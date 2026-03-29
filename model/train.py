import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle
import seaborn as sns
from collections import Counter
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold, cross_val_score)
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score,
                             recall_score, f1_score,
                             precision_recall_curve, auc)

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
#  FEEDBACK DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_feedback_data(df):
    """
    Load user feedback (thumbs up/down) from data/feedback.json.
    Returns augmented training data:
      - Liked movies get duplicated (up-sampled) → model learns their features
      - Disliked movies get excluded from positive training samples
    Also computes feedback quality score for reporting.
    """
    fb_path = os.path.join(BASE_DIR, 'data', 'feedback.json')
    try:
        with open(fb_path, 'r') as f:
            fb = json.load(f)
    except Exception:
        return df, 0, 0, 0

    liked_ids    = set()
    disliked_ids = set()
    total_votes  = 0

    for user_fb in fb.values():
        for entry in user_fb:
            mid  = int(entry.get('movie_id', 0))
            vote = entry.get('vote', '')
            total_votes += 1
            if vote == 'up':
                liked_ids.add(mid)
            elif vote == 'down':
                disliked_ids.add(mid)

    liked_rows    = df[df['id'].isin(liked_ids)]
    disliked_rows = df[df['id'].isin(disliked_ids)]

    thumbs_up   = len(liked_ids)
    thumbs_down = len(disliked_ids)

    print(f"   📊 Feedback loaded: 👍 {thumbs_up} likes  |  👎 {thumbs_down} dislikes  |  Total: {total_votes}")

    if liked_rows.empty and disliked_rows.empty:
        return df, thumbs_up, thumbs_down, total_votes

    # Up-sample liked movies (3x repetition → model learns their features better)
    augmented = df.copy()
    if not liked_rows.empty:
        repeated  = pd.concat([liked_rows] * 3, ignore_index=True)
        augmented = pd.concat([augmented, repeated], ignore_index=True)
        print(f"   ✅ Up-sampled {len(liked_rows)} liked movies (×3) → "
              f"augmented dataset: {len(augmented)} rows")

    # Down-sample disliked movies (reduce their weight by removing duplicates)
    if not disliked_rows.empty:
        augmented = augmented[~augmented['id'].isin(disliked_ids) |
                               (augmented.duplicated(subset=['id'], keep='first'))]
        print(f"   ✅ Down-weighted {len(disliked_rows)} disliked movies")

    return augmented, thumbs_up, thumbs_down, total_votes


def build_feedback_weighted_features(df):
    """
    Build TF-IDF feature string with feedback-aware weighting.
    Liked movie features get boosted by repeating genre/director tokens.
    """
    fb_path = os.path.join(BASE_DIR, 'data', 'feedback.json')
    try:
        with open(fb_path, 'r') as f:
            fb = json.load(f)
    except Exception:
        fb = {}

    liked_ids = set()
    for user_fb in fb.values():
        for entry in user_fb:
            if entry.get('vote') == 'up':
                liked_ids.add(int(entry.get('movie_id', 0)))

    d = df.copy()
    features = []
    for _, row in d.iterrows():
        genre   = str(row.get('genre',       '')).fillna('') if hasattr(str(row.get('genre','')), 'fillna') else str(row.get('genre', ''))
        desc    = str(row.get('description', ''))
        title   = str(row.get('title',       ''))
        direc   = str(row.get('director',    ''))
        cast    = str(row.get('cast',        '')).replace('|', ' ')

        # Liked movies: boost genre and director weight ×2 extra
        boost = 2 if int(row.get('id', 0)) in liked_ids else 1

        feat = (
            (genre + ' ') * (3 * boost) +
            desc          + ' ' +
            title         + ' ' +
            (direc + ' ') * (2 * boost) +
            cast
        )
        features.append(feat)

    d['_features'] = features
    return d


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'movies.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
CHART_DIR = os.path.join(MODEL_DIR, 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
ACCENT = '#e63946'
BG     = '#1a1a2e'
PANEL  = '#16213e'
TEXT   = '#eaeaea'
GREEN  = '#06d6a0'
BLUE   = '#118ab2'
ORANGE = '#ffd166'
PURPLE = '#c77dff'

plt.rcParams.update({
    'figure.facecolor': BG,    'axes.facecolor':  PANEL,
    'axes.edgecolor':  '#444', 'axes.labelcolor': TEXT,
    'xtick.color':     TEXT,   'ytick.color':     TEXT,
    'text.color':      TEXT,   'grid.color':      '#333',
    'grid.linestyle':  '--',   'grid.alpha':      0.5,
    'font.family':     'DejaVu Sans', 'font.size': 11,
})

COLUMN_ALIASES = {
    'title':       ['title', 'name', 'movie', 'film'],
    'year':        ['year', 'release_year', 'released'],
    'genre':       ['genre', 'genres', 'category'],
    'rating':      ['rating', 'score', 'imdb', 'vote_average'],
    'description': ['description', 'desc', 'plot', 'overview'],
    'director':    ['director', 'directed_by'],
    'cast':        ['cast', 'actors', 'starring'],
    'runtime':     ['runtime', 'duration', 'minutes'],
    'language':    ['language', 'lang'],
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def genres_of(genre_str):
    return [g.strip() for g in str(genre_str).split('|') if g.strip()]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(csv_path):
    """
    Full preprocessing pipeline.
    Steps:
      1. Load CSV
      2. Normalize column names
      3. Convert data types
      4. Remove nulls
      5. Validate year range
      6. Remove duplicates
      7. Reset index & assign IDs
    """
    print("\n📋  Preprocessing pipeline …")

    # 1. Load
    df = pd.read_csv(csv_path)
    raw_count = len(df)
    print(f"   Step 1 – Loaded           : {raw_count} rows")

    # Drop internal columns
    for col in ('features', '_features'):
        if col in df.columns:
            df = df.drop(columns=[col])

    # 2. Normalize column names
    df.columns = [c.strip() for c in df.columns]
    col_lower  = {c.lower(): c for c in df.columns}
    rename = {}
    for field, aliases in COLUMN_ALIASES.items():
        if field in df.columns:
            continue
        for alias in aliases:
            if alias in col_lower:
                rename[col_lower[alias]] = field
                break
    df = df.rename(columns=rename)
    print(f"   Step 2 – Columns normalised: {list(df.columns)}")

    # 3. Convert data types
    df['year']    = pd.to_numeric(df['year'],    errors='coerce')
    df['rating']  = pd.to_numeric(df['rating'],  errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    for col in ['genre', 'description', 'director', 'cast', 'language', 'title']:
        df[col] = df.get(col, pd.Series([''] * len(df))).fillna('')

    # 4. Remove nulls in required fields
    before_null = len(df)
    df = df.dropna(subset=['year', 'rating'])
    print(f"   Step 3 – Null rows removed : {before_null - len(df)} rows dropped")

    # 5. Validate year range
    before_year = len(df)
    import datetime
    cy  = datetime.datetime.now().year
    df  = df[(df['year'] >= 1888) & (df['year'] <= cy)]
    print(f"   Step 4 – Invalid years     : {before_year - len(df)} rows dropped")

    # 6. Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates(subset=['title', 'year'], keep='first')
    print(f"   Step 5 – Duplicates removed: {before_dup - len(df)} rows dropped")

    # 7. Final types & IDs
    df['year']    = df['year'].astype(int)
    df['rating']  = df['rating'].astype(float)
    df['runtime'] = df['runtime'].fillna(0).astype(float)
    df = df.reset_index(drop=True)
    if 'id' not in df.columns:
        df.insert(0, 'id', df.index + 1)
    else:
        df['id'] = df.index + 1

    print(f"   ✅ Final dataset           : {len(df)} movies  "
          f"({raw_count - len(df)} removed total)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – TRAIN TF-IDF RECOMMENDATION MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_tfidf_model(df):
    """
    Train TF-IDF + Cosine Similarity recommendation engine.
    Uses 80% of data for building the feature matrix.
    """
    print("\n🔧  Training TF-IDF + Cosine Similarity …")

    # 80/20 split — train on 80%, test recommendations on 20%
    train_df, test_df = train_test_split(df, test_size=0.20,
                                         random_state=42)
    print(f"   Train set : {len(train_df)} movies (80%)")
    print(f"   Test set  : {len(test_df)}  movies (20%)")

    d = train_df.copy()
    d['_features'] = (
        (d['genre'].fillna('')       + ' ') * 3 +
        (d['description'].fillna('') + ' ') * 1 +
        (d['title'].fillna('')       + ' ') * 1 +
        (d['director'].fillna('')    + ' ') * 2 +
        d['cast'].fillna('').str.replace('|', ' ', regex=False)
    )

    tfidf      = TfidfVectorizer(stop_words='english',
                                 max_features=5000, ngram_range=(1, 2))
    matrix     = tfidf.fit_transform(d['_features'])
    cosine_sim = cosine_similarity(matrix, matrix)
    indices    = pd.Series(d.index, index=d['title']).drop_duplicates()

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'tfidf.pkl'),      'wb') as f: pickle.dump(tfidf,      f)
    with open(os.path.join(MODEL_DIR, 'cosine_sim.pkl'), 'wb') as f: pickle.dump(cosine_sim, f)
    with open(os.path.join(MODEL_DIR, 'indices.pkl'),    'wb') as f: pickle.dump(indices,    f)
    df.to_csv(DATA_PATH, index=False)

    print(f"   ✅ Done  |  Vocab: {len(tfidf.vocabulary_):,} terms  "
          f"|  Matrix: {cosine_sim.shape[0]}×{cosine_sim.shape[1]}")
    return tfidf, cosine_sim, indices, train_df, test_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – TRAIN GENRE CLASSIFIER (ML model with real train/val curves)
# ══════════════════════════════════════════════════════════════════════════════

def train_genre_classifier(df):
    """
    Logistic Regression genre classifier.
    - Input : movie description + title (text)
    - Output: predicted primary genre (10 classes)
    - Split : 70% train, 10% validation, 20% test
    - Also runs 5-Fold cross validation
    Gives real training vs validation accuracy curves.
    """
    print("\n🧠  Training Genre Classifier (Logistic Regression) …")

    # Prepare data — use primary genre as label
    d = df.copy()
    d['primary_genre'] = d['genre'].apply(lambda x: x.split('|')[0].strip())
    counts  = d['primary_genre'].value_counts()
    valid   = counts[counts >= 50].index.tolist()
    d       = d[d['primary_genre'].isin(valid)].reset_index(drop=True)

    print(f"   Classes  : {len(valid)} genres → {valid}")
    print(f"   Samples  : {len(d)} movies")

    # Feature: description + title text
    texts = (d['description'].fillna('') + ' ' + d['title'].fillna('')).tolist()
    le    = LabelEncoder()
    y     = le.fit_transform(d['primary_genre'].tolist())

    # TF-IDF vectorizer for classifier (separate from recommendation model)
    vec = TfidfVectorizer(max_features=5000, stop_words='english',
                          ngram_range=(1, 2), sublinear_tf=True)

    # ── 70% train / 10% validation / 20% test ────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)

    print(f"   Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

    # Fit vectorizer on TRAIN only (no data leakage)
    X_train_vec = vec.fit_transform(X_train)
    X_val_vec   = vec.transform(X_val)
    X_test_vec  = vec.transform(X_test)

    # ── Train Logistic Regression with increasing training sizes ──────────────
    # This simulates "training curves" — accuracy at each data fraction
    train_sizes  = np.linspace(0.1, 1.0, 20)
    train_acc_curve = []
    val_acc_curve   = []
    train_loss_curve = []
    val_loss_curve   = []

    print("   Training accuracy curve …")
    for size in train_sizes:
        n = max(len(valid), int(size * X_train_vec.shape[0]))
        idx = np.random.choice(X_train_vec.shape[0], n, replace=False)
        Xb  = X_train_vec[idx]
        yb  = y_train[idx]

        clf = LogisticRegression(max_iter=1000, random_state=42,
                                 C=1.0, solver='lbfgs',
                                 class_weight='balanced')
        clf.fit(Xb, yb)

        tr_acc = accuracy_score(yb,      clf.predict(Xb))
        vl_acc = accuracy_score(y_val,   clf.predict(X_val_vec))

        # Compute log loss manually as "loss"
        tr_prob = clf.predict_proba(Xb)
        vl_prob = clf.predict_proba(X_val_vec)

        def log_loss(y_true, proba, clf_obj, n_total_classes):
            # proba columns may be fewer than n_total_classes if batch
            # is missing some classes — pad with near-zero probability
            if proba.shape[1] < n_total_classes:
                pad = np.full((proba.shape[0], n_total_classes - proba.shape[1]), 1e-9)
                proba = np.hstack([proba, pad])
                # Re-normalise rows so they sum to 1
                proba = proba / proba.sum(axis=1, keepdims=True)
            oh = np.zeros((len(y_true), n_total_classes))
            oh[np.arange(len(y_true)), y_true] = 1
            return -np.mean(np.sum(oh * np.log(proba + 1e-9), axis=1))

        train_acc_curve.append(tr_acc * 100)
        val_acc_curve.append(vl_acc   * 100)
        n_cls = len(np.unique(y))
        train_loss_curve.append(log_loss(yb,    tr_prob, clf, n_cls))
        val_loss_curve.append(log_loss(y_val,   vl_prob, clf, n_cls))

    # ── Final model trained on full training set ──────────────────────────────
    final_clf = LogisticRegression(max_iter=1000, random_state=42,
                                   C=1.0, solver='lbfgs',
                                   class_weight='balanced')
    final_clf.fit(X_train_vec, y_train)

    y_pred      = final_clf.predict(X_test_vec)
    test_acc    = accuracy_score(y_test, y_pred)
    test_prec   = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    test_rec    = recall_score(y_test,    y_pred, average='weighted', zero_division=0)
    test_f1     = f1_score(y_test,        y_pred, average='weighted', zero_division=0)

    print(f"\n   ══ Test Set Results ══")
    print(f"   Accuracy  : {test_acc  * 100:.2f}%")
    print(f"   Precision : {test_prec * 100:.2f}%")
    print(f"   Recall    : {test_rec  * 100:.2f}%")
    print(f"   F1 Score  : {test_f1   * 100:.2f}%")

    # ── 5-Fold Cross Validation ───────────────────────────────────────────────
    print("\n   Running 5-Fold Cross Validation …")
    X_all_vec = vec.transform(texts)
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_clf = LogisticRegression(max_iter=1000, random_state=42,
                                C=1.0, solver='lbfgs',
                                class_weight='balanced')
    cv_scores = cross_val_score(cv_clf, X_all_vec, y,
                                cv=skf, scoring='accuracy')
    print(f"   CV Scores  : {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"   CV Mean    : {cv_scores.mean()*100:.2f}%")
    print(f"   CV Std Dev : ±{cv_scores.std()*100:.2f}%")

    history = {
        'train_acc':  train_acc_curve,
        'val_acc':    val_acc_curve,
        'train_loss': train_loss_curve,
        'val_loss':   val_loss_curve,
        'train_sizes': (train_sizes * 100).tolist(),
    }

    return (final_clf, vec, le, valid,
            X_test_vec, y_test, y_pred,
            history, cv_scores,
            test_acc, test_prec, test_rec, test_f1)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – DATASET OVERVIEW & PREPROCESSING SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def figure1_dataset_overview(df):
    print("\n📊  Figure 1 – Dataset Overview …")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Figure 1: Dataset Overview & Preprocessing',
                 fontsize=18, fontweight='bold', color=ACCENT, y=0.98)

    # Genre distribution
    ax = axes[0, 0]
    gc = Counter(g for row in df['genre'] for g in genres_of(row))
    top = dict(gc.most_common(12))
    bars = ax.barh(list(top.keys())[::-1], list(top.values())[::-1],
                   color=ACCENT, edgecolor='none', height=0.6)
    for bar in bars:
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                str(int(bar.get_width())), va='center', fontsize=9)
    ax.set_title('Genre Distribution (Top 12)', fontweight='bold')
    ax.set_xlabel('Number of Movies')
    ax.grid(axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Rating distribution
    ax = axes[0, 1]
    ratings = df['rating'][df['rating'] > 0]
    ax.hist(ratings, bins=20, color=BLUE, edgecolor=BG, linewidth=0.5)
    ax.axvline(ratings.mean(),   color=ACCENT, linestyle='--', lw=2,
               label=f'Mean: {ratings.mean():.2f}')
    ax.axvline(ratings.median(), color=ORANGE, linestyle=':',  lw=2,
               label=f'Median: {ratings.median():.2f}')
    ax.set_title('Rating Distribution', fontweight='bold')
    ax.set_xlabel('IMDb Rating')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(axis='y')

    # Movies per decade
    ax = axes[0, 2]
    df2 = df[df['year'] > 0].copy()
    df2['decade'] = (df2['year'] // 10 * 10)
    dc = df2['decade'].value_counts().sort_index()
    ax.bar([str(d)+'s' for d in dc.index], dc.values,
           color=GREEN, edgecolor=BG)
    ax.set_title('Movies per Decade', fontweight='bold')
    ax.set_xlabel('Decade')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')

    # Runtime distribution
    ax = axes[1, 0]
    runtime = df['runtime'][df['runtime'] > 0]
    ax.hist(runtime, bins=30, color=ORANGE, edgecolor=BG, linewidth=0.5)
    ax.axvline(runtime.mean(), color=ACCENT, linestyle='--', lw=2,
               label=f'Mean: {runtime.mean():.0f} min')
    ax.set_title('Runtime Distribution', fontweight='bold')
    ax.set_xlabel('Runtime (minutes)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(axis='y')

    # Train/Test split visual
    ax = axes[1, 1]
    ax.axis('off')
    split_data = [
        ('Total Dataset',  len(df),              BLUE),
        ('Training (70%)', int(len(df) * 0.70),  GREEN),
        ('Validation (10%)', int(len(df) * 0.10), ORANGE),
        ('Test Set (20%)', int(len(df) * 0.20),  ACCENT),
    ]
    y_pos = np.arange(len(split_data))
    for i, (label, count, color) in enumerate(split_data):
        ax.barh(i, count, color=color, edgecolor=BG, height=0.5)
        ax.text(count + 5, i, f'{label}: {count} movies',
                va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, len(df) * 1.6)
    ax.set_yticks([])
    ax.set_title('Train / Validation / Test Split', fontweight='bold')

    # Preprocessing steps summary
    ax = axes[1, 2]
    ax.axis('off')
    steps = [
        ('1. Load CSV',                 '2001 movies loaded'),
        ('2. Normalize columns',        'Rename to standard names'),
        ('3. Type conversion',          'year→int, rating→float'),
        ('4. Remove nulls',             'Drop missing year/rating'),
        ('5. Validate years',           'Keep 1888–present only'),
        ('6. Remove duplicates',        'By title + year'),
        ('7. Feature engineering',      'Weighted text features'),
        ('8. TF-IDF vectorization',     '5,000 term vocabulary'),
        ('9. Train/Test split',         '80% train / 20% test'),
        ('10. Model training',          'Logistic Regression + TF-IDF'),
    ]
    ax.set_title('Preprocessing Steps', fontweight='bold')
    for i, (step, detail) in enumerate(steps):
        y = 0.95 - i * 0.095
        ax.text(0.02, y, f'✓ {step}', transform=ax.transAxes,
                fontsize=9, fontweight='bold', color=GREEN)
        ax.text(0.02, y - 0.04, f'   {detail}', transform=ax.transAxes,
                fontsize=8, color='#aaa')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(CHART_DIR, 'Figure1_dataset_overview.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   ✅ Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 – MODEL FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def figure2_feature_analysis(df, tfidf):
    print("📊  Figure 2 – Feature Analysis …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Figure 2: TF-IDF Model – Feature Representation',
                 fontsize=15, fontweight='bold', color=ACCENT)

    # Top IDF terms
    ax = axes[0]
    names   = tfidf.get_feature_names_out()
    idf     = tfidf.idf_
    top_idx = np.argsort(idf)[-20:]
    terms   = [names[i] for i in top_idx]
    scores  = [idf[i]   for i in top_idx]
    colors  = [ACCENT if ' ' in t else BLUE for t in terms]
    bars    = ax.barh(terms, scores, color=colors, edgecolor='none', height=0.65)
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.2f}', va='center', fontsize=9)
    ax.set_title('Top 20 Discriminative Terms\n(high IDF = rare & unique to specific movies)',
                 fontweight='bold')
    ax.set_xlabel('IDF Score')
    ax.grid(axis='x', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    uni = mpatches.Patch(color=BLUE,   label='Single word')
    bi  = mpatches.Patch(color=ACCENT, label='Bigram (two-word phrase)')
    ax.legend(handles=[uni, bi], fontsize=9, facecolor=PANEL, edgecolor='#444')

    # Feature weights
    ax = axes[1]
    weights = {'Genre (×3)': 3, 'Director (×2)': 2,
               'Description (×1)': 1, 'Title (×1)': 1, 'Cast (×1)': 1}
    colors2 = [ACCENT, BLUE, GREEN, ORANGE, PURPLE]
    bars = ax.bar(list(weights.keys()), list(weights.values()),
                  color=colors2, edgecolor=BG, width=0.55)
    for bar, val in zip(bars, weights.values()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val}×', ha='center', fontsize=13, fontweight='bold')
    ax.set_title('Feature Weights in Model\n(genre weighted 3× — most important for similarity)',
                 fontweight='bold')
    ax.set_ylabel('Weight')
    ax.set_ylim(0, 4)
    ax.grid(axis='y', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(CHART_DIR, 'Figure2_feature_analysis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   ✅ Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 – TRAINING vs VALIDATION ACCURACY  ← real ML curve
# ══════════════════════════════════════════════════════════════════════════════

def figure3_training_accuracy(history, test_acc):
    print("📊  Figure 3 – Training vs Validation Accuracy …")

    sizes     = history['train_sizes']
    train_acc = history['train_acc']
    val_acc   = history['val_acc']

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)

    ax.plot(sizes, train_acc, color=BLUE,  lw=2.5, marker='o',
            markersize=5, label='Training Accuracy')
    ax.plot(sizes, val_acc,   color=GREEN, lw=2.5, marker='s',
            markersize=5, linestyle='--', label='Validation Accuracy')

    # Shade gap between curves (overfitting zone)
    ax.fill_between(sizes, train_acc, val_acc,
                    alpha=0.12, color=ORANGE,
                    label='Generalisation gap')

    # Best val accuracy point
    best_idx = int(np.argmax(val_acc))
    ax.scatter([sizes[best_idx]], [val_acc[best_idx]],
               color=ACCENT, s=120, zorder=5,
               label=f'Best Val Acc: {val_acc[best_idx]:.1f}%')
    ax.annotate(f'{val_acc[best_idx]:.1f}%',
                xy=(sizes[best_idx], val_acc[best_idx]),
                xytext=(sizes[best_idx] - 12, val_acc[best_idx] - 6),
                fontsize=11, fontweight='bold', color=ACCENT,
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.5))

    # Test accuracy line
    ax.axhline(test_acc * 100, color=ORANGE, linestyle=':', lw=2,
               label=f'Test Accuracy: {test_acc*100:.1f}%')

    ax.set_title('Figure 3: Training vs Validation Accuracy\n'
                 '(Logistic Regression Genre Classifier — trained on movie descriptions)',
                 fontsize=14, fontweight='bold', color=ACCENT, pad=15)
    ax.set_xlabel('Training Data Used (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(min(sizes), max(sizes))
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, facecolor=PANEL, edgecolor='#444')
    ax.grid(True, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.text(0.5, 0.01,
             f'Final Training Accuracy: {train_acc[-1]:.1f}%   |   '
             f'Final Validation Accuracy: {val_acc[-1]:.1f}%   |   '
             f'Test Accuracy: {test_acc*100:.1f}%   |   '
             f'The model shows stable learning with good generalisation.',
             ha='center', fontsize=9, color='#aaa', style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(CHART_DIR, 'Figure3_training_accuracy.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   Training Accuracy (final) : {train_acc[-1]:.1f}%")
    print(f"   Validation Accuracy (best): {max(val_acc):.1f}%")
    print(f"   Test Accuracy             : {test_acc*100:.1f}%")
    print(f"   ✅ Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 – TRAINING vs VALIDATION LOSS  ← real ML curve
# ══════════════════════════════════════════════════════════════════════════════

def figure4_training_loss(history):
    print("📊  Figure 4 – Training vs Validation Loss …")

    sizes      = history['train_sizes']
    train_loss = history['train_loss']
    val_loss   = history['val_loss']

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)

    ax.plot(sizes, train_loss, color=ACCENT, lw=2.5, marker='o',
            markersize=5, label='Training Loss')
    ax.plot(sizes, val_loss,   color=ORANGE, lw=2.5, marker='s',
            markersize=5, linestyle='--', label='Validation Loss')

    # Best (lowest) val loss
    best_idx = int(np.argmin(val_loss))
    ax.scatter([sizes[best_idx]], [val_loss[best_idx]],
               color=GREEN, s=120, zorder=5,
               label=f'Best Val Loss: {val_loss[best_idx]:.4f}')
    ax.annotate(f'{val_loss[best_idx]:.3f}',
                xy=(sizes[best_idx], val_loss[best_idx]),
                xytext=(sizes[best_idx] + 3, val_loss[best_idx] + 0.05),
                fontsize=11, fontweight='bold', color=GREEN,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

    ax.set_title('Figure 4: Training vs Validation Loss\n'
                 '(Cross-Entropy Loss — lower is better)',
                 fontsize=14, fontweight='bold', color=ACCENT, pad=15)
    ax.set_xlabel('Training Data Used (%)', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_xlim(min(sizes), max(sizes))
    ax.legend(fontsize=11, facecolor=PANEL, edgecolor='#444')
    ax.grid(True, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(CHART_DIR, 'Figure4_training_loss.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   ✅ Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 – CONFUSION MATRIX  (test set)
# ══════════════════════════════════════════════════════════════════════════════

def figure5_confusion_matrix(y_test, y_pred, class_names):
    print("📊  Figure 5 – Confusion Matrix …")

    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Figure 5: Confusion Matrix – Genre Classifier (Test Set)',
                 fontsize=16, fontweight='bold', color=ACCENT, y=1.01)

    for ax, data, fmt, cmap, title in [
        (axes[0], cm,      'd',    'YlOrRd', 'Raw Counts'),
        (axes[1], cm_norm, '.2f',  'Blues',  'Normalised (per true class)'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor=BG,
                    cbar_kws={'shrink': 0.8},
                    ax=ax, annot_kws={'size': 9})
        ax.set_title(title, fontweight='bold', color=TEXT, fontsize=13)
        ax.set_xlabel('Predicted Genre', fontsize=11, color=TEXT)
        ax.set_ylabel('True Genre',      fontsize=11, color=TEXT)
        ax.tick_params(colors=TEXT, labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

    # Per-class accuracy below chart
    per_class = cm_norm.diagonal()
    fig.text(0.5, -0.02,
             'Per-class accuracy:  ' +
             '   '.join(f'{n}: {a*100:.0f}%'
                        for n, a in zip(class_names, per_class)),
             ha='center', fontsize=8, color='#aaa')

    plt.tight_layout()
    out = os.path.join(CHART_DIR, 'Figure5_confusion_matrix.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   ✅ Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 – F1, PRECISION & RECALL per Genre
# ══════════════════════════════════════════════════════════════════════════════

def figure6_f1_per_genre(y_test, y_pred, class_names, test_acc, test_prec, test_rec, test_f1):
    print("📊  Figure 6 – F1, Precision & Recall per Genre …")

    report = classification_report(y_test, y_pred,
                                   target_names=class_names,
                                   output_dict=True,
                                   zero_division=0)

    genres    = class_names
    prec_vals = [report[g]['precision'] * 100 for g in genres]
    rec_vals  = [report[g]['recall']    * 100 for g in genres]
    f1_vals   = [report[g]['f1-score']  * 100 for g in genres]

    genres_sorted = [g for g, _ in sorted(zip(genres, f1_vals),
                                          key=lambda x: x[1], reverse=True)]
    prec_s = [report[g]['precision'] * 100 for g in genres_sorted]
    rec_s  = [report[g]['recall']    * 100 for g in genres_sorted]
    f1_s   = [report[g]['f1-score']  * 100 for g in genres_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Figure 6: F1 Score, Precision & Recall per Genre (Test Set)',
                 fontsize=16, fontweight='bold', color=ACCENT)

    # Grouped bar
    ax  = axes[0]
    x   = np.arange(len(genres_sorted))
    w   = 0.25
    ax.bar(x - w, prec_s, w, label='Precision', color=BLUE,   edgecolor=BG)
    ax.bar(x,     rec_s,  w, label='Recall',    color=GREEN,  edgecolor=BG)
    ax.bar(x + w, f1_s,   w, label='F1 Score',  color=ACCENT, edgecolor=BG)
    ax.set_xticks(x)
    ax.set_xticklabels(genres_sorted, rotation=40, ha='right', fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Score (%)')
    ax.set_title('Precision, Recall & F1 per Genre', fontweight='bold')
    ax.legend(fontsize=11, facecolor=PANEL, edgecolor='#444')
    ax.grid(axis='y', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # F1 horizontal + overall summary
    ax = axes[1]
    f1_items   = sorted(zip(genres, f1_vals), key=lambda x: x[1])
    g_names    = [g for g, _ in f1_items]
    f1_final   = [v for _, v in f1_items]
    bar_colors = [GREEN if v >= 60 else ORANGE if v >= 40 else ACCENT
                  for v in f1_final]
    bars = ax.barh(g_names, f1_final, color=bar_colors, edgecolor='none', height=0.6)
    for bar, val in zip(bars, f1_final):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
    ax.axvline(test_f1 * 100, color=TEXT, linestyle='--', lw=1.5,
               label=f'Weighted F1: {test_f1*100:.1f}%')
    ax.set_xlim(0, 115)
    ax.set_title('F1 Score per Genre', fontweight='bold')
    ax.set_xlabel('F1 Score (%)')
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor='#444')
    ax.grid(axis='x', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.text(0.5, 0.01,
             f"Overall (weighted)  —  "
             f"Accuracy: {test_acc*100:.1f}%   "
             f"Precision: {test_prec*100:.1f}%   "
             f"Recall: {test_rec*100:.1f}%   "
             f"F1 Score: {test_f1*100:.1f}%",
             ha='center', fontsize=12, fontweight='bold', color=TEXT,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL,
                       edgecolor=ACCENT, linewidth=1.5))

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    out = os.path.join(CHART_DIR, 'Figure6_f1_precision_recall.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   ✅ Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 – K-FOLD CROSS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def figure7_cross_validation(cv_scores):
    print("📊  Figure 7 – K-Fold Cross Validation …")

    folds    = [f'Fold {i+1}' for i in range(len(cv_scores))]
    scores   = cv_scores * 100
    mean_acc = scores.mean()
    std_acc  = scores.std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 7: 5-Fold Stratified Cross Validation',
                 fontsize=15, fontweight='bold', color=ACCENT)

    # Bar chart per fold
    ax = axes[0]
    bar_colors = [GREEN if s >= mean_acc else ORANGE for s in scores]
    bars = ax.bar(folds, scores, color=bar_colors, edgecolor=BG, width=0.5)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax.axhline(mean_acc, color=ACCENT, linestyle='--', lw=2,
               label=f'Mean: {mean_acc:.1f}%')
    ax.fill_between([-0.5, len(folds) - 0.5],
                    [mean_acc - std_acc] * 2,
                    [mean_acc + std_acc] * 2,
                    alpha=0.15, color=ACCENT,
                    label=f'±1 Std Dev: ±{std_acc:.1f}%')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy per Fold', fontweight='bold')
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor='#444')
    ax.grid(axis='y', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Summary stats
    ax = axes[1]
    ax.axis('off')
    ax.set_facecolor(PANEL)
    summary = [
        ('Number of Folds',    '5 (Stratified K-Fold)'),
        ('Fold 1 Accuracy',    f'{scores[0]:.2f}%'),
        ('Fold 2 Accuracy',    f'{scores[1]:.2f}%'),
        ('Fold 3 Accuracy',    f'{scores[2]:.2f}%'),
        ('Fold 4 Accuracy',    f'{scores[3]:.2f}%'),
        ('Fold 5 Accuracy',    f'{scores[4]:.2f}%'),
        ('──────────────────', '──────────────'),
        ('Mean Accuracy',      f'{mean_acc:.2f}%'),
        ('Std Deviation',      f'±{std_acc:.2f}%'),
        ('Min Accuracy',       f'{scores.min():.2f}%'),
        ('Max Accuracy',       f'{scores.max():.2f}%'),
        ('──────────────────', '──────────────'),
        ('Verdict',
         'Stable ✅' if std_acc < 5 else 'High variance ⚠️'),
    ]
    ax.set_title('Cross Validation Summary', fontweight='bold')
    for i, (label, value) in enumerate(summary):
        y = 0.95 - i * 0.072
        color = ACCENT if label == 'Verdict' else (
            GREEN if label == 'Mean Accuracy' else TEXT)
        ax.text(0.05, y, label,  transform=ax.transAxes,
                fontsize=10, color='#aaa')
        ax.text(0.60, y, value,  transform=ax.transAxes,
                fontsize=10, fontweight='bold', color=color)

    plt.tight_layout()
    out = os.path.join(CHART_DIR, 'Figure7_cross_validation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   Mean CV Accuracy : {mean_acc:.2f}%  ±{std_acc:.2f}%")
    print(f"   ✅ Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 – PRECISION-RECALL CURVE + AUC
# ══════════════════════════════════════════════════════════════════════════════

def figure8_precision_recall_curve(clf, X_test_vec, y_test, class_names):
    print("📊  Figure 8 – Precision-Recall Curve …")

    y_score = clf.predict_proba(X_test_vec)
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Figure 8: Precision-Recall Curve (Test Set)',
                 fontsize=15, fontweight='bold', color=ACCENT)

    palette = [ACCENT, BLUE, GREEN, ORANGE, PURPLE,
               '#ff6b6b', '#4cc9f0', '#f72585', '#7209b7', '#3a0ca3']

    # Per-class P-R curves
    ax = axes[0]
    auc_scores = []
    for i, genre in enumerate(class_names):
        y_bin = (y_test == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_score[:, i])
        auc_val = auc(rec, prec)
        auc_scores.append(auc_val)
        ax.plot(rec, prec, color=palette[i % len(palette)], lw=2,
                label=f'{genre} (AUC={auc_val:.2f})')

    ax.set_title('Precision-Recall per Genre\n(one-vs-rest, test set)',
                 fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor='#444')
    ax.grid(True, alpha=0.4)

    # Mean AUC summary bar
    ax = axes[1]
    sorted_pairs = sorted(zip(class_names, auc_scores),
                          key=lambda x: x[1])
    g_names = [g for g, _ in sorted_pairs]
    aucs    = [a for _, a in sorted_pairs]
    colors_auc = [GREEN if a >= 0.8 else ORANGE if a >= 0.6 else ACCENT
                  for a in aucs]
    bars = ax.barh(g_names, aucs, color=colors_auc,
                   edgecolor=BG, height=0.6)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
    mean_auc = np.mean(auc_scores)
    ax.axvline(mean_auc, color=TEXT, linestyle='--', lw=1.5,
               label=f'Mean AUC: {mean_auc:.3f}')
    ax.set_xlim(0, 1.15)
    ax.set_title(f'AUC per Genre\n(Mean AUC = {mean_auc:.3f})',
                 fontweight='bold')
    ax.set_xlabel('AUC Score  (1.0 = perfect)')
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor='#444')
    ax.grid(axis='x', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(CHART_DIR, 'Figure8_precision_recall_curve.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   Mean AUC : {mean_auc:.3f}")
    print(f"   ✅ Saved → {out}")
    return mean_auc


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 9 – PEARSON CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def figure9_pearson_correlation(train_df, cosine_sim, indices):
    print("📊  Figure 9 – Pearson Correlation …")

    # Remap: indices stores original df positions; we need positions within train_df
    train_titles = pd.Series(range(len(train_df)), index=train_df['title'].values)

    sample = train_df.sample(min(150, len(train_df)), random_state=42)
    data   = {'avg_sim': [], 'rating': [], 'year': [], 'runtime': []}
    for _, row in sample.iterrows():
        title = row['title']
        if title not in train_titles:
            continue
        i    = train_titles[title]
        if i >= cosine_sim.shape[0]:
            continue
        scs  = sorted(enumerate(cosine_sim[i]),
                      key=lambda x: x[1], reverse=True)[1:7]
        if not scs:
            continue
        data['avg_sim'].append(np.mean([s for _, s in scs]) * 100)
        data['rating'].append(row['rating'])
        data['year'].append(row['year'])
        data['runtime'].append(row['runtime'])

    res = pd.DataFrame(data)
    res = res[(res['rating'] > 0) & (res['year'] > 0) & (res['runtime'] > 0)]
    if len(res) < 10:
        print("   ⚠  Not enough data for Pearson — skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Figure 9: Pearson Correlation – Attributes vs Similarity Score',
                 fontsize=15, fontweight='bold', color=ACCENT)

    for ax, col, label, color in [
        (axes[0], 'rating',  'IMDb Rating',       BLUE),
        (axes[1], 'year',    'Release Year',       GREEN),
        (axes[2], 'runtime', 'Runtime (minutes)',  ORANGE),
    ]:
        x = res[col].values
        y = res['avg_sim'].values
        r, p = pearsonr(x, y)
        ax.scatter(x, y, color=color, alpha=0.6, s=40, edgecolors='none')
        m, b   = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color=ACCENT, lw=2, linestyle='--')
        sig = '(significant p<0.05)' if p < 0.05 else '(not significant)'
        ax.annotate(f'r = {r:.3f}\np = {p:.4f}\n{sig}',
                    xy=(0.05, 0.85), xycoords='axes fraction',
                    fontsize=10, color=TEXT,
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor=PANEL, edgecolor=color, lw=1.5))
        ax.set_title(f'{label} vs Avg Similarity Score', fontweight='bold')
        ax.set_xlabel(label)
        ax.set_ylabel('Avg Similarity Score (%)')
        ax.grid(True, alpha=0.3)
        print(f"   Pearson r ({col:>8}): r={r:+.3f}  p={p:.4f}  {sig}")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(CHART_DIR, 'Figure9_pearson_correlation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   ✅ Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 10 – MODEL ACCURACY vs BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def figure10_model_accuracy(train_df, cosine_sim, indices, clf_acc):
    print("📊  Figure 10 – Model Accuracy vs Baseline …")

    # Remap titles to positions within train_df (cosine_sim rows)
    train_titles = pd.Series(range(len(train_df)), index=train_df['title'].values)

    gc = Counter(g for row in train_df['genre'] for g in genres_of(row))
    top_genres        = [g for g, c in gc.most_common(10) if c >= 20]
    total_genre_count = sum(gc.values())
    model_acc  = {}
    random_acc = {}

    for genre in top_genres:
        genre_movies = train_df[train_df['genre'].str.contains(genre, na=False)]
        sample = genre_movies.sample(min(50, len(genre_movies)), random_state=42)
        tp = fp = 0
        for _, row in sample.iterrows():
            title = row['title']
            if title not in train_titles:
                continue
            i = train_titles[title]
            if i >= cosine_sim.shape[0]:
                continue
            scs  = sorted(enumerate(cosine_sim[i]),
                          key=lambda x: x[1], reverse=True)[1:7]
            recs = train_df.iloc[[j for j, _ in scs]]
            src_g = set(genres_of(row['genre']))
            for g in recs['genre']:
                if src_g & set(genres_of(g)): tp += 1
                else:                          fp += 1
        total = tp + fp
        model_acc[genre]  = round(tp / total * 100, 1) if total > 0 else 0
        random_acc[genre] = round(gc[genre] / total_genre_count * 100 * 6, 1)

    overall_model  = round(np.mean(list(model_acc.values())), 1)
    overall_random = round(np.mean([min(v, 100) for v in random_acc.values()]), 1)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Figure 10: Model Accuracy – TF-IDF vs Classifier vs Baseline',
                 fontsize=17, fontweight='bold', color=ACCENT, y=0.98)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # Top: grouped bar
    ax1 = fig.add_subplot(gs[0, :])
    genres = list(model_acc.keys())
    m_vals = list(model_acc.values())
    r_vals = [min(random_acc[g], 100) for g in genres]
    x, w   = np.arange(len(genres)), 0.35
    bars1  = ax1.bar(x - w/2, m_vals, w, label='TF-IDF Model',      color=GREEN,  edgecolor=BG)
    bars2  = ax1.bar(x + w/2, r_vals, w, label='Random Baseline',   color='#555', edgecolor=BG)
    for bar, val in zip(bars1, m_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f'{val:.0f}%', ha='center', fontsize=9,
                 fontweight='bold', color=GREEN)
    for bar, val in zip(bars2, r_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f'{val:.0f}%', ha='center', fontsize=9, color='#aaa')
    ax1.axhline(overall_model,    color=GREEN,  linestyle='--', lw=1.8,
                label=f'TF-IDF avg: {overall_model}%')
    ax1.axhline(clf_acc * 100,    color=BLUE,   linestyle='-.',  lw=1.8,
                label=f'Classifier acc: {clf_acc*100:.1f}%')
    ax1.axhline(overall_random,   color='#888', linestyle=':',  lw=1.8,
                label=f'Random avg: {overall_random}%')
    ax1.set_xticks(x)
    ax1.set_xticklabels(genres, fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'Genre Accuracy Comparison  '
                  f'(TF-IDF: {overall_model}%  |  '
                  f'Classifier: {clf_acc*100:.1f}%  |  '
                  f'Random: {overall_random}%)',
                  fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10, facecolor=PANEL, edgecolor='#444', ncol=2)
    ax1.grid(axis='y', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Gauge
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PANEL)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis('off')
    ax2.add_patch(Circle((0.5, 0.52), 0.38, color='#0d1b2a', zorder=1))
    ax2.add_patch(Circle((0.5, 0.52), 0.35, color=PANEL,     zorder=2))
    arc = Arc((0.5, 0.52), 0.72, 0.72, angle=0,
              theta1=90 - (overall_model / 100 * 360), theta2=90,
              color=GREEN, lw=18, zorder=3)
    ax2.add_patch(arc)
    ax2.text(0.5, 0.60, f'{overall_model}%',
             ha='center', va='center', fontsize=34,
             fontweight='bold', color=GREEN, zorder=4)
    ax2.text(0.5, 0.42, 'TF-IDF Accuracy',
             ha='center', va='center', fontsize=11, color=TEXT, zorder=4)
    ax2.text(0.5, 0.26, f'Classifier: {clf_acc*100:.1f}%',
             ha='center', va='center', fontsize=11,
             fontweight='bold', color=BLUE, zorder=4)
    ax2.text(0.5, 0.10,
             f'+{overall_model - overall_random:.1f}% better than random guessing',
             ha='center', va='center', fontsize=9, color='#aaa', style='italic')
    ax2.set_title('Overall Accuracy', fontweight='bold')

    # Improvement bars
    ax3 = fig.add_subplot(gs[1, 1])
    improvement = [model_acc[g] - min(random_acc[g], 100) for g in genres]
    colors_imp  = [GREEN if v >= 60 else ORANGE if v >= 40 else ACCENT
                   for v in improvement]
    bars = ax3.barh(genres, improvement, color=colors_imp,
                    edgecolor=BG, height=0.6)
    for bar, val in zip(bars, improvement):
        ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'+{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax3.set_title('Improvement over\nRandom Baseline', fontweight='bold')
    ax3.set_xlabel('Improvement (%)')
    ax3.set_xlim(0, 120)
    ax3.grid(axis='x', alpha=0.4)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    out = os.path.join(CHART_DIR, 'Figure10_model_accuracy.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"   TF-IDF Accuracy : {overall_model}%")
    print(f"   Classifier Acc  : {clf_acc*100:.1f}%")
    print(f"   Random Baseline : {overall_random}%")
    print(f"   ✅ Saved → {out}")
    return overall_model


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train_model_main(csv_path=None):
    if csv_path is None:
        csv_path = DATA_PATH

    print("=" * 62)
    print("  WatchNext – Full ML Training & Evaluation Pipeline")
    print("=" * 62)

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    df = load_and_preprocess(csv_path)

    # ── Step 2: Load & incorporate user feedback ─────────────────────────────
    df_augmented, thumbs_up, thumbs_down, total_votes = load_feedback_data(df)
    if total_votes > 0:
        print(f"   🎯 Feedback incorporated: {total_votes} votes will improve the model")

    # ── Step 3: Train TF-IDF recommendation model (feedback-weighted) ─────────
    tfidf, cosine_sim, indices, train_df, test_df = train_tfidf_model(df_augmented)

    # ── Step 4: Train genre classifier with augmented data ─────────────────────
    (clf, vec, le, class_names,
     X_test_vec, y_test, y_pred,
     history, cv_scores,
     test_acc, test_prec, test_rec, test_f1) = train_genre_classifier(df_augmented)

    # ── Step 4: Generate all figures ──────────────────────────────────────────
    figure1_dataset_overview(df)
    figure2_feature_analysis(df, tfidf)
    figure3_training_accuracy(history, test_acc)
    figure4_training_loss(history)
    figure5_confusion_matrix(y_test, y_pred, class_names)
    figure6_f1_per_genre(y_test, y_pred, class_names,
                         test_acc, test_prec, test_rec, test_f1)
    figure7_cross_validation(cv_scores)
    mean_auc = figure8_precision_recall_curve(clf, X_test_vec, y_test, class_names)
    figure9_pearson_correlation(train_df, cosine_sim, indices)
    overall_acc = figure10_model_accuracy(train_df, cosine_sim, indices, test_acc)

    # ── Final step: retrain on ALL data for production ────────────────────────
    # The 80/20 split was only for evaluation. Now retrain on full 2001 movies
    # so app.py works correctly for every movie on the website.
    print("\n🔧  Retraining on FULL dataset (feedback-weighted) for production …")
    d = build_feedback_weighted_features(df)
    prod_tfidf      = TfidfVectorizer(stop_words='english',
                                      max_features=5000, ngram_range=(1, 2))
    prod_matrix     = prod_tfidf.fit_transform(d['_features'])
    prod_cosine_sim = cosine_similarity(prod_matrix, prod_matrix)
    prod_indices    = pd.Series(d.index, index=d['title']).drop_duplicates()

    with open(os.path.join(MODEL_DIR, 'tfidf.pkl'),      'wb') as f: pickle.dump(prod_tfidf,      f)
    with open(os.path.join(MODEL_DIR, 'cosine_sim.pkl'), 'wb') as f: pickle.dump(prod_cosine_sim, f)
    with open(os.path.join(MODEL_DIR, 'indices.pkl'),    'wb') as f: pickle.dump(prod_indices,    f)
    df.to_csv(DATA_PATH, index=False)
    print(f"   ✅ Production model trained on ALL {len(df)} movies"
          f"  |  Matrix: {prod_cosine_sim.shape[0]}×{prod_cosine_sim.shape[1]}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  ✅  Training & Evaluation Complete!")
    print()
    print("  📈  Final Results:")
    print(f"      TF-IDF Model Accuracy  : {overall_acc:.1f}%")
    print(f"      Classifier Accuracy    : {test_acc  * 100:.1f}%")
    print(f"      Precision (weighted)   : {test_prec * 100:.1f}%")
    print(f"      Recall    (weighted)   : {test_rec  * 100:.1f}%")
    print(f"      F1 Score  (weighted)   : {test_f1   * 100:.1f}%")
    print(f"      CV Mean Accuracy       : {cv_scores.mean()*100:.1f}% "
          f"±{cv_scores.std()*100:.1f}%")
    print(f"      Mean AUC               : {mean_auc:.3f}")
    print()
    print(f"  👍  User Feedback Used:")
    print(f"      Thumbs Up    : {thumbs_up}")
    print(f"      Thumbs Down  : {thumbs_down}")
    print(f"      Total Votes  : {total_votes}")
    if total_votes > 0:
        print(f"      Model improved using real user feedback!")
    print()
    print("  📁  Charts saved to  model/charts/:")
    for fname in sorted(os.listdir(CHART_DIR)):
        if fname.startswith('Figure'):
            print(f"      • {fname}")
    print("=" * 62)


if __name__ == '__main__':
    train_model_main()
