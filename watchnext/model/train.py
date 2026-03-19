import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# ── Column name aliases ───────────────────────────────────────────────────────
# Maps our internal field names → possible column names in any uploaded CSV.
# Add more aliases here if you need to support other CSV formats.
COLUMN_ALIASES = {
    'title':       ['title', 'name', 'movie', 'film', 'movie_title'],
    'year':        ['year', 'release_year', 'released', 'release_date'],
    'genre':       ['genre', 'genres', 'category', 'categories'],
    'rating':      ['rating', 'score', 'imdb', 'vote_average', 'stars'],
    'description': ['description', 'desc', 'plot', 'summary', 'overview', 'synopsis'],
    'director':    ['director', 'directed_by', 'directors'],
    'cast':        ['cast', 'actors', 'starring', 'stars', 'crew'],
    'runtime':     ['runtime', 'duration', 'length', 'minutes', 'mins'],
    'language':    ['language', 'lang', 'original_language'],
    'poster_url':  ['poster_url', 'poster', 'image', 'img', 'thumbnail', 'poster_path'],
    'trailer_id':  ['trailer_id', 'trailer', 'youtube', 'yt_id', 'youtube_id'],
}

REQUIRED_FIELDS = ['title', 'year', 'genre', 'rating', 'description']


def normalize_columns(df, column_map=None):
    """
    Rename CSV columns to our standard internal names.

    Two modes:
      1. Auto-detect  (column_map is None):   uses COLUMN_ALIASES to match headers.
      2. Manual map   (column_map is a dict): uses admin-supplied mapping from the
                                              CSV upload form (col_title=..., etc.)

    Returns the normalised DataFrame and a report dict.
    """
    df.columns = [c.strip() for c in df.columns]
    col_lower  = {c.lower(): c for c in df.columns}

    rename = {}

    if column_map:
        # Manual mapping supplied by the upload form
        for field, csv_col in column_map.items():
            if csv_col and csv_col in df.columns:
                rename[csv_col] = field
    else:
        # Auto-detect via aliases
        for field, aliases in COLUMN_ALIASES.items():
            if field in df.columns:
                continue  # already has the right name
            for alias in aliases:
                if alias in col_lower:
                    rename[col_lower[alias]] = field
                    break

    df = df.rename(columns=rename)

    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    report  = {'renamed': rename, 'missing_required': missing}
    return df, report


def validate_years(df, current_year=None):
    """Clamp / drop rows where year is out of range."""
    import datetime
    if current_year is None:
        current_year = datetime.datetime.now().year

    if 'year' not in df.columns:
        return df, []

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    bad = df[(df['year'] < 1888) | (df['year'] > current_year) | df['year'].isna()]
    if len(bad):
        print(f"⚠️  Dropping {len(bad)} rows with invalid year (allowed: 1888–{current_year})")
    df = df[(df['year'] >= 1888) & (df['year'] <= current_year)].copy()
    df['year'] = df['year'].astype(int)
    return df, bad.index.tolist()


def train_model(csv_path='data/movies.csv', column_map=None):
    """
    Train the recommendation model from any CSV file.

    Args:
        csv_path   : path to the CSV file
        column_map : optional dict {internal_field: csv_column_name}
                     If None, auto-detection via COLUMN_ALIASES is used.
    """
    # 1. Load
    df = pd.read_csv(csv_path)
    print(f"📄 Loaded {len(df)} rows from {csv_path}")

    # 2. Normalise columns
    df, report = normalize_columns(df, column_map)
    if report['renamed']:
        print(f"   → Renamed columns: {report['renamed']}")
    if report['missing_required']:
        raise ValueError(
            f"Missing required columns: {report['missing_required']}. "
            f"Available columns: {list(df.columns)}"
        )

    # 3. Validate years
    df, bad_years = validate_years(df)

    # 4. Assign sequential IDs if not present
    if 'id' not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, 'id', df.index + 1)

    # 5. Build feature string for TF-IDF
    #    Weight genre 3×, director 2× so the model favours genre/director similarity
    df['features'] = (
        (df.get('genre', pd.Series([''] * len(df))).fillna('') + ' ') * 3 +
        (df.get('description', pd.Series([''] * len(df))).fillna('') + ' ') +
        (df.get('title', pd.Series([''] * len(df))).fillna('') + ' ') +
        (df.get('director', pd.Series([''] * len(df))).fillna('') + ' ') * 2 +
        (df.get('cast', pd.Series([''] * len(df))).fillna(''))
    )

    # 6. TF-IDF + cosine similarity
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 7. Persist artefacts
    os.makedirs('model', exist_ok=True)
    with open('model/tfidf.pkl',      'wb') as f: pickle.dump(tfidf,      f)
    with open('model/cosine_sim.pkl', 'wb') as f: pickle.dump(cosine_sim, f)

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    with open('model/indices.pkl', 'wb') as f: pickle.dump(indices, f)

    # Also save the cleaned dataframe so app.py can reload it
    df.to_csv('data/movies.csv', index=False)

    print(f"✅ Model trained — {len(df)} movies, matrix {cosine_sim.shape}")
    return df, cosine_sim, indices


def _shared_genres(genres_a, genres_b):
    """Return list of genres shared between two pipe-separated genre strings."""
    a = set(g.strip() for g in str(genres_a).split('|') if g.strip())
    b = set(g.strip() for g in str(genres_b).split('|') if g.strip())
    return sorted(a & b)


def get_recommendations(title, cosine_sim, df, indices, top_n=6):
    """
    Return top_n similar movies with:
      - similarity_pct  : 0-100 integer
      - match_reasons   : list of human-readable strings explaining why they match
    """
    if title not in indices:
        return []

    source_row = df[df['title'] == title].iloc[0]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    results = df.iloc[[i for i, _ in sim_scores]].to_dict('records')

    for i, record in enumerate(results):
        raw  = float(sim_scores[i][1])
        pct  = round(raw * 100)
        record['similarity_pct'] = pct

        # ── Build match_reasons ───────────────────────────────────────────
        reasons = []

        # Shared genres
        shared = _shared_genres(
            source_row.get('genre', ''),
            record.get('genre', '')
        )
        if shared:
            reasons.append(f"Genre: {', '.join(shared)}")

        # Same director
        src_dir = str(source_row.get('director', '')).strip()
        rec_dir = str(record.get('director', '')).strip()
        if src_dir and rec_dir and src_dir.lower() == rec_dir.lower():
            reasons.append(f"Same director: {src_dir}")

        # Shared cast members
        src_cast = set(a.strip() for a in str(source_row.get('cast', '')).split('|') if a.strip())
        rec_cast = set(a.strip() for a in str(record.get('cast', '')).split('|') if a.strip())
        shared_cast = src_cast & rec_cast
        if shared_cast:
            reasons.append(f"Cast: {', '.join(list(shared_cast)[:2])}")

        # Same decade
        try:
            src_y = int(source_row.get('year', 0))
            rec_y = int(record.get('year', 0))
            if src_y and rec_y and abs(src_y - rec_y) <= 5:
                reasons.append(f"Same era ({rec_y})")
        except (ValueError, TypeError):
            pass

        record['match_reasons'] = reasons if reasons else ['Similar story & themes']

    return results


if __name__ == '__main__':
    train_model()
