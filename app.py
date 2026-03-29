from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, make_response
import pandas as pd
import pickle, os, io, math, datetime, json, hashlib
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

# Related genres — when filtering by one genre, also show similar genres
GENRE_RELATED = {
    'Animation':       ['Animation', 'Family', 'Adventure', 'Comedy', 'Fantasy'],
    'Family':          ['Family', 'Animation', 'Comedy', 'Adventure'],
    'Comedy':          ['Comedy', 'Romance', 'Family'],
    'Romance':         ['Romance', 'Comedy', 'Drama'],
    'Drama':           ['Drama', 'History', 'Crime', 'Romance'],
    'Action':          ['Action', 'Adventure', 'Thriller', 'Crime'],
    'Adventure':       ['Adventure', 'Action', 'Fantasy', 'Science Fiction'],
    'Thriller':        ['Thriller', 'Crime', 'Mystery', 'Horror'],
    'Horror':          ['Horror', 'Thriller', 'Mystery'],
    'Science Fiction': ['Science Fiction', 'Adventure', 'Action', 'Fantasy'],
    'Fantasy':         ['Fantasy', 'Adventure', 'Science Fiction', 'Animation'],
    'Crime':           ['Crime', 'Thriller', 'Drama', 'Action'],
    'Mystery':         ['Mystery', 'Thriller', 'Crime'],
    'History':         ['History', 'Drama', 'War'],
    'War':             ['War', 'History', 'Drama', 'Action'],
    'Music':           ['Music', 'Drama', 'Romance'],
    'Documentary':     ['Documentary', 'History'],
}

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-me-in-production')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB upload limit

ADMIN_JSON_PATH = 'data/admin.json'

def load_admin():
    """Load admin credentials from data/admin.json"""
    try:
        with open(ADMIN_JSON_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        # Fallback defaults if file missing
        return {'username': 'admin',
                'password': hashlib.sha256('admin@123'.encode()).hexdigest()}

DATA_PATH    = 'data/movies.csv'
MODEL_DIR    = 'model'
USERS_PATH   = 'data/users.json'
HISTORY_PATH = 'data/history.json'
CURRENT_YEAR = datetime.datetime.now().year

ALLOWED_EXTENSIONS = {'csv'}

# ── Column aliases for flexible CSV import ────────────────────────────────────
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


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Drop the auto-generated features column if present from old saves
    if 'features' in df.columns:
        df = df.drop(columns=['features'])
    for col in ['director', 'cast', 'language', 'trailer_id', 'poster_url']:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('')
    # Always store runtime as string so it accepts empty values and mixed input
    if 'runtime' not in df.columns:
        df['runtime'] = ''
    else:
        df['runtime'] = df['runtime'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True)
    # Ensure numeric columns have correct types
    df['year']   = pd.to_numeric(df['year'],   errors='coerce').fillna(0).astype(int)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0).astype(float)
    if 'id' not in df.columns:
        df.insert(0, 'id', df.index + 1)
    else:
        df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
    return df


def train_and_save_model(df):
    df = df.copy()
    # Weight genre 3x, director 2x for better matching
    df['_features'] = (
        (df['genre'].fillna('') + ' ') * 3 +
        df['description'].fillna('') + ' ' +
        df['title'].fillna('') + ' ' +
        (df['director'].fillna('') + ' ') * 2 +
        df['cast'].fillna('').str.replace('|', ' ', regex=False)
    )
    tfidf      = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    matrix     = tfidf.fit_transform(df['_features'])
    cosine_sim = cosine_similarity(matrix, matrix)
    indices    = pd.Series(df.index, index=df['title']).drop_duplicates()
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f'{MODEL_DIR}/tfidf.pkl',      'wb') as f: pickle.dump(tfidf,      f)
    with open(f'{MODEL_DIR}/cosine_sim.pkl', 'wb') as f: pickle.dump(cosine_sim, f)
    with open(f'{MODEL_DIR}/indices.pkl',    'wb') as f: pickle.dump(indices,    f)
    return cosine_sim, indices


def load_model():
    try:
        with open(f'{MODEL_DIR}/cosine_sim.pkl', 'rb') as f: cs  = pickle.load(f)
        with open(f'{MODEL_DIR}/indices.pkl',    'rb') as f: idx = pickle.load(f)
        return cs, idx
    except Exception:
        return train_and_save_model(load_data())


def save_data(df):
    """Save to CSV, always stripping internal temp columns."""
    out = df.copy()
    for col in ('_features', 'features'):
        if col in out.columns:
            out = out.drop(columns=[col])
    out.to_csv(DATA_PATH, index=False)


df         = load_data()
cosine_sim, indices = load_model()


# ── Utility ───────────────────────────────────────────────────────────────────
def clean_movie(m):
    """Make all values in a movie dict JSON-safe, strip internal columns."""
    result = {}
    for k, v in m.items():
        if k in ('_features', 'features'):
            continue
        if isinstance(v, float) and math.isnan(v):
            result[k] = ''
        else:
            result[k] = v
    return result


def validate_year(year_str):
    """Returns (int_year, error_msg). error_msg is None if valid."""
    try:
        y = int(year_str)
    except (ValueError, TypeError):
        return None, 'Year must be a number.'
    if y < 1888:
        return None, 'Year must be 1888 or later (first film ever made).'
    if y > CURRENT_YEAR:
        return None, f'Year cannot be in the future (max allowed: {CURRENT_YEAR}).'
    return y, None


def check_duplicate(title, year, exclude_id=None):
    """
    Returns True if a movie with the same title (case-insensitive) AND
    the same year already exists in df. Pass exclude_id when editing so
    the movie being edited doesn't flag itself as a duplicate.
    """
    mask = (
        df['title'].str.strip().str.lower() == title.strip().lower()
    ) & (
        df['year'] == int(year)
    )
    if exclude_id is not None:
        mask = mask & (df['id'] != int(exclude_id))
    return not df[mask].empty


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_csv_columns(uploaded_df, column_map=None):
    """
    Rename uploaded CSV columns to our standard names.
    column_map: {internal_field: csv_column} from upload form.
    Falls back to COLUMN_ALIASES auto-detection if None.
    Returns (df, list_of_missing_required_fields).
    """
    uploaded_df.columns = [c.strip() for c in uploaded_df.columns]
    col_lower = {c.lower(): c for c in uploaded_df.columns}
    rename = {}

    if column_map:
        for field, csv_col in column_map.items():
            if csv_col and csv_col in uploaded_df.columns:
                rename[csv_col] = field
    else:
        for field, aliases in COLUMN_ALIASES.items():
            if field in uploaded_df.columns:
                continue
            for alias in aliases:
                if alias in col_lower:
                    rename[col_lower[alias]] = field
                    break

    uploaded_df = uploaded_df.rename(columns=rename)
    missing = [f for f in REQUIRED_FIELDS if f not in uploaded_df.columns]
    return uploaded_df, missing


def _shared_genres(genres_a, genres_b):
    a = set(g.strip() for g in str(genres_a).split('|') if g.strip())
    b = set(g.strip() for g in str(genres_b).split('|') if g.strip())
    return sorted(a & b)


# ── User helpers ──────────────────────────────────────────────────────────────

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_PATH):
        return {}
    try:
        with open(USERS_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users):
    os.makedirs('data', exist_ok=True)
    with open(USERS_PATH, 'w') as f:
        json.dump(users, f, indent=2)

def load_history():
    if not os.path.exists(HISTORY_PATH):
        return {}
    try:
        with open(HISTORY_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_history(history):
    os.makedirs('data', exist_ok=True)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

def get_user_history(username):
    """Return list of movie IDs the user has viewed."""
    history = load_history()
    return history.get(username, [])

def add_to_history(username, movie_id):
    """Add a movie to user's watch history (max 50, no duplicates)."""
    if not username:
        return
    history = load_history()
    user_hist = history.get(username, [])
    movie_id  = int(movie_id)
    if movie_id in user_hist:
        user_hist.remove(movie_id)
    user_hist.insert(0, movie_id)
    history[username] = user_hist[:50]
    save_history(history)

def get_user_feedback_weights(username):
    """
    Read thumbs up/down feedback and return boost/penalty weights.
    👍 movies boost their genre+director in future recommendations.
    👎 movies penalise their genre so similar movies rank lower.
    """
    fb_path = 'data/feedback.json'
    try:
        with open(fb_path, 'r') as f:
            fb = json.load(f)
    except Exception:
        fb = {}

    user_fb       = fb.get(username, [])
    liked_ids     = set()
    disliked_ids  = set()
    genre_boost   = {}
    genre_penalty = {}
    dir_boost     = {}

    for entry in user_fb:
        mid  = int(entry.get('movie_id', 0))
        vote = entry.get('vote', '')
        row  = df[df['id'] == mid]
        if row.empty:
            continue
        row    = row.iloc[0]
        genres = [g.strip() for g in str(row.get('genre', '')).split('|') if g.strip()]
        direc  = str(row.get('director', '')).strip()

        if vote == 'up':
            liked_ids.add(mid)
            for g in genres:
                genre_boost[g] = genre_boost.get(g, 0) + 2
            if direc:
                dir_boost[direc] = dir_boost.get(direc, 0) + 2
        elif vote == 'down':
            disliked_ids.add(mid)
            for g in genres:
                genre_penalty[g] = genre_penalty.get(g, 0) + 1

    return genre_boost, genre_penalty, dir_boost, liked_ids, disliked_ids


def get_personalised_recommendations(username, top_n=10):
    """
    Personalised recommendations using 3 signals:
      1. Watch history  → what genres/directors user watches
      2. Thumbs UP  👍  → strongly boost those genres/directors
      3. Thumbs DOWN 👎 → penalise those genres
    Result: every user sees a completely different feed.
    """
    global df

    history = get_user_history(username)
    genre_boost, genre_penalty, dir_boost, liked_ids, disliked_ids =         get_user_feedback_weights(username)

    if not history and not liked_ids:
        return df.nlargest(top_n, 'rating').to_dict('records')

    watched_ids  = set(history[:20])
    watched_rows = df[df['id'].isin(watched_ids)]

    # Genre/director counts from watch history
    genre_counts = {}
    for g_str in watched_rows['genre'].fillna(''):
        for g in g_str.split('|'):
            g = g.strip()
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1

    dir_counts = {}
    for d in watched_rows['director'].fillna(''):
        if d.strip():
            dir_counts[d.strip()] = dir_counts.get(d.strip(), 0) + 1

    # Exclude watched + liked + disliked
    exclude_ids = watched_ids | liked_ids | disliked_ids

    scores = []
    for _, row in df.iterrows():
        if row['id'] in exclude_ids:
            continue
        score  = 0
        genres = [g.strip() for g in str(row.get('genre', '')).split('|') if g.strip()]
        direc  = str(row.get('director', '')).strip()

        # History signal
        for g in genres:
            score += genre_counts.get(g, 0) * 3
        score += dir_counts.get(direc, 0) * 2

        # 👍 Boost signal (strong)
        for g in genres:
            score += genre_boost.get(g, 0) * 4
        score += dir_boost.get(direc, 0) * 3

        # 👎 Penalty signal
        for g in genres:
            score -= genre_penalty.get(g, 0) * 3

        # Small rating bonus
        score += float(row.get('rating', 0)) * 0.3
        scores.append((score, row.to_dict()))

    scores.sort(key=lambda x: x[0], reverse=True)
    results = [clean_movie(m) for _, m in scores[:top_n]]

    if len(results) < top_n:
        extra = df[~df['id'].isin(exclude_ids)].nlargest(
            top_n - len(results), 'rating').to_dict('records')
        results += [clean_movie(m) for m in extra]

    return results


def get_trending_movies(top_n=10):
    """Return most viewed movies based on all users' history."""
    history = load_history()
    click_counts = {}
    for user_hist in history.values():
        for mid in user_hist:
            click_counts[mid] = click_counts.get(mid, 0) + 1

    if not click_counts:
        return df.nlargest(top_n, 'rating').to_dict('records')

    trending_ids = sorted(click_counts, key=click_counts.get, reverse=True)[:top_n]
    results = []
    for mid in trending_ids:
        row = df[df['id'] == mid]
        if not row.empty:
            results.append(clean_movie(row.iloc[0].to_dict()))
    return results

def user_login_required(f):
    """Decorator for user-only routes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_logged_in'):
            flash('Please log in to access that page.', 'error')
            return redirect(url_for('user_login'))
        return f(*args, **kwargs)
    return decorated


# ── Auth decorator ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


# ── Recommendations with similarity % and match reasons ──────────────────────
def get_recommendations(title, top_n=6):
    global cosine_sim, indices, df
    if title not in indices:
        return []

    source_rows = df[df['title'] == title]
    if source_rows.empty:
        return []
    source = source_rows.iloc[0]

    idx    = indices[title]
    scores = sorted(list(enumerate(cosine_sim[idx])),
                    key=lambda x: x[1], reverse=True)[1:top_n + 1]
    results = df.iloc[[i for i, _ in scores]].to_dict('records')

    for i, rec in enumerate(results):
        rec['similarity_pct'] = round(float(scores[i][1]) * 100)

        reasons = []

        shared_g = _shared_genres(source.get('genre', ''), rec.get('genre', ''))
        if shared_g:
            reasons.append(f"Genre: {', '.join(shared_g)}")

        src_dir = str(source.get('director', '')).strip()
        rec_dir = str(rec.get('director', '')).strip()
        if src_dir and rec_dir and src_dir.lower() == rec_dir.lower():
            reasons.append(f"Director: {src_dir}")

        src_cast = set(a.strip() for a in str(source.get('cast', '')).split('|') if a.strip())
        rec_cast = set(a.strip() for a in str(rec.get('cast', '')).split('|') if a.strip())
        shared_cast = src_cast & rec_cast
        if shared_cast:
            reasons.append(f"Cast: {', '.join(list(shared_cast)[:2])}")

        try:
            if abs(int(source.get('year', 0)) - int(rec.get('year', 0))) <= 5:
                reasons.append(f"Same era ({int(rec.get('year'))})")
        except (ValueError, TypeError):
            pass

        rec['match_reasons'] = reasons if reasons else ['Similar story & themes']

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    global df
    genres   = sorted(set(g for row in df['genre'].fillna('') for g in row.split('|') if g))
    username = session.get('username')

    if username:
        hist_ids      = get_user_history(username)
        history_count = len(hist_ids)

        # ── Recency-weighted genre scoring ────────────────────────────────
        # Recent clicks matter MORE than old ones.
        # Position 0 = most recent (weight 10), position 49 = oldest (weight 1)
        genre_scores = {}
        dir_scores   = {}
        for rank, mid in enumerate(hist_ids[:30]):
            weight = max(1, 10 - rank)          # 10 for newest, down to 1
            row    = df[df['id'] == mid]
            if row.empty:
                continue
            row = row.iloc[0]
            for g in str(row.get('genre', '')).split('|'):
                g = g.strip()
                if g:
                    genre_scores[g] = genre_scores.get(g, 0) + weight
            d = str(row.get('director', '')).strip()
            if d:
                dir_scores[d] = dir_scores.get(d, 0) + weight

        # Normalise so no single genre dominates just because user clicked it once
        if genre_scores:
            max_gs = max(genre_scores.values())
            genre_scores = {g: v / max_gs for g, v in genre_scores.items()}
        if dir_scores:
            max_ds = max(dir_scores.values())
            dir_scores = {d: v / max_ds for d, v in dir_scores.items()}

        # ── Feedback weights ───────────────────────────────────────────────
        gb, gp, db, liked_ids, disliked_ids = get_user_feedback_weights(username)

        # ── Watched IDs to exclude from top picks ─────────────────────────
        watched_ids = set(hist_ids)

        def score_movie(row):
            s      = 0.0
            genres = [g.strip() for g in str(row.get('genre', '')).split('|') if g.strip()]
            direc  = str(row.get('director', '')).strip()
            mid    = int(row.get('id', 0))

            # Already watched — push to bottom
            if mid in watched_ids:
                return -999

            # History signal (normalised 0-1 × weights)
            for g in genres:
                s += genre_scores.get(g, 0) * 5
            s += dir_scores.get(direc, 0) * 3

            # 👍 Feedback boost
            for g in genres:
                s += gb.get(g, 0) * 0.4
            s += db.get(direc, 0) * 0.3

            # 👎 Feedback penalty
            for g in genres:
                s -= gp.get(g, 0) * 0.3

            # Rating bonus (minor)
            s += float(row.get('rating', 0)) * 0.05
            return s

        records = df.to_dict('records')
        if genre_scores or gb:
            records.sort(key=lambda r: score_movie(r), reverse=True)

        # ── Diversity: interleave across ALL genres user has shown interest in
        # Each genre gets a slot in rotation so no single genre floods the feed
        if genre_scores:
            # Build buckets for every genre user scored, ordered by score desc
            genre_buckets = {}
            for r in records:
                if int(r.get('id', 0)) in watched_ids:
                    continue
                for pg in str(r.get('genre', '')).split('|'):
                    pg = pg.strip()
                    if pg in genre_scores:
                        genre_buckets.setdefault(pg, []).append(r)
                        break

            # Also add an "other" bucket for movies not matching user genres
            other_bucket = [r for r in records
                            if int(r.get('id', 0)) not in watched_ids
                            and not any(
                                g.strip() in genre_scores
                                for g in str(r.get('genre', '')).split('|')
                            )]

            # Sort genre buckets by user preference score (highest first)
            sorted_genres = sorted(genre_scores, key=genre_scores.get, reverse=True)
            bucket_iters  = {g: iter(genre_buckets.get(g, [])) for g in sorted_genres}

            interleaved = []
            added       = set()
            # Round-robin: take 1 from each genre in turn, repeat
            max_rounds = len(records)
            for _ in range(max_rounds):
                any_added = False
                for g in sorted_genres:
                    it  = bucket_iters[g]
                    mov = next((m for m in it if m.get('id') not in added), None)
                    if mov:
                        interleaved.append(mov)
                        added.add(mov.get('id'))
                        any_added = True
                if not any_added:
                    break

            # Fill remaining with "other" movies
            for r in other_bucket:
                if r.get('id') not in added:
                    interleaved.append(r)
                    added.add(r.get('id'))

            # Append any remaining scored movies not yet added
            for r in records:
                if r.get('id') not in added and int(r.get('id', 0)) not in watched_ids:
                    interleaved.append(r)

            records = interleaved if interleaved else records

        # Put already-watched movies at the very end
        watched_records = [r for r in df.to_dict('records') if int(r.get('id', 0)) in watched_ids]
        records = records + watched_records

        movies  = [clean_movie(m) for m in records]
        for_you = movies[:10]
    else:
        # Guest: rank by session clicks (no save, resets on browser close)
        guest_clicks = session.get('guest_clicks', [])
        if guest_clicks:
            watched_rows = df[df['id'].isin(guest_clicks)]
            genre_counts = {}
            for g_str in watched_rows['genre'].fillna(''):
                for g in g_str.split('|'):
                    g = g.strip()
                    if g: genre_counts[g] = genre_counts.get(g, 0) + 1
            dir_counts = {}
            for d in watched_rows['director'].fillna(''):
                if d.strip(): dir_counts[d.strip()] = dir_counts.get(d.strip(), 0) + 1

            def guest_score(row):
                s = 0
                for g in str(row.get('genre', '')).split('|'):
                    s += genre_counts.get(g.strip(), 0) * 3
                s += dir_counts.get(str(row.get('director', '')).strip(), 0) * 2
                return s

            records = df.to_dict('records')
            records.sort(key=lambda r: guest_score(r), reverse=True)
            movies  = [clean_movie(m) for m in records]
        else:
            movies = [clean_movie(m) for m in df.to_dict('records')]
        for_you       = []
        history_count = 0

    resp = make_response(render_template('index.html',
                           movies=movies,
                           genres=genres,
                           for_you=for_you,
                           username=username,
                           history_count=history_count,
                           is_guest_personalised=bool(not username and session.get('guest_clicks'))))
    # Never cache homepage — always fetch fresh personalised rankings
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma']        = 'no-cache'
    return resp


@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.get_json().get('title', '')
    recs  = [clean_movie(r) for r in get_recommendations(title)]
    return jsonify({'recommendations': recs, 'query': title})


@app.route('/search')
def search():
    q     = request.args.get('q', '').lower()
    genre = request.args.get('genre', '')
    sort  = request.args.get('sort', '')
    global df
    res = df.copy()
    if q:
        res = res[
            res['title'].str.lower().str.contains(q, na=False)        |
            res['description'].str.lower().str.contains(q, na=False)  |
            res['cast'].str.lower().str.contains(q, na=False)         |
            res['director'].str.lower().str.contains(q, na=False)
        ]
    if genre:
        related = GENRE_RELATED.get(genre, [genre])
        mask = res['genre'].str.contains(related[0], na=False)
        for g in related[1:]:
            mask = mask | res['genre'].str.contains(g, na=False)
        res = res[mask]
        # Sort: exact genre match first, then related
        res = res.copy()
        res['_exact'] = res['genre'].str.contains(genre, na=False).astype(int)
        res = res.sort_values('_exact', ascending=False).drop(columns=['_exact'])
    if sort == 'rating': res = res.sort_values('rating', ascending=False)
    elif sort == 'year':  res = res.sort_values('year',   ascending=False)
    elif sort == 'title': res = res.sort_values('title')

    movies       = [clean_movie(m) for m in res.to_dict('records')]
    not_found    = False
    suggestions  = []

    # If no results found for a search query → suggest similar titles
    if q and not movies:
        not_found = True
        # Fuzzy-style: find movies where any word in query matches
        words = q.split()
        for word in words:
            if len(word) < 3:
                continue
            partial = df[
                df['title'].str.lower().str.contains(word, na=False) |
                df['genre'].str.lower().str.contains(word, na=False) |
                df['director'].str.lower().str.contains(word, na=False)
            ]
            suggestions.extend(partial.to_dict('records'))

        # Deduplicate and take top 6
        seen = set()
        unique_sugg = []
        for m in suggestions:
            if m['id'] not in seen:
                seen.add(m['id'])
                unique_sugg.append(clean_movie(m))
        suggestions = unique_sugg[:6]

    return jsonify({
        'movies':      movies,
        'not_found':   not_found,
        'query':       q,
        'suggestions': suggestions
    })


@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    global df
    row = df[df['id'] == movie_id]
    if row.empty:
        return redirect(url_for('index'))
    movie     = clean_movie(row.iloc[0].to_dict())
    recs      = [clean_movie(r) for r in get_recommendations(movie['title'])]
    cast_list = [c.strip() for c in str(movie.get('cast', '')).split('|') if c.strip()]

    username = session.get('username')
    if username:
        # Logged-in: save to persistent history
        add_to_history(username, movie_id)
    else:
        # Guest: save to session only (no DB write, resets when browser closes)
        guest_clicks = session.get('guest_clicks', [])
        if movie_id in guest_clicks:
            guest_clicks.remove(movie_id)
        guest_clicks.insert(0, movie_id)
        session['guest_clicks'] = guest_clicks[:20]

    return render_template('movie_detail.html', movie=movie,
                           recommendations=recs, cast_list=cast_list,
                           username=username)


@app.route('/feedback', methods=['POST'])
def feedback():
    """Store thumbs up/down feedback for recommendations."""
    data     = request.get_json()
    movie_id = int(data.get('movie_id', 0))
    vote     = data.get('vote', '')        # 'up' or 'down'
    username = session.get('username', 'anonymous')

    if not movie_id or vote not in ('up', 'down'):
        return jsonify({'status': 'error'}), 400

    # Load existing feedback
    fb_path = 'data/feedback.json'
    try:
        with open(fb_path, 'r') as f:
            fb = json.load(f)
    except Exception:
        fb = {}

    # Store: {username: [{movie_id, vote, timestamp}]}
    entry = {
        'movie_id':  movie_id,
        'vote':      vote,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    fb.setdefault(username, [])
    # Remove old vote for same movie
    fb[username] = [x for x in fb[username] if x['movie_id'] != movie_id]
    fb[username].append(entry)

    os.makedirs('data', exist_ok=True)
    with open(fb_path, 'w') as f:
        json.dump(fb, f, indent=2)

    # Count user's total feedback for response message
    user_fb   = fb.get(username, [])
    liked_cnt = sum(1 for x in user_fb if x['vote'] == 'up')
    disliked_cnt = sum(1 for x in user_fb if x['vote'] == 'down')

    return jsonify({
        'status':       'ok',
        'liked_count':  liked_cnt,
        'disliked_count': disliked_cnt,
        'message': (
            f'Your feed now boosts movies like this one.'
            if vote == 'up'
            else f'Got it! We will show fewer movies like this.'
        )
    })


@app.route('/user/search_for_you')
def user_search_for_you():
    """
    When a logged-in user searches, return results ranked by
    how well they match the user's taste (genre/director preference).
    """
    global df
    q      = request.args.get('q', '').lower()
    genre  = request.args.get('genre', '')
    sort   = request.args.get('sort', '')
    username = session.get('username')

    # Start with normal search filter
    res = df.copy()
    if q:
        res = res[
            res['title'].str.lower().str.contains(q, na=False) |
            res['description'].str.lower().str.contains(q, na=False) |
            res['cast'].str.lower().str.contains(q, na=False) |
            res['director'].str.lower().str.contains(q, na=False)
        ]
    if genre:
        related = GENRE_RELATED.get(genre, [genre])
        mask = res['genre'].str.contains(related[0], na=False)
        for g in related[1:]:
            mask = mask | res['genre'].str.contains(g, na=False)
        res  = res[mask].copy()
        res['_exact'] = res['genre'].str.contains(genre, na=False).astype(int)
        res = res.sort_values('_exact', ascending=False).drop(columns=['_exact'])

    if sort == 'rating': res = res.sort_values('rating', ascending=False)
    elif sort == 'year':  res = res.sort_values('year',   ascending=False)
    elif sort == 'title': res = res.sort_values('title')

    if not username or res.empty:
        return jsonify({'movies': [clean_movie(m) for m in res.to_dict('records')]})

    # Score results by user preference
    hist_ids     = get_user_history(username)
    watched_rows = df[df['id'].isin(hist_ids)]

    genre_counts = {}
    for g_str in watched_rows['genre'].fillna(''):
        for g in g_str.split('|'):
            g = g.strip()
            if g: genre_counts[g] = genre_counts.get(g, 0) + 1

    dir_counts = {}
    for d in watched_rows['director'].fillna(''):
        if d.strip(): dir_counts[d.strip()] = dir_counts.get(d.strip(), 0) + 1

    gb, gp, db, liked_ids, disliked_ids = get_user_feedback_weights(username)

    def score_row(row):
        s      = 0
        genres = [g.strip() for g in str(row.get('genre', '')).split('|') if g.strip()]
        direc  = str(row.get('director', '')).strip()
        for g in genres:
            s += genre_counts.get(g, 0) * 3
        s += dir_counts.get(direc, 0) * 2
        for g in genres:
            s += gb.get(g, 0) * 4
        s += db.get(direc, 0) * 3
        for g in genres:
            s -= gp.get(g, 0) * 3
        return s

    records = res.to_dict('records')
    if genre_counts or gb:
        records.sort(key=lambda r: score_row(r), reverse=True)

    return jsonify({'movies': [clean_movie(m) for m in records]})
    """Watchlist is stored client-side in localStorage."""
    # Return updated for-you list so frontend can refresh instantly
    username = session.get('username', 'anonymous')
    updated  = []
    if username != 'anonymous':
        updated = get_personalised_recommendations(username, top_n=10)

    return jsonify({'status': 'ok', 'updated_picks': updated})


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        admin     = load_admin()
        uname     = request.form.get('username', '').strip()
        pwd_hash  = hashlib.sha256(
                      request.form.get('password', '').encode()
                    ).hexdigest()
        if uname == admin['username'] and pwd_hash == admin['password']:
            session['admin_logged_in'] = True
            flash('Welcome back, Admin!', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('Invalid credentials.', 'error')
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))


@app.route('/admin')
@login_required
def admin_dashboard():
    global df
    movies     = [clean_movie(m) for m in df.to_dict('records')]
    genres_set = set(g for row in df['genre'].fillna('') for g in row.split('|') if g)
    genre_counts = {}
    for row in df['genre'].fillna(''):
        for g in row.split('|'):
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    rating_dist = {
        str(i): int(((df['rating'] >= i) & (df['rating'] < i + 1)).sum())
        for i in range(6, 11)
    }
    stats = {
        'total':        len(df),
        'genres':       len(genres_set),
        'avg_rating':   round(float(df['rating'].mean()), 1),
        'top_rated':    df.nlargest(1, 'rating').iloc[0]['title'],
        'genre_counts': genre_counts,
        'rating_dist':  rating_dist,
    }
    return render_template('admin_dashboard.html', movies=movies, stats=stats)


@app.route('/admin/movie/add', methods=['GET', 'POST'])
@login_required
def admin_add_movie():
    global df, cosine_sim, indices
    if request.method == 'POST':
        title = request.form.get('title', '').strip()

        year, year_err = validate_year(request.form.get('year', ''))
        if year_err:
            flash(year_err, 'error')
            return redirect(url_for('admin_add_movie'))

        # ── Duplicate check: same title (case-insensitive) + same year ────────
        if check_duplicate(title, year):
            flash(
                f'Duplicate movie detected: "{title}" ({year}) already exists in the database. '
                f'A movie with the same title and release year cannot be added twice.',
                'error'
            )
            return redirect(url_for('admin_add_movie'))

        new = {
            'id':          int(df['id'].max()) + 1,
            'title':       title,
            'genre':       request.form.get('genre', '').strip(),
            'description': request.form.get('description', '').strip(),
            'year':        year,
            'rating':      float(request.form.get('rating', 0)),
            'poster_url':  request.form.get('poster_url', '').strip() or
                           'https://via.placeholder.com/300x450/e63946/ffffff?text=' +
                           title.replace(' ', '+'),
            'director':    request.form.get('director', '').strip(),
            'cast':        request.form.get('cast', '').strip(),
            'runtime':     request.form.get('runtime', '').strip(),
            'language':    request.form.get('language', '').strip(),
            'trailer_id':  request.form.get('trailer_id', '').strip(),
        }
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        save_data(df)
        cosine_sim, indices = train_and_save_model(df)
        flash(f'"{new["title"]}" added and model retrained!', 'success')
        return redirect(url_for('admin_dashboard'))

    return render_template('admin_add_movie.html')


@app.route('/admin/movie/edit/<int:movie_id>', methods=['GET', 'POST'])
@login_required
def admin_edit_movie(movie_id):
    global df, cosine_sim, indices
    row = df[df['id'] == movie_id]
    if row.empty:
        flash('Movie not found.', 'error')
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        title = request.form.get('title', '').strip()

        year, year_err = validate_year(request.form.get('year', ''))
        if year_err:
            flash(year_err, 'error')
            return redirect(url_for('admin_edit_movie', movie_id=movie_id))

        # -- Duplicate check: exclude current movie from the check -------------
        if check_duplicate(title, year, exclude_id=movie_id):
            flash(
                f'Duplicate movie detected: "{title}" ({year}) already exists in the database. '
                f'A movie with the same title and release year cannot be saved.',
                'error'
            )
            return redirect(url_for('admin_edit_movie', movie_id=movie_id))

        idx = df.index[df['id'] == movie_id][0]

        # String fields
        for field in ['title', 'genre', 'description', 'director',
                      'cast', 'language', 'trailer_id', 'poster_url']:
            df.at[idx, field] = request.form.get(field, '').strip()

        # Numeric fields — must match the column's actual dtype
        df.at[idx, 'year'] = int(year)

        try:
            df.at[idx, 'rating'] = float(request.form.get('rating', 0))
        except (ValueError, TypeError):
            flash('Invalid rating value.', 'error')
            return redirect(url_for('admin_edit_movie', movie_id=movie_id))

        # Runtime: column is always string
        runtime_raw = request.form.get('runtime', '').strip()
        df.at[idx, 'runtime'] = runtime_raw.replace('.0', '') if runtime_raw else ''

        # Re-normalize numeric columns in case of any drift
        df['year']   = pd.to_numeric(df['year'],   errors='coerce').fillna(0).astype(int)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0).astype(float)

        save_data(df)
        cosine_sim, indices = train_and_save_model(df)
        flash('Movie updated and model retrained!', 'success')
        return redirect(url_for('admin_dashboard'))

    return render_template('admin_edit_movie.html',
                           movie=clean_movie(row.iloc[0].to_dict()))


@app.route('/admin/movie/delete/<int:movie_id>', methods=['POST'])
@login_required
def admin_delete_movie(movie_id):
    global df, cosine_sim, indices
    match = df[df['id'] == movie_id]
    title = match.iloc[0]['title'] if not match.empty else 'Movie'
    df    = df[df['id'] != movie_id].reset_index(drop=True)
    save_data(df)
    cosine_sim, indices = train_and_save_model(df)
    flash(f'"{title}" deleted.', 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/upload-csv', methods=['POST'])
@login_required
def admin_upload_csv():
    global df, cosine_sim, indices

    file = request.files.get('csv_file')
    if not file or file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('admin_add_movie'))
    if not allowed_file(file.filename):
        flash('Only .csv files are allowed.', 'error')
        return redirect(url_for('admin_add_movie'))

    # Parse
    try:
        uploaded_df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    except Exception as e:
        flash(f'Could not parse CSV: {e}', 'error')
        return redirect(url_for('admin_add_movie'))

    # Build column map from form (col_title, col_year, …)
    column_map = {}
    all_fields = REQUIRED_FIELDS + ['director', 'cast', 'runtime',
                                     'language', 'poster_url', 'trailer_id']
    for field in all_fields:
        val = request.form.get(f'col_{field}', '').strip()
        if val and val not in ('', '— skip —', '-- skip --'):
            column_map[field] = val

    # Normalise columns
    uploaded_df, missing = normalize_csv_columns(uploaded_df, column_map or None)
    if missing:
        flash(
            f"Missing required columns: {', '.join(missing)}. "
            f"Please re-map your columns and try again.",
            'error'
        )
        return redirect(url_for('admin_add_movie'))

    # Validate years
    uploaded_df['year'] = pd.to_numeric(uploaded_df['year'], errors='coerce')
    bad_count = int(uploaded_df[
        (uploaded_df['year'] < 1888) |
        (uploaded_df['year'] > CURRENT_YEAR) |
        uploaded_df['year'].isna()
    ].shape[0])
    uploaded_df = uploaded_df[
        (uploaded_df['year'] >= 1888) &
        (uploaded_df['year'] <= CURRENT_YEAR)
    ].copy()
    uploaded_df['year'] = uploaded_df['year'].astype(int)

    if bad_count:
        flash(f'Skipped {bad_count} rows with invalid or future years.', 'warning')
    if uploaded_df.empty:
        flash('No valid rows found after validation.', 'error')
        return redirect(url_for('admin_add_movie'))

    # Merge & deduplicate by title + year (not just title)
    merged = pd.concat([df, uploaded_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=['title', 'year'], keep='first')
    merged = merged.reset_index(drop=True)
    merged['id'] = merged.index + 1
    for col in ['director', 'cast', 'runtime', 'language', 'trailer_id', 'poster_url']:
        if col not in merged.columns:
            merged[col] = ''
    save_data(merged)

    # Retrain
    df = merged
    cosine_sim, indices = train_and_save_model(df)

    flash(f'✅ {len(uploaded_df)} movies imported and model retrained!', 'success')
    return redirect(url_for('admin_dashboard'))



# ══════════════════════════════════════════════════════════════════════════════
# USER AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/user/register', methods=['GET', 'POST'])
def user_register():
    if session.get('user_logged_in'):
        return redirect(url_for('user_dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '').strip()
        confirm  = request.form.get('confirm',  '').strip()

        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('user_register'))
        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
            return redirect(url_for('user_register'))
        if ' ' in username:
            flash('Username cannot contain spaces.', 'error')
            return redirect(url_for('user_register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return redirect(url_for('user_register'))
        if password != confirm:
            flash('Passwords do not match. Please try again.', 'error')
            return redirect(url_for('user_register'))

        # Check username already taken
        users = load_users()
        if username in users:
            flash(f'Username "{username}" is already taken. Please choose another.', 'error')
            return redirect(url_for('user_register'))

        # Save new user
        users[username] = {
            'password':   hash_password(password),
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'name':       username,
        }
        save_users(users)
        session['user_logged_in'] = True
        session['username']       = username
        session['user_name']      = username
        flash(f'Welcome to WatchNext, {username}!', 'success')
        return redirect(url_for('index'))

    return render_template('user_register.html')


@app.route('/user/login', methods=['GET', 'POST'])
def user_login():
    if session.get('user_logged_in'):
        return redirect(url_for('user_dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '').strip()
        users    = load_users()

        if username not in users or \
           users[username]['password'] != hash_password(password):
            flash('Invalid username or password.', 'error')
            return redirect(url_for('user_login'))

        session['user_logged_in'] = True
        session['username']       = username
        session['user_name']      = users[username].get('name', username)
        flash(f'Welcome back, {session["user_name"]}!', 'success')
        return redirect(url_for('index'))

    return render_template('user_login.html')


@app.route('/user/logout')
def user_logout():
    session.pop('user_logged_in', None)
    session.pop('username',       None)
    session.pop('user_name',      None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))


@app.route('/user/dashboard')
@user_login_required
def user_dashboard():
    global df
    username = session.get('username')
    name     = session.get('user_name', username)

    # Watch history
    hist_ids   = get_user_history(username)
    hist_movies = []
    for mid in hist_ids[:20]:
        row = df[df['id'] == mid]
        if not row.empty:
            hist_movies.append(clean_movie(row.iloc[0].to_dict()))

    # Personalised recommendations
    for_you = get_personalised_recommendations(username, top_n=12)

    # Favourite genre from history
    genre_counts = {}
    watched_rows = df[df['id'].isin(hist_ids)]
    for g_str in watched_rows['genre'].fillna(''):
        for g in g_str.split('|'):
            g = g.strip()
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    fav_genre = max(genre_counts, key=genre_counts.get) if genre_counts else None

    # Favourite director
    dir_counts = {}
    for d in watched_rows['director'].fillna(''):
        if d.strip():
            dir_counts[d.strip()] = dir_counts.get(d.strip(), 0) + 1
    fav_director = max(dir_counts, key=dir_counts.get) if dir_counts else None

    # Feedback stats
    fb_path = 'data/feedback.json'
    try:
        with open(fb_path, 'r') as f2:
            fb_data = json.load(f2)
    except Exception:
        fb_data = {}
    user_fb      = fb_data.get(username, [])
    liked_count  = sum(1 for x in user_fb if x['vote'] == 'up')
    disliked_count = sum(1 for x in user_fb if x['vote'] == 'down')

    stats = {
        'watched':        len(hist_ids),
        'fav_genre':      fav_genre,
        'fav_director':   fav_director,
        'avg_rating':     round(float(watched_rows['rating'].mean()), 1)
                          if not watched_rows.empty else 0,
        'liked_count':    liked_count,
        'disliked_count': disliked_count,
    }

    return render_template('user_dashboard.html',
                           name=name, username=username,
                           history=hist_movies,
                           for_you=for_you,
                           stats=stats)


@app.route('/user/history/clear', methods=['POST'])
@user_login_required
def clear_history():
    username = session.get('username')
    history  = load_history()
    history[username] = []
    save_history(history)
    flash('Watch history cleared.', 'success')
    return redirect(url_for('user_dashboard'))


@app.route('/user/refresh_for_you')
def refresh_for_you():
    """
    AJAX endpoint — returns fresh personalised picks after
    user gives thumbs up/down feedback or watches new movies.
    """
    username = session.get('username')
    if not username:
        return jsonify({'movies': []})
    movies = get_personalised_recommendations(username, top_n=10)
    return jsonify({'movies': movies})




# ══════════════════════════════════════════════════════════════════════════════
# ADMIN — USER MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/admin/users')
@login_required
def admin_users():
    users   = load_users()
    history = load_history()
    fb_path = 'data/feedback.json'
    try:
        with open(fb_path) as f:
            fb = json.load(f)
    except Exception:
        fb = {}

    user_list = []
    for uname, data in users.items():
        user_list.append({
            'username':   uname,
            'name':       data.get('name', uname),
            'created_at': data.get('created_at', '—'),
            'movies_watched': len(history.get(uname, [])),
            'feedback_count': len(fb.get(uname, [])),
        })
    user_list.sort(key=lambda x: x['username'])
    return render_template('admin_users.html', users=user_list)


@app.route('/admin/users/delete/<username>', methods=['POST'])
@login_required
def admin_delete_user(username):
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
        # Also clear their history and feedback
        history = load_history()
        if username in history:
            del history[username]
            save_history(history)
        flash(f'User "{username}" deleted.', 'success')
    else:
        flash(f'User "{username}" not found.', 'error')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/reset_password/<username>', methods=['POST'])
@login_required
def admin_reset_password(username):
    users    = load_users()
    new_pwd  = request.form.get('new_password', '').strip()
    if not new_pwd or len(new_pwd) < 6:
        flash('Password must be at least 6 characters.', 'error')
        return redirect(url_for('admin_users'))
    if username not in users:
        flash(f'User "{username}" not found.', 'error')
        return redirect(url_for('admin_users'))
    users[username]['password'] = hash_password(new_pwd)
    save_users(users)
    flash(f'Password for "{username}" updated.', 'success')
    return redirect(url_for('admin_users'))


if __name__ == '__main__':
    app.run(debug=True)
