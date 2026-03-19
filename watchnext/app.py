from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import pickle, os, io, math, datetime
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-me-in-production')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB upload limit

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'changeme')

DATA_PATH    = 'data/movies.csv'
MODEL_DIR    = 'model'
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
    movies    = [clean_movie(m) for m in df.to_dict('records')]
    genres    = sorted(set(g for row in df['genre'].fillna('') for g in row.split('|') if g))
    top_rated = df.nlargest(5, 'rating').to_dict('records')
    trending  = df.sample(min(5, len(df)), random_state=42).to_dict('records')
    return render_template('index.html', movies=movies, genres=genres,
                           top_rated=top_rated, trending=trending)


@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.get_json().get('title', '')
    recs  = [clean_movie(r) for r in get_recommendations(title)]
    return jsonify({'recommendations': recs, 'query': title})


@app.route('/search')
def search():
    q          = request.args.get('q', '').lower()
    genre      = request.args.get('genre', '')
    sort       = request.args.get('sort', '')
    rating_min = float(request.args.get('rating_min', 0))
    rating_max = float(request.args.get('rating_max', 10))
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
        res = res[res['genre'].str.contains(genre, na=False)]
    res = res[(res['rating'] >= rating_min) & (res['rating'] <= rating_max)]
    if sort == 'rating':  res = res.sort_values('rating', ascending=False)
    elif sort == 'year':  res = res.sort_values('year',   ascending=False)
    elif sort == 'title': res = res.sort_values('title')
    return jsonify({'movies': [clean_movie(m) for m in res.to_dict('records')]})


@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    global df
    row = df[df['id'] == movie_id]
    if row.empty:
        return redirect(url_for('index'))
    movie     = clean_movie(row.iloc[0].to_dict())
    recs      = [clean_movie(r) for r in get_recommendations(movie['title'])]
    cast_list = [c.strip() for c in str(movie.get('cast', '')).split('|') if c.strip()]
    return render_template('movie_detail.html', movie=movie,
                           recommendations=recs, cast_list=cast_list)


@app.route('/watchlist/toggle', methods=['POST'])
def watchlist_toggle():
    """Watchlist is stored client-side in localStorage."""
    return jsonify({'status': 'ok'})


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        if (request.form.get('username') == ADMIN_USERNAME and
                request.form.get('password') == ADMIN_PASSWORD):
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


if __name__ == '__main__':
    app.run(debug=True)
