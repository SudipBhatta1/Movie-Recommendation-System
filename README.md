# WatchNext — Movie Recommender

A Flask-based movie recommender with ML-powered similarity search, admin panel, and watchlist.

---

## First Time Setup

### 1. Create the virtual environment
Do this only once after extracting the project.
```
cd watchnext
python -m venv env
```

### 2. Activate the virtual environment
Run this every time you open a new terminal.

Windows:
```
env\Scripts\activate
```

Mac / Linux:
```
source env/bin/activate
```

Your prompt will change to `(env)` — that confirms it is active.

### 3. Install dependencies
Do this once per fresh env.
```
pip install -r requirements.txt
```

### 4. Run the app
```
python app.py
```

Visit:  http://127.0.0.1:5000
Admin:  http://127.0.0.1:5000/admin

### 5. Deactivate when done
```
deactivate
```

---

## Daily Workflow

Every day, just run these three commands:
```
cd watchnext
env\Scripts\activate
python app.py
```

---

## Troubleshooting

### Activation blocked on Windows
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again.

### env broken or corrupted — reset it completely
```
rmdir /s /q env
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### Model files missing (first run or after reset)
The model files in model/ are auto-generated — just run python app.py and they will be created automatically from data/movies.csv.

You can also retrain manually:
```
python model/train.py
```

### Test model accuracy
```
python test_model.py
```

---

## Environment Variables

The .env file is already included. Edit it to change credentials:
```
SECRET_KEY=your-secret-key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin@1
TMDB_API_KEY=your-tmdb-key
```

Never commit .env to git — it is already in .gitignore.

---

## Project Structure

```
watchnext/
├── app.py               # Main Flask app
├── requirements.txt     # Python dependencies
├── test_model.py        # Model accuracy tests
├── .env                 # Environment variables (do not commit)
├── .gitignore
├── data/
│   └── movies.csv       # Movie dataset
├── model/
│   └── train.py         # ML training script (pkl files are auto-generated)
├── static/
│   ├── css/style.css
│   └── js/main.js
└── templates/
    ├── base.html
    ├── index.html
    ├── movie_detail.html
    ├── admin_login.html
    ├── admin_dashboard.html
    ├── admin_add_movie.html
    └── admin_edit_movie.html
```

---

## Features

- ML-based movie recommendations (TF-IDF + cosine similarity)
- Search by title, actor, director
- Filter by genre and rating
- Watchlist stored in browser localStorage
- Admin panel: add, edit, delete movies
- CSV bulk import with auto column mapping
- Duplicate prevention: same title + same release year is blocked on add and edit
- Model auto-retrains on every data change

---

## Notes

- The env/ folder is large (~200-400 MB) — this is normal. Never commit or share it. Share only requirements.txt and others recreate it with pip install -r requirements.txt.
- The model/*.pkl files are auto-generated at runtime and are gitignored.
- Always activate the env before running any python command.