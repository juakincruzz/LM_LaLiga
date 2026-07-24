import aiosqlite
from pathlib import Path

DB_PATH = Path("data/db.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS leagues (
    id INTEGER PRIMARY KEY,
    code TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    country TEXT,
    season TEXT,
    last_updated TEXT,
    is_active INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    league_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    alias TEXT,
    FOREIGN KEY (league_id) REFERENCES leagues(id)
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    league_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_goals INTEGER,
    away_goals INTEGER,
    result TEXT,
    season TEXT,
    matchday INTEGER,
    status TEXT DEFAULT 'played',
    FOREIGN KEY (league_id) REFERENCES leagues(id),
    FOREIGN KEY (home_team_id) REFERENCES teams(id),
    FOREIGN KEY (away_team_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY,
    league_id INTEGER NOT NULL,
    created_at TEXT,
    status TEXT DEFAULT 'pending',
    model_type TEXT,
    accuracy REAL,
    log TEXT,
    FOREIGN KEY (league_id) REFERENCES leagues(id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    model_run_id INTEGER NOT NULL,
    match_id INTEGER NOT NULL,
    predicted_result TEXT,
    prob_home REAL,
    prob_draw REAL,
    prob_away REAL,
    FOREIGN KEY (model_run_id) REFERENCES model_runs(id),
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS simulations (
    id INTEGER PRIMARY KEY,
    model_run_id INTEGER NOT NULL,
    sim_count INTEGER,
    standings_json TEXT,
    FOREIGN KEY (model_run_id) REFERENCES model_runs(id)
);

CREATE TABLE IF NOT EXISTS transfers (
    id INTEGER PRIMARY KEY,
    league_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    player_name TEXT,
    position TEXT,
    transfer_type TEXT,
    market_value_eur REAL,
    from_club TEXT,
    to_club TEXT,
    date TEXT,
    FOREIGN KEY (league_id) REFERENCES leagues(id),
    FOREIGN KEY (team_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS job_status (
    id INTEGER PRIMARY KEY,
    job_type TEXT NOT NULL,
    league_id INTEGER,
    status TEXT DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    result_json TEXT,
    created_at TEXT,
    updated_at TEXT
);
"""


async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()


async def get_db():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db
