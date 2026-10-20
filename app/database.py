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
    league_code TEXT NOT NULL,
    name TEXT NOT NULL,
    alias TEXT,
    FOREIGN KEY (league_code) REFERENCES leagues(code)
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    league_code TEXT NOT NULL,
    date TEXT NOT NULL,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_goals INTEGER,
    away_goals INTEGER,
    result TEXT,
    season TEXT,
    matchday INTEGER,
    status TEXT DEFAULT 'played',
    FOREIGN KEY (league_code) REFERENCES leagues(code),
    FOREIGN KEY (home_team_id) REFERENCES teams(id),
    FOREIGN KEY (away_team_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY,
    league_code TEXT NOT NULL,
    created_at TEXT,
    status TEXT DEFAULT 'pending',
    model_type TEXT,
    accuracy REAL,
    log TEXT,
    FOREIGN KEY (league_code) REFERENCES leagues(code)
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
    model_run_id INTEGER,
    sim_count INTEGER,
    standings_json TEXT,
    label TEXT DEFAULT 'Temporada actual',
    season TEXT,
    FOREIGN KEY (model_run_id) REFERENCES model_runs(id)
);

CREATE TABLE IF NOT EXISTS job_status (
    id INTEGER PRIMARY KEY,
    job_type TEXT NOT NULL,
    league_code TEXT,
    status TEXT DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    result_json TEXT,
    created_at TEXT,
    updated_at TEXT,
    log TEXT
);
"""


# Migraciones idempotentes: se intentan en cada arranque y fallan en silencio si
# la columna ya existe. Es la estrategia adecuada para una BD local desechable;
# añadir aquí toda columna nueva, además de al SCHEMA de arriba.
MIGRATIONS = [
    # Log-loss de referencia de la evaluación walk-forward: el mercado y las
    # frecuencias de resultados (no saber nada). Ver core/evaluation.py.
    "ALTER TABLE model_runs ADD COLUMN logloss_bookmaker REAL",
    "ALTER TABLE model_runs ADD COLUMN logloss_frequencies REAL",
    "ALTER TABLE simulations ADD COLUMN label TEXT DEFAULT 'Temporada actual'",
    # Sin league_code, la consulta de la página de liga tenía que adivinar a qué
    # liga pertenecía cada simulación, y mezclaba unas con otras.
    "ALTER TABLE simulations ADD COLUMN league_code TEXT",
    "ALTER TABLE simulations ADD COLUMN seed INTEGER",
    "ALTER TABLE simulations ADD COLUMN created_at TEXT",
    # Métricas honestas del entrenamiento (ver app/services/ml_service.py)
    "ALTER TABLE model_runs ADD COLUMN baseline_home REAL",
    "ALTER TABLE model_runs ADD COLUMN n_train INTEGER",
    "ALTER TABLE model_runs ADD COLUMN n_test INTEGER",
    "ALTER TABLE model_runs ADD COLUMN test_period TEXT",
    # Multi-temporada: qué temporadas entrenaron el modelo, y dos baselines
    # más además de "siempre local" — clase mayoritaria y casas de apuestas —
    # más log-loss y Brier, que delatan un modelo que no predice empates
    # aunque su accuracy parezca decente.
    "ALTER TABLE model_runs ADD COLUMN seasons_used TEXT",
    "ALTER TABLE model_runs ADD COLUMN baseline_majority REAL",
    "ALTER TABLE model_runs ADD COLUMN baseline_bookmaker REAL",
    "ALTER TABLE model_runs ADD COLUMN logloss REAL",
    "ALTER TABLE model_runs ADD COLUMN brier REAL",
    # Mensaje de progreso legible, separado de `log` (que lleva errores)
    "ALTER TABLE job_status ADD COLUMN message TEXT",
    # El análisis de fichajes (Transfermarkt) se eliminó: fuera su tabla.
    # teams.tm_club_id se queda en las BD antiguas; SQLite no borra columnas
    # con facilidad y no molesta.
    "DROP TABLE IF EXISTS transfers",
    # Temporada de la que parte cada simulación: al cambiar de temporada, la
    # portada mostraba la proyección hecha desde la anterior como si fuera actual.
    "ALTER TABLE simulations ADD COLUMN season TEXT",
]

# Los índices UNIQUE son los que hacen idempotente la ingesta (ON CONFLICT).
INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_teams_unique ON teams(league_code, name)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_matches_unique "
    "ON matches(league_code, season, date, home_team_id, away_team_id)",
    "CREATE INDEX IF NOT EXISTS idx_matches_league_season ON matches(league_code, season, date)",
    "CREATE INDEX IF NOT EXISTS idx_simulations_league ON simulations(league_code, id)",
]


async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        # WAL permite leer mientras un job escribe; busy_timeout evita el
        # "database is locked" inmediato cuando coinciden dos escrituras.
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=10000")

        await db.executescript(SCHEMA)

        for statement in MIGRATIONS:
            try:
                await db.execute(statement)
            except Exception:
                pass  # la columna ya existe

        for statement in INDEXES:
            try:
                await db.execute(statement)
            except Exception:
                pass  # datos previos duplicados: se ignora, no es fatal

        # Backfill de league_code para simulaciones anteriores a la migración
        await db.execute(
            """
            UPDATE simulations
               SET league_code = (SELECT league_code FROM model_runs WHERE id = simulations.model_run_id)
             WHERE league_code IS NULL AND model_run_id IS NOT NULL
            """
        )
        await db.commit()


async def mark_interrupted_jobs() -> int:
    """Marca como fallidos los trabajos que quedaron 'queued' o 'running'.

    Se llama al arrancar el servidor: las BackgroundTasks viven en el proceso,
    así que tras un reinicio nadie va a terminar esos trabajos, y la tarjeta
    se quedaría «en marcha» para siempre.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            UPDATE job_status
               SET status = 'failed', progress = 0, updated_at = datetime('now'),
                   log = 'Interrumpido: el servidor se reinició mientras se ejecutaba.'
             WHERE status IN ('queued', 'running')
            """
        )
        await db.commit()
        return cursor.rowcount


async def get_db():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db
