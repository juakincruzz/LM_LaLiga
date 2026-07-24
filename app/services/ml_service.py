import aiosqlite
from datetime import datetime

from scripts.preprocess_generic import preprocess_league
from models.predictor import LeaguePredictor


async def _get_league_id(db: aiosqlite.Connection, league_code: str) -> int:
    cursor = await db.execute("SELECT id FROM leagues WHERE code=?", (league_code,))
    row = await cursor.fetchone()
    if not row:
        raise ValueError(f"League {league_code} not found in database")
    return row["id"]


async def train_league_model(db: aiosqlite.Connection, league_code: str, season: str = "2526"):
    """Train and save a model for a league."""
    league_id = await _get_league_id(db, league_code)

    # Preprocess
    X, y = preprocess_league(league_code, season)

    # Create model run record
    cursor = await db.execute(
        "INSERT INTO model_runs (league_id, created_at, status, model_type) VALUES (?, ?, ?, ?)",
        (league_id, datetime.now().isoformat(), "running", "xgboost")
    )
    await db.commit()
    run_id = cursor.lastrowid

    # Train
    predictor = LeaguePredictor(league_code)
    predictor.train_xgboost(X, y)

    # Quick accuracy on training set
    import pandas as pd
    from sklearn.metrics import accuracy_score
    y_pred = predictor.model.predict(X)
    if predictor.model_name in ("XGBoost", "LightGBM"):
        y_eval = pd.Series(y).map({-1: 0, 0: 1, 1: 2})
    else:
        y_eval = y
    accuracy = float(accuracy_score(y_eval, y_pred))

    # Save
    predictor.save_model()

    # Update record
    await db.execute(
        "UPDATE model_runs SET status=?, accuracy=? WHERE id=?",
        ("completed", accuracy, run_id)
    )
    await db.commit()
    return run_id, accuracy
