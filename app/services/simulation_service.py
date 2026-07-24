import aiosqlite
import json
from datetime import datetime

from scripts.season_simulator import simulate_season


async def run_simulation(db: aiosqlite.Connection, league_code: str, sim_count: int = 500):
    """Run Monte Carlo and save results."""
    # Resolve league_id from code
    cursor = await db.execute("SELECT id FROM leagues WHERE code=?", (league_code,))
    row = await cursor.fetchone()
    if not row:
        raise ValueError(f"League {league_code} not found in database")
    league_id = row["id"]

    # Find latest completed model run
    cursor = await db.execute(
        "SELECT id FROM model_runs WHERE league_id=? AND status='completed' ORDER BY created_at DESC LIMIT 1",
        (league_id,)
    )
    row = await cursor.fetchone()
    if not row:
        raise ValueError("No trained model found for this league")

    run_id = row["id"]
    standings = simulate_season(league_code, n_sims=sim_count)

    await db.execute(
        "INSERT INTO simulations (model_run_id, sim_count, standings_json) VALUES (?, ?, ?)",
        (run_id, sim_count, json.dumps(standings))
    )
    await db.commit()
    return standings
