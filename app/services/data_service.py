import aiosqlite
from datetime import datetime
from scripts.data_manager_generic import ensure_league_data


async def update_league_data(db: aiosqlite.Connection, league_code: str, season: str = "2526"):
    """Download CSV and update league last_updated timestamp."""
    ensure_league_data(league_code, season)
    await db.execute(
        "UPDATE leagues SET last_updated=?, season=? WHERE code=?",
        (datetime.now().isoformat(), season, league_code)
    )
    await db.commit()
