import pytest
from pathlib import Path
from app.database import init_db, DB_PATH


@pytest.mark.asyncio
async def test_init_db_creates_tables():
    if DB_PATH.exists():
        DB_PATH.unlink()
    await init_db()
    assert DB_PATH.exists()

    import aiosqlite
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in await cursor.fetchall()}
        expected = {
            "leagues", "teams", "matches", "model_runs",
            "predictions", "simulations", "transfers", "job_status"
        }
        assert expected.issubset(tables)
