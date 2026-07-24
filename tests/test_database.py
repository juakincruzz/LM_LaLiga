import pytest
from pathlib import Path
import tempfile

from app.database import init_db, DB_PATH


@pytest.mark.asyncio
async def test_init_db_creates_tables():
    # Use a temporary DB so we don't wipe the production database
    temp_db = Path(tempfile.mktemp(suffix=".sqlite"))
    original_path = DB_PATH

    try:
        # Monkey-patch DB_PATH for this test
        import app.database as db_module
        db_module.DB_PATH = temp_db

        await init_db()
        assert temp_db.exists()

        import aiosqlite
        async with aiosqlite.connect(temp_db) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in await cursor.fetchall()}
            expected = {
                "leagues", "teams", "matches", "model_runs",
                "predictions", "simulations", "transfers", "job_status"
            }
            assert expected.issubset(tables)
    finally:
        # Restore original DB_PATH
        db_module.DB_PATH = original_path
        if temp_db.exists():
            temp_db.unlink()
