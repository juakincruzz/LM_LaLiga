import aiosqlite

from app.database import init_db


async def test_init_db_creates_tables(temp_db):
    # `temp_db` (autouse) ya redirigió DB_PATH y ejecutó init_db; volver a
    # llamarlo comprueba además que el esquema es idempotente.
    await init_db()
    assert temp_db.exists()

    async with aiosqlite.connect(temp_db) as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}

    expected = {
        "leagues", "teams", "matches", "model_runs",
        "predictions", "simulations", "transfers", "job_status",
    }
    assert expected.issubset(tables)
