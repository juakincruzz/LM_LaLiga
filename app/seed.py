import aiosqlite
from app.database import DB_PATH
from app.config import get_current_season, list_leagues


async def seed_leagues():
    async with aiosqlite.connect(DB_PATH) as db:
        for league in list_leagues():
            await db.execute(
                """
                INSERT OR IGNORE INTO leagues (code, name, country, season, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (league["code"], league["name"], league.get("country"), get_current_season(), 1)
            )
        await db.commit()


if __name__ == "__main__":
    import asyncio
    from app.database import init_db
    asyncio.run(init_db())
    asyncio.run(seed_leagues())
    print("Leagues seeded.")
