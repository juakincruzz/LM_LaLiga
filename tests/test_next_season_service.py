import pytest
import aiosqlite
from app.database import init_db, DB_PATH
from app.services.next_season_service import project_next_season


@pytest.mark.asyncio
async def test_project_next_season_returns_valid_table():
    await init_db()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        result = await project_next_season(db, "SP1", sim_count=50, seed=42)

    assert len(result) == 20

    # Descendidos 25/26 no aparecen; entran los ascendidos configurados
    names = {t["name"] for t in result}
    assert "Oviedo" not in names
    assert "Girona" not in names
    assert "Levante" not in names
    assert "Ascendido 1" in names

    # Estructura de cada fila
    first = result[0]
    assert "points" in first and "gf" in first and "ga" in first
    assert "champion_prob" in first and "relegation_prob" in first

    # Tabla ordenada por puntos descendentes
    points = [t["points"] for t in result]
    assert points == sorted(points, reverse=True)

    # Las probabilidades de título suman ~1.0
    total_champ = sum(t["champion_prob"] or 0 for t in result)
    assert 0.95 <= total_champ <= 1.05


@pytest.mark.asyncio
async def test_project_next_season_persists_simulation():
    await init_db()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await project_next_season(db, "SP1", sim_count=20, seed=1)

        cursor = await db.execute(
            "SELECT label FROM simulations ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert "Proyección" in row["label"]
