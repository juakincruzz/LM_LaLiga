import aiosqlite
import pytest

from app.config import get_league_config
from app.services.next_season_service import project_next_season
from core.errors import PromotedTeamsNotConfiguredError, SeasonInProgressError

# Los que subieron de verdad tras la 25/26 de la fixture
ASCENDIDOS = ["La Coruna", "Malaga", "Santander"]


@pytest.fixture(autouse=True)
def ascendidos_reales(monkeypatch):
    """config/leagues.yaml se distribuye con «Ascendido N», que ahora la
    proyección rechaza a propósito."""
    monkeypatch.setitem(get_league_config("SP1")["next_season"], "promoted_teams", list(ASCENDIDOS))


async def test_project_next_season_returns_valid_table(temp_db):
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        result = await project_next_season(db, "SP1", sim_count=50, seed=42)

    config = get_league_config("SP1")
    promoted = config["next_season"]["promoted_teams"]

    assert len(result) == config["teams"]

    names = {t["name"] for t in result}
    # Los ascendidos configurados entran; los colistas de 25/26 salen.
    assert set(promoted).issubset(names)
    assert not {"Oviedo", "Girona", "Levante"} & names

    first = result[0]
    assert {"points", "gf", "ga", "champion_prob", "relegation_prob"} <= first.keys()

    points = [t["points"] for t in result]
    assert points == sorted(points, reverse=True)

    total_champ = sum(t["champion_prob"] or 0 for t in result)
    assert 0.95 <= total_champ <= 1.05


async def test_project_next_season_persists_simulation(temp_db):
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        await project_next_season(db, "SP1", sim_count=20, seed=1)

        cursor = await db.execute("SELECT label FROM simulations ORDER BY id DESC LIMIT 1")
        row = await cursor.fetchone()

    assert row is not None
    assert "Proyección" in row["label"]


async def test_project_next_season_is_reproducible(temp_db):
    """Con la misma semilla, la proyección debe ser idéntica."""
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        primera = await project_next_season(db, "SP1", sim_count=30, seed=7)
        segunda = await project_next_season(db, "SP1", sim_count=30, seed=7)

    assert primera == segunda


async def test_no_proyecta_con_la_temporada_a_medias(temp_db, media_temporada):
    """Con 190 partidos jugados, proyectar daba por descendidos a los colistas
    de ese momento: debe negarse y decir por qué."""
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        with pytest.raises(SeasonInProgressError, match="aún no ha terminado"):
            await project_next_season(db, "SP1", sim_count=5)


async def test_no_proyecta_con_ascendidos_provisionales(temp_db, monkeypatch):
    monkeypatch.setitem(
        get_league_config("SP1")["next_season"], "promoted_teams", ["Ascendido 1", "Ascendido 2", "Ascendido 3"]
    )
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        with pytest.raises(PromotedTeamsNotConfiguredError, match="Ascendido 1"):
            await project_next_season(db, "SP1", sim_count=5)
