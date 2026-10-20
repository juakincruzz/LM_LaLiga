"""Fase B: clasificación real e historial de aciertos frente al mercado."""
import pytest

from tests.conftest import JUGADOS


# --- Clasificación real ------------------------------------------------------


def test_clasificacion_real_coincide_con_league_state():
    from app.services.standings_service import clasificacion_actual
    from core.dataset import load_season, played_only
    from core.features import build_state

    tabla = clasificacion_actual("SP1")
    esperada = build_state(played_only(load_season("SP1", "2526", download=False))).standings()

    assert len(tabla) == 20
    assert [t["equipo"] for t in tabla] == list(esperada["Team"])
    assert [t["puntos"] for t in tabla] == list(esperada["Points"])
    assert all(t["pj"] == 38 and len(t["forma"]) == 5 for t in tabla)
    assert all(t["v"] + t["e"] + t["d"] == t["pj"] for t in tabla)


def test_clasificacion_a_mitad_de_temporada(media_temporada):
    from app.services.standings_service import clasificacion_actual

    tabla = clasificacion_actual("SP1")
    assert sum(t["pj"] for t in tabla) == 2 * JUGADOS


def test_clasificacion_sin_datos_es_none(data_root):
    from app.services.standings_service import clasificacion_actual

    (data_root / "SP1" / "2526.csv").unlink()
    assert clasificacion_actual("SP1") is None


async def test_portada_muestra_la_clasificacion_real(client):
    resp = await client.get("/")
    assert "Clasificación real de la temporada" in resp.text
    assert "Proyección final" in resp.text


# --- Historial de aciertos ----------------------------------------------------


@pytest.fixture
def entrenado(media_temporada, mock_download, temp_db):
    """Entrena de verdad (guarda también el modelo del historial)."""
    import asyncio

    import aiosqlite

    import app.services.history_service as hs
    from app.services.ml_service import train_league_model

    async def entrenar():
        async with aiosqlite.connect(temp_db) as db:
            db.row_factory = aiosqlite.Row
            await train_league_model(db, "SP1")

    asyncio.run(entrenar())
    hs._cache.clear()


def test_entrenar_guarda_el_modelo_del_historial(entrenado, models_dir):
    assert (models_dir / "SP1_predictor_historial.pkl").exists()


def test_historial_por_jornada(entrenado):
    from app.services.history_service import historial

    h = historial("SP1")
    total = h["total"]

    assert total["partidos"] == JUGADOS
    assert sum(j["partidos"] for j in h["jornadas"]) == JUGADOS
    assert [j["numero"] for j in h["jornadas"]] == list(range(1, JUGADOS // 10 + 1))
    assert 0 <= total["aciertos_modelo"] <= JUGADOS
    assert 0.8 < total["logloss_modelo"] < 1.3
    # La fixture trae cuotas: el mercado se compara en todos los partidos
    assert total["con_cuotas"] == JUGADOS and total["logloss_mercado"] is not None
    assert sum(j["aciertos_modelo"] for j in h["jornadas"]) == total["aciertos_modelo"]


def test_el_historial_no_conoce_los_resultados(entrenado, models_dir):
    """El modelo del historial no puede haber visto la temporada actual: su
    log-loss sobre ella debe ser la de una predicción, no la de un ajuste."""
    from app.services.history_service import historial

    h = historial("SP1")
    assert h["total"]["acierto_modelo"] < 0.75


async def test_endpoint_historial(client, entrenado):
    resp = await client.get("/prediccion/historial")
    assert resp.status_code == 200
    assert "Favoritos acertados por el modelo" in resp.text
    assert "Partidos de la jornada 19" in resp.text


async def test_historial_sin_modelo_lo_dice(client, media_temporada):
    import app.services.history_service as hs

    hs._cache.clear()
    resp = await client.get("/prediccion/historial")
    assert resp.status_code == 200
    assert "Entrenar modelo" in resp.text
