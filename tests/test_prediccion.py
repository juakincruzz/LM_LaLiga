"""Calendario, próxima jornada y predicción de partidos concretos.

Escenario: la 25/26 de la fixture cortada a mitad (190 jugados) más un
calendario con los 380 partidos, que es la situación real a mitad de
temporada: football-data solo trae los jugados, el calendario los pendientes.
"""
from pathlib import Path

import aiosqlite
import numpy as np
import pandas as pd
import pytest

from core.dataset import load_season, load_seasons, merge_calendar, played_only
from core.features import build_training_matrix
from core.goals import scoreline_distribution, top_scorelines
from models.predictor import LeaguePredictor

from tests.conftest import JUGADOS

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def modelo(media_temporada, mock_download):
    """Un modelo entrenado de verdad (pequeño) en el MODELS_DIR temporal."""
    import app.services.prediction_service as ps

    df = played_only(load_seasons("SP1", ["2324", "2425", "2526"]))
    X, y, _ = build_training_matrix(df)
    predictor = LeaguePredictor("SP1")
    predictor.train_xgboost(X, y)
    predictor.save_model()
    ps._cache.clear()
    return predictor


# --- Marcadores ------------------------------------------------------------


def test_distribucion_de_marcadores_respeta_el_1x2():
    tabla = scoreline_distribution(0.5, 0.3, 0.2, 1.6, 1.0)
    i, j = np.indices(tabla.shape)

    assert tabla.sum() == pytest.approx(1.0)
    assert tabla[i > j].sum() == pytest.approx(0.5)
    assert tabla[i == j].sum() == pytest.approx(0.3)
    assert tabla[i < j].sum() == pytest.approx(0.2)

    top = top_scorelines(tabla, n=5)
    probs = [p for _, _, p in top]
    assert probs == sorted(probs, reverse=True)


# --- Calendario --------------------------------------------------------------


def test_merge_calendar_empareja_por_par_y_no_por_fecha():
    resultados = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-15", "2026-08-16"]),
        "HomeTeam": ["Barcelona", "Sevilla"],
        "AwayTeam": ["Getafe", "Betis"],
        "Home_Goals": [2, 1], "Away_Goals": [0, 1], "Result": ["H", "D"],
    })
    calendario = pd.DataFrame({
        # Barcelona-Getafe aparece con otra fecha (aplazado): no debe duplicarse
        "Date": ["20/08/2026", "16/08/2026", "23/08/2026", "23/08/2026"],
        "HomeTeam": ["Barcelona", "Sevilla", "Getafe", "Betis"],
        "AwayTeam": ["Getafe", "Betis", "Barcelona", "Sevilla"],
        "Matchday": [1, 1, 2, 2],
    })

    df = merge_calendar(resultados, calendario)

    assert len(df) == 4
    assert df["Result"].notna().sum() == 2
    assert list(df["Matchday"]) == [1, 1, 2, 2]


def test_load_season_incluye_los_pendientes_del_calendario(media_temporada):
    df = load_season("SP1", "2526", download=False)
    assert len(df) == 380
    assert df["Result"].notna().sum() == JUGADOS
    assert df["Matchday"].notna().all()


async def test_actualizar_con_calendario_deja_la_temporada_en_curso(temp_db, media_temporada):
    from app.services.data_service import update_league_data
    from app.services.season_state import EstadoTemporada, get_season_state

    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        resumen = await update_league_data(db, "SP1")
        await update_league_data(db, "SP1")  # idempotente también con pendientes
        estado = await get_season_state(db, "SP1")
        cursor = await db.execute("SELECT COUNT(*) AS n FROM matches WHERE matchday IS NULL")
        sin_jornada = (await cursor.fetchone())["n"]

    assert "aviso" not in resumen
    assert estado.estado is EstadoTemporada.EN_CURSO
    assert (estado.total, estado.jugados, estado.pendientes) == (380, JUGADOS, 380 - JUGADOS)
    assert sin_jornada == 0


# --- Estado y predicción -----------------------------------------------------


def test_el_estado_actual_conserva_el_h2h_de_temporadas_anteriores(media_temporada, mock_download):
    """El simulador llegó a ver 0 enfrentamientos previos donde el entrenamiento
    veía ~2,9: el estado actual debe llevar el historial completo."""
    from scripts.season_simulator import build_current_state

    state, actual = build_current_state("SP1", "2526")

    assert max(len(v) for v in state.h2h_records.values()) >= 2
    # Solo los equipos de esta temporada; los descendidos no aparecen en la tabla
    assert set(state.teams) == set(actual["HomeTeam"]) | set(actual["AwayTeam"])
    assert len(state.standings()) == 20


def test_jornada_predice_los_pendientes(modelo):
    from app.services.prediction_service import jornada

    j = jornada("SP1")

    # Todas las fechas de la fixture son pasadas: la próxima es la primera pendiente
    assert j["numero"] == JUGADOS // 10 + 1
    assert len(j["partidos"]) == 10
    for partido in j["partidos"]:
        p = partido["prediccion"]
        assert p["p1"] + p["px"] + p["p2"] == pytest.approx(1.0)
        assert len(p["marcadores"]) == 5


def test_predecir_un_partido_cualquiera(modelo):
    from app.services.prediction_service import predecir_partido

    p = predecir_partido("SP1", "Barcelona", "Real Madrid")

    assert p["local"]["nombre"] == "Barcelona"
    assert p["local"]["posicion"] is not None
    assert p["favorito"] in {"1", "X", "2"}
    assert p["xg_local"] > 0 and p["xg_visitante"] > 0


async def test_endpoints_de_prediccion(client, modelo):
    resp = await client.get("/prediccion/jornada")
    assert resp.status_code == 200
    assert "Jornada 20" in resp.text

    resp = await client.get("/prediccion/partido", params={"local": "Barcelona", "visitante": "Real Madrid"})
    assert resp.status_code == 200
    assert "Marcador más probable" in resp.text

    resp = await client.get("/prediccion/partido", params={"local": "Barcelona", "visitante": "Barcelona"})
    assert "Elige dos equipos distintos" in resp.text

    resp = await client.get("/prediccion/partido", params={"local": "Barcelona", "visitante": "Bayern"})
    assert "no juega esta temporada" in resp.text


async def test_sin_modelo_lo_dice(client, media_temporada, mock_download):
    import app.services.prediction_service as ps

    ps._cache.clear()
    resp = await client.get("/prediccion/jornada")
    assert resp.status_code == 200
    assert "No hay modelo entrenado" in resp.text


def test_simular_a_mitad_de_temporada_con_historial(modelo):
    """El camino real en plena temporada: pendientes del calendario + estado
    con historial. Los descendidos de temporadas anteriores no deben colarse."""
    from scripts.season_simulator import simulate_season

    tabla = simulate_season("SP1", n_sims=3, season="2526", seed=1)

    assert len(tabla) == 20
    assert sum(t["champion_prob"] or 0 for t in tabla) == pytest.approx(1.0)


def test_un_aplazado_no_se_convierte_en_la_proxima_jornada(modelo, monkeypatch):
    """Caso real de la 26/27: Levante-Athletic (jornada 6) aplazado al 21/10,
    en plena jornada 10. La próxima jornada sigue siendo la de las fechas."""
    import app.services.prediction_service as ps

    df = ps._contexto("SP1").temporada
    hoy = df.loc[df["Matchday"] == 25, "Date"].min()  # "hoy" = inicio de la jornada 25
    aplazado = df.index[(df["Matchday"] == 20)][0]
    df.loc[aplazado, "Date"] = hoy + pd.Timedelta(days=3)  # un partido de la 20, movido

    class _Fecha(ps.date):
        @classmethod
        def today(cls):
            return hoy.date()

    monkeypatch.setattr(ps, "date", _Fecha)
    j = ps.jornada("SP1")

    assert 20 in j["jornadas"]
    assert j["numero"] == 25


def test_la_tabla_simulada_es_coherente_con_sus_probabilidades(modelo):
    """Puntos y probabilidades salen del mismo Monte Carlo: la tabla está
    ordenada por puntos esperados y nadie tiene «—» donde hay un 0 %."""
    from scripts.season_simulator import simulate_season

    tabla = simulate_season("SP1", n_sims=20, season="2526", seed=3)

    puntos = [t["points"] for t in tabla]
    assert puntos == sorted(puntos, reverse=True)
    assert all(t["champion_prob"] is not None and t["relegation_prob"] is not None for t in tabla)
    assert sum(t["relegation_prob"] for t in tabla) == pytest.approx(3.0)


async def test_predicciones_llevan_escudos(client, modelo):
    jornada = (await client.get("/prediccion/jornada")).text
    assert jornada.count('class="escudo"') == 20  # 10 partidos × 2 equipos

    duelo = (await client.get("/prediccion/partido", params={"local": "Barcelona", "visitante": "Betis"})).text
    assert duelo.count('width="32"') == 2
