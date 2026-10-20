"""Paridad del motor de features y simulación en core/.

Guarda que `core.features.build_training_matrix` (por lotes) y
`core.state.LeagueState` usado incrementalmente (simulación) generen siempre
el mismo vector, y que ninguna de las dos rutas se desvíe del snapshot
congelado en `tests/fixtures/SP1_2526_features.csv.gz` — capturado en la fase 3
al unificar lo que antes eran dos implementaciones independientes
(`FeatureEngineer` y el estado interno de `SeasonSimulator`, ya eliminadas).
"""
from pathlib import Path

import pandas as pd
import pytest

from core.dataset import load_season, played_only
from core.features import FEATURE_COLS, build_training_matrix
from core.state import LeagueState

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def partidos():
    return played_only(load_season("SP1", "2526", download=False))


def test_matriz_identica_al_snapshot_congelado(partidos):
    """Cualquier cambio no intencionado en las fórmulas (Elo, forma continua)
    o en el orden de los partidos rompe este test."""
    esperado = pd.read_csv(FIXTURES / "SP1_2526_features.csv.gz")
    X, y, _ = build_training_matrix(partidos, reset_per_season=False)

    obtenido = X.copy()
    obtenido["Target"] = y
    pd.testing.assert_frame_equal(esperado, obtenido, check_dtype=False, atol=1e-9)


def test_estado_incremental_coincide_con_el_de_lotes(partidos):
    """El vector generado partido a partido (camino del simulador) debe ser el
    mismo que el de la matriz por lotes (camino del entrenamiento).

    Es la desalineación que el diseño duplicado hacía posible: entrenar con un
    vector y predecir con otro.
    """
    X_lotes, _, _ = build_training_matrix(partidos, reset_per_season=False)

    state = LeagueState()
    # Mismo orden estable que usa build_training_matrix
    df = partidos.sort_values("Date", kind="mergesort").reset_index(drop=True)

    for i, row in enumerate(df.itertuples(index=False)):
        incremental = state.features(row.HomeTeam, row.AwayTeam, row.Date)
        esperado = X_lotes.iloc[i]
        for col in FEATURE_COLS:
            assert incremental[col] == pytest.approx(esperado[col]), f"fila {i}, columna {col}"
        state.apply_match(
            row.HomeTeam, row.AwayTeam,
            int(row.Home_Goals), int(row.Away_Goals), row.Result, row.Date,
            row.Home_ShotsOnTarget, row.Away_ShotsOnTarget,
        )


def test_clasificacion_coincide_con_la_del_simulador(partidos):
    """core.state.LeagueState.standings ≡ SeasonSimulator._get_standings"""
    from scripts.season_simulator import SeasonSimulator

    sim = SeasonSimulator()
    state = LeagueState()
    for row in partidos.itertuples(index=False):
        args = (row.HomeTeam, row.AwayTeam, int(row.Home_Goals), int(row.Away_Goals),
                row.Result, row.Date)
        sim.play_match(*args)
        state.apply_match(*args)

    esperada = sim._get_standings()
    obtenida = state.standings()

    pd.testing.assert_frame_equal(
        esperada.reset_index(drop=True), obtenida[esperada.columns], check_dtype=False
    )


def test_start_new_season_resetea_la_tabla_y_regresa_el_elo(partidos):
    state = LeagueState()
    for row in partidos.itertuples(index=False):
        state.apply_match(
            row.HomeTeam, row.AwayTeam,
            int(row.Home_Goals), int(row.Away_Goals), row.Result, row.Date,
            row.Home_ShotsOnTarget, row.Away_ShotsOnTarget,
        )

    h2h_antes = {k: list(v) for k, v in state.h2h_records.items()}
    elo_antes = {n: t.elo for n, t in state.teams.items()}
    ew_antes = {n: dict(t.ew) for n, t in state.teams.items()}
    media = sum(elo_antes.values()) / len(elo_antes)
    assert any(t.total.points > 0 for t in state.teams.values())

    state.start_new_season()

    assert all(t.total.matches == 0 and t.total.points == 0 for t in state.teams.values())
    assert state.h2h_records == h2h_antes
    # La forma continua sigue; el Elo se acerca un tercio a la media
    assert {n: t.ew for n, t in state.teams.items()} == ew_antes
    for nombre, antes in elo_antes.items():
        assert state.team(nombre).elo == pytest.approx(antes + 0.33 * (media - antes))


def test_ascendido_empieza_por_debajo(partidos):
    """Quien aparece tras el primer cambio de temporada viene de Segunda."""
    state = LeagueState()
    state.team("Barcelona")
    state.start_new_season()
    assert state.team("Recien Ascendido").elo < state.team("Barcelona").elo


def test_partido_simulado_sin_tiros_no_mueve_la_forma_de_tiros():
    state = LeagueState()
    antes = dict(state.team("Betis").ew)
    state.apply_match("Betis", "Sevilla", 2, 0, "H")
    despues = state.team("Betis").ew
    assert despues["tiros_puerta_favor"] == antes["tiros_puerta_favor"]
    assert despues["goles_favor"] > antes["goles_favor"]


def test_snapshot_es_independiente(partidos):
    state = LeagueState()
    fila = next(partidos.itertuples(index=False))
    state.apply_match(fila.HomeTeam, fila.AwayTeam, 1, 0, "H", fila.Date)

    copia = state.snapshot()
    copia.apply_match(fila.HomeTeam, fila.AwayTeam, 3, 0, "H", fila.Date)

    assert state.team(fila.HomeTeam).total.matches == 1
    assert copia.team(fila.HomeTeam).total.matches == 2
