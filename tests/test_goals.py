"""core.goals: marcadores Poisson coherentes con el resultado ya decidido.

Antes los goles eran tres marcadores fijos (H->2-1, A->1-2, D->1-1), así que
la diferencia de goles simulada siempre valía ±1.
"""
import numpy as np
import pytest

from core.goals import sample_scoreline
from core.state import LeagueState


@pytest.mark.parametrize("outcome", ["H", "A", "D"])
def test_marcador_coherente_con_el_resultado(outcome):
    rng = np.random.default_rng(1)
    for _ in range(200):
        hg, ag = sample_scoreline(rng, outcome, lambda_home=1.4, lambda_away=1.1)
        if outcome == "H":
            assert hg > ag
        elif outcome == "A":
            assert hg < ag
        else:
            assert hg == ag


def test_reproducible_con_semilla():
    a = sample_scoreline(np.random.default_rng(7), "H", 1.5, 1.0)
    b = sample_scoreline(np.random.default_rng(7), "H", 1.5, 1.0)
    assert a == b


def test_la_diferencia_de_goles_ya_no_es_siempre_uno():
    """Antes cualquier victoria local simulada daba 2-1: DG=+1 siempre."""
    rng = np.random.default_rng(3)
    diferencias = {
        sample_scoreline(rng, "H", 1.6, 1.0)[0] - sample_scoreline(rng, "H", 1.6, 1.0)[1]
        for _ in range(100)
    }
    assert len(diferencias) > 1


def test_expected_goals_neutro_sin_historial():
    """Un equipo sin partidos aún (ascendido, inicio de temporada) usa el
    valor neutro, no 0 ni una media basada en un puñado de partidos ruidosos."""
    from core.state import GOLES_LIGA_NEUTRO

    state = LeagueState()
    lh, la = state.expected_goals("Nuevo A", "Nuevo B")
    assert lh == pytest.approx(GOLES_LIGA_NEUTRO)
    assert la == pytest.approx(GOLES_LIGA_NEUTRO)


def test_expected_goals_usa_el_historial_propio_tras_suficientes_partidos():
    state = LeagueState()
    fecha = np.datetime64("2025-08-01")
    # Un equipo goleador de local, tres partidos (el mínimo)
    for _ in range(3):
        state.apply_match("Goleador", "Rival", 4, 0, "H", fecha)

    lh, _ = state.expected_goals("Goleador", "Cualquiera")
    from core.state import GOLES_LIGA_NEUTRO

    assert lh > GOLES_LIGA_NEUTRO
