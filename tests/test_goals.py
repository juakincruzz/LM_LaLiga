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


def test_expected_goals_sin_historial_parte_de_la_forma_inicial():
    """Un equipo sin partidos (ascendido) arranca algo peor que la media de la
    liga, y el local sigue teniendo su ventaja de campo."""
    from core.state import EW_INICIAL, FACTOR_GOLES_LOCAL

    state = LeagueState()
    lh, la = state.expected_goals("Nuevo A", "Nuevo B")
    media = (EW_INICIAL["goles_favor"] + EW_INICIAL["goles_contra"]) / 2
    assert lh == pytest.approx(media * FACTOR_GOLES_LOCAL)
    assert lh > la


def test_expected_goals_sigue_la_forma_y_no_se_reinicia_en_verano():
    state = LeagueState()
    sin_historial, _ = state.expected_goals("Goleador", "Cualquiera")
    for _ in range(3):
        state.apply_match("Goleador", "Rival", 4, 0, "H")

    goleador, _ = state.expected_goals("Goleador", "Cualquiera")
    assert goleador > sin_historial

    state.start_new_season()
    assert state.expected_goals("Goleador", "Cualquiera")[0] == pytest.approx(goleador)
