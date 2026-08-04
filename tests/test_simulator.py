"""El simulador sobre una temporada a medias: el caso que nunca se ejercitó.

Con la temporada 25/26 completa no quedaba ni un partido pendiente, así que este
camino no se ejecutaba nunca y escondía un crash.
"""
import numpy as np
import pytest

from core.dataset import load_season, played_only
from core.errors import SeasonFinishedError
from core.features import build_training_matrix
from models.predictor import LeaguePredictor
from scripts.season_simulator import SeasonSimulator, simulate_season


@pytest.fixture
def media_temporada():
    """Temporada con la segunda mitad sin jugar."""
    df = load_season("SP1", "2526", download=False)
    corte = df["Date"].iloc[190]
    df = df.copy()
    df.loc[df["Date"] >= corte, ["Home_Goals", "Away_Goals", "Result"]] = None
    return df


@pytest.fixture
def predictor(media_temporada):
    X, y, _ = build_training_matrix(played_only(media_temporada), reset_per_season=False)
    p = LeaguePredictor("SP1")
    p.train_xgboost(X, y)
    return p


def test_simula_los_partidos_pendientes(media_temporada, predictor):
    sim = SeasonSimulator()
    standings, simulados = sim.simulate(
        media_temporada, predictor, stochastic=True, verbose=False, rng=np.random.default_rng(7)
    )

    assert len(simulados) == 190
    assert len(standings) == 20
    # Cada equipo termina con los 38 partidos de la temporada
    assert set(standings["Games"]) == {38}
    assert (standings["Points"].diff().dropna() <= 0).all()  # ordenada


def test_monte_carlo_es_reproducible(media_temporada, predictor):
    sim = SeasonSimulator()
    kwargs = dict(n_sims=5, verbose=False)
    a = sim.monte_carlo_simulation(media_temporada, predictor, rng=np.random.default_rng(3), **kwargs)
    b = sim.monte_carlo_simulation(media_temporada, predictor, rng=np.random.default_rng(3), **kwargs)

    assert a["champion_probabilities"] == b["champion_probabilities"]


def test_probabilidades_de_campeon_suman_uno(media_temporada, predictor):
    sim = SeasonSimulator()
    res = sim.monte_carlo_simulation(
        media_temporada, predictor, n_sims=10, verbose=False, rng=np.random.default_rng(1)
    )
    assert sum(res["champion_probabilities"].values()) == pytest.approx(1.0)


def test_temporada_completa_lanza_error():
    """Sin partidos pendientes no hay nada que simular; repetir la temporada
    real devolvería un campeón al 100 % que parece un resultado del modelo."""
    with pytest.raises(SeasonFinishedError, match="ya ha terminado"):
        simulate_season("SP1", n_sims=5)


def test_monte_carlo_no_reproduce_los_partidos_jugados_en_cada_simulacion(media_temporada, predictor):
    """El estado base se construye una vez y se clona con snapshot()."""
    sim = SeasonSimulator()
    played, _ = sim.split_played(media_temporada)
    base = sim.state_from_played(played)

    copia = base.snapshot()
    copia.apply_match("Barcelona", "Girona", 5, 0, "H", media_temporada["Date"].iloc[0])

    # Modificar la copia no toca el estado base reutilizable
    assert base.team("Barcelona").total.matches != copia.team("Barcelona").total.matches
