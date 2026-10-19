"""Marcadores simulados por Poisson, condicionados al resultado ya decidido.

Antes `SeasonSimulator` asignaba tres marcadores fijos según quién ganaba
(H→2-1, A→1-2, D→1-1), así que la diferencia de goles de cualquier partido
simulado era siempre ±1 y el desempate por DG en la clasificación era un
artefacto sin información real.

El modelo de ML sigue siendo quien decide el resultado (H/D/A) — para eso está
entrenado y es la parte que este proyecto valida con accuracy/baselines. Los
goles, condicionados a ese resultado, se muestrean por rechazo de dos Poisson
independientes con las medias de ataque/defensa de cada equipo
(`LeagueState.expected_goals`).

No se comparte con `next_season_service._simulate_one_season`: ese simulador
no tiene un resultado que condicionar (no hay modelo de ML fuera de temporada,
solo ratings de ataque/defensa relativos a la liga), así que ahí los goles SON
la variable primaria y el resultado se deriva de ellos, no al revés. Son dos
problemas distintos que resulta que ambos usan Poisson, no una duplicación.
"""
import numpy as np

MAX_INTENTOS = 30


def sample_scoreline(
    rng: np.random.Generator, outcome: str, lambda_home: float, lambda_away: float
) -> tuple[int, int]:
    """Marcador Poisson(lambda_home, lambda_away) coherente con `outcome` ('H'/'A'/'D').

    Rechazo simple: se sortean goles hasta que el marcador cuadra con el
    resultado ya decidido. Con λ razonables (goles esperados de fútbol,
    típicamente 0.5–3) converge casi siempre en la primera o segunda
    iteración; el resultado favorece a quien tiene que ganar en el fallback.
    """
    home_goals = away_goals = 0
    for _ in range(MAX_INTENTOS):
        home_goals = int(rng.poisson(lambda_home))
        away_goals = int(rng.poisson(lambda_away))
        if outcome == "H" and home_goals > away_goals:
            return home_goals, away_goals
        if outcome == "A" and home_goals < away_goals:
            return home_goals, away_goals
        if outcome == "D" and home_goals == away_goals:
            return home_goals, away_goals

    if outcome == "H":
        return max(home_goals, away_goals + 1), away_goals
    if outcome == "A":
        return home_goals, max(away_goals, home_goals + 1)
    empate = round((home_goals + away_goals) / 2)
    return empate, empate


MAX_GOLES_TABLA = 10


def scoreline_distribution(
    p_home: float, p_draw: float, p_away: float, lambda_home: float, lambda_away: float
) -> np.ndarray:
    """Matriz P[goles_local, goles_visitante] coherente con las probabilidades 1X2.

    Es la versión analítica de `sample_scoreline`: el modelo decide cuánto pesa
    cada resultado y, dentro de cada uno, los marcadores se reparten según dos
    Poisson independientes. Así la suma de la zona "gana el local" es
    exactamente `p_home`, y el marcador más probable no contradice al modelo.
    """
    goles = np.arange(MAX_GOLES_TABLA + 1)
    fact = np.cumprod(np.r_[1, goles[1:]]).astype(float)
    poisson_h = np.exp(-lambda_home) * lambda_home**goles / fact
    poisson_a = np.exp(-lambda_away) * lambda_away**goles / fact
    conjunta = np.outer(poisson_h, poisson_a)

    i, j = np.indices(conjunta.shape)
    zonas = {"H": i > j, "D": i == j, "A": i < j}
    pesos = {"H": p_home, "D": p_draw, "A": p_away}

    tabla = np.zeros_like(conjunta)
    for clave, zona in zonas.items():
        masa = conjunta[zona].sum()
        if masa > 0:
            tabla[zona] = conjunta[zona] / masa * pesos[clave]
    return tabla / tabla.sum()


def top_scorelines(tabla: np.ndarray, n: int = 5) -> list[tuple[int, int, float]]:
    """Los n marcadores más probables como (goles_local, goles_visitante, prob)."""
    orden = np.argsort(tabla, axis=None)[::-1][:n]
    return [(int(k // tabla.shape[1]), int(k % tabla.shape[1]), float(tabla.flat[k])) for k in orden]
