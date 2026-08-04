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
