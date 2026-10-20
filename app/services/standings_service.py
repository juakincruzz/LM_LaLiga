"""Clasificación real de la temporada en curso.

Hasta ahora la portada solo enseñaba simulaciones. La tabla sale de
`LeagueState.standings()` con los partidos jugados de la temporada, la misma
implementación que usan la simulación y la ficha de equipo.
"""
import asyncio

from app.config import get_current_season
from core.dataset import load_season, played_only
from core.features import build_state

LETRA = {3: "V", 1: "E", 0: "D"}


def clasificacion_actual(league_code: str, season: str | None = None) -> list[dict] | None:
    """Una fila por equipo, ordenada. `None` si la temporada no está
    descargada; lista vacía si aún no se ha jugado ningún partido."""
    season = season or get_current_season()
    try:
        df = load_season(league_code, season, download=False)
    except FileNotFoundError:
        return None

    jugados = played_only(df)
    if jugados.empty:
        return []

    state = build_state(jugados, reset_per_season=False)
    return [
        {
            "posicion": int(fila.Position),
            "equipo": fila.Team,
            "pj": int(fila.Games),
            "v": int(fila.Wins),
            "e": int(fila.Draws),
            "d": int(fila.Losses),
            "gf": int(fila.GF),
            "gc": int(fila.GA),
            "dg": int(fila.goal_diff),
            "puntos": int(fila.Points),
            "forma": [LETRA[r] for r in state.team(fila.Team).total.results[-5:]],
        }
        for fila in state.standings().itertuples(index=False)
    ]


async def clasificacion_actual_async(league_code: str, season: str | None = None):
    return await asyncio.to_thread(clasificacion_actual, league_code, season)
