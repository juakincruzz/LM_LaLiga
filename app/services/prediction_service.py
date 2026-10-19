"""Predicción de partidos concretos: la próxima jornada y el «cara a cara».

Usa el mismo estado que la simulación (`build_current_state`: historial de las
temporadas de entrenamiento + jugados de la actual), así que las features de
cada partido son las mismas que el modelo vio al entrenar.

El modelo da las probabilidades de 1/X/2; el marcador sale de repartir cada
una entre los marcadores compatibles según dos Poisson con los goles esperados
de cada equipo (`core.goals.scoreline_distribution`).
"""
import asyncio
import threading
from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.config import get_current_season
from core.errors import LMLaLigaError
from core.features import feature_vector
from core.goals import scoreline_distribution, top_scorelines
from core.state import LeagueState
from models.predictor import LeaguePredictor
from scripts.data_manager_generic import calendar_csv_path, league_csv_path
from scripts.season_simulator import build_current_state

LETRA_RESULTADO = {3: "V", 1: "E", 0: "D"}


class EquipoNoValidoError(LMLaLigaError):
    """Equipo que no juega esta temporada, o el mismo equipo en los dos lados."""


@dataclass
class _Contexto:
    state: LeagueState
    predictor: LeaguePredictor
    temporada: pd.DataFrame
    posiciones: dict[str, int]
    ultimo_jugado: pd.Timestamp | None


_cache: dict[tuple, _Contexto] = {}
_cache_lock = threading.Lock()


def _clave(league_code: str, season: str) -> tuple:
    """El contexto vale mientras no cambien el modelo ni los datos de la temporada."""
    rutas = (
        LeaguePredictor(league_code).model_path,
        league_csv_path(league_code, season),
        calendar_csv_path(league_code, season),
    )
    return (league_code, season, *(r.stat().st_mtime_ns if r.exists() else None for r in rutas))


def _contexto(league_code: str) -> _Contexto:
    season = get_current_season()
    clave = _clave(league_code, season)
    with _cache_lock:
        if clave not in _cache:
            predictor = LeaguePredictor(league_code)
            predictor.load_model()  # ModelNotTrainedError si no hay .pkl
            state, temporada = build_current_state(league_code, season)
            tabla = state.standings()
            jugados = temporada.dropna(subset=["Result"])
            _cache.clear()
            _cache[clave] = _Contexto(
                state=state,
                predictor=predictor,
                temporada=temporada,
                posiciones=dict(zip(tabla["Team"], tabla["Position"])) if not tabla.empty else {},
                ultimo_jugado=jugados["Date"].max() if not jugados.empty else None,
            )
        return _cache[clave]


def _equipo(ctx: _Contexto, nombre: str) -> dict:
    t = ctx.state.team(nombre).total
    return {
        "nombre": nombre,
        "posicion": ctx.posiciones.get(nombre),
        "partidos": t.matches,
        "puntos": t.points,
        "ppp": t.points_per_game,
        "gf": t.goals_for_avg,
        "gc": t.goals_against_avg,
        "forma": [LETRA_RESULTADO[r] for r in t.results[-5:]],
    }


def _predecir(ctx: _Contexto, local: str, visitante: str, fecha: pd.Timestamp) -> dict:
    equipos = ctx.state.teams
    for nombre in (local, visitante):
        if nombre not in equipos:
            raise EquipoNoValidoError(f"«{nombre}» no juega esta temporada.")
    if local == visitante:
        raise EquipoNoValidoError("Elige dos equipos distintos.")

    pred = ctx.predictor.predict_match(feature_vector(ctx.state, local, visitante, fecha))
    p1, px, p2 = pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]
    total = p1 + px + p2
    p1, px, p2 = p1 / total, px / total, p2 / total

    lambda_local, lambda_visitante = ctx.state.expected_goals(local, visitante)
    marcadores = top_scorelines(scoreline_distribution(p1, px, p2, lambda_local, lambda_visitante), n=5)

    h2h = ctx.state.h2h_records.get((local, visitante), [])
    favorito = max((p1, "1"), (px, "X"), (p2, "2"))[1]

    return {
        "local": _equipo(ctx, local),
        "visitante": _equipo(ctx, visitante),
        "fecha": fecha.date().isoformat(),
        "p1": p1,
        "px": px,
        "p2": p2,
        "favorito": favorito,
        "xg_local": lambda_local,
        "xg_visitante": lambda_visitante,
        "marcadores": [{"local": a, "visitante": b, "prob": p} for a, b, p in marcadores],
        "h2h": {
            "partidos": len(h2h),
            "victorias": sum(1 for r in h2h if r == 3),
            "empates": sum(1 for r in h2h if r == 1),
            "derrotas": sum(1 for r in h2h if r == 0),
        },
    }


def predecir_partido(league_code: str, local: str, visitante: str, fecha: date | None = None) -> dict:
    ctx = _contexto(league_code)
    return _predecir(ctx, local, visitante, pd.Timestamp(fecha or date.today()))


def jornada(league_code: str, numero: int | None = None) -> dict:
    """Partidos de una jornada con su predicción (o su resultado, si ya se jugó).

    Sin `numero`, la «próxima jornada» es la primera con partidos pendientes
    cuya fecha MEDIANA (la de toda la jornada) aún no ha pasado. Con la fecha
    máxima, un partido aplazado de la jornada 6 al 21/10 convertía la jornada 6
    en la «próxima» en plena jornada 10. Si todas quedan en el pasado
    (football-data publica con días de retraso), la primera pendiente.
    """
    ctx = _contexto(league_code)
    df = ctx.temporada
    if "Matchday" not in df.columns:
        return {"jornadas": [], "numero": None, "partidos": [], "datos_hasta": ctx.ultimo_jugado}

    pendientes = df[df["Result"].isna() & df["Matchday"].notna()]
    jornadas = sorted(int(n) for n in pendientes["Matchday"].unique())
    if not jornadas:
        return {"jornadas": [], "numero": None, "partidos": [], "datos_hasta": ctx.ultimo_jugado}

    if numero not in jornadas:
        hoy = pd.Timestamp(date.today())
        mediana = df.groupby("Matchday")["Date"].median()
        futuras = [n for n in jornadas if mediana[n] >= hoy]
        numero = futuras[0] if futuras else jornadas[0]

    partidos = []
    for row in df[df["Matchday"] == numero].itertuples(index=False):
        partido = {"fecha": row.Date.date().isoformat(), "local": row.HomeTeam, "visitante": row.AwayTeam}
        if pd.notna(row.Result):
            partido["resultado"] = f"{int(row.Home_Goals)}-{int(row.Away_Goals)}"
        else:
            partido["prediccion"] = _predecir(ctx, row.HomeTeam, row.AwayTeam, row.Date)
        partidos.append(partido)

    return {
        "jornadas": jornadas,
        "numero": numero,
        "partidos": partidos,
        "datos_hasta": ctx.ultimo_jugado.date().isoformat() if ctx.ultimo_jugado is not None else None,
    }


# --- Envoltorios asíncronos: el trabajo de CPU fuera del event loop ---------


async def predecir_partido_async(league_code: str, local: str, visitante: str, fecha: date | None = None):
    return await asyncio.to_thread(predecir_partido, league_code, local, visitante, fecha)


async def jornada_async(league_code: str, numero: int | None = None):
    return await asyncio.to_thread(jornada, league_code, numero)
