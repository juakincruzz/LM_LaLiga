"""Ingesta de datos de una liga: descarga → normaliza → BD.

Punto único donde entran los partidos al sistema. Antes esta función solo
estampaba un timestamp: la tabla `matches` nunca se rellenaba, así que la
página de liga mostraba siempre 0 partidos y creía estar en off-season.
"""
import asyncio
from datetime import datetime

import aiosqlite
import pandas as pd

import requests

from app.config import get_current_season
from core.dataset import load_season
from core.equipos import EquipoDesconocidoError
from scripts.data_manager_generic import download_calendar, download_league_data
from scripts.understat import download_understat


async def _upsert_teams(db: aiosqlite.Connection, league_code: str, names: list[str]) -> dict[str, int]:
    """Inserta los equipos que falten y devuelve el mapa nombre → id."""
    for name in names:
        await db.execute(
            "INSERT INTO teams (league_code, name) VALUES (?, ?) "
            "ON CONFLICT(league_code, name) DO NOTHING",
            (league_code, name),
        )

    cursor = await db.execute("SELECT id, name FROM teams WHERE league_code=?", (league_code,))
    return {row["name"]: row["id"] for row in await cursor.fetchall()}


async def _upsert_matches(
    db: aiosqlite.Connection, league_code: str, season: str, df: pd.DataFrame, team_ids: dict[str, int]
) -> int:
    """Inserta o actualiza los partidos. Idempotente gracias a idx_matches_unique,
    así que pulsar 'Actualizar datos' dos veces no duplica nada."""
    # Los pendientes se regeneran en cada actualización: si un partido se
    # aplaza, su fecha cambia y la fila vieja se quedaría como un pendiente
    # fantasma (la clave única incluye la fecha).
    await db.execute(
        "DELETE FROM matches WHERE league_code=? AND season=? AND status='scheduled'",
        (league_code, season),
    )

    filas = []
    for row in df.itertuples(index=False):
        home_goals = getattr(row, "Home_Goals", None)
        away_goals = getattr(row, "Away_Goals", None)
        jugado = pd.notna(home_goals) and pd.notna(away_goals)
        jornada = getattr(row, "Matchday", None)
        filas.append(
            (
                league_code,
                row.Date.date().isoformat(),
                team_ids.get(row.HomeTeam),
                team_ids.get(row.AwayTeam),
                int(home_goals) if jugado else None,
                int(away_goals) if jugado else None,
                row.Result if jugado else None,
                season,
                int(jornada) if pd.notna(jornada) else None,
                "played" if jugado else "scheduled",
            )
        )

    await db.executemany(
        """
        INSERT INTO matches
            (league_code, date, home_team_id, away_team_id, home_goals, away_goals, result,
             season, matchday, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(league_code, season, date, home_team_id, away_team_id) DO UPDATE SET
            home_goals = excluded.home_goals,
            away_goals = excluded.away_goals,
            result     = excluded.result,
            matchday   = excluded.matchday,
            status     = excluded.status
        """,
        filas,
    )
    return len(filas)


async def update_league_data(
    db: aiosqlite.Connection, league_code: str, season: str | None = None, on_progress=None
) -> dict:
    """Descarga resultados y calendario y vuelca equipos y partidos en la BD.

    Descarga siempre (no usa la caché en disco): el sentido del botón
    'Actualizar datos' es traer los resultados de la última jornada.

    El calendario es lo que da los partidos pendientes (el CSV de resultados
    solo trae los jugados). Si no se puede descargar, se sigue solo con los
    resultados y se avisa: sin él la temporada parecería terminada.

    Lo mismo con Understat (alineaciones, para la ficha de equipo): si falla,
    la actualización sigue siendo un éxito y lo dice en `aviso_jugadores`.
    `on_progress(hechos, total)` se llama desde un hilo, partido a partido.
    """
    season = season or get_current_season()
    download_league_data(league_code, season)
    try:
        download_calendar(league_code, season)
        aviso_calendario = None
    except (requests.RequestException, EquipoDesconocidoError, KeyError, ValueError) as e:
        aviso_calendario = f"sin calendario ({type(e).__name__}: {e})"
    df = load_season(league_code, season, download=False)

    nombres = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    team_ids = await _upsert_teams(db, league_code, nombres)
    n_partidos = await _upsert_matches(db, league_code, season, df, team_ids)

    await db.execute(
        "UPDATE leagues SET last_updated=?, season=? WHERE code=?",
        (datetime.now().isoformat(), season, league_code),
    )
    await db.commit()

    jugados = int(df["Result"].notna().sum())
    resumen = {"equipos": len(nombres), "partidos": n_partidos, "jugados": jugados}
    if aviso_calendario:
        resumen["aviso"] = aviso_calendario

    # En un hilo: la primera vez son ~380 peticiones con pausa entre ellas, y
    # en el event loop congelarían el servidor (y el sondeo del progreso).
    try:
        understat = await asyncio.to_thread(download_understat, league_code, season, on_progress)
        resumen["jugadores"] = understat["partidos"] - understat["faltan"]
        if understat["faltan"]:
            resumen["aviso_jugadores"] = (
                f"faltan {understat['faltan']} alineaciones; se completarán en la próxima actualización"
            )
    except (requests.RequestException, EquipoDesconocidoError, KeyError, ValueError) as e:
        resumen["aviso_jugadores"] = f"sin datos de jugadores ({type(e).__name__}: {e})"
    return resumen
