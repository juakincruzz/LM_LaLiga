"""Servicio de fichajes vía Transfermarkt API (self-hosted en localhost:8001).

Estrategia: comparar la plantilla de cada club entre la temporada actual y la
siguiente. Los jugadores nuevos son altas ('in') y los que desaparecen son
bajas ('out'). Cada jugador trae `market_value` (int EUR) y `position`.
"""
import asyncio
import unicodedata
from collections import defaultdict
from datetime import datetime

import aiosqlite
import httpx
import pandas as pd

from app.config import get_current_season, get_league_config
from scripts.data_manager_generic import league_csv_path

TRANSFERMARKT_BASE = "http://localhost:8001"
REQUEST_DELAY_S = 1.0  # cortesía con Transfermarkt

# Mapeo de posiciones de Transfermarkt → grupos usados por el ajuste
POSITION_MAP = {
    "Goalkeeper": "GK",
    "Centre-Back": "DEF", "Left-Back": "DEF", "Right-Back": "DEF",
    "Defender": "DEF", "Defence": "DEF",
    "Defensive Midfield": "MID", "Central Midfield": "MID",
    "Attacking Midfield": "MID", "Midfielder": "MID",
    "Left Midfield": "MID", "Right Midfield": "MID",
    "Centre-Forward": "FWD", "Left Winger": "FWD", "Right Winger": "FWD",
    "Second Striker": "FWD", "Attack": "FWD", "Striker": "FWD",
}

# Aliases: nombre en football-data.co.uk → subcadena esperada en el nombre TM.
# Solo hace falta para los que no casan por normalización simple.
TEAM_ALIASES = {
    "SP1": {
        "Ath Madrid": "atletico",
        "Ath Bilbao": "athletic",
        "Sociedad": "sociedad",
        "Vallecano": "rayo",
        "Betis": "betis",
        "Celta": "celta",
        "Espanol": "espanyol",
        "Alaves": "alaves",
        "Barcelona": "barcelona",
        "Sevilla": "sevilla",
        "Valencia": "valencia",
        "Villarreal": "villarreal",
        "Getafe": "getafe",
        "Girona": "girona",
        "Elche": "elche",
        "Levante": "levante",
        "Oviedo": "oviedo",
        "Osasuna": "osasuna",
        "Mallorca": "mallorca",
    },
}


def _normalize(name: str) -> str:
    """Minúsculas, sin acentos, sin tokens genéricos de club."""
    nfkd = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    tokens = no_accents.lower().split()
    drop = {"fc", "cf", "ud", "cd", "de", "ca", "rc", "rcd", "sc", "ac", "real"}
    return " ".join(t for t in tokens if t not in drop)


def map_position(tm_position: str) -> str:
    return POSITION_MAP.get(tm_position, "MID")


async def _tm_get(client: httpx.AsyncClient, path: str) -> dict:
    resp = await client.get(f"{TRANSFERMARKT_BASE}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


async def check_transfermarkt_available(client: httpx.AsyncClient) -> None:
    """Falla rápido y con un mensaje claro si el servicio no está arrancado.

    Sin esto, el primer fallo de red llegaba 41 peticiones más tarde con
    "All connection attempts failed", sin ninguna pista de que hay que
    arrancar `transfermarkt-api` en el puerto 8001.
    """
    try:
        await client.get(f"{TRANSFERMARKT_BASE}/", timeout=3)
    except httpx.HTTPError as e:
        raise ConnectionError(
            f"El servicio Transfermarkt no responde en {TRANSFERMARKT_BASE}. "
            f"Arráncalo con: cd transfermarkt-api && PYTHONPATH=$PWD uvicorn app.main:app --port 8001"
        ) from e


async def fetch_competition_clubs(client, tm_league_id: str, season_id: str) -> list[dict]:
    data = await _tm_get(client, f"/competitions/{tm_league_id}/clubs?season_id={season_id}")
    return data.get("clubs", [])


async def fetch_club_players(client, club_id: str, season_id: str) -> list[dict]:
    data = await _tm_get(client, f"/clubs/{club_id}/players?season_id={season_id}")
    return data.get("players", [])


def match_teams(fd_names: list[str], tm_clubs: list[dict], league_code: str) -> dict:
    """Empareja nombres football-data con clubs TM. Devuelve {fd_name: tm_club_id}."""
    aliases = TEAM_ALIASES.get(league_code, {})
    matches = {}
    tm_index = [(_normalize(c["name"]), c["id"]) for c in tm_clubs]

    for fd_name in fd_names:
        fd_norm = _normalize(fd_name)
        alias_hint = aliases.get(fd_name)

        for tm_norm, tm_id in tm_index:
            if alias_hint and alias_hint in tm_norm:
                matches[fd_name] = tm_id
                break
            if fd_norm and fd_norm in tm_norm:
                matches[fd_name] = tm_id
                break
            if tm_norm and tm_norm in fd_norm:
                matches[fd_name] = tm_id
                break

    return matches


async def ensure_teams_populated(
    db: aiosqlite.Connection, league_code: str, season: str | None = None
) -> list[str]:
    """Inserta en `teams` los equipos encontrados en el CSV si la tabla está vacía."""
    cursor = await db.execute("SELECT COUNT(*) AS n FROM teams WHERE league_code=?", (league_code,))
    if (await cursor.fetchone())["n"] > 0:
        cursor = await db.execute("SELECT name FROM teams WHERE league_code=?", (league_code,))
        return [r["name"] for r in await cursor.fetchall()]

    csv_path = league_csv_path(league_code, season or get_current_season())
    if not csv_path.exists():
        raise FileNotFoundError(f"No hay CSV en {csv_path}. Pulsa 'Actualizar datos' primero.")

    df = pd.read_csv(csv_path)
    names = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    for name in names:
        await db.execute(
            "INSERT INTO teams (league_code, name) VALUES (?, ?)", (league_code, name)
        )
    await db.commit()
    return names


async def analyze_league_transfers(db: aiosqlite.Connection, league_code: str) -> dict:
    """Descarga plantillas TM (temporada actual y siguiente), calcula altas/bajas
    y las persiste en la tabla `transfers`. Devuelve un resumen del análisis."""
    config = get_league_config(league_code)
    tm_league_id = config["transfermarkt_league_id"]
    next_cfg = config.get("next_season", {})
    next_tm_season = next_cfg.get("tm_season_id", "2026")
    current_tm_season = str(int(next_tm_season) - 1)

    fd_names = await ensure_teams_populated(db, league_code)

    async with httpx.AsyncClient() as client:
        await check_transfermarkt_available(client)

        # Clubs de la competición en la temporada siguiente
        tm_clubs = await fetch_competition_clubs(client, tm_league_id, next_tm_season)
        await asyncio.sleep(REQUEST_DELAY_S)

        matches = match_teams(fd_names, tm_clubs, league_code)
        unmatched = [n for n in fd_names if n not in matches]
        # Nota: los clubs TM sin emparejar NO son necesariamente ascendidos —
        # en las ligas sin alias configurados (todas menos SP1) el emparejamiento
        # por subcadena falla para nombres como "Man United"/"Manchester United",
        # así que reportarlos como "ascendido detectado" era simplemente falso.

        # Persistir tm_club_id
        for fd_name, tm_id in matches.items():
            await db.execute(
                "UPDATE teams SET tm_club_id=? WHERE league_code=? AND name=?",
                (tm_id, league_code, fd_name),
            )
        await db.commit()

        # Limpiar fichajes previos de la liga
        await db.execute("DELETE FROM transfers WHERE league_code=?", (league_code,))
        await db.commit()

        # Diff de plantillas por club
        summary = {"arrivals": 0, "departures": 0, "net_value_eur": 0}
        cursor = await db.execute(
            "SELECT id, name, tm_club_id FROM teams WHERE league_code=?", (league_code,)
        )
        teams = await cursor.fetchall()

        today = datetime.now().date().isoformat()
        for team in teams:
            tm_id = team["tm_club_id"]
            if not tm_id:
                continue  # descendido o no emparejado

            try:
                current_players = await fetch_club_players(client, tm_id, current_tm_season)
                await asyncio.sleep(REQUEST_DELAY_S)
                next_players = await fetch_club_players(client, tm_id, next_tm_season)
                await asyncio.sleep(REQUEST_DELAY_S)
            except httpx.HTTPError as e:
                # Club sin página de la próxima temporada todavía: saltar
                print(f"⚠️  {team['name']}: {e}")
                continue

            current_by_id = {p["id"]: p for p in current_players}
            next_by_id = {p["id"]: p for p in next_players}

            arrivals = [p for pid, p in next_by_id.items() if pid not in current_by_id]
            departures = [p for pid, p in current_by_id.items() if pid not in next_by_id]

            for p in arrivals:
                mv = p.get("market_value") or 0
                summary["arrivals"] += 1
                summary["net_value_eur"] += mv
                await db.execute(
                    """INSERT INTO transfers
                       (league_code, team_id, player_name, position, transfer_type,
                        market_value_eur, from_club, to_club, date)
                       VALUES (?, ?, ?, ?, 'in', ?, ?, ?, ?)""",
                    (league_code, team["id"], p["name"], map_position(p.get("position", "")),
                     mv, p.get("signed_from"), team["name"], today),
                )

            for p in departures:
                mv = p.get("market_value") or 0
                summary["departures"] += 1
                summary["net_value_eur"] -= mv
                await db.execute(
                    """INSERT INTO transfers
                       (league_code, team_id, player_name, position, transfer_type,
                        market_value_eur, from_club, to_club, date)
                       VALUES (?, ?, ?, ?, 'out', ?, ?, ?, ?)""",
                    (league_code, team["id"], p["name"], map_position(p.get("position", "")),
                     mv, team["name"], None, today),
                )

            # Confirmar por equipo: 20 clubes * 2 peticiones * 1s de cortesía son
            # varios minutos con la transacción abierta si se deja todo para el
            # final, y eso es lo que producía "database is locked" en cualquier
            # otro job que intentara escribir mientras tanto.
            await db.commit()

    return {
        **summary,
        "teams_matched": len(matches),
        "teams_unmatched": unmatched,
    }
