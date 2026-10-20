"""Proyección de la siguiente temporada basada en la temporada completada.

Estrategia:
1. Calcula ratings de ataque/defensa de cada equipo a partir de la temporada
   finalizada (goles a favor/en contra relativos a la media de liga).
2. Sustituye los equipos descendidos (cola de la tabla) por los ascendidos
   configurados en config/leagues.yaml con ratings típicos de recién ascendido.
3. Genera un fixture round-robin completo y ejecuta Monte Carlo con goles
   Poisson para estimar la tabla final y probabilidades.
"""
import json
from collections import defaultdict
from datetime import datetime

import aiosqlite
import numpy as np
import pandas as pd

import re

from app.config import get_current_season, get_league_config, season_label
from core.dataset import load_season, played_only
from core.errors import PromotedTeamsNotConfiguredError, SeasonInProgressError

# Ratings típicos de un equipo recién ascendido (relativos a media de liga = 1.0)
PROMOTED_ATTACK = 0.75
PROMOTED_DEFENSE = 1.25


# Los nombres de relleno con que se distribuye config/leagues.yaml
PROVISIONAL = re.compile(r"^Ascendido \d+$")


def _load_season_df(league_code: str, season: str) -> pd.DataFrame:
    """Partidos jugados de la temporada, que tiene que estar TERMINADA.

    Con la temporada a medias, la proyección tomaba los partidos jugados como
    si fueran la temporada entera, daba por descendidos a los colistas de ese
    momento y ponía en su lugar a los ascendidos provisionales. Para la
    temporada en curso está «Simular temporada».
    """
    completa = load_season(league_code, season, download=False)  # FileNotFoundError si no hay datos
    df = played_only(completa)
    if df.empty:
        raise ValueError(f"La temporada {season_label(season)} no tiene partidos con resultado.")
    pendientes = len(completa) - len(df)
    if pendientes:
        raise SeasonInProgressError(
            f"La temporada {season_label(season)} aún no ha terminado (quedan {pendientes} partidos): "
            f"la proyección de la siguiente se hace con la temporada completa. "
            f"Mientras tanto, usa «Simular temporada»."
        )
    return df


def _comprobar_ascendidos(promoted: list[str]) -> None:
    provisionales = [t for t in promoted if PROVISIONAL.match(t)]
    if provisionales:
        raise PromotedTeamsNotConfiguredError(
            f"Faltan los ascendidos reales: en config/leagues.yaml (next_season.promoted_teams) "
            f"siguen los provisionales {', '.join(provisionales)}. Pon los nombres de "
            f"football-data.co.uk de los equipos que suben."
        )


def _team_stats(df: pd.DataFrame) -> dict:
    """GF/GA/partidos por equipo y medias de liga (casa/fuera)."""
    stats = defaultdict(lambda: {"gf": 0, "ga": 0, "matches": 0, "pts": 0})
    home_goals_total, away_goals_total = 0, 0

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        hg, ag = int(row["Home_Goals"]), int(row["Away_Goals"])
        home_goals_total += hg
        away_goals_total += ag

        stats[h]["gf"] += hg
        stats[h]["ga"] += ag
        stats[h]["matches"] += 1
        stats[a]["gf"] += ag
        stats[a]["ga"] += hg
        stats[a]["matches"] += 1

        if hg > ag:
            stats[h]["pts"] += 3
        elif hg < ag:
            stats[a]["pts"] += 3
        else:
            stats[h]["pts"] += 1
            stats[a]["pts"] += 1

    n_matches = len(df)
    league_avg_home = home_goals_total / n_matches
    league_avg_away = away_goals_total / n_matches
    league_avg_team = (home_goals_total + away_goals_total) / (2 * n_matches)

    return stats, league_avg_home, league_avg_away, league_avg_team


def _final_table_order(stats: dict) -> list[str]:
    """Equipos ordenados por puntos (desc) para detectar descendidos."""
    return [
        team
        for team, _ in sorted(
            stats.items(),
            key=lambda kv: (kv[1]["pts"], kv[1]["gf"] - kv[1]["ga"]),
            reverse=True,
        )
    ]


def _build_ratings(stats: dict, league_avg_team: float, teams_next: list[str]) -> dict:
    """Ratings attack/defense relativos a la media de liga."""
    ratings = {}
    for team in teams_next:
        if team in stats:
            s = stats[team]
            ratings[team] = {
                "attack": (s["gf"] / s["matches"]) / league_avg_team,
                "defense": (s["ga"] / s["matches"]) / league_avg_team,
            }
        else:
            # Equipo ascendido: rating por defecto de recién ascendido
            ratings[team] = {"attack": PROMOTED_ATTACK, "defense": PROMOTED_DEFENSE}
    return ratings


def _round_robin(teams: list[str]) -> list[tuple[str, str]]:
    """Cada equipo juega contra todos, ida y vuelta."""
    return [(h, a) for h in teams for a in teams if h != a]


def _simulate_one_season(teams, ratings, fixtures, league_avg_home, league_avg_away, rng):
    """Simula una temporada completa con goles Poisson. Devuelve tabla ordenada."""
    table = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    for home, away in fixtures:
        home_xg = ratings[home]["attack"] * ratings[away]["defense"] * league_avg_home
        away_xg = ratings[away]["attack"] * ratings[home]["defense"] * league_avg_away
        hg = rng.poisson(home_xg)
        ag = rng.poisson(away_xg)

        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[away]["gf"] += ag
        table[away]["ga"] += hg
        if hg > ag:
            table[home]["pts"] += 3
        elif hg < ag:
            table[away]["pts"] += 3
        else:
            table[home]["pts"] += 1
            table[away]["pts"] += 1

    return sorted(table.items(), key=lambda kv: (kv[1]["pts"], kv[1]["gf"] - kv[1]["ga"]), reverse=True)


async def project_next_season(
    db: aiosqlite.Connection,
    league_code: str,
    season: str | None = None,
    sim_count: int = 500,
    seed: int | None = None,
) -> list[dict]:
    """Proyecta la siguiente temporada. Devuelve lista de equipos con
    puntos/goles medios y probabilidades (mismo formato que simulate_season)."""
    config = get_league_config(league_code)
    next_cfg = config.get("next_season", {})
    promoted = next_cfg.get("promoted_teams", [])
    label = next_cfg.get("label", "próxima temporada")

    season = season or get_current_season()
    df = _load_season_df(league_code, season)
    _comprobar_ascendidos(promoted)
    stats, league_avg_home, league_avg_away, league_avg_team = _team_stats(df)

    # Descendidos: últimos N según nº de ascendidos configurados
    order = _final_table_order(stats)
    relegated = order[-len(promoted):] if promoted else []

    # Plantel de la próxima temporada
    teams_next = [t for t in order if t not in relegated] + list(promoted)

    ratings = _build_ratings(stats, league_avg_team, teams_next)

    fixtures = _round_robin(teams_next)
    rng = np.random.default_rng(seed)

    # Monte Carlo
    champion_counts = defaultdict(int)
    relegation_counts = defaultdict(int)
    points_acc = defaultdict(list)
    gf_acc = defaultdict(list)
    ga_acc = defaultdict(list)

    n_relegation = len(promoted) if promoted else 3
    for _ in range(sim_count):
        table = _simulate_one_season(teams_next, ratings, fixtures, league_avg_home, league_avg_away, rng)
        champion_counts[table[0][0]] += 1
        for team, _ in table[-n_relegation:]:
            relegation_counts[team] += 1
        for team, s in table:
            points_acc[team].append(s["pts"])
            gf_acc[team].append(s["gf"])
            ga_acc[team].append(s["ga"])

    # Tabla media proyectada
    projected = []
    for team in teams_next:
        projected.append({
            "name": team,
            "points": round(float(np.mean(points_acc[team]))),
            "gf": round(float(np.mean(gf_acc[team]))),
            "ga": round(float(np.mean(ga_acc[team]))),
            # 0.0 es un resultado ("no ganó ninguna simulación"), no un hueco:
            # devolver None hacía que la tabla mostrara '—', igual que si no se
            # hubiera calculado nada.
            "champion_prob": champion_counts[team] / sim_count,
            "relegation_prob": relegation_counts[team] / sim_count,
        })
    projected.sort(key=lambda t: t["points"], reverse=True)

    # Persistir como simulación etiquetada
    cursor = await db.execute(
        "SELECT id FROM model_runs WHERE league_code=? AND status='completed' ORDER BY created_at DESC LIMIT 1",
        (league_code,),
    )
    row = await cursor.fetchone()
    model_run_id = row["id"] if row else None

    await db.execute(
        """
        INSERT INTO simulations
            (model_run_id, league_code, sim_count, standings_json, label, seed, created_at, season)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_run_id,
            league_code,
            sim_count,
            json.dumps(projected),
            f"Proyección {label}",
            seed,
            datetime.now().isoformat(),
            season,
        ),
    )
    await db.commit()

    return projected
