"""Portada: la página de La Liga.

Antes había un dashboard con una tarjeta por liga y una página por liga en
/league/{code}. Con una sola liga, la portada ES la página de la liga.
"""
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse

from app.config import LEAGUE_CODE, get_current_season, get_league_config, season_label
from app.dependencies import get_db_dependency
from app.services.season_state import get_season_state
from app.templating import templates
from core.equipos import nombre_visible

router = APIRouter()


@router.get("/")
async def portada(request: Request, db=Depends(get_db_dependency)):
    config = get_league_config(LEAGUE_CODE)

    cursor = await db.execute("SELECT * FROM leagues WHERE code=?", (LEAGUE_CODE,))
    league_row = await cursor.fetchone()

    temporada = await get_season_state(db, LEAGUE_CODE)

    # Última simulación hecha DESDE la temporada actual. `label` distingue el
    # Monte Carlo de la temporada en curso de la proyección de la siguiente.
    cursor = await db.execute(
        """
        SELECT standings_json, label, sim_count, created_at
          FROM simulations
         WHERE league_code = ? AND season = ?
         ORDER BY id DESC
         LIMIT 1
        """,
        (LEAGUE_CODE, get_current_season()),
    )
    sim_row = await cursor.fetchone()

    # Último entrenamiento con métricas honestas. baseline_home IS NOT NULL
    # descarta los entrenamientos antiguos, cuya accuracy estaba medida sobre
    # el propio conjunto de entrenamiento (~0,99) y no significa nada.
    cursor = await db.execute(
        """
        SELECT accuracy, baseline_home, baseline_majority, baseline_bookmaker,
               logloss, brier, seasons_used, n_train, n_test, test_period,
               model_type, created_at
          FROM model_runs
         WHERE league_code = ? AND status = 'completed' AND baseline_home IS NOT NULL
         ORDER BY created_at DESC
         LIMIT 1
        """,
        (LEAGUE_CODE,),
    )
    run_row = await cursor.fetchone()

    # Equipos de la temporada en curso, para el selector del «cara a cara»
    cursor = await db.execute(
        """
        SELECT DISTINCT t.name
          FROM matches m
          JOIN teams t ON t.id IN (m.home_team_id, m.away_team_id)
         WHERE m.league_code = ? AND m.season = ?
        """,
        (LEAGUE_CODE, get_current_season()),
    )
    equipos = sorted((row["name"] for row in await cursor.fetchall()), key=nombre_visible)

    return templates.TemplateResponse(
        request,
        "liga.html",
        {
            "league": config,
            "league_code": LEAGUE_CODE,
            "db_league": dict(league_row) if league_row else None,
            "temporada": temporada,
            "temporada_label": season_label(temporada.season or get_current_season()),
            "modelo": dict(run_row) if run_row else None,
            "sim_standings": json.loads(sim_row["standings_json"]) if sim_row else None,
            "sim_label": sim_row["label"] if sim_row else None,
            "sim_count": sim_row["sim_count"] if sim_row else None,
            "next_season_label": config.get("next_season", {}).get("label", "próxima temporada"),
            "equipos": equipos,
        },
    )


@router.get("/league/{league_code}")
async def ruta_antigua(league_code: str):
    """Compatibilidad con los enlaces de la versión multi-liga."""
    get_league_config(league_code)  # cualquier otra liga → UnknownLeagueError → 404
    return RedirectResponse("/", status_code=301)
