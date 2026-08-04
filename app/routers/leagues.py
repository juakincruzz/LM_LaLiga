import json

from fastapi import APIRouter, Depends, Request

from app.config import get_league_config
from app.dependencies import get_db_dependency
from app.templating import templates
from app.services.season_state import get_season_state

router = APIRouter()


@router.get("/league/{league_code}")
async def league_detail(request: Request, league_code: str, db=Depends(get_db_dependency)):
    config = get_league_config(league_code)

    cursor = await db.execute("SELECT * FROM leagues WHERE code=?", (league_code,))
    league_row = await cursor.fetchone()

    temporada = await get_season_state(db, league_code)

    # Última simulación DE ESTA LIGA. Antes el filtro era por model_run_id, que
    # es NULL en las proyecciones de ligas sin modelo entrenado, y esas filas
    # acababan mostrándose en la página de todas las demás ligas.
    cursor = await db.execute(
        """
        SELECT standings_json, label
          FROM simulations
         WHERE league_code = ?
         ORDER BY id DESC
         LIMIT 1
        """,
        (league_code,),
    )
    sim_row = await cursor.fetchone()
    sim_standings = json.loads(sim_row["standings_json"]) if sim_row else None
    sim_label = sim_row["label"] if sim_row else None

    # Último entrenamiento, para mostrar las métricas honestas junto a la tabla.
    # baseline_home IS NOT NULL descarta los entrenamientos anteriores a esta
    # versión, cuya accuracy estaba medida sobre el propio conjunto de
    # entrenamiento (~0,99) y no significa nada.
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
        (league_code,),
    )
    run_row = await cursor.fetchone()

    next_label = config.get("next_season", {}).get("label", "próxima temporada")

    return templates.TemplateResponse(
        request,
        "league_detail.html",
        {
            "league": config,
            "league_code": league_code,
            "db_league": dict(league_row) if league_row else None,
            "temporada": temporada,
            "match_count": temporada.total,
            "modelo": dict(run_row) if run_row else None,
            "sim_standings": sim_standings,
            "sim_label": sim_label,
            "next_season_label": next_label,
        },
    )
