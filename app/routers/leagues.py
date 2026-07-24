import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
import aiosqlite

from app.dependencies import get_db_dependency
from app.config import get_league_config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


async def is_offseason(db: aiosqlite.Connection, league_code: str) -> bool:
    cursor = await db.execute(
        "SELECT MAX(date) as last_match FROM matches WHERE league_id=?", (league_code,)
    )
    row = await cursor.fetchone()
    if not row or not row["last_match"]:
        return True
    last_match = datetime.fromisoformat(row["last_match"])
    return (datetime.now() - last_match).days > 30


@router.get("/league/{league_code}")
async def league_detail(request: Request, league_code: str, db=Depends(get_db_dependency)):
    config = get_league_config(league_code)

    cursor = await db.execute(
        "SELECT * FROM leagues WHERE code=?", (league_code,)
    )
    league_row = await cursor.fetchone()

    # Count matches
    cursor = await db.execute(
        "SELECT COUNT(*) as total FROM matches WHERE league_id=?", (league_code,)
    )
    match_count = (await cursor.fetchone())["total"]

    offseason = await is_offseason(db, league_code)

    cursor = await db.execute(
        """
        SELECT s.standings_json
        FROM simulations s
        JOIN model_runs mr ON s.model_run_id = mr.id
        WHERE mr.league_id = ?
        ORDER BY s.id DESC
        LIMIT 1
        """,
        (league_code,)
    )
    sim_row = await cursor.fetchone()
    sim_standings = json.loads(sim_row["standings_json"]) if sim_row else None

    return templates.TemplateResponse(
        request,
        "league_detail.html",
        {
            "league": config,
            "league_code": league_code,
            "db_league": dict(league_row) if league_row else None,
            "match_count": match_count,
            "is_offseason": offseason,
            "sim_standings": sim_standings
        }
    )
