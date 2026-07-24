from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
import aiosqlite

from app.dependencies import get_db_dependency
from app.config import get_league_config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


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

    return templates.TemplateResponse(
        request,
        "league_detail.html",
        {
            "league": config,
            "league_code": league_code,
            "db_league": dict(league_row) if league_row else None,
            "match_count": match_count
        }
    )
