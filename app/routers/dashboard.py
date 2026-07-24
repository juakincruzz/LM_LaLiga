from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
import aiosqlite

from app.dependencies import get_db_dependency
from app.config import list_leagues

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def dashboard(request: Request, db=Depends(get_db_dependency)):
    leagues = list_leagues()
    # Fetch last update per league from DB
    league_status = []
    for league in leagues:
        cursor = await db.execute(
            "SELECT last_updated, is_active FROM leagues WHERE code = ?",
            (league["code"],)
        )
        row = await cursor.fetchone()
        if row:
            league_status.append({
                **league,
                "last_updated": row["last_updated"],
                "is_active": bool(row["is_active"])
            })
        else:
            league_status.append({**league, "last_updated": None, "is_active": True})

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"leagues": league_status}
    )
