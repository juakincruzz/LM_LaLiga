from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.database import init_db
from app.routers import jobs, liga, prediccion
from app.templating import templates
from core.errors import UnknownLeagueError


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="LM LaLiga", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(liga.router)
app.include_router(jobs.router)
app.include_router(prediccion.router)


@app.exception_handler(UnknownLeagueError)
async def unknown_league_handler(request: Request, exc: UnknownLeagueError):
    """Una liga inexistente es un 404, no un 500 con traza en consola."""
    return templates.TemplateResponse(
        request, "404.html", {"mensaje": str(exc)}, status_code=404
    )
