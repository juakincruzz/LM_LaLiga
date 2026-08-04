from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.database import init_db
from app.routers import dashboard, jobs, leagues
from core.errors import UnknownLeagueError


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="LM_LaLiga Dashboard", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(dashboard.router)
app.include_router(jobs.router)
app.include_router(leagues.router)


@app.exception_handler(UnknownLeagueError)
async def unknown_league_handler(request: Request, exc: UnknownLeagueError) -> HTMLResponse:
    """Una liga inexistente es un 404, no un 500 con traza en consola."""
    return HTMLResponse(
        status_code=404,
        content=(
            "<h1>404 &mdash; Liga no encontrada</h1>"
            f"<p>{exc}</p>"
            '<p><a href="/">Volver al Dashboard</a></p>'
        ),
    )
