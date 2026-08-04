"""Lanzamiento y seguimiento de trabajos en segundo plano.

Cada tarea abre su propia conexión a SQLite: la del request
(`get_db_dependency`) se cierra al enviar la respuesta, así que no se puede
reutilizar desde una BackgroundTask.

El trabajo pesado de CPU se saca del event loop dentro de cada servicio
(`asyncio.to_thread`), no aquí: así el acceso a la BD sigue siendo asíncrono y
el sondeo HTMX puede responder mientras el modelo entrena.

Los trabajos se serializan con `_job_lock`: nada impedía antes lanzar cinco
entrenamientos XGBoost (n_jobs=-1) a la vez, cada uno reclamando todos los
núcleos, más cinco escritores SQLite concurrentes. Con el lock, 'queued'
significa de verdad "esperando su turno" en vez de arrancar al instante.

(Se descartó una cola con hilo trabajador dedicado: los cinco servicios ya son
asíncronos y usan aiosqlite, así que forzarlos a síncrono para encajar en un
hilo puro habría significado reescribir el acceso a datos de los cinco sin
ganar nada que el lock no dé ya.)
"""
import asyncio
import json
from datetime import datetime
from typing import Awaitable, Callable

import aiosqlite
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from app.config import get_league_config
from app.database import DB_PATH
from app.dependencies import get_db_dependency
from app.templating import templates

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Un solo job de CPU/red a la vez. asyncio.Lock sirve a los que esperan en el
# orden en que llamaron a acquire(), así que además da FIFO gratis.
_job_lock = asyncio.Lock()


async def _update_job(
    db: aiosqlite.Connection,
    job_id: int,
    status: str | None = None,
    progress: int | None = None,
    result_json: str | None = None,
    log: str | None = None,
    message: str | None = None,
):
    """Actualiza solo las columnas indicadas.

    Antes escribía siempre todas, de modo que pasar a 'running' borraba el
    resultado o el log anteriores.
    """
    campos = {
        "status": status,
        "progress": progress,
        "result_json": result_json,
        "log": log,
        "message": message,
    }
    asignaciones = {k: v for k, v in campos.items() if v is not None}
    asignaciones["updated_at"] = datetime.now().isoformat()

    sets = ", ".join(f"{col}=?" for col in asignaciones)
    await db.execute(f"UPDATE job_status SET {sets} WHERE id=?", (*asignaciones.values(), job_id))
    await db.commit()


# --- Handlers: (ctx, league_code) -> dict de resultado ---------------------


async def _job_update_data(ctx, league_code) -> dict:
    from app.services.data_service import update_league_data

    await ctx.progress(20, "Descargando datos…")
    resumen = await update_league_data(ctx.db, league_code)
    return {**resumen, "mensaje": f"{resumen['partidos']} partidos ({resumen['jugados']} jugados)"}


async def _job_train(ctx, league_code) -> dict:
    from app.services.ml_service import train_league_model

    await ctx.progress(30, "Generando features y entrenando…")
    run_id, accuracy = await train_league_model(ctx.db, league_code)
    return {"run_id": run_id, "accuracy": accuracy, "mensaje": f"Precisión fuera de muestra: {accuracy:.1%}"}


async def _job_simulate(ctx, league_code) -> dict:
    from app.services.simulation_service import run_simulation

    await ctx.progress(10, "Reconstruyendo el estado de la temporada…")
    await run_simulation(ctx.db, league_code, on_progress=ctx.progress_threadsafe)
    return {"mensaje": "Simulación completada"}


async def _job_project_next_season(ctx, league_code) -> dict:
    from app.services.next_season_service import project_next_season

    await ctx.progress(20, "Proyectando la próxima temporada…")
    await project_next_season(ctx.db, league_code)
    return {"mensaje": "Proyección completada"}


async def _job_analyze_transfers(ctx, league_code) -> dict:
    from app.services.transfer_service import analyze_league_transfers

    await ctx.progress(15, "Consultando Transfermarkt…")
    return await analyze_league_transfers(ctx.db, league_code)


JobHandler = Callable[["JobContext", str], Awaitable[dict]]

JOB_HANDLERS: dict[str, JobHandler] = {
    "update_data": _job_update_data,
    "train": _job_train,
    "simulate": _job_simulate,
    "project_next_season": _job_project_next_season,
    "analyze_transfers": _job_analyze_transfers,
}


class JobContext:
    """Lo que ve un handler: su conexión, su id, y cómo reportar progreso."""

    def __init__(self, db: aiosqlite.Connection, job_id: int, loop: asyncio.AbstractEventLoop):
        self.db = db
        self.job_id = job_id
        self._loop = loop

    async def progress(self, pct: int, message: str) -> None:
        await _update_job(self.db, self.job_id, progress=pct, message=message)

    def progress_threadsafe(self, done: int, total: int) -> None:
        """Para llamar desde un hilo de `asyncio.to_thread` (p.ej. dentro del
        Monte Carlo), donde no hay event loop propio para hacer `await`."""
        pct = 15 + int(80 * done / total)
        coro = self.progress(pct, f"Simulación {done}/{total}")
        asyncio.run_coroutine_threadsafe(coro, self._loop)


async def _run_job(job_id: int, job_type: str, league_code: str):
    """Ejecuta un job con su propia conexión, en exclusión con cualquier otro."""
    loop = asyncio.get_running_loop()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if _job_lock.locked():
            await _update_job(db, job_id, message="En cola: esperando su turno…")

        async with _job_lock:
            await _update_job(db, job_id, status="running", progress=10, message="En marcha…")
            ctx = JobContext(db, job_id, loop)
            try:
                resultado = await JOB_HANDLERS[job_type](ctx, league_code)
                await _update_job(
                    db,
                    job_id,
                    status="done",
                    progress=100,
                    result_json=json.dumps({**resultado, "url": f"/league/{league_code}"}),
                    message=resultado.get("mensaje", "Completado"),
                )
            except Exception as e:
                # El tipo de excepción es la mitad del diagnóstico y antes se perdía
                await _update_job(db, job_id, status="failed", progress=0, log=f"{type(e).__name__}: {e}")


@router.post("/")
async def create_job(
    request: Request,
    job_type: str,
    league_code: str,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_dependency),
):
    # Validar ANTES de insertar: un job_type inválido creaba una fila que nadie
    # ejecutaba nunca y que el navegador sondeaba indefinidamente.
    if job_type not in JOB_HANDLERS:
        raise HTTPException(status_code=400, detail=f"Tipo de trabajo desconocido: {job_type}")
    get_league_config(league_code)  # lanza UnknownLeagueError → 404

    ahora = datetime.now().isoformat()
    cursor = await db.execute(
        "INSERT INTO job_status (job_type, league_code, status, progress, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (job_type, league_code, "queued", 0, ahora, ahora),
    )
    await db.commit()
    job_id = cursor.lastrowid

    background_tasks.add_task(_run_job, job_id, job_type, league_code)

    job = {
        "id": job_id,
        "job_type": job_type,
        "league_code": league_code,
        "status": "queued",
        "progress": 0,
        "result_json": None,
        "log": None,
        "message": "En cola: esperando su turno…" if _job_lock.locked() else "En cola",
        "result_url": "",
    }
    return templates.TemplateResponse(request, "partials/job_status.html", {"job": job})


@router.get("/{job_id}")
async def get_job(job_id: int, db=Depends(get_db_dependency)):
    cursor = await db.execute("SELECT * FROM job_status WHERE id=?", (job_id,))
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    return dict(row)


@router.get("/{job_id}/poll")
async def poll_job(request: Request, job_id: int, db=Depends(get_db_dependency)):
    cursor = await db.execute("SELECT * FROM job_status WHERE id=?", (job_id,))
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")

    job = dict(row)
    try:
        resultado = json.loads(job.get("result_json") or "{}")
    except json.JSONDecodeError:
        resultado = {}
    job["result_url"] = resultado.get("url", "")
    return templates.TemplateResponse(request, "partials/job_status.html", {"job": job})
