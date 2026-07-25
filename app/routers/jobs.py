import json
from datetime import datetime
from fastapi import APIRouter, Request, Depends, BackgroundTasks
from fastapi.templating import Jinja2Templates
import aiosqlite

from app.dependencies import get_db_dependency
from app.database import DB_PATH

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
templates = Jinja2Templates(directory="app/templates")


async def _update_job(db: aiosqlite.Connection, job_id: int, status: str, progress: int, result_json: str = None, log: str = None):
    await db.execute(
        "UPDATE job_status SET status=?, progress=?, result_json=?, updated_at=?, log=? WHERE id=?",
        (status, progress, result_json, datetime.now().isoformat(), log, job_id)
    )
    await db.commit()


async def _run_data_update_job(job_id: int, league_code: str):
    """Background task: open its own DB connection so it outlives the request."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _update_job(db, job_id, "running", 10)
        try:
            from app.services.data_service import update_league_data
            await update_league_data(db, league_code)
            await _update_job(db, job_id, "done", 100, result_json='{"url": "/league/' + league_code + '"}')
        except Exception as e:
            await _update_job(db, job_id, "failed", 0, log=str(e))


async def _run_train_job(job_id: int, league_code: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _update_job(db, job_id, "running", 5)
        try:
            from app.services.ml_service import train_league_model
            run_id, accuracy = await train_league_model(db, league_code)
            await _update_job(
                db, job_id, "done", 100,
                result_json=json.dumps({"run_id": run_id, "accuracy": accuracy, "url": f"/league/{league_code}"})
            )
        except Exception as e:
            await _update_job(db, job_id, "failed", 0, log=str(e))


async def _run_simulate_job(job_id: int, league_code: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _update_job(db, job_id, "running", 10)
        try:
            from app.services.simulation_service import run_simulation
            standings = await run_simulation(db, league_code)
            await _update_job(
                db, job_id, "done", 100,
                result_json=json.dumps({"url": f"/league/{league_code}"})
            )
        except Exception as e:
            await _update_job(db, job_id, "failed", 0, log=str(e))


async def _run_project_next_season_job(job_id: int, league_code: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _update_job(db, job_id, "running", 10)
        try:
            from app.services.next_season_service import project_next_season
            standings = await project_next_season(db, league_code)
            await _update_job(
                db, job_id, "done", 100,
                result_json=json.dumps({"url": f"/league/{league_code}"})
            )
        except Exception as e:
            await _update_job(db, job_id, "failed", 0, log=str(e))


@router.post("/")
async def create_job(request: Request, job_type: str, league_code: str, background_tasks: BackgroundTasks, db=Depends(get_db_dependency)):
    cursor = await db.execute(
        "INSERT INTO job_status (job_type, league_code, status, progress, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (job_type, league_code, "queued", 0, datetime.now().isoformat(), datetime.now().isoformat())
    )
    await db.commit()
    job_id = cursor.lastrowid

    if job_type == "update_data":
        background_tasks.add_task(_run_data_update_job, job_id, league_code)
    elif job_type == "train":
        background_tasks.add_task(_run_train_job, job_id, league_code)
    elif job_type == "simulate":
        background_tasks.add_task(_run_simulate_job, job_id, league_code)
    elif job_type == "project_next_season":
        background_tasks.add_task(_run_project_next_season_job, job_id, league_code)

    # Return the job status partial so HTMX inserts it into the DOM immediately
    job = {
        "id": job_id,
        "job_type": job_type,
        "league_code": league_code,
        "status": "queued",
        "progress": 0,
        "result_json": None,
        "log": None,
    }
    job["result_url"] = ""
    return templates.TemplateResponse(request, "partials/job_status.html", {"job": job})


@router.get("/{job_id}")
async def get_job(job_id: int, db=Depends(get_db_dependency)):
    cursor = await db.execute("SELECT * FROM job_status WHERE id=?", (job_id,))
    row = await cursor.fetchone()
    if not row:
        return {"error": "Job not found"}
    return dict(row)


@router.get("/{job_id}/poll")
async def poll_job(request: Request, job_id: int, db=Depends(get_db_dependency)):
    cursor = await db.execute("SELECT * FROM job_status WHERE id=?", (job_id,))
    row = await cursor.fetchone()
    if not row:
        return "Job not found"
    job = dict(row)
    result_json_str = job.get("result_json", "{}") or "{}"
    result = json.loads(result_json_str)
    job["result_url"] = result.get("url", "")
    return templates.TemplateResponse(request, "partials/job_status.html", {"job": job})
