"""Concurrencia acotada de los trabajos en segundo plano.

Antes cada botón lanzaba una BackgroundTask que arrancaba al instante: nada
impedía pulsar los cinco a la vez y tener cinco XGBoost (n_jobs=-1) y cinco
escritores SQLite compitiendo. `_job_lock` los serializa.

Nota: con `httpx.ASGITransport`, `await client.post(...)` no vuelve hasta que
la BackgroundTask ha terminado (Starlette la ejecuta como parte del propio
ciclo ASGI), así que dos POST secuenciales del cliente de test NUNCA se
solapan — no sirven para probar concurrencia. Para eso hay que lanzar dos
`_run_job` con `asyncio.gather` directamente.
"""
import asyncio

import aiosqlite
import pytest


async def _crear_job_status(db, league_code: str) -> int:
    cursor = await db.execute(
        "INSERT INTO job_status (job_type, league_code, status, progress, created_at, updated_at) "
        "VALUES ('update_data', ?, 'queued', 0, '', '')",
        (league_code,),
    )
    await db.commit()
    return cursor.lastrowid


async def test_el_lock_impide_que_dos_jobs_corran_a_la_vez(temp_db, monkeypatch):
    import app.routers.jobs as jobs_module

    intervalos = []

    async def handler_lento(ctx, league_code):
        inicio = asyncio.get_event_loop().time()
        await asyncio.sleep(0.05)
        intervalos.append((inicio, asyncio.get_event_loop().time()))
        return {"mensaje": "ok"}

    monkeypatch.setitem(jobs_module.JOB_HANDLERS, "update_data", handler_lento)

    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        job1 = await _crear_job_status(db, "SP1")
        job2 = await _crear_job_status(db, "E0")

    # Lanzados de verdad a la vez: si no hubiera lock, sus 0.05s se solaparían.
    await asyncio.gather(
        jobs_module._run_job(job1, "update_data", "SP1"),
        jobs_module._run_job(job2, "update_data", "E0"),
    )

    assert len(intervalos) == 2
    (inicio_a, fin_a), (inicio_b, fin_b) = intervalos
    assert fin_a <= inicio_b or fin_b <= inicio_a, "los dos handlers se solaparon en el tiempo"

    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        for job_id in (job1, job2):
            cur = await db.execute("SELECT status FROM job_status WHERE id=?", (job_id,))
            assert (await cur.fetchone())["status"] == "done"


async def test_segundo_job_muestra_que_espera_turno(temp_db, monkeypatch):
    import app.routers.jobs as jobs_module

    liberar = asyncio.Event()

    async def handler_bloqueante(ctx, league_code):
        await liberar.wait()
        return {"mensaje": "ok"}

    monkeypatch.setitem(jobs_module.JOB_HANDLERS, "update_data", handler_bloqueante)

    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        job1 = await _crear_job_status(db, "SP1")
        job2 = await _crear_job_status(db, "E0")

    tarea1 = asyncio.create_task(jobs_module._run_job(job1, "update_data", "SP1"))
    await asyncio.sleep(0.02)  # deja que job1 entre en el lock y quede bloqueado

    tarea2 = asyncio.create_task(jobs_module._run_job(job2, "update_data", "E0"))
    await asyncio.sleep(0.02)  # job2 debería estar esperando el lock, no ejecutando

    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT status, message FROM job_status WHERE id=?", (job2,))
        fila = await cur.fetchone()

    assert fila["status"] == "queued"
    assert "esperando su turno" in fila["message"]

    liberar.set()
    await asyncio.wait_for(asyncio.gather(tarea1, tarea2), timeout=2)
