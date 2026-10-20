import asyncio
import json
from datetime import datetime

import aiosqlite

from app.config import get_current_season
from core.errors import ModelNotTrainedError
from scripts.season_simulator import simulate_season


async def run_simulation(
    db: aiosqlite.Connection,
    league_code: str,
    sim_count: int = 500,
    seed: int = 42,
    on_progress=None,
):
    """Ejecuta el Monte Carlo de la temporada en curso y guarda la clasificación.

    `on_progress(hechas, total)`, si se pasa, se llama desde el hilo de
    `asyncio.to_thread` — quien la reciba debe saber volver al event loop
    (ver `JobContext.progress_threadsafe` en app/routers/jobs.py).
    """
    cursor = await db.execute(
        "SELECT id FROM model_runs WHERE league_code=? AND status='completed' "
        "ORDER BY created_at DESC LIMIT 1",
        (league_code,),
    )
    row = await cursor.fetchone()
    if not row:
        raise ModelNotTrainedError(
            "No hay modelo entrenado para esta liga. Pulsa 'Entrenar modelo' primero."
        )

    run_id = row["id"]
    # Fuera del event loop: el Monte Carlo bloquea durante minutos.
    # season=None dentro de simulate_season se resuelve a la temporada actual.
    standings = await asyncio.to_thread(
        simulate_season, league_code, sim_count, None, seed, on_progress
    )

    await db.execute(
        """
        INSERT INTO simulations
            (model_run_id, league_code, sim_count, standings_json, label, seed, created_at, season)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            league_code,
            sim_count,
            json.dumps(standings),
            "Temporada actual",
            seed,
            datetime.now().isoformat(),
            get_current_season(),
        ),
    )
    await db.commit()
    return standings
