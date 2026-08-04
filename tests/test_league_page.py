"""Comprobaciones de la página de liga: aislamiento, estado y códigos HTTP."""
import json

import aiosqlite

from app.services.data_service import update_league_data
from app.services.season_state import EstadoTemporada, get_season_state


async def test_liga_desconocida_devuelve_404(client):
    resp = await client.get("/league/XX")
    assert resp.status_code == 404


async def test_job_type_invalido_devuelve_400_y_no_crea_fila(client, temp_db):
    resp = await client.post("/api/jobs/?job_type=inventado&league_code=SP1")
    assert resp.status_code == 400

    # Una fila huérfana en 'queued' haría que el navegador sondeara para siempre
    async with aiosqlite.connect(temp_db) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM job_status")
        assert (await cursor.fetchone())[0] == 0


async def test_simulacion_de_otra_liga_no_aparece(client, temp_db):
    """Una proyección de Bundesliga no debe mostrarse en la página de La Liga."""
    tabla = json.dumps([{"name": "Bayern", "points": 80, "gf": 90, "ga": 30,
                         "champion_prob": 0.9, "relegation_prob": 0.0}])
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            "INSERT INTO simulations (model_run_id, league_code, sim_count, standings_json, label) "
            "VALUES (NULL, 'D1', 10, ?, 'Proyección 26/27')",
            (tabla,),
        )
        await db.commit()

    resp = await client.get("/league/SP1")
    assert resp.status_code == 200
    assert "Bayern" not in resp.text

    resp = await client.get("/league/D1")
    assert "Bayern" in resp.text


async def test_estado_temporada_finalizada(temp_db, mock_download):
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        await update_league_data(db, "SP1")
        estado = await get_season_state(db, "SP1")

    # La fixture es una temporada completa: 380 jugados, 0 pendientes
    assert estado.estado is EstadoTemporada.FINALIZADA
    assert (estado.total, estado.jugados, estado.pendientes) == (380, 380, 0)
    assert estado.puede_simular is False


async def test_estado_sin_datos(temp_db):
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        estado = await get_season_state(db, "SP1")

    assert estado.estado is EstadoTemporada.SIN_DATOS
    assert estado.puede_simular is False


async def test_pagina_muestra_partidos_reales(client, temp_db, mock_download):
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        await update_league_data(db, "SP1")

    resp = await client.get("/league/SP1")
    assert resp.status_code == 200
    assert "380" in resp.text
    assert "Temporada finalizada" in resp.text
