"""Comprobaciones de la portada (página de La Liga): estado y códigos HTTP."""
import json

import aiosqlite

from app.services.data_service import update_league_data
from app.services.season_state import EstadoTemporada, get_season_state


async def test_liga_desconocida_devuelve_404(client):
    # Las ligas que se eliminaron (p. ej. la Premier) ya no existen
    resp = await client.get("/league/E0")
    assert resp.status_code == 404


async def test_job_type_invalido_devuelve_400_y_no_crea_fila(client, temp_db):
    resp = await client.post("/api/jobs/?job_type=inventado&league_code=SP1")
    assert resp.status_code == 400

    # Una fila huérfana en 'queued' haría que el navegador sondeara para siempre
    async with aiosqlite.connect(temp_db) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM job_status")
        assert (await cursor.fetchone())[0] == 0


async def test_portada_muestra_la_ultima_simulacion(client, temp_db):
    """La tabla de la portada es la última fila de `simulations`, con su etiqueta."""
    tabla = json.dumps([{"name": "Barcelona", "points": 88, "gf": 90, "ga": 30,
                         "champion_prob": 0.61, "relegation_prob": 0.0}])
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            "INSERT INTO simulations (model_run_id, league_code, sim_count, standings_json, label, season) "
            "VALUES (NULL, 'SP1', 10, ?, 'Proyección 26/27', '2526')",
            (tabla,),
        )
        await db.commit()

    resp = await client.get("/")
    assert resp.status_code == 200
    assert "Barcelona" in resp.text
    assert "Proyección 26/27" in resp.text
    assert "61.0%" in resp.text


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

    resp = await client.get("/")
    assert resp.status_code == 200
    assert "380" in resp.text
    assert "Temporada finalizada" in resp.text


async def test_no_muestra_simulaciones_de_otra_temporada(client, temp_db):
    """Al pasar a una temporada nueva, la proyección hecha desde la anterior
    (con «Ascendido 1»…) no debe presentarse como la tabla actual."""
    tabla = json.dumps([{"name": "Ascendido 1", "points": 30, "gf": 30, "ga": 60,
                         "champion_prob": 0.0, "relegation_prob": 0.9}])
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            "INSERT INTO simulations (model_run_id, league_code, sim_count, standings_json, label, season) "
            "VALUES (NULL, 'SP1', 10, ?, 'Proyección vieja', '2425')",
            (tabla,),
        )
        await db.commit()

    resp = await client.get("/")
    assert "Ascendido 1" not in resp.text


async def test_clasificacion_lleva_escudos(client, temp_db):
    tabla = json.dumps([
        {"name": n, "points": 50, "gf": 40, "ga": 30, "champion_prob": 0.1, "relegation_prob": 0.0}
        for n in ("Barcelona", "Betis")
    ])
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            "INSERT INTO simulations (model_run_id, league_code, sim_count, standings_json, label, season) "
            "VALUES (NULL, 'SP1', 10, ?, 'Simulación', '2526')",
            (tabla,),
        )
        await db.commit()

    resp = await client.get("/")
    assert resp.text.count('class="escudo"') == 2
    assert "#00aa00" in resp.text  # el verde del Betis
