import aiosqlite

from app.services.data_service import update_league_data


async def _contar(db, tabla: str, league_code: str = "SP1") -> int:
    cursor = await db.execute(f"SELECT COUNT(*) AS n FROM {tabla} WHERE league_code=?", (league_code,))
    return (await cursor.fetchone())["n"]


async def test_update_puebla_teams_y_matches(temp_db, mock_download):
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        resumen = await update_league_data(db, "SP1")

        assert await _contar(db, "teams") == 20
        assert await _contar(db, "matches") == 380

        # Los partidos quedan enlazados a equipos, no con IDs sueltos
        cursor = await db.execute(
            "SELECT COUNT(*) AS n FROM matches WHERE league_code='SP1' AND home_team_id IS NULL"
        )
        assert (await cursor.fetchone())["n"] == 0

    # El calendario de test no existe (404): se sigue con los resultados y se avisa
    assert {k: resumen[k] for k in ("equipos", "partidos", "jugados")} == {
        "equipos": 20, "partidos": 380, "jugados": 380,
    }
    assert "sin calendario" in resumen["aviso"]


async def test_update_es_idempotente(temp_db, mock_download):
    """Pulsar 'Actualizar datos' dos veces no duplica filas."""
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        await update_league_data(db, "SP1")
        await update_league_data(db, "SP1")

        assert await _contar(db, "teams") == 20
        assert await _contar(db, "matches") == 380


async def test_update_descarga_siempre(temp_db, mock_download):
    """El CSV ya está en disco, pero actualizar debe volver a descargarlo:
    si no, el botón nunca traería los resultados de la última jornada."""
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        await update_league_data(db, "SP1")

    descargas_csv = [c for c in mock_download.call_args_list if "football-data" in c.args[0]]
    assert len(descargas_csv) == 1


async def test_actualizar_trae_alineaciones(temp_db, mock_download, mock_understat):
    progreso = []
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        resumen = await update_league_data(db, "SP1", on_progress=lambda h, t: progreso.append((h, t)))

    assert resumen["jugadores"] == 6
    assert "aviso_jugadores" not in resumen
    assert progreso[-1] == (6, 6)


async def test_actualizar_sin_understat_sigue(temp_db, mock_download):
    # Sin mock_understat: la fixture autouse deja Understat sin red
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        resumen = await update_league_data(db, "SP1")
        assert await _contar(db, "matches") == 380

    assert resumen["aviso_jugadores"].startswith("sin datos de jugadores")


async def test_job_de_actualizar_menciona_el_aviso_de_jugadores(client, temp_db, mock_download):
    import asyncio

    resp = await client.post("/api/jobs/?job_type=update_data")
    assert resp.status_code == 200
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        for _ in range(100):
            fila = await (await db.execute("SELECT status, message FROM job_status")).fetchone()
            if fila["status"] in ("done", "failed"):
                break
            await asyncio.sleep(0.05)
    assert fila["status"] == "done"
    assert "sin datos de jugadores" in fila["message"]


async def test_actualizar_sobrevive_a_un_cambio_de_formato_de_understat(temp_db, mock_download, monkeypatch):
    import scripts.understat as understat

    monkeypatch.setattr(understat, "_get", lambda url: {"dates": None})
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        resumen = await update_league_data(db, "SP1")
        assert await _contar(db, "matches") == 380
    assert resumen["aviso_jugadores"].startswith("sin datos de jugadores (TypeError")
