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

    assert resumen == {"equipos": 20, "partidos": 380, "jugados": 380}


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

    mock_download.assert_called_once()
