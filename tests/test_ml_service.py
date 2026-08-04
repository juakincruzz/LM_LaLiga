import aiosqlite

from app.services.data_service import update_league_data
from app.services.ml_service import train_league_model


async def test_accuracy_es_fuera_de_muestra(temp_db, mock_download):
    """La accuracy reportada debe medirse sobre partidos no vistos.

    Este test es el que habría cazado el 0,9868 que llegó a producción: era
    la accuracy sobre el propio conjunto de entrenamiento.
    """
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        await update_league_data(db, "SP1")
        run_id, accuracy = await train_league_model(db, "SP1")

        cursor = await db.execute("SELECT * FROM model_runs WHERE id=?", (run_id,))
        run = dict(await cursor.fetchone())

        cursor = await db.execute(
            "SELECT COUNT(*) AS n FROM predictions WHERE model_run_id=?", (run_id,)
        )
        n_predicciones = (await cursor.fetchone())["n"]

    # Un modelo de resultados de fútbol no pasa del ~55 % fuera de muestra
    assert 0.30 <= accuracy <= 0.65, f"accuracy sospechosa: {accuracy}"
    assert run["status"] == "completed"
    # 5 temporadas de fixture (2122..2526): las 4 anteriores entrenan, la
    # actual (2526, la que ingiere update_league_data) es el test.
    assert run["n_train"] == 1520 and run["n_test"] == 380
    assert run["seasons_used"] == "2122,2223,2324,2425,2526"
    assert run["baseline_home"] is not None
    assert run["baseline_majority"] is not None
    assert run["logloss"] is not None and run["brier"] is not None
    assert run["test_period"]

    # Las predicciones del test se persisten y quedan enlazadas a matches
    # (solo la temporada actual está en `matches`, que es justo la de test)
    assert n_predicciones == run["n_test"]


async def test_probabilidades_suman_uno(temp_db, mock_download):
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        await update_league_data(db, "SP1")
        run_id, _ = await train_league_model(db, "SP1")

        cursor = await db.execute(
            "SELECT prob_home, prob_draw, prob_away FROM predictions WHERE model_run_id=? LIMIT 5",
            (run_id,),
        )
        filas = await cursor.fetchall()

    assert filas
    for fila in filas:
        total = fila["prob_home"] + fila["prob_draw"] + fila["prob_away"]
        assert abs(total - 1.0) < 1e-6
