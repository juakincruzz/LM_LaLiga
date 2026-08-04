"""Entrenamiento del modelo de una liga, con métricas honestas.

Antes esta función entrenaba con todos los partidos y medía la accuracy sobre
esos mismos partidos: reportaba ~0,99 donde lo real fuera de muestra era ~0,52.
Ahora entrena sobre varias temporadas (ver app.config.get_training_seasons),
reserva la temporada actual como test, y guarda junto a la accuracy tres
baselines de referencia: "siempre gana el local", la clase mayoritaria del
entrenamiento, y lo que ya acierta el mercado de apuestas.
"""
import asyncio
from datetime import datetime

import aiosqlite
import pandas as pd

from app.config import get_training_seasons
from core.evaluation import EvaluationReport, evaluate_predictions, season_train_test_split
from models.predictor import LeaguePredictor
from scripts.preprocess_generic import preprocess_league


def _probabilidades(predictor: LeaguePredictor, proba_row) -> tuple[float, float, float]:
    """(local, empate, visitante), indexando por model.classes_ — no por
    posición, para no asignar una probabilidad a la clase equivocada si
    alguna faltara en el entrenamiento."""
    prob = predictor._probs_by_outcome(proba_row)
    return prob.get(1, 0.0), prob.get(0, 0.0), prob.get(-1, 0.0)


async def _guardar_predicciones(
    db: aiosqlite.Connection,
    run_id: int,
    league_code: str,
    df_test: pd.DataFrame,
    X_test: pd.DataFrame,
    predictor: LeaguePredictor,
) -> int:
    """Persiste las predicciones del conjunto de test, enlazadas a `matches`."""
    cursor = await db.execute(
        """
        SELECT m.id, m.date, h.name AS home, a.name AS away
          FROM matches m
          JOIN teams h ON h.id = m.home_team_id
          JOIN teams a ON a.id = m.away_team_id
         WHERE m.league_code = ?
        """,
        (league_code,),
    )
    match_ids = {(r["date"], r["home"], r["away"]): r["id"] for r in await cursor.fetchall()}
    if not match_ids:
        return 0  # todavía no se ha pulsado 'Actualizar datos'

    probas = predictor.model.predict_proba(X_test)

    filas = []
    for (_, fila), proba_row in zip(df_test.iterrows(), probas):
        clave = (fila["Date"].date().isoformat(), fila["HomeTeam"], fila["AwayTeam"])
        match_id = match_ids.get(clave)
        if match_id is None:
            continue
        p_home, p_draw, p_away = _probabilidades(predictor, proba_row)
        predicho = max((p_home, "H"), (p_draw, "D"), (p_away, "A"))[1]
        filas.append((run_id, match_id, predicho, p_home, p_draw, p_away))

    await db.executemany(
        """
        INSERT INTO predictions
            (model_run_id, match_id, predicted_result, prob_home, prob_draw, prob_away)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        filas,
    )
    return len(filas)


def _entrenar_y_evaluar(league_code: str):
    """Todo el trabajo de CPU en una función síncrona: descarga de temporadas
    históricas, features y entrenamiento, con un único asyncio.to_thread."""
    seasons = get_training_seasons(league_code)
    X, y, df = preprocess_league(league_code, seasons)
    X_train, X_test, y_train, y_test, df_train, df_test = season_train_test_split(df, X, y, seasons)

    predictor = LeaguePredictor(league_code)
    predictor.train_xgboost(X_train, y_train)

    proba = predictor.model.predict_proba(X_test)
    y_pred_encoded = proba.argmax(axis=1)
    clases = list(predictor.model.classes_)

    reporte = evaluate_predictions(
        y_train=y_train,
        y_test=y_test,
        y_pred_encoded=y_pred_encoded,
        proba=proba,
        classes=clases,
        df_test=df_test,
        encode=predictor._encode_target,
    )

    predictor.accuracy = reporte.accuracy
    predictor.save_model()

    return predictor, reporte, df_test, X_test, seasons


async def train_league_model(db: aiosqlite.Connection, league_code: str) -> tuple[int, float]:
    """Entrena y guarda el modelo de una liga. Devuelve (run_id, accuracy honesta).

    Un solo INSERT con la fila ya completa, no INSERT+UPDATE: para cuando se
    llega aquí todo el trabajo (descarga, features, entrenamiento, evaluación)
    ya ha terminado en el hilo de `asyncio.to_thread`. De propina, si algo
    revienta antes de este punto no queda ninguna fila a medias en
    `model_runs` marcada 'running' para siempre — el error lo captura el
    `job_status.log` del job, que es donde se busca de todos modos.
    """
    predictor, reporte, df_test, X_test, seasons = await asyncio.to_thread(
        _entrenar_y_evaluar, league_code
    )

    cursor = await db.execute(
        """
        INSERT INTO model_runs
            (league_code, created_at, status, model_type, seasons_used,
             accuracy, baseline_home, baseline_majority, baseline_bookmaker,
             logloss, brier, n_train, n_test, test_period)
        VALUES (?, ?, 'completed', 'xgboost', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            league_code, datetime.now().isoformat(), ",".join(seasons),
            reporte.accuracy, reporte.baseline_home, reporte.baseline_majority,
            reporte.baseline_bookmaker, reporte.logloss, reporte.brier,
            reporte.n_train, reporte.n_test, reporte.test_period,
        ),
    )
    await db.commit()
    run_id = cursor.lastrowid

    await _guardar_predicciones(db, run_id, league_code, df_test, X_test, predictor)
    await db.commit()

    return run_id, reporte.accuracy
