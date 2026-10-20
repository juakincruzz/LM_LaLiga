"""Entrenamiento del modelo de una liga, con métricas honestas.

Antes esta función entrenaba con todos los partidos y medía la accuracy sobre
esos mismos partidos: reportaba ~0,99 donde lo real fuera de muestra era ~0,52.

Ahora las métricas son walk-forward (core.evaluation): las últimas
temporadas se predicen cada una con un modelo entrenado solo con las
anteriores. El modelo que se guarda, en cambio, se entrena con TODO lo jugado,
temporada actual incluida: para predecir la próxima jornada no hay motivo para
esconderle los partidos de este año. Junto a las métricas se guardan las
referencias: "siempre local", la clase mayoritaria, el mercado de apuestas
(acierto y log-loss) y las frecuencias de resultados (log-loss de no saber nada).
"""
import asyncio
from datetime import datetime

import aiosqlite
import numpy as np
import pandas as pd

from app.config import EXCLUDED_SEASONS, get_current_season, get_training_seasons, season_label
from core.evaluation import Fold, evaluate_walk_forward, walk_forward_splits
from models.predictor import LeaguePredictor
from scripts.preprocess_generic import preprocess_league


async def _guardar_predicciones(
    db: aiosqlite.Connection,
    run_id: int,
    league_code: str,
    df_test: pd.DataFrame,
    proba: np.ndarray,
) -> int:
    """Persiste las predicciones walk-forward, enlazadas a `matches` (solo
    las de temporadas ingeridas en la BD encuentran su partido)."""
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

    filas = []
    for fila, (p_away, p_draw, p_home) in zip(df_test.itertuples(index=False), proba):
        match_id = match_ids.get((fila.Date.date().isoformat(), fila.HomeTeam, fila.AwayTeam))
        if match_id is None:
            continue
        predicho = max((p_home, "H"), (p_draw, "D"), (p_away, "A"))[1]
        filas.append((run_id, match_id, predicho, float(p_home), float(p_draw), float(p_away)))

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
    históricas, features, walk-forward y modelo final, con un único
    asyncio.to_thread."""
    seasons = get_training_seasons(league_code)
    X, y, df = preprocess_league(league_code, seasons)

    actual = get_current_season()
    folds, previo = [], None
    for season, train, test in walk_forward_splits(df, seasons, excluded=EXCLUDED_SEASONS):
        modelo = LeaguePredictor(league_code)
        modelo.train_xgboost(X[train], y[train])
        folds.append(Fold(season, y[train], y[test], modelo.proba_ordenada(X[test]), df[test]))
        if season == actual:
            previo = modelo

    # Modelo del historial de aciertos: solo temporadas anteriores a la actual
    # (es el corte walk-forward de la actual; si aún no tiene partidos, se
    # entrena aparte). Así el historial predice cada partido sin haberlo visto.
    if previo is None:
        anteriores = (df["Season"] != actual).to_numpy() & ~df["Season"].isin(EXCLUDED_SEASONS).to_numpy()
        if anteriores.any():
            previo = LeaguePredictor(league_code)
            previo.train_xgboost(X[anteriores], y[anteriores])
    if previo is not None:
        previo.save_model(previo.history_model_path)

    # Modelo final: todo lo jugado, salvo las temporadas excluidas
    entrena = ~df["Season"].isin(EXCLUDED_SEASONS).to_numpy()
    predictor = LeaguePredictor(league_code)
    predictor.train_xgboost(X[entrena], y[entrena])

    reporte = evaluate_walk_forward(folds, n_train=int(entrena.sum()), label=season_label)
    predictor.accuracy = reporte.accuracy
    predictor.save_model()

    df_test = pd.concat([f.df_test for f in folds], ignore_index=True)
    proba = np.vstack([f.proba for f in folds])
    usadas = [s for s in seasons if s not in EXCLUDED_SEASONS]
    return predictor, reporte, df_test, proba, usadas


async def train_league_model(db: aiosqlite.Connection, league_code: str) -> tuple[int, float]:
    """Entrena y guarda el modelo de una liga. Devuelve (run_id, accuracy honesta).

    Un solo INSERT con la fila ya completa, no INSERT+UPDATE: para cuando se
    llega aquí todo el trabajo (descarga, features, entrenamiento, evaluación)
    ya ha terminado en el hilo de `asyncio.to_thread`. De propina, si algo
    revienta antes de este punto no queda ninguna fila a medias en
    `model_runs` marcada 'running' para siempre — el error lo captura el
    `job_status.log` del job, que es donde se busca de todos modos.
    """
    predictor, reporte, df_test, proba, seasons = await asyncio.to_thread(
        _entrenar_y_evaluar, league_code
    )

    cursor = await db.execute(
        """
        INSERT INTO model_runs
            (league_code, created_at, status, model_type, seasons_used,
             accuracy, baseline_home, baseline_majority, baseline_bookmaker,
             logloss, brier, logloss_bookmaker, logloss_frequencies,
             n_train, n_test, test_period)
        VALUES (?, ?, 'completed', 'xgboost', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            league_code, datetime.now().isoformat(), ",".join(seasons),
            reporte.accuracy, reporte.baseline_home, reporte.baseline_majority,
            reporte.baseline_bookmaker, reporte.logloss, reporte.brier,
            reporte.logloss_bookmaker, reporte.logloss_frequencies,
            reporte.n_train, reporte.n_test, reporte.test_period,
        ),
    )
    await db.commit()
    run_id = cursor.lastrowid

    await _guardar_predicciones(db, run_id, league_code, df_test, proba)
    await db.commit()

    return run_id, reporte.accuracy
