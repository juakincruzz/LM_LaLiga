"""Historial de aciertos de la temporada en curso, jornada a jornada.

Cada partido jugado se predice con el modelo del historial (entrenado SOLO con
temporadas anteriores, ver ml_service) y con el estado justo antes de jugarse:
es una predicción que de verdad no conocía el resultado. Al lado, lo que daba
el mercado de apuestas, como referencia (no es una entrada del modelo).

No hace falta reentrenar para que aparezca una jornada nueva: basta con
«Actualizar datos», porque el modelo no cambia y las features se recalculan.
"""
import asyncio
import threading

import numpy as np
import pandas as pd

from app.config import get_current_season, get_training_seasons
from core.dataset import load_seasons, played_only
from core.errors import ModelNotTrainedError
from core.evaluation import bookmaker_probabilities
from core.features import FEATURE_COLS, build_training_matrix
from models.predictor import LeaguePredictor
from scripts.data_manager_generic import calendar_csv_path, league_csv_path

SIGNOS = ("2", "X", "1")  # columnas de proba_ordenada: visitante, empate, local
CODIGO = {"A": 0, "D": 1, "H": 2}

_cache: dict[tuple, dict] = {}
_cache_lock = threading.Lock()


def _clave(league_code: str, season: str) -> tuple:
    rutas = (
        LeaguePredictor(league_code).history_model_path,
        league_csv_path(league_code, season),
        calendar_csv_path(league_code, season),
    )
    return (league_code, season, *(r.stat().st_mtime_ns if r.exists() else None for r in rutas))


def _resumen(filas: pd.DataFrame) -> dict:
    """Aciertos y log-loss de un grupo de partidos. El mercado solo cuenta
    los partidos con cuotas."""
    con_cuotas = filas["mercado_ok"].notna()
    mercado = filas[con_cuotas]
    return {
        "partidos": len(filas),
        "aciertos_modelo": int(filas["modelo_ok"].sum()),
        "acierto_modelo": float(filas["modelo_ok"].mean()),
        "logloss_modelo": float(filas["ll_modelo"].mean()),
        "con_cuotas": int(con_cuotas.sum()),
        "aciertos_mercado": int(mercado["mercado_ok"].sum()) if len(mercado) else None,
        "acierto_mercado": float(mercado["mercado_ok"].mean()) if len(mercado) else None,
        "logloss_mercado": float(mercado["ll_mercado"].mean()) if len(mercado) else None,
    }


def _calcular(league_code: str, season: str) -> dict:
    predictor = LeaguePredictor(league_code)
    try:
        predictor.load_model(predictor.history_model_path)
    except ModelNotTrainedError as e:
        raise ModelNotTrainedError(
            "Todavía no hay modelo para el historial. Pulsa «Entrenar modelo»."
        ) from e

    # Mismo recorrido que el entrenamiento: las features de cada partido
    # salen del estado con TODA la historia anterior a él.
    temporadas = get_training_seasons(league_code)
    _, _, df = build_training_matrix(played_only(load_seasons(league_code, temporadas, download=False)))
    df = df[df["Season"] == season].reset_index(drop=True)
    if df.empty:
        return {"jornadas": [], "total": None}

    proba = predictor.proba_ordenada(df[FEATURE_COLS])
    real = df["Result"].map(CODIGO).to_numpy()
    filas = pd.DataFrame({
        "jornada": df["Matchday"] if "Matchday" in df.columns else pd.NA,
        "fecha": df["Date"].dt.date.astype(str),
        "local": df["HomeTeam"],
        "visitante": df["AwayTeam"],
        "marcador": df["Home_Goals"].astype(int).astype(str) + "-" + df["Away_Goals"].astype(int).astype(str),
        "real": [SIGNOS[c] for c in real],
        "modelo": [SIGNOS[c] for c in proba.argmax(axis=1)],
        "p_modelo": proba.max(axis=1),
        "modelo_ok": proba.argmax(axis=1) == real,
        "ll_modelo": -np.log(np.clip(proba[np.arange(len(real)), real], 1e-15, 1)),
    })

    mercado = bookmaker_probabilities(df)
    if mercado is not None:
        m = mercado.to_numpy()
        hay = ~np.isnan(m).any(axis=1)
        eleccion = np.where(hay, np.nan_to_num(m).argmax(axis=1), 0)
        filas["mercado"] = [SIGNOS[c] if h else None for c, h in zip(eleccion, hay)]
        filas["mercado_ok"] = np.where(hay, (eleccion == real).astype(float), np.nan)
        p_real = np.where(hay, np.nan_to_num(m)[np.arange(len(real)), real], np.nan)
        filas["ll_mercado"] = -np.log(np.clip(p_real, 1e-15, 1))
    else:
        filas["mercado"] = None
        filas["mercado_ok"] = np.nan
        filas["ll_mercado"] = np.nan

    jornadas = []
    for numero, grupo in filas.groupby(filas["jornada"].fillna(0).astype(int), sort=True):
        jornadas.append({
            "numero": numero or None,
            **_resumen(grupo),
            "partidos_detalle": grupo.drop(columns=["jornada"]).to_dict("records"),
        })
    return {"jornadas": jornadas, "total": _resumen(filas)}


def historial(league_code: str, season: str | None = None) -> dict:
    """{"jornadas": [...], "total": {...}} de la temporada en curso, con caché
    mientras no cambien el modelo del historial ni los datos de la temporada."""
    season = season or get_current_season()
    clave = _clave(league_code, season)
    with _cache_lock:
        if clave not in _cache:
            resultado = _calcular(league_code, season)
            _cache.clear()
            _cache[clave] = resultado
        return _cache[clave]


async def historial_async(league_code: str, season: str | None = None) -> dict:
    return await asyncio.to_thread(historial, league_code, season)
