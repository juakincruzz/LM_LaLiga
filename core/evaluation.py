"""Evaluación honesta: split por temporadas y baselines de referencia.

Una accuracy sin baseline no dice nada — 0,52 puede ser un buen modelo o un
modelo inútil según lo que "siempre gana el local" o las casas de apuestas ya
consiguen sin aprender nada. Este módulo calcula el split y las tres
referencias que se muestran junto a la accuracy en la UI.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Codificación del target: ver models/predictor.py._target_map
HOME, DRAW, AWAY = 1, 0, -1


def season_train_test_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, seasons: list[str]):
    """Separa por temporadas: todo menos la última para entrenar, la última
    (la actual) para test. Se toca una sola vez y es el número que ve la UI.

    No se reserva una temporada de validación aparte porque hoy no hay
    selección de modelo ni búsqueda de hiperparámetros que la consuma (un solo
    modelo, XGBoost, con hiperparámetros fijos) — reservarla sería un tercer
    trozo de datos sin ningún consumidor, y con historia limitada cada
    temporada que se aparta duele. Si en el futuro se compara más de un
    modelo, ahí es donde se separaría del final de `train_seasons`.
    """
    if len(seasons) < 2:
        raise ValueError(
            f"Hacen falta al menos 2 temporadas para separar train/test; hay {len(seasons)}."
        )

    test_season = seasons[-1]
    train_seasons = seasons[:-1]

    train_mask = df["Season"].isin(train_seasons)
    test_mask = df["Season"] == test_season

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
        df[train_mask], df[test_mask],
    )


def baseline_always_home(y_test: pd.Series) -> float:
    """Accuracy de predecir siempre "gana el local"."""
    return float((y_test == HOME).mean())


def baseline_majority_class(y_train: pd.Series, y_test: pd.Series) -> float:
    """Accuracy de predecir siempre la clase mayoritaria... del TRAIN.

    Usar la moda del propio test (como hacía scripts/model_evaluation.py) es un
    baseline oráculo: mide el test contra sí mismo y siempre queda
    artificialmente alto.
    """
    clase_mayoritaria = y_train.mode().iloc[0]
    return float((y_test == clase_mayoritaria).mean())


def bookmaker_baseline(df_test: pd.DataFrame) -> float | None:
    """Accuracy de apostar siempre por la cuota más baja (Bet365).

    None si el CSV no traía columnas de cuotas — no todas las ligas/temporadas
    las tienen.
    """
    cols = {"Odds_Home", "Odds_Draw", "Odds_Away"}
    if not cols <= set(df_test.columns):
        return None

    odds = df_test[["Odds_Home", "Odds_Draw", "Odds_Away"]].dropna()
    if odds.empty:
        return None

    # Cuota más baja = probabilidad implícita más alta = lo que el mercado cree más probable
    prediccion = odds.idxmin(axis=1).map({"Odds_Home": HOME, "Odds_Draw": DRAW, "Odds_Away": AWAY})
    real = df_test.loc[odds.index, "Result"].map({"H": HOME, "D": DRAW, "A": AWAY})
    return float((prediccion == real).mean())


def multiclass_brier_score(y_true: pd.Series, proba: np.ndarray, classes: list[int]) -> float:
    """Brier multiclase: error cuadrático medio entre la probabilidad predicha
    y el resultado real codificado one-hot. sklearn no lo da para multiclase."""
    one_hot = np.zeros((len(y_true), len(classes)))
    indice = {clase: i for i, clase in enumerate(classes)}
    for row, valor in enumerate(y_true):
        one_hot[row, indice[valor]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


@dataclass
class EvaluationReport:
    """Todo lo que hace falta para interpretar una accuracy, en un solo sitio."""

    accuracy: float
    baseline_home: float
    baseline_majority: float
    baseline_bookmaker: float | None
    logloss: float
    brier: float
    n_train: int
    n_test: int
    test_period: str


def evaluate_predictions(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_encoded: np.ndarray,
    proba: np.ndarray,
    classes: list[int],
    df_test: pd.DataFrame,
    encode,
) -> EvaluationReport:
    """`encode` mapea {-1,0,1} -> {0,1,2}; y_pred_encoded y proba ya vienen del modelo."""
    y_test_encoded = encode(y_test)
    accuracy = float((y_test_encoded.to_numpy() == y_pred_encoded).mean())

    return EvaluationReport(
        accuracy=accuracy,
        baseline_home=baseline_always_home(y_test),
        baseline_majority=baseline_majority_class(y_train, y_test),
        baseline_bookmaker=bookmaker_baseline(df_test),
        logloss=float(log_loss(y_test_encoded, proba, labels=classes)),
        brier=multiclass_brier_score(y_test_encoded.to_numpy(), proba, classes),
        n_train=len(y_train),
        n_test=len(y_test),
        test_period=f"{df_test['Date'].min().date()} → {df_test['Date'].max().date()}",
    )
