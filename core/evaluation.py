"""Evaluación honesta: walk-forward por temporadas y baselines de referencia.

Una accuracy sin baseline no dice nada — 0,52 puede ser un buen modelo o un
modelo inútil según lo que "siempre gana el local" o las casas de apuestas ya
consiguen sin aprender nada. Este módulo calcula los cortes y las referencias
que se muestran junto a las métricas en la UI.

La validación es walk-forward: cada temporada de test se predice con un modelo
entrenado SOLO con las anteriores, y las métricas agregan varias temporadas.
Evaluar solo la temporada actual daba un número muy ruidoso (a principio de
temporada, ~80 partidos).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Codificación del target: ver models/predictor.py._target_map
HOME, DRAW, AWAY = 1, 0, -1


TEMPORADAS_DE_TEST = 4


def walk_forward_splits(df: pd.DataFrame, seasons: list[str], n_test: int = TEMPORADAS_DE_TEST,
                        excluded=frozenset()):
    """Cortes walk-forward: las últimas `n_test` temporadas con partidos, cada
    una con todas las anteriores como entrenamiento.

    Devuelve una lista de (temporada, máscara de train, máscara de test). Las
    temporadas de `excluded` no entrenan ni se evalúan (sus partidos sí han
    alimentado el estado al construir las features).
    """
    con_partidos = [s for s in seasons if s not in excluded and (df["Season"] == s).any()]
    if len(con_partidos) < 2:
        raise ValueError(
            f"Hacen falta al menos 2 temporadas con partidos para evaluar; hay {len(con_partidos)}."
        )

    cortes = []
    for test in con_partidos[1:][-n_test:]:
        anteriores = con_partidos[: con_partidos.index(test)]
        cortes.append((test, df["Season"].isin(anteriores).to_numpy(), (df["Season"] == test).to_numpy()))
    return cortes


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


def bookmaker_probabilities(df: pd.DataFrame) -> pd.DataFrame | None:
    """Probabilidades implícitas de Bet365 sin el margen de la casa, en el
    orden (visitante, empate, local) del target codificado. Filas sin cuotas
    quedan como NaN; None si el CSV no trae cuotas."""
    cols = ["Odds_Away", "Odds_Draw", "Odds_Home"]
    if not set(cols) <= set(df.columns):
        return None
    inversas = 1 / df[cols].apply(pd.to_numeric, errors="coerce")
    return inversas.div(inversas.sum(axis=1), axis=0)


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
    """Todo lo que hace falta para interpretar las métricas, en un solo sitio."""

    accuracy: float
    baseline_home: float
    baseline_majority: float
    baseline_bookmaker: float | None
    logloss: float
    brier: float
    logloss_bookmaker: float | None
    logloss_frequencies: float
    n_train: int
    n_test: int
    test_period: str


@dataclass
class Fold:
    """Un corte walk-forward ya predicho."""

    season: str
    y_train: pd.Series  # {-1, 0, 1}
    y_test: pd.Series  # {-1, 0, 1}
    proba: np.ndarray  # columnas en el orden codificado: visitante, empate, local
    df_test: pd.DataFrame


def _codificar(y: pd.Series) -> np.ndarray:
    return y.map({AWAY: 0, DRAW: 1, HOME: 2}).to_numpy()


def evaluate_walk_forward(folds: list[Fold], n_train: int, label=str) -> EvaluationReport:
    """Agrega los cortes en un único informe, sobre todos sus partidos.

    Las referencias se calculan corte a corte con lo que se sabía entonces:
    la clase mayoritaria y las frecuencias salen del TRAIN de cada corte.
    `n_train` es el tamaño del entrenamiento del modelo final (el que se
    guarda), no el de los cortes.
    """
    y_test = pd.concat([f.y_test for f in folds], ignore_index=True)
    y_cod = _codificar(y_test)
    proba = np.vstack([f.proba for f in folds])
    df_test = pd.concat([f.df_test for f in folds], ignore_index=True)

    mayoritaria = np.concatenate([np.full(len(f.y_test), f.y_train.mode().iloc[0]) for f in folds])
    frecuencias = np.vstack([
        np.tile(np.bincount(_codificar(f.y_train), minlength=3) / len(f.y_train), (len(f.y_test), 1))
        for f in folds
    ])

    mercado = bookmaker_probabilities(df_test)
    logloss_mercado = None
    if mercado is not None:
        con_cuotas = mercado.notna().all(axis=1).to_numpy()
        if con_cuotas.any():
            logloss_mercado = float(log_loss(y_cod[con_cuotas], mercado.to_numpy()[con_cuotas], labels=[0, 1, 2]))

    return EvaluationReport(
        accuracy=float((proba.argmax(axis=1) == y_cod).mean()),
        baseline_home=baseline_always_home(y_test),
        baseline_majority=float((y_test.to_numpy() == mayoritaria).mean()),
        baseline_bookmaker=bookmaker_baseline(df_test),
        logloss=float(log_loss(y_cod, proba, labels=[0, 1, 2])),
        brier=multiclass_brier_score(y_cod, proba, [0, 1, 2]),
        logloss_bookmaker=logloss_mercado,
        logloss_frequencies=float(log_loss(y_cod, frecuencias, labels=[0, 1, 2])),
        n_train=n_train,
        n_test=len(y_test),
        test_period=" · ".join(label(f.season) for f in folds),
    )
