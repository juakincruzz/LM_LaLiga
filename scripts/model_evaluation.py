"""
Módulo de evaluación temporal para modelos de predicción deportiva.

En datos deportivos no es válido usar train_test_split aleatorio porque
los partidos futuros no pueden usarse para predecir partidos pasados.
Este módulo proporciona estrategias de validación temporal:

- División temporal simple.
- Walk-forward validation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def temporal_split(X, y, df, test_size=0.2):
    """
    Divide los datos cronológicamente.

    El conjunto de entrenamiento contiene los partidos más antiguos y el de
    test los partidos más recientes. Esto refleja cómo se usaría el modelo
    en la realidad.

    Parámetros
    ----------
    X : pd.DataFrame
        Features (debe estar ordenado cronológicamente).
    y : pd.Series
        Target (debe estar ordenado cronológicamente).
    df : pd.DataFrame
        DataFrame original ordenado cronológicamente.
    test_size : float
        Proporción de datos reservada para test (por defecto 0.2).

    Retorna
    -------
    tuple
        (X_train, X_test, y_train, y_test, df_train, df_test)
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    print(f"✓ Split temporal: {len(X_train)} train ({df_train['Date'].min().date()} → {df_train['Date'].max().date()}), "
          f"{len(X_test)} test ({df_test['Date'].min().date()} → {df_test['Date'].max().date()})")

    return X_train, X_test, y_train, y_test, df_train, df_test


def walk_forward_validation(X, y, df, predictor, min_train_size=None, step=38):
    """
    Realiza walk-forward validation.

    En cada iteración entrena con todos los partidos disponibles hasta un
    punto determinado y predice los siguientes `step` partidos. Luego avanza
    la ventana y repite el proceso.

    Parámetros
    ----------
    X : pd.DataFrame
        Features ordenadas cronológicamente.
    y : pd.Series
        Target ordenado cronológicamente.
    df : pd.DataFrame
        DataFrame original ordenado cronológicamente.
    predictor : LaLigaPredictor
        Instancia del predictor a evaluar.
    min_train_size : int | None
        Tamaño mínimo del conjunto de entrenamiento en la primera iteración.
        Si es None, usa el 60% de los datos.
    step : int
        Número de partidos a predecir en cada iteración (por defecto 38,
        aproximadamente una jornada de La Liga).

    Retorna
    -------
    tuple
        (scores, predictions, actuals, test_dates)
    """
    if min_train_size is None:
        min_train_size = int(len(X) * 0.6)

    scores = []
    predictions = []
    actuals = []
    test_dates = []

    print(f"📈 Iniciando walk-forward validation (min_train={min_train_size}, step={step})...")

    for end_train in range(min_train_size, len(X) - step + 1, step):
        X_train = X.iloc[:end_train]
        y_train = y.iloc[:end_train]
        X_test = X.iloc[end_train:end_train + step]
        y_test = y.iloc[end_train:end_train + step]
        df_test = df.iloc[end_train:end_train + step]

        # Entrenar y predecir
        predictor.train_xgboost(X_train, y_train, X_test, y_test)
        y_pred = predictor.model.predict(X_test)

        # Calcular accuracy
        y_test_enc = y_test.map({-1: 0, 0: 1, 1: 2})
        acc = accuracy_score(y_test_enc, y_pred)

        scores.append(acc)
        predictions.extend(y_pred.tolist())
        actuals.extend(y_test_enc.tolist())
        test_dates.append(df_test['Date'].min())

        print(f"   Fold {len(scores)}: train hasta {df.iloc[end_train - 1]['Date'].date()} | "
              f"test {df_test['Date'].min().date()} → {df_test['Date'].max().date()} | "
              f"accuracy={acc:.4f}")

    return scores, predictions, actuals, test_dates


def print_walk_forward_summary(scores, predictions, actuals):
    """Muestra un resumen de la validación walk-forward."""
    print("\n📊 Resumen de walk-forward validation:")
    print(f"   Accuracy promedio: {np.mean(scores):.4f}")
    print(f"   Accuracy mínima:   {np.min(scores):.4f}")
    print(f"   Accuracy máxima:   {np.max(scores):.4f}")
    print(f"   Desviación std:    {np.std(scores):.4f}")

    print("\nMatriz de confusión global:")
    print(confusion_matrix(actuals, predictions))

    print("\nReporte de clasificación global:")
    print(classification_report(
        actuals, predictions,
        target_names=['Away Win', 'Draw', 'Home Win'],
        labels=[0, 1, 2]
    ))


def temporal_baseline_accuracy(y_test):
    """
    Calcula la accuracy de un baseline que siempre predice la clase mayoritaria
    en el conjunto de entrenamiento (útil para comparar con el modelo).
    """
    mode = y_test.mode()
    if len(mode) == 0:
        return 0.0
    most_common = mode[0]
    return (y_test == most_common).mean()


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.load_data import DataLoader
    from scripts.preprocess import FeatureEngineer
    from models.predictor import LaLigaPredictor

    loader = DataLoader()
    loader.load_csv_files()
    loader.clean_data()

    engineer = FeatureEngineer(loader.df)
    X, y, df_features = engineer.prepare_features()

    # División temporal simple
    X_train, X_test, y_train, y_test, _, _ = temporal_split(X, y, df_features, test_size=0.2)

    predictor = LaLigaPredictor()
    predictor.train_xgboost(X_train, y_train, X_test, y_test)
    predictor.evaluate(X_test, y_test)
