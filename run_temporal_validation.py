"""
Script de validación temporal para LM_LaLiga.

Ejecuta walk-forward validation: entrena con datos históricos y predice
ventanas futuras de partidos, avanzando progresivamente en el tiempo.

Uso:
    python run_temporal_validation.py
"""

import sys
import os

# Agregar la raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.data_manager import ensure_raw_data
from scripts.load_data import DataLoader
from scripts.preprocess import FeatureEngineer
from scripts.model_evaluation import (
    temporal_split,
    walk_forward_validation,
    print_walk_forward_summary,
    temporal_baseline_accuracy
)
from models.predictor import LaLigaPredictor


def main():
    print("=" * 90)
    print("📈 VALIDACIÓN TEMPORAL - LM_LaLiga")
    print("=" * 90)

    # 1. Asegurar datos
    print("\n[1/4] VERIFICANDO DATOS...")
    if not ensure_raw_data():
        print("❌ No se pudieron obtener datos. Abortando.")
        return False

    # 2. Cargar y preparar datos
    print("\n[2/4] CARGANDO Y PREPARANDO DATOS...")
    try:
        loader = DataLoader()
        loader.load_csv_files()
        loader.clean_data()

        engineer = FeatureEngineer(loader.df)
        X, y, df_features = engineer.prepare_features()
        print(f"✓ Features preparados: {X.shape[1]} características, {X.shape[0]} muestras")
    except Exception as e:
        print(f"❌ Error preparando datos: {e}")
        return False

    # 3. División temporal simple
    print("\n[3/4] EVALUACIÓN CON SPLIT TEMPORAL SIMPLE...")
    try:
        X_train, X_test, y_train, y_test, _, _ = temporal_split(X, y, df_features, test_size=0.2)

        predictor = LaLigaPredictor()
        predictor.train_xgboost(X_train, y_train, X_test, y_test)
        predictor.evaluate(X_test, y_test)

        baseline = temporal_baseline_accuracy(y_test)
        print(f"\n📌 Baseline (clase mayoritaria): {baseline:.4f}")
    except Exception as e:
        print(f"❌ Error en split temporal: {e}")
        return False

    # 4. Walk-forward validation
    print("\n[4/4] WALK-FORWARD VALIDATION...")
    try:
        predictor_wf = LaLigaPredictor()
        scores, predictions, actuals, test_dates = walk_forward_validation(
            X, y, df_features, predictor_wf, min_train_size=None, step=38
        )
        print_walk_forward_summary(scores, predictions, actuals)
    except Exception as e:
        print(f"❌ Error en walk-forward validation: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 90)
    print("✓ Validación temporal completada")
    print("=" * 90)
    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
