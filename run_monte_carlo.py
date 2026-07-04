"""
Script de Monte Carlo para LM_LaLiga.

Ejecuta miles de simulaciones estocásticas de los partidos restantes para
estimar probabilidades de:
- Ganar La Liga.
- Clasificar a Champions League (top 4).
- Descender (últimos 3).

Uso:
    python run_monte_carlo.py [n_simulaciones]

Ejemplo:
    python run_monte_carlo.py 1000
"""

import sys
import os
import argparse

# Agregar la raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.data_manager import ensure_raw_data
from scripts.load_data import DataLoader
from scripts.preprocess import FeatureEngineer
from scripts.model_evaluation import temporal_split
from scripts.season_simulator import (
    SeasonSimulator,
    print_monte_carlo_results,
    plot_monte_carlo_probabilities
)
from models.predictor import LaLigaPredictor


def main(n_sims=1000, start_date=None):
    print("=" * 90)
    print("🎲 MONTE CARLO - LM_LaLiga")
    print("=" * 90)

    # 1. Asegurar datos
    print("\n[1/5] VERIFICANDO DATOS...")
    if not ensure_raw_data():
        print("❌ No se pudieron obtener datos. Abortando.")
        return False

    # 2. Cargar y preparar datos
    print("\n[2/5] CARGANDO Y PREPARANDO DATOS...")
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

    # 3. Dividir datos temporalmente
    print("\n[3/5] DIVIDIENDO DATOS...")
    try:
        X_train, X_test, y_train, y_test, _, _ = temporal_split(
            X, y, df_features, test_size=0.2
        )
    except Exception as e:
        print(f"❌ Error dividiendo datos: {e}")
        return False

    # 4. Entrenar y comparar modelos
    print("\n[4/5] ENTRENANDO Y COMPARANDO MODELOS...")
    try:
        predictor = LaLigaPredictor()
        predictor.compare_models(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"❌ Error entrenando modelos: {e}")
        return False

    # 5. Monte Carlo
    print("\n[5/5] EJECUTANDO SIMULACIONES MONTE CARLO...")
    try:
        simulator = SeasonSimulator()
        results = simulator.monte_carlo_simulation(
            loader.df, predictor, n_sims=n_sims, start_date=start_date, verbose=True
        )

        print_monte_carlo_results(results)
        plot_monte_carlo_probabilities(results)

        # Guardar resultados en CSV
        os.makedirs('data/processed', exist_ok=True)
        champ_df = pd.DataFrame(
            sorted(results['champion_probabilities'].items(), key=lambda x: x[1], reverse=True),
            columns=['Team', 'Champion_Probability']
        )
        champ_df.to_csv('data/processed/monte_carlo_champion_probabilities.csv', index=False)
        print("\n✓ Probabilidades guardadas en data/processed/monte_carlo_champion_probabilities.csv")

    except Exception as e:
        print(f"❌ Error en Monte Carlo: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 90)
    print("✓ Monte Carlo completado")
    print("=" * 90)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo para LM_LaLiga")
    parser.add_argument(
        "--sims", type=int, default=1000,
        help="Número de simulaciones Monte Carlo (por defecto 1000)"
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Fecha desde la cual simular (formato YYYY-MM-DD). Si no se indica, usa los partidos no disputados."
    )

    args = parser.parse_args()

    start_date = None
    if args.start_date:
        import pandas as pd
        start_date = pd.Timestamp(args.start_date)

    try:
        success = main(n_sims=args.sims, start_date=start_date)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
