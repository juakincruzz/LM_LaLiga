"""
Pipeline unificado para LM_LaLiga.

Ejecuta todo el flujo en un solo comando:
1. Verifica o descarga datos.
2. Carga y limpia datos.
3. Genera características.
4. Entrena y evalúa el modelo.
5. Calcula la clasificación actual.
6. Proyecta el campeón de La Liga.

Uso:
    python run_full_pipeline.py
"""

import sys
import os
from datetime import datetime

import pandas as pd

# Agregar la raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.data_manager import ensure_raw_data
from scripts.load_data import DataLoader
from scripts.preprocess import FeatureEngineer
from scripts.model_evaluation import temporal_split
from scripts.season_simulator import SeasonSimulator, print_simulated_standings, print_monte_carlo_results, plot_monte_carlo_probabilities
from models.predictor import LaLigaPredictor
from update_and_predict import (
    calculate_current_standings,
    project_season,
    print_standings_compact
)


def print_header():
    """Imprime el encabezado del pipeline unificado."""
    print("\n" + "=" * 90)
    print("🏆 LM_LaLiga - PIPELINE COMPLETO (Datos + Modelo + Predicción)")
    print("=" * 90)
    print(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 90)


def print_champion_prediction(standings, projected):
    """Muestra la predicción final del campeón."""
    champion = projected.index[0]
    champion_current = int(standings.loc[champion]['points'])
    champion_projected = int(projected.iloc[0]['projected_points'])
    second_projected = int(projected.iloc[1]['projected_points'])
    point_gap = champion_projected - second_projected
    jornada = int(standings['games'].max())

    print(f"\n{'='*90}")
    print("🏅 PREDICCIÓN FINAL DE CAMPEÓN")
    print(f"{'='*90}")
    print(f"\n🥇 CAMPEÓN PREDICHO: {champion.upper()}")
    print(f"   • Puntos actuales: {champion_current} ({jornada}/38)")
    print(f"   • Puntos proyectados: {champion_projected}")
    print(f"   • Ventaja: +{point_gap} puntos sobre 2º")

    if point_gap >= 5:
        confidence = "MUY ALTA ⭐⭐⭐⭐⭐"
    elif point_gap >= 3:
        confidence = "ALTA ⭐⭐⭐⭐"
    else:
        confidence = "MEDIA ⭐⭐⭐"

    print(f"   • Confianza: {confidence}")

    print(f"\n🏆 PODIO FINAL:")
    for pos, (team, row) in enumerate(projected.head(3).iterrows(), 1):
        print(f"   {pos}. {team:<25} {int(row['projected_points']):>3} pts")

    print(f"\n⚠️  ZONA DE RIESGO (últimas 3):")
    for team, row in projected.tail(3).iterrows():
        print(f"   {team:<25} {int(row['projected_points']):>3} pts")

    print(f"\n{'='*90}\n")


def save_projections(projected, output_path='data/processed/projected_standings.csv'):
    """Guarda la proyección final en un CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    projected.to_csv(output_path)
    print(f"✓ Proyección guardada en {output_path}")


def main():
    print_header()

    # ------------------------------------------------------------------
    # 1. ASEGURAR DATOS DISPONIBLES
    # ------------------------------------------------------------------
    print("\n[1/7] VERIFICANDO Y ACTUALIZANDO DATOS...")
    if not ensure_raw_data():
        print("❌ No se pudieron obtener datos. Abortando.")
        return False

    # ------------------------------------------------------------------
    # 2. CARGAR Y LIMPIAR DATOS
    # ------------------------------------------------------------------
    print("\n[2/7] CARGANDO Y LIMPIANDO DATOS...")
    try:
        loader = DataLoader()
        loader.load_csv_files()
        loader.clean_data()
        loader.save_processed_data()
        loader.get_summary()
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return False

    # ------------------------------------------------------------------
    # 3. PREPARAR CARACTERÍSTICAS
    # ------------------------------------------------------------------
    print("\n[3/7] PREPARANDO CARACTERÍSTICAS...")
    try:
        engineer = FeatureEngineer(loader.df)
        X, y, df_features = engineer.prepare_features()
        print(f"✓ Features preparados: {X.shape[1]} características, {X.shape[0]} muestras")
    except Exception as e:
        print(f"❌ Error preparando características: {e}")
        return False

    # ------------------------------------------------------------------
    # 4. DIVIDIR DATOS (temporalmente)
    # ------------------------------------------------------------------
    print("\n[4/7] DIVIDIENDO DATOS...")
    try:
        X_train, X_test, y_train, y_test, _, _ = temporal_split(
            X, y, df_features, test_size=0.2
        )
        print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    except Exception as e:
        print(f"❌ Error dividiendo datos: {e}")
        return False

    # ------------------------------------------------------------------
    # 5. ENTRENAR Y COMPARAR MODELOS
    # ------------------------------------------------------------------
    print("\n[5/7] ENTRENANDO Y COMPARANDO MODELOS...")
    try:
        predictor = LaLigaPredictor()
        predictor.compare_models(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"❌ Error entrenando modelos: {e}")
        return False

    # ------------------------------------------------------------------
    # 6. EVALUAR MEJOR MODELO
    # ------------------------------------------------------------------
    print("\n[6/7] EVALUANDO MEJOR MODELO...")
    try:
        predictor.evaluate(X_test, y_test)
        predictor.feature_importance(X.columns.tolist(), top_n=15)
        predictor.save_model()
    except Exception as e:
        print(f"❌ Error evaluando modelo: {e}")
        return False

    # ------------------------------------------------------------------
    # 7. CALCULAR CLASIFICACIÓN Y PROYECTAR CAMPEÓN
    # ------------------------------------------------------------------
    print("\n[7/7] CALCULANDO CLASIFICACIÓN Y PROYECTANDO CAMPEÓN...")
    try:
        # Clasificación actual
        standings = calculate_current_standings(loader.df)
        print_standings_compact(standings, "📊 CLASIFICACIÓN ACTUAL")

        # Proyección basada en el modelo ML (simulación de partidos restantes)
        simulator = SeasonSimulator()
        simulated_standings, simulated_matches = simulator.simulate(loader.df, predictor)
        print_simulated_standings(simulated_standings, "🎮 CLASIFICACIÓN SIMULADA CON MODELO ML")

        # Proyección lineal clásica (comparación)
        projected = project_season(standings)
        print_standings_compact(projected, "📈 PROYECCIÓN LINEAL (Puntos por partido)")

        # Predicción final del campeón usando la simulación del modelo
        champion = simulated_standings.iloc[0]
        champion_name = champion['Team']
        champion_points = int(champion['Points'])
        second_points = int(simulated_standings.iloc[1]['Points'])
        point_gap = champion_points - second_points
        jornada = int(standings['games'].max())

        print(f"\n{'='*90}")
        print("🏅 PREDICCIÓN FINAL DE CAMPEÓN (SIMULACIÓN ML)")
        print(f"{'='*90}")
        print(f"\n🥇 CAMPEÓN PREDICHO: {champion_name.upper()}")
        print(f"   • Modelo usado: {predictor.model_name}")
        print(f"   • Puntos actuales: {int(standings.loc[champion_name]['points'])} ({jornada}/38)")
        print(f"   • Puntos simulados: {champion_points}")
        print(f"   • Ventaja: +{point_gap} puntos sobre 2º")

        if point_gap >= 5:
            confidence = "MUY ALTA ⭐⭐⭐⭐⭐"
        elif point_gap >= 3:
            confidence = "ALTA ⭐⭐⭐⭐"
        else:
            confidence = "MEDIA ⭐⭐⭐"
        print(f"   • Confianza: {confidence}")

        print(f"\n🏆 PODIO FINAL SIMULADO:")
        for pos, row in simulated_standings.head(3).iterrows():
            print(f"   {int(row['Position'])}. {row['Team']:<25} {int(row['Points']):>3} pts")

        print(f"\n⚠️  ZONA DE RIESGO (últimas 3):")
        for _, row in simulated_standings.tail(3).iterrows():
            print(f"   {int(row['Position'])}. {row['Team']:<25} {int(row['Points']):>3} pts")

        print(f"\n{'='*90}\n")

        # Guardar proyecciones
        save_projections(projected)
        os.makedirs('data/processed', exist_ok=True)
        simulated_standings.to_csv('data/processed/simulated_standings.csv', index=False)
        simulated_matches.to_csv('data/processed/simulated_matches.csv', index=False)
        print("✓ Simulación guardada en data/processed/simulated_standings.csv")
        print("✓ Partidos simulados guardados en data/processed/simulated_matches.csv")

        # Monte Carlo: estimar probabilidades de campeón, top 4 y descenso
        print("\n🎲 EJECUTANDO MONTE CARLO (500 simulaciones)...")
        mc_simulator = SeasonSimulator()
        mc_results = mc_simulator.monte_carlo_simulation(
            loader.df, predictor, n_sims=500, verbose=False
        )
        print_monte_carlo_results(mc_results, top_n=10)
        plot_monte_carlo_probabilities(mc_results)

        # Guardar probabilidades de campeón
        champ_df = pd.DataFrame(
            sorted(mc_results['champion_probabilities'].items(), key=lambda x: x[1], reverse=True),
            columns=['Team', 'Champion_Probability']
        )
        champ_df.to_csv('data/processed/monte_carlo_champion_probabilities.csv', index=False)
        print("\n✓ Probabilidades guardadas en data/processed/monte_carlo_champion_probabilities.csv")

    except Exception as e:
        print(f"❌ Error proyectando campeón: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 90)
    print("✓ ¡Pipeline completo finalizado exitosamente!")
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
