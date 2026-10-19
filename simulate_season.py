"""
🏆 Simulador Monte Carlo para La Liga 2025-2026

Usa el modelo ML entrenado para simular miles de temporadas
y calcular la probabilidad real de cada equipo de ser campeón.

Uso:
    python simulate_season.py                    # 1000 simulaciones (default)
    python simulate_season.py --simulations 5000 # 5000 simulaciones
    python simulate_season.py --retrain           # Re-entrenar modelo antes de simular
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.predictor import LaLigaPredictor
from scripts.load_data import DataLoader
from scripts.preprocess import FeatureEngineer
from scripts.scraper_laliga import LigaScraper
from simulator.monte_carlo import MonteCarloSimulator
from sklearn.model_selection import train_test_split


def load_or_train_model(force_retrain=False):
    """Carga el modelo entrenado o lo re-entrena si no existe."""
    predictor = LaLigaPredictor()
    model_path = 'models/laliga_predictor.pkl'
    
    if not force_retrain and os.path.exists(model_path):
        print("📦 Cargando modelo entrenado...")
        predictor.load_model(model_path)
        
        # Verificar que tiene scaler ajustado
        if predictor._is_fitted:
            return predictor
        else:
            print("⚠️  Modelo sin scaler ajustado, re-entrenando...")
    
    # Entrenar modelo
    print("🔧 Entrenando modelo ML...")
    loader = DataLoader()
    loader.load_csv_files()
    loader.clean_data()
    
    engineer = FeatureEngineer(loader.df)
    X, y, _ = engineer.prepare_features()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    predictor.train_xgboost(X_train, y_train, X_test, y_test)
    predictor.evaluate(X_test, y_test)
    predictor.save_model(model_path)
    
    return predictor


def get_historical_data():
    """Carga los datos históricos de partidos."""
    loader = DataLoader()
    loader.load_csv_files()
    loader.clean_data()
    return loader.df


def get_current_standings():
    """Obtiene la clasificación actual usando el scraper."""
    scraper = LigaScraper()
    standings = scraper.fetch_standings()
    gameweek = scraper.get_current_gameweek()
    return standings, gameweek


def print_header():
    """Imprime encabezado del simulador."""
    print("\n" + "═" * 90)
    print("🎲 SIMULADOR MONTE CARLO — LA LIGA 2025-2026")
    print("═" * 90)
    print(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("═" * 90)


def print_championship_results(results, gameweek):
    """Muestra los resultados de la simulación de forma visual."""
    
    # Ordenar por probabilidad de campeón
    champion_sorted = sorted(
        results['champion_probs'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    print("\n" + "═" * 90)
    print("🏆 PROBABILIDAD DE SER CAMPEÓN")
    print("═" * 90)
    
    # Barra visual de probabilidad
    max_bar_width = 40
    
    for rank, (team, prob) in enumerate(champion_sorted, 1):
        pct = prob * 100
        bar_width = int(prob * max_bar_width)
        bar = "█" * bar_width + "░" * (max_bar_width - bar_width)
        
        current_pts = results['current_points'].get(team, 0)
        avg_pts = results['avg_points'].get(team, 0)
        pts_min, pts_max = results['points_range'].get(team, (0, 0))
        
        if pct >= 1.0:
            # Mostrar equipos con >= 1% de probabilidad
            emoji = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
            print(f"\n{emoji} {rank:>2}. {team:<22} {pct:>5.1f}%  {bar}")
            print(f"      Pts actual: {current_pts:>3} | Pts promedio final: {avg_pts:>5.1f} | Rango: [{pts_min}-{pts_max}]")
        elif pct > 0:
            print(f"   {rank:>2}. {team:<22} {pct:>5.1f}%  {'░' * max_bar_width}")
    
    # Equipos con 0% probabilidad
    zero_teams = [t for t, p in champion_sorted if p == 0]
    if zero_teams:
        print(f"\n   ── Equipos sin opciones de título ({len(zero_teams)}): ", end="")
        print(", ".join(zero_teams[:5]))
        if len(zero_teams) > 5:
            print(f"      y {len(zero_teams) - 5} más")


def print_top4_results(results):
    """Muestra probabilidades de clasificarse para Champions League."""
    
    top4_sorted = sorted(
        results['top4_probs'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\n" + "─" * 90)
    print("⭐ PROBABILIDAD TOP 4 (Champions League)")
    print("─" * 90)
    
    max_bar_width = 30
    
    for rank, (team, prob) in enumerate(top4_sorted[:10], 1):
        pct = prob * 100
        bar_width = int(prob * max_bar_width)
        bar = "▓" * bar_width + "░" * (max_bar_width - bar_width)
        
        print(f"   {rank:>2}. {team:<22} {pct:>5.1f}%  {bar}")


def print_relegation_results(results):
    """Muestra probabilidades de descenso."""
    
    relegation_sorted = sorted(
        results['relegation_probs'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Solo mostrar equipos con > 0% probabilidad de descenso
    at_risk = [(t, p) for t, p in relegation_sorted if p > 0]
    
    if not at_risk:
        return
    
    print("\n" + "─" * 90)
    print("⚠️  PROBABILIDAD DE DESCENSO (últimos 3)")
    print("─" * 90)
    
    max_bar_width = 30
    
    for rank, (team, prob) in enumerate(at_risk[:10], 1):
        pct = prob * 100
        bar_width = int(prob * max_bar_width)
        bar = "▒" * bar_width + "░" * (max_bar_width - bar_width)
        
        print(f"   {rank:>2}. {team:<22} {pct:>5.1f}%  {bar}")


def print_projected_table(results):
    """Muestra la clasificación proyectada promedio."""
    
    table_sorted = sorted(
        results['avg_points'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\n" + "─" * 90)
    print("📊 CLASIFICACIÓN PROYECTADA (promedio de simulaciones)")
    print("─" * 90)
    print(f"{'POS':<5}{'EQUIPO':<25}{'PTS ACT':<10}{'PTS PROM':<10}{'PTS MED':<10}{'RANGO':<15}")
    print("─" * 75)
    
    for pos, (team, avg_pts) in enumerate(table_sorted, 1):
        current = results['current_points'].get(team, 0)
        median = results['median_points'].get(team, 0)
        pts_min, pts_max = results['points_range'].get(team, (0, 0))
        
        zone = ""
        if pos <= 4:
            zone = "🟢"
        elif pos >= 18:
            zone = "🔴"
        else:
            zone = "  "
        
        print(f"{zone}{pos:<4}{team:<25}{current:<10}{avg_pts:<10.1f}{median:<10.1f}[{pts_min}-{pts_max}]")


def print_summary(results, gameweek):
    """Muestra resumen final."""
    
    champion_sorted = sorted(
        results['champion_probs'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    top_team, top_prob = champion_sorted[0]
    
    print("\n" + "═" * 90)
    print("📋 RESUMEN DE LA SIMULACIÓN")
    print("═" * 90)
    print(f"   🎲 Simulaciones realizadas: {results['n_simulations']:,}")
    print(f"   ⚽ Partidos simulados por temporada: {results['n_remaining_fixtures']}")
    print(f"   📅 Jornada actual: {gameweek}/38")
    print(f"   🏟️  Partidos restantes totales: {results['n_remaining_fixtures']}")
    
    print(f"\n   🏆 FAVORITO: {top_team.upper()} ({top_prob*100:.1f}% probabilidad)")
    
    if len(champion_sorted) > 1:
        second_team, second_prob = champion_sorted[1]
        print(f"   🥈 Segundo: {second_team} ({second_prob*100:.1f}%)")
    
    if len(champion_sorted) > 2:
        third_team, third_prob = champion_sorted[2]
        print(f"   🥉 Tercero: {third_team} ({third_prob*100:.1f}%)")
    
    # Nivel de certeza
    if top_prob >= 0.7:
        certainty = "MUY ALTA ⭐⭐⭐⭐⭐"
    elif top_prob >= 0.5:
        certainty = "ALTA ⭐⭐⭐⭐"
    elif top_prob >= 0.3:
        certainty = "MODERADA ⭐⭐⭐"
    elif top_prob >= 0.15:
        certainty = "BAJA ⭐⭐"
    else:
        certainty = "MUY ABIERTA ⭐"
    
    print(f"\n   📊 Certeza: {certainty}")
    print("═" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Simulador Monte Carlo para La Liga')
    parser.add_argument('--simulations', '-n', type=int, default=1000,
                       help='Número de simulaciones (default: 1000)')
    parser.add_argument('--retrain', action='store_true',
                       help='Re-entrenar el modelo antes de simular')
    parser.add_argument('--seed', type=int, default=42,
                       help='Semilla aleatoria (default: 42)')
    args = parser.parse_args()
    
    print_header()
    
    # 1. Cargar/entrenar modelo
    print("\n[1/4] PREPARANDO MODELO ML...")
    predictor = load_or_train_model(force_retrain=args.retrain)
    
    # 2. Cargar datos históricos
    print("\n[2/4] CARGANDO DATOS HISTÓRICOS...")
    historical_df = get_historical_data()
    print(f"   ✓ {len(historical_df)} partidos históricos cargados")
    
    # 3. Obtener clasificación actual
    print("\n[3/4] OBTENIENDO CLASIFICACIÓN ACTUAL...")
    standings, gameweek = get_current_standings()
    
    if standings is None or len(standings) == 0:
        print("❌ No se pudo obtener la clasificación actual")
        return False
    
    print(f"   ✓ {len(standings)} equipos | Jornada {gameweek}/38")
    
    # 4. Ejecutar simulación Monte Carlo
    print(f"\n[4/4] SIMULACIÓN MONTE CARLO ({args.simulations:,} simulaciones)...")
    
    simulator = MonteCarloSimulator(
        predictor=predictor,
        historical_df=historical_df,
        n_simulations=args.simulations,
        random_seed=args.seed
    )
    
    results = simulator.simulate(standings, verbose=True)
    
    # Mostrar resultados
    print_championship_results(results, gameweek)
    print_top4_results(results)
    print_relegation_results(results)
    print_projected_table(results)
    print_summary(results, gameweek)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
