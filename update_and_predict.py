"""
Script completo: Actualiza datos + Predice ganador
Ejecutar diariamente para tener predicciones al día
"""

import sys
import os
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.fetch_updated_data import FootballDataFetcher
from scripts.load_data import DataLoader
from scripts.preprocess import FeatureEngineer
from models.predictor import LaLigaPredictor
from sklearn.model_selection import train_test_split
from collections import defaultdict

def calculate_current_standings(df):
    """Calcula la clasificación actual"""
    standings = defaultdict(lambda: {'points': 0, 'games': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0})
    
    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = int(row['Home_Goals'])
        away_goals = int(row['Away_Goals'])
        result = row['Result']
        
        # Home team
        standings[home_team]['games'] += 1
        standings[home_team]['goals_for'] += home_goals
        standings[home_team]['goals_against'] += away_goals
        
        if result == 'H':
            standings[home_team]['points'] += 3
            standings[home_team]['wins'] += 1
        elif result == 'D':
            standings[home_team]['points'] += 1
            standings[home_team]['draws'] += 1
        else:
            standings[home_team]['losses'] += 1
        
        # Away team
        standings[away_team]['games'] += 1
        standings[away_team]['goals_for'] += away_goals
        standings[away_team]['goals_against'] += home_goals
        
        if result == 'A':
            standings[away_team]['points'] += 3
            standings[away_team]['wins'] += 1
        elif result == 'D':
            standings[away_team]['points'] += 1
            standings[away_team]['draws'] += 1
        else:
            standings[away_team]['losses'] += 1
    
    standings_df = pd.DataFrame.from_dict(standings, orient='index')
    standings_df = standings_df.sort_values('points', ascending=False)
    standings_df['goal_diff'] = standings_df['goals_for'] - standings_df['goals_against']
    
    return standings_df

def get_la_liga_teams():
    """Retorna los 20 equipos oficiales de La Liga"""
    return [
        'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Villarreal',
        'Real Betis', 'Sevilla', 'Athletic Club', 'Valencia',
        'Real Sociedad', 'Getafe', 'Osasuna', 'Celta Vigo',
        'Rayo Vallecano', 'Las Palmas', 'Mallorca', 'Girona',
        'Alaves', 'Elche', 'Granada', 'Oviedo'
    ]

def clean_team_names(df):
    """Limpia y estandariza los nombres de equipos"""
    replacements = {
        'Ath Madrid': 'Atletico Madrid',
        'Ath Bilbao': 'Athletic Club',
        'Espanol': 'Espanyol',
        'Real  Sociedad': 'Real Sociedad',
        'Betis': 'Real Betis',
        'Celta': 'Celta Vigo',
        'Vallecano': 'Rayo Vallecano',
        'Sociedad': 'Real Sociedad'
    }
    
    for old, new in replacements.items():
        df['HomeTeam'] = df['HomeTeam'].replace(old, new)
        df['AwayTeam'] = df['AwayTeam'].replace(old, new)
    
    return df

def filter_laliga_only(df):
    """Filtra solo partidos de La Liga"""
    la_liga_teams = get_la_liga_teams()
    mask = (df['HomeTeam'].isin(la_liga_teams)) & (df['AwayTeam'].isin(la_liga_teams))
    return df[mask].copy()

def project_season(standings_df, total_teams=20, matches_per_team=38):
    """Proyecta los puntos finales"""
    standings_df_copy = standings_df.copy()
    
    standings_df_copy['points_per_game'] = standings_df_copy['points'] / standings_df_copy['games']
    standings_df_copy['games_remaining'] = matches_per_team - standings_df_copy['games']
    standings_df_copy['projected_points'] = standings_df_copy['points'] + (standings_df_copy['points_per_game'] * standings_df_copy['games_remaining'])
    
    standings_df_copy['goals_for_per_game'] = standings_df_copy['goals_for'] / standings_df_copy['games']
    standings_df_copy['goals_against_per_game'] = standings_df_copy['goals_against'] / standings_df_copy['games']
    
    standings_df_copy['projected_goals_for'] = standings_df_copy['goals_for'] + (standings_df_copy['goals_for_per_game'] * standings_df_copy['games_remaining'])
    standings_df_copy['projected_goals_against'] = standings_df_copy['goals_against'] + (standings_df_copy['goals_against_per_game'] * standings_df_copy['games_remaining'])
    
    standings_df_copy['projected_goal_diff'] = standings_df_copy['projected_goals_for'] - standings_df_copy['projected_goals_against']
    
    return standings_df_copy.sort_values('projected_points', ascending=False)

def print_header():
    """Imprime encabezado"""
    print("\n" + "=" * 90)
    print("🏆 PREDICTOR DE LA LIGA 2025-2026 - VERSIÓN ACTUALIZADA")
    print("=" * 90)
    print(f"📅 Fecha de actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 90)

def print_standings_compact(standings_df, title):
    """Imprime clasificación de forma compacta"""
    print(f"\n{title}")
    print(f"{'POS':<5}{'EQUIPO':<25}{'PJ':<5}{'V':<5}{'E':<5}{'D':<5}{'DG':<5}{'PTS':<5}")
    print("-" * 60)
    
    for pos, (team, row) in enumerate(standings_df.iterrows(), 1):
        print(f"{pos:<5}{team:<25}{int(row['games']):<5}{int(row['wins']):<5}{int(row['draws']):<5}{int(row['losses']):<5}{int(row['goal_diff']):<5}{int(row['points']):<5}")

def main():
    print_header()
    
    # PASO 1: ACTUALIZAR DATOS
    print("\n[1/4] ACTUALIZANDO DATOS...")
    fetcher = FootballDataFetcher()
    df_new = fetcher.fetch_from_football_data_org_latest()
    
    if df_new is None:
        print("⚠️  No se pudieron descargar datos nuevos, usando datos locales")
        df_new = pd.read_csv('data/raw/SP1.csv')
    
    # PASO 2: CARGAR Y LIMPIAR
    print("\n[2/4] CARGANDO Y LIMPIANDO DATOS...")
    
    df_new = clean_team_names(df_new)
    df_new = filter_laliga_only(df_new)
    df_new = df_new.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first')
    
    print(f"✓ Partidos disponibles: {len(df_new)}")
    
        # Asegurar que existen las columnas necesarias
    print("✓ Verificando columnas necesarias...")
    
    # Si no existen las columnas de goles, intentar usarlas del CSV
    if 'Home_Goals' not in df_new.columns and 'FTHG' in df_new.columns:
        df_new['Home_Goals'] = df_new['FTHG']
    if 'Away_Goals' not in df_new.columns and 'FTAG' in df_new.columns:
        df_new['Away_Goals'] = df_new['FTAG']
    if 'Result' not in df_new.columns and 'FTR' in df_new.columns:
        df_new['Result'] = df_new['FTR']
    
    # Crear Result si aún no existe
    if 'Result' not in df_new.columns:
        if 'Home_Goals' in df_new.columns and 'Away_Goals' in df_new.columns:
            df_new['Result'] = df_new.apply(
                lambda row: 'H' if row['Home_Goals'] > row['Away_Goals'] 
                           else ('A' if row['Home_Goals'] < row['Away_Goals'] else 'D'), 
                axis=1
            )
    
    # PASO 3: CALCULAR CLASIFICACIÓN
    print("\n[3/4] CALCULANDO CLASIFICACIÓN...")
    
    standings = calculate_current_standings(df_new)
    la_liga_teams = get_la_liga_teams()
    standings = standings[standings.index.isin(la_liga_teams)]
    
    print_standings_compact(standings, "📊 CLASIFICACIÓN ACTUAL")
    
    jornada = int(standings['games'].max())
    print(f"\n⭐ LÍDER: {standings.index[0].upper()} con {int(standings.iloc[0]['points'])} puntos ({jornada}/38 partidos)")
    
    # PASO 4: PREDICCIÓN
    print("\n[4/4] GENERANDO PREDICCIÓN...")
    
    projected_standings = project_season(standings)
    
    champion = projected_standings.index[0]
    champion_projected = int(projected_standings.iloc[0]['projected_points'])
    champion_current = int(standings.loc[champion]['points'])
    second_projected = int(projected_standings.iloc[1]['projected_points'])
    point_gap = champion_projected - second_projected
    
    print_standings_compact(projected_standings, "📈 CLASIFICACIÓN PROYECTADA AL FINAL")
    
    # RESULTADO FINAL
    print(f"\n{'='*90}")
    print("🏅 PREDICCIÓN FINAL")
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
    for pos, (team, row) in enumerate(projected_standings.head(3).iterrows(), 1):
        print(f"   {pos}. {team:<25} {int(row['projected_points']):>3} pts")
    
    print(f"\n{'='*90}\n")
    
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)