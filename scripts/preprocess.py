import pandas as pd
import numpy as np
from pathlib import Path

class FeatureEngineer:
    """Clase para crear características (features) para el modelo ML"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.team_stats = {}
    
    def calculate_team_stats(self):
        """Calcula estadísticas históricas por equipo"""
        print("📊 Calculando estadísticas por equipo...")
        
        teams = set(list(self.df['HomeTeam'].unique()) + list(self.df['AwayTeam'].unique()))
        
        for team in teams:
            # Partidos en casa
            home_matches = self.df[self.df['HomeTeam'] == team]
            away_matches = self.df[self.df['AwayTeam'] == team]
            
            home_goals_favor = home_matches['Home_Goals'].sum()
            home_goals_contra = home_matches['Away_Goals'].sum()
            home_wins = len(home_matches[home_matches['Result'] == 'H'])
            home_draws = len(home_matches[home_matches['Result'] == 'D'])
            home_losses = len(home_matches[home_matches['Result'] == 'A'])
            
            away_goals_favor = away_matches['Away_Goals'].sum()
            away_goals_contra = away_matches['Home_Goals'].sum()
            away_wins = len(away_matches[away_matches['Result'] == 'A'])
            away_draws = len(away_matches[away_matches['Result'] == 'D'])
            away_losses = len(away_matches[away_matches['Result'] == 'H'])
            
            total_wins = home_wins + away_wins
            total_matches = len(home_matches) + len(away_matches)
            
            self.team_stats[team] = {
                'total_matches': total_matches,
                'total_wins': total_wins,
                'total_points': home_wins * 3 + home_draws + away_wins * 3 + away_draws,
                'home_wins': home_wins,
                'away_wins': away_wins,
                'home_goals_avg': home_goals_favor / max(len(home_matches), 1),
                'away_goals_avg': away_goals_favor / max(len(away_matches), 1),
                'defense_home': home_goals_contra / max(len(home_matches), 1),
                'defense_away': away_goals_contra / max(len(away_matches), 1),
            }
        
        return self.team_stats
    
    def add_team_features(self):
        """Agrega características de equipos al dataframe"""
        print("🔧 Agregando características de equipos...")
        
        # Estadísticas del equipo de casa
        self.df['Home_Wins'] = self.df['HomeTeam'].map(lambda x: self.team_stats.get(x, {}).get('home_wins', 0))
        self.df['Home_Goals_Avg'] = self.df['HomeTeam'].map(lambda x: self.team_stats.get(x, {}).get('home_goals_avg', 0))
        self.df['Home_Defense_Avg'] = self.df['HomeTeam'].map(lambda x: self.team_stats.get(x, {}).get('defense_home', 0))
        self.df['Home_Total_Points'] = self.df['HomeTeam'].map(lambda x: self.team_stats.get(x, {}).get('total_points', 0))
        
        # Estadísticas del equipo visitante
        self.df['Away_Wins'] = self.df['AwayTeam'].map(lambda x: self.team_stats.get(x, {}).get('away_wins', 0))
        self.df['Away_Goals_Avg'] = self.df['AwayTeam'].map(lambda x: self.team_stats.get(x, {}).get('away_goals_avg', 0))
        self.df['Away_Defense_Avg'] = self.df['AwayTeam'].map(lambda x: self.team_stats.get(x, {}).get('defense_away', 0))
        self.df['Away_Total_Points'] = self.df['AwayTeam'].map(lambda x: self.team_stats.get(x, {}).get('total_points', 0))
        
        return self.df
    
    def add_match_features(self):
        """Agrega características derivadas del partido"""
        print("🔧 Agregando características derivadas...")
        
        # Diferencia en puntos
        self.df['Points_Diff'] = self.df['Home_Total_Points'] - self.df['Away_Total_Points']
        
        # Diferencia en victorias
        self.df['Wins_Diff'] = self.df['Home_Wins'] - self.df['Away_Wins']
        
        # Diferencia en goles promedio
        self.df['Goals_Avg_Diff'] = self.df['Home_Goals_Avg'] - self.df['Away_Goals_Avg']
        
        # Diferencia defensiva
        self.df['Defense_Diff'] = self.df['Home_Defense_Avg'] - self.df['Away_Defense_Avg']
        
        # Tasa de efectividad ofensiva (goles / tiros)
        self.df['Home_Shot_Effectiveness'] = self.df.apply(
            lambda x: x['Home_Goals'] / max(x['Home_Shots'], 1), axis=1
        )
        self.df['Away_Shot_Effectiveness'] = self.df.apply(
            lambda x: x['Away_Goals'] / max(x['Away_Shots'], 1), axis=1
        )
        
        # Total de goles en el partido
        self.df['Total_Goals'] = self.df['Home_Goals'] + self.df['Away_Goals']
        
        # Goles esperados (xG simplificado)
        self.df['Home_xG'] = self.df['Home_ShotsOnTarget'] * 0.08
        self.df['Away_xG'] = self.df['Away_ShotsOnTarget'] * 0.08
        
        return self.df
    
    def encode_target(self):
        """Codifica la variable objetivo (ganador de La Liga)"""
        # Para predicción de ganador de temporada, necesitamos datos de toda la temporada
        # Por ahora, codificamos el resultado del partido
        self.df['Target'] = self.df['Result'].map({'H': 1, 'D': 0, 'A': -1})
        
        return self.df
    
    def prepare_features(self):
        """Prepara todas las características"""
        self.calculate_team_stats()
        self.add_team_features()
        self.add_match_features()
        self.encode_target()
        
        # Seleccionar features relevantes
        feature_cols = [
            'Home_Goals', 'Away_Goals', 'Home_Shots', 'Away_Shots',
            'Home_ShotsOnTarget', 'Away_ShotsOnTarget', 'Home_Fouls', 'Away_Fouls',
            'Home_Corners', 'Away_Corners', 'Home_Yellows', 'Away_Yellows',
            'Home_Wins', 'Away_Wins', 'Home_Goals_Avg', 'Away_Goals_Avg',
            'Home_Defense_Avg', 'Away_Defense_Avg', 'Home_Total_Points', 'Away_Total_Points',
            'Points_Diff', 'Wins_Diff', 'Goals_Avg_Diff', 'Defense_Diff',
            'Home_Shot_Effectiveness', 'Away_Shot_Effectiveness', 'Total_Goals',
            'Home_xG', 'Away_xG'
        ]
        
        X = self.df[feature_cols].fillna(0)
        y = self.df['Target']
        
        return X, y, self.df


if __name__ == "__main__":
    from scripts.load_data import DataLoader
    
    # Cargar datos
    loader = DataLoader()
    loader.load_csv_files()
    loader.clean_data()
    df = loader.df
    
    # Preparar features
    engineer = FeatureEngineer(df)
    X, y, df_features = engineer.prepare_features()
    
    print("\n✓ Features preparados:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"\nPrimeras features:")
    print(X.head())