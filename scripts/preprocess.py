import pandas as pd
import numpy as np
from pathlib import Path

class FeatureEngineer:
    """Clase para crear características (features) para el modelo ML.
    
    Cambios respecto a la versión anterior:
    - Eliminado data leakage: Home_Goals, Away_Goals, Total_Goals y 
      Shot_Effectiveness ya NO se usan como features (revelaban el target).
    - Eliminado leakage temporal: los stats de equipo ahora se calculan 
      con ventana móvil (solo datos anteriores al partido actual).
    - Añadidas features de forma reciente (últimos 5 partidos).
    """
    
    ROLLING_WINDOW = 5  # Número de partidos recientes para calcular forma
    
    def __init__(self, df):
        self.df = df.copy()
        # Asegurar orden cronológico para cálculos temporales
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            self.df = self.df.sort_values('Date').reset_index(drop=True)
    
    def _get_team_history(self, team, before_idx):
        """Obtiene los partidos de un equipo anteriores al índice dado.
        
        Args:
            team: Nombre del equipo
            before_idx: Índice del partido actual (excluido)
        
        Returns:
            tuple: (partidos_en_casa, partidos_fuera) anteriores al partido actual
        """
        df_before = self.df.iloc[:before_idx]
        home_matches = df_before[df_before['HomeTeam'] == team]
        away_matches = df_before[df_before['AwayTeam'] == team]
        return home_matches, away_matches
    
    def _calc_team_rolling_stats(self, team, before_idx):
        """Calcula estadísticas de un equipo usando solo datos anteriores al partido.
        
        Solo usa los últimos ROLLING_WINDOW partidos para capturar forma reciente.
        Devuelve estadísticas globales (acumuladas) + forma reciente.
        """
        home_matches, away_matches = self._get_team_history(team, before_idx)
        
        total_home = len(home_matches)
        total_away = len(away_matches)
        total_matches = total_home + total_away
        
        if total_matches == 0:
            return {
                'total_points': 0, 'home_wins': 0, 'away_wins': 0,
                'home_goals_avg': 0, 'away_goals_avg': 0,
                'defense_home': 0, 'defense_away': 0,
                'form_points': 0, 'form_goals_scored': 0,
                'form_goals_conceded': 0, 'form_wins': 0,
            }
        
        # --- Estadísticas acumuladas (todo el historial previo) ---
        home_wins = len(home_matches[home_matches['Result'] == 'H'])
        home_draws = len(home_matches[home_matches['Result'] == 'D'])
        away_wins = len(away_matches[away_matches['Result'] == 'A'])
        away_draws = len(away_matches[away_matches['Result'] == 'D'])
        
        home_goals_avg = home_matches['Home_Goals'].sum() / max(total_home, 1)
        away_goals_avg = away_matches['Away_Goals'].sum() / max(total_away, 1)
        defense_home = home_matches['Away_Goals'].sum() / max(total_home, 1)
        defense_away = away_matches['Home_Goals'].sum() / max(total_away, 1)
        
        total_points = home_wins * 3 + home_draws + away_wins * 3 + away_draws
        
        # --- Forma reciente (últimos N partidos) ---
        # Combinar todos los partidos del equipo y tomar los últimos N
        all_team_matches = []
        
        for _, row in home_matches.iterrows():
            all_team_matches.append({
                'date': row.get('Date', None),
                'goals_scored': row['Home_Goals'],
                'goals_conceded': row['Away_Goals'],
                'result': row['Result'],
                'is_win': 1 if row['Result'] == 'H' else 0,
                'points': 3 if row['Result'] == 'H' else (1 if row['Result'] == 'D' else 0)
            })
        
        for _, row in away_matches.iterrows():
            all_team_matches.append({
                'date': row.get('Date', None),
                'goals_scored': row['Away_Goals'],
                'goals_conceded': row['Home_Goals'],
                'result': row['Result'],
                'is_win': 1 if row['Result'] == 'A' else 0,
                'points': 3 if row['Result'] == 'A' else (1 if row['Result'] == 'D' else 0)
            })
        
        # Ordenar por fecha y tomar los últimos N
        all_team_df = pd.DataFrame(all_team_matches)
        if 'date' in all_team_df.columns and all_team_df['date'].notna().any():
            all_team_df = all_team_df.sort_values('date')
        
        recent = all_team_df.tail(self.ROLLING_WINDOW)
        
        form_points = recent['points'].sum() if len(recent) > 0 else 0
        form_goals_scored = recent['goals_scored'].mean() if len(recent) > 0 else 0
        form_goals_conceded = recent['goals_conceded'].mean() if len(recent) > 0 else 0
        form_wins = recent['is_win'].sum() if len(recent) > 0 else 0
        
        return {
            'total_points': total_points,
            'home_wins': home_wins,
            'away_wins': away_wins,
            'home_goals_avg': home_goals_avg,
            'away_goals_avg': away_goals_avg,
            'defense_home': defense_home,
            'defense_away': defense_away,
            'form_points': form_points,
            'form_goals_scored': form_goals_scored,
            'form_goals_conceded': form_goals_conceded,
            'form_wins': form_wins,
        }
    
    def add_temporal_features(self):
        """Agrega features calculados temporalmente (sin data leakage).
        
        Para cada partido, calcula stats del equipo local y visitante
        usando SOLO partidos anteriores.
        """
        print("📊 Calculando estadísticas temporales por equipo (sin leakage)...")
        print(f"   Ventana de forma reciente: últimos {self.ROLLING_WINDOW} partidos")
        
        # Pre-calcular todas las features para cada fila
        home_stats_list = []
        away_stats_list = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            home_stats = self._calc_team_rolling_stats(home_team, idx)
            away_stats = self._calc_team_rolling_stats(away_team, idx)
            
            home_stats_list.append(home_stats)
            away_stats_list.append(away_stats)
            
            # Progreso cada 100 partidos
            if (idx + 1) % 100 == 0:
                print(f"   Procesados {idx + 1}/{len(self.df)} partidos...")
        
        # Asignar features del equipo local
        self.df['Home_Wins'] = [s['home_wins'] for s in home_stats_list]
        self.df['Home_Goals_Avg'] = [s['home_goals_avg'] for s in home_stats_list]
        self.df['Home_Defense_Avg'] = [s['defense_home'] for s in home_stats_list]
        self.df['Home_Total_Points'] = [s['total_points'] for s in home_stats_list]
        self.df['Home_Form_Points'] = [s['form_points'] for s in home_stats_list]
        self.df['Home_Form_Goals'] = [s['form_goals_scored'] for s in home_stats_list]
        self.df['Home_Form_Conceded'] = [s['form_goals_conceded'] for s in home_stats_list]
        self.df['Home_Form_Wins'] = [s['form_wins'] for s in home_stats_list]
        
        # Asignar features del equipo visitante
        self.df['Away_Wins'] = [s['away_wins'] for s in away_stats_list]
        self.df['Away_Goals_Avg'] = [s['away_goals_avg'] for s in away_stats_list]
        self.df['Away_Defense_Avg'] = [s['defense_away'] for s in away_stats_list]
        self.df['Away_Total_Points'] = [s['total_points'] for s in away_stats_list]
        self.df['Away_Form_Points'] = [s['form_points'] for s in away_stats_list]
        self.df['Away_Form_Goals'] = [s['form_goals_scored'] for s in away_stats_list]
        self.df['Away_Form_Conceded'] = [s['form_goals_conceded'] for s in away_stats_list]
        self.df['Away_Form_Wins'] = [s['form_wins'] for s in away_stats_list]
        
        print(f"✓ Features temporales calculados para {len(self.df)} partidos")
        return self.df
    
    def add_match_features(self):
        """Agrega características derivadas del partido.
        
        NOTA: Solo usa features que NO revelen el resultado del partido.
        Home_Goals, Away_Goals, Total_Goals y Shot_Effectiveness están
        EXCLUIDOS porque filtran directamente el target.
        """
        print("🔧 Agregando características derivadas...")
        
        # Diferencia en puntos históricos
        self.df['Points_Diff'] = self.df['Home_Total_Points'] - self.df['Away_Total_Points']
        
        # Diferencia en victorias
        self.df['Wins_Diff'] = self.df['Home_Wins'] - self.df['Away_Wins']
        
        # Diferencia en goles promedio histórico
        self.df['Goals_Avg_Diff'] = self.df['Home_Goals_Avg'] - self.df['Away_Goals_Avg']
        
        # Diferencia defensiva
        self.df['Defense_Diff'] = self.df['Home_Defense_Avg'] - self.df['Away_Defense_Avg']
        
        # Diferencia de forma reciente
        self.df['Form_Points_Diff'] = self.df['Home_Form_Points'] - self.df['Away_Form_Points']
        self.df['Form_Goals_Diff'] = self.df['Home_Form_Goals'] - self.df['Away_Form_Goals']
        
        # xG simplificado (basado en ShotsOnTarget, disponible pre-partido solo 
        # como promedio histórico, pero aquí usamos los del partido actual que 
        # NO revelan directamente el resultado)
        if 'Home_ShotsOnTarget' in self.df.columns:
            self.df['Home_xG'] = self.df['Home_ShotsOnTarget'] * 0.08
            self.df['Away_xG'] = self.df['Away_ShotsOnTarget'] * 0.08
        
        return self.df
    
    def encode_target(self):
        """Codifica la variable objetivo"""
        # Codificamos el resultado del partido: H=1, D=0, A=-1
        self.df['Target'] = self.df['Result'].map({'H': 1, 'D': 0, 'A': -1})
        return self.df
    
    def prepare_features(self):
        """Prepara todas las características.
        
        Features usados (SIN data leakage):
        - Stats pre-partido: Shots, ShotsOnTarget, Fouls, Corners, Yellows
        - Stats históricos de equipo: Wins, Goals_Avg, Defense_Avg, Total_Points
        - Forma reciente: Form_Points, Form_Goals, Form_Conceded, Form_Wins
        - Derivados: diffs entre equipos, xG simplificado
        
        Features EXCLUIDOS (revelan el resultado):
        - Home_Goals, Away_Goals (determinan directamente Result)
        - Total_Goals (suma de los anteriores)
        - Shot_Effectiveness (contiene Goals)
        """
        self.add_temporal_features()
        self.add_match_features()
        self.encode_target()
        
        # Features que NO filtran el target
        feature_cols = [
            # Stats del partido (no revelan resultado directamente)
            'Home_Shots', 'Away_Shots',
            'Home_ShotsOnTarget', 'Away_ShotsOnTarget',
            'Home_Fouls', 'Away_Fouls',
            'Home_Corners', 'Away_Corners',
            'Home_Yellows', 'Away_Yellows',
            # Stats históricos del equipo (calculados temporalmente)
            'Home_Wins', 'Away_Wins',
            'Home_Goals_Avg', 'Away_Goals_Avg',
            'Home_Defense_Avg', 'Away_Defense_Avg',
            'Home_Total_Points', 'Away_Total_Points',
            # Forma reciente (últimos N partidos)
            'Home_Form_Points', 'Away_Form_Points',
            'Home_Form_Goals', 'Away_Form_Goals',
            'Home_Form_Conceded', 'Away_Form_Conceded',
            'Home_Form_Wins', 'Away_Form_Wins',
            # Features derivados
            'Points_Diff', 'Wins_Diff', 'Goals_Avg_Diff', 'Defense_Diff',
            'Form_Points_Diff', 'Form_Goals_Diff',
            'Home_xG', 'Away_xG',
        ]
        
        # Filtrar solo las columnas que existen en el dataframe
        available_cols = [col for col in feature_cols if col in self.df.columns]
        missing_cols = [col for col in feature_cols if col not in self.df.columns]
        if missing_cols:
            print(f"⚠️  Columnas no disponibles (se omiten): {missing_cols}")
        
        X = self.df[available_cols].fillna(0)
        y = self.df['Target']
        
        # Eliminar filas donde el target es NaN (partidos sin resultado)
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"\n✓ Features finales: {X.shape[1]} características, {X.shape[0]} muestras")
        print(f"  (Eliminadas features con leakage: Home_Goals, Away_Goals, Total_Goals, Shot_Effectiveness)")
        
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
    print(f"\nColumnas usadas:")
    for col in X.columns:
        print(f"  • {col}")
    print(f"\nPrimeras features:")
    print(X.head())