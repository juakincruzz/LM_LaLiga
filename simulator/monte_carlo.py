"""
Simulador Monte Carlo para predicción del campeón de La Liga.

Usa el modelo ML entrenado para simular los partidos restantes de la temporada
miles de veces, calculando la probabilidad de cada equipo de ser campeón.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
import copy


class MonteCarloSimulator:
    """Simula el resto de la temporada de La Liga usando Monte Carlo.
    
    Para cada simulación:
    1. Toma la clasificación actual como punto de partida
    2. Predice cada partido restante usando las probabilidades del modelo ML
    3. Muestrea el resultado (victoria local, empate, victoria visitante)
    4. Actualiza la clasificación
    5. Registra quién termina campeón
    
    Tras N simulaciones, calcula la probabilidad de cada equipo de ser campeón.
    """
    
    TOTAL_MATCHDAYS = 38  # Jornadas en una temporada de La Liga
    
    def __init__(self, predictor, historical_df, n_simulations=1000, random_seed=42):
        """
        Args:
            predictor: LaLigaPredictor con modelo entrenado y scaler ajustado
            historical_df: DataFrame con partidos históricos (para calcular promedios)
            n_simulations: Número de simulaciones Monte Carlo
            random_seed: Semilla para reproducibilidad
        """
        self.predictor = predictor
        self.historical_df = historical_df.copy()
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(random_seed)
        
        # Pre-calcular promedios históricos por equipo para generar features
        self.team_averages = self._compute_team_averages()
    
    def _compute_team_averages(self):
        """Calcula promedios históricos por equipo para usar como features en partidos futuros.
        
        Cuando predecimos un partido futuro, no tenemos stats del partido (tiros, faltas, etc.).
        Usamos los promedios históricos de cada equipo como estimación.
        """
        df = self.historical_df
        averages = {}
        
        # Columnas de stats del partido
        stat_cols_home = ['Home_Shots', 'Home_ShotsOnTarget', 'Home_Fouls', 'Home_Corners', 'Home_Yellows']
        stat_cols_away = ['Away_Shots', 'Away_ShotsOnTarget', 'Away_Fouls', 'Away_Corners', 'Away_Yellows']
        
        teams = set(list(df['HomeTeam'].unique()) + list(df['AwayTeam'].unique()))
        
        for team in teams:
            home_matches = df[df['HomeTeam'] == team]
            away_matches = df[df['AwayTeam'] == team]
            
            total_home = len(home_matches)
            total_away = len(away_matches)
            total = total_home + total_away
            
            # Promedios cuando juega en casa
            home_stats = {}
            for col in stat_cols_home:
                if col in home_matches.columns:
                    home_stats[col] = home_matches[col].mean() if total_home > 0 else 0
                else:
                    home_stats[col] = 0
                    
            # Promedios cuando juega fuera
            away_stats = {}
            for col in stat_cols_away:
                if col in away_matches.columns:
                    away_stats[col] = away_matches[col].mean() if total_away > 0 else 0
                else:
                    away_stats[col] = 0
            
            # Stats de rendimiento
            home_wins = len(home_matches[home_matches['Result'] == 'H']) if total_home > 0 else 0
            home_draws = len(home_matches[home_matches['Result'] == 'D']) if total_home > 0 else 0
            away_wins = len(away_matches[away_matches['Result'] == 'A']) if total_away > 0 else 0
            away_draws = len(away_matches[away_matches['Result'] == 'D']) if total_away > 0 else 0
            
            home_goals_avg = home_matches['Home_Goals'].mean() if total_home > 0 else 0
            away_goals_avg = away_matches['Away_Goals'].mean() if total_away > 0 else 0
            defense_home = home_matches['Away_Goals'].mean() if total_home > 0 else 0
            defense_away = away_matches['Home_Goals'].mean() if total_away > 0 else 0
            
            total_points = home_wins * 3 + home_draws + away_wins * 3 + away_draws
            
            # Forma reciente (últimos 5 partidos)
            all_matches = []
            for _, row in home_matches.iterrows():
                all_matches.append({
                    'date': row.get('Date', None),
                    'goals_scored': row['Home_Goals'],
                    'goals_conceded': row['Away_Goals'],
                    'is_win': 1 if row['Result'] == 'H' else 0,
                    'points': 3 if row['Result'] == 'H' else (1 if row['Result'] == 'D' else 0)
                })
            for _, row in away_matches.iterrows():
                all_matches.append({
                    'date': row.get('Date', None),
                    'goals_scored': row['Away_Goals'],
                    'goals_conceded': row['Home_Goals'],
                    'is_win': 1 if row['Result'] == 'A' else 0,
                    'points': 3 if row['Result'] == 'A' else (1 if row['Result'] == 'D' else 0)
                })
            
            match_df = pd.DataFrame(all_matches)
            if 'date' in match_df.columns and match_df['date'].notna().any():
                match_df = match_df.sort_values('date')
            
            recent = match_df.tail(5)
            
            averages[team] = {
                # Stats del partido (promedios)
                'home_shots': home_stats.get('Home_Shots', 0),
                'home_shots_on_target': home_stats.get('Home_ShotsOnTarget', 0),
                'home_fouls': home_stats.get('Home_Fouls', 0),
                'home_corners': home_stats.get('Home_Corners', 0),
                'home_yellows': home_stats.get('Home_Yellows', 0),
                'away_shots': away_stats.get('Away_Shots', 0),
                'away_shots_on_target': away_stats.get('Away_ShotsOnTarget', 0),
                'away_fouls': away_stats.get('Away_Fouls', 0),
                'away_corners': away_stats.get('Away_Corners', 0),
                'away_yellows': away_stats.get('Away_Yellows', 0),
                # Stats históricas
                'home_wins': home_wins,
                'away_wins': away_wins,
                'home_goals_avg': home_goals_avg,
                'away_goals_avg': away_goals_avg,
                'defense_home': defense_home,
                'defense_away': defense_away,
                'total_points': total_points,
                # Forma reciente
                'form_points': recent['points'].sum() if len(recent) > 0 else 0,
                'form_goals_scored': recent['goals_scored'].mean() if len(recent) > 0 else 0,
                'form_goals_conceded': recent['goals_conceded'].mean() if len(recent) > 0 else 0,
                'form_wins': recent['is_win'].sum() if len(recent) > 0 else 0,
            }
        
        return averages
    
    def _build_match_features(self, home_team, away_team):
        """Construye el vector de features para un partido futuro.
        
        Usa los promedios históricos de cada equipo ya que no tenemos
        los stats reales del partido (aún no se ha jugado).
        
        Returns:
            pd.DataFrame con una fila y las columnas que espera el modelo
        """
        home_avg = self.team_averages.get(home_team, {})
        away_avg = self.team_averages.get(away_team, {})
        
        # Construir features en el mismo orden que el modelo espera
        features = {
            # Stats del partido (usamos promedios históricos como estimación)
            'Home_Shots': home_avg.get('home_shots', 0),
            'Away_Shots': away_avg.get('away_shots', 0),
            'Home_ShotsOnTarget': home_avg.get('home_shots_on_target', 0),
            'Away_ShotsOnTarget': away_avg.get('away_shots_on_target', 0),
            'Home_Fouls': home_avg.get('home_fouls', 0),
            'Away_Fouls': away_avg.get('away_fouls', 0),
            'Home_Corners': home_avg.get('home_corners', 0),
            'Away_Corners': away_avg.get('away_corners', 0),
            'Home_Yellows': home_avg.get('home_yellows', 0),
            'Away_Yellows': away_avg.get('away_yellows', 0),
            # Stats históricos
            'Home_Wins': home_avg.get('home_wins', 0),
            'Away_Wins': away_avg.get('away_wins', 0),
            'Home_Goals_Avg': home_avg.get('home_goals_avg', 0),
            'Away_Goals_Avg': away_avg.get('away_goals_avg', 0),
            'Home_Defense_Avg': home_avg.get('defense_home', 0),
            'Away_Defense_Avg': away_avg.get('defense_away', 0),
            'Home_Total_Points': home_avg.get('total_points', 0),
            'Away_Total_Points': away_avg.get('total_points', 0),
            # Forma reciente
            'Home_Form_Points': home_avg.get('form_points', 0),
            'Away_Form_Points': away_avg.get('form_points', 0),
            'Home_Form_Goals': home_avg.get('form_goals_scored', 0),
            'Away_Form_Goals': away_avg.get('form_goals_scored', 0),
            'Home_Form_Conceded': home_avg.get('form_goals_conceded', 0),
            'Away_Form_Conceded': away_avg.get('form_goals_conceded', 0),
            'Home_Form_Wins': home_avg.get('form_wins', 0),
            'Away_Form_Wins': away_avg.get('form_wins', 0),
            # Features derivados
            'Points_Diff': home_avg.get('total_points', 0) - away_avg.get('total_points', 0),
            'Wins_Diff': home_avg.get('home_wins', 0) - away_avg.get('away_wins', 0),
            'Goals_Avg_Diff': home_avg.get('home_goals_avg', 0) - away_avg.get('away_goals_avg', 0),
            'Defense_Diff': home_avg.get('defense_home', 0) - away_avg.get('defense_away', 0),
            'Form_Points_Diff': home_avg.get('form_points', 0) - away_avg.get('form_points', 0),
            'Form_Goals_Diff': home_avg.get('form_goals_scored', 0) - away_avg.get('form_goals_scored', 0),
            'Home_xG': home_avg.get('home_shots_on_target', 0) * 0.08,
            'Away_xG': away_avg.get('away_shots_on_target', 0) * 0.08,
        }
        
        return pd.DataFrame([features])
    
    def _predict_match_probabilities(self, home_team, away_team):
        """Predice las probabilidades de un partido usando el modelo ML.
        
        Returns:
            tuple: (prob_home_win, prob_draw, prob_away_win)
        """
        features = self._build_match_features(home_team, away_team)
        
        # Escalar features
        features_scaled = self.predictor.scaler.transform(features) if self.predictor._is_fitted else features
        
        # Obtener probabilidades
        probabilities = self.predictor.model.predict_proba(features_scaled)[0]
        
        # XGBoost encoding: 0=Away Win, 1=Draw, 2=Home Win
        prob_away = probabilities[0]
        prob_draw = probabilities[1]
        prob_home = probabilities[2]
        
        return prob_home, prob_draw, prob_away
    
    def _sample_result(self, prob_home, prob_draw, prob_away):
        """Muestrea un resultado basado en las probabilidades.
        
        Returns:
            str: 'H' (home win), 'D' (draw), o 'A' (away win)
        """
        r = self.rng.random()
        if r < prob_home:
            return 'H'
        elif r < prob_home + prob_draw:
            return 'D'
        else:
            return 'A'
    
    def generate_remaining_fixtures(self, standings_df, teams=None):
        """Genera los partidos restantes de la temporada.
        
        En La Liga, cada equipo juega contra todos los demás dos veces
        (una en casa, otra fuera) = 38 jornadas, 380 partidos totales.
        
        Usamos los partidos ya jugados (de historical_df) para determinar
        qué partidos faltan.
        
        Args:
            standings_df: DataFrame con la clasificación actual
            teams: Lista de equipos (si None, se extrae de standings)
        
        Returns:
            list de tuples: [(home_team, away_team), ...]
        """
        if teams is None:
            if 'Team' in standings_df.columns:
                teams = standings_df['Team'].tolist()
            else:
                teams = standings_df.index.tolist()
        
        # Todos los partidos posibles (ida y vuelta)
        all_fixtures = set()
        for home in teams:
            for away in teams:
                if home != away:
                    all_fixtures.add((home, away))
        
        # Partidos ya jugados
        played = set()
        for _, row in self.historical_df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            if home in teams and away in teams:
                played.add((home, away))
        
        # Partidos restantes
        remaining = all_fixtures - played
        
        return list(remaining)
    
    def simulate(self, standings_df, teams=None, verbose=True):
        """Ejecuta la simulación Monte Carlo completa.
        
        Args:
            standings_df: DataFrame con clasificación actual
                Debe tener columnas: Team (o index), Points, Games
                (o equivalentes: points, games)
            teams: Lista de equipos
            verbose: Si True, muestra progreso
        
        Returns:
            dict con resultados:
                'champion_probs': {equipo: probabilidad}
                'top4_probs': {equipo: probabilidad}
                'relegation_probs': {equipo: probabilidad}
                'avg_points': {equipo: puntos_promedio_finales}
                'median_points': {equipo: puntos_medianos_finales}
                'points_range': {equipo: (min, max)}
        """
        if verbose:
            print(f"\n🎲 Iniciando simulación Monte Carlo ({self.n_simulations} simulaciones)...")
        
        # Normalizar standings a un dict {equipo: puntos_actuales}
        current_points = {}
        if 'Team' in standings_df.columns:
            for _, row in standings_df.iterrows():
                team = row['Team']
                pts = row.get('Points', row.get('points', 0))
                current_points[team] = int(pts)
        else:
            for team, row in standings_df.iterrows():
                pts = row.get('Points', row.get('points', 0))
                current_points[team] = int(pts)
        
        if teams is None:
            teams = list(current_points.keys())
        
        # Generar partidos restantes
        remaining_fixtures = self.generate_remaining_fixtures(standings_df, teams)
        
        if verbose:
            print(f"   📋 Equipos: {len(teams)}")
            print(f"   ⚽ Partidos restantes: {len(remaining_fixtures)}")
        
        # Pre-calcular probabilidades para cada fixture (para no recalcular en cada simulación)
        if verbose:
            print(f"   🤖 Calculando probabilidades con el modelo ML...")
        
        fixture_probs = {}
        for home, away in remaining_fixtures:
            try:
                probs = self._predict_match_probabilities(home, away)
                fixture_probs[(home, away)] = probs
            except Exception as e:
                # Si no puede predecir, usa probabilidades por defecto (ventaja local)
                fixture_probs[(home, away)] = (0.45, 0.27, 0.28)
        
        if verbose:
            print(f"   ✓ Probabilidades calculadas para {len(fixture_probs)} partidos")
            print(f"\n   🎰 Simulando {self.n_simulations} temporadas...")
        
        # Contadores
        champion_count = defaultdict(int)
        top4_count = defaultdict(int)
        relegation_count = defaultdict(int)
        all_final_points = defaultdict(list)
        
        # Ejecutar simulaciones
        progress_step = max(1, self.n_simulations // 10)
        
        for sim in range(self.n_simulations):
            # Copiar puntos actuales
            sim_points = dict(current_points)
            
            # Simular cada partido restante
            for (home, away), (ph, pd, pa) in fixture_probs.items():
                result = self._sample_result(ph, pd, pa)
                
                if result == 'H':
                    sim_points[home] = sim_points.get(home, 0) + 3
                elif result == 'D':
                    sim_points[home] = sim_points.get(home, 0) + 1
                    sim_points[away] = sim_points.get(away, 0) + 1
                else:  # 'A'
                    sim_points[away] = sim_points.get(away, 0) + 3
            
            # Clasificación final de esta simulación
            final_standings = sorted(sim_points.items(), key=lambda x: x[1], reverse=True)
            
            # Registrar resultados
            champion = final_standings[0][0]
            champion_count[champion] += 1
            
            for team, pts in final_standings[:4]:
                top4_count[team] += 1
            
            for team, pts in final_standings[-3:]:
                relegation_count[team] += 1
            
            for team, pts in final_standings:
                all_final_points[team].append(pts)
            
            if verbose and (sim + 1) % progress_step == 0:
                pct = (sim + 1) / self.n_simulations * 100
                print(f"      Progreso: {pct:.0f}% ({sim + 1}/{self.n_simulations})")
        
        # Calcular resultados finales
        results = {
            'champion_probs': {},
            'top4_probs': {},
            'relegation_probs': {},
            'avg_points': {},
            'median_points': {},
            'points_range': {},
            'current_points': current_points,
            'n_simulations': self.n_simulations,
            'n_remaining_fixtures': len(remaining_fixtures),
        }
        
        for team in teams:
            results['champion_probs'][team] = champion_count[team] / self.n_simulations
            results['top4_probs'][team] = top4_count[team] / self.n_simulations
            results['relegation_probs'][team] = relegation_count[team] / self.n_simulations
            
            pts_list = all_final_points[team]
            results['avg_points'][team] = np.mean(pts_list) if pts_list else 0
            results['median_points'][team] = np.median(pts_list) if pts_list else 0
            results['points_range'][team] = (
                int(np.min(pts_list)) if pts_list else 0,
                int(np.max(pts_list)) if pts_list else 0
            )
        
        if verbose:
            print(f"   ✓ Simulación completada")
        
        return results
