"""
Simulador de temporada para LM_LaLiga.

Permite simular los partidos restantes de una temporada usando el modelo de
Machine Learning entrenado, en lugar de proyectar puntos linealmente.

El simulador procesa los partidos ya disputados para reconstruir el estado de
cada equipo y luego va prediciendo los resultados de los partidos pendientes
uno a uno, actualizando el estado tras cada predicción.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from scripts.preprocess import FEATURE_COLS


class SeasonSimulator:
    """Simula el final de una temporada usando un modelo entrenado."""

    def __init__(self):
        # Estado de cada equipo (misma estructura que FeatureEngineer)
        self.team_stats = defaultdict(lambda: {
            'matches': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'points': 0,
            'goals_for': 0,
            'goals_against': 0,
            'results': [],

            'home_matches': 0,
            'home_wins': 0,
            'home_draws': 0,
            'home_losses': 0,
            'home_points': 0,
            'home_goals_for': 0,
            'home_goals_against': 0,
            'home_results': [],

            'away_matches': 0,
            'away_wins': 0,
            'away_draws': 0,
            'away_losses': 0,
            'away_points': 0,
            'away_goals_for': 0,
            'away_goals_against': 0,
            'away_results': []
        })

        self.last_match_date = {}
        self.h2h_records = defaultdict(list)

    @staticmethod
    def _safe_avg(value, count):
        return value / max(count, 1)

    @staticmethod
    def _form_n(results, n):
        if not results:
            return 0.0
        recent = results[-n:]
        return sum(recent) / len(recent)

    def play_match(self, home_team, away_team, home_goals, away_goals, result, match_date):
        """
        Actualiza el estado del simulador con un partido ya disputado.
        """
        home = self.team_stats[home_team]
        away = self.team_stats[away_team]

        # Estadísticas generales
        home['matches'] += 1
        away['matches'] += 1
        home['goals_for'] += home_goals
        home['goals_against'] += away_goals
        away['goals_for'] += away_goals
        away['goals_against'] += home_goals

        # Local / visitante
        home['home_matches'] += 1
        home['home_goals_for'] += home_goals
        home['home_goals_against'] += away_goals

        away['away_matches'] += 1
        away['away_goals_for'] += away_goals
        away['away_goals_against'] += home_goals

        # Resultados y puntos
        if result == 'H':
            home['wins'] += 1
            home['home_wins'] += 1
            home['points'] += 3
            home['home_points'] += 3
            home['results'].append(3)
            home['home_results'].append(3)

            away['losses'] += 1
            away['away_losses'] += 1
            away['results'].append(0)
            away['away_results'].append(0)

        elif result == 'A':
            away['wins'] += 1
            away['away_wins'] += 1
            away['points'] += 3
            away['away_points'] += 3
            away['results'].append(3)
            away['away_results'].append(3)

            home['losses'] += 1
            home['home_losses'] += 1
            home['results'].append(0)
            home['home_results'].append(0)

        else:  # Empate
            home['draws'] += 1
            home['home_draws'] += 1
            home['points'] += 1
            home['home_points'] += 1
            home['results'].append(1)
            home['home_results'].append(1)

            away['draws'] += 1
            away['away_draws'] += 1
            away['points'] += 1
            away['away_points'] += 1
            away['results'].append(1)
            away['away_results'].append(1)

        # Fechas y H2H
        self.last_match_date[home_team] = match_date
        self.last_match_date[away_team] = match_date

        if result == 'H':
            self.h2h_records[(home_team, away_team)].append(3)
        elif result == 'A':
            self.h2h_records[(home_team, away_team)].append(0)
        else:
            self.h2h_records[(home_team, away_team)].append(1)

    def generate_features(self, home_team, away_team, current_date):
        """
        Genera el vector de features para un partido futuro usando el estado
        actual del simulador.
        """
        home = self.team_stats[home_team]
        away = self.team_stats[away_team]

        features = {}

        # Estadísticas generales
        features['Home_Matches'] = home['matches']
        features['Away_Matches'] = away['matches']
        features['Home_Win_Rate'] = home['wins'] / max(home['matches'], 1)
        features['Away_Win_Rate'] = away['wins'] / max(away['matches'], 1)
        features['Home_Points_Per_Game'] = home['points'] / max(home['matches'], 1)
        features['Away_Points_Per_Game'] = away['points'] / max(away['matches'], 1)
        features['Home_Goals_For_Avg'] = self._safe_avg(home['goals_for'], home['matches'])
        features['Away_Goals_For_Avg'] = self._safe_avg(away['goals_for'], away['matches'])
        features['Home_Goals_Against_Avg'] = self._safe_avg(home['goals_against'], home['matches'])
        features['Away_Goals_Against_Avg'] = self._safe_avg(away['goals_against'], away['matches'])

        # Forma reciente
        features['Home_Form_3'] = self._form_n(home['results'], 3)
        features['Away_Form_3'] = self._form_n(away['results'], 3)
        features['Home_Form_5'] = self._form_n(home['results'], 5)
        features['Away_Form_5'] = self._form_n(away['results'], 5)

        # Rendimiento local/visitante específico
        features['Home_Home_Win_Rate'] = home['home_wins'] / max(home['home_matches'], 1)
        features['Away_Away_Win_Rate'] = away['away_wins'] / max(away['away_matches'], 1)
        features['Home_Home_Points_Per_Game'] = home['home_points'] / max(home['home_matches'], 1)
        features['Away_Away_Points_Per_Game'] = away['away_points'] / max(away['away_matches'], 1)
        features['Home_Home_Goals_For_Avg'] = self._safe_avg(home['home_goals_for'], home['home_matches'])
        features['Away_Away_Goals_For_Avg'] = self._safe_avg(away['away_goals_for'], away['away_matches'])
        features['Home_Home_Goals_Against_Avg'] = self._safe_avg(home['home_goals_against'], home['home_matches'])
        features['Away_Away_Goals_Against_Avg'] = self._safe_avg(away['away_goals_against'], away['away_matches'])
        features['Home_Home_Form_3'] = self._form_n(home['home_results'], 3)
        features['Away_Away_Form_3'] = self._form_n(away['away_results'], 3)

        # Diferencias
        features['Points_Diff'] = features['Home_Points_Per_Game'] - features['Away_Points_Per_Game']
        features['Win_Rate_Diff'] = features['Home_Win_Rate'] - features['Away_Win_Rate']
        features['Goals_For_Diff'] = features['Home_Goals_For_Avg'] - features['Away_Goals_For_Avg']
        features['Goals_Against_Diff'] = features['Home_Goals_Against_Avg'] - features['Away_Goals_Against_Avg']
        features['Form_5_Diff'] = features['Home_Form_5'] - features['Away_Form_5']
        features['Home_Advantage'] = features['Home_Home_Points_Per_Game'] - features['Home_Points_Per_Game']

        # Descanso
        home_last = self.last_match_date.get(home_team)
        away_last = self.last_match_date.get(away_team)
        features['Home_Days_Rest'] = (current_date - home_last).days if home_last is not None else 7
        features['Away_Days_Rest'] = (current_date - away_last).days if away_last is not None else 7

        # H2H
        h2h = self.h2h_records[(home_team, away_team)]
        features['H2H_Matches'] = len(h2h)
        features['H2H_Home_Points_Per_Game'] = sum(h2h) / len(h2h) if h2h else 1.5
        features['H2H_Home_Win_Rate'] = sum(1 for r in h2h if r == 3) / len(h2h) if h2h else 0.33

        return pd.DataFrame([features], columns=FEATURE_COLS).fillna(0)

    @staticmethod
    def _result_from_prediction(pred, stochastic=False):
        """
        Convierte la predicción del modelo en resultado y goles simulados.

        Parámetros
        ----------
        pred : dict
            Diccionario con 'prediction' y probabilidades.
        stochastic : bool
            Si es True, samplea el resultado según las probabilidades del modelo
            (Monte Carlo). Si es False, usa la clase con mayor probabilidad.
        """
        if stochastic:
            outcomes = ['A', 'D', 'H']
            probs = [
                pred['away_win_prob'],
                pred['draw_prob'],
                pred['home_win_prob']
            ]
            result = np.random.choice(outcomes, p=probs)
        else:
            result = {'Away Win': 'A', 'Draw': 'D', 'Home Win': 'H'}.get(
                pred['prediction'], 'D'
            )

        # Goles simulados simplificados según el resultado
        if result == 'H':
            return 'H', 2, 1
        elif result == 'A':
            return 'A', 1, 2
        else:
            return 'D', 1, 1

    def simulate(self, df, predictor, start_date=None, max_gameweek=None, stochastic=False, verbose=True):
        """
        Simula los partidos restantes de una temporada.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con todos los partidos de la temporada (jugados y pendientes).
        predictor : LaLigaPredictor
            Modelo entrenado para predecir resultados.
        start_date : datetime | None
            Fecha desde la cual simular. Si es None, simula desde el primer partido
            no disputado. Un partido se considera disputado si tiene 'Result' no nulo.
        max_gameweek : int | None
            Número máximo de jornadas a simular (no implementado actualmente).
        stochastic : bool
            Si es True, samplea resultados según probabilidades (Monte Carlo).
            Si es False, usa siempre la predicción más probable.
        verbose : bool
            Si es True, imprime información de progreso.

        Retorna
        -------
        tuple
            (standings_df, simulated_matches)
        """
        df = df.copy().sort_values('Date').reset_index(drop=True)

        if start_date is None:
            # Simular desde el primer partido sin resultado
            played_mask = df['Result'].notna()
        else:
            played_mask = df['Date'] < start_date

        played = df[played_mask].copy()
        remaining = df[~played_mask].copy()

        if verbose:
            print(f"📅 Partidos ya disputados: {len(played)}")
            print(f"🔮 Partidos pendientes: {len(remaining)}")

        # Reproducir partidos ya jugados
        for _, row in played.iterrows():
            self.play_match(
                row['HomeTeam'], row['AwayTeam'],
                int(row['Home_Goals']), int(row['Away_Goals']),
                row['Result'], row['Date']
            )

        simulated_matches = []

        # Simular partidos pendientes
        for _, row in remaining.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_date = row['Date']

            features = self.generate_features(home_team, away_team, match_date)
            pred = predictor.predict_match(features)
            result, home_goals, away_goals = self._result_from_prediction(pred, stochastic=stochastic)

            self.play_match(home_team, away_team, home_goals, away_goals, result, match_date)

            simulated_matches.append({
                'Date': match_date,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'Home_Goals': home_goals,
                'Away_Goals': away_goals,
                'Result': result,
                'Predicted_Home_Prob': pred['home_win_prob'],
                'Predicted_Draw_Prob': pred['draw_prob'],
                'Predicted_Away_Prob': pred['away_win_prob']
            })

        standings = self._get_standings()
        return standings, pd.DataFrame(simulated_matches)

    def _get_standings(self):
        """Calcula la clasificación a partir del estado actual."""
        standings = []
        for team, stats in self.team_stats.items():
            standings.append({
                'Team': team,
                'Games': stats['matches'],
                'Wins': stats['wins'],
                'Draws': stats['draws'],
                'Losses': stats['losses'],
                'GF': stats['goals_for'],
                'GA': stats['goals_against'],
                'Points': stats['points']
            })

        standings_df = pd.DataFrame(standings)
        standings_df['goal_diff'] = standings_df['GF'] - standings_df['GA']
        standings_df = standings_df.sort_values(['Points', 'goal_diff', 'GF'], ascending=False)
        standings_df['Position'] = range(1, len(standings_df) + 1)

        return standings_df.reset_index(drop=True)

    def monte_carlo_simulation(self, df, predictor, n_sims=1000, start_date=None, verbose=True):
        """
        Ejecuta múltiples simulaciones estocásticas de la temporada y acumula
        resultados para estimar probabilidades de campeón, Champions League,
        descenso, etc.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con todos los partidos de la temporada.
        predictor : LaLigaPredictor
            Modelo entrenado.
        n_sims : int
            Número de simulaciones (por defecto 1000).
        start_date : datetime | None
            Fecha desde la cual simular. Si es None, usa los partidos pendientes.
        verbose : bool
            Si es True, muestra progreso por consola.

        Retorna
        -------
        dict
            Diccionario con conteos y probabilidades por objetivo.
        """
        if verbose:
            print(f"\n🎲 Iniciando Monte Carlo con {n_sims} simulaciones...")

        champions = []
        top4 = defaultdict(int)
        relegation = defaultdict(int)
        all_final_points = defaultdict(list)

        for i in range(n_sims):
            sim = SeasonSimulator()
            standings, _ = sim.simulate(df, predictor, start_date=start_date, stochastic=True, verbose=False)

            champion = standings.iloc[0]['Team']
            champions.append(champion)

            for team in standings.head(4)['Team']:
                top4[team] += 1

            for team in standings.tail(3)['Team']:
                relegation[team] += 1

            for _, row in standings.iterrows():
                all_final_points[row['Team']].append(row['Points'])

            if verbose and (i + 1) % 100 == 0:
                print(f"   ✓ Completadas {i + 1}/{n_sims} simulaciones")

        champion_counts = Counter(champions)
        total = n_sims

        results = {
            'champion_probabilities': {
                team: count / total for team, count in champion_counts.items()
            },
            'top4_probabilities': {
                team: count / total for team, count in top4.items()
            },
            'relegation_probabilities': {
                team: count / total for team, count in relegation.items()
            },
            'final_points_distribution': {
                team: {
                    'mean': np.mean(points),
                    'std': np.std(points),
                    'min': np.min(points),
                    'max': np.max(points)
                }
                for team, points in all_final_points.items()
            }
        }

        if verbose:
            print(f"✓ Monte Carlo completado ({n_sims} simulaciones)")

        return results


def print_simulated_standings(standings_df, title="🏆 CLASIFICACIÓN SIMULADA"):
    """Imprime la clasificación simulada de forma formateada."""
    print(f"\n{title}")
    print(f"{'POS':<5}{'EQUIPO':<25}{'PJ':<5}{'V':<5}{'E':<5}{'D':<5}{'DG':<5}{'PTS':<5}")
    print("-" * 60)

    for _, row in standings_df.iterrows():
        print(f"{int(row['Position']):<5}{row['Team']:<25}{int(row['Games']):<5}"
              f"{int(row['Wins']):<5}{int(row['Draws']):<5}{int(row['Losses']):<5}"
              f"{int(row['goal_diff']):<5}{int(row['Points']):<5}")


def print_monte_carlo_results(results, title="🎲 RESULTADOS MONTE CARLO", top_n=10):
    """Imprime las probabilidades calculadas por Monte Carlo."""
    print(f"\n{title}")
    print("=" * 90)

    print("\n🏆 Probabilidades de campeón:")
    probs = sorted(results['champion_probabilities'].items(), key=lambda x: x[1], reverse=True)
    for i, (team, prob) in enumerate(probs[:top_n], 1):
        bar = "█" * int(prob * 40)
        print(f"   {i:>2}. {team:<25} {prob:>6.1%} {bar}")

    print("\n🥇 Probabilidades de entrar en Champions (Top 4):")
    top4 = sorted(results['top4_probabilities'].items(), key=lambda x: x[1], reverse=True)
    for i, (team, prob) in enumerate(top4[:top_n], 1):
        bar = "█" * int(prob * 40)
        print(f"   {i:>2}. {team:<25} {prob:>6.1%} {bar}")

    print("\n⚠️  Probabilidades de descenso (últimos 3):")
    relegation = sorted(results['relegation_probabilities'].items(), key=lambda x: x[1], reverse=True)
    for i, (team, prob) in enumerate(relegation[:top_n], 1):
        bar = "█" * int(prob * 40)
        print(f"   {i:>2}. {team:<25} {prob:>6.1%} {bar}")

    print("\n📊 Puntos finales esperados (media ± std):")
    points = sorted(
        results['final_points_distribution'].items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    )
    for i, (team, stats) in enumerate(points[:top_n], 1):
        print(f"   {i:>2}. {team:<25} {stats['mean']:>5.1f} ± {stats['std']:>4.1f} "
              f"({stats['min']:.0f}-{stats['max']:.0f})")


def plot_monte_carlo_probabilities(results, output_path='data/processed/monte_carlo_probabilities.png'):
    """Genera un gráfico de barras con probabilidades de campeón."""
    try:
        import matplotlib.pyplot as plt

        probs = sorted(results['champion_probabilities'].items(), key=lambda x: x[1], reverse=True)
        teams = [p[0] for p in probs]
        values = [p[1] for p in probs]

        plt.figure(figsize=(10, 8))
        bars = plt.barh(teams[::-1], values[::-1], color='steelblue')
        plt.xlabel('Probabilidad de campeón')
        plt.title('Probabilidades de ganar La Liga (Monte Carlo)')
        plt.xlim(0, 1)

        # Añadir porcentajes en las barras
        for bar, val in zip(bars, values[::-1]):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.1%}", va='center', fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"✓ Gráfico guardado en {output_path}")

    except ImportError:
        print("⚠️  matplotlib no disponible. No se generó el gráfico.")


def simulate_season(league_code: str, n_sims: int = 500):
    """
    Run Monte Carlo simulation for a league season.

    This is a provisional wrapper that loads generic league data, trains or loads
    a predictor, and runs the Monte Carlo simulation. It will be refined in later tasks.
    """
    import pandas as pd
    from scripts.data_manager_generic import ensure_league_data
    from scripts.preprocess import FeatureEngineer
    from models.predictor import LeaguePredictor
    from scripts.model_evaluation import temporal_split

    csv_path = ensure_league_data(league_code, "2526")
    df = pd.read_csv(csv_path)

    # Standardize football-data.co.uk column names
    rename_map = {
        "FTHG": "Home_Goals",
        "FTAG": "Away_Goals",
        "FTR": "Result",
        "HS": "Home_Shots",
        "AS": "Away_Shots",
        "HST": "Home_ShotsOnTarget",
        "AST": "Away_ShotsOnTarget",
        "HF": "Home_Fouls",
        "AF": "Away_Fouls",
        "HC": "Home_Corners",
        "AC": "Away_Corners",
        "HY": "Home_Yellows",
        "AY": "Away_Yellows",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    for col in ["Home_Goals", "Away_Goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive Result from goals if column missing
    if "Result" not in df.columns or df["Result"].isna().all():
        df["Result"] = df.apply(
            lambda r: "H"
            if r["Home_Goals"] > r["Away_Goals"]
            else ("A" if r["Home_Goals"] < r["Away_Goals"] else "D"),
            axis=1,
        )

    # Train predictor on played matches only
    played_df = df.dropna(subset=["Home_Goals", "Away_Goals", "Result"]).copy()
    if played_df.empty:
        raise ValueError(f"No played matches found for {league_code}")

    engineer = FeatureEngineer(played_df)
    X, y, _ = engineer.prepare_features()

    predictor = LeaguePredictor(league_code)
    try:
        predictor.load_model()
    except FileNotFoundError:
        X_train, X_test, y_train, y_test, _, _ = temporal_split(X, y, played_df, test_size=0.2)
        predictor.train_xgboost(X_train, y_train, X_test, y_test)
        predictor.save_model()

    # Run a single deterministic projection for the standings table
    sim = SeasonSimulator()
    standings_df, _ = sim.simulate(df, predictor, stochastic=True, verbose=False)

    # Run Monte Carlo for probabilities
    mc_results = sim.monte_carlo_simulation(
        df, predictor, n_sims=n_sims, stochastic=True, verbose=False
    )

    champion_probs = mc_results.get('champion_probabilities', {})

    standings_list = []
    for _, row in standings_df.iterrows():
        team_name = row['Team']
        standings_list.append({
            'name': team_name,
            'points': int(row['Points']),
            'gf': int(row['GF']),
            'ga': int(row['GA']),
            'champion_prob': champion_probs.get(team_name, None),
        })

    return standings_list


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_manager import ensure_raw_data
    from scripts.load_data import DataLoader
    from scripts.preprocess import FeatureEngineer
    from scripts.model_evaluation import temporal_split
    from models.predictor import LaLigaPredictor

    ensure_raw_data()

    loader = DataLoader()
    loader.load_csv_files()
    loader.clean_data()

    engineer = FeatureEngineer(loader.df)
    X, y, df_features = engineer.prepare_features()

    X_train, X_test, y_train, y_test, _, _ = temporal_split(X, y, df_features, test_size=0.2)

    predictor = LaLigaPredictor()
    predictor.compare_models(X_train, y_train, X_test, y_test)

    simulator = SeasonSimulator()
    standings, simulated = simulator.simulate(loader.df, predictor)
    print_simulated_standings(standings)
