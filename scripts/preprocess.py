import pandas as pd
import numpy as np
from collections import defaultdict


# Columnas de features usadas por el modelo. Se exportan para que otros
# módulos (como el simulador de temporada) puedan generar vectores consistentes.
FEATURE_COLS = [
    # Estadísticas generales acumuladas
    'Home_Matches', 'Away_Matches',
    'Home_Win_Rate', 'Away_Win_Rate',
    'Home_Points_Per_Game', 'Away_Points_Per_Game',
    'Home_Goals_For_Avg', 'Away_Goals_For_Avg',
    'Home_Goals_Against_Avg', 'Away_Goals_Against_Avg',

    # Forma reciente
    'Home_Form_3', 'Away_Form_3',
    'Home_Form_5', 'Away_Form_5',

    # Rendimiento local/visitante específico
    'Home_Home_Win_Rate', 'Away_Away_Win_Rate',
    'Home_Home_Points_Per_Game', 'Away_Away_Points_Per_Game',
    'Home_Home_Goals_For_Avg', 'Away_Away_Goals_For_Avg',
    'Home_Home_Goals_Against_Avg', 'Away_Away_Goals_Against_Avg',
    'Home_Home_Form_3', 'Away_Away_Form_3',

    # Diferencias y factor campo
    'Points_Diff', 'Win_Rate_Diff',
    'Goals_For_Diff', 'Goals_Against_Diff',
    'Form_5_Diff', 'Home_Advantage',

    # Descanso y enfrentamientos directos
    'Home_Days_Rest', 'Away_Days_Rest',
    'H2H_Matches', 'H2H_Home_Points_Per_Game', 'H2H_Home_Win_Rate'
]


class FeatureEngineer:
    """
    Clase para crear características (features) para el modelo ML.

    IMPORTANTE: todas las características se calculan usando solo información
    disponible ANTES del partido. No se utiliza ningún dato del propio partido
    para evitar data leakage.
    """

    def __init__(self, df):
        self.df = df.copy().sort_values('Date').reset_index(drop=True)
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

        # Última fecha de partido de cada equipo (para calcular descanso)
        self.last_match_date = {}

        # Historial de enfrentamientos directos (H2H) entre pares de equipos
        # Clave: (equipo_local, equipo_visitante) en el orden del partido
        self.h2h_records = defaultdict(list)


    @staticmethod
    def _safe_avg(value, count):
        """Calcula un promedio seguro evitando división por cero."""
        return value / max(count, 1)

    @staticmethod
    def _form_n(results, n):
        """Calcula el promedio de puntos por partido en los últimos n resultados."""
        if not results:
            return 0.0
        recent = results[-n:]
        return sum(recent) / len(recent)

    def calculate_rolling_features(self):
        """
        Calcula características acumulativas y de forma reciente para cada partido
        usando únicamente datos previos a ese partido.
        """
        print("📊 Calculando estadísticas acumulativas sin data leakage...")

        for idx, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            home = self.team_stats[home_team]
            away = self.team_stats[away_team]

            # --------------------------
            # Features del equipo local
            # --------------------------
            self.df.at[idx, 'Home_Matches'] = home['matches']
            self.df.at[idx, 'Home_Win_Rate'] = home['wins'] / max(home['matches'], 1)
            self.df.at[idx, 'Home_Points_Per_Game'] = home['points'] / max(home['matches'], 1)
            self.df.at[idx, 'Home_Goals_For_Avg'] = self._safe_avg(home['goals_for'], home['matches'])
            self.df.at[idx, 'Home_Goals_Against_Avg'] = self._safe_avg(home['goals_against'], home['matches'])
            self.df.at[idx, 'Home_Form_3'] = self._form_n(home['results'], 3)
            self.df.at[idx, 'Home_Form_5'] = self._form_n(home['results'], 5)

            # Rendimiento específico como local
            self.df.at[idx, 'Home_Home_Win_Rate'] = home['home_wins'] / max(home['home_matches'], 1)
            self.df.at[idx, 'Home_Home_Points_Per_Game'] = home['home_points'] / max(home['home_matches'], 1)
            self.df.at[idx, 'Home_Home_Goals_For_Avg'] = self._safe_avg(home['home_goals_for'], home['home_matches'])
            self.df.at[idx, 'Home_Home_Goals_Against_Avg'] = self._safe_avg(home['home_goals_against'], home['home_matches'])
            self.df.at[idx, 'Home_Home_Form_3'] = self._form_n(home['home_results'], 3)

            # --------------------------
            # Features del equipo visitante
            # --------------------------
            self.df.at[idx, 'Away_Matches'] = away['matches']
            self.df.at[idx, 'Away_Win_Rate'] = away['wins'] / max(away['matches'], 1)
            self.df.at[idx, 'Away_Points_Per_Game'] = away['points'] / max(away['matches'], 1)
            self.df.at[idx, 'Away_Goals_For_Avg'] = self._safe_avg(away['goals_for'], away['matches'])
            self.df.at[idx, 'Away_Goals_Against_Avg'] = self._safe_avg(away['goals_against'], away['matches'])
            self.df.at[idx, 'Away_Form_3'] = self._form_n(away['results'], 3)
            self.df.at[idx, 'Away_Form_5'] = self._form_n(away['results'], 5)

            # Rendimiento específico como visitante
            self.df.at[idx, 'Away_Away_Win_Rate'] = away['away_wins'] / max(away['away_matches'], 1)
            self.df.at[idx, 'Away_Away_Points_Per_Game'] = away['away_points'] / max(away['away_matches'], 1)
            self.df.at[idx, 'Away_Away_Goals_For_Avg'] = self._safe_avg(away['away_goals_for'], away['away_matches'])
            self.df.at[idx, 'Away_Away_Goals_Against_Avg'] = self._safe_avg(away['away_goals_against'], away['away_matches'])
            self.df.at[idx, 'Away_Away_Form_3'] = self._form_n(away['away_results'], 3)

            # --------------------------
            # Diferencias entre equipos
            # --------------------------
            self.df.at[idx, 'Points_Diff'] = (
                self.df.at[idx, 'Home_Points_Per_Game'] - self.df.at[idx, 'Away_Points_Per_Game']
            )
            self.df.at[idx, 'Win_Rate_Diff'] = (
                self.df.at[idx, 'Home_Win_Rate'] - self.df.at[idx, 'Away_Win_Rate']
            )
            self.df.at[idx, 'Goals_For_Diff'] = (
                self.df.at[idx, 'Home_Goals_For_Avg'] - self.df.at[idx, 'Away_Goals_For_Avg']
            )
            self.df.at[idx, 'Goals_Against_Diff'] = (
                self.df.at[idx, 'Home_Goals_Against_Avg'] - self.df.at[idx, 'Away_Goals_Against_Avg']
            )
            self.df.at[idx, 'Form_5_Diff'] = (
                self.df.at[idx, 'Home_Form_5'] - self.df.at[idx, 'Away_Form_5']
            )

            # Factor campo: diferencia entre rendimiento local y rendimiento general
            self.df.at[idx, 'Home_Advantage'] = (
                self.df.at[idx, 'Home_Home_Points_Per_Game'] - self.df.at[idx, 'Home_Points_Per_Game']
            )

            # --------------------------
            # Features de descanso
            # --------------------------
            current_date = row['Date']
            home_last = self.last_match_date.get(home_team)
            away_last = self.last_match_date.get(away_team)

            self.df.at[idx, 'Home_Days_Rest'] = (
                (current_date - home_last).days if home_last is not None else 7
            )
            self.df.at[idx, 'Away_Days_Rest'] = (
                (current_date - away_last).days if away_last is not None else 7
            )

            # --------------------------
            # Features de enfrentamientos directos (H2H)
            # --------------------------
            h2h = self.h2h_records[(home_team, away_team)]
            self.df.at[idx, 'H2H_Matches'] = len(h2h)
            self.df.at[idx, 'H2H_Home_Points_Per_Game'] = (
                sum(h2h) / len(h2h) if h2h else 1.5  # 1.5 es el valor esperado neutral
            )
            self.df.at[idx, 'H2H_Home_Win_Rate'] = (
                sum(1 for r in h2h if r == 3) / len(h2h) if h2h else 0.33
            )

            # ---------------------------------
            # Actualizar estadísticas con el partido actual
            # (solo después de haber generado las features)
            # ---------------------------------
            home_goals = int(row['Home_Goals'])
            away_goals = int(row['Away_Goals'])
            result = row['Result']

            # Estadísticas generales
            home['matches'] += 1
            away['matches'] += 1
            home['goals_for'] += home_goals
            home['goals_against'] += away_goals
            away['goals_for'] += away_goals
            away['goals_against'] += home_goals

            # Estadísticas como local del equipo local
            home['home_matches'] += 1
            home['home_goals_for'] += home_goals
            home['home_goals_against'] += away_goals

            # Estadísticas como visitante del equipo visitante
            away['away_matches'] += 1
            away['away_goals_for'] += away_goals
            away['away_goals_against'] += home_goals

            # Actualizar resultados y puntos según el marcador
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

            # Actualizar fecha del último partido de cada equipo
            self.last_match_date[home_team] = current_date
            self.last_match_date[away_team] = current_date

            # Actualizar H2H: puntos obtenidos por el equipo local
            if result == 'H':
                self.h2h_records[(home_team, away_team)].append(3)
            elif result == 'A':
                self.h2h_records[(home_team, away_team)].append(0)
            else:
                self.h2h_records[(home_team, away_team)].append(1)

        return self.df

    def encode_target(self):
        """Codifica la variable objetivo: resultado del partido."""
        self.df['Target'] = self.df['Result'].map({'H': 1, 'D': 0, 'A': -1})
        return self.df

    def prepare_features(self):
        """Prepara todas las características y devuelve X, y y el dataframe completo."""
        self.calculate_rolling_features()
        self.encode_target()

        X = self.df[FEATURE_COLS].fillna(0)
        y = self.df['Target']

        return X, y, self.df


if __name__ == "__main__":
    from scripts.load_data import DataLoader

    loader = DataLoader()
    loader.load_csv_files()
    loader.clean_data()
    df = loader.df

    engineer = FeatureEngineer(df)
    X, y, df_features = engineer.prepare_features()

    print("\n✓ Features preparados:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"\nPrimeras features:")
    print(X.head())
