"""Simulador de temporada.

Simula los partidos que quedan por jugar usando el modelo entrenado: predice
cada partido, actualiza el estado y sigue con el siguiente.

El estado por equipo, las features y la clasificación NO se implementan aquí:
viven en core.state.LeagueState. Este módulo solo orquesta.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from app.config import get_current_season
from core.features import FEATURE_COLS, feature_row, feature_vector
from core.goals import sample_scoreline
from core.state import LeagueState


class SeasonSimulator:
    """Simula el final de una temporada usando un modelo entrenado."""

    def __init__(self, state: LeagueState | None = None):
        self.state = state if state is not None else LeagueState()

    # --- Delegación en LeagueState (API mantenida para los scripts legacy) ---

    def play_match(self, home_team, away_team, home_goals, away_goals, result, match_date):
        self.state.apply_match(home_team, away_team, home_goals, away_goals, result, match_date)

    def generate_features(self, home_team, away_team, current_date):
        return feature_row(self.state, home_team, away_team, current_date)

    def _get_standings(self):
        return self.state.standings()

    def _result_from_prediction(self, pred, home_team, away_team, stochastic=False, rng=None):
        """Convierte la predicción del modelo en resultado y marcador simulados.

        El modelo decide quién gana (para eso está entrenado y validado con
        accuracy/baselines); el marcador se muestrea por Poisson una vez
        decidido el resultado, con los goles esperados de cada equipo según su
        historial (`LeagueState.expected_goals`) — así la diferencia de goles
        dejar de ser siempre ±1, que era lo que pasaba con los tres marcadores
        fijos de antes.
        """
        rng = rng if rng is not None else np.random.default_rng()

        if stochastic:
            outcomes = ['A', 'D', 'H']
            # XGBoost devuelve float32; Generator.choice valida que sumen 1 con
            # tolerancia de float64, así que hay que ampliar y renormalizar o
            # revienta con "probabilities do not sum to 1".
            probs = np.array(
                [pred['away_win_prob'], pred['draw_prob'], pred['home_win_prob']],
                dtype=np.float64,
            )
            probs /= probs.sum()
            result = rng.choice(outcomes, p=probs)
        else:
            result = {'Away Win': 'A', 'Draw': 'D', 'Home Win': 'H'}.get(pred['prediction'], 'D')

        lambda_home, lambda_away = self.state.expected_goals(home_team, away_team)
        home_goals, away_goals = sample_scoreline(rng, result, lambda_home, lambda_away)
        return result, home_goals, away_goals

    # --- Simulación -------------------------------------------------------

    @staticmethod
    def split_played(df, start_date=None):
        """Separa partidos disputados de pendientes."""
        df = df.copy().sort_values('Date', kind='mergesort').reset_index(drop=True)
        jugados = df['Result'].notna() if start_date is None else df['Date'] < start_date
        return df[jugados].copy(), df[~jugados].copy()

    @classmethod
    def state_from_played(cls, played):
        """Reconstruye el estado a partir de los partidos ya disputados.

        Se hace UNA vez y luego se clona con snapshot() en cada simulación: antes
        cada una de las n_sims volvía a reproducir los 380 partidos jugados.
        """
        state = LeagueState()
        for row in played.itertuples(index=False):
            state.apply_match(
                row.HomeTeam, row.AwayTeam,
                int(row.Home_Goals), int(row.Away_Goals), row.Result, row.Date,
            )
        return state

    def simulate_remaining(self, remaining, predictor, stochastic=False, rng=None):
        """Simula los partidos pendientes sobre el estado actual."""
        simulados = []
        for row in remaining.itertuples(index=False):
            # feature_vector (ndarray) en vez de generate_features (DataFrame):
            # aquí se hace una predicción por partido y por simulación
            features = feature_vector(self.state, row.HomeTeam, row.AwayTeam, row.Date)
            pred = predictor.predict_match(features)
            result, home_goals, away_goals = self._result_from_prediction(
                pred, row.HomeTeam, row.AwayTeam, stochastic=stochastic, rng=rng
            )
            self.play_match(row.HomeTeam, row.AwayTeam, home_goals, away_goals, result, row.Date)
            simulados.append({
                'Date': row.Date,
                'HomeTeam': row.HomeTeam,
                'AwayTeam': row.AwayTeam,
                'Home_Goals': home_goals,
                'Away_Goals': away_goals,
                'Result': result,
                'Predicted_Home_Prob': pred['home_win_prob'],
                'Predicted_Draw_Prob': pred['draw_prob'],
                'Predicted_Away_Prob': pred['away_win_prob'],
            })
        return self._get_standings(), pd.DataFrame(simulados)

    def simulate(self, df, predictor, start_date=None, stochastic=False, verbose=True, rng=None):
        """Simula una temporada completa desde cero. Devuelve (clasificación, partidos)."""
        played, remaining = self.split_played(df, start_date)

        if verbose:
            print(f"📅 Partidos ya disputados: {len(played)}")
            print(f"🔮 Partidos pendientes: {len(remaining)}")

        self.state = self.state_from_played(played)
        return self.simulate_remaining(remaining, predictor, stochastic=stochastic, rng=rng)

    def monte_carlo_simulation(self, df, predictor, n_sims=1000, start_date=None, verbose=True,
                               rng=None, on_progress=None):
        """Repite la simulación n_sims veces y acumula probabilidades."""
        if verbose:
            print(f"\n🎲 Iniciando Monte Carlo con {n_sims} simulaciones...")

        rng = rng if rng is not None else np.random.default_rng()
        played, remaining = self.split_played(df, start_date)
        base = self.state_from_played(played)

        champions = []
        top4 = defaultdict(int)
        relegation = defaultdict(int)
        all_final_points = defaultdict(list)

        for i in range(n_sims):
            sim = SeasonSimulator(base.snapshot())
            standings, _ = sim.simulate_remaining(remaining, predictor, stochastic=True, rng=rng)

            champions.append(standings.iloc[0]['Team'])
            for team in standings.head(4)['Team']:
                top4[team] += 1
            for team in standings.tail(3)['Team']:
                relegation[team] += 1
            for row in standings.itertuples(index=False):
                all_final_points[row.Team].append(row.Points)

            if on_progress is not None and (i + 1) % 10 == 0:
                on_progress(i + 1, n_sims)
            if verbose and (i + 1) % 100 == 0:
                print(f"   ✓ Completadas {i + 1}/{n_sims} simulaciones")

        champion_counts = Counter(champions)
        results = {
            'champion_probabilities': {t: c / n_sims for t, c in champion_counts.items()},
            'top4_probabilities': {t: c / n_sims for t, c in top4.items()},
            'relegation_probabilities': {t: c / n_sims for t, c in relegation.items()},
            'final_points_distribution': {
                team: {
                    'mean': np.mean(points),
                    'std': np.std(points),
                    'min': np.min(points),
                    'max': np.max(points),
                }
                for team, points in all_final_points.items()
            },
        }

        if verbose:
            print(f"✓ Monte Carlo completado ({n_sims} simulaciones)")

        return results


def simulate_season(
    league_code: str, n_sims: int = 500, season: str | None = None, seed: int = 42, on_progress=None
):
    """Monte Carlo de los partidos que quedan por jugar de una temporada.

    Devuelve la clasificación proyectada con la probabilidad de título de cada
    equipo. Lanza SeasonFinishedError si no queda nada por simular: repetir la
    temporada ya jugada devolvería un 100 % para el campeón real, que es una
    respuesta sin información y parece un resultado del modelo.

    `on_progress(hechas, total)` se llama cada 10 simulaciones; ver
    `SeasonSimulator.monte_carlo_simulation`.
    """
    from core.dataset import load_season, played_only
    from core.errors import SeasonFinishedError
    from models.predictor import LeaguePredictor

    season = season or get_current_season()
    df = load_season(league_code, season)

    played_df = played_only(df)
    if played_df.empty:
        raise ValueError(f"No hay partidos jugados en {league_code}/{season}")

    pendientes = len(df) - len(played_df)
    if pendientes == 0:
        raise SeasonFinishedError(
            f"La temporada {season} ya ha terminado: no quedan partidos que simular. "
            f"Usa 'Proyectar' para estimar la temporada siguiente."
        )

    predictor = LeaguePredictor(league_code)
    predictor.load_model()  # lanza ModelNotTrainedError si no hay .pkl

    rng = np.random.default_rng(seed)

    sim = SeasonSimulator()
    standings_df, _ = sim.simulate(df, predictor, stochastic=True, verbose=False, rng=rng)

    mc_results = sim.monte_carlo_simulation(
        df, predictor, n_sims=n_sims, verbose=False, rng=rng, on_progress=on_progress
    )
    champion_probs = mc_results.get('champion_probabilities', {})
    relegation_probs = mc_results.get('relegation_probabilities', {})

    standings_list = []
    for _, row in standings_df.iterrows():
        team_name = row['Team']
        standings_list.append({
            'name': team_name,
            'points': int(row['Points']),
            'gf': int(row['GF']),
            'ga': int(row['GA']),
            'champion_prob': champion_probs.get(team_name),
            'relegation_prob': relegation_probs.get(team_name),
        })

    return standings_list
