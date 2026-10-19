"""Simulador de temporada.

Simula los partidos que quedan por jugar usando el modelo entrenado: predice
cada partido, actualiza el estado y sigue con el siguiente.

El estado por equipo, las features y la clasificación NO se implementan aquí:
viven en core.state.LeagueState. Este módulo solo orquesta.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from app.config import get_current_season, get_training_seasons
from core.features import build_state, feature_row, feature_vector
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
        Solo ve los partidos que recibe; para el historial completo (H2H de
        temporadas anteriores) usa `build_current_state` y pasa `base_state`.
        """
        return build_state(played, reset_per_season=False)

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

    def simulate(self, df, predictor, start_date=None, stochastic=False, verbose=True, rng=None,
                 base_state=None):
        """Simula una temporada completa desde cero. Devuelve (clasificación, partidos).

        `base_state`, si se pasa, es el estado con el que arrancar (ver
        `build_current_state`); si no, se reconstruye con los jugados de `df`.
        """
        played, remaining = self.split_played(df, start_date)

        if verbose:
            print(f"📅 Partidos ya disputados: {len(played)}")
            print(f"🔮 Partidos pendientes: {len(remaining)}")

        self.state = base_state.snapshot() if base_state is not None else self.state_from_played(played)
        return self.simulate_remaining(remaining, predictor, stochastic=stochastic, rng=rng)

    def monte_carlo_simulation(self, df, predictor, n_sims=1000, start_date=None, verbose=True,
                               rng=None, on_progress=None, base_state=None):
        """Repite la simulación n_sims veces y acumula probabilidades."""
        if verbose:
            print(f"\n🎲 Iniciando Monte Carlo con {n_sims} simulaciones...")

        rng = rng if rng is not None else np.random.default_rng()
        played, remaining = self.split_played(df, start_date)
        base = base_state if base_state is not None else self.state_from_played(played)

        champions = []
        top4 = defaultdict(int)
        relegation = defaultdict(int)
        all_final_points = defaultdict(list)
        all_gf = defaultdict(list)
        all_ga = defaultdict(list)

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
                all_gf[row.Team].append(row.GF)
                all_ga[row.Team].append(row.GA)

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
            'mean_goals': {
                team: {'gf': np.mean(all_gf[team]), 'ga': np.mean(all_ga[team])} for team in all_gf
            },
        }

        if verbose:
            print(f"✓ Monte Carlo completado ({n_sims} simulaciones)")

        return results


def build_current_state(league_code: str, season: str | None = None):
    """Estado de la liga hoy, con la MISMA historia con la que se entrena.

    Recorre las temporadas de entrenamiento anteriores (conservando el H2H y
    reseteando acumuladores en cada cambio, como `build_training_matrix`) y
    después los partidos ya jugados de la temporada actual. Devuelve
    (estado, df de la temporada actual con jugados y pendientes).
    """
    from core.dataset import load_season, load_seasons, played_only

    season = season or get_current_season()
    actual = load_season(league_code, season)
    anteriores = [s for s in get_training_seasons(league_code) if s != season]

    historia = played_only(load_seasons(league_code, anteriores)) if anteriores else actual.iloc[0:0]
    jugados = played_only(actual)
    state = build_state(pd.concat([historia, jugados], ignore_index=True))
    if jugados.empty:
        state.start_new_season()  # pretemporada: nadie arrastra los puntos de la anterior

    equipos = set(actual["HomeTeam"]) | set(actual["AwayTeam"])
    state.keep_only(equipos)
    for equipo in equipos:
        state.team(equipo)  # los ascendidos existen aunque aún no hayan jugado
    return state, actual


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

    # Después de las comprobaciones: cargar el historial cuesta (y puede descargar)
    base_state, df = build_current_state(league_code, season)

    rng = np.random.default_rng(seed)

    # Todo sale del mismo Monte Carlo: antes los puntos venían de UNA simulación
    # suelta y las probabilidades de otras 500, y un equipo podía salir 4.º con
    # más probabilidad de título que el 3.º.
    mc_results = SeasonSimulator().monte_carlo_simulation(
        df, predictor, n_sims=n_sims, verbose=False, rng=rng, on_progress=on_progress,
        base_state=base_state,
    )
    champion_probs = mc_results['champion_probabilities']
    relegation_probs = mc_results['relegation_probabilities']
    puntos = mc_results['final_points_distribution']
    goles = mc_results['mean_goals']

    standings_list = [
        {
            'name': team,
            'points': round(float(puntos[team]['mean'])),
            'points_std': float(puntos[team]['std']),
            'gf': round(float(goles[team]['gf'])),
            'ga': round(float(goles[team]['ga'])),
            # 0.0 es un resultado ("no ganó ninguna simulación"), no un hueco
            'champion_prob': champion_probs.get(team, 0.0),
            'relegation_prob': relegation_probs.get(team, 0.0),
        }
        for team in puntos
    ]
    # Puntos esperados sin redondear, y la diferencia de goles como desempate
    standings_list.sort(
        key=lambda t: (puntos[t['name']]['mean'], goles[t['name']]['gf'] - goles[t['name']]['ga']),
        reverse=True,
    )
    return standings_list
