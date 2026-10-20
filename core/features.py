"""Construcción de la matriz de entrenamiento a partir de LeagueState.

Sustituye a `FeatureEngineer.calculate_rolling_features`, que escribía las 35
columnas celda a celda con `df.at[idx, col] = ...` sobre un DataFrame que iba
creciendo (lento y frágil bajo pandas 3).
"""
import numpy as np
import pandas as pd

from core.state import LeagueState

# Orden canónico de las features. El modelo se entrena y predice con este orden,
# así que cambiarlo invalida los .pkl guardados: hay que reentrenar.
FEATURE_COLS = [
    # Elo (arrastrado entre temporadas con regresión a la media)
    "Home_Elo", "Away_Elo", "Elo_Diff",

    # Forma continua: medias exponenciales sin reinicio de temporada
    "Home_EW_Goals_For", "Home_EW_Goals_Against",
    "Home_EW_SoT_For", "Home_EW_SoT_Against",
    "Away_EW_Goals_For", "Away_EW_Goals_Against",
    "Away_EW_SoT_For", "Away_EW_SoT_Against",
    "EW_Goal_Diff", "EW_SoT_Diff",
]

TARGET_MAP = {"H": 1, "D": 0, "A": -1}


def feature_row(state: LeagueState, home_team: str, away_team: str, current_date) -> pd.DataFrame:
    """Una fila con las features de un partido, en el orden que espera el modelo."""
    return pd.DataFrame([state.features(home_team, away_team, current_date)], columns=FEATURE_COLS)


def feature_vector(state: LeagueState, home_team: str, away_team: str, current_date) -> np.ndarray:
    """Igual que `feature_row` pero como array 1×len(FEATURE_COLS).

    Es lo que usa el simulador: construir un DataFrame por partido hacía que la
    conversión a XGBoost costara más que la propia predicción.
    """
    f = state.features(home_team, away_team, current_date)
    return np.array([[f[col] for col in FEATURE_COLS]], dtype=np.float32)


def _recorrer(df: pd.DataFrame, state: LeagueState, reset_per_season: bool):
    """Recorre partidos jugados en orden, cediendo (fila, estado) ANTES de
    aplicar cada partido al estado. Es el único sitio que decide cuándo se
    cambia de temporada, para que entrenar y predecir vean el mismo estado.
    """
    temporada_actual = None
    for row in df.itertuples(index=False):
        if reset_per_season:
            temporada = getattr(row, "Season", None)
            if temporada_actual is not None and temporada != temporada_actual:
                state.start_new_season()
            temporada_actual = temporada

        yield row, state
        state.apply_match(
            row.HomeTeam,
            row.AwayTeam,
            int(row.Home_Goals),
            int(row.Away_Goals),
            row.Result,
            row.Date,
            getattr(row, "Home_ShotsOnTarget", None),
            getattr(row, "Away_ShotsOnTarget", None),
        )


def _ordenar(df: pd.DataFrame) -> pd.DataFrame:
    # mergesort = estable. Con quicksort (el defecto), dos ordenaciones del
    # mismo dataframe barajan de forma distinta los partidos de una misma
    # fecha, y las features de esos partidos salían diferentes en cada pasada.
    return df.copy().sort_values("Date", kind="mergesort").reset_index(drop=True)


def build_state(df: pd.DataFrame, reset_per_season: bool = True) -> LeagueState:
    """Estado tras aplicar todos los partidos JUGADOS de `df`, con la misma
    semántica de cambio de temporada que `build_training_matrix`.

    Es lo que hay que usar para predecir o simular: un estado construido solo
    con la temporada actual no tiene el Elo ni la forma que el modelo sí vio
    al entrenar.
    """
    state = LeagueState()
    for _ in _recorrer(_ordenar(df), state, reset_per_season):
        pass
    return state


def build_training_matrix(df: pd.DataFrame, reset_per_season: bool = True):
    """Recorre los partidos en orden y genera (X, y, df).

    Cada fila se calcula ANTES de aplicar su propio partido al estado: ahí está
    la garantía de que no hay data leakage.
    """
    df = _ordenar(df)
    filas = [
        state.features(row.HomeTeam, row.AwayTeam, row.Date)
        for row, state in _recorrer(df, LeagueState(), reset_per_season)
    ]

    X = pd.DataFrame(filas, columns=FEATURE_COLS).fillna(0)
    y = df["Result"].map(TARGET_MAP).rename("Target")

    df = pd.concat([df, X], axis=1)
    df["Target"] = y
    return X, y, df
