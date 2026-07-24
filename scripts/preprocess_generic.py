"""Generic preprocessing for any league.

Provisional implementation — Task 13 will provide the definitive generic pipeline.
For now, loads league CSV via data_manager_generic and reuses FeatureEngineer.
"""

import pandas as pd
from scripts.data_manager_generic import ensure_league_data
from scripts.preprocess import FeatureEngineer


def preprocess_league(league_code: str, season: str = "2526"):
    """Load and preprocess league data, returning X, y."""
    csv_path = ensure_league_data(league_code, season)
    df = pd.read_csv(csv_path)

    # Standardize football-data.co.uk column names
    rename_map = {
        "FTHG": "Home_Goals",
        "FTAG": "Away_Goals",
        "FTR": "Result",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parse dates (football-data.co.uk uses DD/MM/YYYY)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Keep only played matches with complete data
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "Home_Goals", "Away_Goals", "Result"])
    df["Home_Goals"] = pd.to_numeric(df["Home_Goals"], errors="coerce")
    df["Away_Goals"] = pd.to_numeric(df["Away_Goals"], errors="coerce")
    df = df.dropna(subset=["Home_Goals", "Away_Goals"])

    if df.empty:
        raise ValueError(f"No valid played matches found for {league_code}/{season}")

    engineer = FeatureEngineer(df)
    X, y, _ = engineer.prepare_features()
    return X, y
