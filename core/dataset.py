"""Carga y normalización de los CSV de football-data.co.uk.

Punto único de entrada de datos crudos. Antes existían cuatro mapas de
renombrado distintos (uno de ellos con solo tres entradas, que reventaba con
KeyError en CSV sin FTR) y tres implementaciones de "derivar Result de los
goles"; todo eso vive aquí ahora.
"""
from pathlib import Path

import pandas as pd

from scripts.data_manager_generic import ensure_league_data, league_csv_path

# Nombres de football-data.co.uk → nombres internos.
COLUMN_MAP = {
    "FTHG": "Home_Goals",
    "FTAG": "Away_Goals",
    "FTR": "Result",
    "HTHG": "Home_Goals_HT",
    "HTAG": "Away_Goals_HT",
    "HTR": "Result_HT",
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
    "HR": "Home_Reds",
    "AR": "Away_Reds",
}

# Cuotas de Bet365, usadas como baseline de referencia en la evaluación.
ODDS_COLUMNS = {"B365H": "Odds_Home", "B365D": "Odds_Draw", "B365A": "Odds_Away"}

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam"]


def normalize_frame(df: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Normaliza un CSV crudo: nombres, fechas, tipos y columna Result.

    Devuelve tanto partidos jugados como pendientes; filtrar es cosa de quien
    llama (entrenar necesita jugados, simular necesita los pendientes).
    """
    df = df.rename(columns={k: v for k, v in {**COLUMN_MAP, **ODDS_COLUMNS}.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"El CSV no tiene las columnas {missing}; ¿se descargó una página de error?")

    # football-data.co.uk usa DD/MM/YYYY
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    for col in ("Home_Goals", "Away_Goals"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derivar Result de los goles cuando falte (única implementación)
    if "Result" not in df.columns:
        df["Result"] = pd.NA
    if {"Home_Goals", "Away_Goals"} <= set(df.columns):
        derivar = df["Result"].isna() & df["Home_Goals"].notna() & df["Away_Goals"].notna()
        if derivar.any():
            diff = df.loc[derivar, "Home_Goals"] - df.loc[derivar, "Away_Goals"]
            df.loc[derivar, "Result"] = diff.map(lambda d: "H" if d > 0 else ("A" if d < 0 else "D"))

    # .copy() consolida el frame: sin él, pandas avisa de fragmentación al
    # añadir Season sobre las ~120 columnas del CSV original.
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True).copy()
    if season is not None:
        df["Season"] = season

    return df


def read_csv(path: Path) -> pd.DataFrame:
    """Lee un CSV de football-data.co.uk. `utf-8-sig` quita el BOM que, si no,
    hace que la primera columna se llame '﻿Div'."""
    return pd.read_csv(path, encoding="utf-8-sig")


def load_season(league_code: str, season: str, download: bool = True) -> pd.DataFrame:
    """Carga una temporada normalizada, descargándola si falta."""
    path = ensure_league_data(league_code, season) if download else league_csv_path(league_code, season)
    if not path.exists():
        raise FileNotFoundError(f"No hay datos en {path}. Pulsa 'Actualizar datos' primero.")
    return normalize_frame(read_csv(path), season=season)


def load_seasons(league_code: str, seasons: list[str], download: bool = True) -> pd.DataFrame:
    """Carga y concatena varias temporadas, ordenadas cronológicamente.

    Las temporadas de football-data.co.uk no se solapan en fechas, así que
    ordenar por Date tras concatenar ya agrupa cada temporada en bloque sin
    necesidad de tratamiento especial.
    """
    marcos = [load_season(league_code, season, download=download) for season in seasons]
    return (
        pd.concat(marcos, ignore_index=True)
        .sort_values("Date", kind="mergesort")
        .reset_index(drop=True)
    )


def played_only(df: pd.DataFrame) -> pd.DataFrame:
    """Solo partidos con resultado, para entrenar."""
    return df.dropna(subset=["Home_Goals", "Away_Goals", "Result"]).reset_index(drop=True)
