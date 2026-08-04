"""Generic data manager supporting multiple leagues."""
from pathlib import Path

import requests

from app.config import get_league_config

DATA_DIR = Path("data/raw")


def league_csv_path(league_code: str, season: str) -> Path:
    """Ruta del CSV de una liga/temporada.

    Punto único donde se construye la ruta: resuelve `DATA_DIR` en tiempo de
    llamada, de forma que los tests puedan redirigirla parcheando la constante.
    """
    return DATA_DIR / league_code / f"{season}.csv"


def download_league_data(league_code: str, season: str) -> Path:
    """
    Download CSV for a league and season from football-data.co.uk.
    Season format: '2526' for 2025-26.
    """
    config = get_league_config(league_code)
    data_code = config["data_code"]
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{data_code}.csv"

    file_path = league_csv_path(league_code, season)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_path


def ensure_league_data(league_code: str, season: str) -> Path:
    """Download if not present locally."""
    file_path = league_csv_path(league_code, season)
    if not file_path.exists():
        return download_league_data(league_code, season)
    return file_path
