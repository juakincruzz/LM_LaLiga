"""Generic data manager supporting multiple leagues."""
from pathlib import Path
import requests
from app.config import get_league_config

DATA_DIR = Path("data/raw")


def download_league_data(league_code: str, season: str) -> Path:
    """
    Download CSV for a league and season from football-data.co.uk.
    Season format: '2526' for 2025-26.
    """
    config = get_league_config(league_code)
    data_code = config["data_code"]
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{data_code}.csv"

    league_dir = DATA_DIR / league_code
    league_dir.mkdir(parents=True, exist_ok=True)

    file_path = league_dir / f"{season}.csv"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_path


def ensure_league_data(league_code: str, season: str) -> Path:
    """Download if not present locally."""
    file_path = DATA_DIR / league_code / f"{season}.csv"
    if not file_path.exists():
        return download_league_data(league_code, season)
    return file_path
