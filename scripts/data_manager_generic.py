"""Descarga y rutas de los datos crudos.

Dos fuentes:
  - football-data.co.uk: resultados (y cuotas, tiros…) de los partidos JUGADOS.
    Su CSV no trae los partidos futuros.
  - openfootball: el calendario completo con jornadas, que es lo único que dice
    qué partidos quedan y cuál es la próxima jornada. Se guarda como CSV con los
    nombres ya traducidos a los de football-data (core/equipos.py).
"""
import csv
from datetime import date
from pathlib import Path

import requests

from app.config import get_league_config, season_long
from core.equipos import desde_openfootball

DATA_DIR = Path("data/raw")


def league_csv_path(league_code: str, season: str) -> Path:
    """Ruta del CSV de una liga/temporada.

    Punto único donde se construye la ruta: resuelve `DATA_DIR` en tiempo de
    llamada, de forma que los tests puedan redirigirla parcheando la constante.
    """
    return DATA_DIR / league_code / f"{season}.csv"


def calendar_csv_path(league_code: str, season: str) -> Path:
    """Ruta del calendario de una temporada (ver `download_calendar`)."""
    return DATA_DIR / league_code / f"{season}_calendario.csv"


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


def download_calendar(league_code: str, season: str) -> Path:
    """Descarga el calendario de openfootball y lo guarda como CSV
    (Date en DD/MM/YYYY como football-data, HomeTeam, AwayTeam, Matchday)."""
    config = get_league_config(league_code)
    url = (
        "https://raw.githubusercontent.com/openfootball/football.json/master/"
        f"{season_long(season)}/{config['calendar_code']}.json"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    filas = []
    for partido in response.json()["matches"]:
        jornada = int(str(partido["round"]).split()[-1])  # "Matchday 10" -> 10
        dia = date.fromisoformat(partido["date"])
        filas.append({
            "Date": dia.strftime("%d/%m/%Y"),
            "HomeTeam": desde_openfootball(partido["team1"]),
            "AwayTeam": desde_openfootball(partido["team2"]),
            "Matchday": jornada,
        })

    file_path = calendar_csv_path(league_code, season)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "HomeTeam", "AwayTeam", "Matchday"])
        writer.writeheader()
        writer.writerows(filas)
    return file_path
