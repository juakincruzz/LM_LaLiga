from pathlib import Path
from typing import Any, Dict

import yaml

from core.errors import UnknownLeagueError

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "leagues.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _RAW_CONFIG: Dict[str, Any] = yaml.safe_load(f)

DEFAULTS: Dict[str, Any] = _RAW_CONFIG.pop("defaults", {})
LEAGUE_CONFIG: Dict[str, Any] = _RAW_CONFIG

# La única liga del proyecto. Sigue siendo un código de texto porque es la
# clave foránea de todas las tablas, el directorio de data/raw/ y el nombre del
# .pkl; centralizarlo aquí evita escribir "SP1" a mano por el código.
LEAGUE_CODE = "SP1"

# Temporada 2020-21: estadios vacíos por la pandemia distorsionaron la ventaja
# de campo, así que se excluye del entrenamiento (sus partidos sí cuentan para
# el Elo y la forma: ver core.features.build_training_matrix).
EXCLUDED_SEASONS = {"2021"}


def get_league_config(league_code: str = LEAGUE_CODE) -> Dict[str, Any]:
    if league_code not in LEAGUE_CONFIG:
        raise UnknownLeagueError(f"Liga desconocida: {league_code}")
    return LEAGUE_CONFIG[league_code]


def list_leagues() -> list:
    return [{"code": code, **cfg} for code, cfg in LEAGUE_CONFIG.items()]


def season_label(code: str) -> str:
    """'2526' -> '25/26', para mostrar en la interfaz."""
    return f"{code[:2]}/{code[2:]}"


def season_long(code: str) -> str:
    """'2627' -> '2026-27', el formato de las carpetas de openfootball."""
    return f"20{code[:2]}-{code[2:]}"


def get_current_season() -> str:
    """Única fuente de verdad para "qué temporada es la actual".

    Antes "2526" estaba escrito a mano en 9 sitios distintos; cambiar de
    temporada significaba editarlos todos.
    """
    return DEFAULTS.get("current_season", "2526")


def previous_season(code: str) -> str:
    """'2526' -> '2425'. Los códigos de football-data.co.uk son mecánicos:
    dos años de dos dígitos concatenados, cada uno un año anterior."""
    y1, y2 = int(code[:2]), int(code[2:])
    return f"{(y1 - 1) % 100:02d}{(y2 - 1) % 100:02d}"


def get_training_seasons(league_code: str | None = None) -> list[str]:
    """Temporadas a usar para entrenar, de la más antigua a la actual.

    Desde `first_season` hasta la actual: con más historia el modelo mejora
    (estudio de fiabilidad, docs/ROADMAP.md §2). `league_code` no se usa: se
    acepta para no cambiar la firma que usan los servicios.
    """
    primera = DEFAULTS.get("first_season", "1415")
    seasons = []
    code = get_current_season()
    while True:
        seasons.append(code)
        if code == primera or len(seasons) > 50:
            break
        code = previous_season(code)
    return list(reversed(seasons))


def is_excluded(season: str) -> bool:
    """Temporadas cuyos partidos no se usan como ejemplos de entrenamiento."""
    return season in EXCLUDED_SEASONS
