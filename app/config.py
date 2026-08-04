from pathlib import Path
from typing import Any, Dict

import yaml

from core.errors import UnknownLeagueError

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "leagues.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _RAW_CONFIG: Dict[str, Any] = yaml.safe_load(f)

DEFAULTS: Dict[str, Any] = _RAW_CONFIG.pop("defaults", {})
LEAGUE_CONFIG: Dict[str, Any] = _RAW_CONFIG

# Temporada 2020-21: estadios vacíos por la pandemia distorsionaron la ventaja
# de campo (una de las 35 features), así que se excluye del entrenamiento
# aunque caiga dentro de la profundidad histórica configurada.
EXCLUDED_SEASONS = {"2021"}


def get_league_config(league_code: str) -> Dict[str, Any]:
    if league_code not in LEAGUE_CONFIG:
        raise UnknownLeagueError(f"Liga desconocida: {league_code}")
    return LEAGUE_CONFIG[league_code]


def list_leagues() -> list:
    return [{"code": code, **cfg} for code, cfg in LEAGUE_CONFIG.items()]


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

    `history_depth` temporadas anteriores + la actual, salvo las excluidas
    (ver EXCLUDED_SEASONS). `league_code` no se usa hoy — todas las ligas
    comparten la misma profundidad — pero se acepta para no tener que tocar
    la firma el día que una liga necesite una distinta.
    """
    depth = DEFAULTS.get("history_depth", 5)
    seasons = []
    code = get_current_season()
    for _ in range(depth + 1):
        if code not in EXCLUDED_SEASONS:
            seasons.append(code)
        code = previous_season(code)
    return list(reversed(seasons))
