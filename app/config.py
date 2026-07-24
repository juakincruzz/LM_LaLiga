import yaml
from pathlib import Path
from typing import Dict, Any

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "leagues.yaml"

with open(_CONFIG_PATH, "r") as f:
    LEAGUE_CONFIG: Dict[str, Any] = yaml.safe_load(f)


def get_league_config(league_code: str) -> Dict[str, Any]:
    if league_code not in LEAGUE_CONFIG:
        raise ValueError(f"Unknown league code: {league_code}")
    return LEAGUE_CONFIG[league_code]


def list_leagues() -> list:
    return [
        {"code": code, **cfg}
        for code, cfg in LEAGUE_CONFIG.items()
    ]
