"""Instancia única de Jinja2.

Antes cada router creaba la suya (cuatro entornos con cachés independientes) y
la ruta era relativa al directorio de trabajo. Aquí se resuelve desde __file__
y se expone la configuración de la liga como global, para que la cabecera y la
barra de estado salgan de config/leagues.yaml y no estén escritas a mano.
"""
from pathlib import Path

from fastapi.templating import Jinja2Templates

from app.config import LEAGUE_CODE, get_current_season, get_league_config, season_label
from app.escudos import escudo_svg
from core.equipos import nombre_visible

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _temporadas(codigos: str) -> str:
    """'2122,2223' (como se guarda en model_runs.seasons_used) -> '21/22 · 22/23'."""
    return " · ".join(season_label(c) for c in codigos.split(",") if c)


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["liga"] = get_league_config(LEAGUE_CODE)
templates.env.globals["league_code"] = LEAGUE_CODE
templates.env.globals["temporada_actual"] = season_label(get_current_season())
templates.env.filters["temporadas"] = _temporadas
templates.env.filters["equipo"] = nombre_visible
templates.env.globals["escudo"] = escudo_svg

# Versión de los estáticos para las URL (?v=…): sin ella el navegador sigue
# usando el CSS/JS antiguo de su caché después de un cambio.
templates.env.globals["version_estaticos"] = max(
    int(f.stat().st_mtime) for f in (Path(__file__).parent / "static").rglob("*") if f.is_file()
)
