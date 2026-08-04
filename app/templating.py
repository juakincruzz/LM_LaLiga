"""Instancia única de Jinja2.

Antes cada router creaba la suya (cuatro entornos con cachés independientes) y
la ruta era relativa al directorio de trabajo. Aquí se resuelve desde __file__
y se expone `ligas` como global, para que el menú salga de config/leagues.yaml
en vez de estar escrito a mano en base.html.
"""
from pathlib import Path

from fastapi.templating import Jinja2Templates

from app.config import list_leagues

TEMPLATES_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["ligas"] = list_leagues()
