"""Ficha de equipo: /equipo/{nombre}, con el nombre de football-data ("Ath Madrid")."""
from fastapi import APIRouter, Depends, Request

from app.config import LEAGUE_CODE
from app.dependencies import get_db_dependency
from app.services.team_service import ficha
from app.templating import templates
from core.equipos import ESCUDOS

router = APIRouter(prefix="/equipo", tags=["equipo"])


@router.get("/{nombre}")
async def ver_equipo(request: Request, nombre: str, db=Depends(get_db_dependency)):
    datos = await ficha(db, LEAGUE_CODE, nombre)
    if datos is None:
        return templates.TemplateResponse(
            request,
            "404.html",
            {"titulo": "Equipo no encontrado", "mensaje": f"«{nombre}» no juega en {templates.env.globals['liga']['name']}."},
            status_code=404,
        )
    # Colores del escudo para las camisetas del campo
    colores = (ESCUDOS.get(nombre) or {"colores": ("gris",)})["colores"]
    return templates.TemplateResponse(request, "equipo.html", {"f": datos, "colores": colores})
