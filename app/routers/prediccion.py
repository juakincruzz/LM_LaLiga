"""Fragmentos HTMX de predicción: próxima jornada y «cara a cara».

Los errores esperables (sin modelo, sin datos, equipo no válido) se devuelven
como el propio fragmento con un mensaje, con estado 200: HTMX no sustituye el
contenido en respuestas 4xx/5xx, y el usuario se quedaría sin saber qué pasa.
"""
from datetime import date

from fastapi import APIRouter, Depends, Request

from app.config import LEAGUE_CODE
from app.dependencies import get_db_dependency
from app.services.history_service import historial_async
from app.services.prediction_service import jornada_async, predecir_partido_async
from app.templating import templates
from core.errors import LMLaLigaError

router = APIRouter(prefix="/prediccion", tags=["prediccion"])


async def _fiabilidad(db) -> dict | None:
    """Precisión del último modelo y del mercado, para poner la predicción en contexto."""
    cursor = await db.execute(
        """
        SELECT accuracy, baseline_bookmaker, n_test
          FROM model_runs
         WHERE league_code = ? AND status = 'completed' AND baseline_home IS NOT NULL
         ORDER BY created_at DESC
         LIMIT 1
        """,
        (LEAGUE_CODE,),
    )
    fila = await cursor.fetchone()
    return dict(fila) if fila else None


def _mensaje(e: Exception) -> str:
    if isinstance(e, FileNotFoundError):
        return "Todavía no hay datos de la temporada. Pulsa «Actualizar datos»."
    return str(e)


@router.get("/jornada")
async def ver_jornada(request: Request, n: int | None = None, db=Depends(get_db_dependency)):
    contexto = {"fiabilidad": await _fiabilidad(db)}
    try:
        contexto["jornada"] = await jornada_async(LEAGUE_CODE, n)
    except (LMLaLigaError, FileNotFoundError) as e:
        contexto["error"] = _mensaje(e)
    return templates.TemplateResponse(request, "partials/jornada.html", contexto)


@router.get("/partido")
async def ver_partido(
    request: Request,
    local: str,
    visitante: str,
    fecha: date | None = None,
    db=Depends(get_db_dependency),
):
    contexto = {"fiabilidad": await _fiabilidad(db)}
    try:
        contexto["p"] = await predecir_partido_async(LEAGUE_CODE, local, visitante, fecha)
    except (LMLaLigaError, FileNotFoundError) as e:
        contexto["error"] = _mensaje(e)
    return templates.TemplateResponse(request, "partials/prediccion.html", contexto)


@router.get("/historial")
async def ver_historial(request: Request):
    """Aciertos del modelo jornada a jornada en la temporada en curso."""
    contexto = {}
    try:
        contexto["h"] = await historial_async(LEAGUE_CODE)
    except (LMLaLigaError, FileNotFoundError) as e:
        contexto["error"] = _mensaje(e)
    return templates.TemplateResponse(request, "partials/historial.html", contexto)
