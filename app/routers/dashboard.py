from fastapi import APIRouter, Depends, Request

from app.config import list_leagues
from app.dependencies import get_db_dependency
from app.templating import templates

router = APIRouter()


@router.get("/")
async def dashboard(request: Request, db=Depends(get_db_dependency)):
    # Una sola consulta en vez de una por liga
    cursor = await db.execute("SELECT code, last_updated FROM leagues")
    estado_db = {row["code"]: dict(row) for row in await cursor.fetchall()}

    # Partidos jugados por liga, para que la tarjeta diga algo real
    cursor = await db.execute(
        """
        SELECT league_code,
               COUNT(*) AS total,
               SUM(CASE WHEN result IS NOT NULL THEN 1 ELSE 0 END) AS jugados
          FROM matches
         GROUP BY league_code
        """
    )
    partidos = {row["league_code"]: dict(row) for row in await cursor.fetchall()}

    leagues = []
    for league in list_leagues():
        fila = estado_db.get(league["code"], {})
        conteo = partidos.get(league["code"], {})
        leagues.append(
            {
                **league,
                "last_updated": fila.get("last_updated"),
                # 'Activa' significa que hay datos cargados, no un flag fijo en la BD
                "is_active": bool(conteo.get("total")),
                "total_partidos": conteo.get("total", 0),
                "partidos_jugados": conteo.get("jugados", 0),
            }
        )

    return templates.TemplateResponse(request, "dashboard.html", {"leagues": leagues})
