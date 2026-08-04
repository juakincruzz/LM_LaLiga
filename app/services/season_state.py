"""Estado de la temporada de una liga, deducido de la tabla `matches`.

Sustituye al antiguo `is_offseason()`, que preguntaba por la fecha del último
partido: como `matches` nunca se rellenaba, devolvía siempre True y todas las
ligas mostraban el cartel de off-season de forma permanente.
"""
from dataclasses import dataclass
from enum import Enum

import aiosqlite


class EstadoTemporada(str, Enum):
    SIN_DATOS = "sin_datos"
    PRETEMPORADA = "pretemporada"
    EN_CURSO = "en_curso"
    FINALIZADA = "finalizada"


ETIQUETAS = {
    EstadoTemporada.SIN_DATOS: "Sin datos",
    EstadoTemporada.PRETEMPORADA: "Pretemporada",
    EstadoTemporada.EN_CURSO: "Temporada en curso",
    EstadoTemporada.FINALIZADA: "Temporada finalizada",
}


@dataclass
class ResumenTemporada:
    estado: EstadoTemporada
    total: int
    jugados: int
    pendientes: int
    ultimo_partido: str | None
    season: str | None

    @property
    def etiqueta(self) -> str:
        return ETIQUETAS[self.estado]

    @property
    def puede_simular(self) -> bool:
        """Simular solo tiene sentido si quedan partidos por jugar."""
        return self.pendientes > 0

    @property
    def es_offseason(self) -> bool:
        return self.estado in (EstadoTemporada.FINALIZADA, EstadoTemporada.SIN_DATOS)


async def get_season_state(db: aiosqlite.Connection, league_code: str) -> ResumenTemporada:
    cursor = await db.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN result IS NOT NULL THEN 1 ELSE 0 END) AS jugados,
               MAX(CASE WHEN result IS NOT NULL THEN date END) AS ultimo,
               MAX(season) AS season
          FROM matches
         WHERE league_code = ?
        """,
        (league_code,),
    )
    row = await cursor.fetchone()

    total = row["total"] or 0
    jugados = row["jugados"] or 0
    pendientes = total - jugados

    if total == 0:
        estado = EstadoTemporada.SIN_DATOS
    elif jugados == 0:
        estado = EstadoTemporada.PRETEMPORADA
    elif pendientes == 0:
        estado = EstadoTemporada.FINALIZADA
    else:
        estado = EstadoTemporada.EN_CURSO

    return ResumenTemporada(
        estado=estado,
        total=total,
        jugados=jugados,
        pendientes=pendientes,
        ultimo_partido=row["ultimo"],
        season=row["season"],
    )
