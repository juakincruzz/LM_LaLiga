"""Errores de dominio, con mensajes en español pensados para verse en la UI.

Los jobs guardan `str(e)` en `job_status.log`, así que el texto de estas
excepciones es literalmente lo que lee el usuario cuando algo falla.
"""


class LMLaLigaError(Exception):
    """Base de los errores propios del proyecto."""


class UnknownLeagueError(LMLaLigaError, ValueError):
    """Código de liga que no está en config/leagues.yaml."""


class SeasonFinishedError(LMLaLigaError):
    """No quedan partidos por jugar, así que no hay nada que simular."""


class ModelNotTrainedError(LMLaLigaError):
    """No hay modelo entrenado para esta liga."""


class SeasonInProgressError(LMLaLigaError):
    """La temporada aún no ha terminado: proyectar la siguiente no tiene sentido."""


class PromotedTeamsNotConfiguredError(LMLaLigaError):
    """Los ascendidos de config/leagues.yaml siguen siendo los provisionales."""
