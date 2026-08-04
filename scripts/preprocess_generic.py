"""Preprocesado genérico para cualquier liga.

Fino a propósito: la carga vive en core.dataset y las features en core.features.
"""

from app.config import get_training_seasons
from core.dataset import load_seasons, played_only
from core.features import build_training_matrix


def preprocess_league(league_code: str, seasons: list[str] | None = None):
    """Carga y preprocesa una liga sobre varias temporadas. Devuelve (X, y, df_jugados).

    Por defecto entrena con `app.config.get_training_seasons` (la profundidad
    histórica configurada). `core.features.build_training_matrix` resetea los
    acumuladores en cada cambio de temporada mientras conserva el histórico de
    enfrentamientos directos (`reset_per_season=True`, su valor por defecto).
    """
    if seasons is None:
        seasons = get_training_seasons(league_code)

    df = played_only(load_seasons(league_code, seasons))
    if df.empty:
        raise ValueError(f"No hay partidos jugados en {league_code} ({', '.join(seasons)})")

    return build_training_matrix(df)
