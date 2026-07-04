"""
Módulo de gestión de datos para LM_LaLiga.

Se encarga de verificar que existan datos locales y, si faltan,
descargar la temporada actual desde football-data.org.
"""

import os
import pandas as pd
from scripts.fetch_updated_data import FootballDataFetcher


def _local_csv_exists(raw_path='data/raw'):
    """Comprueba si existe al menos un CSV de datos en la ruta indicada."""
    sp1_path = os.path.join(raw_path, 'SP1.csv')
    fm_path = os.path.join(raw_path, 'football_matches.csv')
    return os.path.exists(sp1_path) or os.path.exists(fm_path)


def ensure_raw_data(raw_path='data/raw', season='2526', force_download=False):
    """
    Asegura que existan datos locales para entrenar el modelo.

    Si no hay archivos CSV en `raw_path` o se fuerza la descarga,
    descarga la temporada indicada desde football-data.org y la guarda
    como `data/raw/SP1.csv`.

    Parámetros
    ----------
    raw_path : str
        Carpeta donde se buscan/guardan los CSVs.
    season : str
        Código de temporada usado por football-data.org (ej. '2526').
    force_download : bool
        Si es True, descarga los datos aunque ya existan locales.

    Retorna
    -------
    bool
        True si hay datos disponibles, False en caso contrario.
    """
    os.makedirs(raw_path, exist_ok=True)

    has_local = _local_csv_exists(raw_path)

    if has_local and not force_download:
        print("✓ Datos locales encontrados")
        return True

    if force_download:
        print("🔄 Forzando descarga de datos actualizados...")
    else:
        print("⚠️  No se encontraron datos locales. Descargando desde football-data.org...")

    fetcher = FootballDataFetcher()
    df = fetcher.fetch_from_football_data_org_latest()

    if df is None or len(df) == 0:
        print("❌ No se pudieron descargar datos de football-data.org")
        return False

    output_path = os.path.join(raw_path, 'SP1.csv')
    fetcher.save_updated_data(df, output_path)
    return True


def load_latest_raw_data(raw_path='data/raw'):
    """
    Carga el CSV más reciente disponible: SP1.csv o football_matches.csv.

    Retorna
    -------
    pd.DataFrame | None
        DataFrame con los datos crudos, o None si no hay archivos.
    """
    sp1_path = os.path.join(raw_path, 'SP1.csv')
    fm_path = os.path.join(raw_path, 'football_matches.csv')

    if os.path.exists(sp1_path):
        print(f"✓ Cargando datos desde {sp1_path}")
        return pd.read_csv(sp1_path)
    elif os.path.exists(fm_path):
        print(f"✓ Cargando datos desde {fm_path}")
        return pd.read_csv(fm_path)
    else:
        print("❌ No se encontró ningún archivo de datos local")
        return None


if __name__ == "__main__":
    success = ensure_raw_data(force_download=False)
    if success:
        df = load_latest_raw_data()
        if df is not None:
            print(f"✓ Datos listos: {len(df)} registros")
