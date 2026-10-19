"""
Configuración de fuentes de datos confiables para La Liga
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Football-Data.org API
FOOTBALL_DATA_API = {
    'base_url': 'https://api.football-data.org/v4',
    'api_key': os.getenv('FOOTBALL_DATA_API_KEY'),
    'league': 'PD',  # La Liga española (Primera División)
    'headers': {
        'X-Auth-Token': os.getenv('FOOTBALL_DATA_API_KEY')
    }
}

# RapidAPI Football
RAPIDAPI_FOOTBALL = {
    'base_url': 'https://api-football-v1.p.rapidapi.com/v3',
    'api_key': os.getenv('RAPIDAPI_KEY'),
    'host': 'api-football-v1.p.rapidapi.com',
    'headers': {
        'x-rapidapi-key': os.getenv('RAPIDAPI_KEY'),
        'x-rapidapi-host': 'api-football-v1.p.rapidapi.com'
    }
}

# Understat (requiere scraping especial o contacto)
UNDERSTAT_CONFIG = {
    'base_url': 'https://understat.com/api/v1',
    'requires_browser': True  # Requiere Selenium o similar
}

# Configuración de datos
DATA_SOURCES = {
    'primary': 'football_data',      # Principal
    'secondary': 'rapidapi',         # Secundaria para actualizaciones
    'advanced_stats': 'understat'    # Opcional: estadísticas avanzadas
}