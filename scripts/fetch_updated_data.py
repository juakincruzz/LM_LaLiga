"""
Script para obtener datos actualizados de La Liga desde APIs
y mantener los datos siempre al día
"""

import pandas as pd
import requests
import json
import os
from datetime import datetime, timedelta
import time

class FootballDataFetcher:
    """Clase para obtener datos actualizados de La Liga desde APIs"""
    
    def __init__(self):
        self.apis = {
            'api_football': 'https://v3.football.api-sports.io/',
            'rapidapi': 'https://api-football-v1.p.rapidapi.com/v3/',
        }
    
    def fetch_from_api_football_free(self):
        """
        Obtiene datos de api-sports.io usando RapidAPI (versión gratuita)
        Requiere API key de RapidAPI: https://rapidapi.com/api-sports/api/api-football
        """
        print("🌐 Intentando obtener datos de api-sports.io (RapidAPI)...")
        
        # Nota: Necesitas obtener tu propia key de RapidAPI
        api_key = os.getenv('RAPIDAPI_KEY')
        
        if not api_key:
            print("⚠️  RAPIDAPI_KEY no configurada")
            print("   Para obtener datos en tiempo real:")
            print("   1. Regístrate en: https://rapidapi.com/api-sports/api/api-football")
            print("   2. Copia tu API key")
            print("   3. Ejecuta: export RAPIDAPI_KEY='tu_key_aqui'")
            print("   O configura la variable de entorno en tu sistema")
            return None
        
        headers = {
            'x-rapidapi-key': api_key,
            'x-rapidapi-host': 'api-football-v1.p.rapidapi.com'
        }
        
        try:
            # La Liga ID: 302, Temporada: 2026
            url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
            params = {
                'league': 302,
                'season': 2026,
                'status': 'all'
            }
            
            print("   📡 Conectando con API...")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['results'] == 0:
                print("   ⚠️  Sin datos disponibles en la API")
                return None
            
            print(f"   ✓ Datos obtenidos: {data['results']} partidos")
            
            return self.process_api_response(data)
        
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error de conexión: {e}")
            return None
        except Exception as e:
            print(f"   ❌ Error procesando datos: {e}")
            return None
    
    def fetch_from_football_data_org_latest(self):
        """
        Obtiene datos de football-data.org
        Actualiza los datos locales con los últimos disponibles
        """
        print("🌐 Obteniendo datos de football-data.org...")
        
        try:
            # Descargar CSV de La Liga temporada 2025-2026
            url = "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"
            
            print("   📡 Descargando datos...")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            # Guardar en archivo temporal
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"   ✓ Datos obtenidos: {len(df)} partidos")
            
            # Renombrar columnas según el formato de football-data.org
            column_mapping = {
                'HomeTeam': 'HomeTeam',
                'AwayTeam': 'AwayTeam',
                'FTHG': 'Home_Goals',      # Full Time Home Goals
                'FTAG': 'Away_Goals',      # Full Time Away Goals
                'FTR': 'Result',           # Full Time Result
                'HTHG': 'HT_Home_Goals',
                'HTAG': 'HT_Away_Goals',
                'HS': 'Home_Shots',
                'AS': 'Away_Shots',
                'HST': 'Home_ShotsOnTarget',
                'AST': 'Away_ShotsOnTarget',
                'HF': 'Home_Fouls',
                'AF': 'Away_Fouls',
                'HC': 'Home_Corners',
                'AC': 'Away_Corners',
                'HY': 'Home_Yellows',
                'AY': 'Away_Yellows',
                'HR': 'Home_Reds',
                'AR': 'Away_Reds',
                'Date': 'Date'
            }
            
            # Renombrar solo las columnas que existen
            rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
            df = df.rename(columns=rename_dict)
            
            # Filtrar solo partidos jugados (con resultados)
            df_completed = df[df['Result'].notna()].copy()
            
            # Rellenar columnas faltantes con 0 si es necesario
            columns_needed = ['Home_Goals', 'Away_Goals', 'Result']
            for col in columns_needed:
                if col not in df_completed.columns:
                    df_completed[col] = 0
            
            print(f"   ✓ Partidos completados: {len(df_completed)}")
            
            return df_completed
        
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error descargando: {e}")
            return None
        except Exception as e:
            print(f"   ❌ Error procesando: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_api_response(self, data):
        """Procesa la respuesta de la API"""
        matches = []
        
        for fixture in data.get('response', []):
            # Solo incluir partidos que ya se jugaron
            if fixture['fixture']['status']['short'] not in ['FT', 'AET', 'PEN']:
                continue
            
            match = {
                'Date': fixture['fixture']['date'],
                'HomeTeam': fixture['teams']['home']['name'],
                'AwayTeam': fixture['teams']['away']['name'],
                'Home_Goals': fixture['goals']['home'],
                'Away_Goals': fixture['goals']['away'],
                'Status': fixture['fixture']['status']['short']
            }
            matches.append(match)
        
        return pd.DataFrame(matches)
    
    def save_updated_data(self, df, output_path='data/raw/laliga_updated.csv'):
        """Guarda los datos actualizados"""
        if df is None or len(df) == 0:
            print("❌ No hay datos para guardar")
            return False
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Datos guardados en {output_path}")
        print(f"  Último partido: {df['Date'].max()}")
        
        return True
    
    def merge_datasets(self, df_old, df_new):
        """Fusiona datasets antiguo y nuevo sin duplicados"""
        
        if df_new is None or len(df_new) == 0:
            print("⚠️  Usando datos antiguos")
            return df_old
        
        # Convertir fechas a datetime
        df_old['Date'] = pd.to_datetime(df_old['Date'], errors='coerce')
        df_new['Date'] = pd.to_datetime(df_new['Date'], errors='coerce')
        
        # Combinar
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        
        # Eliminar duplicados (mantener el más reciente)
        df_merged = df_merged.drop_duplicates(
            subset=['Date', 'HomeTeam', 'AwayTeam'],
            keep='last'
        )
        
        # Ordenar por fecha
        df_merged = df_merged.sort_values('Date').reset_index(drop=True)
        
        print(f"✓ Datos fusionados: {len(df_merged)} partidos únicos")
        
        return df_merged
    
    def get_today_date(self):
        """Retorna la fecha de hoy"""
        return datetime.now().strftime("%Y-%m-%d")


def main():
    fetcher = FootballDataFetcher()
    
    print("=" * 80)
    print("📊 ACTUALIZADOR DE DATOS - LA LIGA 2025-2026")
    print("=" * 80)
    print(f"📅 Fecha actual: {fetcher.get_today_date()}\n")
    
    # Opción 1: Obtener de football-data.org (sin API key)
    print("[1/3] Obteniendo datos de football-data.org...")
    df_new = fetcher.fetch_from_football_data_org_latest()
    
    # Opción 2: Obtener de RapidAPI (con API key)
    print("\n[2/3] Intentando obtener datos de RapidAPI (opcional)...")
    df_api = fetcher.fetch_from_api_football_free()
    
    if df_api is not None and len(df_api) > len(df_new):
        print("   ✓ Usando datos de RapidAPI (más actualizados)")
        df_new = df_api
    elif df_new is not None:
        print("   ✓ Usando datos de football-data.org")
    else:
        print("   ❌ Error: No se pudieron obtener datos")
        return False
    
    # Cargar datos antiguos si existen
    print("\n[3/3] Fusionando con datos anteriores...")
    
    old_file = 'data/raw/SP1.csv'
    if os.path.exists(old_file):
        df_old = pd.read_csv(old_file)
        df_merged = fetcher.merge_datasets(df_old, df_new)
    else:
        df_merged = df_new
    
    # Guardar datos actualizados
    fetcher.save_updated_data(df_merged, 'data/raw/laliga_updated.csv')
    
    print("\n" + "=" * 80)
    print("✓ ¡Datos actualizados exitosamente!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()