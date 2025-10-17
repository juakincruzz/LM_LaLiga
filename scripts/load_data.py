import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataLoader:
    """Clase para cargar y combinar datos de La Liga"""
    
    def __init__(self, raw_data_path='data/raw'):
        self.raw_data_path = raw_data_path
        self.df = None
    
    def load_csv_files(self):
        """Carga los archivos CSV disponibles"""
        print("🔄 Cargando archivos CSV...")
        
        dfs = []
        
        # Cargar SP1.csv
        sp1_path = os.path.join(self.raw_data_path, 'SP1.csv')
        if os.path.exists(sp1_path):
            print(f"✓ Cargando {sp1_path}")
            df_sp1 = pd.read_csv(sp1_path)
            # Seleccionar solo columnas relevantes
            cols_sp1 = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                        'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY']
            df_sp1 = df_sp1[cols_sp1].copy()
            df_sp1.rename(columns={
                'FTHG': 'Home_Goals',
                'FTAG': 'Away_Goals',
                'FTR': 'Result',
                'HS': 'Home_Shots',
                'AS': 'Away_Shots',
                'HST': 'Home_ShotsOnTarget',
                'AST': 'Away_ShotsOnTarget',
                'HF': 'Home_Fouls',
                'AF': 'Away_Fouls',
                'HC': 'Home_Corners',
                'AC': 'Away_Corners',
                'HY': 'Home_Yellows',
                'AY': 'Away_Yellows'
            }, inplace=True)
            dfs.append(df_sp1)
        
        # Cargar football_matches.csv
        fm_path = os.path.join(self.raw_data_path, 'football_matches.csv')
        if os.path.exists(fm_path):
            print(f"✓ Cargando {fm_path}")
            df_fm = pd.read_csv(fm_path)
            # Filtrar solo La Liga
            df_fm = df_fm[df_fm['League'] == 'La Liga'].copy()
            # Seleccionar y renombrar columnas
            cols_fm = ['Date', 'Home_Team', 'Away_Team', 'Home_Team_Score', 
                       'Away_Team_Score', 'Home_Team_Shots', 'Away_Team_Shots',
                       'Home_Team_ShotsonTarget', 'Away_Team_ShotsonTarget',
                       'Home_Team_Fouls', 'Away_Team_Fouls', 'Home_Team_Corner', 'Away_Team_Corner',
                       'Home_Team_Yellowcard', 'Away_Team_Yellowcard']
            df_fm = df_fm[cols_fm].copy()
            df_fm.rename(columns={
                'Home_Team': 'HomeTeam',
                'Away_Team': 'AwayTeam',
                'Home_Team_Score': 'Home_Goals',
                'Away_Team_Score': 'Away_Goals',
                'Home_Team_Shots': 'Home_Shots',
                'Away_Team_Shots': 'Away_Shots',
                'Home_Team_ShotsonTarget': 'Home_ShotsOnTarget',
                'Away_Team_ShotsonTarget': 'Away_ShotsOnTarget',
                'Home_Team_Fouls': 'Home_Fouls',
                'Away_Team_Fouls': 'Away_Fouls',
                'Home_Team_Corner': 'Home_Corners',
                'Away_Team_Corner': 'Away_Corners',
                'Home_Team_Yellowcard': 'Home_Yellows',
                'Away_Team_Yellowcard': 'Away_Yellows'
            }, inplace=True)
            # Agregar columna de resultado
            df_fm['Result'] = df_fm.apply(lambda row: 'H' if row['Home_Goals'] > row['Away_Goals'] 
                                          else ('A' if row['Home_Goals'] < row['Away_Goals'] else 'D'), axis=1)
            dfs.append(df_fm)
        
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            print(f"✓ Total de partidos cargados: {len(self.df)}")
            return self.df
        else:
            print("❌ No se encontraron archivos CSV")
            return None
    
    def clean_data(self):
        """Limpia y prepara los datos"""
        if self.df is None:
            print("❌ No hay datos cargados")
            return None
        
        print("\n🧹 Limpiando datos...")
        
        # Convertir fecha a datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        # Eliminar filas con valores nulos en columnas críticas
        self.df = self.df.dropna(subset=['HomeTeam', 'AwayTeam', 'Home_Goals', 'Away_Goals', 'Result'])
        
        # Convertir columnas numéricas
        numeric_cols = ['Home_Goals', 'Away_Goals', 'Home_Shots', 'Away_Shots', 
                       'Home_ShotsOnTarget', 'Away_ShotsOnTarget', 'Home_Fouls', 
                       'Away_Fouls', 'Home_Corners', 'Away_Corners', 'Home_Yellows', 'Away_Yellows']
        
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Rellenar NaN en columnas numéricas con 0
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
        
        # Eliminar duplicados
        self.df = self.df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first')
        
        # Ordenar por fecha
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        print(f"✓ Datos limpios. Total de partidos: {len(self.df)}")
        print(f"✓ Fechas: {self.df['Date'].min()} a {self.df['Date'].max()}")
        
        return self.df
    
    def save_processed_data(self, output_path='data/processed/laliga_processed.csv'):
        """Guarda los datos procesados"""
        if self.df is None:
            print("❌ No hay datos para guardar")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✓ Datos guardados en {output_path}")
    
    def get_summary(self):
        """Obtiene un resumen de los datos"""
        if self.df is None:
            return None
        
        print("\n📊 RESUMEN DE DATOS:")
        print(f"Total de partidos: {len(self.df)}")
        print(f"Equipos únicos: {len(set(list(self.df['HomeTeam'].unique()) + list(self.df['AwayTeam'].unique())))}")
        print(f"Fecha más antigua: {self.df['Date'].min()}")
        print(f"Fecha más reciente: {self.df['Date'].max()}")
        print(f"\nDistribución de resultados:")
        print(self.df['Result'].value_counts())
        print(f"\nPromedio de goles:")
        print(f"  - Casa: {self.df['Home_Goals'].mean():.2f}")
        print(f"  - Fuera: {self.df['Away_Goals'].mean():.2f}")


if __name__ == "__main__":
    loader = DataLoader()
    loader.load_csv_files()
    loader.clean_data()
    loader.save_processed_data()
    loader.get_summary()