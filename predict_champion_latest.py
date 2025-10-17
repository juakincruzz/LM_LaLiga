"""
Predictor con datos en tiempo real desde native-stats.org
"""

import sys
import os
import pandas as pd
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.scraper_laliga import LigaScraper

def project_season_from_scraper(standings_df, matches_per_team=38):
    """Proyecta la temporada basado en datos del scraper"""
    
    standings_df = standings_df.copy()
    
    # Calcular promedio de puntos por partido
    standings_df['points_per_game'] = standings_df['Points'] / standings_df['Games']
    
    # Partidos restantes
    standings_df['games_remaining'] = matches_per_team - standings_df['Games']
    
    # Proyectar puntos finales
    standings_df['projected_points'] = standings_df['Points'] + (
        standings_df['points_per_game'] * standings_df['games_remaining']
    )
    
    # Calcular diferencia de goles
    standings_df['goal_diff'] = standings_df['GF'] - standings_df['GA']
    
    return standings_df.sort_values('projected_points', ascending=False)

def clean_team_name(team):
    """Limpia el nombre del equipo removiendo números iniciales"""
    team = str(team).strip()
    team = re.sub(r'^\d+\s*', '', team).strip()
    return team

def main():
    print("\n" + "=" * 90)
    print("🏆 PREDICTOR DE LA LIGA 2025-2026 - DATOS EN TIEMPO REAL")
    print("=" * 90)
    print(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 90)
    
    # Obtener datos
    print("\n[1/3] OBTENIENDO DATOS EN TIEMPO REAL...")
    scraper = LigaScraper()
    standings = scraper.fetch_standings()
    gameweek = scraper.get_current_gameweek()
    
    if standings is None:
        print("❌ Error: No se pudieron obtener datos")
        return False
    
    print(f"\n✓ Datos obtenidos: {len(standings)} equipos")
    print(f"✓ Jornada actual: {gameweek if gameweek else 'Desconocida'}/38")
    
    # Mostrar clasificación actual
    print("\n[2/3] CLASIFICACIÓN ACTUAL:")
    print("\n" + "-" * 90)
    print(f"{'POS':<5}{'EQUIPO':<25}{'PJ':<5}{'V':<5}{'E':<5}{'D':<5}{'GF':<5}{'GC':<5}{'DG':<5}{'PTS':<5}")
    print("-" * 90)
    
    for _, row in standings.iterrows():
        pos = int(row['Position'])
        team = clean_team_name(row['Team'])
        pj = int(row['Games'])
        w = int(row['Wins'])
        d = int(row['Draws'])
        l = int(row['Losses'])
        gf = int(row['GF']) if 'GF' in row else 0
        ga = int(row['GA']) if 'GA' in row else 0
        dg = gf - ga
        pts = int(row['Points'])
        
        print(f"{pos:<5}{team:<25}{pj:<5}{w:<5}{d:<5}{l:<5}{gf:<5}{ga:<5}{dg:<5}{pts:<5}")
    
    # Proyección
    print("\n[3/3] GENERANDO PROYECCIÓN...")
    
    try:
        projected = project_season_from_scraper(standings)
        
        # Resultados
        print("\n" + "=" * 90)
        print("🏅 PREDICCIÓN FINAL")
        print("=" * 90)
        
        champion = projected.iloc[0]
        second = projected.iloc[1] if len(projected) > 1 else None
        
        champion_team = clean_team_name(champion['Team'])
        
        print(f"\n🥇 CAMPEÓN PREDICHO: {champion_team.upper()}")
        print(f"   • Puntos actuales: {int(champion['Points'])} ({int(champion['Games'])}/38 partidos)")
        print(f"   • Puntos proyectados: {int(champion['projected_points'])}")
        print(f"   • Promedio: {champion['points_per_game']:.2f} pts/partido")
        
        if second is not None:
            gap = champion['projected_points'] - second['projected_points']
            print(f"   • Ventaja: +{gap:.0f} puntos sobre 2º")
            
            if gap >= 5:
                confidence = "MUY ALTA ⭐⭐⭐⭐⭐"
            elif gap >= 3:
                confidence = "ALTA ⭐⭐⭐⭐"
            else:
                confidence = "MEDIA ⭐⭐⭐"
        else:
            confidence = "DESCONOCIDA"
        
        print(f"   • Confianza: {confidence}")
        
        # Podio
        print(f"\n🏆 PODIO FINAL PREDICHO:")
        for pos, (_, row) in enumerate(projected.head(3).iterrows(), 1):
            team = clean_team_name(row['Team'])
            print(f"   {pos}. {team:<30} {int(row['projected_points']):>3} pts")
        
        # Zona de riesgo
        print(f"\n⚠️  ZONA DE RIESGO (últimas 3):")
        for pos, (_, row) in enumerate(projected.tail(3).iterrows(), 1):
            team = clean_team_name(row['Team'])
            actual_pos = int(row['Position'])
            print(f"   {actual_pos}. {team:<30} {int(row['projected_points']):>3} pts")
        
        print("\n" + "=" * 90 + "\n")
        
        return True
    
    except Exception as e:
        print(f"❌ Error en proyección: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)