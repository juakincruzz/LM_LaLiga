"""Nombres de equipo: el de football-data.co.uk es el canónico.

Todo el proyecto (BD, features, modelo) usa los nombres cortos de
football-data.co.uk ("Ath Madrid"). Este módulo traduce los de las otras
fuentes a ese nombre, y da un nombre legible para mostrar en la interfaz.
"""
from core.errors import LMLaLigaError

# openfootball (calendario) → football-data.co.uk. Verificado contra los JSON de
# 2021-22 a 2026-27 y los CSV de 1415 a 2627. Si asciende un equipo que no está
# aquí, la descarga del calendario falla con su nombre: basta con añadirlo.
OPENFOOTBALL = {
    "Athletic Club": "Ath Bilbao",
    "CA Osasuna": "Osasuna",
    "CD Leganés": "Leganes",
    "Club Atlético de Madrid": "Ath Madrid",
    "Cádiz CF": "Cadiz",
    "Deportivo Alavés": "Alaves",
    "Elche CF": "Elche",
    "FC Barcelona": "Barcelona",
    "Getafe CF": "Getafe",
    "Girona FC": "Girona",
    "Granada CF": "Granada",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "RC Celta de Vigo": "Celta",
    "RC Deportivo La Coruña": "La Coruna",
    "RCD Espanyol de Barcelona": "Espanol",
    "RCD Mallorca": "Mallorca",
    "Rayo Vallecano de Madrid": "Vallecano",
    "Real Betis Balompié": "Betis",
    "Real Madrid CF": "Real Madrid",
    "Real Oviedo": "Oviedo",
    "Real Racing Club de Santander": "Santander",
    "Real Sociedad de Fútbol": "Sociedad",
    "Real Valladolid CF": "Valladolid",
    "Sevilla FC": "Sevilla",
    "UD Almería": "Almeria",
    "UD Las Palmas": "Las Palmas",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
}

# football-data.co.uk → nombre para la interfaz (sin tildes perdidas ni abreviaturas)
NOMBRES_VISIBLES = {
    "Alaves": "Alavés",
    "Almeria": "Almería",
    "Ath Bilbao": "Athletic",
    "Ath Madrid": "Atlético",
    "Cadiz": "Cádiz",
    "Cordoba": "Córdoba",
    "Espanol": "Espanyol",
    "La Coruna": "Deportivo",
    "Leganes": "Leganés",
    "Malaga": "Málaga",
    "Santander": "Racing",
    "Sociedad": "Real Sociedad",
    "Sp Gijon": "Sporting",
    "Vallecano": "Rayo Vallecano",
}


class EquipoDesconocidoError(LMLaLigaError):
    """Un nombre de otra fuente que no está en la tabla de traducción."""


def desde_openfootball(nombre: str) -> str:
    try:
        return OPENFOOTBALL[nombre]
    except KeyError:
        raise EquipoDesconocidoError(
            f"Equipo del calendario sin traducir: «{nombre}». "
            f"Añádelo a OPENFOOTBALL en core/equipos.py."
        ) from None


def nombre_visible(nombre: str) -> str:
    return NOMBRES_VISIBLES.get(nombre, nombre)
