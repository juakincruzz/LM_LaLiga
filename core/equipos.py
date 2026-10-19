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

# Understat (alineaciones y datos de jugadores) → football-data.co.uk.
# Verificado contra Understat 2021 a 2026. Mismo trato que OPENFOOTBALL: un
# ascendido que no esté aquí hace fallar la descarga con su nombre.
UNDERSTAT = {
    "Alaves": "Alaves",
    "Almeria": "Almeria",
    "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Barcelona": "Barcelona",
    "Cadiz": "Cadiz",
    "Celta Vigo": "Celta",
    "Deportivo La Coruna": "La Coruna",
    "Elche": "Elche",
    "Espanyol": "Espanol",
    "Getafe": "Getafe",
    "Girona": "Girona",
    "Granada": "Granada",
    "Las Palmas": "Las Palmas",
    "Leganes": "Leganes",
    "Levante": "Levante",
    "Malaga": "Malaga",
    "Mallorca": "Mallorca",
    "Osasuna": "Osasuna",
    "Racing Santander": "Santander",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Madrid": "Real Madrid",
    "Real Oviedo": "Oviedo",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "Sevilla": "Sevilla",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",
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


# Paleta EGA/VGA, la misma que app/static/css/retro.css
EGA = {
    "negro": "#000000",
    "azul": "#0000aa",
    "verde": "#00aa00",
    "cian": "#00aaaa",
    "rojo": "#aa0000",
    "magenta": "#aa00aa",
    "marron": "#aa5500",
    "gris": "#aaaaaa",
    "gris_osc": "#555555",
    "azul_cl": "#5555ff",
    "verde_cl": "#55ff55",
    "cian_cl": "#55ffff",
    "rojo_cl": "#ff5555",
    "magenta_cl": "#ff55ff",
    "amarillo": "#ffff55",
    "blanco": "#ffffff",
}


def _escudo(patron: str, *colores: str, ini: str) -> dict:
    return {"patron": patron, "colores": colores, "iniciales": ini}


# Escudos propios (app/escudos.py los dibuja): una interpretación libre de la
# equipación de cada club, NO su escudo oficial, que es marca registrada. Si
# asciende un equipo que no está aquí, sale un escudo gris con sus iniciales.
ESCUDOS = {
    "Alaves": _escudo("rayas_v", "azul", "blanco", ini="ALA"),
    "Almeria": _escudo("rayas_v", "rojo", "blanco", ini="ALM"),
    "Ath Bilbao": _escudo("rayas_v", "rojo", "blanco", ini="ATH"),
    "Ath Madrid": _escudo("rayas_esquina", "rojo", "blanco", "azul", ini="ATM"),
    "Barcelona": _escudo("rayas_v", "azul", "rojo", ini="FCB"),
    "Betis": _escudo("rayas_v", "verde", "blanco", ini="RBB"),
    "Cadiz": _escudo("franja_h", "amarillo", "azul", ini="CAD"),
    "Celta": _escudo("liso", "cian_cl", ini="CEL"),
    "Elche": _escudo("franja_h", "blanco", "verde", ini="ELC"),
    "Espanol": _escudo("rayas_v", "azul_cl", "blanco", ini="ESP"),
    "Getafe": _escudo("liso", "azul_cl", ini="GET"),
    "Girona": _escudo("rayas_v", "rojo", "blanco", ini="GIR"),
    "Granada": _escudo("rayas_h", "rojo", "blanco", ini="GRA"),
    "La Coruna": _escudo("rayas_v", "azul", "blanco", ini="RCD"),
    "Las Palmas": _escudo("mitades", "amarillo", "azul", ini="LPA"),
    "Leganes": _escudo("rayas_v", "azul", "blanco", ini="LEG"),
    "Levante": _escudo("mitades", "azul", "rojo", ini="LEV"),
    "Malaga": _escudo("rayas_v", "azul_cl", "blanco", ini="MAL"),
    "Mallorca": _escudo("diagonal", "rojo", "negro", ini="MLL"),
    "Osasuna": _escudo("diagonal", "rojo", "azul", ini="OSA"),
    "Oviedo": _escudo("liso", "azul", ini="OVI"),
    "Real Madrid": _escudo("banda", "blanco", "magenta", ini="RMA"),
    "Santander": _escudo("liso", "verde", ini="RAC"),
    "Sevilla": _escudo("diagonal", "blanco", "rojo", ini="SFC"),
    "Sociedad": _escudo("rayas_v", "azul", "blanco", ini="RSO"),
    "Valencia": _escudo("franja_h", "blanco", "negro", ini="VCF"),
    "Valladolid": _escudo("rayas_v", "magenta", "blanco", ini="VLL"),
    "Vallecano": _escudo("banda", "blanco", "rojo", ini="RAY"),
    "Villarreal": _escudo("liso", "amarillo", ini="VIL"),
}


def iniciales(nombre: str) -> str:
    """'Recien Ascendido' -> 'RA'; 'Girona' -> 'GIR'."""
    palabras = nombre.split()
    if len(palabras) > 1:
        return "".join(p[0] for p in palabras[:3]).upper()
    return nombre[:3].upper()


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


def desde_understat(nombre: str) -> str:
    try:
        return UNDERSTAT[nombre]
    except KeyError:
        raise EquipoDesconocidoError(
            f"Equipo de Understat sin traducir: «{nombre}». "
            f"Añádelo a UNDERSTAT en core/equipos.py."
        ) from None


def nombre_visible(nombre: str) -> str:
    return NOMBRES_VISIBLES.get(nombre, nombre)
