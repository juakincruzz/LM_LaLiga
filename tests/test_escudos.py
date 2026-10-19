"""Escudos pixelados propios (diseño nuestro, no los oficiales de los clubes)."""
from app.escudos import escudo_svg
from core.equipos import ESCUDOS, iniciales

# Todos los equipos que aparecen en los CSV de 21/22 a 26/27
EQUIPOS_DATOS = [
    "Alaves", "Almeria", "Ath Bilbao", "Ath Madrid", "Barcelona", "Betis", "Cadiz", "Celta",
    "Elche", "Espanol", "Getafe", "Girona", "Granada", "La Coruna", "Las Palmas", "Leganes",
    "Levante", "Malaga", "Mallorca", "Osasuna", "Oviedo", "Real Madrid", "Santander", "Sevilla",
    "Sociedad", "Valencia", "Valladolid", "Vallecano", "Villarreal",
]


def test_todos_los_equipos_de_los_datos_tienen_escudo():
    assert set(EQUIPOS_DATOS) <= set(ESCUDOS)


def test_svg_es_pixelado_y_usa_los_colores_del_club():
    svg = str(escudo_svg("Betis", 16))
    assert svg.startswith("<svg") and 'shape-rendering="crispEdges"' in svg
    assert 'viewBox="0 0 16 16"' in svg and 'width="16"' in svg
    assert "#00aa00" in svg and "#ffffff" in svg  # verde y blanco EGA


def test_iniciales_solo_a_partir_de_48px():
    assert "RBB" not in str(escudo_svg("Betis", 32))
    assert "RBB" in str(escudo_svg("Betis", 64))


def test_equipo_desconocido_da_escudo_gris_con_iniciales():
    svg = str(escudo_svg("Recien Ascendido", 64))
    assert "#aaaaaa" in svg and ">RA<" in svg


def test_iniciales_derivadas():
    assert iniciales("Recien Ascendido") == "RA"
    assert iniciales("Girona") == "GIR"


def test_decorativo_oculta_a_lectores_y_no_decorativo_lleva_etiqueta():
    assert 'aria-hidden="true"' in str(escudo_svg("Betis"))
    assert 'aria-label="Escudo de Atlético"' in str(escudo_svg("Ath Madrid", decorativo=False))


def test_nombre_se_escapa():
    assert "<script>" not in str(escudo_svg("<script>", 64, decorativo=False))
