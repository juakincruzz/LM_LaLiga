"""Escudos pixelados de 16×16 como SVG inline.

Diseño propio: silueta común para todos, rellena con un patrón y los colores
del club (core.equipos.ESCUDOS). Se inserta inline, no como <img>, para que las
iniciales usen la fuente «Press Start 2P» de la página: un SVG cargado por
<img> no ve las fuentes del documento.
"""
from markupsafe import Markup, escape

from core.equipos import EGA, ESCUDOS, iniciales, nombre_visible

LADO = 16
TAM_INICIALES = 48  # por debajo, las iniciales no se leen


def _dentro(x: int, y: int) -> bool:
    """Silueta: recta arriba, en punta abajo."""
    if y < 1 or y > 15:
        return False
    i = 0 if y < 10 else y - 9
    return 1 + i <= x <= 14 - i


def _borde(x: int, y: int) -> bool:
    return _dentro(x, y) and not all(
        _dentro(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
    )


def _color(patron: str, colores: tuple[str, ...], x: int, y: int) -> str:
    c1 = colores[0]
    c2 = colores[1] if len(colores) > 1 else c1
    if patron == "rayas_v":
        return c1 if (x // 2) % 2 == 0 else c2
    if patron == "rayas_h":
        return c1 if (y // 2) % 2 == 0 else c2
    if patron == "mitades":
        return c1 if x < 8 else c2
    if patron == "franja_h":
        return c2 if 3 <= y <= 5 else c1
    if patron == "diagonal":
        return c2 if y > x else c1
    if patron == "banda":
        return c2 if abs(x - y) <= 1 else c1
    if patron == "rayas_esquina":
        if x < 7 and y < 6:
            return colores[2]
        return c1 if (x // 2) % 2 == 0 else c2
    return c1  # liso


def escudo_svg(nombre: str, tam: int = 16, decorativo: bool = True) -> Markup:
    datos = ESCUDOS.get(nombre) or {"patron": "liso", "colores": ("gris",), "iniciales": iniciales(nombre)}
    colores = tuple(EGA[c] for c in datos["colores"])
    con_iniciales = tam >= TAM_INICIALES

    def pixel(x: int, y: int) -> str | None:
        if not _dentro(x, y):
            return None
        if _borde(x, y) or (con_iniciales and 6 <= y <= 9):
            return EGA["negro"]
        return _color(datos["patron"], colores, x, y)

    # Un <rect> por tramo horizontal del mismo color, no uno por píxel
    rects = []
    for y in range(LADO):
        x = 0
        while x < LADO:
            color = pixel(x, y)
            ancho = 1
            while x + ancho < LADO and pixel(x + ancho, y) == color:
                ancho += 1
            if color:
                rects.append(f'<rect x="{x}" y="{y}" width="{ancho}" height="1" fill="{color}"/>')
            x += ancho

    if decorativo:
        accesible = 'aria-hidden="true" focusable="false"'
    else:
        accesible = f'role="img" aria-label="Escudo de {escape(nombre_visible(nombre))}"'

    texto = ""
    if con_iniciales:
        texto = (
            '<text x="8" y="8.4" text-anchor="middle" dominant-baseline="middle" '
            "font-family=\"'Press Start 2P', monospace\" font-size=\"2.6\" "
            f'fill="{EGA["amarillo"]}">{escape(datos["iniciales"])}</text>'
        )

    return Markup(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {LADO} {LADO}" width="{tam}" '
        f'height="{tam}" shape-rendering="crispEdges" class="escudo" {accesible}>'
        f'{"".join(rects)}{texto}</svg>'
    )
