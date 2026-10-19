"""Página de la ficha de equipo: /equipo/{nombre}."""
import json

import aiosqlite
import pytest

from app.services.data_service import update_league_data


async def _actualizar(temp_db, season=None):
    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        await update_league_data(db, "SP1", season)


@pytest.fixture
async def poblada(temp_db, mock_download):
    """La 25/26 completa en `matches`; Understat sin red (sin caché de jugadores)."""
    await _actualizar(temp_db)


@pytest.fixture
async def poblada_con_jugadores(temp_db, mock_download, mock_understat):
    """Igual, con las alineaciones del Betis en la caché de Understat."""
    await _actualizar(temp_db)


async def test_ficha_200_con_todas_las_ventanas(client, poblada_con_jugadores):
    r = await client.get("/equipo/Betis")
    assert r.status_code == 200
    for titulo in ("Estado de forma", "Puntos fuertes y débiles", "11 más habitual",
                   "Mejores jugadores", "Palmarés"):
        assert titulo in r.text
    assert 'width="80"' in r.text  # escudo grande en la cabecera, como en el boceto
    assert "Copa del Rey" in r.text and "×3" in r.text
    assert "Héctor Bellerín" in r.text


async def test_ficha_nombre_con_espacio(client, poblada):
    r = await client.get("/equipo/Real%20Madrid")
    assert r.status_code == 200
    assert "Real Madrid" in r.text


async def test_equipo_inexistente_404(client, poblada):
    r = await client.get("/equipo/Narnia")
    assert r.status_code == 404
    assert "Equipo no encontrado" in r.text
    assert "Liga no encontrada" not in r.text


async def test_ficha_sin_understat(client, poblada):
    r = await client.get("/equipo/Betis")
    assert r.status_code == 200
    assert "Datos de jugadores no disponibles" in r.text
    assert "Estado de forma" in r.text  # el resto de la ficha sigue


async def test_ficha_sin_cuotas_ni_puntos_por_partido(client, poblada_con_jugadores):
    texto = (await client.get("/equipo/Betis")).text.lower()
    for prohibido in ("cuota", "apuesta", "pts/partido", "al alza"):
        assert prohibido not in texto


async def test_ficha_sin_datos_usa_temporada_anterior(client, temp_db, mock_download):
    await _actualizar(temp_db, season="2425")  # la actual (25/26) queda sin filas
    r = await client.get("/equipo/Betis")
    assert r.status_code == 200
    assert "Aún no ha jugado esta temporada" in r.text
    assert "Palmarés" in r.text


async def test_ficha_en_curso_once_contra_el_rival(client, temp_db, media_temporada, mock_understat):
    await _actualizar(temp_db)
    r = await client.get("/equipo/Betis")
    assert r.status_code == 200
    assert "Posible 11 · contra" in r.text
    assert "Próximo:" in r.text


async def test_enlaces_a_la_ficha(client, temp_db):
    tabla = json.dumps([
        {"name": n, "points": 50, "gf": 40, "ga": 30, "champion_prob": 0.1, "relegation_prob": 0.0}
        for n in ("Ath Madrid", "Ascendido 1")
    ])
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            "INSERT INTO simulations (model_run_id, league_code, sim_count, standings_json, label, season) "
            "VALUES (NULL, 'SP1', 10, ?, 'Proyección 26/27', '2526')",
            (tabla,),
        )
        await db.commit()

    texto = (await client.get("/")).text
    assert 'href="/equipo/Ath%20Madrid"' in texto
    assert 'href="/equipo/Ascendido%201"' not in texto


async def test_equipo_sin_alineaciones_muestra_mensaje_en_jugadores(client, poblada_con_jugadores):
    # Getafe no sale en las alineaciones de la fixture: como un equipo en pretemporada
    texto = (await client.get("/equipo/Getafe")).text
    assert texto.count("Aún no hay alineaciones de este equipo esta temporada.") == 2


async def test_ascendido_de_la_proyeccion_tiene_ficha(client, poblada, monkeypatch):
    # Fuera de temporada, la proyección lista ascendidos reales que aún no
    # tienen partidos en la BD: su enlace no debe dar 404
    import app.services.team_service as ts

    monkeypatch.setattr(ts, "_ascendidos", lambda: {"Malaga"})
    r = await client.get("/equipo/Malaga")
    assert r.status_code == 200
    assert "Aún no ha jugado esta temporada" in r.text
    assert (await client.get("/equipo/Ascendido%201")).status_code == 404
