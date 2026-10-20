"""Fixtures compartidas.

Objetivo: ningún test puede tocar `data/db.sqlite` ni `data/raw/`. Las dos
fixtures de aislamiento son `autouse`, así que un test nuevo queda protegido
aunque su autor no se acuerde de pedirlas.
"""
import asyncio
import csv
import json
import shutil
from pathlib import Path
from unittest import mock

import httpx
import pytest
import requests

FIXTURES_DIR = Path(__file__).parent / "fixtures"
JUGADOS = 190  # partidos jugados en `media_temporada`


@pytest.fixture(autouse=True)
def temporada_de_fixtures(monkeypatch):
    """Congela la temporada actual en 25/26, la de los CSV de tests/fixtures,
    y la primera de entrenamiento en 21/22, la más antigua que hay.

    config/leagues.yaml sigue a la temporada real; sin esto, cambiarla rompería
    todos los tests que cuentan con la 25/26 completa de la fixture. Debe ir
    antes que `temp_db`, que siembra la liga con la temporada actual.
    """
    import app.config as config

    monkeypatch.setitem(config.DEFAULTS, "current_season", "2526")
    # La fixture más antigua: la configuración real entrena desde la 14/15
    monkeypatch.setitem(config.DEFAULTS, "first_season", "2122")


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch, temporada_de_fixtures):
    """Redirige la base de datos a un fichero temporal y la inicializa.

    `DB_PATH` se importa por valor en varios módulos (`from app.database import
    DB_PATH`), así que hay que parchear cada binding, no solo el original.
    """
    import app.database as database
    import app.routers.jobs as jobs_router
    import app.seed as seed

    db_path = tmp_path / "test.sqlite"
    for module in (database, jobs_router, seed):
        monkeypatch.setattr(module, "DB_PATH", db_path, raising=False)

    asyncio.run(database.init_db())
    asyncio.run(seed.seed_leagues())

    return db_path


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    """Redirige `data/raw/` a un directorio temporal con la temporada de prueba.

    Todos los módulos que leen CSV pasan por `league_csv_path()`, que resuelve
    `DATA_DIR` en tiempo de llamada, así que basta con parchear la constante.
    """
    import scripts.data_manager_generic as dmg

    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(dmg, "DATA_DIR", raw_dir)

    league_dir = raw_dir / "SP1"
    league_dir.mkdir(parents=True)
    shutil.copy(FIXTURES_DIR / "SP1_2526.csv", league_dir / "2526.csv")

    return raw_dir


@pytest.fixture(autouse=True)
def _job_lock_por_test(monkeypatch):
    """`asyncio.Lock()` se enlaza al primer event loop que lo usa (`acquire()`).

    En producción no pasa nada: uvicorn tiene un único loop durante toda la
    vida del proceso. Pero pytest-asyncio crea un loop nuevo por test, así que
    el lock global de app.routers.jobs quedaría enlazado al loop del primer
    test y reventaría en el segundo con 'bound to a different event loop'.
    """
    import asyncio

    import app.routers.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "_job_lock", asyncio.Lock())


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    """Los .pkl entrenados en tests van a un temporal, no a models/."""
    import models.predictor as predictor

    destino = tmp_path / "models"
    destino.mkdir()
    monkeypatch.setattr(predictor, "MODELS_DIR", destino)
    return destino


@pytest.fixture
def mock_download():
    """Sustituye la descarga por los CSV de fixture reales, eligiendo la
    temporada a partir de la URL.

    Antes devolvía siempre el mismo contenido sin mirar qué temporada se
    pedía: entrenar "con 5 temporadas" en un test eran en realidad 5 copias
    idénticas de 2526, y por eso la accuracy salía 1.0 en vez de realista.
    """
    contenidos = {
        p.stem.split("_")[-1]: p.read_bytes() for p in FIXTURES_DIR.glob("SP1_*.csv")
    }

    def responder(url, timeout=30):
        if "openfootball" in url:
            # Sin calendario: los tests que lo necesitan lo escriben en disco
            respuesta = mock.MagicMock()
            respuesta.raise_for_status.side_effect = requests.HTTPError("404 (calendario de test)")
            return respuesta
        temporada = url.rstrip("/").split("/")[-2]  # .../mmz4281/{temporada}/SP1.csv
        return mock.MagicMock(content=contenidos[temporada])

    with mock.patch("scripts.data_manager_generic.requests.get", side_effect=responder) as mock_get:
        yield mock_get


@pytest.fixture(autouse=True)
def understat_offline(monkeypatch):
    """Understat sin red por defecto: `update_league_data` lo descarga, y un
    test que no pida `mock_understat` no debe salir a internet.

    Se parchea `scripts.understat._get` y no `requests.get`: el módulo
    `requests` es uno solo, y parchearlo pisaría el `mock_download` de
    football-data."""
    import scripts.understat as understat

    monkeypatch.setattr(understat, "PAUSA", 0)
    monkeypatch.setattr(
        understat, "_get", mock.MagicMock(side_effect=requests.ConnectionError("sin red en tests"))
    )


@pytest.fixture
def mock_understat(monkeypatch):
    """Understat servido desde tests/fixtures/understat: la liga 25/26 recortada
    a los 6 últimos partidos del Betis y sus alineaciones reales."""
    import scripts.understat as understat

    carpeta = FIXTURES_DIR / "understat"

    def responder(url):
        prefijo = "liga" if "getLeagueData" in url else "partido"
        fichero = carpeta / f"{prefijo}_{url.rstrip('/').split('/')[-1]}.json"
        if not fichero.exists():
            raise requests.HTTPError(f"404 {url}")
        return json.loads(fichero.read_text())

    falso = mock.MagicMock(side_effect=responder)
    monkeypatch.setattr(understat, "_get", falso)
    return falso


@pytest.fixture
async def client():
    """Cliente HTTP contra la app ASGI, sin levantar servidor."""
    from app.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def media_temporada(data_root, monkeypatch):
    """La 25/26 a medias (190 jugados) con su calendario completo de 380.

    Escribe en `data_root` la 25/26 a medias y su calendario completo, y
    neutraliza las descargas de `update_league_data` (que si no, traerían la
    temporada entera de la fixture)."""
    with open(FIXTURES_DIR / "SP1_2526.csv", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))

    destino = data_root / "SP1"
    with open(destino / "2526.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas[:JUGADOS])

    with open(destino / "2526_calendario.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "HomeTeam", "AwayTeam", "Matchday"])
        for i, fila in enumerate(filas):
            writer.writerow([fila["Date"], fila["HomeTeam"], fila["AwayTeam"], i // 10 + 1])

    import app.services.data_service as ds

    monkeypatch.setattr(ds, "download_league_data", lambda league, season: destino / f"{season}.csv")
    monkeypatch.setattr(ds, "download_calendar", lambda league, season: destino / f"{season}_calendario.csv")
    return destino
