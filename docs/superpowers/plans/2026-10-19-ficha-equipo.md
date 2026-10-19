# Ficha de equipo y escudos retro — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Escudos pixelados propios en toda la web y una ficha por equipo en `/equipo/{nombre}` con
forma, fuertes/débiles, posible 11, mejores jugadores y palmarés.

**Architecture:** Los escudos son SVG inline generados en Python desde datos de `core/equipos.py`.
Understat se descarga en el job `update` a una caché JSON (`scripts/understat.py`, único módulo que lo
conoce). La ficha la compone `app/services/team_service.py` con funciones puras sobre el DataFrame
de la temporada (football-data) y otro de alineaciones (Understat), y la sirve un router nuevo.

**Tech Stack:** Python 3.14, FastAPI, Jinja2, HTMX, pandas, requests, pytest. Sin dependencias nuevas.

**Spec:** [docs/superpowers/specs/2026-10-19-ficha-equipo-design.md](../specs/2026-10-19-ficha-equipo-design.md)

## Global Constraints

- Textos de usuario, docstrings y comentarios en castellano; identificadores en inglés o como el
  fichero que tocas (el repo mezcla: `nombre_visible`, `load_season`). Sigue el fichero.
- Nombres de equipo canónicos = football-data.co.uk (`"Ath Madrid"`). Todo lo demás se traduce.
- Usa `app.config.LEAGUE_CODE`, nunca `"SP1"` a mano. `league_code` sigue siendo string.
- Plantillas: el `templates` compartido de `app/templating.py`; nada de `Jinja2Templates` nuevo.
- CSS solo en `app/static/css/retro.css`; nada de frameworks, npm ni build.
- Press Start 2P solo para texto sin mayúsculas acentuadas (cifras, iniciales, nombres sin tilde);
  títulos de ventana en VT323 (`.ventana__titulo` ya lo hace).
- Ningún test toca la red: Understat se simula en `tests/conftest.py`.
- Nada de cuotas en la ficha. El estado de forma son **solo las 5 fichas V/E/D**, sin puntos por
  partido ni flechas.
- La ficha nunca llama a Understat; solo lee la caché.
- Nada nuevo entra en `FEATURE_COLS`; `tests/test_feature_parity.py` debe seguir pasando sin tocar
  su snapshot.
- Entorno: `source .venv/bin/activate` (ya funciona). Suite: `pytest -q`.

## Review Focus

1. **Equipo recién ascendido sin entrada en `ESCUDOS` ni en `UNDERSTAT`** → escudo gris con
   iniciales y la descarga de Understat avisa sin tumbar la actualización. Tests en Tasks 1 y 3.
2. **Nombre con espacio o tilde en la URL** (`/equipo/Ath%20Madrid`, `/equipo/Real%20Madrid`) → 200,
   y enlaces generados con `urlencode`. Test en Task 8.
3. **Equipo con menos de 5 partidos jugados** (jornada 2) → forma con los que haya, 11 con los
   titulares disponibles, sin excepción. Tests en Tasks 5 y 6.
4. **Caché de Understat parcial** (liga descargada, algún partido sin JSON por un corte) → se usan
   los partidos que hay y la siguiente actualización completa los que faltan. Test en Task 3.
5. **Jugador traspasado a mitad de temporada** → solo cuenta lo jugado con este equipo (sale gratis
   al agregar desde las alineaciones por equipo). Test en Task 6.

---

### Task 1: Datos de escudos y generador SVG

**Files:**
- Modify: `core/equipos.py` (añadir `EGA`, `ESCUDOS`, `iniciales()`)
- Create: `app/escudos.py`
- Modify: `app/templating.py` (global `escudo`)
- Test: `tests/test_escudos.py`

**Interfaces:**
- Produces: `core.equipos.ESCUDOS: dict[str, dict]` con claves `patron: str`, `colores: tuple[str, ...]`
  (nombres de `EGA`), `iniciales: str`.
- Produces: `core.equipos.iniciales(nombre: str) -> str`.
- Produces: `app.escudos.escudo_svg(nombre: str, tam: int = 16, decorativo: bool = True) -> markupsafe.Markup`,
  expuesta en Jinja como `escudo(nombre, tam)`.

- [ ] **Step 1: Escribir los tests**

```python
from app.escudos import escudo_svg
from core.equipos import ESCUDOS, iniciales

EQUIPOS_DATOS = ['Alaves', 'Almeria', 'Ath Bilbao', 'Ath Madrid', 'Barcelona', 'Betis', 'Cadiz',
    'Celta', 'Elche', 'Espanol', 'Getafe', 'Girona', 'Granada', 'La Coruna', 'Las Palmas', 'Leganes',
    'Levante', 'Malaga', 'Mallorca', 'Osasuna', 'Oviedo', 'Real Madrid', 'Santander', 'Sevilla',
    'Sociedad', 'Valencia', 'Valladolid', 'Vallecano', 'Villarreal']

def test_todos_los_equipos_de_los_datos_tienen_escudo():
    assert set(EQUIPOS_DATOS) <= set(ESCUDOS)

def test_svg_es_pixelado_y_usa_los_colores_del_club():
    svg = str(escudo_svg("Betis", 16))
    assert svg.startswith("<svg") and 'shape-rendering="crispEdges"' in svg
    assert 'viewBox="0 0 16 16"' in svg and 'width="16"' in svg
    assert "#00aa00" in svg and "#ffffff" in svg      # verde y blanco EGA

def test_iniciales_solo_a_partir_de_48px():
    assert "RBB" not in str(escudo_svg("Betis", 32))
    assert "RBB" in str(escudo_svg("Betis", 64))

def test_equipo_desconocido_da_escudo_gris_con_iniciales():
    svg = str(escudo_svg("Recien Ascendido", 64))
    assert "#aaaaaa" in svg and "RA" in svg

def test_iniciales_derivadas():
    assert iniciales("Recien Ascendido") == "RA"
    assert iniciales("Girona") == "GIR"

def test_decorativo_oculta_a_lectores_y_no_decorativo_lleva_etiqueta():
    assert 'aria-hidden="true"' in str(escudo_svg("Betis"))
    assert 'aria-label="Escudo de Betis"' in str(escudo_svg("Betis", decorativo=False))
```

(El `aria-label` usa `nombre_visible`.)

- [ ] **Step 2: Ejecutar y ver que fallan** — `pytest tests/test_escudos.py -q` → ImportError.

- [ ] **Step 3: Implementar los datos en `core/equipos.py`**

`EGA` = los 16 colores: negro `#000000`, azul `#0000aa`, verde `#00aa00`, cian `#00aaaa`, rojo
`#aa0000`, magenta `#aa00aa`, marron `#aa5500`, gris `#aaaaaa`, gris_osc `#555555`, azul_cl
`#5555ff`, verde_cl `#55ff55`, cian_cl `#55ffff`, rojo_cl `#ff5555`, magenta_cl `#ff55ff`,
amarillo `#ffff55`, blanco `#ffffff`.

`ESCUDOS` (patrón, colores, iniciales) — interpretación libre de la equipación:

| Equipo | patrón | colores | ini |
|---|---|---|---|
| Alaves | rayas_v | azul, blanco | ALA |
| Almeria | rayas_v | rojo, blanco | ALM |
| Ath Bilbao | rayas_v | rojo, blanco | ATH |
| Ath Madrid | rayas_esquina | rojo, blanco, azul | ATM |
| Barcelona | rayas_v | azul, rojo | FCB |
| Betis | rayas_v | verde, blanco | RBB |
| Cadiz | franja_h | amarillo, azul | CAD |
| Celta | liso | cian_cl | CEL |
| Elche | franja_h | blanco, verde | ELC |
| Espanol | rayas_v | azul_cl, blanco | ESP |
| Getafe | liso | azul_cl | GET |
| Girona | rayas_v | rojo, blanco | GIR |
| Granada | rayas_h | rojo, blanco | GRA |
| La Coruna | rayas_v | azul, blanco | RCD |
| Las Palmas | mitades | amarillo, azul | LPA |
| Leganes | rayas_v | azul, blanco | LEG |
| Levante | mitades | azul, rojo | LEV |
| Malaga | rayas_v | azul_cl, blanco | MAL |
| Mallorca | diagonal | rojo, negro | MLL |
| Osasuna | diagonal | rojo, azul | OSA |
| Oviedo | liso | azul | OVI |
| Real Madrid | banda | blanco, magenta | RMA |
| Santander | liso | verde | RAC |
| Sevilla | diagonal | blanco, rojo | SFC |
| Sociedad | rayas_v | azul, blanco | RSO |
| Valencia | franja_h | blanco, negro | VCF |
| Valladolid | rayas_v | magenta, blanco | VLL |
| Vallecano | banda | blanco, rojo | RAY |
| Villarreal | liso | amarillo | VIL |

`iniciales(nombre)`: con varias palabras, la primera letra de cada una (máx. 3), en mayúsculas;
con una, sus 3 primeras letras en mayúsculas.

- [ ] **Step 4: Implementar `app/escudos.py`**

Silueta 16×16 (la del boceto aprobado):

```python
def _dentro(x, y):
    if y < 1 or y > 15:
        return False
    i = 0 if y < 10 else y - 9
    return 1 + i <= x <= 14 - i
# borde = dentro y algún vecino 4-conectado fuera → negro
```

Patrones `(x, y) -> color` con c1, c2, c3: `liso` c1 · `rayas_v` c1 si `(x // 2) % 2 == 0` si no c2 ·
`rayas_h` igual con `y` · `mitades` c1 si `x < 8` · `franja_h` c2 en `3 <= y <= 5`, si no c1 ·
`diagonal` c2 si `y > x`, si no c1 · `banda` c2 si `abs(x - y) <= 1`, si no c1 ·
`rayas_esquina` como `rayas_v` salvo `x < 7 and y < 6` → c3.
Desconocido: `liso` gris + `iniciales(nombre)`.

Salida: `<svg xmlns=… viewBox="0 0 16 16" width="{tam}" height="{tam}" shape-rendering="crispEdges" class="escudo">`,
un `<rect>` por **tramo horizontal** de píxeles del mismo color (no uno por píxel). Con `tam >= 48`:
franja negra en filas 6–9 y `<text x="8" y="8.4" text-anchor="middle" dominant-baseline="middle"
font-family="'Press Start 2P', monospace" font-size="2.6" fill="#ffff55">{ini}</text>`.
Decorativo → `aria-hidden="true" focusable="false"`; si no, `role="img" aria-label="Escudo de {nombre_visible}"`.
Escapa el nombre (`markupsafe.escape`). Registrar en `app/templating.py`:
`templates.env.globals["escudo"] = escudo_svg`.

- [ ] **Step 5: Tests en verde** — `pytest tests/test_escudos.py -q` → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat: escudos pixelados propios por equipo"`

---

### Task 2: Escudos en clasificación y predicciones

**Files:**
- Modify: `app/templates/partials/standings_table.html`, `app/templates/partials/jornada.html`,
  `app/templates/partials/prediccion.html`, `app/static/css/retro.css`
- Test: `tests/test_league_page.py`, `tests/test_prediccion.py` (añadir)

**Interfaces:**
- Consumes: global Jinja `escudo(nombre, tam)` (Task 1).

- [ ] **Step 1: Tests** — en `tests/test_league_page.py`, con una simulación ya guardada (reutiliza
  el patrón de los tests existentes que pintan la clasificación), `test_clasificacion_lleva_escudos`:
  la respuesta de `/` contiene `class="escudo"` al menos 20 veces. En `tests/test_prediccion.py`,
  con la fixture `media_temporada`, `test_jornada_lleva_escudos`: `/prediccion/jornada` contiene
  `class="escudo"`; y `test_partido_lleva_escudos_grandes`: `/prediccion/partido?...` contiene
  `width="32"`.
- [ ] **Step 2: Ver que fallan.**
- [ ] **Step 3: Insertar** — clasificación: `{{ escudo(team.name, 16) }}` delante del nombre en el
  `<th scope="row">`. Jornada: `escudo(p.local, 16)` / `escudo(p.visitante, 16)` junto a cada nombre.
  Cara a cara (`.duelo__equipo`): `escudo(..., 32)` encima de cada nombre. CSS: `.escudo { vertical-align: middle; flex: none; }`
  y el margen justo para que no pegue al texto; en `.duelo__equipo` el escudo en bloque centrado.
  Comprueba en el navegador (`python run_web.py`) que la tabla no se desborda en 375 px de ancho.
- [ ] **Step 4: Tests en verde** — `pytest tests/test_league_page.py tests/test_prediccion.py -q`.
- [ ] **Step 5: Commit** — `git commit -m "feat: escudos en clasificación y predicciones"`

---

### Task 3: Cliente y caché de Understat

**Files:**
- Modify: `core/equipos.py` (`UNDERSTAT`, `desde_understat`)
- Create: `scripts/understat.py`
- Create: `tests/fixtures/understat/liga_2025.json`, `tests/fixtures/understat/partido_{id}.json` (6)
- Modify: `tests/conftest.py`, `.gitignore`
- Test: `tests/test_understat.py`

**Interfaces:**
- Produces: `core.equipos.desde_understat(nombre: str) -> str` (lanza `EquipoDesconocidoError`
  pidiendo añadirlo a `UNDERSTAT`).
- Produces: `scripts.understat.understat_dir(league_code: str) -> Path` =
  `scripts.data_manager_generic.DATA_DIR / league_code / "understat"` (resuelto en cada llamada).
- Produces: `scripts.understat.download_understat(league_code: str, season: str, on_progress=None) -> dict`
  → `{"partidos": int, "nuevos": int, "faltan": int}`. `on_progress(hechos, total)` por partido.
- Produces: `scripts.understat.load_lineups(league_code: str, season: str) -> pd.DataFrame`, una fila
  por jugador y partido con columnas `match_id, date (Timestamp), team, opponent, home (bool),
  player_id, player, position, starter (bool), minutes, goals, assists, xg, key_passes, yellow, red`
  (equipos ya traducidos a football-data; numéricos como int/float). Lanza `FileNotFoundError` si no
  hay `liga_{año}.json` en caché.
- Produces: `scripts.understat.PAUSA: float = 0.4` (segundos entre peticiones; los tests la ponen a 0).

- [ ] **Step 1: Crear las fixtures (una vez, con red)** — script temporal en el scratchpad, no en el
  repo: descarga `getLeagueData/La_liga/2025`, se queda con los 6 últimos partidos jugados
  (`isResult`) del Real Betis y guarda `{"dates": [esos 6]}` en `liga_2025.json`; para cada uno,
  `getMatchData/{id}` guardando solo `{"rosters": ...}` en `partido_{id}.json`. Cabeceras:
  `User-Agent` de navegador y `X-Requested-With: XMLHttpRequest`; `requests` descomprime el gzip.
  Anota en el test, en un comentario, qué partidos son.
- [ ] **Step 2: `UNDERSTAT`** en `core/equipos.py`: Alaves→Alaves, Almeria→Almeria, Athletic
  Club→Ath Bilbao, Atletico Madrid→Ath Madrid, Barcelona, Cadiz, Celta Vigo→Celta, Deportivo La
  Coruna→La Coruna, Elche, Espanyol→Espanol, Getafe, Girona, Granada, Las Palmas, Leganes, Levante,
  Malaga, Mallorca, Osasuna, Racing Santander→Santander, Rayo Vallecano→Vallecano, Real
  Betis→Betis, Real Madrid, Real Oviedo→Oviedo, Real Sociedad→Sociedad, Real Valladolid→Valladolid,
  Sevilla, Valencia, Villarreal (verificado contra Understat 2021–2026).
- [ ] **Step 3: conftest** — fixture **autouse** `understat_offline` que parchea
  `scripts.understat.requests.get` para lanzar `requests.ConnectionError("sin red en tests")` y pone
  `PAUSA = 0`. Fixture `mock_understat` (no autouse) que sirve `liga_2025.json` para
  `getLeagueData` y `partido_{id}.json` para `getMatchData/{id}` (404 → `HTTPError` si no existe la
  fixture) y devuelve el mock para contar llamadas. `.gitignore`: añade `data/raw/**/understat/`.
- [ ] **Step 4: Tests**

```python
def test_descarga_y_carga_alineaciones(mock_understat):
    r = download_understat("SP1", "2526")
    assert r == {"partidos": 6, "nuevos": 6, "faltan": 0}
    df = load_lineups("SP1", "2526")
    assert set(df["team"]) >= {"Betis"} and df["starter"].groupby(df["match_id"]).sum().min() == 22
    assert df.loc[df["position"] == "Sub", "starter"].eq(False).all()

def test_partidos_jugados_no_se_vuelven_a_pedir(mock_understat):
    download_understat("SP1", "2526"); n = mock_understat.call_count
    assert download_understat("SP1", "2526")["nuevos"] == 0
    assert mock_understat.call_count == n + 1          # solo la liga

def test_cache_parcial_se_completa(mock_understat, data_root):
    download_understat("SP1", "2526")
    un_partido = next((data_root / "SP1" / "understat" / "partidos").glob("*.json")); un_partido.unlink()
    assert download_understat("SP1", "2526")["nuevos"] == 1

def test_sin_cache_es_file_not_found():
    with pytest.raises(FileNotFoundError): load_lineups("SP1", "2526")

def test_equipo_desconocido(mock_understat, monkeypatch):
    monkeypatch.delitem(core.equipos.UNDERSTAT, "Real Betis")
    with pytest.raises(EquipoDesconocidoError): download_understat("SP1", "2526")
```

  Más `test_carga_ignora_partido_sin_json` (borrar un JSON y `load_lineups` devuelve 5 partidos).
- [ ] **Step 5: Ver que fallan.**
- [ ] **Step 6: Implementar `scripts/understat.py`** — año de Understat = `2000 + int(season[:2])`.
  Caché: `understat_dir/liga_{año}.json` (se reescribe siempre) y
  `understat_dir/partidos/{id}.json` (solo partidos `isResult`, solo si no existen; `time.sleep(PAUSA)`
  entre peticiones). Traduce los nombres al descargar la liga, para fallar pronto. `faltan` = partidos
  jugados sin JSON tras la descarga (un error en un partido concreto se cuenta ahí y no aborta el
  resto). `starter = position != "Sub"`. Docstring del módulo: es el único que conoce Understat;
  para cambiar de fuente, reimplementar estas dos funciones.
- [ ] **Step 7: Tests en verde** — `pytest tests/test_understat.py -q` y la suite entera.
- [ ] **Step 8: Commit** — incluye las fixtures JSON. `git commit -m "feat: caché de alineaciones de Understat"`

---

### Task 4: Understat dentro de «Actualizar datos»

**Files:**
- Modify: `app/services/data_service.py`, `app/routers/jobs.py`
- Test: `tests/test_data_service.py`

**Interfaces:**
- Consumes: `download_understat` (Task 3).
- Produces: `update_league_data(db, league_code, season=None, on_progress=None) -> dict`; el resumen
  gana `"jugadores": int` (partidos con alineación) o un aviso en `"aviso_jugadores"`.
- Produces: `JobContext.progress_threadsafe(done, total, etiqueta="Simulación")`.

- [ ] **Step 1: Tests** — `test_actualizar_trae_alineaciones(mock_download, mock_understat)`: el
  resumen tiene `jugadores == 6` y no `aviso_jugadores`. `test_actualizar_sin_understat_sigue(mock_download)`
  (con la fixture autouse offline): la actualización termina, `matches` tiene filas y
  `"aviso_jugadores"` empieza por `"sin datos de jugadores"`.
- [ ] **Step 2: Ver que fallan.**
- [ ] **Step 3: Implementar** — tras volcar partidos, `await asyncio.to_thread(download_understat,
  league_code, season, on_progress)` dentro de `try/except (requests.RequestException,
  EquipoDesconocidoError, KeyError, ValueError)`; el aviso: `f"sin datos de jugadores ({type(e).__name__}: {e})"`.
  Si `faltan > 0`, aviso `f"faltan {n} alineaciones; se completarán en la próxima actualización"`.
  En `_job_update_data`: pasa `on_progress=lambda h, t: ctx.progress_threadsafe(h, t, "Alineaciones")`
  y añade el aviso al `mensaje` como ya se hace con el del calendario.
- [ ] **Step 4: Tests en verde** — `pytest tests/test_data_service.py tests/test_jobs_queue.py -q`.
- [ ] **Step 5: Commit** — `git commit -m "feat: descargar alineaciones al actualizar datos"`

---

### Task 5: Forma, cabecera y fuertes/débiles

**Files:**
- Modify: `core/dataset.py` (`"HxG": "Home_xG", "AxG": "Away_xG"` en `COLUMN_MAP`)
- Create: `app/services/team_service.py`
- Test: `tests/test_team_service.py`

**Interfaces:**
- Produces, todas puras sobre `df` = `core.dataset.load_season(...)` (jugados y pendientes):
  - `forma(df, equipo: str, n: int = 5) -> list[dict]` — más antiguo primero; cada dict
    `{"letra": "V"|"E"|"D", "rival": str, "local": bool, "marcador": "2-0" (goles del equipo primero), "fecha": "YYYY-MM-DD"}`.
  - `cabecera(df, equipo: str, hoy: date) -> dict` — `{"posicion": int|None, "puntos", "v", "e", "d", "gf", "gc", "jugados": int, "proximo": dict|None}`;
    `proximo = {"rival", "local", "fecha", "jornada"}`. Posición y números desde
    `core.features.build_state(played_only(df)).standings()`.
  - `fuertes_debiles(df, equipo: str, n: int = 3) -> {"fuertes": list[dict], "debiles": list[dict]}`;
    cada dict `{"clave", "texto", "valor": float, "media": float, "pct": float, "pct_media": float, "formato": str}`.

- [ ] **Step 1: Tests** (con la fixture 25/26 de `data_root`; `load_season("SP1", "2526", download=False)`)

```python
def test_forma_ultimos_cinco_en_orden(df):
    f = forma(df, "Betis")
    assert len(f) == 5 and [x["fecha"] for x in f] == sorted(x["fecha"] for x in f)
    assert all(x["letra"] in "VED" for x in f)

def test_forma_con_menos_de_cinco_partidos(df):
    corto = df[df["Date"] <= df["Date"].min() + pd.Timedelta(days=8)]
    assert 0 < len(forma(corto, "Betis")) < 5

def test_cabecera_coincide_con_league_state(df): ...   # posición == standings() de build_state
def test_proximo_partido_primer_pendiente_desde_hoy(media_temporada): ...  # usa la fixture de test_prediccion (muévela a conftest)
def test_fuertes_y_debiles_son_tres_y_tres_con_signo_correcto(df):
    r = fuertes_debiles(df, "Barcelona")
    assert len(r["fuertes"]) == 3 and len(r["debiles"]) <= 3
    gf = next((m for m in r["fuertes"] if m["clave"] == "goles_favor"), None)
    assert gf is None or gf["valor"] > gf["media"]
def test_goles_encajados_menos_es_mejor(df): ...   # el equipo con menos GC/partido tiene "goles_contra" en fuertes
def test_xg_solo_si_hay_columnas(df):
    assert not any(m["clave"].startswith("xg") for lado in fuertes_debiles(df, "Betis").values() for m in lado)
    con_xg = df.assign(Home_xG=1.0 * df["Home_Goals"], Away_xG=1.0 * df["Away_Goals"])
    claves = {m["clave"] for lado in fuertes_debiles(con_xg, "Barcelona", n=20).values() for m in lado}
    assert "xg_favor" in claves
```

  Mueve la fixture `media_temporada` de `tests/test_prediccion.py` a `tests/conftest.py` (no autouse)
  para reutilizarla.
- [ ] **Step 2: Ver que fallan.**
- [ ] **Step 3: Implementar.** Medidas (por partido jugado; `menos` = menos es mejor):

| clave | valor | sentido | texto fuerte / débil |
|---|---|---|---|
| goles_favor | GF | más | Mucho gol / Le cuesta marcar |
| goles_contra | GC | menos | Defensa sólida / Encaja mucho |
| goles_favor_casa | GF en casa | más | Golea en casa / Poco gol en casa |
| goles_contra_casa | GC en casa | menos | Muy sólido en casa / Frágil en casa |
| goles_favor_fuera | GF fuera | más | Marca a domicilio / Sin gol fuera |
| goles_contra_fuera | GC fuera | menos | Fiable a domicilio / Encaja mucho fuera |
| tiros | tiros | más | Dispara mucho / Apenas dispara |
| punteria | tiros a puerta / tiros | más | Mucha puntería / Poca puntería |
| conversion | goles / tiros a puerta | más | Muy eficaz de cara a puerta / Falla ocasiones claras |
| contra_segunda | GC − GC al descanso | menos | Aguanta las segundas partes / Se cae en las segundas partes |
| corners | córners | más | Mucho juego por banda / Pocos córners |
| tarjetas | amarillas + rojas | menos | Juego limpio / Muchas tarjetas |
| xg_favor | xG | más | Genera mucho peligro / Genera poco peligro |
| xg_contra | xG rival | menos | Concede poco / Concede mucho peligro |
| goles_menos_xg | GF − xG | más | Marca más de lo que genera / Marca menos de lo que genera |

  Una medida se omite si faltan sus columnas o son todas NaN. Para cada medida: valor de los 20
  equipos con ≥1 partido, `z = (valor − media) / std` (omitir si `std == 0`), invertido si `menos`.
  Fuertes = las `n` de mayor `z > 0`; débiles = las `n` de menor `z < 0`. `pct` y `pct_media` =
  posición del valor y de la media entre el mínimo y el máximo de la liga (0–100), invertidos si
  `menos`, para que barra larga = mejor. `formato`: `"0.0"` para medias, `"0%"` para ratios.
- [ ] **Step 4: Tests en verde** — `pytest tests/test_team_service.py tests/test_feature_parity.py -q`.
- [ ] **Step 5: Commit** — `git commit -m "feat: forma y puntos fuertes/débiles del equipo"`

---

### Task 6: Posible 11 y mejores jugadores

**Files:**
- Modify: `app/services/team_service.py`
- Test: `tests/test_team_service.py`

**Interfaces:**
- Consumes: DataFrame de `load_lineups` (Task 3).
- Produces: `posible_once(lineups, equipo: str, n_partidos: int = 5) -> dict` →
  `{"partidos": int, "jugadores": [{"nombre", "posicion", "x": float, "y": float, "titularidades": int, "aviso_amarillas": bool, "expulsado": bool}]}`.
- Produces: `mejores_jugadores(lineups, equipo: str, n: int = 3) -> dict[str, list[tuple[str, float]]]`
  con claves en este orden: `"Goles"`, `"Asistencias"`, `"Ocasiones (xG)"`, `"Pases clave"`,
  `"Minutos"`, `"Tarjetas"`.

- [ ] **Step 1: Tests** (con `mock_understat` + `download_understat` + `load_lineups`)

```python
def test_once_tiene_once_con_un_portero(lineups):
    o = posible_once(lineups, "Betis")
    assert len(o["jugadores"]) == 11 and o["partidos"] == 5
    assert sum(j["posicion"] == "GK" for j in o["jugadores"]) == 1
    assert all(0 < j["x"] < 100 and 0 < j["y"] < 100 for j in o["jugadores"])

def test_once_con_un_solo_partido(lineups):
    uno = lineups[lineups["match_id"] == lineups["match_id"].max()]
    assert len(posible_once(uno, "Betis")["jugadores"]) == 11

def test_once_prefiere_los_mas_titulares(lineups): ...   # el más titular en los 5 entra siempre
def test_aviso_amarillas_con_cuatro(lineups): ...        # fuerza yellow para que un jugador sume 4 → aviso; 5 → no
def test_expulsado_en_el_ultimo_partido(lineups): ...
def test_mejores_jugadores_ordenados_y_sin_ceros(lineups):
    m = mejores_jugadores(lineups, "Betis")
    assert list(m) == ["Goles", "Asistencias", "Ocasiones (xG)", "Pases clave", "Minutos", "Tarjetas"]
    assert all(v > 0 for lista in m.values() for _, v in lista)
    assert all([v for _, v in l] == sorted((v for _, v in l), reverse=True) for l in m.values())
def test_traspasado_solo_cuenta_con_este_equipo(lineups): ...  # mismo player_id con filas de otro team: no suma
```

- [ ] **Step 2: Ver que fallan.**
- [ ] **Step 3: Implementar.** Selección: últimos `n_partidos` partidos del equipo por fecha;
  candidatos = titulares; orden `(titularidades desc, titular en el más reciente desc, minutos desc)`;
  primero el mejor `GK`, luego los 10 mejores no porteros. Posición = la de su titularidad más
  reciente. Coordenadas (y desde arriba, ataque arriba): `GK` 90 · `D*` 72 · `DM*` 58 · `M*` 46 ·
  `AM*` 30 · `FW*` 13. Dentro de cada fila, ordena por sufijo (`L` < `C`/sin sufijo < `R`) y reparte
  `x = (i + 1) * 100 / (k + 1)`. Avisos: amarillas de la temporada con este equipo `% 5 == 4`;
  `expulsado` = roja en el partido más reciente del equipo. Mejores: suma por `player_id` de las
  filas del equipo (Tarjetas = amarillas + rojas); se descartan valores 0; xG redondeado a 1 decimal.
- [ ] **Step 4: Tests en verde.**
- [ ] **Step 5: Commit** — `git commit -m "feat: posible 11 y mejores jugadores"`

---

### Task 7: Palmarés

**Files:**
- Create: `config/palmares.yaml`
- Modify: `app/services/team_service.py`
- Test: `tests/test_team_service.py`

**Interfaces:**
- Produces: `palmares(equipo: str) -> list[dict] | None` — `None` si el equipo no está en el fichero;
  si está, `[{"competicion": str (etiqueta), "titulos": int, "ultimo": str}]` en el orden de
  `competiciones`, solo las que tienen títulos (lista vacía = sin títulos).

- [ ] **Step 1: Tests** — `test_palmares_orden_y_sin_ceros` (Real Madrid: primera entrada "Liga",
  todas con `titulos > 0`), `test_palmares_vacio` (un equipo con `{}`, p. ej. Getafe → `[]`),
  `test_palmares_ausente` (`"Recien Ascendido"` → `None`), `test_palmares_cubre_la_temporada_actual`
  (los 20 equipos de 26/27 de Task 3/Understat están en el fichero:
  Alaves, Ath Bilbao, Ath Madrid, Barcelona, Celta, La Coruna, Elche, Espanol, Getafe, Levante,
  Malaga, Osasuna, Santander, Vallecano, Betis, Real Madrid, Sociedad, Sevilla, Valencia, Villarreal).
- [ ] **Step 2: Ver que fallan.**
- [ ] **Step 3: Rellenar `config/palmares.yaml`** con WebFetch a la Wikipedia en español de cada
  club (sección Palmarés). Estructura:

```yaml
# Regla: solo títulos reconocidos por la RFEF (nacionales) y la UEFA (europeos), recuentos de la
# Wikipedia en español. Copa de Ferias y Copa Eva Duarte en filas propias; la Copa de la
# Coronación de 1902 no cuenta. Última revisión: 2026-10-19.
competiciones:
  liga: Liga
  copa: Copa del Rey
  supercopa: Supercopa de España
  champions: Copa de Europa
  uefa: UEFA / Europa League
  conference: Conference League
  recopa: Recopa
  supercopa_europa: Supercopa de Europa
  copa_ferias: Copa de Ferias
  eva_duarte: Copa Eva Duarte
equipos:
  Betis:
    liga: {titulos: 1, ultimo: "1934/35"}
    copa: {titulos: 3, ultimo: "2021/22"}
  Getafe: {}
```

  Rellena también, si es rápido, los otros 9 equipos de `ESCUDOS`. Si una cifra no está clara en la
  página, no la inventes: deja el equipo fuera (→ «Palmarés no disponible») y anótalo en el commit.
  Carga el YAML una vez (módulo) con `yaml.safe_load`, ruta relativa a la raíz del repo resuelta
  desde `__file__`.
- [ ] **Step 4: Tests en verde.**
- [ ] **Step 5: Commit** — `git commit -m "feat: palmarés de los clubes"`

---

### Task 8: Página de la ficha, enlaces y documentación

**Files:**
- Modify: `app/services/team_service.py` (orquestación async)
- Create: `app/routers/equipo.py`, `app/templates/equipo.html`
- Modify: `app/main.py`, `app/templates/404.html`, `app/static/css/retro.css`,
  `app/templates/partials/standings_table.html`, `partials/jornada.html`, `partials/prediccion.html`,
  `CLAUDE.md`
- Test: `tests/test_equipo.py`

**Interfaces:**
- Consumes: todo lo de las Tasks 1, 3, 5, 6 y 7.
- Produces: `async ficha(db, league_code: str, equipo: str) -> dict | None` (`None` = equipo no
  válido) con claves `equipo, season, estado (EstadoTemporada), cabecera, forma, fuertes_debiles,
  once, mejores, palmares, aviso_jugadores (str|None)`. Las lecturas de pandas/JSON en
  `asyncio.to_thread`.
- Produces: `GET /equipo/{nombre}` → 200 `equipo.html` o 404 `404.html`.

- [ ] **Step 1: Tests** — dos fixtures locales en `tests/test_equipo.py`: `poblada` (`mock_download`
  y `update_league_data` sobre una conexión a `temp_db`; Understat queda offline → sin caché) y
  `poblada_con_jugadores` (igual, pero pidiendo también `mock_understat` antes de actualizar). En los
  tests de abajo, donde pone `poblada, mock_understat` usa `poblada_con_jugadores`.

```python
async def test_ficha_200_con_todas_las_ventanas(client, poblada, mock_understat): ...
    # 200; contiene "Estado de forma", "Puntos fuertes", "Posible 11", "Mejores jugadores", "Palmarés", 'width="64"'
async def test_ficha_nombre_con_espacio(client, poblada):
    assert (await client.get("/equipo/Real%20Madrid")).status_code == 200
async def test_equipo_inexistente_404(client, poblada):
    r = await client.get("/equipo/Narnia"); assert r.status_code == 404 and "Equipo no encontrado" in r.text
async def test_ficha_sin_understat(client, poblada):   # sin mock_understat
    assert "Datos de jugadores no disponibles" in (await client.get("/equipo/Betis")).text
async def test_ficha_sin_cuotas_ni_ppp(client, poblada, mock_understat):
    t = (await client.get("/equipo/Betis")).text
    assert "cuota" not in t.lower() and "apuesta" not in t.lower() and "pts/partido" not in t
async def test_ficha_sin_datos_usa_temporada_anterior(client): ...
    # matches vacío para 2526 pero con filas 2425 → /equipo/Betis 200, sin ventana de forma (mensaje "Aún no ha jugado")
async def test_finalizada_once_mas_habitual(client, poblada, mock_understat):
    assert "11 más habitual" in (await client.get("/equipo/Betis")).text   # la fixture 25/26 está completa
async def test_enlaces_a_la_ficha(client, poblada, ...):  # la clasificación enlaza a /equipo/Ath%20Madrid
```

- [ ] **Step 2: Ver que fallan.**
- [ ] **Step 3: `ficha()`** — equipos válidos: `SELECT DISTINCT` de `teams` unidos a `matches` de la
  temporada actual; si no hay filas, de la temporada más reciente con filas (y se usa esa temporada
  para `load_season`). Estado con `get_season_state`. `load_season(..., download=False)`; si falla
  con `FileNotFoundError`, cabecera mínima y ventanas de partidos con mensaje. `load_lineups`; si
  `FileNotFoundError`, `aviso_jugadores = "Datos de jugadores no disponibles: pulsa ACTUALIZAR"`.
  `hoy = date.today()`.
- [ ] **Step 4: Router y plantilla.** Router `prefix="/equipo"`, `GET /{nombre}`; `None` →
  `404.html` con `titulo="Equipo no encontrado"`, `mensaje=f"«{nombre}» no juega en {liga}."`.
  `404.html`: `{{ titulo or "Liga no encontrada" }}` y la nota de «solo cubre…» solo sin `titulo`.
  Registrar en `app/main.py`. `equipo.html` extiende `base.html`, título
  `{{ equipo|equipo }} · LM LaLiga`, ventanas en el orden del spec (cabecera, Estado de forma,
  Puntos fuertes y débiles, Posible 11 — «contra {rival}» o «11 más habitual» —, Mejores jugadores,
  Palmarés), maquetadas como el boceto aprobado
  (`.superpowers/brainstorm/24706-1792423655/content/ficha.html`, solo en esta máquina) **sin** la
  línea de puntos por partido. Leyendas fijas: bajo la barra «La raya amarilla es la media de la
  liga.»; bajo el campo «Estimado con los titulares de los últimos 5 partidos. No incluye
  lesionados.» y «⚠ = a una amarilla de la sanción». Palmarés: trofeo pixelado (SVG inline 8×8,
  amarillo) + `COMPETICIÓN ×N · último AAAA/AA`; `[]` → «Sin títulos oficiales todavía»; `None` →
  «Palmarés no disponible». Un enlace «Esc Volver a la portada» como en `404.html`. Las camisetas del
  campo usan los colores de `ESCUDOS` del equipo.
- [ ] **Step 5: Enlaces.** En clasificación, jornada y cara a cara, escudo + nombre envueltos en
  `<a class="enlace-equipo" href="/equipo/{{ nombre|urlencode }}">`. Los nombres provisionales
  (`"Ascendido N"`, que no están en `teams`) no llevan enlace: pasa a las plantillas el conjunto de
  equipos válidos o comprueba `nombre in ESCUDOS`.
- [ ] **Step 6: CSS** en `retro.css` para la ficha (rejilla de 2 columnas que pasa a 1 bajo 900 px,
  fichas V/E/D reutilizando `.forma--V/E/D` si encaja, barras, campo, ranking, palmarés, enlaces).
  Revisa en el navegador a 1280 y 375 px, con CRT encendido y apagado.
- [ ] **Step 7: Tests en verde** — `pytest -q` (la suite entera).
- [ ] **Step 8: CLAUDE.md** — añade, en el estilo del fichero: Understat como tercera fuente
  (`scripts/understat.py`, caché en `data/raw/SP1/understat/`, descargada en `update`, la ficha solo
  lee caché, `UNDERSTAT` en `core/equipos.py` para recién ascendidos), la ficha (`/equipo/{nombre}`,
  `team_service.py`), los escudos (`ESCUDOS`, `app/escudos.py`, añadir un ascendido) y
  `config/palmares.yaml` (regla de conteo, editar a mano). Y las fixtures `understat_offline` /
  `mock_understat` en la sección de tests.
- [ ] **Step 9: Commit** — `git commit -m "feat: ficha de equipo"`
