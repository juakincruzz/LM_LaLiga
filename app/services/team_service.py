"""Ficha de equipo: estado de forma, fuertes/débiles, posible 11, jugadores y palmarés.

Funciones puras sobre dos DataFrames:
  - la temporada de `core.dataset.load_season` (football-data + calendario);
  - las alineaciones de `scripts.understat.load_lineups`.
La clasificación sale de `LeagueState`, como en el resto del proyecto. Las
medidas de fuertes/débiles son estadística descriptiva para mostrar, no
features del modelo.
"""
import asyncio
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.config import LEAGUE_CODE, get_current_season, get_league_config
from app.services.season_state import get_season_state
from core.dataset import load_season, played_only
from core.equipos import ESCUDOS
from core.features import build_state
from scripts.understat import load_lineups

LETRAS = {1: "V", 0: "E", -1: "D"}


# --- Partidos del equipo -----------------------------------------------------


def _col(df: pd.DataFrame, nombre: str) -> pd.Series:
    """La columna si existe; si no, NaN (temporadas o fuentes sin ese dato)."""
    return pd.to_numeric(df[nombre], errors="coerce") if nombre in df.columns else pd.Series(np.nan, index=df.index)


def _por_equipo(df: pd.DataFrame) -> pd.DataFrame:
    """Dos filas por partido jugado, una por equipo, con sus datos y los del rival."""
    jugados = played_only(df) if len(df) else df
    if jugados.empty:
        return pd.DataFrame(columns=["team", "rival", "local", "fecha", "gf", "gc"])

    def lado(propio: str, ajeno: str, local: bool) -> pd.DataFrame:
        return pd.DataFrame({
            "team": jugados[f"{propio}Team"],
            "rival": jugados[f"{ajeno}Team"],
            "local": local,
            "fecha": jugados["Date"],
            "gf": _col(jugados, f"{propio}_Goals"),
            "gc": _col(jugados, f"{ajeno}_Goals"),
            "gc_ht": _col(jugados, f"{ajeno}_Goals_HT"),
            "tiros": _col(jugados, f"{propio}_Shots"),
            "tiros_puerta": _col(jugados, f"{propio}_ShotsOnTarget"),
            "corners": _col(jugados, f"{propio}_Corners"),
            "tarjetas": _col(jugados, f"{propio}_Yellows") + _col(jugados, f"{propio}_Reds"),
            "xg": _col(jugados, f"{propio}_xG"),
            "xga": _col(jugados, f"{ajeno}_xG"),
        })

    largo = pd.concat([lado("Home", "Away", True), lado("Away", "Home", False)], ignore_index=True)
    return largo.sort_values("fecha", kind="mergesort").reset_index(drop=True)


def forma(df: pd.DataFrame, equipo: str, n: int = 5) -> list[dict]:
    """Los últimos `n` partidos jugados, el más antiguo primero."""
    largo = _por_equipo(df)
    ultimos = largo[largo["team"] == equipo].tail(n)
    return [
        {
            "letra": LETRAS[int(np.sign(r.gf - r.gc))],
            "rival": r.rival,
            "local": bool(r.local),
            "marcador": f"{int(r.gf)}-{int(r.gc)}",
            "fecha": r.fecha.date().isoformat(),
        }
        for r in ultimos.itertuples(index=False)
    ]


def cabecera(df: pd.DataFrame, equipo: str, hoy: date) -> dict:
    """Posición y números de la temporada (de `LeagueState`) y próximo partido."""
    datos = {"posicion": None, "puntos": 0, "v": 0, "e": 0, "d": 0, "gf": 0, "gc": 0, "jugados": 0, "proximo": None}
    if df.empty:
        return datos

    jugados = played_only(df)
    if not jugados.empty:
        tabla = build_state(jugados).standings().set_index("Team")
        if equipo in tabla.index:
            fila = tabla.loc[equipo]
            datos.update(
                posicion=int(fila["Position"]), puntos=int(fila["Points"]), v=int(fila["Wins"]),
                e=int(fila["Draws"]), d=int(fila["Losses"]), gf=int(fila["GF"]), gc=int(fila["GA"]),
                jugados=int(fila["Games"]),
            )

    pendientes = df[df["Result"].isna() & ((df["HomeTeam"] == equipo) | (df["AwayTeam"] == equipo))]
    if not pendientes.empty:
        # El primero desde hoy; si todos quedan atrás (football-data publica
        # con retraso, o hay aplazados), el primero pendiente
        futuros = pendientes[pendientes["Date"].dt.date >= hoy]
        p = (futuros if not futuros.empty else pendientes).iloc[0]
        local = p["HomeTeam"] == equipo
        jornada = p["Matchday"] if "Matchday" in p.index else None
        datos["proximo"] = {
            "rival": p["AwayTeam"] if local else p["HomeTeam"],
            "local": bool(local),
            "fecha": p["Date"].date().isoformat(),
            "jornada": int(jornada) if pd.notna(jornada) else None,
        }
    return datos


# --- Puntos fuertes y débiles ------------------------------------------------


# Para contar como punto fuerte o débil, el equipo tiene que separarse de la
# media de la liga al menos 0,75 desviaciones típicas: sin umbral, «Muchas
# tarjetas» salía con 2,5 por partido frente a 2,25 de media, y con 0,5 aún
# pasaba «Marca menos de lo que genera» con −0,08 frente a 0,08. Y con menos
# de 5 partidos las medias son casi ruido, así que no se muestra nada.
Z_MINIMO = 0.75
PARTIDOS_MINIMOS = 5


@dataclass(frozen=True)
class Medida:
    clave: str
    fuerte: str
    debil: str
    menos_es_mejor: bool
    formato: str  # "0.0"/"0.00" (media por partido) o "0%" (proporción)
    calcular: object  # (partidos del equipo) -> float
    # Medidas que dicen casi lo mismo (goles totales, en casa, fuera): de
    # cada familia sale como mucho una por lado, la más destacada.
    familia: str = ""


def _ratio(a: pd.Series, b: pd.Series) -> float:
    return a.sum() / b.sum() if b.sum() else np.nan


MEDIDAS = [
    Medida("goles_favor", "Mucho gol", "Le cuesta marcar", False, "0.0", lambda g: g["gf"].mean(),
           familia="goles_favor"),
    Medida("goles_contra", "Defensa sólida", "Encaja mucho", True, "0.0", lambda g: g["gc"].mean(),
           familia="goles_contra"),
    Medida("goles_favor_casa", "Golea en casa", "Poco gol en casa", False, "0.0",
           lambda g: g.loc[g["local"], "gf"].mean(), familia="goles_favor"),
    Medida("goles_contra_casa", "Muy sólido en casa", "Frágil en casa", True, "0.0",
           lambda g: g.loc[g["local"], "gc"].mean(), familia="goles_contra"),
    Medida("goles_favor_fuera", "Marca a domicilio", "Sin gol fuera", False, "0.0",
           lambda g: g.loc[~g["local"], "gf"].mean(), familia="goles_favor"),
    Medida("goles_contra_fuera", "Fiable a domicilio", "Encaja mucho fuera", True, "0.0",
           lambda g: g.loc[~g["local"], "gc"].mean(), familia="goles_contra"),
    Medida("tiros", "Dispara mucho", "Apenas dispara", False, "0.0", lambda g: g["tiros"].mean()),
    Medida("punteria", "Mucha puntería", "Poca puntería", False, "0%",
           lambda g: _ratio(g["tiros_puerta"], g["tiros"])),
    Medida("conversion", "Muy eficaz de cara a puerta", "Falla ocasiones claras", False, "0%",
           lambda g: _ratio(g["gf"], g["tiros_puerta"])),
    Medida("contra_segunda", "Aguanta las segundas partes", "Se cae en las segundas partes", True, "0.0",
           lambda g: (g["gc"] - g["gc_ht"]).mean()),
    Medida("corners", "Mucho juego por banda", "Pocos córners", False, "0.0", lambda g: g["corners"].mean()),
    Medida("tarjetas", "Juego limpio", "Muchas tarjetas", True, "0.0", lambda g: g["tarjetas"].mean()),
    Medida("xg_favor", "Genera mucho peligro", "Genera poco peligro", False, "0.0", lambda g: g["xg"].mean()),
    Medida("xg_contra", "Concede poco", "Concede mucho peligro", True, "0.0", lambda g: g["xga"].mean()),
    Medida("goles_menos_xg", "Marca más de lo que genera", "Marca menos de lo que genera", False, "0.00",
           lambda g: (g["gf"] - g["xg"]).mean()),
]


def fuertes_debiles(df: pd.DataFrame, equipo: str, n: int = 3) -> dict:
    """Las `n` medidas en que el equipo más destaca por arriba y por abajo
    respecto a los equipos de la temporada, en desviaciones típicas.

    Solo cuentan las que se separan al menos `Z_MINIMO` de la media, una por
    familia y lado, y nada hasta que el equipo lleva `PARTIDOS_MINIMOS`.
    """
    largo = _por_equipo(df)
    if (largo["team"] == equipo).sum() < PARTIDOS_MINIMOS:
        return {"fuertes": [], "debiles": []}

    puntuadas = []
    grupos = dict(tuple(largo.groupby("team")))
    for m in MEDIDAS:
        with np.errstate(all="ignore"):
            valores = pd.Series({t: m.calcular(g) for t, g in grupos.items()}, dtype=float).dropna()
        if equipo not in valores.index or len(valores) < 2:
            continue
        media, std = valores.mean(), valores.std()
        if not std or np.isnan(std):
            continue
        signo = -1 if m.menos_es_mejor else 1
        z = signo * (valores[equipo] - media) / std

        lo, hi = valores.min(), valores.max()

        def escala(v):
            pct = 100 * (v - lo) / (hi - lo)
            return 100 - pct if m.menos_es_mejor else pct

        puntuadas.append((z, {
            "clave": m.clave,
            "familia": m.familia or m.clave,
            "texto": m.fuerte if z > 0 else m.debil,
            "valor": float(valores[equipo]),
            "media": float(media),
            "pct": float(escala(valores[equipo])),
            "pct_media": float(escala(media)),
            "formato": m.formato,
            "menos_es_mejor": m.menos_es_mejor,
        }))

    puntuadas.sort(key=lambda par: par[0], reverse=True)

    def elegir(candidatas) -> list[dict]:
        elegidas, familias = [], set()
        for d in candidatas:
            if d["familia"] not in familias:
                elegidas.append(d)
                familias.add(d["familia"])
        return elegidas[:n]

    return {
        "fuertes": elegir(d for z, d in puntuadas if z >= Z_MINIMO),
        "debiles": elegir(d for z, d in reversed(puntuadas) if z <= -Z_MINIMO),
    }


# --- Posible 11 y mejores jugadores (alineaciones de Understat) --------------

# Altura en el campo (0 arriba = ataque, 100 abajo = portería propia) según el
# prefijo de la posición de Understat: GK, DR/DC/DL, DMC, MC, AML/AMC, FW…
FILAS = [("GK", 90), ("DM", 58), ("AM", 30), ("FW", 13), ("D", 72), ("M", 46)]
ORDEN_LADO = {"L": 0, "C": 1, "R": 2}
CICLO_AMARILLAS = 5  # en La Liga, cada 5 amarillas un partido de sanción


def _fila(posicion: str) -> int:
    return next((y for prefijo, y in FILAS if posicion.startswith(prefijo)), 46)


def _lado(posicion: str) -> int:
    sufijo = posicion[-1] if len(posicion) > 1 and posicion not in ("GK",) else "C"
    return ORDEN_LADO.get(sufijo, 1)


def _numerico(lineups: pd.DataFrame) -> pd.DataFrame:
    df = lineups.copy()
    for col in ("minutes", "goals", "assists", "xg", "key_passes", "yellow", "red"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["starter"] = df["starter"].astype(bool)
    return df


def posible_once(lineups: pd.DataFrame, equipo: str, n_partidos: int = 5) -> dict:
    """El 11 que más se repite en los últimos `n_partidos` del equipo.

    Es una estimación: titularidades, luego si fue titular en el último
    partido, luego minutos. Un portero obligatorio: si ninguno fue titular en
    el tramo, el último que lo fue en la temporada. Cada jugador va en la
    posición de su titularidad más reciente. No sabe de lesiones.
    """
    df = _numerico(lineups)
    propios = df[df["team"] == equipo]
    if propios.empty:
        return {"partidos": 0, "jugadores": []}

    partidos = propios.drop_duplicates("match_id").sort_values("date", kind="mergesort")["match_id"]
    recientes = list(partidos.tail(n_partidos))
    ultimo = recientes[-1]
    tramo = propios[propios["match_id"].isin(recientes)]
    titulares = tramo[tramo["starter"]].sort_values("date", kind="mergesort")

    resumen = titulares.groupby("player_id").agg(
        nombre=("player", "last"),
        posicion=("position", "last"),
        titularidades=("match_id", "nunique"),
        minutos=("minutes", "sum"),
    )
    resumen["en_ultimo"] = resumen.index.isin(titulares.loc[titulares["match_id"] == ultimo, "player_id"])
    resumen = resumen.sort_values(["titularidades", "en_ultimo", "minutos"], ascending=False, kind="mergesort")

    porteros = resumen[resumen["posicion"] == "GK"].head(1)
    if porteros.empty:
        # Ningún portero titular en el tramo (p. ej. datos incompletos): el
        # último que lo fue en la temporada. Sin ninguno, 11 de campo.
        de_temporada = propios[propios["starter"] & (propios["position"] == "GK")]
        if not de_temporada.empty:
            ultimo_gk = de_temporada.sort_values("date", kind="mergesort").iloc[-1]
            porteros = pd.DataFrame(
                {"nombre": [ultimo_gk["player"]], "posicion": ["GK"], "titularidades": [0],
                 "minutos": [0], "en_ultimo": [False]},
                index=pd.Index([ultimo_gk["player_id"]], name="player_id"),
            )
    campo = resumen[resumen["posicion"] != "GK"].head(11 - len(porteros))
    once = pd.concat([porteros, campo])

    amarillas = propios.groupby("player_id")["yellow"].sum()
    expulsados = set(propios.loc[(propios["match_id"] == ultimo) & (propios["red"] > 0), "player_id"])

    jugadores = []
    for pid, j in once.iterrows():
        jugadores.append({
            "nombre": j["nombre"],
            "posicion": j["posicion"],
            "y": float(_fila(j["posicion"])),
            "titularidades": int(j["titularidades"]),
            "aviso_amarillas": int(amarillas.get(pid, 0)) % CICLO_AMARILLAS == CICLO_AMARILLAS - 1,
            "expulsado": pid in expulsados,
        })

    # Reparto horizontal dentro de cada línea: izquierda, centro, derecha
    for y in {j["y"] for j in jugadores}:
        linea = sorted((j for j in jugadores if j["y"] == y), key=lambda j: _lado(j["posicion"]))
        for i, j in enumerate(linea):
            j["x"] = (i + 1) * 100 / (len(linea) + 1)

    return {"partidos": len(recientes), "jugadores": jugadores}


CATEGORIAS = {
    "Goles": "goals",
    "Asistencias": "assists",
    "Ocasiones (xG)": "xg",
    "Pases clave": "key_passes",
    "Minutos": "minutes",
    "Tarjetas": "tarjetas",
}


def mejores_jugadores(lineups: pd.DataFrame, equipo: str, n: int = 3) -> dict[str, list[tuple[str, float]]]:
    """Top `n` del equipo por categoría, sumando solo lo jugado CON este equipo
    (un traspasado a mitad de temporada no arrastra lo del club anterior)."""
    df = _numerico(lineups)
    propios = df[df["team"] == equipo].assign(tarjetas=lambda d: d["yellow"] + d["red"])
    if propios.empty:
        return {cat: [] for cat in CATEGORIAS}

    totales = propios.groupby("player_id").agg(
        nombre=("player", "last"), **{col: (col, "sum") for col in CATEGORIAS.values()}
    )
    totales["xg"] = totales["xg"].round(1)

    mejores = {}
    for categoria, col in CATEGORIAS.items():
        top = totales[totales[col] > 0].sort_values(col, ascending=False, kind="mergesort").head(n)
        mejores[categoria] = [
            (fila["nombre"], float(fila[col]) if col == "xg" else int(fila[col])) for _, fila in top.iterrows()
        ]
    return mejores


# --- Palmarés ------------------------------------------------------------------

RUTA_PALMARES = Path(__file__).resolve().parents[2] / "config" / "palmares.yaml"
with open(RUTA_PALMARES, encoding="utf-8") as _f:
    _PALMARES = yaml.safe_load(_f)


def palmares(equipo: str) -> list[dict] | None:
    """Títulos por competición, en el orden del fichero y sin las que están a
    cero. `[]` = sin títulos oficiales; `None` = el equipo no está en el fichero."""
    titulos = _PALMARES["equipos"].get(equipo)
    if titulos is None:
        return None
    return [
        {"competicion": etiqueta, "titulos": titulos[clave]["titulos"], "ultimo": titulos[clave]["ultimo"]}
        for clave, etiqueta in _PALMARES["competiciones"].items()
        if titulos.get(clave, {}).get("titulos")
    ]


# --- La ficha completa ---------------------------------------------------------


async def _equipos_de(db, league_code: str, season: str) -> set[str]:
    cursor = await db.execute(
        """
        SELECT DISTINCT t.name
          FROM matches m
          JOIN teams t ON t.id IN (m.home_team_id, m.away_team_id)
         WHERE m.league_code = ? AND m.season = ?
        """,
        (league_code, season),
    )
    return {fila[0] for fila in await cursor.fetchall()}


async def _temporada_con_datos(db, league_code: str) -> str | None:
    """La actual si tiene partidos; si no (SIN_DATOS), la más reciente que tenga."""
    actual = get_current_season()
    cursor = await db.execute(
        "SELECT season FROM matches WHERE league_code = ? AND season IS NOT NULL "
        "GROUP BY season ORDER BY season = ? DESC, MAX(date) DESC LIMIT 1",
        (league_code, actual),
    )
    fila = await cursor.fetchone()
    return fila[0] if fila else None


def _cargar(league_code: str, season: str):
    """Lecturas de disco (pandas/JSON), fuera del event loop."""
    try:
        df = load_season(league_code, season, download=False)
    except FileNotFoundError:
        df = None
    try:
        lineups = load_lineups(league_code, season)
    except FileNotFoundError:
        lineups = None
    return df, lineups


def _ascendidos() -> set[str]:
    """Ascendidos reales de la proyección (los provisionales «Ascendido N» no
    tienen escudo): aún sin partidos en la BD, pero la portada los enlaza."""
    config = get_league_config(LEAGUE_CODE).get("next_season", {})
    return {e for e in config.get("promoted_teams", []) if e in ESCUDOS}


def _componer(df, lineups, equipo: str, es_actual: bool) -> dict:
    """La parte de pandas de la ficha; corre en un hilo, fuera del event loop."""
    hay_partidos = df is not None and es_actual and equipo in set(df["HomeTeam"]) | set(df["AwayTeam"]) \
        and df["Result"].notna().any()
    datos = {
        "es_actual": es_actual,
        "hay_partidos": bool(hay_partidos),
        "cabecera": cabecera(df if df is not None and es_actual else pd.DataFrame(), equipo, date.today()),
        "forma": forma(df, equipo) if hay_partidos else [],
        "fuertes_debiles": fuertes_debiles(df, equipo) if hay_partidos else {"fuertes": [], "debiles": []},
        "once": None,
        "mejores": None,
        "aviso_jugadores": None,
    }
    if lineups is None or not es_actual:
        datos["aviso_jugadores"] = "Datos de jugadores no disponibles: pulsa ACTUALIZAR"
    else:
        datos["once"] = posible_once(lineups, equipo)
        datos["mejores"] = mejores_jugadores(lineups, equipo)
        if not datos["once"]["jugadores"]:
            # Pretemporada (o equipo sin alineaciones): mensaje en las dos
            # ventanas de jugadores, no seis columnas vacías
            datos["once"] = datos["mejores"] = None
            datos["aviso_jugadores"] = "Aún no hay alineaciones de este equipo esta temporada."
    return datos


def _cargar_y_componer(league_code: str, season: str, equipo: str, es_actual: bool) -> dict:
    df, lineups = _cargar(league_code, season)
    return _componer(df, lineups, equipo, es_actual)


async def ficha(db, league_code: str, equipo: str) -> dict | None:
    """Todo lo que pinta /equipo/{nombre}; `None` si el equipo no juega."""
    season = await _temporada_con_datos(db, league_code)
    juega = season is not None and equipo in await _equipos_de(db, league_code, season)
    if not juega and equipo not in _ascendidos():
        return None

    estado = await get_season_state(db, league_code)
    es_actual = season == get_current_season()
    if season is None:
        compuesto = _componer(None, None, equipo, es_actual=False)
    else:
        compuesto = await asyncio.to_thread(_cargar_y_componer, league_code, season, equipo, es_actual)
    return {
        "equipo": equipo,
        "season": season,
        "estado": estado,
        "palmares": palmares(equipo),
        **compuesto,
    }
