"""Ficha de equipo: estado de forma, fuertes/débiles, posible 11, jugadores y palmarés.

Funciones puras sobre dos DataFrames:
  - la temporada de `core.dataset.load_season` (football-data + calendario);
  - las alineaciones de `scripts.understat.load_lineups`.
La clasificación sale de `LeagueState`, como en el resto del proyecto. Las
medidas de fuertes/débiles son estadística descriptiva para mostrar, no
features del modelo.
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from core.dataset import played_only
from core.features import build_state

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


@dataclass(frozen=True)
class Medida:
    clave: str
    fuerte: str
    debil: str
    menos_es_mejor: bool
    formato: str  # "0.0" (media por partido) o "0%" (proporción)
    calcular: object  # (partidos del equipo) -> float


def _ratio(a: pd.Series, b: pd.Series) -> float:
    return a.sum() / b.sum() if b.sum() else np.nan


MEDIDAS = [
    Medida("goles_favor", "Mucho gol", "Le cuesta marcar", False, "0.0", lambda g: g["gf"].mean()),
    Medida("goles_contra", "Defensa sólida", "Encaja mucho", True, "0.0", lambda g: g["gc"].mean()),
    Medida("goles_favor_casa", "Golea en casa", "Poco gol en casa", False, "0.0",
           lambda g: g.loc[g["local"], "gf"].mean()),
    Medida("goles_contra_casa", "Muy sólido en casa", "Frágil en casa", True, "0.0",
           lambda g: g.loc[g["local"], "gc"].mean()),
    Medida("goles_favor_fuera", "Marca a domicilio", "Sin gol fuera", False, "0.0",
           lambda g: g.loc[~g["local"], "gf"].mean()),
    Medida("goles_contra_fuera", "Fiable a domicilio", "Encaja mucho fuera", True, "0.0",
           lambda g: g.loc[~g["local"], "gc"].mean()),
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
    Medida("goles_menos_xg", "Marca más de lo que genera", "Marca menos de lo que genera", False, "0.0",
           lambda g: (g["gf"] - g["xg"]).mean()),
]


def fuertes_debiles(df: pd.DataFrame, equipo: str, n: int = 3) -> dict:
    """Las `n` medidas en que el equipo más destaca por arriba y por abajo
    respecto a los equipos de la temporada, en desviaciones típicas."""
    largo = _por_equipo(df)
    if equipo not in set(largo["team"]):
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
            "texto": m.fuerte if z > 0 else m.debil,
            "valor": float(valores[equipo]),
            "media": float(media),
            "pct": float(escala(valores[equipo])),
            "pct_media": float(escala(media)),
            "formato": m.formato,
            "menos_es_mejor": m.menos_es_mejor,
        }))

    puntuadas.sort(key=lambda par: par[0], reverse=True)
    fuertes = [d for z, d in puntuadas if z > 0][:n]
    debiles = [d for z, d in reversed(puntuadas) if z < 0][:n]
    return {"fuertes": fuertes, "debiles": debiles}
