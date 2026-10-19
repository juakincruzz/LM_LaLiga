"""Ficha de equipo: funciones puras sobre la temporada y las alineaciones."""
from datetime import date

import pandas as pd
import pytest

from app.services.team_service import (
    cabecera, forma, fuertes_debiles, mejores_jugadores, palmares, posible_once,
)
from core.dataset import load_season, played_only
from core.features import build_state


@pytest.fixture
def df():
    """La 25/26 completa de la fixture."""
    return load_season("SP1", "2526", download=False)


# --- Estado de forma -------------------------------------------------------


def test_forma_ultimos_cinco_en_orden(df):
    f = forma(df, "Betis")
    assert len(f) == 5
    assert [x["fecha"] for x in f] == sorted(x["fecha"] for x in f)
    assert all(x["letra"] in "VED" for x in f)


def test_forma_marcador_con_los_goles_del_equipo_primero(df):
    ultimo = played_only(df)
    ultimo = ultimo[(ultimo["HomeTeam"] == "Betis") | (ultimo["AwayTeam"] == "Betis")].iloc[-1]
    f = forma(df, "Betis")[-1]
    local = ultimo["HomeTeam"] == "Betis"
    gf, gc = (ultimo["Home_Goals"], ultimo["Away_Goals"]) if local else (ultimo["Away_Goals"], ultimo["Home_Goals"])
    assert f["marcador"] == f"{int(gf)}-{int(gc)}"
    assert f["local"] == local
    assert f["rival"] == (ultimo["AwayTeam"] if local else ultimo["HomeTeam"])
    assert f["letra"] == ("V" if gf > gc else "E" if gf == gc else "D")


def test_forma_con_menos_de_cinco_partidos(df):
    corto = df[df["Date"] <= df["Date"].min() + pd.Timedelta(days=8)]
    assert 0 < len(forma(corto, "Betis")) < 5


def test_forma_sin_partidos_es_lista_vacia(df):
    assert forma(df.iloc[0:0], "Betis") == []


# --- Cabecera ----------------------------------------------------------------


def test_cabecera_coincide_con_league_state(df):
    tabla = build_state(played_only(df)).standings().set_index("Team")
    c = cabecera(df, "Betis", date(2026, 10, 19))
    assert c["posicion"] == tabla.loc["Betis", "Position"]
    assert c["puntos"] == tabla.loc["Betis", "Points"]
    assert (c["v"], c["e"], c["d"]) == tuple(tabla.loc["Betis", ["Wins", "Draws", "Losses"]])
    assert c["jugados"] == 38
    assert c["proximo"] is None  # temporada terminada


def test_proximo_partido_primer_pendiente_desde_hoy(media_temporada):
    df = load_season("SP1", "2526", download=False)
    pendientes = df[df["Result"].isna() & ((df["HomeTeam"] == "Betis") | (df["AwayTeam"] == "Betis"))]

    # Con «hoy» antes de todo, el primer pendiente
    p = cabecera(df, "Betis", date(2000, 1, 1))["proximo"]
    primero = pendientes.iloc[0]
    assert p["fecha"] == primero["Date"].date().isoformat()
    assert p["local"] == (primero["HomeTeam"] == "Betis")
    assert p["jornada"] == primero["Matchday"]

    # Con todas las fechas en el pasado (football-data va con retraso), también el primero
    assert cabecera(df, "Betis", date(2100, 1, 1))["proximo"]["fecha"] == p["fecha"]


def test_cabecera_sin_partidos(df):
    c = cabecera(df.iloc[0:0], "Betis", date(2026, 10, 19))
    assert c["posicion"] is None and c["jugados"] == 0 and c["proximo"] is None


# --- Puntos fuertes y débiles -----------------------------------------------


def _todas(r):
    return [m for lado in r.values() for m in lado]


def test_fuertes_y_debiles_tres_y_tres_con_signo_correcto(df):
    r = fuertes_debiles(df, "Barcelona")
    assert len(r["fuertes"]) == 3 and len(r["debiles"]) <= 3
    for m in r["fuertes"]:
        mejor = m["valor"] < m["media"] if m["menos_es_mejor"] else m["valor"] > m["media"]
        assert mejor, m
    for m in r["debiles"]:
        peor = m["valor"] > m["media"] if m["menos_es_mejor"] else m["valor"] < m["media"]
        assert peor, m


def test_barra_larga_es_mejor(df):
    r = fuertes_debiles(df, "Barcelona")
    assert all(m["pct"] > m["pct_media"] for m in r["fuertes"])
    assert all(m["pct"] < m["pct_media"] for m in r["debiles"])
    assert all(0 <= m["pct"] <= 100 and 0 <= m["pct_media"] <= 100 for m in _todas(r))


def test_goles_encajados_menos_es_mejor(df):
    jugados = played_only(df)
    gc = pd.concat([
        jugados.groupby("HomeTeam")["Away_Goals"].sum(), jugados.groupby("AwayTeam")["Home_Goals"].sum()
    ]).groupby(level=0).sum()
    mejor_defensa = gc.idxmin()
    claves = {m["clave"] for m in fuertes_debiles(df, mejor_defensa, n=20)["fuertes"]}
    assert "goles_contra" in claves


def test_xg_solo_si_hay_columnas(df):
    assert not any(m["clave"].startswith("xg") for m in _todas(fuertes_debiles(df, "Betis", n=20)))
    con_xg = df.assign(Home_xG=df["Home_Shots"] / 10, Away_xG=df["Away_Shots"] / 10)
    claves = {m["clave"] for m in _todas(fuertes_debiles(con_xg, "Barcelona", n=20))}
    assert {"xg_favor", "xg_contra"} & claves


def test_fuertes_debiles_sin_partidos(df):
    assert fuertes_debiles(df.iloc[0:0], "Betis") == {"fuertes": [], "debiles": []}


def test_dataset_normaliza_el_xg_de_football_data():
    from core.dataset import normalize_frame

    crudo = pd.DataFrame({"Date": ["15/08/2026"], "HomeTeam": ["Betis"], "AwayTeam": ["Elche"],
                          "FTHG": [1], "FTAG": [0], "HxG": [1.4], "AxG": [0.6]})
    df = normalize_frame(crudo)
    assert df.loc[0, "Home_xG"] == 1.4 and df.loc[0, "Away_xG"] == 0.6


# --- Posible 11 y mejores jugadores (alineaciones reales del Betis, ver test_understat) ---


@pytest.fixture
def lineups(mock_understat):
    from scripts.understat import download_understat, load_lineups

    download_understat("SP1", "2526")
    return load_lineups("SP1", "2526")


def test_once_tiene_once_con_un_portero(lineups):
    o = posible_once(lineups, "Betis")
    assert len(o["jugadores"]) == 11 and o["partidos"] == 5
    assert sum(j["posicion"] == "GK" for j in o["jugadores"]) == 1
    assert all(0 < j["x"] < 100 and 0 < j["y"] < 100 for j in o["jugadores"])
    assert len({(j["x"], j["y"]) for j in o["jugadores"]}) == 11  # nadie encima de otro


def test_once_con_un_solo_partido(lineups):
    uno = lineups[lineups["match_id"] == lineups["match_id"].max()]
    o = posible_once(uno, "Betis")
    assert len(o["jugadores"]) == 11 and o["partidos"] == 1


def test_once_prefiere_los_mas_titulares(lineups):
    betis = lineups[lineups["team"] == "Betis"]
    ultimos5 = betis.drop_duplicates("match_id").nlargest(5, "date")["match_id"]
    titulares = betis[betis["match_id"].isin(ultimos5) & betis["starter"]]
    conteo = titulares.groupby("player")["match_id"].nunique()
    siempre = set(conteo[conteo == 5].index)
    assert siempre  # en la fixture hay titulares fijos

    nombres = {j["nombre"]: j for j in posible_once(lineups, "Betis")["jugadores"]}
    assert siempre <= set(nombres)
    assert all(nombres[n]["titularidades"] == conteo[n] for n in nombres)
    # Nadie fuera del 11 tiene más titularidades que alguien de campo dentro
    campo = [j["titularidades"] for j in nombres.values() if j["posicion"] != "GK"]
    fuera = conteo.drop(list(nombres))
    assert fuera.empty or fuera.max() <= min(campo)


def test_once_portero_abajo_delantero_arriba(lineups):
    js = posible_once(lineups, "Betis")["jugadores"]
    portero = next(j for j in js if j["posicion"] == "GK")
    assert portero["y"] == max(j["y"] for j in js)


def test_aviso_amarillas_con_cuatro(lineups):
    natan = lineups["player"] == "Natan"
    con_cuatro = lineups.copy()
    con_cuatro.loc[natan, "yellow"] = 0
    con_cuatro.loc[con_cuatro.index[natan][:4], "yellow"] = 1
    jugadores = {j["nombre"]: j for j in posible_once(con_cuatro, "Betis")["jugadores"]}
    assert jugadores["Natan"]["aviso_amarillas"]

    con_cinco = con_cuatro.copy()
    con_cinco.loc[natan, "yellow"] = 1  # 5 o más partidos con amarilla → ya cumplió
    if natan.sum() >= 5:
        jugadores = {j["nombre"]: j for j in posible_once(con_cinco, "Betis")["jugadores"]}
        assert not jugadores["Natan"]["aviso_amarillas"]


def test_expulsado_en_el_ultimo_partido(lineups):
    ultimo = lineups["match_id"] == lineups.loc[lineups["team"] == "Betis", "match_id"].max()
    expulsion = lineups.copy()
    expulsion.loc[ultimo & (expulsion["player"] == "Héctor Bellerín"), "red"] = 1
    jugadores = {j["nombre"]: j for j in posible_once(expulsion, "Betis")["jugadores"]}
    assert jugadores.get("Héctor Bellerín", {"expulsado": True})["expulsado"]
    assert not any(j["expulsado"] for n, j in jugadores.items() if n != "Héctor Bellerín")


def test_once_sin_alineaciones(lineups):
    assert posible_once(lineups.iloc[0:0], "Betis") == {"partidos": 0, "jugadores": []}


def test_mejores_jugadores_ordenados_y_sin_ceros(lineups):
    m = mejores_jugadores(lineups, "Betis")
    assert list(m) == ["Goles", "Asistencias", "Ocasiones (xG)", "Pases clave", "Minutos", "Tarjetas"]
    assert all(len(lista) <= 3 for lista in m.values())
    assert all(v > 0 for lista in m.values() for _, v in lista)
    assert all([v for _, v in lista] == sorted((v for _, v in lista), reverse=True) for lista in m.values())
    assert m["Tarjetas"][0] == ("Diego Llorente", 2)


def test_traspasado_solo_cuenta_con_este_equipo(lineups):
    goleador, goles = mejores_jugadores(lineups, "Betis")["Goles"][0]
    fila = lineups[lineups["player"] == goleador].iloc[0].copy()
    fila["team"], fila["opponent"], fila["goals"], fila["match_id"] = "Getafe", "Betis", 10, 1
    con_otro_club = pd.concat([lineups, fila.to_frame().T], ignore_index=True)
    assert dict(mejores_jugadores(con_otro_club, "Betis")["Goles"])[goleador] == goles


# --- Palmarés ---------------------------------------------------------------

EQUIPOS_2627 = ["Alaves", "Ath Bilbao", "Ath Madrid", "Barcelona", "Celta", "La Coruna", "Elche",
                "Espanol", "Getafe", "Levante", "Malaga", "Osasuna", "Santander", "Vallecano", "Betis",
                "Real Madrid", "Sociedad", "Sevilla", "Valencia", "Villarreal"]


def test_palmares_orden_y_sin_ceros():
    p = palmares("Real Madrid")
    assert p[0]["competicion"] == "Liga"
    assert all(t["titulos"] > 0 and t["ultimo"] for t in p)
    assert any(t["competicion"] == "Copa de Europa" for t in p)


def test_palmares_vacio():
    assert palmares("Getafe") == []


def test_palmares_ausente():
    assert palmares("Recien Ascendido") is None


def test_palmares_cubre_la_temporada_actual():
    assert all(palmares(e) is not None for e in EQUIPOS_2627)
