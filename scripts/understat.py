"""Alineaciones y datos de jugadores desde Understat.

Es el ÚNICO módulo que conoce Understat (URLs, formato, nombres). El resto del
proyecto solo ve `download_understat` y el DataFrame de `load_lineups`: para
cambiar de fuente (p. ej. una API de pago al publicar), basta con
reimplementar esas dos funciones.

Understat no tiene API oficial; estos son los endpoints JSON que usa su propia
web. La caché vive en data/raw/{liga}/understat/:
  - liga_{año}.json: los partidos de la temporada (se reescribe en cada descarga).
  - partidos/{id}.json: la alineación de cada partido jugado. Un partido jugado
    ya no cambia, así que se descarga una sola vez.
La ficha de equipo solo lee esta caché; descargar es cosa de «Actualizar datos».
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

import scripts.data_manager_generic as dmg
from core.equipos import desde_understat

URL_LIGA = "https://understat.com/getLeagueData/La_liga/{anio}"
URL_PARTIDO = "https://understat.com/getMatchData/{id}"
CABECERAS = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}
PAUSA = 0.4  # segundos entre peticiones, por cortesía con Understat


def understat_dir(league_code: str) -> Path:
    """Resuelve DATA_DIR en cada llamada, para que los tests puedan redirigirlo."""
    return dmg.DATA_DIR / league_code / "understat"


def _anio(season: str) -> int:
    """Understat nombra la temporada por su año de inicio: '2526' -> 2025."""
    return 2000 + int(season[:2])


def _get(url: str) -> dict:
    respuesta = requests.get(url, headers=CABECERAS, timeout=30)
    respuesta.raise_for_status()
    return respuesta.json()


def download_understat(league_code: str, season: str, on_progress=None) -> dict:
    """Descarga los partidos de la temporada y las alineaciones que falten.

    Un fallo al pedir la liga se propaga (no hay nada que hacer sin ella); un
    fallo en un partido concreto solo lo deja en `faltan`, y la siguiente
    descarga lo vuelve a intentar.
    """
    carpeta = understat_dir(league_code)
    (carpeta / "partidos").mkdir(parents=True, exist_ok=True)

    liga = _get(URL_LIGA.format(anio=_anio(season)))
    # Traducir ya: un ascendido sin traducir debe fallar aquí, no en la ficha
    for partido in liga["dates"]:
        desde_understat(partido["h"]["title"])
        desde_understat(partido["a"]["title"])
    (carpeta / f"liga_{_anio(season)}.json").write_text(
        json.dumps({"dates": liga["dates"]}, ensure_ascii=False)
    )

    jugados = [p["id"] for p in liga["dates"] if p["isResult"]]
    pendientes = [i for i in jugados if not (carpeta / "partidos" / f"{i}.json").exists()]
    nuevos = 0
    for n, match_id in enumerate(pendientes, start=1):
        if nuevos:
            time.sleep(PAUSA)
        try:
            datos = _get(URL_PARTIDO.format(id=match_id))
            (carpeta / "partidos" / f"{match_id}.json").write_text(
                json.dumps({"rosters": datos["rosters"]}, ensure_ascii=False)
            )
            nuevos += 1
        except (requests.RequestException, OSError, ValueError, KeyError):
            pass
        if on_progress:
            on_progress(n, len(pendientes))

    faltan = sum(1 for i in jugados if not (carpeta / "partidos" / f"{i}.json").exists())
    return {"partidos": len(jugados), "nuevos": nuevos, "faltan": faltan}


def load_lineups(league_code: str, season: str) -> pd.DataFrame:
    """Una fila por jugador y partido jugado, con los equipos ya en nombres de
    football-data. `FileNotFoundError` si nunca se descargó la temporada."""
    carpeta = understat_dir(league_code)
    fichero_liga = carpeta / f"liga_{_anio(season)}.json"
    if not fichero_liga.exists():
        raise FileNotFoundError(f"No hay datos de Understat en {fichero_liga}.")

    filas = []
    for partido in json.loads(fichero_liga.read_text())["dates"]:
        fichero = carpeta / "partidos" / f"{partido['id']}.json"
        if not partido["isResult"] or not fichero.exists():
            continue
        equipos = {"h": desde_understat(partido["h"]["title"]), "a": desde_understat(partido["a"]["title"])}
        rosters = json.loads(fichero.read_text())["rosters"]
        for lado, rival in (("h", "a"), ("a", "h")):
            for j in rosters[lado].values():
                filas.append({
                    "match_id": int(partido["id"]),
                    "date": partido["datetime"],
                    "team": equipos[lado],
                    "opponent": equipos[rival],
                    "home": lado == "h",
                    "player_id": int(j["player_id"]),
                    "player": j["player"],
                    "position": j["position"],
                    "starter": j["position"] != "Sub",
                    "minutes": int(j["time"]),
                    "goals": int(j["goals"]),
                    "assists": int(j["assists"]),
                    "xg": float(j["xG"]),
                    "key_passes": int(j["key_passes"]),
                    "yellow": int(j["yellow_card"]),
                    "red": int(j["red_card"]),
                })

    columnas = ["match_id", "date", "team", "opponent", "home", "player_id", "player", "position",
                "starter", "minutes", "goals", "assists", "xg", "key_passes", "yellow", "red"]
    df = pd.DataFrame(filas, columns=columnas)
    df["date"] = pd.to_datetime(df["date"])
    return df
