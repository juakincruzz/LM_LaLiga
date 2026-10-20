"""Estudio: Elo inicial de los ascendidos con datos de Segunda (SP2).

    python experimentos/elo_ascendidos.py     # desde la raíz del repo

Descarga las temporadas de Segunda que falten a data/raw/SP2/. Variantes del
Elo con que un ascendido empieza su primera temporada en Primera:

  actual       lo de producción: 1430 si es nuevo; si ya estuvo en Primera,
               el Elo que dejó, regresado hacia la media cada verano fuera.
  fijo         1430 siempre.
  ppg(β)       1430 + β · (puntos por partido en Segunda − media de los
               ascendidos de temporadas anteriores).
  compartido(X) Primera y Segunda en un mismo Elo: el que baja lleva su Elo a
               Segunda, el que sube trae el que tenga allí (como clubelo).
               X = Elo inicial de un equipo que nunca estuvo en Primera.

Parámetros elegidos en las temporadas de validación (17/18–21/22) y evaluados
en las de test (22/23–25/26 y lo jugado de la actual), en walk-forward.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.data_manager_generic as dmg  # noqa: E402
from app.config import EXCLUDED_SEASONS, get_training_seasons, previous_season  # noqa: E402
from core.dataset import load_seasons, normalize_frame, played_only, read_csv  # noqa: E402
from core.features import FEATURE_COLS, _ordenar  # noqa: E402
from core.state import (  # noqa: E402
    ELO_ASCENDIDO, ELO_K, ELO_REGRESION, ELO_VENTAJA_LOCAL, LeagueState,
)
from models.predictor import LeaguePredictor  # noqa: E402

VALIDACION = ["1718", "1819", "1920", "2122"]
TEST = ["2223", "2324", "2425", "2526"]


def segunda(season: str) -> pd.DataFrame:
    ruta = dmg.DATA_DIR / "SP2" / f"{season}.csv"
    if not ruta.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"https://www.football-data.co.uk/mmz4281/{season}/SP2.csv", timeout=30)
        r.raise_for_status()
        ruta.write_bytes(r.content)
    return _ordenar(played_only(normalize_frame(read_csv(ruta), season=season)))


def elo_partido(r: dict, local: str, visitante: str, gl: int, gv: int, inicial: float) -> None:
    """La fórmula de core.state.LeagueState.apply_match, sobre un dict."""
    h, a = r.setdefault(local, inicial), r.setdefault(visitante, inicial)
    esperado = 1 / (1 + 10 ** (-(h + ELO_VENTAJA_LOCAL - a) / 400))
    real = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
    dg = abs(gl - gv)
    delta = ELO_K * (1.0 if dg <= 1 else (1.5 if dg == 2 else (11 + dg) / 8)) * (real - esperado)
    r[local], r[visitante] = h + delta, a - delta


def ppg(df: pd.DataFrame) -> dict[str, float]:
    puntos = {}
    for row in df.itertuples(index=False):
        ph, pa = (3, 0) if row.Result == "H" else ((0, 3) if row.Result == "A" else (1, 1))
        for eq, p in ((row.HomeTeam, ph), (row.AwayTeam, pa)):
            puntos.setdefault(eq, []).append(p)
    return {eq: np.mean(p) for eq, p in puntos.items()}


def matriz(df: pd.DataFrame, variante: str, param: float | None):
    """Recorre Primera como build_training_matrix, fijando el Elo de cada
    ascendido al empezar su temporada según la variante.

    Calendario de la variante compartido, al pasar de la temporada S−1 a la S:
      1. Se guarda el Elo de los que bajan al final de S−1 (jugarán Segunda en S).
      2. Cambio de temporada en Primera (regresión del Elo, como siempre).
      3. Se juega en el Elo de Segunda la temporada S−1 de Segunda (ya terminada),
         con los que bajaron al final de S−2 dentro.
      4. Los que suben a S entran con su Elo de Segunda y salen de ese Elo.
      5. Los que bajaron al final de S−1 entran en el Elo de Segunda.
    """
    state = LeagueState()
    pool2: dict[str, float] = {}
    ppg_ascendidos: list[float] = []
    filas, ascendido_en = [], []
    temporada, equipos_ant, nuevos = None, set(), set()
    por_temporada = {s: set(g["HomeTeam"]) | set(g["AwayTeam"]) for s, g in df.groupby("Season")}

    def jugar_segunda(season):
        sp2 = segunda(season)
        media = np.mean(list(pool2.values())) if pool2 else param
        for e in list(pool2):
            pool2[e] += ELO_REGRESION * (media - pool2[e])
        for p in sp2.itertuples(index=False):
            elo_partido(pool2, p.HomeTeam, p.AwayTeam, int(p.Home_Goals), int(p.Away_Goals), param)

    for row in df.itertuples(index=False):
        if row.Season != temporada:
            equipos = por_temporada[row.Season]
            if temporada is None:
                if variante == "compartido":
                    jugar_segunda(previous_season(row.Season))  # calentamiento
            else:
                bajados = {e: state.team(e).elo for e in equipos_ant - equipos}
                state.start_new_season()
                nuevos = equipos - equipos_ant
                anterior = previous_season(row.Season)
                puntos = ppg(segunda(anterior))
                if variante == "compartido":
                    jugar_segunda(anterior)
                for eq in nuevos:
                    if variante == "fijo":
                        state.team(eq).elo = ELO_ASCENDIDO
                    elif variante == "ppg":
                        centro = np.mean(ppg_ascendidos) if ppg_ascendidos else 1.8
                        state.team(eq).elo = ELO_ASCENDIDO + param * (puntos.get(eq, centro) - centro)
                    elif variante == "compartido":
                        state.team(eq).elo = pool2.pop(eq, param)
                ppg_ascendidos += [puntos[e] for e in nuevos if e in puntos]
                if variante == "compartido":
                    pool2.update(bajados)
            temporada, equipos_ant = row.Season, equipos

        filas.append(state.features(row.HomeTeam, row.AwayTeam, row.Date))
        ascendido_en.append(row.HomeTeam in nuevos or row.AwayTeam in nuevos)
        state.apply_match(row.HomeTeam, row.AwayTeam, int(row.Home_Goals), int(row.Away_Goals), row.Result,
                          row.Date, row.Home_ShotsOnTarget, row.Away_ShotsOnTarget)
    return pd.DataFrame(filas, columns=FEATURE_COLS), np.array(ascendido_en)


def evaluar(df, X, ascendido, y):
    temporadas = [s for s in get_training_seasons() if s not in EXCLUDED_SEASONS]
    proba = np.full((len(df), 3), np.nan)
    for test in temporadas[temporadas.index("1718"):]:
        train = df["Season"].isin(temporadas[: temporadas.index(test)]).to_numpy()
        prueba = (df["Season"] == test).to_numpy()
        if not prueba.any():
            continue
        modelo = LeaguePredictor()
        modelo.train_xgboost(X[train], df["Target"][train])
        proba[prueba] = modelo.proba_ordenada(X[prueba])

    n = df.groupby("Season").cumcount().to_numpy()
    res = {}
    for nombre, temps in (("val", VALIDACION), ("test", TEST), ("actual", [get_training_seasons()[-1]])):
        m = df["Season"].isin(temps).to_numpy()
        res[f"{nombre}"] = log_loss(y[m], proba[m], labels=[0, 1, 2])
        res[f"{nombre}_asc"] = log_loss(y[m & ascendido], proba[m & ascendido], labels=[0, 1, 2])
        if nombre == "test":
            res["test_j1_6"] = log_loss(y[m & (n < 60)], proba[m & (n < 60)], labels=[0, 1, 2])
            res["test_asc_j1_6"] = log_loss(y[m & ascendido & (n < 60)], proba[m & ascendido & (n < 60)],
                                            labels=[0, 1, 2])
            res["n_asc"] = int((m & ascendido).sum())
    return res


def main():
    df = _ordenar(played_only(load_seasons("SP1", get_training_seasons(), download=False)))
    df["Target"] = df["Result"].map({"H": 1, "D": 0, "A": -1})
    y = df["Result"].map({"A": 0, "D": 1, "H": 2}).to_numpy()

    variantes = [("actual", None), ("fijo", None)]
    variantes += [("ppg", b) for b in (100, 200, 300, 400)]
    variantes += [("compartido", x) for x in (1300, 1350, 1400, 1450)]

    # La variante «actual» tiene que ser exactamente la matriz de producción
    from core.features import build_training_matrix
    X_prod, _, _ = build_training_matrix(df)
    X_actual, _ = matriz(df, "actual", None)
    pd.testing.assert_frame_equal(X_prod, X_actual.fillna(0), check_dtype=False)

    filas = []
    for variante, param in variantes:
        X, asc = matriz(df, variante, param)
        filas.append({"variante": variante, "param": param, **evaluar(df, X, asc, y)})
        print(filas[-1], file=sys.stderr)

    res = pd.DataFrame(filas)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
