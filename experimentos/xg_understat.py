"""Estudio: ¿mejora el modelo con la forma continua de xG (Understat)?

    python experimentos/xg_understat.py     # desde la raíz del repo

Descarga (una petición por temporada) el JSON de liga de Understat a la caché
de producción, data/raw/SP1/understat/liga_{año}.json, que trae el xG de cada
partido. Se usa SIEMPRE el xG de Understat, nunca el de football-data: son
modelos de xG distintos (Understat da ~0,2 goles más al local).

Variantes, todas sobre las 13 features de producción y el mismo XGBoost:
  base           las 13 de producción (Elo + forma de goles y tiros a puerta)
  +xG(α)         + media exponencial de xG a favor/en contra de cada equipo y
                 su diferencia (5 features más)
  xG por tiros(α) la forma de tiros a puerta sustituida por la de xG (13)

α elegido en validación (17/18–21/22) y evaluado en test (22/23–25/26 y lo
jugado de la actual), walk-forward.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EXCLUDED_SEASONS, get_training_seasons  # noqa: E402
from core.dataset import load_seasons, played_only  # noqa: E402
from core.equipos import desde_understat  # noqa: E402
from core.features import build_training_matrix  # noqa: E402
from models.predictor import LeaguePredictor  # noqa: E402
from scripts.understat import URL_LIGA, _anio, _get, understat_dir  # noqa: E402

VALIDACION = ["1718", "1819", "1920", "2122"]
TEST = ["2223", "2324", "2425", "2526"]
XG_INICIAL = {"favor": 1.2, "contra": 1.5}  # como la forma de goles: un recién llegado, algo peor que la media


def xg_understat(temporadas: list[str]) -> pd.DataFrame:
    filas = []
    for season in temporadas:
        ruta = understat_dir("SP1") / f"liga_{_anio(season)}.json"
        if not ruta.exists():
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_text(json.dumps({"dates": _get(URL_LIGA.format(anio=_anio(season)))["dates"]}))
            time.sleep(0.5)
        for p in json.loads(ruta.read_text())["dates"]:
            if p["isResult"]:
                filas.append({"Season": season, "HomeTeam": desde_understat(p["h"]["title"]),
                              "AwayTeam": desde_understat(p["a"]["title"]),
                              "xg_h": float(p["xG"]["h"]), "xg_a": float(p["xG"]["a"])})
    return pd.DataFrame(filas)


def forma_xg(df: pd.DataFrame, alfa: float) -> pd.DataFrame:
    """Media exponencial de xG a favor/en contra, previa a cada partido, sin
    reinicio de temporada (la misma idea que la forma de goles de LeagueState)."""
    ew, filas = {}, []
    for row in df.itertuples(index=False):
        loc = ew.setdefault(row.HomeTeam, dict(XG_INICIAL))
        vis = ew.setdefault(row.AwayTeam, dict(XG_INICIAL))
        filas.append({"Home_EW_xG_For": loc["favor"], "Home_EW_xG_Against": loc["contra"],
                      "Away_EW_xG_For": vis["favor"], "Away_EW_xG_Against": vis["contra"],
                      "EW_xG_Diff": (loc["favor"] - loc["contra"]) - (vis["favor"] - vis["contra"])})
        for estado, a_favor, en_contra in ((loc, row.xg_h, row.xg_a), (vis, row.xg_a, row.xg_h)):
            estado["favor"] += alfa * (a_favor - estado["favor"])
            estado["contra"] += alfa * (en_contra - estado["contra"])
    return pd.DataFrame(filas)


def evaluar(df, X, y):
    temporadas = [s for s in get_training_seasons() if s not in EXCLUDED_SEASONS]
    proba = np.full((len(df), 3), np.nan)
    for test in temporadas[temporadas.index("1718"):]:
        train = df["Season"].isin(temporadas[: temporadas.index(test)]).to_numpy()
        prueba = (df["Season"] == test).to_numpy()
        if prueba.any():
            modelo = LeaguePredictor()
            modelo.train_xgboost(X[train], df["Target"][train])
            proba[prueba] = modelo.proba_ordenada(X[prueba])

    n = df.groupby("Season").cumcount().to_numpy()
    res = {}
    for nombre, temps in (("val", VALIDACION), ("test", TEST), ("actual", [get_training_seasons()[-1]])):
        m = df["Season"].isin(temps).to_numpy()
        res[nombre] = log_loss(y[m], proba[m], labels=[0, 1, 2])
        res[f"{nombre}_acierto"] = float((proba[m].argmax(axis=1) == y[m]).mean())
        if nombre == "test":
            res["test_j1_6"] = log_loss(y[m & (n < 60)], proba[m & (n < 60)], labels=[0, 1, 2])
            for t in TEST:
                mt = (df["Season"] == t).to_numpy()
                res[t] = log_loss(y[mt], proba[mt], labels=[0, 1, 2])
    return res


def main():
    temporadas = get_training_seasons()
    X_base, _, df = build_training_matrix(played_only(load_seasons("SP1", temporadas, download=False)))
    df = df.drop(columns=X_base.columns)
    xg = xg_understat(temporadas)
    df = df.merge(xg, on=["Season", "HomeTeam", "AwayTeam"], how="left", validate="one_to_one")
    assert df["xg_h"].notna().all(), f"{df['xg_h'].isna().sum()} partidos sin xG de Understat"
    y = df["Result"].map({"A": 0, "D": 1, "H": 2}).to_numpy()
    tiros = [c for c in X_base.columns if "SoT" in c]

    filas = [{"variante": "base (13)", "alfa": None, **evaluar(df, X_base, y)}]
    print(filas[-1], file=sys.stderr)
    for alfa in (0.06, 0.12, 0.2):
        W = forma_xg(df, alfa)
        filas.append({"variante": "+xG (18)", "alfa": alfa, **evaluar(df, pd.concat([X_base, W], axis=1), y)})
        print(filas[-1], file=sys.stderr)
        filas.append({"variante": "xG por tiros (13)", "alfa": alfa,
                      **evaluar(df, pd.concat([X_base.drop(columns=tiros), W.drop(columns="EW_xG_Diff")], axis=1)
                                .assign(EW_xG_Diff=W["EW_xG_Diff"]), y)})
        print(filas[-1], file=sys.stderr)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(filas).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
