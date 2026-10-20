"""Estudio de calibración del modelo: ¿cuando dice X %, pasa el X %?

    python experimentos/calibracion.py     # desde la raíz del repo

Usa el pipeline de producción (mismas features, mismo XGBoost). Pasos:

1. Predicciones fuera de muestra de cada temporada desde la 17/18, cada una con
   un modelo entrenado SOLO con las anteriores (walk-forward).
2. Diagrama de fiabilidad por resultado (local, empate, visitante) y ECE.
3. Dos correcciones, cada una ajustada solo con las predicciones fuera de
   muestra de temporadas ANTERIORES a la que se evalúa:
     - temperatura: un único parámetro que aplana o afila las probabilidades;
     - logística sobre log-probabilidades (calibración «Dirichlet», 3×3 + sesgo).
   Se evalúan en 22/23–25/26 y en lo jugado de la actual.

La lectura de los resultados está en docs/ROADMAP.md, sección 2.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EXCLUDED_SEASONS, get_current_season, get_training_seasons  # noqa: E402
from models.predictor import LeaguePredictor  # noqa: E402
from scripts.preprocess_generic import preprocess_league  # noqa: E402

NOMBRES = ["visitante", "empate", "local"]  # columnas de proba_ordenada
PRIMERA_PREDICHA = "1718"  # antes, el modelo tendría menos de 3 temporadas de historia
EVALUADAS = ["2223", "2324", "2425", "2526"]


def predicciones_fuera_de_muestra():
    temporadas = get_training_seasons()
    X, y, df = preprocess_league("SP1", temporadas)
    cod = y.map({-1: 0, 0: 1, 1: 2}).to_numpy()
    validas = [s for s in temporadas if s not in EXCLUDED_SEASONS]

    proba = np.full((len(df), 3), np.nan)
    for test in validas[validas.index(PRIMERA_PREDICHA):]:
        train = df["Season"].isin(validas[: validas.index(test)]).to_numpy()
        prueba = (df["Season"] == test).to_numpy()
        if not prueba.any():
            continue
        modelo = LeaguePredictor()
        modelo.train_xgboost(X[train], y[train])
        proba[prueba] = modelo.proba_ordenada(X[prueba])
        print(f"  {test}: {prueba.sum()} partidos predichos", file=sys.stderr)
    hay = ~np.isnan(proba).any(axis=1)
    return df.loc[hay, "Season"].to_numpy(), proba[hay], cod[hay]


# --- Métricas ------------------------------------------------------------------


def ece_clase(p, acierto, bins=10):
    """Error de calibración de una clase: media ponderada de |frecuencia − p| por tramo."""
    tramos = np.minimum((p * bins).astype(int), bins - 1)
    return sum(
        (tramos == b).mean() * abs(acierto[tramos == b].mean() - p[tramos == b].mean())
        for b in range(bins) if (tramos == b).any()
    )


def metricas(proba, y):
    onehot = np.eye(3)[y]
    return {
        "logloss": log_loss(y, proba, labels=[0, 1, 2]),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        **{f"ece_{n}": ece_clase(proba[:, k], onehot[:, k]) for k, n in enumerate(NOMBRES)},
    }


def diagrama(proba, y, bins=10):
    """Tabla de fiabilidad por clase: en cada tramo de probabilidad predicha,
    cuántos partidos, la media predicha y la frecuencia real."""
    onehot = np.eye(3)[y]
    filas = []
    for k, nombre in enumerate(NOMBRES):
        tramos = np.minimum((proba[:, k] * bins).astype(int), bins - 1)
        for b in range(bins):
            m = tramos == b
            if m.sum() >= 20:
                filas.append({"resultado": nombre, "tramo": f"{b * 10}-{b * 10 + 10} %", "n": int(m.sum()),
                              "predicho": proba[m, k].mean(), "real": onehot[m, k].mean()})
    return pd.DataFrame(filas)


# --- Calibradores --------------------------------------------------------------


def temperatura(p_ajuste, y_ajuste):
    logp = np.log(np.clip(p_ajuste, 1e-12, 1))

    def coste(t):
        q = np.exp(logp / t)
        return log_loss(y_ajuste, q / q.sum(axis=1, keepdims=True), labels=[0, 1, 2])

    t = minimize_scalar(coste, bounds=(0.3, 3.0), method="bounded").x

    def aplicar(p):
        q = np.exp(np.log(np.clip(p, 1e-12, 1)) / t)
        return q / q.sum(axis=1, keepdims=True)

    return aplicar, t


def dirichlet(p_ajuste, y_ajuste, c=1.0):
    modelo = LogisticRegression(C=c, max_iter=2000).fit(np.log(np.clip(p_ajuste, 1e-12, 1)), y_ajuste)
    return lambda p: modelo.predict_proba(np.log(np.clip(p, 1e-12, 1))), None


def main():
    temporada, proba, y = predicciones_fuera_de_muestra()
    actual = get_current_season()
    evaluadas = EVALUADAS + ([actual] if actual not in EVALUADAS and (temporada == actual).any() else [])

    filas = []
    for test in evaluadas:
        ajuste = temporada < test  # solo predicciones fuera de muestra de temporadas anteriores
        prueba = temporada == test
        filas.append({"temporada": test, "metodo": "sin calibrar", **metricas(proba[prueba], y[prueba])})
        aplicar, t = temperatura(proba[ajuste], y[ajuste])
        filas.append({"temporada": test, "metodo": f"temperatura", "param": t,
                      **metricas(aplicar(proba[prueba]), y[prueba])})
        aplicar, _ = dirichlet(proba[ajuste], y[ajuste])
        filas.append({"temporada": test, "metodo": "dirichlet", **metricas(aplicar(proba[prueba]), y[prueba])})

    res = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    cuatro = res[res["temporada"].isin(EVALUADAS)]
    print("Media 22/23–25/26:")
    print(cuatro.groupby("metodo")[["logloss", "brier", "ece_local", "ece_empate", "ece_visitante"]]
          .mean().round(4).to_string())
    print("\nLog-loss por temporada:")
    print(res.pivot(index="metodo", columns="temporada", values="logloss").round(4).to_string())
    print("\nTemperatura ajustada por temporada:",
          res.loc[res["metodo"] == "temperatura", ["temporada", "param"]].round(3).to_dict("records"))

    m = np.isin(temporada, EVALUADAS)
    print("\nDiagrama de fiabilidad sin calibrar (22/23–25/26):")
    print(diagrama(proba[m], y[m]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
