"""Estudio de fiabilidad del modelo: validación walk-forward sobre 4 temporadas.

    python experimentos/fiabilidad_modelo.py     # desde la raíz del repo

Descarga a data/raw/SP1/ las temporadas que falten (desde la 14/15). Tarda
unos minutos. Resultados por temporada en experimentos/resultados.csv; la
lectura de los resultados está en docs/ROADMAP.md, sección 2.


Para cada temporada de test T (2223..2526) se entrena SOLO con temporadas
anteriores a T y se evalúa sobre T entera. Se comparan variantes de features y
de modelo con log-loss (la métrica principal: el Monte Carlo muestrea de las
probabilidades), Brier, accuracy y error de calibración (ECE).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.dataset import normalize_frame, read_csv  # noqa: E402
from core.features import build_training_matrix  # noqa: E402
from scripts.data_manager_generic import ensure_league_data  # noqa: E402
TODAS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TESTS = ["2223", "2324", "2425", "2526"]
CLASES = [0, 1, 2]  # A, D, H
COD = {"A": 0, "D": 1, "H": 2}


def cargar():
    marcos = [normalize_frame(read_csv(ensure_league_data("SP1", s)), season=s) for s in TODAS]
    df = pd.concat(marcos, ignore_index=True).sort_values("Date", kind="mergesort").reset_index(drop=True)
    return df.dropna(subset=["Home_Goals", "Away_Goals", "Result"]).reset_index(drop=True)


# ---------------------------------------------------------------- features extra

def elo(df, k=20.0, ventaja=60.0, regresion=0.33, inicial_ascendido=1430.0):
    """Elo clásico con multiplicador por diferencia de goles, arrastrado entre
    temporadas con regresión a la media. Devuelve rating PRE-partido."""
    r, temporada, eh, ea = {}, None, [], []
    vistos_temporada = set()
    for row in df.itertuples(index=False):
        if row.Season != temporada:
            if temporada is not None:
                media = np.mean(list(r.values()))
                r = {t: v + regresion * (media - v) for t, v in r.items()}
            temporada = row.Season
            vistos_temporada = set()
        for t in (row.HomeTeam, row.AwayTeam):
            if t not in r:
                r[t] = inicial_ascendido if len(r) else 1500.0
            vistos_temporada.add(t)
        h, a = r[row.HomeTeam], r[row.AwayTeam]
        eh.append(h)
        ea.append(a)
        esperado = 1 / (1 + 10 ** (-(h + ventaja - a) / 400))
        real = 1.0 if row.Result == "H" else (0.5 if row.Result == "D" else 0.0)
        dg = abs(row.Home_Goals - row.Away_Goals)
        mult = 1.0 if dg <= 1 else (1.5 if dg == 2 else (11 + dg) / 8)
        delta = k * mult * (real - esperado)
        r[row.HomeTeam] = h + delta
        r[row.AwayTeam] = a - delta
    return pd.DataFrame({"Elo_Home": eh, "Elo_Away": ea, "Elo_Diff": np.array(eh) - np.array(ea)})


def ewma(df, alpha=0.12):
    """Medias exponenciales por equipo SIN reinicio de temporada: goles y tiros a
    puerta a favor/en contra. Sin datos -> media de liga."""
    cols = {"g": ("Home_Goals", "Away_Goals"), "sot": ("Home_ShotsOnTarget", "Away_ShotsOnTarget")}
    st = {}
    salida = []
    for row in df.itertuples(index=False):
        f = {}
        for lado, eq in (("H", row.HomeTeam), ("A", row.AwayTeam)):
            s = st.setdefault(eq, {"g_f": 1.2, "g_c": 1.5, "sot_f": 3.8, "sot_c": 4.6})  # ascendido: algo peor que la media
            for k, v in s.items():
                f[f"EW_{lado}_{k}"] = v
        salida.append(f)
        for eq, propio, rival in ((row.HomeTeam, 0, 1), (row.AwayTeam, 1, 0)):
            s = st[eq]
            for nombre, (ch, ca) in cols.items():
                vals = (getattr(row, ch), getattr(row, ca))
                if pd.isna(vals[0]) or pd.isna(vals[1]):
                    continue
                s[f"{nombre}_f"] += alpha * (vals[propio] - s[f"{nombre}_f"])
                s[f"{nombre}_c"] += alpha * (vals[rival] - s[f"{nombre}_c"])
    out = pd.DataFrame(salida)
    out["EW_g_diff"] = (out.EW_H_g_f - out.EW_H_g_c) - (out.EW_A_g_f - out.EW_A_g_c)
    out["EW_sot_diff"] = (out.EW_H_sot_f - out.EW_H_sot_c) - (out.EW_A_sot_f - out.EW_A_sot_c)
    return out


def mercado(df):
    """Probabilidades implícitas (Bet365 y, si existe, media del mercado), sin margen."""
    def implicitas(h, d, a):
        inv = pd.DataFrame({"A": 1 / df[a], "D": 1 / df[d], "H": 1 / df[h]})
        return inv.div(inv.sum(axis=1), axis=0)
    p = implicitas("Odds_Home", "Odds_Draw", "Odds_Away")
    return pd.DataFrame({"Mkt_A": p.A, "Mkt_D": p.D, "Mkt_H": p.H})


# ---------------------------------------------------------------- métricas

def ece(y, proba, bins=10):
    conf = proba.max(axis=1)
    acierto = proba.argmax(axis=1) == y
    total, err = len(y), 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            err += m.sum() / total * abs(acierto[m].mean() - conf[m].mean())
    return err


def metricas(y, proba):
    onehot = np.eye(3)[y]
    return {
        "logloss": log_loss(y, proba, labels=CLASES),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "acc": float((proba.argmax(axis=1) == y).mean()),
        "ece": ece(y, proba),
        "pred_D": float((proba.argmax(axis=1) == 1).mean()),
    }


# ---------------------------------------------------------------- modelos

def xgb_actual():
    return XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42,
                         n_jobs=-1, eval_metric="mlogloss")


def xgb_regularizado():
    return XGBClassifier(n_estimators=400, max_depth=2, learning_rate=0.02, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0,
                         random_state=42, n_jobs=-1, eval_metric="mlogloss")


def logistica(c=0.05):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def main():
    df = cargar()
    X_base, _, _ = build_training_matrix(df)
    y = df["Result"].map(COD).to_numpy()
    E, W, M = elo(df), ewma(df), mercado(df)
    tiene_cuotas = M.notna().all(axis=1).to_numpy()

    conjuntos = {
        "actual (35)": X_base,
        "actual + Elo": pd.concat([X_base, E], axis=1),
        "Elo + EWMA (11)": pd.concat([E, W], axis=1),
        "actual + Elo + EWMA": pd.concat([X_base, E, W], axis=1),
        "Elo + EWMA + mercado": pd.concat([E, W, M], axis=1),
    }
    modelos = {"XGB actual": xgb_actual, "XGB regularizado": xgb_regularizado, "Logística": logistica}

    filas = []
    for test in TESTS:
        anteriores = TODAS[: TODAS.index(test)]
        for historia, temporadas in (("5 temp.", [s for s in anteriores if s != "2021"][-5:]),
                                     ("todas", [s for s in anteriores if s != "2021"])):
            tr = df.Season.isin(temporadas).to_numpy() & tiene_cuotas
            te = (df.Season == test).to_numpy() & tiene_cuotas
            for nombre_x, X in conjuntos.items():
                for nombre_m, fabrica in modelos.items():
                    if historia == "todas" and nombre_m != "Logística" and nombre_x not in ("actual (35)", "Elo + EWMA (11)"):
                        continue
                    if historia == "todas" and nombre_m == "XGB actual" and nombre_x != "actual (35)":
                        continue
                    modelo = fabrica()
                    modelo.fit(X[tr], y[tr])
                    proba = modelo.predict_proba(X[te])
                    filas.append({"test": test, "historia": historia, "features": nombre_x,
                                  "modelo": nombre_m, "n": int(te.sum()), **metricas(y[te], proba)})
            # Referencias
            yt = y[te]
            filas.append({"test": test, "historia": "-", "features": "MERCADO (Bet365)", "modelo": "cuotas",
                          "n": int(te.sum()), **metricas(yt, M[te][["Mkt_A", "Mkt_D", "Mkt_H"]].to_numpy())})
            frec = np.bincount(y[tr], minlength=3) / tr.sum()
            filas.append({"test": test, "historia": "-", "features": "FRECUENCIAS", "modelo": "constante",
                          "n": int(te.sum()), **metricas(yt, np.tile(frec, (len(yt), 1)))})

    res = pd.DataFrame(filas)
    res.to_csv(Path(__file__).parent / "resultados.csv", index=False)  # gitignored
    resumen = (res.groupby(["historia", "features", "modelo"])[["logloss", "brier", "acc", "ece", "pred_D"]]
               .mean().sort_values("logloss"))
    pd.set_option("display.width", 200)
    print(resumen.round(4).to_string())
    print()
    piv = res.pivot_table(index=["historia", "features", "modelo"], columns="test", values="logloss")
    print(piv.loc[resumen.index[:6]].round(4).to_string())


if __name__ == "__main__":
    main()
