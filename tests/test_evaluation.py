"""core.evaluation: split por temporadas y baselines honestos."""
import numpy as np
import pandas as pd
import pytest

from core.evaluation import (
    baseline_always_home,
    baseline_majority_class,
    bookmaker_baseline,
    multiclass_brier_score,
    season_train_test_split,
)


def test_split_usa_la_ultima_temporada_como_test():
    df = pd.DataFrame({
        "Season": ["2223"] * 3 + ["2324"] * 3 + ["2425"] * 3,
        "Date": pd.to_datetime(["2022-09-01"] * 3 + ["2023-09-01"] * 3 + ["2024-09-01"] * 3),
    })
    X = pd.DataFrame({"f": range(9)})
    y = pd.Series(range(9))

    X_train, X_test, y_train, y_test, df_train, df_test = season_train_test_split(
        df, X, y, ["2223", "2324", "2425"]
    )

    assert len(X_train) == 6 and len(X_test) == 3
    assert set(df_train["Season"]) == {"2223", "2324"}
    assert set(df_test["Season"]) == {"2425"}


def test_split_exige_al_menos_dos_temporadas():
    df = pd.DataFrame({"Season": ["2526"]})
    with pytest.raises(ValueError, match="al menos 2"):
        season_train_test_split(df, df, df["Season"], ["2526"])


def test_baseline_majority_usa_el_train_no_el_test():
    """Antes el baseline de clase mayoritaria se calculaba sobre el propio
    test (un oráculo: mide el test contra sí mismo). Aquí debe usar el train."""
    y_train = pd.Series([1, 1, 1, 1, 0])  # mayoría: local (1)
    y_test = pd.Series([-1, -1, -1, 1])  # aquí la mayoría sería visitante, pero no cuenta

    resultado = baseline_majority_class(y_train, y_test)
    assert resultado == pytest.approx(0.25)  # solo 1 de 4 es "1" (local)


def test_baseline_siempre_local():
    y_test = pd.Series([1, 1, 0, -1])
    assert baseline_always_home(y_test) == pytest.approx(0.5)


def test_bookmaker_baseline_con_cuotas_reales():
    df_test = pd.DataFrame({
        "Odds_Home": [1.5, 3.0, 2.0],
        "Odds_Draw": [4.0, 3.2, 3.3],
        "Odds_Away": [6.0, 2.3, 3.5],
        "Result": ["H", "A", "H"],
    })
    # Favoritos: fila0 H (acierta), fila1 A (acierta), fila2 H (acierta)
    assert bookmaker_baseline(df_test) == pytest.approx(1.0)


def test_bookmaker_baseline_none_sin_columnas_de_cuotas():
    df_test = pd.DataFrame({"Result": ["H", "A"]})
    assert bookmaker_baseline(df_test) is None


def test_brier_score_perfecto_es_cero():
    y_true = pd.Series([2, 0, 1])  # codificado {0,1,2}
    proba = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    assert multiclass_brier_score(y_true, proba, classes=[0, 1, 2]) == pytest.approx(0.0)


def test_brier_score_uniforme_es_peor_que_perfecto():
    y_true = pd.Series([2, 0, 1])
    perfecto = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    uniforme = np.full((3, 3), 1 / 3)
    peor = multiclass_brier_score(y_true, uniforme, classes=[0, 1, 2])
    mejor = multiclass_brier_score(y_true, perfecto, classes=[0, 1, 2])
    assert peor > mejor
