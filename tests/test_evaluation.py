"""core.evaluation: split por temporadas y baselines honestos."""
import numpy as np
import pandas as pd
import pytest

from core.evaluation import (
    baseline_always_home,
    baseline_majority_class,
    bookmaker_baseline,
    Fold,
    evaluate_walk_forward,
    multiclass_brier_score,
    walk_forward_splits,
)


def _temporadas(*codigos, por_temporada=3):
    return pd.DataFrame({"Season": [c for c in codigos for _ in range(por_temporada)]})


def test_walk_forward_cada_test_entrena_solo_con_las_anteriores():
    df = _temporadas("2122", "2223", "2324", "2425")
    cortes = walk_forward_splits(df, ["2122", "2223", "2324", "2425"], n_test=2)

    assert [t for t, _, _ in cortes] == ["2324", "2425"]
    for test, train, prueba in cortes:
        assert set(df.loc[prueba, "Season"]) == {test}
        assert set(df.loc[train, "Season"]) == {s for s in ["2122", "2223", "2324"] if s < test}


def test_walk_forward_salta_las_excluidas_y_las_vacias():
    df = _temporadas("1920", "2021", "2122")
    cortes = walk_forward_splits(df, ["1920", "2021", "2122", "2223"], excluded={"2021"})

    assert [t for t, _, _ in cortes] == ["2122"]  # 2223 no tiene partidos
    _, train, _ = cortes[0]
    assert set(df.loc[train, "Season"]) == {"1920"}


def test_walk_forward_exige_al_menos_dos_temporadas():
    with pytest.raises(ValueError, match="al menos 2"):
        walk_forward_splits(_temporadas("2526"), ["2526"])


def test_informe_walk_forward_con_referencias():
    """Probabilidades perfectas: acierto 1 y log-loss casi 0; el mercado y las
    frecuencias se calculan aparte, con lo que se sabía en cada corte."""
    y_test = pd.Series([1, -1, 0, 1])
    df_test = pd.DataFrame({
        "Date": pd.to_datetime(["2025-09-01"] * 4),
        "Result": ["H", "A", "D", "H"],
        "Odds_Home": [1.5, 4.0, 3.0, 2.0], "Odds_Draw": [4.0, 3.5, 2.8, 3.3], "Odds_Away": [6.0, 1.8, 3.0, 3.5],
    })
    perfecta = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) * 0.98 + 0.02 / 3
    fold = Fold("2526", pd.Series([1, 1, 0, -1]), y_test, perfecta, df_test)

    r = evaluate_walk_forward([fold], n_train=4, label=lambda s: s)

    assert r.accuracy == pytest.approx(1.0)
    assert r.logloss < r.logloss_bookmaker < r.logloss_frequencies + 1
    assert r.baseline_home == pytest.approx(0.5)
    assert r.baseline_majority == pytest.approx(0.5)  # la mayoría del train es «local»
    assert (r.n_train, r.n_test, r.test_period) == (4, 4, "2526")


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
