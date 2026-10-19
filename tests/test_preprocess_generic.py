from scripts.preprocess_generic import preprocess_league


def test_preprocess_returns_xy(mock_download):
    # Sin mock_download, las temporadas que `data_root` no trae (todas menos la
    # 25/26) se descargaban de verdad: un test rápido que dependía de la red.
    X, y, df = preprocess_league("SP1")
    assert len(X) == len(y) == len(df)
    assert len(X.columns) > 0
    # El target vive en {-1, 0, 1}; el remapeo a {0,1,2} es cosa del predictor
    assert set(y.unique()) <= {-1, 0, 1}
