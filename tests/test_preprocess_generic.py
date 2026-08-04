from scripts.preprocess_generic import preprocess_league


def test_preprocess_returns_xy():
    X, y, df = preprocess_league("SP1")
    assert len(X) == len(y) == len(df)
    assert len(X.columns) > 0
    # El target vive en {-1, 0, 1}; el remapeo a {0,1,2} es cosa del predictor
    assert set(y.unique()) <= {-1, 0, 1}
