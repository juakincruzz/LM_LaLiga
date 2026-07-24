import pytest
from scripts.preprocess_generic import preprocess_league

def test_preprocess_returns_xy():
    # Requires data/raw/SP1/2526.csv to exist (seeded in earlier tasks or existing)
    X, y = preprocess_league("SP1")
    assert len(X) == len(y)
    assert len(X.columns) > 0
