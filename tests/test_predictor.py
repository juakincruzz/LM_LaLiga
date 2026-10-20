from models.predictor import LeaguePredictor

def test_predictor_initializes_with_league_code():
    predictor = LeaguePredictor("SP1")
    assert predictor.league_code == "SP1"
    assert "SP1" in str(predictor.model_path)


def test_un_pkl_con_otras_features_pide_reentrenar(models_dir):
    """Un modelo de las 35 features antiguas no debe reventar en mitad de una
    predicción con un error de forma: debe pedir reentrenar."""
    import pickle

    import pytest

    from core.errors import ModelNotTrainedError

    with open(models_dir / "SP1_predictor.pkl", "wb") as f:
        pickle.dump({"model": object(), "feature_names": ["Home_Matches"]}, f)

    with pytest.raises(ModelNotTrainedError, match="reentrenar|Entrenar"):
        LeaguePredictor("SP1").load_model()
