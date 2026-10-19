from models.predictor import LeaguePredictor

def test_predictor_initializes_with_league_code():
    predictor = LeaguePredictor("SP1")
    assert predictor.league_code == "SP1"
    assert "SP1" in str(predictor.model_path)
