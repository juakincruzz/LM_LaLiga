from models.predictor import LeaguePredictor

def test_predictor_initializes_with_league_code():
    predictor = LeaguePredictor("E0")
    assert predictor.league_code == "E0"
    assert "E0" in str(predictor.model_path)
