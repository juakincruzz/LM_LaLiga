from app.config import get_league_config, list_leagues

def test_get_laliga():
    cfg = get_league_config("SP1")
    assert cfg["name"] == "La Liga"
    assert cfg["teams"] == 20

def test_list_leagues():
    leagues = list_leagues()
    assert len(leagues) == 5
    codes = {l["code"] for l in leagues}
    assert codes == {"SP1", "E0", "D1", "I1", "F1"}
