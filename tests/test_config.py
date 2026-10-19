import pytest

from app.config import LEAGUE_CODE, get_league_config, list_leagues, season_label
from core.errors import UnknownLeagueError


def test_get_laliga():
    cfg = get_league_config()
    assert cfg["name"] == "La Liga"
    assert cfg["teams"] == 20


def test_solo_queda_la_liga():
    assert LEAGUE_CODE == "SP1"
    assert [l["code"] for l in list_leagues()] == ["SP1"]


def test_otras_ligas_ya_no_existen():
    with pytest.raises(UnknownLeagueError):
        get_league_config("E0")


def test_season_label():
    assert season_label("2526") == "25/26"
