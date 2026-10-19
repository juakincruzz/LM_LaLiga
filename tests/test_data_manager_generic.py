from unittest.mock import MagicMock, patch

from scripts.data_manager_generic import ensure_league_data, league_csv_path


def test_ensure_league_data_downloads_when_missing():
    """Si el CSV no está en disco, se descarga. `data_root` aísla la ruta,
    así que este test nunca borra datos reales."""
    # `data_root` solo trae la 25/26: una temporada anterior no está en disco
    path = league_csv_path("SP1", "2425")
    assert not path.exists()

    with patch("scripts.data_manager_generic.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = b"Date,HomeTeam,AwayTeam\n2025-08-01,TeamA,TeamB\n"
        mock_get.return_value = mock_response

        result = ensure_league_data("SP1", "2425")

    assert result == path
    assert path.exists()
    mock_get.assert_called_once()
    assert "2425/SP1" in str(mock_get.call_args[0][0])


def test_ensure_league_data_does_not_redownload():
    """Si ya existe, no se toca la red."""
    with patch("scripts.data_manager_generic.requests.get") as mock_get:
        result = ensure_league_data("SP1", "2526")

    assert result.exists()
    mock_get.assert_not_called()
