import pytest
from pathlib import Path
from scripts.data_manager_generic import ensure_league_data, DATA_DIR
from unittest.mock import patch, MagicMock


def test_ensure_league_data_downloads_when_missing():
    with patch("scripts.data_manager_generic.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = b"Date,HomeTeam,AwayTeam\n2025-08-01,TeamA,TeamB\n"
        mock_get.return_value = mock_response

        path = ensure_league_data("SP1", "2526")
        assert path.exists()
        mock_get.assert_called_once()
        assert "SP1" in str(mock_get.call_args[0][0])

    # Cleanup
    if path.exists():
        path.unlink()
