"""Caché de alineaciones de Understat.

Las fixtures son los 6 últimos partidos del Betis en 25/26 (ids 29474–29529):
Betis-R. Madrid, Betis-Oviedo, R. Sociedad-Betis, Betis-Elche, Barcelona-Betis
y Betis-Levante, con sus alineaciones reales.
"""
import pytest

import core.equipos
from core.equipos import EquipoDesconocidoError
from scripts.understat import download_understat, load_lineups


def test_descarga_y_carga_alineaciones(mock_understat):
    assert download_understat("SP1", "2526") == {"partidos": 6, "nuevos": 6, "faltan": 0}
    df = load_lineups("SP1", "2526")
    assert df["match_id"].nunique() == 6
    assert "Betis" in set(df["team"]) and "Real Madrid" in set(df["opponent"])
    assert df.groupby("match_id")["starter"].sum().eq(22).all()
    assert not df.loc[df["position"] == "Sub", "starter"].any()
    assert str(df["date"].dtype).startswith("datetime64")


def test_partidos_jugados_no_se_vuelven_a_pedir(mock_understat):
    download_understat("SP1", "2526")
    n = mock_understat.call_count
    assert download_understat("SP1", "2526")["nuevos"] == 0
    assert mock_understat.call_count == n + 1  # solo la liga


def test_cache_parcial_se_completa(mock_understat, data_root):
    download_understat("SP1", "2526")
    next((data_root / "SP1" / "understat" / "partidos").glob("*.json")).unlink()
    assert download_understat("SP1", "2526")["nuevos"] == 1


def test_carga_ignora_partido_sin_json(mock_understat, data_root):
    download_understat("SP1", "2526")
    next((data_root / "SP1" / "understat" / "partidos").glob("*.json")).unlink()
    assert load_lineups("SP1", "2526")["match_id"].nunique() == 5


def test_un_partido_que_falla_no_aborta_el_resto(mock_understat, monkeypatch):
    original = mock_understat.side_effect

    def con_un_fallo(url, **kw):
        if url.endswith("/29504"):
            raise ConnectionError("corte")
        return original(url, **kw)

    mock_understat.side_effect = con_un_fallo
    assert download_understat("SP1", "2526") == {"partidos": 6, "nuevos": 5, "faltan": 1}


def test_sin_cache_es_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_lineups("SP1", "2526")


def test_sin_red_lanza_error_de_requests():
    import requests

    with pytest.raises(requests.RequestException):
        download_understat("SP1", "2526")


def test_equipo_desconocido(mock_understat, monkeypatch):
    monkeypatch.delitem(core.equipos.UNDERSTAT, "Real Betis")
    with pytest.raises(EquipoDesconocidoError):
        download_understat("SP1", "2526")
