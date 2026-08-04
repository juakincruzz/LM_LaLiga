"""Arreglos mínimos de transfer_service.py, sin invertir en la integración completa."""
import httpx
import pytest

from app.services.transfer_service import check_transfermarkt_available


async def test_falla_con_mensaje_claro_si_el_servicio_no_responde():
    """Antes el primer fallo llegaba 41 peticiones más tarde como
    'All connection attempts failed', sin pista de qué arrancar."""
    async with httpx.AsyncClient() as client:
        with pytest.raises(ConnectionError, match="transfermarkt-api"):
            await check_transfermarkt_available(client)


async def test_no_reporta_ascendidos_falsos(monkeypatch, temp_db):
    """Un club TM sin emparejar no es necesariamente un ascendido: en las ligas
    sin alias configurados, el emparejamiento por subcadena falla para nombres
    como 'Man United' / 'Manchester United', así que antes se reportaban como
    'ascendido detectado' clubes que llevaban años en la liga."""
    import aiosqlite

    from app.services import transfer_service as ts

    async def falso_ok(client):
        return None

    async def falsos_clubs(client, league_id, season):
        return [{"id": "1", "name": "Manchester United"}]

    async def sin_jugadores(client, club_id, season):
        return []

    async def equipos_falsos(db, league_code, season="2526"):
        return ["Man United"]

    monkeypatch.setattr(ts, "check_transfermarkt_available", falso_ok)
    monkeypatch.setattr(ts, "fetch_competition_clubs", falsos_clubs)
    monkeypatch.setattr(ts, "fetch_club_players", sin_jugadores)
    monkeypatch.setattr(ts, "ensure_teams_populated", equipos_falsos)

    async with aiosqlite.connect(temp_db) as db:
        db.row_factory = aiosqlite.Row
        resumen = await ts.analyze_league_transfers(db, "E0")

    assert "promoted_detected" not in resumen
    assert resumen["teams_unmatched"] == ["Man United"]
