"""Service to fetch transfer data from self-hosted Transfermarkt API."""
import httpx
from app.config import get_league_config

TRANSFERMARKT_BASE = "http://localhost:8001"


async def fetch_club_transfers(club_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRANSFERMARKT_BASE}/clubs/{club_id}/transfers",
            timeout=30
        )
        response.raise_for_status()
        return response.json()


async def fetch_league_transfers(league_code: str):
    config = get_league_config(league_code)
    tm_league_id = config.get("transfermarkt_league_id")
    if not tm_league_id:
        return []

    # TODO: Map league_id to clubs list; this requires a clubs endpoint or a mapping table
    # Stub implementation returning empty list until mapping is built
    return []
