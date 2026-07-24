import pytest
import httpx
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "LM_LaLiga API"
