import pytest
import httpx
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "La Liga" in response.text


@pytest.mark.asyncio
async def test_dashboard():
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "La Liga" in response.text


@pytest.mark.asyncio
async def test_create_and_get_job():
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        create_resp = await client.post("/api/jobs/?job_type=update_data&league_code=SP1")
        assert create_resp.status_code == 200
        job = create_resp.json()
        assert job["status"] == "queued"

        get_resp = await client.get(f"/api/jobs/{job['job_id']}")
        assert get_resp.status_code == 200
        assert get_resp.json()["job_type"] == "update_data"


@pytest.mark.asyncio
async def test_league_detail():
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/league/SP1")
    assert response.status_code == 200
    assert "La Liga" in response.text
