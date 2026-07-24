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


import re

@pytest.mark.asyncio
async def test_create_and_get_job():
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        create_resp = await client.post("/api/jobs/?job_type=update_data&league_code=SP1")
        assert create_resp.status_code == 200
        # POST now returns HTML partial, not JSON
        assert "QUEUED" in create_resp.text

        # Extract job_id from the HTML partial (id="job-{id}")
        match = re.search(r'id="job-(\d+)"', create_resp.text)
        assert match is not None
        job_id = int(match.group(1))

        get_resp = await client.get(f"/api/jobs/{job_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["job_type"] == "update_data"


@pytest.mark.asyncio
async def test_league_detail():
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/league/SP1")
    assert response.status_code == 200
    assert "La Liga" in response.text
