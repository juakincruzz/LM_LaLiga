"""End-to-end integration test for the full web pipeline."""
import pytest
import asyncio
import httpx
from app.main import app


@pytest.mark.asyncio
async def test_full_pipeline_sp1():
    """
    Full end-to-end test:
    1. Update data for SP1
    2. Train model for SP1
    3. Simulate season for SP1
    4. Verify standings appear on league page
    """
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:

        # Step 1: Update data
        print("\n[1/3] Updating data for SP1...")
        resp = await client.post("/api/jobs/?job_type=update_data&league_code=SP1")
        assert resp.status_code == 200
        assert "QUEUED" in resp.text
        job_id = _extract_job_id(resp.text)

        # Wait for completion
        await _wait_for_job(client, job_id, max_wait=30)
        print("  ✓ Data updated")

        # Step 2: Train model
        print("\n[2/3] Training model for SP1...")
        resp = await client.post("/api/jobs/?job_type=train&league_code=SP1")
        assert resp.status_code == 200
        assert "QUEUED" in resp.text
        job_id = _extract_job_id(resp.text)

        # Wait for completion (training takes ~1-2 minutes)
        await _wait_for_job(client, job_id, max_wait=180)
        print("  ✓ Model trained")

        # Step 3: Simulate season
        print("\n[3/3] Simulating season for SP1...")
        resp = await client.post("/api/jobs/?job_type=simulate&league_code=SP1")
        assert resp.status_code == 200
        assert "QUEUED" in resp.text
        job_id = _extract_job_id(resp.text)

        # Wait for completion
        await _wait_for_job(client, job_id, max_wait=60)
        print("  ✓ Season simulated")

        # Verify standings appear on league page
        resp = await client.get("/league/SP1")
        assert resp.status_code == 200
        assert "Clasificación Proyectada" in resp.text or "Simular temporada" in resp.text
        print("\n✅ Full pipeline test passed!")


def _extract_job_id(html: str) -> int:
    import re
    match = re.search(r'id="job-(\d+)"', html)
    assert match is not None
    return int(match.group(1))


async def _wait_for_job(client, job_id: int, max_wait: int = 60):
    for _ in range(max_wait):
        resp = await client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        job = resp.json()
        if job["status"] in ("done", "failed"):
            if job["status"] == "failed":
                raise AssertionError(f"Job {job_id} failed: {job.get('log', 'Unknown error')}")
            return
        await asyncio.sleep(1)
    raise TimeoutError(f"Job {job_id} did not complete within {max_wait}s")
