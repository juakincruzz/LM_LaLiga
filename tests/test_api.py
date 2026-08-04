import re


async def test_dashboard_lista_las_ligas(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "La Liga" in response.text
    assert "Premier League" in response.text


async def test_league_detail(client):
    response = await client.get("/league/SP1")
    assert response.status_code == 200
    assert "La Liga" in response.text


async def test_create_and_get_job(client, mock_download):
    # El job de update_data se ejecuta de verdad como BackgroundTask, así que
    # `mock_download` intercepta la descarga para no depender de la red.
    create_resp = await client.post("/api/jobs/?job_type=update_data&league_code=SP1")

    assert create_resp.status_code == 200
    # El POST devuelve el fragmento HTML del job, no JSON
    assert "QUEUED" in create_resp.text

    match = re.search(r'id="job-(\d+)"', create_resp.text)
    assert match is not None
    job_id = int(match.group(1))

    get_resp = await client.get(f"/api/jobs/{job_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["job_type"] == "update_data"
