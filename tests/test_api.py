import re


async def test_portada_es_la_liga(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "La Liga" in response.text
    assert "Premier League" not in response.text


async def test_ruta_antigua_redirige_a_portada(client):
    response = await client.get("/league/SP1")
    assert response.status_code == 301
    assert response.headers["location"] == "/"


async def test_create_and_get_job(client, mock_download):
    # El job de update_data se ejecuta de verdad como BackgroundTask, así que
    # `mock_download` intercepta la descarga para no depender de la red.
    # league_code es opcional: por defecto, La Liga.
    create_resp = await client.post("/api/jobs/?job_type=update_data")

    assert create_resp.status_code == 200
    # El POST devuelve el fragmento HTML del job, no JSON
    assert 'data-status="queued"' in create_resp.text
    assert "Actualizar datos" in create_resp.text

    match = re.search(r'id="job-(\d+)"', create_resp.text)
    assert match is not None
    job_id = int(match.group(1))

    get_resp = await client.get(f"/api/jobs/{job_id}")
    assert get_resp.status_code == 200
    job = get_resp.json()
    assert job["job_type"] == "update_data"
    assert job["league_code"] == "SP1"
