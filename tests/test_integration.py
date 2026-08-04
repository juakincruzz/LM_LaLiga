"""Prueba end-to-end del pipeline web completo.

Marcada como `slow`: descarga de red y entrena un modelo real. Se ejecuta con
`pytest -m slow`.
"""
import asyncio
import re

import pytest


@pytest.mark.slow
async def test_pipeline_completo_sp1(client, temp_db):
    """Actualizar datos → entrenar → proyectar, y comprobar lo que ve el usuario."""
    # 1. Actualizar datos (descarga real)
    job_id = await _lanzar(client, "update_data", "SP1")
    job = await _esperar(client, job_id, max_wait=60)
    assert "partidos" in (job["message"] or "")

    # 2. Entrenar. La accuracy debe ser honesta, no la del conjunto de entrenamiento.
    job_id = await _lanzar(client, "train", "SP1")
    job = await _esperar(client, job_id, max_wait=300)
    resultado = _resultado(job)
    assert 0.30 <= resultado["accuracy"] <= 0.65, f"accuracy sospechosa: {resultado['accuracy']}"

    # 3. La temporada 25/26 está completa: simular debe fallar con un mensaje claro
    #    en lugar de repetir la temporada real y devolver un campeón al 100 %.
    job_id = await _lanzar(client, "simulate", "SP1")
    job = await _esperar(client, job_id, max_wait=120, permitir_fallo=True)
    assert job["status"] == "failed"
    assert "ya ha terminado" in job["log"]

    # 4. Proyectar la temporada siguiente sí funciona
    job_id = await _lanzar(client, "project_next_season", "SP1")
    await _esperar(client, job_id, max_wait=120)

    resp = await client.get("/league/SP1")
    assert resp.status_code == 200
    assert "Temporada finalizada" in resp.text
    assert "Precisión fuera de muestra" in resp.text
    assert "Proyección" in resp.text


async def _lanzar(client, job_type: str, league_code: str) -> int:
    resp = await client.post(f"/api/jobs/?job_type={job_type}&league_code={league_code}")
    assert resp.status_code == 200
    assert "QUEUED" in resp.text
    match = re.search(r'id="job-(\d+)"', resp.text)
    assert match is not None
    return int(match.group(1))


async def _esperar(client, job_id: int, max_wait: int = 60, permitir_fallo: bool = False) -> dict:
    for _ in range(max_wait):
        resp = await client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        job = resp.json()
        if job["status"] in ("done", "failed"):
            if job["status"] == "failed" and not permitir_fallo:
                raise AssertionError(f"El trabajo {job_id} falló: {job.get('log')}")
            return job
        await asyncio.sleep(1)
    raise TimeoutError(f"El trabajo {job_id} no terminó en {max_wait}s")


def _resultado(job: dict) -> dict:
    import json

    return json.loads(job["result_json"] or "{}")
