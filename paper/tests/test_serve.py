import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from dexposure_agent.serve import create_app
    return create_app(mock_mode=True)


@pytest.mark.asyncio
async def test_health_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models" in data


@pytest.mark.asyncio
async def test_forecast_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/forecast", json={
            "date": "2025-01-01",
            "horizon": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "edge_probs" in data
        assert "edge_weights" in data
        assert "weight_stds" in data
        assert "node_ids" in data


@pytest.mark.asyncio
async def test_models_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "loaded" in data


@pytest.mark.asyncio
async def test_run_epoch_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/run-epoch", json={"date": "2025-01-01"})
        assert resp.status_code == 200
        data = resp.json()
        assert "epoch_date" in data
        assert "data_health" in data
        assert "alerts" in data
        assert "tickets" in data


@pytest.mark.asyncio
async def test_batch_forecast_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/batch-forecast", json={
            "date": "2025-01-01",
            "horizons": [1, 4],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
