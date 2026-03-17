# tests/agent/test_integration.py
"""End-to-end integration test: full agent pipeline with mock server."""
import pytest
from httpx import AsyncClient, ASGITransport
from lib.agent.serve import create_app
from lib.agent.types import AgentOutput


@pytest.fixture
def mock_app():
    return create_app(mock_mode=True)


@pytest.mark.asyncio
async def test_full_pipeline_via_api(mock_app):
    """Run /run-epoch endpoint and verify complete AgentOutput structure."""
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/run-epoch", json={"date": "2025-03-01"})

    assert resp.status_code == 200
    data = resp.json()

    # Verify all top-level fields exist
    assert "epoch_date" in data
    assert data["epoch_date"] == "2025-03-01"
    assert "data_health" in data
    assert "alerts" in data
    assert "scenario_summary" in data
    assert "tickets" in data
    assert "suppressed" in data

    # Verify data_health structure
    dh = data["data_health"]
    assert 0.0 <= dh["score"] <= 1.0
    assert isinstance(dh["safe_mode"], bool)
    assert isinstance(dh["checks"], dict)

    # Verify scenario_summary structure
    ss = data["scenario_summary"]
    assert "ranked_losses" in ss
    assert "worst_scenario" in ss

    # Verify it deserializes back to AgentOutput
    output = AgentOutput.model_validate(data)
    assert output.epoch_date == "2025-03-01"


@pytest.mark.asyncio
async def test_pipeline_determinism(mock_app):
    """Same date should produce same results (mock uses seeded RNG)."""
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post("/run-epoch", json={"date": "2025-01-15"})
        r2 = await client.post("/run-epoch", json={"date": "2025-01-15"})

    d1, d2 = r1.json(), r2.json()
    assert d1["data_health"]["score"] == d2["data_health"]["score"]
    assert len(d1["alerts"]) == len(d2["alerts"])


@pytest.mark.asyncio
async def test_different_dates_produce_valid_results(mock_app):
    """Different dates should each produce a structurally valid AgentOutput."""
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post("/run-epoch", json={"date": "2025-01-01"})
        r2 = await client.post("/run-epoch", json={"date": "2025-06-01"})

    assert r1.status_code == 200
    assert r2.status_code == 200

    d1, d2 = r1.json(), r2.json()

    # Each response deserialises to a valid AgentOutput
    out1 = AgentOutput.model_validate(d1)
    out2 = AgentOutput.model_validate(d2)

    # epoch_date echoes back the requested date
    assert out1.epoch_date == "2025-01-01"
    assert out2.epoch_date == "2025-06-01"

    # The two runs are for different dates
    assert out1.epoch_date != out2.epoch_date

    # Data-health scores are both valid
    assert 0.0 <= out1.data_health.score <= 1.0
    assert 0.0 <= out2.data_health.score <= 1.0
