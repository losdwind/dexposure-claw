"""Tests for lib/agent/agent_loop.py — Algorithm 1 end-to-end orchestration."""
import pytest
from unittest.mock import AsyncMock, patch
from dexposure_agent.agent_loop import run_epoch
from dexposure_agent.config import AgentConfig
from dexposure_agent.types import GraphSnapshot, NodeFeatures, Edge, AgentOutput


@pytest.fixture
def mock_prediction():
    """Mock API response for forecast endpoint."""
    return {
        "edge_probs": {
            ("aave-v3", "lido"): 0.9,
            ("lido", "uniswap-v3"): 0.5,
            ("uniswap-v3", "aave-v3"): 0.3,
        },
        "edge_weights": {
            ("aave-v3", "lido"): 7.5,
            ("lido", "uniswap-v3"): 6.2,
            ("uniswap-v3", "aave-v3"): 5.8,
        },
        "weight_stds": {
            ("aave-v3", "lido"): 0.5,
            ("lido", "uniswap-v3"): 0.8,
            ("uniswap-v3", "aave-v3"): 1.2,
        },
        "node_ids": ["aave-v3", "lido", "uniswap-v3"],
    }


@pytest.mark.asyncio
async def test_run_epoch_returns_agent_output(sample_graph, mock_prediction):
    """Full pipeline with mocked API returns valid AgentOutput."""
    cfg = AgentConfig(horizons=[1, 4], mc_samples=5)
    with patch("dexposure_agent.agent_loop.call_forecast_api", new_callable=AsyncMock, return_value=mock_prediction):
        output = await run_epoch(
            graph=sample_graph,
            baseline_history=[],
            config=cfg,
        )
    assert isinstance(output, AgentOutput)
    assert output.epoch_date == "2025-01-01"
    assert output.data_health.score > 0


@pytest.mark.asyncio
async def test_run_epoch_safe_mode_propagates(mock_prediction):
    """Empty graph triggers safe mode, which suppresses interventions."""
    empty = GraphSnapshot(date="2025-01-01", nodes={}, edges=[])
    cfg = AgentConfig(horizons=[1], mc_samples=3)
    with patch("dexposure_agent.agent_loop.call_forecast_api", new_callable=AsyncMock, return_value=mock_prediction):
        output = await run_epoch(graph=empty, baseline_history=[], config=cfg)
    assert output.data_health.safe_mode is True
    assert output.suppressed is True
    for t in output.tickets:
        assert t.action in ("Monitor", "Investigate")


@pytest.mark.asyncio
async def test_run_epoch_multiple_horizons(sample_graph, mock_prediction):
    """Multiple horizons produce scenario losses for each."""
    cfg = AgentConfig(horizons=[1, 4, 8], mc_samples=3)
    with patch("dexposure_agent.agent_loop.call_forecast_api", new_callable=AsyncMock, return_value=mock_prediction):
        output = await run_epoch(graph=sample_graph, baseline_history=[], config=cfg)
    assert output.scenario_summary.ranked_losses is not None
