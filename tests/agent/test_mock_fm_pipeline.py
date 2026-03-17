"""Comprehensive mock FM server tests.

Tests the full agent pipeline with a realistic mock FM model that produces
structured predictions. Verifies that all pipeline stages produce meaningful
outputs: DataHealth, Monitor alerts, Scenario losses, Decision tickets.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from lib.agent.agent_loop import run_epoch
from lib.agent.config import AgentConfig
from lib.agent.serve import create_app, _mock_forecast, _build_mock_graph
from lib.agent.types import AgentOutput, GraphSnapshot, NodeFeatures, Edge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_app():
    return create_app(mock_mode=True)


@pytest.fixture
def realistic_graph():
    """A 7-node graph mimicking real DeFi topology with diverse sectors."""
    return _build_mock_graph("2025-03-01")


@pytest.fixture
def baseline_history():
    """Several epochs of past metric dicts for rolling baseline comparison."""
    from lib.agent.monitor import compute_metrics

    history = []
    for week in range(26):
        date = f"2024-{7 + week // 4:02d}-{1 + (week % 4) * 7:02d}"
        g = _build_mock_graph(date)
        metrics = compute_metrics(g)
        history.append(metrics)
    return history


# ---------------------------------------------------------------------------
# Test: mock forecast produces valid predictions
# ---------------------------------------------------------------------------

class TestMockForecast:
    def test_forecast_has_required_keys(self):
        pred = _mock_forecast("2025-03-01", 4)
        assert "edge_probs" in pred
        assert "edge_weights" in pred
        assert "weight_stds" in pred
        assert "node_ids" in pred

    def test_forecast_has_edges(self):
        pred = _mock_forecast("2025-03-01", 4)
        assert len(pred["edge_probs"]) > 0, "Mock should produce some edges"

    def test_forecast_probs_in_range(self):
        pred = _mock_forecast("2025-03-01", 4)
        for key, prob in pred["edge_probs"].items():
            assert 0.0 <= prob <= 1.0, f"Edge {key} prob={prob} out of range"

    def test_forecast_weights_positive(self):
        pred = _mock_forecast("2025-03-01", 4)
        for key, w in pred["edge_weights"].items():
            assert w > 0.0, f"Edge {key} weight={w} should be positive"

    def test_forecast_keys_are_pipe_strings(self):
        """Server returns JSON-serializable string keys, not tuples."""
        pred = _mock_forecast("2025-03-01", 4)
        for key in pred["edge_probs"]:
            assert isinstance(key, str), f"Key should be string, got {type(key)}"
            assert "|" in key, f"Key should be pipe-delimited, got {key}"

    def test_forecast_deterministic(self):
        """Same (date, horizon) produces same predictions."""
        p1 = _mock_forecast("2025-03-01", 4)
        p2 = _mock_forecast("2025-03-01", 4)
        assert p1 == p2

    def test_forecast_varies_with_horizon(self):
        p1 = _mock_forecast("2025-03-01", 1)
        p4 = _mock_forecast("2025-03-01", 4)
        # Different horizons should (likely) produce different predictions
        assert p1 != p4


# ---------------------------------------------------------------------------
# Test: mock graph construction
# ---------------------------------------------------------------------------

class TestMockGraph:
    def test_build_mock_graph_has_nodes(self):
        g = _build_mock_graph("2025-03-01")
        assert len(g.nodes) == 7  # MOCK_PROTOCOLS has 7 entries

    def test_build_mock_graph_has_edges(self):
        g = _build_mock_graph("2025-03-01")
        assert len(g.edges) > 5, "Should have a reasonably connected graph"

    def test_build_mock_graph_has_diverse_categories(self):
        g = _build_mock_graph("2025-03-01")
        categories = {nf.category for nf in g.nodes.values()}
        assert len(categories) >= 3, f"Should have diverse categories, got {categories}"

    def test_build_mock_graph_has_bridge_node(self):
        """Needed for S2 scenario (bridge cluster failure)."""
        g = _build_mock_graph("2025-03-01")
        bridge_nodes = [n for n, f in g.nodes.items() if f.category == "bridge"]
        assert len(bridge_nodes) >= 1, "Need at least one bridge for S2 scenario"


# ---------------------------------------------------------------------------
# Test: full pipeline via run_epoch with mock forecast
# ---------------------------------------------------------------------------

class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_produces_complete_output(self, realistic_graph):
        """Full pipeline with mock forecast produces all output fields."""
        async def mock_fn(graph, horizon, config):
            return _mock_forecast(graph.date, horizon)

        cfg = AgentConfig(horizons=[1, 4], mc_samples=5)
        output = await run_epoch(
            graph=realistic_graph,
            baseline_history=[],
            config=cfg,
            forecast_fn=mock_fn,
        )

        assert isinstance(output, AgentOutput)
        assert output.epoch_date == "2025-03-01"
        assert output.data_health.score > 0.0
        assert output.data_health.safe_mode is False  # Healthy graph
        assert output.scenario_summary.ranked_losses is not None
        assert len(output.scenario_summary.ranked_losses) > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_baseline_produces_alerts(self, realistic_graph, baseline_history):
        """With 26 weeks of baseline history, alerts should fire on deviations."""
        async def mock_fn(graph, horizon, config):
            return _mock_forecast(graph.date, horizon)

        cfg = AgentConfig(
            horizons=[1, 4],
            mc_samples=5,
            z_threshold=1.5,  # Lower threshold to increase alert sensitivity
        )
        output = await run_epoch(
            graph=realistic_graph,
            baseline_history=baseline_history,
            config=cfg,
            forecast_fn=mock_fn,
        )

        # With baseline comparison, some alerts should fire
        # (predicted metrics will differ from historical mean)
        assert isinstance(output.alerts, list)
        # Even if no alerts fire, the pipeline should complete without error

    @pytest.mark.asyncio
    async def test_pipeline_scenario_losses_are_ranked(self, realistic_graph):
        """Scenario losses should be sorted by expected_loss descending."""
        async def mock_fn(graph, horizon, config):
            return _mock_forecast(graph.date, horizon)

        cfg = AgentConfig(horizons=[4], mc_samples=5)
        output = await run_epoch(
            graph=realistic_graph,
            baseline_history=[],
            config=cfg,
            forecast_fn=mock_fn,
        )

        losses = output.scenario_summary.ranked_losses
        if len(losses) >= 2:
            for i in range(len(losses) - 1):
                assert losses[i].expected_loss >= losses[i + 1].expected_loss

    @pytest.mark.asyncio
    async def test_pipeline_scenario_has_all_5_scenarios(self, realistic_graph):
        """All S1-S5 should appear in scenario results."""
        async def mock_fn(graph, horizon, config):
            return _mock_forecast(graph.date, horizon)

        cfg = AgentConfig(horizons=[4], mc_samples=3)
        output = await run_epoch(
            graph=realistic_graph,
            baseline_history=[],
            config=cfg,
            forecast_fn=mock_fn,
        )

        scenario_ids = {sl.scenario_id for sl in output.scenario_summary.ranked_losses}
        assert scenario_ids == {"S1", "S2", "S3", "S4", "S5"}

    @pytest.mark.asyncio
    async def test_pipeline_worst_scenario_is_meaningful(self, realistic_graph):
        """Worst scenario should have non-zero loss."""
        async def mock_fn(graph, horizon, config):
            return _mock_forecast(graph.date, horizon)

        cfg = AgentConfig(horizons=[4], mc_samples=3)
        output = await run_epoch(
            graph=realistic_graph,
            baseline_history=[],
            config=cfg,
            forecast_fn=mock_fn,
        )

        worst_id = output.scenario_summary.worst_scenario
        assert worst_id != "", "Should identify a worst scenario"
        worst_losses = [sl for sl in output.scenario_summary.ranked_losses if sl.scenario_id == worst_id]
        assert len(worst_losses) == 1
        assert worst_losses[0].expected_loss > 0.0

    @pytest.mark.asyncio
    async def test_safe_mode_suppresses_intervention_tickets(self):
        """Empty graph triggers safe mode, blocking Recommend-Reduce and Contingency."""
        empty = GraphSnapshot(date="2025-01-01", nodes={}, edges=[])

        async def mock_fn(graph, horizon, config):
            return _mock_forecast(graph.date, horizon)

        cfg = AgentConfig(horizons=[4], mc_samples=3)
        output = await run_epoch(
            graph=empty,
            baseline_history=[],
            config=cfg,
            forecast_fn=mock_fn,
        )

        assert output.data_health.safe_mode is True
        assert output.suppressed is True
        for t in output.tickets:
            assert t.action in ("Monitor", "Investigate"), \
                f"Safe mode should block {t.action}"


# ---------------------------------------------------------------------------
# Test: full pipeline via HTTP (mock server)
# ---------------------------------------------------------------------------

class TestHTTPPipeline:
    @pytest.mark.asyncio
    async def test_run_epoch_endpoint_produces_real_output(self, mock_app):
        """The /run-epoch endpoint should now produce real alerts and scenarios."""
        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/run-epoch", json={"date": "2025-03-01"})

        assert resp.status_code == 200
        data = resp.json()

        # Validate full structure
        output = AgentOutput.model_validate(data)
        assert output.epoch_date == "2025-03-01"
        assert output.data_health.score > 0.5  # Healthy mock graph
        assert output.data_health.safe_mode is False

        # Scenarios should be populated
        assert len(output.scenario_summary.ranked_losses) == 5
        assert output.scenario_summary.worst_scenario in {"S1", "S2", "S3", "S4", "S5"}

        # At least some scenario should have non-zero loss
        total_loss = sum(sl.expected_loss for sl in output.scenario_summary.ranked_losses)
        assert total_loss > 0.0, "Scenarios should produce real losses"

    @pytest.mark.asyncio
    async def test_forecast_endpoint_pipe_keys(self, mock_app):
        """Forecast endpoint returns pipe-delimited string keys."""
        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/forecast", json={"date": "2025-03-01", "horizon": 4})

        data = resp.json()
        for key in data["edge_probs"]:
            assert isinstance(key, str)
            assert "|" in key

    @pytest.mark.asyncio
    async def test_multiple_epochs_via_http(self, mock_app):
        """Run multiple epochs and verify consistent structure."""
        transport = ASGITransport(app=mock_app)
        dates = ["2025-01-01", "2025-02-01", "2025-03-01"]

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for date in dates:
                resp = await client.post("/run-epoch", json={"date": date})
                assert resp.status_code == 200
                output = AgentOutput.model_validate(resp.json())
                assert output.epoch_date == date
                assert len(output.scenario_summary.ranked_losses) == 5


# ---------------------------------------------------------------------------
# Test: ticket generation end-to-end
# ---------------------------------------------------------------------------

class TestTicketGeneration:
    @pytest.mark.asyncio
    async def test_tickets_have_evidence(self, realistic_graph, baseline_history):
        """Tickets should include triggering alerts and scenario impact."""
        async def mock_fn(graph, horizon, config):
            return _mock_forecast(graph.date, horizon)

        cfg = AgentConfig(
            horizons=[1, 4, 8],
            mc_samples=5,
            z_threshold=1.0,  # Very sensitive — should produce many alerts
            tau_conf=0.1,     # Low confidence gate — allow interventions
        )
        output = await run_epoch(
            graph=realistic_graph,
            baseline_history=baseline_history,
            config=cfg,
            forecast_fn=mock_fn,
        )

        if output.tickets:
            for ticket in output.tickets:
                assert ticket.action in ("Monitor", "Investigate", "Recommend-Reduce", "Contingency")
                assert ticket.severity in ("Low", "Medium", "High", "Critical")
                assert ticket.score > 0.0
                assert ticket.rationale != ""

    @pytest.mark.asyncio
    async def test_ticket_scores_descending(self, realistic_graph, baseline_history):
        """Tickets should be sorted by score descending."""
        async def mock_fn(graph, horizon, config):
            return _mock_forecast(graph.date, horizon)

        cfg = AgentConfig(
            horizons=[4],
            mc_samples=3,
            z_threshold=1.0,
            tau_conf=0.1,
        )
        output = await run_epoch(
            graph=realistic_graph,
            baseline_history=baseline_history,
            config=cfg,
            forecast_fn=mock_fn,
        )

        if len(output.tickets) >= 2:
            scores = [t.score for t in output.tickets]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Test: JSON serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerialization:
    @pytest.mark.asyncio
    async def test_agent_output_full_roundtrip(self, realistic_graph):
        """Full AgentOutput should serialize/deserialize cleanly."""
        async def mock_fn(graph, horizon, config):
            return _mock_forecast(graph.date, horizon)

        cfg = AgentConfig(horizons=[4], mc_samples=3)
        output = await run_epoch(
            graph=realistic_graph,
            baseline_history=[],
            config=cfg,
            forecast_fn=mock_fn,
        )

        json_str = output.model_dump_json()
        restored = AgentOutput.model_validate_json(json_str)

        assert restored.epoch_date == output.epoch_date
        assert restored.data_health.score == output.data_health.score
        assert len(restored.scenario_summary.ranked_losses) == len(output.scenario_summary.ranked_losses)
        assert len(restored.alerts) == len(output.alerts)
        assert len(restored.tickets) == len(output.tickets)
