import pytest
from lib.agent.scenario import SCENARIO_LIBRARY, apply_shock, compute_contagion_loss, run_scenarios
from lib.agent.config import AgentConfig
from lib.agent.types import GraphSnapshot, NodeFeatures, Edge, ScenarioSummary


@pytest.fixture
def stress_graph():
    """A 5-node graph with clear hierarchy for stress testing."""
    return GraphSnapshot(
        date="2025-01-01",
        nodes={
            "aave-v3": NodeFeatures(log_size=9.5, num_tokens=3, max_share=0.6, entropy=1.2, category="lending"),
            "lido": NodeFeatures(log_size=9.1, num_tokens=1, max_share=0.9, entropy=0.3, category="liquid-staking"),
            "uniswap-v3": NodeFeatures(log_size=7.8, num_tokens=5, max_share=0.3, entropy=2.1, category="dex"),
            "compound": NodeFeatures(log_size=8.0, num_tokens=2, max_share=0.5, entropy=1.0, category="lending"),
            "wbtc-bridge": NodeFeatures(log_size=8.5, num_tokens=1, max_share=0.8, entropy=0.5, category="bridge"),
        },
        edges=[
            Edge(source="aave-v3", target="lido", weight=9.0),
            Edge(source="lido", target="uniswap-v3", weight=7.0),
            Edge(source="uniswap-v3", target="aave-v3", weight=6.0),
            Edge(source="compound", target="aave-v3", weight=5.0),
            Edge(source="wbtc-bridge", target="compound", weight=8.0),
            Edge(source="wbtc-bridge", target="lido", weight=7.5),
        ],
    )


def test_scenario_library_has_5_scenarios():
    assert len(SCENARIO_LIBRARY) == 5
    for sid in ["S1", "S2", "S3", "S4", "S5"]:
        assert sid in SCENARIO_LIBRARY


def test_apply_shock_s1_removes_top_node(stress_graph):
    """S1: 100% TVL loss on top-1 node by weight."""
    shocked = apply_shock(stress_graph, SCENARIO_LIBRARY["S1"])
    # Top node by total edge weight involvement should have edges zeroed/removed
    assert len(shocked.edges) < len(stress_graph.edges) or any(
        e.weight < orig_e.weight
        for e, orig_e in zip(shocked.edges, stress_graph.edges)
        if e.source == orig_e.source and e.target == orig_e.target
    )


def test_apply_shock_s2_targets_bridges(stress_graph):
    """S2: 100% TVL loss on bridge nodes."""
    shocked = apply_shock(stress_graph, SCENARIO_LIBRARY["S2"])
    # Bridge node edges should be removed or zeroed
    bridge_edges = [e for e in shocked.edges if "bridge" in e.source or "bridge" in e.target]
    assert all(e.weight == 0.0 for e in bridge_edges) or len(bridge_edges) == 0


def test_compute_contagion_loss_returns_valid(stress_graph):
    loss = compute_contagion_loss(stress_graph, stress_graph, scenario_id="S1", scenario_name="test", horizon=4)
    assert loss.expected_loss >= 0.0
    assert loss.distressed_count >= 0
    assert loss.propagation_depth >= 0


def test_run_scenarios_returns_summary(stress_graph, sample_config):
    summary = run_scenarios(stress_graph, [stress_graph], sample_config, horizon=4)
    assert isinstance(summary, ScenarioSummary)
    assert len(summary.ranked_losses) == 5  # One per scenario
    assert summary.worst_scenario in ["S1", "S2", "S3", "S4", "S5"]


def test_run_scenarios_ranked_by_loss(stress_graph, sample_config):
    summary = run_scenarios(stress_graph, [stress_graph], sample_config, horizon=4)
    losses = [sl.expected_loss for sl in summary.ranked_losses]
    assert losses == sorted(losses, reverse=True)
