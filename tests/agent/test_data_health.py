import pytest
from lib.agent.data_health import compute_data_health
from lib.agent.config import AgentConfig


def test_healthy_graph(sample_graph, sample_config):
    result = compute_data_health(sample_graph, sample_config)
    assert result.score > sample_config.tau_data
    assert result.safe_mode is False


def test_empty_graph_triggers_safe_mode(sample_config):
    from lib.agent.types import GraphSnapshot
    empty = GraphSnapshot(date="2025-01-01", nodes={}, edges=[])
    result = compute_data_health(empty, sample_config)
    assert result.score < sample_config.tau_data
    assert result.safe_mode is True


def test_missing_features_lowers_score(sample_graph, sample_config):
    sample_graph.nodes["aave-v3"].log_size = 0.0
    sample_graph.nodes["aave-v3"].num_tokens = 0
    result = compute_data_health(sample_graph, sample_config)
    assert result.score < 1.0
    assert "missingness" in result.checks


def test_single_node_graph(sample_config):
    from lib.agent.types import GraphSnapshot, NodeFeatures
    single = GraphSnapshot(
        date="2025-01-01",
        nodes={"aave-v3": NodeFeatures(log_size=8.5, num_tokens=3, max_share=0.6, entropy=1.2, category="lending")},
        edges=[],
    )
    result = compute_data_health(single, sample_config)
    assert 0.0 <= result.score <= 1.0
    assert "topology" in result.checks
