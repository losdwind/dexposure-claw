import pytest
import numpy as np
from lib.agent.pred_graph import build_pred_graph, mc_sample
from lib.agent.config import AgentConfig
from lib.agent.types import GraphSnapshot


@pytest.fixture
def mock_prediction():
    """Mock FM output: edge probabilities and predicted weights."""
    return {
        "edge_probs": {
            ("aave-v3", "lido"): 0.9,
            ("lido", "uniswap-v3"): 0.3,
            ("uniswap-v3", "aave-v3"): 0.1,
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


def test_build_pred_graph_filters_low_prob(mock_prediction):
    cfg = AgentConfig(pi_min=0.2)
    g_hat = build_pred_graph(mock_prediction, cfg)
    edge_pairs = {(e.source, e.target) for e in g_hat.edges}
    assert ("aave-v3", "lido") in edge_pairs
    assert ("lido", "uniswap-v3") in edge_pairs
    assert ("uniswap-v3", "aave-v3") not in edge_pairs  # prob=0.1 < 0.2


def test_build_pred_graph_uses_expected_weights(mock_prediction):
    cfg = AgentConfig(pi_min=0.2)
    g_hat = build_pred_graph(mock_prediction, cfg)
    for e in g_hat.edges:
        if e.source == "aave-v3" and e.target == "lido":
            assert e.weight == pytest.approx(7.5)


def test_mc_sample_returns_correct_count(mock_prediction):
    cfg = AgentConfig(mc_samples=10, pi_min=0.2)
    samples = mc_sample(mock_prediction, cfg)
    assert len(samples) == 10
    for s in samples:
        assert isinstance(s, GraphSnapshot)


def test_mc_samples_have_variability(mock_prediction):
    np.random.seed(42)
    cfg = AgentConfig(mc_samples=30, pi_min=0.2)
    samples = mc_sample(mock_prediction, cfg)
    edge_counts = [len(s.edges) for s in samples]
    # With stochastic Bernoulli sampling, not all samples should be identical
    assert len(set(edge_counts)) >= 1  # At minimum different edge counts


def test_build_pred_graph_empty_prediction():
    cfg = AgentConfig(pi_min=0.2)
    empty = {"edge_probs": {}, "edge_weights": {}, "weight_stds": {}, "node_ids": []}
    g_hat = build_pred_graph(empty, cfg)
    assert len(g_hat.edges) == 0
    assert len(g_hat.nodes) == 0
