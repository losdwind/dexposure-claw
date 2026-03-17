import pytest
from lib.agent.types import GraphSnapshot, NodeFeatures, Edge


@pytest.fixture
def sample_graph():
    """Minimal 3-node graph for testing."""
    return GraphSnapshot(
        date="2025-01-01",
        nodes={
            "aave-v3": NodeFeatures(log_size=8.5, num_tokens=3, max_share=0.6, entropy=1.2, category="lending"),
            "lido": NodeFeatures(log_size=9.1, num_tokens=1, max_share=0.9, entropy=0.3, category="liquid-staking"),
            "uniswap-v3": NodeFeatures(log_size=7.8, num_tokens=5, max_share=0.3, entropy=2.1, category="dex"),
        },
        edges=[
            Edge(source="aave-v3", target="lido", weight=7.5),
            Edge(source="lido", target="uniswap-v3", weight=6.2),
            Edge(source="uniswap-v3", target="aave-v3", weight=5.8),
        ],
    )


@pytest.fixture
def sample_config():
    from lib.agent.config import AgentConfig
    return AgentConfig()
