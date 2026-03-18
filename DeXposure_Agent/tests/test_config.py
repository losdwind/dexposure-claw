import pytest
from dexposure_agent.config import AgentConfig


def test_default_config():
    cfg = AgentConfig()
    assert cfg.tau_data == 0.7
    assert cfg.tau_conf == 0.6
    assert cfg.z_threshold == 2.0
    assert cfg.rolling_window == 26
    assert cfg.pi_min == 0.2
    assert cfg.lambda_tail == 0.5
    assert cfg.mc_samples == 50
    assert cfg.top_k == 10
    assert cfg.horizons == [1, 4, 8, 12]


def test_custom_config():
    cfg = AgentConfig(tau_data=0.5, mc_samples=100)
    assert cfg.tau_data == 0.5
    assert cfg.mc_samples == 100


def test_invalid_tau_data():
    with pytest.raises(ValueError):
        AgentConfig(tau_data=1.5)


def test_invalid_tau_data_negative():
    with pytest.raises(ValueError):
        AgentConfig(tau_data=-0.1)
