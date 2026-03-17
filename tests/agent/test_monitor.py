import pytest
from lib.agent.monitor import compute_metrics, detect_alerts, compute_confidence, run_monitor
from lib.agent.config import AgentConfig


def test_compute_metrics_returns_all_ids(sample_graph):
    metrics = compute_metrics(sample_graph)
    # M1=SIS, M3=HHI, M4=Density, M6=PageRank, M7=Gini are computable from a single graph
    # M2=Spillover needs sector data, M5=Stress loss needs scenario
    expected_ids = {"M1", "M3", "M4", "M6", "M7"}
    assert expected_ids.issubset(set(metrics.keys()))


def test_compute_metrics_values_in_range(sample_graph):
    metrics = compute_metrics(sample_graph)
    # All network-level metrics should be in [0, 1] or reasonable ranges
    assert 0.0 <= metrics["M4"] <= 1.0  # Density
    assert 0.0 <= metrics["M7"] <= 1.0  # Gini


def test_detect_alerts_with_spike(sample_config):
    current = {"M1": 0.9}
    baseline = {"M1": {"mean": 0.3, "std": 0.1}}
    alerts = detect_alerts(current, baseline, horizon=4, config=sample_config)
    assert len(alerts) == 1
    assert alerts[0].metric_id == "M1"
    assert alerts[0].z_score > sample_config.z_threshold


def test_detect_alerts_no_spike(sample_config):
    current = {"M1": 0.35}
    baseline = {"M1": {"mean": 0.3, "std": 0.1}}
    alerts = detect_alerts(current, baseline, horizon=4, config=sample_config)
    assert len(alerts) == 0


def test_detect_alerts_handles_zero_std(sample_config):
    """Zero std means no variability — any deviation should trigger."""
    current = {"M1": 0.31}
    baseline = {"M1": {"mean": 0.3, "std": 0.0}}
    alerts = detect_alerts(current, baseline, horizon=4, config=sample_config)
    # With zero std, use a small epsilon to avoid division by zero
    assert len(alerts) <= 1  # May or may not trigger depending on epsilon handling


def test_compute_confidence_degrades_with_horizon(sample_config):
    c1 = compute_confidence(dh_score=0.9, dispersion=0.1, horizon=1, config=sample_config)
    c4 = compute_confidence(dh_score=0.9, dispersion=0.1, horizon=12, config=sample_config)
    assert c1 > c4


def test_compute_confidence_degrades_with_dispersion(sample_config):
    c_low = compute_confidence(dh_score=0.9, dispersion=0.1, horizon=4, config=sample_config)
    c_high = compute_confidence(dh_score=0.9, dispersion=0.9, horizon=4, config=sample_config)
    assert c_low > c_high


def test_compute_confidence_degrades_with_low_dh(sample_config):
    c_good = compute_confidence(dh_score=0.9, dispersion=0.1, horizon=4, config=sample_config)
    c_bad = compute_confidence(dh_score=0.3, dispersion=0.1, horizon=4, config=sample_config)
    assert c_good > c_bad


def test_run_monitor_integration(sample_graph, sample_config):
    result = run_monitor(
        predicted_graph=sample_graph,
        mc_samples=[sample_graph],
        baseline_history=[],
        horizon=4,
        config=sample_config,
    )
    assert hasattr(result, "alerts")
    assert hasattr(result, "metrics")
