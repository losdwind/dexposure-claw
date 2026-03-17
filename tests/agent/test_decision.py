import pytest
from lib.agent.decision import generate_tickets
from lib.agent.types import Alert, ScenarioSummary, DataHealthResult, ScenarioLoss, DecisionResult
from lib.agent.config import AgentConfig


def test_safe_mode_suppresses_interventions():
    """In safe mode, only Monitor/Investigate actions should be generated."""
    dh = DataHealthResult(score=0.3, safe_mode=True, checks={})
    alerts = [Alert(horizon=4, metric_id="M1", metric_name="SIS", value=0.9,
                    baseline_mean=0.3, baseline_std=0.1, z_score=6.0, confidence=0.9)]
    scenario = ScenarioSummary(ranked_losses=[], worst_scenario="S1", worst_horizon=4)
    cfg = AgentConfig()
    result = generate_tickets(alerts, scenario, dh, cfg)
    for t in result.tickets:
        assert t.action in ("Monitor", "Investigate")
    assert result.suppressed is True


def test_high_confidence_enables_intervention():
    dh = DataHealthResult(score=0.9, safe_mode=False, checks={})
    alerts = [
        Alert(horizon=4, metric_id="M1", metric_name="SIS", value=0.9,
              baseline_mean=0.3, baseline_std=0.1, z_score=6.0, confidence=0.9),
        Alert(horizon=4, metric_id="M3", metric_name="HHI", value=0.8,
              baseline_mean=0.4, baseline_std=0.1, z_score=4.0, confidence=0.8),
        Alert(horizon=4, metric_id="M4", metric_name="Density", value=0.1,
              baseline_mean=0.5, baseline_std=0.1, z_score=4.0, confidence=0.7),
    ]
    scenario = ScenarioSummary(
        ranked_losses=[ScenarioLoss(scenario_id="S1", scenario_name="Single failure",
                                     horizon=4, expected_loss=0.8, cvar_loss=0.9,
                                     distressed_count=5, propagation_depth=3, top_targets=["aave-v3"])],
        worst_scenario="S1", worst_horizon=4,
    )
    cfg = AgentConfig()
    result = generate_tickets(alerts, scenario, dh, cfg)
    actions = {t.action for t in result.tickets}
    assert "Contingency" in actions or "Recommend-Reduce" in actions


def test_low_confidence_blocks_intervention():
    dh = DataHealthResult(score=0.9, safe_mode=False, checks={})
    alerts = [Alert(horizon=4, metric_id="M1", metric_name="SIS", value=0.9,
                    baseline_mean=0.3, baseline_std=0.1, z_score=6.0, confidence=0.2)]
    scenario = ScenarioSummary(ranked_losses=[], worst_scenario="S1", worst_horizon=4)
    cfg = AgentConfig()
    result = generate_tickets(alerts, scenario, dh, cfg)
    for t in result.tickets:
        assert t.action in ("Monitor", "Investigate")


def test_no_alerts_no_tickets():
    dh = DataHealthResult(score=0.9, safe_mode=False, checks={})
    scenario = ScenarioSummary(ranked_losses=[], worst_scenario="", worst_horizon=0)
    cfg = AgentConfig()
    result = generate_tickets([], scenario, dh, cfg)
    assert len(result.tickets) == 0


def test_tickets_sorted_by_score():
    dh = DataHealthResult(score=0.9, safe_mode=False, checks={})
    alerts = [
        Alert(horizon=4, metric_id="M1", metric_name="SIS", value=0.9,
              baseline_mean=0.3, baseline_std=0.1, z_score=6.0, confidence=0.9),
        Alert(horizon=4, metric_id="M3", metric_name="HHI", value=0.6,
              baseline_mean=0.4, baseline_std=0.1, z_score=2.0, confidence=0.7),
    ]
    scenario = ScenarioSummary(
        ranked_losses=[ScenarioLoss(scenario_id="S1", scenario_name="Single failure",
                                     horizon=4, expected_loss=0.5, cvar_loss=0.6,
                                     distressed_count=3, propagation_depth=2, top_targets=["aave-v3"])],
        worst_scenario="S1", worst_horizon=4,
    )
    cfg = AgentConfig()
    result = generate_tickets(alerts, scenario, dh, cfg)
    if len(result.tickets) >= 2:
        scores = [t.score for t in result.tickets]
        assert scores == sorted(scores, reverse=True)
