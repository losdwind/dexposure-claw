import pytest
from dexposure_agent.types import (
    GraphSnapshot, NodeFeatures, Edge,
    Alert, AlertEvidence, MonitorResult,
    ScenarioLoss, ScenarioSummary,
    Ticket, DecisionResult, AgentOutput,
    DataHealthResult,
)


def test_graph_snapshot_construction(sample_graph):
    assert len(sample_graph.nodes) > 0
    assert len(sample_graph.edges) > 0
    assert sample_graph.date is not None


def test_alert_construction():
    alert = Alert(
        horizon=4, metric_id="M1", metric_name="Systemic Importance Score",
        value=0.85, baseline_mean=0.5, baseline_std=0.1,
        z_score=3.5, confidence=0.8,
        attribution={"aave-v3": 0.35, "lido": 0.25},
    )
    assert alert.z_score == 3.5
    assert alert.confidence == 0.8


def test_ticket_construction():
    ticket = Ticket(
        action="Recommend-Reduce", severity="High",
        score=0.75, targets=["aave-v3"],
        triggering_alerts=["M1_h4", "M2_h4"],
        scenario_impact={"S1": 0.45},
        rationale="SIS spike on aave-v3 at h=4",
    )
    assert ticket.severity == "High"


def test_agent_output_json_roundtrip():
    output = AgentOutput(
        epoch_date="2025-03-01",
        data_health=DataHealthResult(score=0.85, safe_mode=False, checks={}),
        alerts=[],
        scenario_summary=ScenarioSummary(ranked_losses=[], worst_scenario="S1", worst_horizon=4),
        tickets=[],
    )
    j = output.model_dump_json()
    restored = AgentOutput.model_validate_json(j)
    assert restored.epoch_date == "2025-03-01"
