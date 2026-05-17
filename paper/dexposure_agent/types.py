"""Pydantic models for every data structure in the agent pipeline.

Maps to Algorithm 1 in DeXposure-Agent paper:
  GraphSnapshot -> DataHealthResult -> Alert/MonitorResult -> ScenarioSummary -> Ticket/DecisionResult -> AgentOutput
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class NodeFeatures(BaseModel):
    log_size: float
    num_tokens: int
    max_share: float
    entropy: float
    category: str


class Edge(BaseModel):
    source: str
    target: str
    weight: float


class GraphSnapshot(BaseModel):
    date: str
    nodes: dict[str, NodeFeatures]
    edges: list[Edge]


class DataHealthResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="DH_t in [0,1]")
    safe_mode: bool = Field(description="True if DH_t < tau_data")
    checks: dict[str, float] = Field(default_factory=dict, description="Individual check scores")


class Alert(BaseModel):
    horizon: int
    metric_id: str
    metric_name: str
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    confidence: float = Field(ge=0.0, le=1.0)
    attribution: dict[str, float] = Field(default_factory=dict, description="Top-K contributing nodes")


class AlertEvidence(BaseModel):
    """Evidence bundle for an alert (Eq. 5 in paper)."""
    alert: Alert
    horizon: int
    attribution: dict[str, float] = Field(default_factory=dict)


class MonitorResult(BaseModel):
    alerts: list[Alert] = Field(default_factory=list)
    metrics: dict[str, dict[int, float]] = Field(
        default_factory=dict, description="metric_id -> {horizon: value}"
    )


class ScenarioLoss(BaseModel):
    scenario_id: str
    scenario_name: str
    horizon: int
    expected_loss: float
    cvar_loss: float
    distressed_count: int
    propagation_depth: int
    top_targets: list[str] = Field(default_factory=list)


class ScenarioSummary(BaseModel):
    ranked_losses: list[ScenarioLoss] = Field(default_factory=list)
    worst_scenario: str = ""
    worst_horizon: int = 0


class Ticket(BaseModel):
    action: str = Field(description="One of: Monitor, Investigate, Recommend-Reduce, Contingency")
    severity: str = Field(description="Low, Medium, High, Critical")
    score: float
    targets: list[str] = Field(default_factory=list)
    triggering_alerts: list[str] = Field(default_factory=list)
    scenario_impact: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class DecisionResult(BaseModel):
    tickets: list[Ticket] = Field(default_factory=list)
    suppressed: bool = Field(False, description="True if safe-mode suppressed interventions")


class AgentOutput(BaseModel):
    epoch_date: str
    data_health: DataHealthResult
    alerts: list[Alert] = Field(default_factory=list)
    scenario_summary: ScenarioSummary
    tickets: list[Ticket] = Field(default_factory=list)
    suppressed: bool = Field(False, description="True if safe-mode suppressed interventions")
