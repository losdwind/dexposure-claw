# DeXposure-Agent: Core Library + REST API + Claude Code Plugin

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the DeXposure-Agent as a shared core Python library, expose it via a FastAPI server on GPU, and release it as a Claude Code plugin with multiple skills for interactive DeFi risk monitoring.

**Architecture:** Three-layer design — (1) `lib/agent/` shared core implements Algorithm 1 from the paper (DataHealth, Monitor, Scenario, Decision); (2) `lib/agent/serve.py` wraps DeXposure-FM inference in a FastAPI REST API on GPU server; (3) `plugin/dexposure-agent/` is a Claude Code plugin with skills that call the API and perform analytical reasoning. Experiment scripts in `experiments/` reuse the same core library for batch evaluation.

**Tech Stack:** Python 3.12, PyTorch 2.2, FastAPI, uvicorn, httpx, pydantic, networkx. Claude Code plugin uses markdown skills + bash/Python scripts.

**Existing Code to Reuse:**
- `dexposure_fm/macroprudential_tools.py` — SIS, spillover, contagion simulation (Phi functionals)
- `dexposure_fm/network_statistics.py` — Gini, HHI, density, entropy, PageRank
- `run_task2_model_based.py` — forward risk, predictive contagion, early warning
- `run_macroprudential_tools.py` — CLI for macroprudential tools
- `checkpoints/` — pretrained FM weights
- `data/historical-network_week_*.json` — graph snapshots

---

## File Structure

```
graph-dexposure/
├── lib/agent/                          # SHARED CORE (new)
│   ├── __init__.py                     # Public API surface
│   ├── types.py                        # Pydantic models: AlertEvidence, Ticket, etc.
│   ├── data_health.py                  # DataHealth gate (DH_t, SAFE_MODE)
│   ├── pred_graph.py                   # BuildPredGraph + MC sampler
│   ├── monitor.py                      # Phi functionals, alert rules, confidence
│   ├── scenario.py                     # Stress-test engine (S1-S5)
│   ├── decision.py                     # Playbook constraints, ticket scoring
│   ├── agent_loop.py                   # End-to-end Algorithm 1
│   ├── config.py                       # AgentConfig pydantic model (all thresholds)
│   └── serve.py                        # FastAPI server wrapping FM inference
│
├── tests/agent/                        # Tests for core library (new)
│   ├── conftest.py                     # Shared fixtures (mock graphs, configs)
│   ├── test_types.py
│   ├── test_data_health.py
│   ├── test_pred_graph.py
│   ├── test_monitor.py
│   ├── test_scenario.py
│   ├── test_decision.py
│   ├── test_agent_loop.py
│   └── test_serve.py
│
├── plugin/dexposure-agent/             # CLAUDE CODE PLUGIN (new)
│   ├── .claude-plugin/
│   │   └── plugin.json
│   ├── commands/
│   │   └── dexposure.md                # /dexposure — main entry command
│   ├── agents/
│   │   ├── risk-analyst.md             # Autonomous monitoring agent
│   │   └── stress-tester.md            # Scenario analysis subagent
│   ├── skills/
│   │   ├── data-health/
│   │   │   └── SKILL.md
│   │   ├── forecast/
│   │   │   └── SKILL.md
│   │   ├── monitor/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── metrics-reference.md
│   │   ├── scenario/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── scenario-library.md
│   │   ├── decision/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── playbook-reference.md
│   │   ├── run-epoch/
│   │   │   └── SKILL.md
│   │   └── evaluate/
│   │   │   └── SKILL.md
│   │   └── domain-knowledge/
│   │       ├── SKILL.md
│   │       └── references/
│   │           ├── defi-protocols.md
│   │           └── risk-metrics.md
│   ├── hooks/
│   │   ├── hooks.json
│   │   └── scripts/
│   │       └── check-server-health.sh
│   └── scripts/
│       ├── call-api.py                 # Thin httpx wrapper for skills
│       └── format-output.py            # Format agent outputs for display
│
├── experiments/                        # PAPER BENCHMARKS (new, replaces ad-hoc scripts)
│   ├── run_all.py                      # Master experiment runner
│   ├── b1_risk_forecasting.py
│   ├── b2_early_warning.py
│   ├── b3_uncertainty_calibration.py
│   ├── b4_stress_test.py
│   ├── b5_decision_quality.py
│   ├── b6_robustness.py
│   ├── ablations.py                    # A1-A8
│   └── competitors/
│       ├── roland_agent.py             # C1: ROLAND backbone in agent pipeline
│       ├── persistence_agent.py        # C2: Naive baseline in agent pipeline
│       ├── llm_agent.py                # C3: LLM backbone in agent pipeline
│       └── baselines.py                # C4-C10: Non-agent baselines
│
└── (existing files unchanged)
```

---

## Phase 1: Core Agent Types + Config (Foundation)

### Task 1: Agent Configuration Model

**Files:**
- Create: `lib/agent/__init__.py`
- Create: `lib/agent/config.py`
- Test: `tests/agent/test_config.py` (placeholder)

This task establishes the configuration object that every other module imports. All thresholds from EXPERIMENT_PLAN.txt Section 2.5 live here.

- [ ] **Step 1: Create test file with config validation tests**

```python
# tests/agent/test_config.py
import pytest
from lib.agent.config import AgentConfig


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/figurich/CodeProjects/graph-dexposure && python -m pytest tests/agent/test_config.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement config.py**

```python
# lib/agent/config.py
"""Agent configuration with all tunable hyperparameters.

Maps to EXPERIMENT_PLAN.txt Section 2.5.
"""
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """All hyperparameters for the DeXposure-Agent pipeline."""

    # Data-health gate
    tau_data: float = Field(0.7, ge=0.0, le=1.0, description="Data-health safe-mode threshold")

    # Monitor
    z_threshold: float = Field(2.0, gt=0.0, description="Alert z-score threshold")
    rolling_window: int = Field(26, gt=0, description="Rolling baseline window (weeks)")

    # PredGraph builder
    pi_min: float = Field(0.2, ge=0.0, le=1.0, description="Edge existence probability threshold")

    # MC sampling
    mc_samples: int = Field(50, gt=0, description="Monte Carlo samples for uncertainty")

    # Scenario engine
    lambda_tail: float = Field(0.5, ge=0.0, le=1.0, description="CVaR tail weight")

    # Decision
    tau_conf: float = Field(0.6, ge=0.0, le=1.0, description="Confidence gate for interventions")
    top_k: int = Field(10, gt=0, description="Top-K attribution nodes/edges")

    # Horizons
    horizons: list[int] = Field(default=[1, 4, 8, 12], description="Forecast horizons in weeks")

    # Server
    api_base_url: str = Field("http://localhost:8000", description="GPU server API base URL")
```

- [ ] **Step 4: Create `lib/agent/__init__.py`**

```python
# lib/agent/__init__.py
"""DeXposure-Agent: Agentic DeFi risk monitoring system."""
from lib.agent.config import AgentConfig

__all__ = ["AgentConfig"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/figurich/CodeProjects/graph-dexposure && python -m pytest tests/agent/test_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add lib/agent/__init__.py lib/agent/config.py tests/agent/test_config.py
git commit -m "feat(agent): add AgentConfig with all tunable hyperparameters"
```

---

### Task 2: Core Type Definitions

**Files:**
- Create: `lib/agent/types.py`
- Create: `tests/agent/conftest.py`
- Test: `tests/agent/test_types.py`

Pydantic models for every data structure that flows through Algorithm 1. These types are the contract between all modules and the plugin.

- [ ] **Step 1: Write tests for type construction and serialization**

```python
# tests/agent/test_types.py
import pytest
import numpy as np
from lib.agent.types import (
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
```

- [ ] **Step 2: Create conftest with shared fixtures**

```python
# tests/agent/conftest.py
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/figurich/CodeProjects/graph-dexposure && python -m pytest tests/agent/test_types.py -v`
Expected: FAIL (types module not found)

- [ ] **Step 4: Implement types.py**

```python
# lib/agent/types.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/figurich/CodeProjects/graph-dexposure && python -m pytest tests/agent/test_types.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Update __init__.py and commit**

Add type exports to `lib/agent/__init__.py`, then:

```bash
git add lib/agent/types.py tests/agent/conftest.py tests/agent/test_types.py lib/agent/__init__.py
git commit -m "feat(agent): add pydantic type definitions for full agent pipeline"
```

---

## Phase 2: Core Agent Modules (Analytical Logic)

### Task 3: DataHealth Gate

**Files:**
- Create: `lib/agent/data_health.py`
- Test: `tests/agent/test_data_health.py`

Implements Section 2.4 of the paper. Aggregates deterministic checks: freshness, missingness, discontinuities, topology sanity. Outputs DH_t in [0,1] and SAFE_MODE flag.

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/test_data_health.py
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
    # Set some features to 0 (missing)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/agent/test_data_health.py -v`
Expected: FAIL

- [ ] **Step 3: Implement data_health.py**

Implement `compute_data_health(graph: GraphSnapshot, config: AgentConfig) -> DataHealthResult` with four checks:
1. **freshness** (1.0 if date is parseable, 0.0 otherwise — in production, check staleness)
2. **missingness** (fraction of non-zero features across all nodes)
3. **topology** (1.0 if edge_count >= node_count, scaled down otherwise)
4. **discontinuity** (1.0 by default for single snapshot; in production, compare to previous)

Aggregate: `DH_t = mean(checks.values())`, `SAFE_MODE = DH_t < tau_data`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/agent/test_data_health.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/agent/data_health.py tests/agent/test_data_health.py
git commit -m "feat(agent): implement DataHealth gate with 4 quality checks"
```

---

### Task 4: PredGraph Builder + MC Sampler

**Files:**
- Create: `lib/agent/pred_graph.py`
- Test: `tests/agent/test_pred_graph.py`

Builds G_hat from prediction distributions and draws MC samples. This module processes outputs FROM the GPU server (it does NOT call the model itself).

- [ ] **Step 1: Write failing tests**

Test `build_pred_graph()` with a mock prediction distribution (dict of edge probabilities + predicted weights). Test `mc_sample()` returns M samples with expected variance.

```python
# tests/agent/test_pred_graph.py
import pytest
import numpy as np
from lib.agent.pred_graph import build_pred_graph, mc_sample
from lib.agent.config import AgentConfig


@pytest.fixture
def mock_prediction():
    """Mock FM output: edge probabilities and predicted weights."""
    return {
        "edge_probs": {
            ("aave-v3", "lido"): 0.9,
            ("lido", "uniswap-v3"): 0.3,       # below pi_min=0.2 but above threshold
            ("uniswap-v3", "aave-v3"): 0.1,     # below pi_min=0.2
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


def test_mc_sample_returns_correct_count(mock_prediction):
    cfg = AgentConfig(mc_samples=10, pi_min=0.2)
    samples = mc_sample(mock_prediction, cfg)
    assert len(samples) == 10
    for s in samples:
        assert hasattr(s, "edges")


def test_mc_samples_vary(mock_prediction):
    cfg = AgentConfig(mc_samples=20, pi_min=0.2)
    samples = mc_sample(mock_prediction, cfg)
    edge_counts = [len(s.edges) for s in samples]
    # With stochastic sampling, not all samples should be identical
    assert len(set(edge_counts)) > 1 or len(samples) < 5
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement pred_graph.py**

`build_pred_graph(prediction, config)`: threshold edges by pi_min, use expected weights.
`mc_sample(prediction, config)`: for each sample, Bernoulli-sample edges by prob, Gaussian-sample weights.

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add lib/agent/pred_graph.py tests/agent/test_pred_graph.py
git commit -m "feat(agent): implement PredGraph builder and MC sampler"
```

---

### Task 5: Monitor Module

**Files:**
- Create: `lib/agent/monitor.py`
- Test: `tests/agent/test_monitor.py`

Implements Section 2.5: compute Phi functionals, compare to rolling baselines, trigger alerts with evidence and confidence. This is the analytical heart of the agent.

**Key design decision:** Reuse functions from `dexposure_fm/macroprudential_tools.py` and `dexposure_fm/network_statistics.py` for Phi functionals. The monitor module wraps them with alert logic.

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/test_monitor.py
import pytest
from lib.agent.monitor import compute_metrics, detect_alerts, compute_confidence, run_monitor
from lib.agent.config import AgentConfig
from lib.agent.types import GraphSnapshot


def test_compute_metrics_returns_all_ids(sample_graph):
    metrics = compute_metrics(sample_graph)
    expected_ids = {"M1", "M3", "M4", "M6", "M7"}  # M2, M5 need sector/scenario data
    assert expected_ids.issubset(set(metrics.keys()))


def test_detect_alerts_with_spike(sample_config):
    """A metric value far above baseline should trigger an alert."""
    current = {"M1": 0.9}
    baseline = {"M1": {"mean": 0.3, "std": 0.1}}
    alerts = detect_alerts(current, baseline, horizon=4, config=sample_config)
    assert len(alerts) == 1
    assert alerts[0].metric_id == "M1"
    assert alerts[0].z_score > sample_config.z_threshold


def test_detect_alerts_no_spike(sample_config):
    """A metric value within baseline should not trigger."""
    current = {"M1": 0.35}
    baseline = {"M1": {"mean": 0.3, "std": 0.1}}
    alerts = detect_alerts(current, baseline, horizon=4, config=sample_config)
    assert len(alerts) == 0


def test_compute_confidence_degrades_with_horizon(sample_config):
    c1 = compute_confidence(dh_score=0.9, dispersion=0.1, horizon=1, config=sample_config)
    c4 = compute_confidence(dh_score=0.9, dispersion=0.1, horizon=12, config=sample_config)
    assert c1 > c4  # Longer horizon = lower confidence


def test_run_monitor_integration(sample_graph, sample_config):
    """End-to-end monitor with empty history (no baseline, no alerts expected on first call)."""
    result = run_monitor(
        predicted_graph=sample_graph,
        mc_samples=[sample_graph],
        baseline_history=[],
        horizon=4,
        config=sample_config,
    )
    assert hasattr(result, "alerts")
    assert hasattr(result, "metrics")
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement monitor.py**

Key functions:
- `compute_metrics(graph) -> dict[str, float]`: calls existing network_statistics + macroprudential_tools, wraps in metric IDs M1-M7
- `detect_alerts(current, baseline, horizon, config) -> list[Alert]`: z-score test per metric
- `compute_confidence(dh_score, dispersion, horizon, config) -> float`: Eq. 6 from paper
- `run_monitor(predicted_graph, mc_samples, baseline_history, horizon, config) -> MonitorResult`

**Import from existing code** (use actual function names from codebase):
```python
from dexposure_fm.network_statistics import gini_coefficient, herfindahl_hirschman_index, network_density
from dexposure_fm.macroprudential_tools import compute_sis_components, compute_sector_spillover_index, simulate_contagion
```

**Note:** Consider wrapping `compute_network_risk_metrics()` from `macroprudential_tools.py` (lines 378-482) and `compute_rolling_statistics()` from `network_statistics.py` — these already compute comprehensive metric sets and rolling baselines. Adapt their dict-format snapshot input via a thin conversion layer from the Pydantic `GraphSnapshot` type.

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add lib/agent/monitor.py tests/agent/test_monitor.py
git commit -m "feat(agent): implement Monitor with Phi functionals and z-score alerts"
```

---

### Task 6: Scenario Engine

**Files:**
- Create: `lib/agent/scenario.py`
- Test: `tests/agent/test_scenario.py`

Implements Section 2.6: apply stress shocks S1-S5 to predicted graphs, compute contagion losses, rank scenarios.

- [ ] **Step 1: Write failing tests**

Test `apply_shock()` for each scenario type S1-S5. Test `run_scenarios()` returns ranked losses. Test that CVaR computation is correct.

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement scenario.py**

Key functions:
- `SCENARIO_LIBRARY`: dict defining S1-S5 from EXPERIMENT_PLAN Section 2.3
- `apply_shock(graph, scenario) -> GraphSnapshot`: apply TVL drops per scenario spec
- `compute_contagion_loss(shocked_graph) -> ScenarioLoss`: wrap `dexposure_fm.macroprudential_tools.simulate_contagion()`
- `run_scenarios(pred_graph, mc_samples, config) -> ScenarioSummary`: run S1-S5, aggregate with CVaR

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add lib/agent/scenario.py tests/agent/test_scenario.py
git commit -m "feat(agent): implement Scenario Engine with S1-S5 stress tests"
```

---

### Task 7: Decision Module

**Files:**
- Create: `lib/agent/decision.py`
- Test: `tests/agent/test_decision.py`

Implements Section 2.7: playbook constraints, ticket generation and scoring (Eq. 8).

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/test_decision.py
import pytest
from lib.agent.decision import generate_tickets
from lib.agent.types import Alert, ScenarioSummary, DataHealthResult, ScenarioLoss
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
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement decision.py**

Implement playbook from EXPERIMENT_PLAN Section 2.4:
- `PLAYBOOK`: dict mapping action -> (severity, requires_safe_mode_off, min_alerts)
- `generate_tickets(alerts, scenario, data_health, config) -> DecisionResult`
- Score formula: `Sev * Conf * Imp` (Eq. 8)
- Constraint: Eq. 7 — check safe_mode and min confidence

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add lib/agent/decision.py tests/agent/test_decision.py
git commit -m "feat(agent): implement Decision module with playbook constraints"
```

---

### Task 8: Agent Loop (Algorithm 1)

**Files:**
- Create: `lib/agent/agent_loop.py`
- Test: `tests/agent/test_agent_loop.py`

End-to-end orchestration: DataHealth -> Forecast (API call) -> Monitor -> Scenario -> Decision. This is Algorithm 1 from the paper.

- [ ] **Step 1: Write failing tests**

Test `run_epoch()` with mocked API responses. Test that safe_mode propagates correctly. Test that output contains all expected fields.

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement agent_loop.py**

```python
async def run_epoch(
    graph: GraphSnapshot,
    baseline_history: list[dict],
    config: AgentConfig,
    api_client: httpx.AsyncClient | None = None,
) -> AgentOutput:
    """Algorithm 1: full agent loop for one epoch."""
    # 1. DataHealth gate
    dh = compute_data_health(graph, config)

    all_alerts = []
    all_scenario_losses = []

    for h in config.horizons:
        # 2. Forecast (call GPU server)
        prediction = await call_forecast_api(graph, h, api_client, config)

        # 3. Build predicted graph + MC samples
        g_hat = build_pred_graph(prediction, config)
        samples = mc_sample(prediction, config)

        # 4. Monitor
        monitor_result = run_monitor(g_hat, samples, baseline_history, h, config)
        all_alerts.extend(monitor_result.alerts)

        # 5. Scenario engine
        scenario_result = run_scenarios(g_hat, samples, config)
        all_scenario_losses.extend(scenario_result.ranked_losses)

    # 6. Aggregate scenarios
    scenario_summary = aggregate_scenarios(all_scenario_losses)

    # 7. Decision
    decision = generate_tickets(all_alerts, scenario_summary, dh, config)

    return AgentOutput(
        epoch_date=graph.date,
        data_health=dh,
        alerts=all_alerts,
        scenario_summary=scenario_summary,
        tickets=decision.tickets,
    )
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add lib/agent/agent_loop.py tests/agent/test_agent_loop.py
git commit -m "feat(agent): implement end-to-end agent loop (Algorithm 1)"
```

---

## Phase 3: GPU Model Serving API

### Task 9: FastAPI Server

**Files:**
- Create: `lib/agent/serve.py`
- Test: `tests/agent/test_serve.py`

REST API that wraps DeXposure-FM inference. Runs on GPU server. Keeps model loaded in VRAM.

- [ ] **Step 1: Write failing tests using httpx TestClient**

```python
# tests/agent/test_serve.py
import pytest
from httpx import AsyncClient, ASGITransport
from lib.agent.serve import create_app


@pytest.fixture
def app():
    return create_app(mock_mode=True)  # Uses mock model for testing


@pytest.mark.asyncio
async def test_health_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_forecast_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/forecast", json={
            "graph": {"date": "2025-01-01", "nodes": {}, "edges": []},
            "horizon": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "edge_probs" in data
        assert "edge_weights" in data


@pytest.mark.asyncio
async def test_models_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/models")
        assert resp.status_code == 200
        assert "dexposure-fm" in resp.json()["loaded"]
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement serve.py**

Endpoints:
- `GET /health` — server status, GPU memory, loaded models
- `POST /forecast` — `{graph, horizon}` -> `{edge_probs, edge_weights, weight_stds, node_ids}`
- `POST /batch-forecast` — `{graph, horizons}` -> list of predictions (parallel)
- `GET /models` — list loaded model backbones
- `POST /data-health-raw` — raw freshness/missingness stats (GPU-side checks)

Model loading: lazy-load on first request, keep in VRAM. Use `mock_mode=True` for tests (returns random predictions with correct structure).

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add lib/agent/serve.py tests/agent/test_serve.py
git commit -m "feat(agent): implement FastAPI model serving with mock mode"
```

---

### Task 10: Server Deployment Script

**Files:**
- Create: `scripts/start_server.sh`
- Create: `scripts/sync_and_serve.sh`

Scripts for deploying the API on gpu-server via SSH (per CLAUDE.md instructions).

- [ ] **Step 1: Create start_server.sh**

```bash
#!/usr/bin/env bash
# Start the DeXposure-Agent API server on GPU server.
# Usage: ssh gpu-server 'bash -s' < scripts/start_server.sh
set -euo pipefail

cd ~/CodeProjects/graph-dexposure
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export DEXPOSURE_CHECKPOINT_DIR=./checkpoints

echo "[$(date)] Starting DeXposure-Agent API server..."
python -m uvicorn lib.agent.serve:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --timeout-keep-alive 300 \
    2>&1 | tee logs/serve_$(date +%Y%m%d_%H%M%S).log
```

- [ ] **Step 2: Create sync_and_serve.sh**

```bash
#!/usr/bin/env bash
# Sync local code to GPU server and (re)start API server.
set -euo pipefail

GPU_HOST="gpu-server"
REMOTE_DIR="~/CodeProjects/graph-dexposure"

echo "[sync] Pushing latest code to ${GPU_HOST}..."
rsync -avz --exclude='.venv' --exclude='data/' --exclude='checkpoints/' \
    ./ "${GPU_HOST}:${REMOTE_DIR}/"

echo "[serve] Restarting API server..."
ssh "${GPU_HOST}" "cd ${REMOTE_DIR} && pkill -f 'uvicorn lib.agent.serve' || true"
ssh "${GPU_HOST}" "cd ${REMOTE_DIR} && nohup bash scripts/start_server.sh > /dev/null 2>&1 &"

echo "[done] Server starting at ${GPU_HOST}:8000"
```

- [ ] **Step 3: Commit**

```bash
chmod +x scripts/start_server.sh scripts/sync_and_serve.sh
git add scripts/start_server.sh scripts/sync_and_serve.sh
git commit -m "feat(agent): add GPU server deployment scripts"
```

---

## Phase 4: Claude Code Plugin

### Task 11: Plugin Scaffold

**Files:**
- Create: `plugin/dexposure-agent/.claude-plugin/plugin.json`
- Create: `plugin/dexposure-agent/scripts/call-api.py`
- Create: `plugin/dexposure-agent/hooks/hooks.json`
- Create: `plugin/dexposure-agent/hooks/scripts/check-server-health.sh`

- [ ] **Step 1: Create plugin.json**

```json
{
  "name": "dexposure-agent",
  "version": "0.1.0",
  "description": "DeFi credit-exposure risk monitoring agent powered by DeXposure-FM. Provides network risk monitoring, stress testing, and decision recommendations for DeFi protocols.",
  "author": {
    "name": "DeXposure Team"
  },
  "license": "MIT",
  "keywords": ["defi", "risk", "monitoring", "graph", "foundation-model", "stress-testing"]
}
```

- [ ] **Step 2: Create call-api.py helper**

```python
#!/usr/bin/env python3
"""Thin httpx wrapper for calling the DeXposure-Agent API from plugin skills.

Usage:
    python call-api.py forecast --date 2025-01-01 --horizon 4
    python call-api.py health
    python call-api.py run-epoch --date 2025-01-01
"""
import argparse
import json
import sys
import httpx

DEFAULT_BASE = "http://gpu-server:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["health", "forecast", "run-epoch", "models"])
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--date", help="Epoch date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--output", choices=["json", "summary"], default="summary")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=300)

    if args.command == "health":
        r = client.get("/health")
        print(json.dumps(r.json(), indent=2))
    elif args.command == "models":
        r = client.get("/models")
        print(json.dumps(r.json(), indent=2))
    elif args.command == "forecast":
        r = client.post("/forecast", json={"date": args.date, "horizon": args.horizon})
        data = r.json()
        if args.output == "summary":
            print(f"Forecast for h={args.horizon}: {len(data.get('edge_probs', {}))} edges predicted")
        else:
            print(json.dumps(data, indent=2))
    elif args.command == "run-epoch":
        r = client.post("/run-epoch", json={"date": args.date})
        data = r.json()
        if args.output == "summary":
            print(f"Epoch {args.date}: {len(data.get('alerts', []))} alerts, {len(data.get('tickets', []))} tickets")
        else:
            print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create hooks.json and health check script**

```json
{
  "SessionStart": [
    {
      "matcher": "startup",
      "hooks": [
        {
          "type": "command",
          "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/check-server-health.sh",
          "timeout": 10
        }
      ]
    }
  ]
}
```

```bash
#!/usr/bin/env bash
# Check if the DeXposure-Agent API server is reachable.
GPU_API="${DEXPOSURE_API_URL:-http://gpu-server:8000}"
if curl -sf "${GPU_API}/health" > /dev/null 2>&1; then
    echo "[dexposure-agent] GPU server is reachable at ${GPU_API}"
else
    echo "[dexposure-agent] WARNING: GPU server not reachable at ${GPU_API}. Some skills will be unavailable."
fi
```

- [ ] **Step 4: Commit**

```bash
git add plugin/dexposure-agent/
git commit -m "feat(plugin): scaffold Claude Code plugin with manifest and helpers"
```

---

### Task 12: Skills — data-health, forecast, monitor

**Files:**
- Create: `plugin/dexposure-agent/skills/data-health/SKILL.md`
- Create: `plugin/dexposure-agent/skills/forecast/SKILL.md`
- Create: `plugin/dexposure-agent/skills/monitor/SKILL.md`
- Create: `plugin/dexposure-agent/skills/monitor/references/metrics-reference.md`

- [ ] **Step 1: Write data-health SKILL.md**

```markdown
---
name: Data Health Assessment
description: >
  This skill should be used when the user asks to "check data quality",
  "assess data health", "is the data fresh", "run data health gate",
  "check if data is safe", or mentions DeFi graph data quality.
  Evaluates freshness, missingness, topology, and discontinuities
  in DeFi credit-exposure graph snapshots.
version: 0.1.0
---

# Data Health Assessment

Assess the quality of a DeFi credit-exposure graph snapshot before
running the agent pipeline. Maps to Section 2.4 of the DeXposure-Agent paper.

## What It Checks

| Check | What | Score 1.0 means |
|-------|------|-----------------|
| Freshness | Date parseable, not stale | Data is current |
| Missingness | Non-zero node features | All features present |
| Topology | Edge/node ratio | Well-connected graph |
| Discontinuity | Consistency with history | No sudden breaks |

## How to Run

Execute the data-health check via the API:

    python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py run-epoch --date YYYY-MM-DD --output json

Parse `data_health` from the response:
- `score`: DH_t in [0,1]
- `safe_mode`: True if score < tau_data (default 0.7)
- `checks`: individual check scores

## Interpreting Results

- **score >= 0.7**: Data is healthy. Full agent pipeline proceeds normally.
- **score < 0.7 (SAFE_MODE)**: Data quality concerns detected.
  The agent suppresses intervention-type recommendations (Recommend-Reduce, Contingency)
  and only emits Monitor/Investigate tickets.

## When Safe Mode Triggers

Common causes: stale data (API down), many missing features (new protocols),
sudden topology change (protocol migration), edge weight outliers.

Action: investigate the specific checks that scored low before trusting any agent outputs.
```

- [ ] **Step 2: Write forecast SKILL.md**

Similar structure — covers how to call the forecast API, interpret edge_probs/weights, understand horizons {1,4,8,12}.

- [ ] **Step 3: Write monitor SKILL.md + metrics-reference.md**

SKILL.md: lean overview of monitoring (alert trigger, confidence, attribution).
metrics-reference.md: detailed table of M1-M7 with formulas and interpretation.

- [ ] **Step 4: Commit**

```bash
git add plugin/dexposure-agent/skills/
git commit -m "feat(plugin): add data-health, forecast, and monitor skills"
```

---

### Task 13: Skills — scenario, decision, run-epoch

**Files:**
- Create: `plugin/dexposure-agent/skills/scenario/SKILL.md`
- Create: `plugin/dexposure-agent/skills/scenario/references/scenario-library.md`
- Create: `plugin/dexposure-agent/skills/decision/SKILL.md`
- Create: `plugin/dexposure-agent/skills/decision/references/playbook-reference.md`
- Create: `plugin/dexposure-agent/skills/run-epoch/SKILL.md`
- Create: `plugin/dexposure-agent/skills/evaluate/SKILL.md`
- Create: `plugin/dexposure-agent/skills/domain-knowledge/SKILL.md`
- Create: `plugin/dexposure-agent/skills/domain-knowledge/references/defi-protocols.md`
- Create: `plugin/dexposure-agent/skills/domain-knowledge/references/risk-metrics.md`

- [ ] **Step 1: Write scenario SKILL.md + scenario-library.md**

SKILL.md: lean overview of stress testing (apply shocks, compute contagion, rank losses).
scenario-library.md: detailed S1-S5 specs from EXPERIMENT_PLAN Section 2.3.

- [ ] **Step 2: Write decision SKILL.md + playbook-reference.md**

SKILL.md: lean overview of ticket generation (constraints, scoring).
playbook-reference.md: full action table from Section 2.4.

- [ ] **Step 3: Write run-epoch SKILL.md**

Covers the full Algorithm 1 end-to-end — how to run a complete analysis epoch.

- [ ] **Step 4: Write evaluate SKILL.md**

Covers self-evaluation: run B1/B3/B4 metrics on user's own data.

- [ ] **Step 5: Write domain-knowledge SKILL.md + references**

DeFi protocol categories, risk metric definitions — background knowledge for analysis.

- [ ] **Step 6: Commit**

```bash
git add plugin/dexposure-agent/skills/
git commit -m "feat(plugin): add scenario, decision, run-epoch, evaluate, and domain-knowledge skills"
```

---

### Task 14: Agents — risk-analyst, stress-tester

**Files:**
- Create: `plugin/dexposure-agent/agents/risk-analyst.md`
- Create: `plugin/dexposure-agent/agents/stress-tester.md`

- [ ] **Step 1: Write risk-analyst.md**

```markdown
---
description: |
  Use this agent when the user asks to "analyze DeFi risk", "run risk monitoring",
  "check protocol exposure", "what's the current risk state", or wants an autonomous
  end-to-end risk assessment. This agent runs the full DeXposure-Agent pipeline
  (Algorithm 1) and presents results as a structured risk report.

  <example>
  Context: User wants to understand current DeFi risk landscape
  user: "What does the risk landscape look like for this week?"
  assistant: "I'll use the risk-analyst agent to run a full monitoring epoch."
  <commentary>
  User is requesting comprehensive risk analysis — trigger the autonomous agent.
  </commentary>
  </example>

  <example>
  Context: User asks about specific protocol risk
  user: "Is Aave showing any warning signs?"
  assistant: "I'll use the risk-analyst agent to check for alerts related to Aave."
  <commentary>
  Protocol-specific query still benefits from full pipeline context.
  </commentary>
  </example>
model: inherit
color: cyan
tools: ["Read", "Bash", "Write"]
---

# DeXposure Risk Analyst

Expert identity: autonomous DeFi risk monitoring agent powered by the DeXposure-FM
foundation model. Specializes in network-level credit exposure analysis, early warning
detection, and risk-mitigating recommendations.

## Core Process

1. Check GPU server health via `${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py health`
2. Run full epoch: `python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py run-epoch --date <date> --output json`
3. Parse the AgentOutput JSON
4. Present results as structured report:

### Report Structure

**Data Health**: DH_t score, safe mode status, individual checks
**Alerts** (sorted by confidence):
  - For each alert: metric name, horizon, z-score, confidence, top attributions
**Stress Tests** (sorted by expected loss):
  - For each scenario: loss, distressed count, propagation depth, targets
**Recommendations** (sorted by score):
  - For each ticket: action, severity, targets, rationale

## Quality Standards

- Always report data health status FIRST — if safe mode is active, prominently warn
- Include confidence scores with every alert — never present uncertain alerts as facts
- Attribute risk to specific protocols/sectors — avoid vague "systemic risk" statements
- When presenting tickets, always include the evidence bundle
```

- [ ] **Step 2: Write stress-tester.md**

Focused subagent for scenario analysis only — runs S1-S5 and presents contagion results.

- [ ] **Step 3: Commit**

```bash
git add plugin/dexposure-agent/agents/
git commit -m "feat(plugin): add risk-analyst and stress-tester agents"
```

---

### Task 15: Main Command — /dexposure

**Files:**
- Create: `plugin/dexposure-agent/commands/dexposure.md`

- [ ] **Step 1: Write command**

```markdown
---
name: dexposure
description: Run DeXposure-Agent risk monitoring analysis
argument-hint: "[date] [--horizon N] [--scenario S1-S5] [--full]"
---

# /dexposure Command

Run the DeXposure-Agent pipeline for DeFi credit-exposure risk monitoring.

## Usage Patterns

- `/dexposure` — Run full analysis for latest available date
- `/dexposure 2025-03-01` — Run analysis for specific date
- `/dexposure --scenario S1` — Run only scenario S1 analysis
- `/dexposure --full` — Full analysis with all scenarios and detailed attribution

## Process

1. Parse arguments to determine: target date, horizons, scenarios
2. Check GPU server health: `python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py health`
3. If server unreachable, inform user and suggest running `scripts/sync_and_serve.sh`
4. Run analysis: `python ${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py run-epoch --date <date> --output json`
5. Present results using the risk-analyst agent's report structure
6. If user asks follow-up questions, use relevant skills (monitor, scenario, decision)
```

- [ ] **Step 2: Commit**

```bash
git add plugin/dexposure-agent/commands/
git commit -m "feat(plugin): add /dexposure main entry command"
```

---

## Phase 5: Experiment Scripts (Paper Benchmarks)

### Task 16: Experiment Runner Skeleton

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/run_all.py`
- Create: `experiments/b1_risk_forecasting.py`

This task creates the experiment framework that reuses `lib/agent/` for batch evaluation. One script per benchmark (B1-B6), matching EXPERIMENT_PLAN Section 3.

- [ ] **Step 1: Create run_all.py skeleton**

Master script that:
- Accepts `--benchmarks B1,B2,...` and `--methods C0,C1,...` flags
- Implements the competitor-benchmark applicability matrix from Section 4.3:
  - B2 (Early Warning): agent methods only (C0-C3)
  - B3 (Uncertainty): methods with MC (C0, C1, C4)
  - B5 (Decision): agent methods only (C0-C3)
  - B6 (Robustness): skip Persistence-Agent and Static GCN
- Calls individual benchmark scripts, skipping inapplicable combinations
- Collects results into Tables 1-7 (LaTeX output)
- Logs everything to `logs/experiments/`

- [ ] **Step 2: Create b1_risk_forecasting.py**

Implements B1 from Section 3.2:
- For each method and horizon, predict risk metrics on G_hat vs actual G_{t+h}
- Compute: PageRank MAE, HHI MAE, Density MAE, Gini MAE, Rank Correlation, Trend Consistency
- Output: Table 1 in LaTeX format

Uses `lib.agent.monitor.compute_metrics()` for Phi functionals — same code path as plugin.

- [ ] **Step 3: Commit**

```bash
git add experiments/
git commit -m "feat(experiments): add benchmark runner skeleton for B1-B6"
```

---

### Task 17: Remaining Benchmarks (B2-B6) + Ablations

**Files:**
- Create: `experiments/b2_early_warning.py`
- Create: `experiments/b3_uncertainty_calibration.py`
- Create: `experiments/b4_stress_test.py`
- Create: `experiments/b5_decision_quality.py`
- Create: `experiments/b6_robustness.py`
- Create: `experiments/ablations.py`

- [ ] **Step 1: Implement B2-B6**

Each script follows the same pattern:
1. Load test-split data
2. For each method: run agent pipeline (or baseline), collect outputs
3. Compute benchmark-specific metrics (from EXPERIMENT_PLAN Sections 3.3-3.7)
4. Output LaTeX table

- [ ] **Step 2: Implement ablations.py**

For each ablation A1-A8 (Section 4.2):
- A1: `AgentConfig(tau_data=0.0)` — disable data-health gating
- A2: `AgentConfig(tau_conf=0.0)` — disable confidence scoring
- A3: Skip scenario engine — set `run_scenarios = lambda *a: empty_summary`
- A4: Skip attribution — set top_k=0
- A5: Fixed baselines — disable rolling window
- A6: Single horizon — `horizons=[1]`
- A7: No MC — `mc_samples=1`
- A8: No playbook constraints — all actions always feasible

Run B1-B5 key metrics for each ablation, output Table 7.

- [ ] **Step 3: Commit**

```bash
git add experiments/
git commit -m "feat(experiments): implement B2-B6 benchmarks and A1-A8 ablations"
```

---

### Task 18: Competitor Wrappers

**Files:**
- Create: `experiments/competitors/roland_agent.py`
- Create: `experiments/competitors/persistence_agent.py`
- Create: `experiments/competitors/llm_agent.py`
- Create: `experiments/competitors/baselines.py`

- [ ] **Step 1: Implement competitor wrappers**

Each wrapper uses the SAME `lib/agent/` pipeline but swaps the backbone:
- **C1 (ROLAND-Agent)**: Replace FM forecast with ROLAND (GCN+GRU) inference
- **C2 (Persistence-Agent)**: Replace FM forecast with `G_{t+h} = G_t`
- **C3 (LLM-Agent)**: Replace FM forecast with Claude/GPT-4 reasoning on tabular metrics
- **C4-C10**: Non-agent baselines — just run backbone, compute B1/B4 metrics directly

- [ ] **Step 2: Commit**

```bash
git add experiments/competitors/
git commit -m "feat(experiments): add competitor wrappers C1-C10"
```

---

## Phase 5b: Hyperparameter Tuning + Walk-Forward

### Task 16b: Hyperparameter Grid Search on Validation Split

**Files:**
- Create: `experiments/tune_agent.py`

Implements the grid search from EXPERIMENT_PLAN Section 2.5 on the validation split (2024-07 ~ 2024-12).

- [ ] **Step 1: Implement tune_agent.py**

Search grid (from Section 2.5):
- `pi_min`: {0.1, 0.2, 0.3}
- `z_threshold`: {1.5, 2.0, 2.5}
- `rolling_window`: {13, 26, 52}
- `tau_data`: {0.5, 0.6, 0.7}
- `tau_conf`: {0.4, 0.5, 0.6}
- `lambda_tail`: {0.0, 0.25, 0.5}
- `mc_samples`: {20, 50, 100}
- `top_k`: {5, 10, 20}

Use Optuna (already a dependency) to search this space. Optimize a composite objective from B1 (Rank Correlation) + B2 (F1-warning) + B5 (Ticket Precision) on the val split.

Save best config to `results/best_agent_config.json`.

- [ ] **Step 2: Commit**

```bash
git add experiments/tune_agent.py
git commit -m "feat(experiments): add hyperparameter grid search on val split"
```

---

### Task 16c: Walk-Forward Cross-Validation

**Files:**
- Create: `experiments/walk_forward.py`

Implements the expanding-window walk-forward protocol (15 folds) from EXPERIMENT_PLAN Section 1.2.

- [ ] **Step 1: Implement walk_forward.py**

Protocol:
- Start with minimal training window (2020-03 ~ 2022-06)
- Expand training window by ~4 weeks per fold
- Evaluate on next 4 weeks
- 15 folds total, covering 2022-07 ~ 2025-08
- Run B1 key metrics at each fold
- Report mean +/- std across folds

- [ ] **Step 2: Commit**

```bash
git add experiments/walk_forward.py
git commit -m "feat(experiments): add 15-fold walk-forward cross-validation"
```

---

## Phase 6: Integration Testing + Polish

### Task 19: End-to-End Integration Test

**Files:**
- Create: `tests/agent/test_integration.py`

- [ ] **Step 1: Write integration test**

Test the full pipeline: load a real graph snapshot from `data/`, run `run_epoch()` with mock API, verify output structure.

- [ ] **Step 2: Run and verify**
- [ ] **Step 3: Commit**

---

### Task 20: Plugin Validation

- [ ] **Step 1: Validate plugin structure**

Verify all SKILL.md files have correct frontmatter, all referenced files exist, hooks.json is valid.

- [ ] **Step 2: Test plugin locally**

```bash
claude --plugin-dir plugin/dexposure-agent
# Test: /dexposure
# Test: ask about risk metrics (should trigger domain-knowledge skill)
# Test: ask to check data quality (should trigger data-health skill)
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(plugin): complete DeXposure-Agent plugin with all skills and agents"
```

---

### Task 21: Figures and Visualization (Phase 5 of Spec)

**Files:**
- Create: `experiments/figures.py`

Generates all figures for the paper (EXPERIMENT_PLAN Section 5 Phase 5).

- [ ] **Step 1: Implement figures.py**

Figures to generate:
1. **Timeline plots**: alerts vs actual stress events (Terra, FTX, SVB) with lead-time markers
2. **Calibration reliability diagrams**: predicted confidence vs actual alert precision
3. **Scenario loss heatmaps**: scenarios (rows) x horizons (cols), color by expected loss
4. **Decision ticket examples**: qualitative samples showing full evidence bundles
5. **Robustness degradation plots**: B6 metrics across regimes

Save all figures to `results/figures/`.

- [ ] **Step 2: Commit**

```bash
git add experiments/figures.py
git commit -m "feat(experiments): add figure generation for paper"
```

---

## Dependency Graph

```
Task 1 (config) ──┐
                   ├── Task 3 (data_health)  ──┐
Task 2 (types)  ──┤                            │
                   ├── Task 4 (pred_graph)   ──┼── Task 8 (agent_loop) ── Task 9 (serve) ── Task 10 (deploy)
                   ├── Task 5 (monitor)      ──┤          │
                   ├── Task 6 (scenario)     ──┤          │
                   └── Task 7 (decision)     ──┘          │
                                                          │
Task 11 (plugin scaffold) ── Task 12-13 (skills) ── Task 14 (agents) ── Task 15 (command) ── Task 20 (validate)
                                                          │
Task 16-17 (experiments) ── Task 18 (competitors) ── Task 19 (integration)
```

**Parallelizable groups:**
- Tasks 3, 4, 5, 6, 7 (all depend on Tasks 1+2, independent of each other)
- Tasks 11-15 (plugin) can run parallel with Tasks 16-18 (experiments) after Task 8
- Task 19 and Task 20 are final integration gates

---

## GPU Server Execution Plan

Per CLAUDE.md: "ssh gpu-server" for compute, sync code first, maximize GPU utilization.

**Batch training sequence (Phase 1 from EXPERIMENT_PLAN):**

```bash
# Run sequentially on GPU server to maximize utilization
ssh gpu-server << 'SCRIPT'
cd ~/CodeProjects/graph-dexposure
source .venv/bin/activate

# 1. Verify DeXposure-FM checkpoint
python -c "from lib.agent.serve import load_model; load_model('dexposure-fm')"

# 2. Train ROLAND baseline (if not already trained)
python DeXposure/src/models/defi_roland_gnn.py --train --save-checkpoint

# 3. Train EvolveGCN, DyRep, TGN, Static GCN
python experiments/competitors/baselines.py --train-all --save-checkpoints

# 4. Start API server for interactive use
nohup python -m uvicorn lib.agent.serve:app --host 0.0.0.0 --port 8000 &

# 5. Run all experiments (B1-B6 + ablations)
python experiments/run_all.py --benchmarks B1,B2,B3,B4,B5,B6 --methods all --output results/
SCRIPT
```
