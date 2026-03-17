"""Monitor module for the DeXposure-Agent pipeline.

Implements Section 2.5 of the DeXposure-Agent paper:
- Phi functionals (network-level risk metrics) computed on predicted GraphSnapshots
- Rolling baseline comparison via z-scores
- Alert generation with confidence scoring
"""
from __future__ import annotations

import math
from typing import Any

from lib.agent.config import AgentConfig
from lib.agent.types import Alert, GraphSnapshot, MonitorResult

# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------

METRIC_NAMES: dict[str, str] = {
    "M1": "SIS Weighted Degree",
    "M3": "HHI Concentration",
    "M4": "Graph Density",
    "M6": "PageRank Concentration (Gini)",
    "M7": "Gini Weighted Degree",
}


# ---------------------------------------------------------------------------
# Helper: Gini coefficient
# ---------------------------------------------------------------------------

def _gini(values: list[float]) -> float:
    """Compute Gini coefficient for a list of non-negative values.

    Returns 0.0 for empty or all-zero input.
    """
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0.0:
        return 0.0
    sorted_vals = sorted(values)
    cumsum = 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumsum += v
        gini_sum += cumsum
    # Gini = 1 - 2 * area under Lorenz curve
    return 1.0 - (2.0 * gini_sum) / (n * total)


# ---------------------------------------------------------------------------
# Helper: simple power-iteration PageRank
# ---------------------------------------------------------------------------

def _pagerank(
    nodes: list[str],
    adjacency: dict[str, dict[str, float]],
    damping: float = 0.85,
    iterations: int = 10,
) -> dict[str, float]:
    """Compute PageRank via power iteration.

    adjacency[src][tgt] = weight (used as uniform weight here for simplicity).
    """
    n = len(nodes)
    if n == 0:
        return {}
    pr: dict[str, float] = {node: 1.0 / n for node in nodes}
    # Build out-degree (unweighted for standard PageRank)
    out_degree: dict[str, int] = {node: len(adjacency.get(node, {})) for node in nodes}
    node_set = set(nodes)
    for _ in range(iterations):
        new_pr: dict[str, float] = {node: (1.0 - damping) / n for node in nodes}
        for src in nodes:
            targets = adjacency.get(src, {})
            od = out_degree[src]
            if od == 0:
                # Dangling node: distribute equally
                share = pr[src] / n
                for node in nodes:
                    new_pr[node] += damping * share
            else:
                share = pr[src] / od
                for tgt in targets:
                    if tgt in node_set:
                        new_pr[tgt] += damping * share
        pr = new_pr
    return pr


# ---------------------------------------------------------------------------
# Core: compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(graph: GraphSnapshot) -> dict[str, float]:
    """Compute network-level Phi functionals from a GraphSnapshot.

    Returns a dict mapping metric IDs (M1, M3, M4, M6, M7) to float values.
    For empty graphs, all metrics are 0.0.
    """
    nodes = list(graph.nodes.keys())
    edges = graph.edges
    n = len(nodes)
    num_edges = len(edges)

    if n == 0:
        return {mid: 0.0 for mid in METRIC_NAMES}

    # Build weighted adjacency and per-node weighted out-degree
    adjacency: dict[str, dict[str, float]] = {node: {} for node in nodes}
    weighted_degree: dict[str, float] = {node: 0.0 for node in nodes}

    for edge in edges:
        if edge.source in adjacency:
            adjacency[edge.source][edge.target] = (
                adjacency[edge.source].get(edge.target, 0.0) + edge.weight
            )
        if edge.source in weighted_degree:
            weighted_degree[edge.source] += edge.weight

    degree_values = list(weighted_degree.values())
    total_weighted_degree = sum(degree_values)

    # M1: SIS — max PageRank (systemic importance of the most important node)
    pr = _pagerank(nodes, adjacency)
    pr_values = list(pr.values())
    m1 = max(pr_values) if pr_values else 0.0
    m1 = max(0.0, min(1.0, m1))

    # M3: HHI on weighted degrees
    if total_weighted_degree > 0:
        m3 = sum((d / total_weighted_degree) ** 2 for d in degree_values)
    else:
        m3 = 0.0

    # M4: Directed graph density = E / (N * (N-1))
    max_possible_edges = n * (n - 1)
    m4 = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0
    m4 = max(0.0, min(1.0, m4))

    # M6: PageRank Concentration — Gini of PageRank distribution (pr already computed above)
    m6 = _gini(pr_values)
    m6 = max(0.0, min(1.0, m6))

    # M7: Gini of weighted degrees
    m7 = _gini(degree_values)
    m7 = max(0.0, min(1.0, m7))

    return {
        "M1": m1,
        "M3": m3,
        "M4": m4,
        "M6": m6,
        "M7": m7,
    }


# ---------------------------------------------------------------------------
# Core: detect_alerts
# ---------------------------------------------------------------------------

def detect_alerts(
    current_metrics: dict[str, float],
    baseline: dict[str, dict[str, float]],
    horizon: int,
    config: AgentConfig,
) -> list[Alert]:
    """Compare current metric values to rolling baseline and return triggered alerts.

    Args:
        current_metrics: metric_id -> current value.
        baseline: metric_id -> {"mean": float, "std": float}.
        horizon: forecast horizon (weeks).
        config: AgentConfig with z_threshold.

    Returns:
        List of Alert objects for metrics exceeding z_threshold.
    """
    alerts: list[Alert] = []
    epsilon = 1e-8

    for metric_id, value in current_metrics.items():
        if metric_id not in baseline:
            continue
        bline = baseline[metric_id]
        mean = bline["mean"]
        std = bline["std"]
        effective_std = max(std, epsilon)
        z_score = abs(value - mean) / effective_std

        if z_score > config.z_threshold:
            alerts.append(
                Alert(
                    horizon=horizon,
                    metric_id=metric_id,
                    metric_name=METRIC_NAMES.get(metric_id, metric_id),
                    value=value,
                    baseline_mean=mean,
                    baseline_std=std,
                    z_score=z_score,
                    confidence=0.0,  # Filled in by run_monitor
                    attribution={},
                )
            )

    return alerts


# ---------------------------------------------------------------------------
# Core: compute_confidence
# ---------------------------------------------------------------------------

def compute_confidence(
    dh_score: float,
    dispersion: float,
    horizon: int,
    config: AgentConfig,  # noqa: ARG001
) -> float:
    """Compute alert confidence score (Eq. 6 in DeXposure-Agent paper).

    C = dh_score * (1 / (1 + dispersion)) * (1 / (1 + log(1 + horizon)))

    Clamped to [0, 1].
    """
    raw = (
        float(dh_score)
        * (1.0 / (1.0 + float(dispersion)))
        * (1.0 / (1.0 + math.log(1.0 + float(horizon))))
    )
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Core: run_monitor
# ---------------------------------------------------------------------------

def _compute_rolling_baseline(
    history: list[dict[str, float]],
    window: int,
) -> dict[str, dict[str, float]]:
    """Compute rolling mean/std from a list of metric dicts."""
    recent = history[-window:]
    if not recent:
        return {}

    all_keys: set[str] = set()
    for entry in recent:
        all_keys.update(entry.keys())

    baseline: dict[str, dict[str, float]] = {}
    for key in all_keys:
        vals = [entry[key] for entry in recent if key in entry]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(variance)
        baseline[key] = {"mean": mean, "std": std}

    return baseline


def _compute_dispersion(
    metric_id: str,
    mc_samples: list[GraphSnapshot],
) -> float:
    """Compute metric dispersion (coefficient of variation) across MC samples."""
    if not mc_samples:
        return 0.0

    sample_values: list[float] = []
    for sample_graph in mc_samples:
        sample_metrics = compute_metrics(sample_graph)
        if metric_id in sample_metrics:
            sample_values.append(sample_metrics[metric_id])

    if len(sample_values) < 2:
        return 0.0

    mean = sum(sample_values) / len(sample_values)
    variance = sum((v - mean) ** 2 for v in sample_values) / len(sample_values)
    std = math.sqrt(variance)

    # Coefficient of variation, clamped to [0, 1]
    if mean == 0.0:
        return 0.0
    cv = std / abs(mean)
    return max(0.0, min(1.0, cv))


def run_monitor(
    predicted_graph: GraphSnapshot,
    mc_samples: list[GraphSnapshot],
    baseline_history: list[Any],
    horizon: int,
    config: AgentConfig,
    dh_score: float = 1.0,
) -> MonitorResult:
    """Run the full monitoring pipeline.

    Args:
        predicted_graph: The forecasted GraphSnapshot for this horizon.
        mc_samples: MC sampled graphs for uncertainty estimation.
        baseline_history: List of past metric dicts or GraphSnapshots.
        horizon: Forecast horizon in weeks.
        config: AgentConfig.
        dh_score: Data-health score (default 1.0 if not provided).

    Returns:
        MonitorResult with alerts and metrics.
    """
    # Step 1: Compute metrics on predicted graph
    current_metrics = compute_metrics(predicted_graph)

    # Step 2: Build baseline
    if not baseline_history:
        # First call — no baseline available, no alerts generated
        baseline: dict[str, dict[str, float]] = {}
    else:
        # Normalize baseline_history: may be list of GraphSnapshots or list of metric dicts
        history_metrics: list[dict[str, float]] = []
        for entry in baseline_history:
            if isinstance(entry, GraphSnapshot):
                history_metrics.append(compute_metrics(entry))
            elif isinstance(entry, dict):
                history_metrics.append(entry)
        baseline = _compute_rolling_baseline(history_metrics, config.rolling_window)

    # Step 3: Detect alerts
    alerts = detect_alerts(current_metrics, baseline, horizon=horizon, config=config)

    # Step 4: Compute dispersion and confidence for each alert
    final_alerts: list[Alert] = []
    for alert in alerts:
        dispersion = _compute_dispersion(alert.metric_id, mc_samples)
        confidence = compute_confidence(
            dh_score=dh_score,
            dispersion=dispersion,
            horizon=horizon,
            config=config,
        )
        final_alerts.append(
            alert.model_copy(update={"confidence": confidence})
        )

    # Step 5: Build MonitorResult
    # metrics structure: metric_id -> {horizon: value}
    metrics_out: dict[str, dict[int, float]] = {
        mid: {horizon: val} for mid, val in current_metrics.items()
    }

    return MonitorResult(alerts=final_alerts, metrics=metrics_out)
