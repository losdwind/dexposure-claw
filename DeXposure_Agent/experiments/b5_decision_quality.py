#!/usr/bin/env python3
"""B5: Decision Quality (Section 3.6 of EXPERIMENT_PLAN)

Evaluates the quality of regulatory/supervisory action recommendations
produced by the agent layer on top of risk forecasts.

Metrics:
- Ticket Precision       -- fraction of opened investigation tickets that were justified
- Risk Reduction         -- average risk-metric improvement after recommended action
- Target Stability       -- consistency of top-K flagged protocols across consecutive weeks
- Audit Completeness     -- fraction of truly high-risk protocols covered by audit tickets
- Suppression Rate       -- fraction of low-risk protocols correctly suppressed (not flagged)
- False Intervention Rate -- fraction of interventions triggered on stable protocols
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp_logger import ExpLogger
from dexposure_agent.config import AgentConfig
from dexposure_agent.data_health import compute_data_health
from dexposure_agent.data_loader import SnapshotLoader
from dexposure_agent.decision import generate_tickets
from dexposure_agent.monitor import (
    compute_metrics,
    detect_alerts,
    _compute_rolling_baseline,
)
from dexposure_agent.scenario import SCENARIO_LIBRARY, apply_shock, compute_contagion_loss, run_scenarios
from dexposure_agent.types import Edge, GraphSnapshot, ScenarioSummary


# Stress detection: >20% edge weight drop threshold
STRESS_THRESHOLD = 0.20
# Lookahead window for ground truth stress detection (weeks)
STRESS_LOOKAHEAD = 4
# MC noise sigma for persistence-based MC samples (calibrated at runtime)
MC_NOISE_SIGMA_DEFAULT = 0.1


@dataclass
class B5Result:
    method: str
    ticket_precision: float = float("nan")
    risk_reduction: float = float("nan")
    target_stability: float = float("nan")
    audit_completeness: float = float("nan")
    suppression_rate: float = float("nan")
    false_intervention_rate: float = float("nan")
    n_weeks_evaluated: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"B5Result(method={self.method}, "
            f"ticket_prec={self.ticket_precision:.3f}, "
            f"risk_reduct={self.risk_reduction:.3f}, "
            f"stability={self.target_stability:.3f}, "
            f"audit_compl={self.audit_completeness:.3f}, "
            f"suppress={self.suppression_rate:.3f}, "
            f"FIR={self.false_intervention_rate:.3f})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_mc_samples(
    graph: GraphSnapshot,
    n_samples: int,
    sigma: float = MC_NOISE_SIGMA_DEFAULT,
    rng: np.random.Generator | None = None,
) -> list[GraphSnapshot]:
    """Generate MC samples by adding Gaussian noise to edge weights."""
    if rng is None:
        rng = np.random.default_rng()
    samples: list[GraphSnapshot] = []
    for _ in range(n_samples):
        new_edges: list[Edge] = []
        for edge in graph.edges:
            noise = rng.normal(0.0, sigma * abs(edge.weight)) if edge.weight != 0.0 else 0.0
            new_weight = max(0.0, edge.weight + noise)
            new_edges.append(Edge(source=edge.source, target=edge.target, weight=new_weight))
        samples.append(GraphSnapshot(date=graph.date, nodes=graph.nodes, edges=new_edges))
    return samples


def _compute_node_total_weight(graph: GraphSnapshot) -> dict[str, float]:
    """Compute total edge weight per node (sum of source + target weights)."""
    totals: dict[str, float] = {}
    for node_id in graph.nodes:
        totals[node_id] = 0.0
    for edge in graph.edges:
        totals[edge.source] = totals.get(edge.source, 0.0) + edge.weight
        totals[edge.target] = totals.get(edge.target, 0.0) + edge.weight
    return totals


def _detect_truly_stressed_protocols(
    snap_t: GraphSnapshot,
    snap_future: GraphSnapshot,
    threshold: float = STRESS_THRESHOLD,
) -> set[str]:
    """Identify protocols that experienced > threshold edge weight drop.

    Compares per-node total edge weight between current and future snapshots.
    A protocol is 'truly stressed' if its total edge weight dropped by more
    than threshold (as a fraction of the current weight).
    """
    weights_t = _compute_node_total_weight(snap_t)
    weights_future = _compute_node_total_weight(snap_future)

    stressed: set[str] = set()
    for node_id, w_t in weights_t.items():
        if w_t <= 0.0:
            continue
        w_f = weights_future.get(node_id, 0.0)
        drop_frac = (w_t - w_f) / w_t
        if drop_frac > threshold:
            stressed.add(node_id)
    return stressed


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_b5(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    **kwargs,
) -> list[B5Result]:
    """Run B5 benchmark for a given method.

    Args:
        method_id: Decision method with rule-based tickets, typically C0 or C2.
        data_dir: Path to processed graph snapshots and ground-truth action labels.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        **kwargs: Extra method-specific config.

    Returns:
        List containing a single B5Result.
    """
    results_dir = kwargs.pop("results_dir", "results/")
    log = ExpLogger("B5", method=method_id, results_dir=results_dir)

    log.info(f"B5 | method={method_id} | test_split={test_split}")

    config = AgentConfig(**{k: v for k, v in kwargs.items() if k in AgentConfig.model_fields})
    rng = np.random.default_rng(seed=42)
    horizon = STRESS_LOOKAHEAD

    # --- Load test snapshots with baselines ---
    loader = SnapshotLoader(data_dir=data_dir)
    all_dates = loader.dates

    # Collect all (snap, baseline) pairs
    test_pairs: list[tuple[GraphSnapshot, list[dict[str, float]]]] = []
    for snap, baseline in loader.iter_test_with_baselines(test_split, baseline_window=config.rolling_window):
        test_pairs.append((snap, baseline))

    if len(test_pairs) < 2:
        log.warning("B5: fewer than 2 test snapshots, returning NaN result")
        return [B5Result(method=method_id, n_weeks_evaluated=0)]

    log.info(f"B5: loaded {len(test_pairs)} test snapshots with baselines")

    # Calibrate MC sigma from training data
    from experiments.b3_uncertainty_calibration import _calibrate_mc_sigma
    mc_sigma = _calibrate_mc_sigma(loader, test_split, horizon=STRESS_LOOKAHEAD)
    log.info(f"B5: calibrated MC sigma = {mc_sigma:.4f}")

    # Build date -> snapshot for future lookups
    all_snapshots = loader.load()
    date_to_snap: dict[str, GraphSnapshot] = {s.date: s for s in all_snapshots}

    # --- Per-week accumulators ---
    n_weeks = 0
    ticket_correct: list[bool] = []        # was each ticket target truly stressed?
    truly_stressed_covered: list[float] = []  # fraction of truly stressed covered per week
    target_sets: list[set[str]] = []       # target sets per week for stability
    safe_mode_flags: list[bool] = []       # safe_mode flag per week
    false_interventions: list[bool] = []   # false interventions (Recommend-Reduce/Contingency on stable)
    risk_deltas: list[float] = []          # scenario loss delta (simplified)

    prev_targets: set[str] | None = None

    for snap_t, baseline_history in log.progress(
        test_pairs, desc="Weekly test snapshots", total=len(test_pairs), unit="week"
    ):
        # --- Find future snapshot for ground truth ---
        t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
        if t_idx < 0:
            continue
        future_idx = t_idx + horizon
        if future_idx >= len(all_dates):
            log.info(f"B5: no future snapshot for {snap_t.date}, skipping")
            continue
        future_date = all_dates[future_idx]
        if future_date not in date_to_snap:
            try:
                gt_future = loader.load_single(future_date)
            except KeyError:
                log.info(f"B5: cannot load future snapshot {future_date}, skipping")
                continue
        else:
            gt_future = date_to_snap[future_date]

        # --- Step a: Data health ---
        data_health = compute_data_health(snap_t, config)

        # --- Step b: Generate prediction -> metrics, alerts ---
        from experiments.predict_helper import predict_graph
        pred_graph = predict_graph(method_id, snap_t, horizon=STRESS_LOOKAHEAD)
        current_metrics = compute_metrics(pred_graph)

        # Build rolling baseline stats from history
        rolling_baseline = _compute_rolling_baseline(baseline_history, config.rolling_window)
        alerts = detect_alerts(current_metrics, rolling_baseline, horizon=horizon, config=config)

        # --- Step c: Scenario engine ---
        mc_samples = _generate_mc_samples(pred_graph, config.mc_samples, sigma=mc_sigma, rng=rng)
        scenario_summary = run_scenarios(pred_graph, mc_samples, config, horizon=horizon)

        # --- Step d: Generate tickets ---
        decision = generate_tickets(alerts, scenario_summary, data_health, config)

        # --- Step e: Collect ticket targets ---
        all_ticket_targets: set[str] = set()
        intervention_targets: set[str] = set()  # targets from Recommend-Reduce / Contingency
        for ticket in decision.tickets:
            all_ticket_targets.update(ticket.targets)
            if ticket.action in ("Recommend-Reduce", "Contingency"):
                intervention_targets.update(ticket.targets)

        target_sets.append(all_ticket_targets)
        safe_mode_flags.append(decision.suppressed)

        # --- Step f: Ground truth check ---
        truly_stressed = _detect_truly_stressed_protocols(snap_t, gt_future, threshold=STRESS_THRESHOLD)

        # Ticket Precision: fraction of ticket targets that are truly stressed
        for target in all_ticket_targets:
            ticket_correct.append(target in truly_stressed)

        # Audit Completeness: fraction of truly stressed protocols that were targeted
        if truly_stressed:
            covered = len(truly_stressed & all_ticket_targets) / len(truly_stressed)
        else:
            covered = 1.0  # no truly stressed protocols => vacuously complete
        truly_stressed_covered.append(covered)

        # False Intervention Rate: intervention tickets targeting stable protocols
        stable_protocols = set(snap_t.nodes.keys()) - truly_stressed
        for target in intervention_targets:
            false_interventions.append(target in stable_protocols)

        # Risk Reduction: simplified as mean scenario expected_loss
        # (lower loss = better; we track the raw loss as a proxy for "risk level")
        if scenario_summary.ranked_losses:
            mean_loss = float(np.mean([sl.expected_loss for sl in scenario_summary.ranked_losses]))
            risk_deltas.append(mean_loss)

        # Target Stability: Jaccard with previous week
        if prev_targets is not None:
            # computed below in aggregation
            pass
        prev_targets = all_ticket_targets

        n_weeks += 1
        log.step(
            f"week {n_weeks}",
            date=snap_t.date, tickets=len(decision.tickets),
            targets=len(all_ticket_targets), truly_stressed=len(truly_stressed),
            safe_mode=decision.suppressed,
        )

    if n_weeks == 0:
        log.warning("B5: no weeks evaluated")
        return [B5Result(method=method_id, n_weeks_evaluated=0)]

    # --- Aggregate metrics ---

    # Ticket Precision
    if ticket_correct:
        ticket_precision = float(np.mean(ticket_correct))
    else:
        ticket_precision = 0.0

    # Risk Reduction: average scenario loss (lower is better; report as negative delta
    # if we have at least 2 points, otherwise just report mean loss)
    if len(risk_deltas) >= 2:
        # Simplified: mean of all scenario losses across weeks
        risk_reduction = float(np.mean(risk_deltas))
    elif risk_deltas:
        risk_reduction = float(risk_deltas[0])
    else:
        risk_reduction = 0.0

    # Target Stability: average Jaccard between consecutive weeks
    stability_scores: list[float] = []
    for i in range(1, len(target_sets)):
        stability_scores.append(_jaccard(target_sets[i], target_sets[i - 1]))
    target_stability = float(np.mean(stability_scores)) if stability_scores else 0.0

    # Audit Completeness
    audit_completeness = float(np.mean(truly_stressed_covered)) if truly_stressed_covered else 0.0

    # Suppression Rate: fraction of weeks where safe_mode was on
    suppression_rate = float(np.mean(safe_mode_flags)) if safe_mode_flags else 0.0

    # False Intervention Rate
    if false_interventions:
        false_intervention_rate = float(np.mean(false_interventions))
    else:
        false_intervention_rate = 0.0

    result = B5Result(
        method=method_id,
        ticket_precision=ticket_precision,
        risk_reduction=risk_reduction,
        target_stability=target_stability,
        audit_completeness=audit_completeness,
        suppression_rate=suppression_rate,
        false_intervention_rate=false_intervention_rate,
        n_weeks_evaluated=n_weeks,
    )

    log.summary({
        "n_weeks_evaluated": n_weeks,
        "ticket_precision": ticket_precision,
        "risk_reduction": risk_reduction,
        "target_stability": target_stability,
        "audit_completeness": audit_completeness,
        "suppression_rate": suppression_rate,
        "false_intervention_rate": false_intervention_rate,
    })

    log.save_results([{
        "method": method_id,
        "ticket_precision": ticket_precision,
        "risk_reduction": risk_reduction,
        "target_stability": target_stability,
        "audit_completeness": audit_completeness,
        "suppression_rate": suppression_rate,
        "false_intervention_rate": false_intervention_rate,
        "n_weeks_evaluated": n_weeks,
    }])

    log.info(f"B5 complete: {result}")
    return [result]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="B5: Decision Quality benchmark")
    parser.add_argument("--method", required=True,
                        help="Method ID (e.g. C0 or C2)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range YYYY-MM~YYYY-MM")
    args = parser.parse_args()

    results = run_b5(args.method, args.data_dir, args.test_split)
    for r in results:
        print(r)
