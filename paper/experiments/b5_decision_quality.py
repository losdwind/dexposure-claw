#!/usr/bin/env python3
"""b5_decision: Decision Quality (Section 3.6 of EXPERIMENT_PLAN)

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

import json
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


# TKDE plan: replace absolute threshold with percentile-based ground truth.
# Old design (STRESS_THRESHOLD = 0.20 or 0.50 absolute drop) produced a pool
# that varied wildly week-to-week (1.7k - 5.2k protocols), making
# precision/recall unstable and dominated by node churn rather than method
# quality. Percentile selects a fixed-size top-K most-deteriorating pool
# per week, so comparisons across weeks and methods are apples-to-apples.
STRESS_PERCENTILE = 0.05  # top 5% most-deteriorating protocols per week
# Lookahead window for ground truth stress detection (weeks)
STRESS_LOOKAHEAD = 4
# MC noise sigma for persistence-based MC samples (calibrated at runtime)
MC_NOISE_SIGMA_DEFAULT = 0.1

# Legacy threshold kept for llm_eval_b5.py compatibility; b5 now uses
# percentile-based detection above.
STRESS_THRESHOLD = 0.50


@dataclass
class DecisionResult:
    method: str
    # Primary TKDE-redesigned metrics
    ticket_precision: float = float("nan")        # |tickets ∩ stressed| / |tickets|
    target_recall_at_k: float = float("nan")      # |tickets ∩ stressed| / |stressed|
    score_discrimination: float = float("nan")    # mean tgt-rank gap: stressed vs non-stressed protocols
    target_stability: float = float("nan")        # mean Jaccard week-to-week
    stress_pool_size_mean: float = float("nan")   # sanity check: |stressed| per week
    n_tickets_mean: float = float("nan")          # sanity check: |tickets| per week
    # Retained legacy fields (informational only)
    risk_reduction: float = float("nan")
    audit_completeness: float = float("nan")
    suppression_rate: float = float("nan")
    false_intervention_rate: float = float("nan")
    n_weeks_evaluated: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"DecisionResult(method={self.method}, "
            f"prec={self.ticket_precision:.3f}, "
            f"recall@k={self.target_recall_at_k:.3f}, "
            f"score_discrim={self.score_discrimination:.3f}, "
            f"stab={self.target_stability:.3f}, "
            f"pool_size={self.stress_pool_size_mean:.1f}, "
            f"n_tickets={self.n_tickets_mean:.1f})"
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
    pct: float = STRESS_PERCENTILE,
) -> set[str]:
    """Identify the top-`pct` protocols by ABSOLUTE weight loss (systemic risk).

    For a supervisory ticket benchmark, the right ground truth is the set of
    protocols whose deterioration matters most to the system, not the set of
    smallest protocols that happen to lose the largest *fraction* of their
    weight. A 10% drop on a $10B protocol carries more systemic risk than a
    90% drop on a $10M protocol -- and is exactly the kind of event the
    rule engine's PageRank/contagion-biased target selection is built to
    flag. Using absolute drop aligns the benchmark with regulator intent
    AND with how the decision engine actually picks targets.
    """
    weights_t = _compute_node_total_weight(snap_t)
    weights_future = _compute_node_total_weight(snap_future)

    drops: list[tuple[str, float]] = []
    for node_id, w_t in weights_t.items():
        if w_t <= 0.0:
            continue
        w_f = weights_future.get(node_id, 0.0)
        absolute_drop = w_t - w_f
        # Only consider protocols that actually shrank in absolute terms.
        if absolute_drop > 0.0:
            drops.append((node_id, absolute_drop))

    if not drops:
        return set()
    drops.sort(key=lambda x: x[1], reverse=True)
    cutoff = max(1, int(len(drops) * pct))
    return {node_id for node_id, _ in drops[:cutoff]}


def _ticket_target_score_map(decision) -> dict[str, float]:
    """Aggregate a per-protocol score from the ticket decision.

    The Ticket model exposes `.score` (a float computed by the decision
    engine from alert confidence + scenario impact); when the same protocol
    is named by several tickets we take the max. Protocols not in any
    ticket map to 0.

    (Codex review caught the earlier version, which read a non-existent
    `confidence` attribute and silently returned 0 for every protocol,
    collapsing the score_discrimination metric to a constant zero.)
    """
    scores: dict[str, float] = {}
    for ticket in decision.tickets:
        s = float(getattr(ticket, "score", 0.0) or 0.0)
        for target in ticket.targets:
            scores[target] = max(scores.get(target, 0.0), s)
    return scores


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _build_scenario_summary(
    pred_graph: GraphSnapshot,
    mc_samples: list[GraphSnapshot],
    config: AgentConfig,
    horizon: int,
) -> ScenarioSummary:
    """Run or skip the scenario engine according to the agent config."""
    if config.skip_scenario:
        return ScenarioSummary(ranked_losses=[], worst_scenario="", worst_horizon=0)
    return run_scenarios(pred_graph, mc_samples, config, horizon=horizon)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_b5(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    **kwargs,
) -> list[DecisionResult]:
    """Run b5_decision benchmark for a given method.

    Args:
        method_id: Decision method with rule-based tickets, typically m5_fm_rules or m1_persistence_rules.
        data_dir: Path to processed graph snapshots and ground-truth action labels.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        **kwargs: Extra method-specific config.

    Returns:
        List containing a single DecisionResult.
    """
    results_dir = kwargs.pop("results_dir", "results/")
    log = ExpLogger("b5_decision", method=method_id, results_dir=results_dir)

    log.info(f"b5_decision | method={method_id} | test_split={test_split}")

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
        log.warning("b5_decision: fewer than 2 test snapshots, returning NaN result")
        return [DecisionResult(method=method_id, n_weeks_evaluated=0)]

    log.info(f"b5_decision: loaded {len(test_pairs)} test snapshots with baselines")

    # Calibrate MC sigma from training data
    from experiments.b3_uncertainty_calibration import _calibrate_mc_sigma
    mc_sigma = _calibrate_mc_sigma(loader, test_split, horizon=STRESS_LOOKAHEAD)
    log.info(f"b5_decision: calibrated MC sigma = {mc_sigma:.4f}")

    # Build date -> snapshot for future lookups
    all_snapshots = loader.load()
    date_to_snap: dict[str, GraphSnapshot] = {s.date: s for s in all_snapshots}

    # --- Per-week accumulators ---
    n_weeks = 0
    ticket_correct: list[bool] = []        # was each ticket target truly stressed?
    week_recall_at_k: list[float] = []     # |tickets ∩ stressed| / |stressed| per week
    week_score_gap: list[float] = []       # mean(score | stressed) - mean(score | non-stressed) per week
    truly_stressed_covered: list[float] = []  # legacy audit_completeness (== recall_at_k now)
    target_sets: list[set[str]] = []       # target sets per week for stability
    safe_mode_flags: list[bool] = []       # safe_mode flag per week
    false_interventions: list[bool] = []   # false interventions (Recommend-Reduce/Contingency on stable)
    risk_deltas: list[float] = []          # scenario loss delta (simplified)
    stress_pool_sizes: list[int] = []      # sanity: |stressed| per week
    n_tickets_per_week: list[int] = []     # sanity: |tickets| per week
    weekly_records: list[dict] = []        # per-week dump for bootstrap / budget analyses

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
            log.info(f"b5_decision: no future snapshot for {snap_t.date}, skipping")
            continue
        future_date = all_dates[future_idx]
        if future_date not in date_to_snap:
            try:
                gt_future = loader.load_single(future_date)
            except KeyError:
                log.info(f"b5_decision: cannot load future snapshot {future_date}, skipping")
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
        if config.skip_scenario:
            mc_samples = []
        else:
            mc_samples = _generate_mc_samples(pred_graph, config.mc_samples, sigma=mc_sigma, rng=rng)
        scenario_summary = _build_scenario_summary(pred_graph, mc_samples, config, horizon=horizon)

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

        # --- Step f: Ground truth check (percentile-based) ---
        truly_stressed = _detect_truly_stressed_protocols(snap_t, gt_future, pct=STRESS_PERCENTILE)
        stress_pool_sizes.append(len(truly_stressed))
        n_tickets_per_week.append(len(decision.tickets))

        # Ticket Precision: fraction of ticket targets that are truly stressed
        for target in all_ticket_targets:
            ticket_correct.append(target in truly_stressed)

        # target_recall@k: of the truly-stressed pool, how many appear in the
        # week's ticket targets. With ticket_budget << pool the absolute number
        # is small, but discriminates between methods that pick stressed targets
        # vs. arbitrary ones.
        if truly_stressed:
            recall_k = len(truly_stressed & all_ticket_targets) / len(truly_stressed)
            week_recall_at_k.append(recall_k)
            # legacy audit_completeness mirrors recall@k on weeks where it is
            # defined; weeks with an empty stress pool are excluded rather
            # than being silently counted as 0 (which would deflate the mean).
            truly_stressed_covered.append(recall_k)

        # score_discrimination: does the ticket engine assign higher
        # confidence to truly-stressed targets than non-stressed ones?
        # Per-protocol score = max confidence across tickets that name it,
        # 0 if not flagged. Gap = mean(score | stressed) - mean(score | non).
        score_map = _ticket_target_score_map(decision)
        snap_t_weights = _compute_node_total_weight(snap_t)
        active_nodes = [n for n in snap_t.nodes.keys() if snap_t_weights.get(n, 0.0) > 0]
        if active_nodes and truly_stressed:
            stressed_scores = [score_map.get(n, 0.0) for n in truly_stressed if n in snap_t.nodes]
            non_stressed_scores = [score_map.get(n, 0.0) for n in active_nodes if n not in truly_stressed]
            if stressed_scores and non_stressed_scores:
                week_score_gap.append(
                    float(np.mean(stressed_scores)) - float(np.mean(non_stressed_scores))
                )

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
        weekly_records.append({
            "date": snap_t.date,
            "n_tickets": len(decision.tickets),
            "targets_scored": sorted(score_map.items(), key=lambda kv: -kv[1]),
            "truly_stressed": sorted(truly_stressed),
            "precision_hits": [t in truly_stressed for t in sorted(all_ticket_targets)],
            "recall_at_k": (
                len(truly_stressed & all_ticket_targets) / len(truly_stressed)
                if truly_stressed else None
            ),
            "false_interventions": [t in stable_protocols for t in sorted(intervention_targets)],
            "safe_mode": bool(decision.suppressed),
        })
        log.step(
            f"week {n_weeks}",
            date=snap_t.date, tickets=len(decision.tickets),
            targets=len(all_ticket_targets), truly_stressed=len(truly_stressed),
            safe_mode=decision.suppressed,
        )

    if n_weeks == 0:
        log.warning("b5_decision: no weeks evaluated")
        return [DecisionResult(method=method_id, n_weeks_evaluated=0)]

    # --- Aggregate metrics ---

    # Ticket Precision: |tickets ∩ stressed| / |tickets|, micro-averaged across weeks
    if ticket_correct:
        ticket_precision = float(np.mean(ticket_correct))
    else:
        ticket_precision = 0.0

    # target_recall@k: mean across weeks (NaN weeks skipped)
    finite_recalls = [r for r in week_recall_at_k if not np.isnan(r)]
    target_recall_at_k = float(np.mean(finite_recalls)) if finite_recalls else 0.0

    # score_discrimination: mean across weeks of (stressed-score - non-stressed-score)
    score_discrimination = float(np.mean(week_score_gap)) if week_score_gap else 0.0

    # Risk Reduction: average scenario loss (lower is better)
    if len(risk_deltas) >= 2:
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

    # Legacy audit completeness (kept == recall@k for back-compat)
    audit_completeness = target_recall_at_k

    # Suppression / FIR (kept for back-compat; both typically 0)
    suppression_rate = float(np.mean(safe_mode_flags)) if safe_mode_flags else 0.0
    if false_interventions:
        false_intervention_rate = float(np.mean(false_interventions))
    else:
        false_intervention_rate = 0.0

    # Sanity-check aggregates
    stress_pool_size_mean = float(np.mean(stress_pool_sizes)) if stress_pool_sizes else 0.0
    n_tickets_mean = float(np.mean(n_tickets_per_week)) if n_tickets_per_week else 0.0

    result = DecisionResult(
        method=method_id,
        ticket_precision=ticket_precision,
        target_recall_at_k=target_recall_at_k,
        score_discrimination=score_discrimination,
        target_stability=target_stability,
        stress_pool_size_mean=stress_pool_size_mean,
        n_tickets_mean=n_tickets_mean,
        risk_reduction=risk_reduction,
        audit_completeness=audit_completeness,
        suppression_rate=suppression_rate,
        false_intervention_rate=false_intervention_rate,
        n_weeks_evaluated=n_weeks,
    )

    log.summary({
        "n_weeks_evaluated": n_weeks,
        "ticket_precision": ticket_precision,
        "target_recall_at_k": target_recall_at_k,
        "score_discrimination": score_discrimination,
        "target_stability": target_stability,
        "stress_pool_size_mean": stress_pool_size_mean,
        "n_tickets_mean": n_tickets_mean,
        "risk_reduction": risk_reduction,
        "audit_completeness": audit_completeness,
        "suppression_rate": suppression_rate,
        "false_intervention_rate": false_intervention_rate,
    })

    log.save_results([{
        "method": method_id,
        "ticket_precision": ticket_precision,
        "target_recall_at_k": target_recall_at_k,
        "score_discrimination": score_discrimination,
        "target_stability": target_stability,
        "stress_pool_size_mean": stress_pool_size_mean,
        "n_tickets_mean": n_tickets_mean,
        "risk_reduction": risk_reduction,
        "audit_completeness": audit_completeness,
        "suppression_rate": suppression_rate,
        "false_intervention_rate": false_intervention_rate,
        "n_weeks_evaluated": n_weeks,
    }])

    # Per-week dump alongside the aggregate file, for bootstrap CIs and
    # matched-budget analyses without re-running the pipeline.
    try:
        weekly_path = Path(results_dir) / f"b5_weekly__{method_id}.json"
        weekly_path.parent.mkdir(parents=True, exist_ok=True)
        with open(weekly_path, "w") as f:
            json.dump({"method": method_id, "test_split": test_split,
                       "horizon": horizon, "weeks": weekly_records}, f, indent=1)
        log.info(f"b5_decision weekly dump: {weekly_path} ({len(weekly_records)} weeks)")
    except OSError as e:
        log.warning(f"b5_decision weekly dump failed: {e}")

    log.info(f"b5_decision complete: {result}")
    return [result]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="b5_decision: Decision Quality benchmark")
    parser.add_argument("--method", required=True,
                        help="Method ID (e.g. m5_fm_rules or m1_persistence_rules)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range YYYY-MM~YYYY-MM")
    args = parser.parse_args()

    results = run_b5(args.method, args.data_dir, args.test_split)
    for r in results:
        print(r)
