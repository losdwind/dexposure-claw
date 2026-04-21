#!/usr/bin/env python3
"""Supplementary ablation experiments for the paper revision.

Addresses two reviewer concerns:
1. A1 (data-health gate) shows no effect under clean test conditions
   → Re-run under B6-style degraded conditions where safe_mode activates
2. A6 (multi-horizon) shows no effect because B5 only uses h=4
   → Create multi-horizon B5 matching agent_loop.py, test on crisis periods

Also re-runs B6_C0 as fresh baseline on this server.

Usage:
    DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 python3 scripts/run_supplementary_ablations.py
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

# Setup paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_AGENT_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _AGENT_DIR.parent
sys.path.insert(0, str(_AGENT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from dexposure_agent.config import AgentConfig
from dexposure_agent.data_health import compute_data_health
from dexposure_agent.data_loader import SnapshotLoader
from dexposure_agent.decision import generate_tickets
from dexposure_agent.monitor import compute_metrics, detect_alerts, _compute_rolling_baseline
from dexposure_agent.scenario import run_scenarios
from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures

DATA_DIR = str(_REPO_ROOT / "DeXposure" / "data")
RESULTS_DIR = str(_AGENT_DIR / "results" / "run_supplementary_ablations")
STRESS_LOOKAHEAD = 4

# ──────────────────────────────────────────────────────────────────────────────
# Degradation helpers (from B6, extended)
# ──────────────────────────────────────────────────────────────────────────────

def _mask_edges(graph: GraphSnapshot, mask_fraction: float, rng: np.random.Generator) -> GraphSnapshot:
    if mask_fraction <= 0.0 or not graph.edges:
        return graph
    n_keep = max(1, int(len(graph.edges) * (1.0 - mask_fraction)))
    indices = rng.choice(len(graph.edges), size=n_keep, replace=False)
    kept_edges = [graph.edges[i] for i in sorted(indices)]
    return GraphSnapshot(date=graph.date, nodes=graph.nodes, edges=kept_edges)


def _drop_features(graph: GraphSnapshot, drop_fraction: float, rng: np.random.Generator) -> GraphSnapshot:
    if drop_fraction <= 0.0:
        return graph
    numeric_attrs = ["log_size", "num_tokens", "max_share", "entropy"]
    new_nodes: dict[str, NodeFeatures] = {}
    for node_id, features in graph.nodes.items():
        vals = {
            "log_size": features.log_size,
            "num_tokens": features.num_tokens,
            "max_share": features.max_share,
            "entropy": features.entropy,
            "category": features.category,
        }
        for attr in numeric_attrs:
            if rng.random() < drop_fraction:
                vals[attr] = 0 if attr == "num_tokens" else 0.0
        new_nodes[node_id] = NodeFeatures(**vals)
    return GraphSnapshot(date=graph.date, nodes=new_nodes, edges=graph.edges)


def apply_degradation(graph: GraphSnapshot, regime: str, rng: np.random.Generator) -> GraphSnapshot:
    """Apply a named degradation regime to a snapshot."""
    regimes = {
        "moderate":       {"feature_drop": 0.50, "edge_mask": 0.50},
        "severe":         {"feature_drop": 0.80, "edge_mask": 0.85},
        "feature_only":   {"feature_drop": 0.80, "edge_mask": 0.00},
        "topology_only":  {"feature_drop": 0.00, "edge_mask": 0.85},
    }
    cfg = regimes.get(regime, {})
    result = graph
    if cfg.get("feature_drop", 0) > 0:
        result = _drop_features(result, cfg["feature_drop"], rng)
    if cfg.get("edge_mask", 0) > 0:
        result = _mask_edges(result, cfg["edge_mask"], rng)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# MC sample helper (from B5)
# ──────────────────────────────────────────────────────────────────────────────

MC_NOISE_SIGMA = 0.1

def _generate_mc_samples(
    graph: GraphSnapshot, n_samples: int, sigma: float = MC_NOISE_SIGMA,
    rng: np.random.Generator | None = None,
) -> list[GraphSnapshot]:
    if rng is None:
        rng = np.random.default_rng()
    samples = []
    for _ in range(n_samples):
        new_edges = []
        for edge in graph.edges:
            noise = rng.normal(0.0, sigma * abs(edge.weight)) if edge.weight != 0.0 else 0.0
            new_edges.append(Edge(source=edge.source, target=edge.target, weight=max(0.0, edge.weight + noise)))
        samples.append(GraphSnapshot(date=graph.date, nodes=graph.nodes, edges=new_edges))
    return samples


def _compute_node_total_weight(graph: GraphSnapshot) -> dict[str, float]:
    totals: dict[str, float] = {n: 0.0 for n in graph.nodes}
    for e in graph.edges:
        totals[e.source] = totals.get(e.source, 0.0) + e.weight
        totals[e.target] = totals.get(e.target, 0.0) + e.weight
    return totals


def _detect_truly_stressed(snap_t: GraphSnapshot, snap_future: GraphSnapshot, threshold: float = 0.20) -> set[str]:
    wt = _compute_node_total_weight(snap_t)
    wf = _compute_node_total_weight(snap_future)
    return {n for n, w in wt.items() if w > 0 and (w - wf.get(n, 0.0)) / w > threshold}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: A1-degraded — data-health gate under degraded conditions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class A1DegradedResult:
    regime: str
    tau_data: float
    n_weeks: int = 0
    mean_dh_score: float = float("nan")
    safe_mode_rate: float = float("nan")
    ticket_precision: float = float("nan")
    false_intervention_rate: float = float("nan")
    target_stability: float = float("nan")
    n_tickets_total: int = 0
    n_intervention_tickets: int = 0


def run_a1_degraded(
    test_split: str = "2025-01~2025-08",
    degradation_regimes: list[str] | None = None,
    tau_data_values: list[float] | None = None,
) -> list[A1DegradedResult]:
    """Run A1 ablation under degraded conditions.

    For each (regime, tau_data) pair:
    1. Load test snapshots
    2. Apply degradation to each input snapshot
    3. Run B5-style pipeline: data_health → predict → monitor → scenario → decide
    4. Measure: DH_t scores, safe_mode rate, ticket precision, FIR
    """
    if degradation_regimes is None:
        degradation_regimes = ["moderate", "severe", "feature_only", "topology_only"]
    if tau_data_values is None:
        tau_data_values = [0.7, 0.0]  # full system vs A1

    from experiments.predict_helper import predict_graph
    from experiments.b3_uncertainty_calibration import _calibrate_mc_sigma

    loader = SnapshotLoader(data_dir=DATA_DIR)
    all_dates = loader.dates
    all_snapshots = loader.load()
    date_to_snap = {s.date: s for s in all_snapshots}

    # Calibrate MC sigma
    mc_sigma = _calibrate_mc_sigma(loader, test_split, horizon=STRESS_LOOKAHEAD)
    logger.info(f"A1-degraded: MC sigma = {mc_sigma:.4f}")

    results: list[A1DegradedResult] = []
    rng = np.random.default_rng(seed=42)

    for regime in degradation_regimes:
        for tau_data in tau_data_values:
            logger.info(f"A1-degraded: regime={regime} tau_data={tau_data}")
            t0 = time.time()

            config = AgentConfig(tau_data=tau_data)
            result = A1DegradedResult(regime=regime, tau_data=tau_data)

            dh_scores: list[float] = []
            safe_modes: list[bool] = []
            ticket_correct: list[bool] = []
            false_interventions: list[bool] = []
            target_sets: list[set[str]] = []
            n_tickets = 0
            n_intervention = 0

            # Load test pairs
            test_pairs = list(loader.iter_test_with_baselines(test_split, baseline_window=config.rolling_window))

            for snap_t, baseline_history in test_pairs:
                # Find future snapshot for ground truth
                t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
                if t_idx < 0:
                    continue
                future_idx = t_idx + STRESS_LOOKAHEAD
                if future_idx >= len(all_dates):
                    continue
                future_date = all_dates[future_idx]
                gt_future = date_to_snap.get(future_date)
                if gt_future is None:
                    continue

                # Apply degradation to input
                degraded_snap = apply_degradation(snap_t, regime, rng)

                # Data health on degraded input
                dh = compute_data_health(degraded_snap, config)
                dh_scores.append(dh.score)
                safe_modes.append(dh.safe_mode)

                # Predict on degraded input
                pred_graph = predict_graph("C0", degraded_snap, horizon=STRESS_LOOKAHEAD)
                current_metrics = compute_metrics(pred_graph)
                rolling_baseline = _compute_rolling_baseline(baseline_history, config.rolling_window)
                alerts = detect_alerts(current_metrics, rolling_baseline, horizon=STRESS_LOOKAHEAD, config=config)

                # Scenario engine
                mc_samples = _generate_mc_samples(pred_graph, config.mc_samples, sigma=mc_sigma, rng=rng)
                scenario_summary = run_scenarios(pred_graph, mc_samples, config, horizon=STRESS_LOOKAHEAD)

                # Decision
                decision = generate_tickets(alerts, scenario_summary, dh, config)

                # Collect targets
                all_targets: set[str] = set()
                intervention_targets: set[str] = set()
                for ticket in decision.tickets:
                    all_targets.update(ticket.targets)
                    n_tickets += 1
                    if ticket.action in ("Recommend-Reduce", "Contingency"):
                        intervention_targets.update(ticket.targets)
                        n_intervention += 1
                target_sets.append(all_targets)

                # Ground truth
                truly_stressed = _detect_truly_stressed(snap_t, gt_future)
                for t in all_targets:
                    ticket_correct.append(t in truly_stressed)
                stable = set(snap_t.nodes.keys()) - truly_stressed
                for t in intervention_targets:
                    false_interventions.append(t in stable)

                result.n_weeks += 1

            # Aggregate
            result.mean_dh_score = float(np.mean(dh_scores)) if dh_scores else float("nan")
            result.safe_mode_rate = float(np.mean(safe_modes)) if safe_modes else 0.0
            result.ticket_precision = float(np.mean(ticket_correct)) if ticket_correct else 0.0
            result.false_intervention_rate = float(np.mean(false_interventions)) if false_interventions else 0.0
            result.n_tickets_total = n_tickets
            result.n_intervention_tickets = n_intervention

            # Stability
            stab_scores = []
            for i in range(1, len(target_sets)):
                stab_scores.append(_jaccard(target_sets[i], target_sets[i - 1]))
            result.target_stability = float(np.mean(stab_scores)) if stab_scores else 0.0

            elapsed = time.time() - t0
            logger.info(
                f"A1-degraded: regime={regime} tau_data={tau_data} done in {elapsed:.1f}s | "
                f"DH={result.mean_dh_score:.3f} safe_mode_rate={result.safe_mode_rate:.2f} "
                f"prec={result.ticket_precision:.3f} FIR={result.false_intervention_rate:.3f} "
                f"n_intervention={n_intervention}"
            )
            results.append(result)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: A6-crisis — multi-horizon on crisis periods
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class A6CrisisResult:
    crisis_period: str
    horizons: list[int] = field(default_factory=list)
    n_weeks: int = 0
    n_alerts_total: int = 0
    n_alerts_per_horizon: dict[int, int] = field(default_factory=dict)
    ticket_precision: float = float("nan")
    false_intervention_rate: float = float("nan")
    target_stability: float = float("nan")
    audit_completeness: float = float("nan")
    n_tickets_total: int = 0


# Crisis test splits: period name -> (test_split, description)
CRISIS_SPLITS = {
    "terra_luna":  ("2022-04~2022-07", "Terra/Luna collapse period"),
    "ftx":         ("2022-10~2023-01", "FTX collapse period"),
    "svb":         ("2023-02~2023-05", "SVB/USDC de-peg period"),
}


def run_a6_crisis(
    crisis_periods: list[str] | None = None,
    horizon_configs: list[list[int]] | None = None,
) -> list[A6CrisisResult]:
    """Run A6 ablation: multi-horizon vs single-horizon on crisis periods.

    Key difference from standard B5: this mimics agent_loop.py by iterating
    over ALL horizons and aggregating alerts before generating tickets.
    """
    if crisis_periods is None:
        crisis_periods = list(CRISIS_SPLITS.keys())
    if horizon_configs is None:
        horizon_configs = [[1, 4, 8, 12], [4]]  # full vs A6

    from experiments.predict_helper import predict_graph

    loader = SnapshotLoader(data_dir=DATA_DIR)
    all_dates = loader.dates
    all_snapshots = loader.load()
    date_to_snap = {s.date: s for s in all_snapshots}

    results: list[A6CrisisResult] = []
    rng = np.random.default_rng(seed=42)

    for crisis_name in crisis_periods:
        test_split, desc = CRISIS_SPLITS[crisis_name]
        logger.info(f"A6-crisis: {crisis_name} ({desc}) test_split={test_split}")

        for horizons in horizon_configs:
            logger.info(f"A6-crisis: {crisis_name} horizons={horizons}")
            t0 = time.time()

            config = AgentConfig(horizons=horizons)
            result = A6CrisisResult(crisis_period=crisis_name, horizons=list(horizons))

            ticket_correct: list[bool] = []
            false_interventions: list[bool] = []
            target_sets: list[set[str]] = []
            audit_coverages: list[float] = []
            alerts_per_h: dict[int, int] = {h: 0 for h in horizons}

            # Load test pairs with baselines
            test_pairs = list(loader.iter_test_with_baselines(test_split, baseline_window=config.rolling_window))
            logger.info(f"A6-crisis: loaded {len(test_pairs)} test weeks for {crisis_name}")

            for snap_t, baseline_history in test_pairs:
                t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
                if t_idx < 0:
                    continue
                # Use h=4 for ground truth lookahead
                future_idx = t_idx + STRESS_LOOKAHEAD
                if future_idx >= len(all_dates):
                    continue
                future_date = all_dates[future_idx]
                gt_future = date_to_snap.get(future_date)
                if gt_future is None:
                    continue

                # Data health (clean input for A6 test)
                dh = compute_data_health(snap_t, config)

                # Multi-horizon: iterate like agent_loop.py
                all_alerts = []
                all_scenario_losses = []

                for h in horizons:
                    pred_graph = predict_graph("C0", snap_t, horizon=h)
                    current_metrics = compute_metrics(pred_graph)
                    rolling_baseline = _compute_rolling_baseline(baseline_history, config.rolling_window)
                    alerts_h = detect_alerts(current_metrics, rolling_baseline, horizon=h, config=config)

                    all_alerts.extend(alerts_h)
                    alerts_per_h[h] += len(alerts_h)

                    mc_samples = _generate_mc_samples(pred_graph, config.mc_samples, sigma=MC_NOISE_SIGMA, rng=rng)
                    scenario_result = run_scenarios(pred_graph, mc_samples, config, horizon=h)
                    all_scenario_losses.extend(scenario_result.ranked_losses)

                # Aggregate scenarios across horizons (like agent_loop._aggregate_scenarios)
                from dexposure_agent.agent_loop import _aggregate_scenarios
                scenario_summary = _aggregate_scenarios(all_scenario_losses)

                result.n_alerts_total += len(all_alerts)

                # Generate tickets using aggregated alerts
                decision = generate_tickets(all_alerts, scenario_summary, dh, config)

                all_targets: set[str] = set()
                intervention_targets: set[str] = set()
                for ticket in decision.tickets:
                    all_targets.update(ticket.targets)
                    result.n_tickets_total += 1
                    if ticket.action in ("Recommend-Reduce", "Contingency"):
                        intervention_targets.update(ticket.targets)
                target_sets.append(all_targets)

                # Ground truth
                truly_stressed = _detect_truly_stressed(snap_t, gt_future)
                for t in all_targets:
                    ticket_correct.append(t in truly_stressed)
                stable = set(snap_t.nodes.keys()) - truly_stressed
                for t in intervention_targets:
                    false_interventions.append(t in stable)
                if truly_stressed:
                    audit_coverages.append(len(truly_stressed & all_targets) / len(truly_stressed))
                else:
                    audit_coverages.append(1.0)

                result.n_weeks += 1

            # Aggregate
            result.n_alerts_per_horizon = alerts_per_h
            result.ticket_precision = float(np.mean(ticket_correct)) if ticket_correct else 0.0
            result.false_intervention_rate = float(np.mean(false_interventions)) if false_interventions else 0.0
            result.audit_completeness = float(np.mean(audit_coverages)) if audit_coverages else 0.0

            stab_scores = []
            for i in range(1, len(target_sets)):
                stab_scores.append(_jaccard(target_sets[i], target_sets[i - 1]))
            result.target_stability = float(np.mean(stab_scores)) if stab_scores else 0.0

            elapsed = time.time() - t0
            logger.info(
                f"A6-crisis: {crisis_name} h={horizons} done in {elapsed:.1f}s | "
                f"weeks={result.n_weeks} alerts={result.n_alerts_total} "
                f"per_h={alerts_per_h} prec={result.ticket_precision:.3f} "
                f"FIR={result.false_intervention_rate:.3f} audit={result.audit_completeness:.4f}"
            )
            results.append(result)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: B6_C0 fresh baseline
# ──────────────────────────────────────────────────────────────────────────────

def run_b6_fresh() -> list:
    """Run fresh B6 for C0 on this server."""
    from experiments.b6_robustness import run_b6
    logger.info("Running fresh B6 for C0...")
    return run_b6("C0", data_dir=DATA_DIR, results_dir=RESULTS_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Saving results
# ──────────────────────────────────────────────────────────────────────────────

def save_results(a1_results: list, a6_results: list, b6_results: list):
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A1-degraded
    a1_data = []
    for r in a1_results:
        a1_data.append(asdict(r))
    (out_dir / "A1_degraded.json").write_text(json.dumps(a1_data, indent=2, default=str))

    # A6-crisis
    a6_data = []
    for r in a6_results:
        a6_data.append(asdict(r))
    (out_dir / "A6_crisis.json").write_text(json.dumps(a6_data, indent=2, default=str))

    # B6-fresh
    if b6_results:
        b6_data = [
            {
                "method": r.method, "regime": r.regime, "horizon": r.horizon,
                "pagerank_mae": r.pagerank_mae, "hhi_mae": r.hhi_mae,
                "rank_correlation": r.rank_correlation,
                "relative_degradation": r.relative_degradation,
                "n_test_snapshots": r.n_test_snapshots,
            }
            for r in b6_results
        ]
        (out_dir / "B6_C0_fresh.json").write_text(json.dumps(b6_data, indent=2, default=str))

    # Combined summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "server": "175.155.64.231:19113",
        "a1_degraded": a1_data,
        "a6_crisis": a6_data,
        "n_b6_regimes": len(b6_results) if b6_results else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"All results saved to {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Supplementary ablation experiments")
    parser.add_argument("--skip-a1", action="store_true", help="Skip A1-degraded experiments")
    parser.add_argument("--skip-a6", action="store_true", help="Skip A6-crisis experiments")
    parser.add_argument("--skip-b6", action="store_true", help="Skip B6 fresh baseline")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SUPPLEMENTARY ABLATION EXPERIMENTS")
    logger.info("=" * 70)

    t_start = time.time()
    a1_results: list = []
    a6_results: list = []
    b6_results: list = []

    # ── Experiment 1: A1 under degradation ──
    if not args.skip_a1:
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT 1: A1 (data-health gate) under degraded conditions")
        logger.info("=" * 70)
        a1_results = run_a1_degraded()
    else:
        logger.info("Skipping A1-degraded")

    # ── Experiment 2: A6 on crisis periods ──
    if not args.skip_a6:
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT 2: A6 (multi-horizon) on crisis periods")
        logger.info("=" * 70)
        a6_results = run_a6_crisis()
    else:
        logger.info("Skipping A6-crisis")

    # ── Experiment 3: B6 fresh baseline ──
    if not args.skip_b6:
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT 3: B6_C0 fresh baseline")
        logger.info("=" * 70)
        b6_results = run_b6_fresh()
    else:
        logger.info("Skipping B6-fresh")

    # ── Save ──
    save_results(a1_results, a6_results, b6_results)

    total = time.time() - t_start
    logger.info(f"\nALL DONE in {total:.0f}s ({total / 60:.1f} min)")
    logger.info(f"Results: {RESULTS_DIR}/")

    # ── Print summary ──
    if a1_results:
        print("\n=== A1 DEGRADED RESULTS ===")
        print(f"{'Regime':<18} {'tau_data':>8} {'DH_t':>6} {'safe%':>6} {'Prec':>6} {'FIR':>6} {'#intv':>6}")
        print("-" * 60)
        for r in a1_results:
            print(f"{r.regime:<18} {r.tau_data:>8.1f} {r.mean_dh_score:>6.3f} "
                  f"{r.safe_mode_rate:>6.2f} {r.ticket_precision:>6.3f} "
                  f"{r.false_intervention_rate:>6.3f} {r.n_intervention_tickets:>6}")

    if a6_results:
        print("\n=== A6 CRISIS RESULTS ===")
        print(f"{'Crisis':<14} {'Horizons':<16} {'Weeks':>5} {'Alerts':>7} {'Prec':>6} {'FIR':>6} {'Audit':>7}")
        print("-" * 65)
        for r in a6_results:
            h_str = str(r.horizons)
            print(f"{r.crisis_period:<14} {h_str:<16} {r.n_weeks:>5} {r.n_alerts_total:>7} "
                  f"{r.ticket_precision:>6.3f} {r.false_intervention_rate:>6.3f} "
                  f"{r.audit_completeness:>7.4f}")
