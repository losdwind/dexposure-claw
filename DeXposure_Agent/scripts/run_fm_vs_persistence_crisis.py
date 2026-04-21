#!/usr/bin/env python3
"""FM vs Persistence on crisis periods.

Core question: Does DeXposure-FM outperform persistence during crises?

Experiments:
1. B1-crisis: Risk metric forecasting (MAE, RankCorr, TrendCons) on crisis periods
   - C0 (FM) vs C2 (persistence) on Terra/Luna, FTX, SVB
2. B5-crisis-multihorizon: Decision quality with multi-horizon alert aggregation
   - C0 vs C2 on same crisis periods

Usage:
    DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 python3 scripts/run_fm_vs_persistence_crisis.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats as scipy_stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_AGENT_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _AGENT_DIR.parent
sys.path.insert(0, str(_AGENT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from dexposure_agent.config import AgentConfig
from dexposure_agent.data_health import compute_data_health
from dexposure_agent.data_loader import SnapshotLoader
from dexposure_agent.decision import generate_tickets
from dexposure_agent.monitor import compute_metrics, detect_alerts, _compute_rolling_baseline, _pagerank
from dexposure_agent.scenario import run_scenarios
from dexposure_agent.types import Edge, GraphSnapshot
from dexposure_agent.agent_loop import _aggregate_scenarios
from experiments.predict_helper import predict_graph

DATA_DIR = str(_REPO_ROOT / "DeXposure" / "data")
RESULTS_DIR = str(_AGENT_DIR / "results" / "run_fm_vs_persistence_crisis")
STRESS_LOOKAHEAD = 4
MC_NOISE_SIGMA = 2.0

CRISIS_SPLITS = {
    "terra_luna": ("2022-04~2022-07", "Terra/Luna collapse"),
    "ftx":        ("2022-10~2023-01", "FTX collapse"),
    "calm_2025":  ("2025-01~2025-08", "2025 calm baseline"),
}

METHODS = ["C0", "C2"]
HORIZONS = [1, 4, 8, 12]
METRIC_IDS = ["M1", "M3", "M4", "M6", "M7"]


# ──────────────────────────────────────────────────────────────────────────────
# B1-crisis: Risk metric forecasting
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class B1CrisisResult:
    method: str
    crisis: str
    horizon: int
    pagerank_mae: float = float("nan")
    hhi_mae: float = float("nan")
    density_mae: float = float("nan")
    gini_mae: float = float("nan")
    rank_correlation: float = float("nan")
    trend_consistency: float = float("nan")
    n_snapshots: int = 0
    # Per-week edge weight prediction error (FM-specific)
    mean_edge_weight_mae: float = float("nan")
    mean_new_edge_count: float = float("nan")


def _compute_edge_weight_mae(pred: GraphSnapshot, gt: GraphSnapshot) -> float:
    """Compute MAE of edge weights between predicted and ground truth graphs."""
    pred_edges = {(e.source, e.target): e.weight for e in pred.edges}
    gt_edges = {(e.source, e.target): e.weight for e in gt.edges}
    common = set(pred_edges.keys()) & set(gt_edges.keys())
    if not common:
        return float("nan")
    errors = [abs(pred_edges[k] - gt_edges[k]) for k in common]
    return float(np.mean(errors))


def _count_new_edges(pred: GraphSnapshot, current: GraphSnapshot) -> int:
    """Count edges in prediction that don't exist in current graph."""
    current_edges = {(e.source, e.target) for e in current.edges}
    pred_edges = {(e.source, e.target) for e in pred.edges}
    return len(pred_edges - current_edges)


def run_b1_crisis() -> list[B1CrisisResult]:
    """Run B1 risk forecasting on crisis and calm periods, comparing C0 vs C2."""
    loader = SnapshotLoader(data_dir=DATA_DIR)
    all_dates = loader.dates
    all_snaps = loader.load()
    date_to_snap = {s.date: s for s in all_snaps}

    results = []

    for crisis_name, (test_split, desc) in CRISIS_SPLITS.items():
        logger.info(f"B1-crisis: {crisis_name} ({desc}) split={test_split}")
        test_snaps = loader.load(date_range=test_split)
        logger.info(f"  Loaded {len(test_snaps)} test snapshots")

        for method in METHODS:
            for h in HORIZONS:
                pr_errors, hhi_errors, density_errors, gini_errors = [], [], [], []
                rank_corrs = []
                edge_maes, new_edge_counts = [], []
                prev_pred_metrics, prev_gt_metrics = None, None
                trend_correct, trend_total = 0, 0
                n_valid = 0

                for snap_t in test_snaps:
                    t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
                    if t_idx < 0:
                        continue
                    future_idx = t_idx + h
                    if future_idx >= len(all_dates):
                        continue
                    gt = date_to_snap.get(all_dates[future_idx])
                    if gt is None:
                        continue

                    pred = predict_graph(method, snap_t, horizon=h)

                    # Compute network-level metrics
                    pred_m = compute_metrics(pred)
                    gt_m = compute_metrics(gt)

                    pr_errors.append(abs(pred_m["M1"] - gt_m["M1"]))
                    hhi_errors.append(abs(pred_m["M3"] - gt_m["M3"]))
                    density_errors.append(abs(pred_m["M4"] - gt_m["M4"]))
                    gini_errors.append(abs(pred_m["M7"] - gt_m["M7"]))

                    # Edge-level accuracy
                    edge_maes.append(_compute_edge_weight_mae(pred, gt))
                    new_edge_counts.append(_count_new_edges(pred, snap_t))

                    # Rank correlation (node-level PageRank)
                    pred_nodes = list(pred.nodes.keys())
                    gt_nodes = list(gt.nodes.keys())
                    pred_adj = {n: {} for n in pred_nodes}
                    for e in pred.edges:
                        if e.source in pred_adj:
                            pred_adj[e.source][e.target] = pred_adj[e.source].get(e.target, 0) + e.weight
                    gt_adj = {n: {} for n in gt_nodes}
                    for e in gt.edges:
                        if e.source in gt_adj:
                            gt_adj[e.source][e.target] = gt_adj[e.source].get(e.target, 0) + e.weight
                    pred_pr = _pagerank(pred_nodes, pred_adj) if pred_nodes else {}
                    gt_pr = _pagerank(gt_nodes, gt_adj) if gt_nodes else {}

                    common_nodes = sorted(set(pred_pr.keys()) & set(gt_pr.keys()))
                    if len(common_nodes) >= 3:
                        rho, _ = scipy_stats.spearmanr(
                            [pred_pr[n] for n in common_nodes],
                            [gt_pr[n] for n in common_nodes]
                        )
                        if not np.isnan(rho):
                            rank_corrs.append(float(rho))

                    # Trend consistency
                    if prev_pred_metrics is not None and prev_gt_metrics is not None:
                        for mid in METRIC_IDS:
                            pd_delta = pred_m.get(mid, 0) - prev_pred_metrics.get(mid, 0)
                            gd_delta = gt_m.get(mid, 0) - prev_gt_metrics.get(mid, 0)
                            if (pd_delta >= 0) == (gd_delta >= 0):
                                trend_correct += 1
                            trend_total += 1

                    prev_pred_metrics = pred_m
                    prev_gt_metrics = gt_m
                    n_valid += 1

                valid_edge_maes = [x for x in edge_maes if not np.isnan(x)]

                r = B1CrisisResult(
                    method=method,
                    crisis=crisis_name,
                    horizon=h,
                    pagerank_mae=float(np.mean(pr_errors)) if pr_errors else float("nan"),
                    hhi_mae=float(np.mean(hhi_errors)) if hhi_errors else float("nan"),
                    density_mae=float(np.mean(density_errors)) if density_errors else float("nan"),
                    gini_mae=float(np.mean(gini_errors)) if gini_errors else float("nan"),
                    rank_correlation=float(np.mean(rank_corrs)) if rank_corrs else float("nan"),
                    trend_consistency=trend_correct / trend_total if trend_total > 0 else float("nan"),
                    n_snapshots=n_valid,
                    mean_edge_weight_mae=float(np.mean(valid_edge_maes)) if valid_edge_maes else float("nan"),
                    mean_new_edge_count=float(np.mean(new_edge_counts)) if new_edge_counts else float("nan"),
                )
                results.append(r)
                logger.info(
                    f"  {method} h={h}: RankCorr={r.rank_correlation:.4f} "
                    f"TrendCons={r.trend_consistency:.4f} HHI_MAE={r.hhi_mae:.4f} "
                    f"EdgeMAE={r.mean_edge_weight_mae:.4f} NewEdges={r.mean_new_edge_count:.1f} "
                    f"n={n_valid}"
                )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# B5-crisis: Multi-horizon decision quality
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class B5CrisisResult:
    method: str
    crisis: str
    horizons: list[int] = field(default_factory=list)
    n_weeks: int = 0
    n_alerts: int = 0
    alerts_per_horizon: dict[int, int] = field(default_factory=dict)
    ticket_precision: float = float("nan")
    false_intervention_rate: float = float("nan")
    target_stability: float = float("nan")
    audit_completeness: float = float("nan")
    n_tickets: int = 0
    n_intervention: int = 0


def _node_weights(g):
    t = {n: 0.0 for n in g.nodes}
    for e in g.edges:
        t[e.source] = t.get(e.source, 0) + e.weight
        t[e.target] = t.get(e.target, 0) + e.weight
    return t


def _stressed(snap_t, snap_f, thresh=0.20):
    wt, wf = _node_weights(snap_t), _node_weights(snap_f)
    return {n for n, w in wt.items() if w > 0 and (w - wf.get(n, 0)) / w > thresh}


def _jaccard(a, b):
    if not a and not b: return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _mc_samples(graph, n, sigma, rng):
    samples = []
    for _ in range(n):
        edges = []
        for e in graph.edges:
            noise = rng.normal(0, sigma * abs(e.weight)) if e.weight else 0
            edges.append(Edge(source=e.source, target=e.target, weight=max(0, e.weight + noise)))
        samples.append(GraphSnapshot(date=graph.date, nodes=graph.nodes, edges=edges))
    return samples


def run_b5_crisis() -> list[B5CrisisResult]:
    """Run multi-horizon B5 on crisis periods, C0 vs C2."""
    loader = SnapshotLoader(data_dir=DATA_DIR)
    all_dates = loader.dates
    all_snaps = loader.load()
    date_to_snap = {s.date: s for s in all_snaps}
    rng = np.random.default_rng(42)
    config = AgentConfig(horizons=HORIZONS)

    results = []

    for crisis_name, (test_split, desc) in CRISIS_SPLITS.items():
        logger.info(f"B5-crisis: {crisis_name} ({desc}) split={test_split}")
        test_pairs = list(loader.iter_test_with_baselines(test_split, baseline_window=config.rolling_window))
        logger.info(f"  Loaded {len(test_pairs)} test weeks")

        for method in METHODS:
            logger.info(f"  Running {method} with horizons={HORIZONS}")
            t0 = time.time()

            r = B5CrisisResult(method=method, crisis=crisis_name, horizons=list(HORIZONS))
            ticket_ok, false_intv, target_sets, audit_covs = [], [], [], []
            alerts_per_h = {h: 0 for h in HORIZONS}

            for snap_t, baseline_history in test_pairs:
                t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
                if t_idx < 0: continue
                fi = t_idx + STRESS_LOOKAHEAD
                if fi >= len(all_dates): continue
                gt = date_to_snap.get(all_dates[fi])
                if gt is None: continue

                dh = compute_data_health(snap_t, config)

                # Multi-horizon aggregation (like agent_loop.py)
                all_alerts = []
                all_scenario_losses = []

                for h in HORIZONS:
                    pred = predict_graph(method, snap_t, horizon=h)
                    metrics = compute_metrics(pred)
                    baseline = _compute_rolling_baseline(baseline_history, config.rolling_window)
                    alerts_h = detect_alerts(metrics, baseline, horizon=h, config=config)
                    all_alerts.extend(alerts_h)
                    alerts_per_h[h] += len(alerts_h)

                    mc = _mc_samples(pred, config.mc_samples, MC_NOISE_SIGMA, rng)
                    scen = run_scenarios(pred, mc, config, horizon=h)
                    all_scenario_losses.extend(scen.ranked_losses)

                scenario_summary = _aggregate_scenarios(all_scenario_losses)
                r.n_alerts += len(all_alerts)

                decision = generate_tickets(all_alerts, scenario_summary, dh, config)

                targets, intv_targets = set(), set()
                for t in decision.tickets:
                    targets.update(t.targets)
                    r.n_tickets += 1
                    if t.action in ("Recommend-Reduce", "Contingency"):
                        intv_targets.update(t.targets)
                        r.n_intervention += 1
                target_sets.append(targets)

                stressed = _stressed(snap_t, gt)
                for t in targets: ticket_ok.append(t in stressed)
                stable = set(snap_t.nodes.keys()) - stressed
                for t in intv_targets: false_intv.append(t in stable)
                if stressed:
                    audit_covs.append(len(stressed & targets) / len(stressed))
                else:
                    audit_covs.append(1.0)
                r.n_weeks += 1

            r.alerts_per_horizon = alerts_per_h
            r.ticket_precision = float(np.mean(ticket_ok)) if ticket_ok else 0.0
            r.false_intervention_rate = float(np.mean(false_intv)) if false_intv else 0.0
            r.audit_completeness = float(np.mean(audit_covs)) if audit_covs else 0.0

            stab = [_jaccard(target_sets[i], target_sets[i-1]) for i in range(1, len(target_sets))]
            r.target_stability = float(np.mean(stab)) if stab else 0.0

            elapsed = time.time() - t0
            logger.info(
                f"  {method}: alerts={r.n_alerts} per_h={alerts_per_h} "
                f"prec={r.ticket_precision:.3f} FIR={r.false_intervention_rate:.3f} "
                f"audit={r.audit_completeness:.4f} stab={r.target_stability:.3f} ({elapsed:.0f}s)"
            )
            results.append(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 65)
    logger.info("FM vs PERSISTENCE ON CRISIS PERIODS")
    logger.info("=" * 65)

    t_start = time.time()

    # B1-crisis
    logger.info("\n" + "=" * 65)
    logger.info("PART 1: B1 Risk Forecasting — C0 vs C2 on crisis periods")
    logger.info("=" * 65)
    b1_results = run_b1_crisis()

    # B5-crisis
    logger.info("\n" + "=" * 65)
    logger.info("PART 2: B5 Decision Quality (multi-horizon) — C0 vs C2")
    logger.info("=" * 65)
    b5_results = run_b5_crisis()

    # Save
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    b1_data = [asdict(r) for r in b1_results]
    (out_dir / "b1_crisis.json").write_text(json.dumps(b1_data, indent=2, default=str))

    b5_data = [asdict(r) for r in b5_results]
    (out_dir / "b5_crisis.json").write_text(json.dumps(b5_data, indent=2, default=str))

    total = time.time() - t_start
    logger.info(f"\nALL DONE in {total:.0f}s ({total/60:.1f} min)")

    # Print B1 summary
    print("\n=== B1: FM vs Persistence on Crisis Periods ===")
    print(f"{'Crisis':<12} {'Method':<6} {'h':>3} {'RankCorr':>9} {'TrendCons':>10} {'HHI_MAE':>8} {'EdgeMAE':>8} {'NewEdge':>8} {'n':>3}")
    print("-" * 75)
    for r in b1_results:
        print(f"{r.crisis:<12} {r.method:<6} {r.horizon:>3} {r.rank_correlation:>9.4f} "
              f"{r.trend_consistency:>10.4f} {r.hhi_mae:>8.4f} "
              f"{r.mean_edge_weight_mae:>8.4f} {r.mean_new_edge_count:>8.1f} {r.n_snapshots:>3}")

    # Print B5 summary
    print("\n=== B5: Decision Quality (multi-horizon) on Crisis ===")
    print(f"{'Crisis':<12} {'Method':<6} {'Alerts':>7} {'Prec':>6} {'FIR':>6} {'Audit':>7} {'Stab':>6} {'#Intv':>6}")
    print("-" * 60)
    for r in b5_results:
        print(f"{r.crisis:<12} {r.method:<6} {r.n_alerts:>7} {r.ticket_precision:>6.3f} "
              f"{r.false_intervention_rate:>6.3f} {r.audit_completeness:>7.4f} "
              f"{r.target_stability:>6.3f} {r.n_intervention:>6}")


if __name__ == "__main__":
    main()
