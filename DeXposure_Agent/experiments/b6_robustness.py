#!/usr/bin/env python3
"""B6: Robustness Under Distribution Shift (Section 3.7 of EXPERIMENT_PLAN)

Evaluates model performance under five data-degradation regimes to measure
robustness to missing / noisy inputs representative of real-world data issues.

Regimes:
- low_data_10pct      -- train on only 10% of available training data
- low_data_25pct      -- train on only 25% of available training data
- partial_graph_30    -- 30% of edges randomly masked at inference time
- noisy_features_01   -- Gaussian noise N(0, 0.1) added to all node features
- missing_features_20 -- 20% of node feature values randomly set to zero

For each regime, we report the same core metrics as B1 (h=4 fixed) plus a
relative degradation score compared to the clean baseline.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp_logger import ExpLogger
from dexposure_agent.config import AgentConfig
from dexposure_agent.data_loader import SnapshotLoader
from dexposure_agent.monitor import compute_metrics, _pagerank
from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures


REGIMES = [
    "low_data_10pct",
    "low_data_25pct",
    "partial_graph_30",
    "noisy_features_01",
    "missing_features_20",
]

REGIME_CONFIGS = {
    "low_data_10pct":      {"train_fraction": 0.10, "edge_mask": 0.0,  "noise_sigma": 0.0, "feature_drop": 0.0},
    "low_data_25pct":      {"train_fraction": 0.25, "edge_mask": 0.0,  "noise_sigma": 0.0, "feature_drop": 0.0},
    "partial_graph_30":    {"train_fraction": 1.0,  "edge_mask": 0.30, "noise_sigma": 0.0, "feature_drop": 0.0},
    "noisy_features_01":   {"train_fraction": 1.0,  "edge_mask": 0.0,  "noise_sigma": 0.1, "feature_drop": 0.0},
    "missing_features_20": {"train_fraction": 1.0,  "edge_mask": 0.0,  "noise_sigma": 0.0, "feature_drop": 0.20},
}

FIXED_HORIZON = 4  # weeks; same as B1 but single horizon for tractability

# Metric IDs used for B1-style evaluation
METRIC_IDS = ["M1", "M3", "M4", "M6", "M7"]


@dataclass
class B6Result:
    method: str
    regime: str
    horizon: int = FIXED_HORIZON
    pagerank_mae: float = float("nan")
    hhi_mae: float = float("nan")
    density_mae: float = float("nan")
    gini_mae: float = float("nan")
    rank_correlation: float = float("nan")
    trend_consistency: float = float("nan")
    relative_degradation: float = float("nan")  # vs clean baseline, higher = worse
    n_test_snapshots: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"B6Result(method={self.method}, regime={self.regime}, h={self.horizon}, "
            f"pr_mae={self.pagerank_mae:.4f}, hhi_mae={self.hhi_mae:.4f}, "
            f"rel_degrad={self.relative_degradation:+.3f})"
        )


# ---------------------------------------------------------------------------
# Data degradation helpers
# ---------------------------------------------------------------------------


def _mask_edges(graph: GraphSnapshot, mask_fraction: float, rng: np.random.Generator) -> GraphSnapshot:
    """Randomly remove a fraction of edges from the graph."""
    if mask_fraction <= 0.0 or not graph.edges:
        return graph
    n_edges = len(graph.edges)
    n_keep = max(1, int(n_edges * (1.0 - mask_fraction)))
    indices = rng.choice(n_edges, size=n_keep, replace=False)
    kept_edges = [graph.edges[i] for i in sorted(indices)]
    return GraphSnapshot(date=graph.date, nodes=graph.nodes, edges=kept_edges)


def _add_feature_noise(graph: GraphSnapshot, noise_sigma: float, rng: np.random.Generator) -> GraphSnapshot:
    """Add Gaussian noise N(0, noise_sigma) to numeric node features."""
    if noise_sigma <= 0.0:
        return graph
    new_nodes: dict[str, NodeFeatures] = {}
    for node_id, features in graph.nodes.items():
        new_nodes[node_id] = NodeFeatures(
            log_size=max(0.0, features.log_size + rng.normal(0.0, noise_sigma)),
            num_tokens=max(0, int(features.num_tokens + rng.normal(0.0, noise_sigma * 10))),
            max_share=max(0.0, min(1.0, features.max_share + rng.normal(0.0, noise_sigma))),
            entropy=max(0.0, features.entropy + rng.normal(0.0, noise_sigma)),
            category=features.category,
        )
    return GraphSnapshot(date=graph.date, nodes=new_nodes, edges=graph.edges)


def _drop_features(graph: GraphSnapshot, drop_fraction: float, rng: np.random.Generator) -> GraphSnapshot:
    """Randomly zero out a fraction of numeric node features."""
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


def _apply_degradation(
    graph: GraphSnapshot,
    regime_cfg: dict,
    rng: np.random.Generator,
) -> GraphSnapshot:
    """Apply a data degradation regime to a snapshot.

    Note: low_data regimes are handled at the snapshot-list level (subsampling),
    not per-snapshot. This function handles per-snapshot degradations.
    """
    result = graph
    edge_mask = regime_cfg.get("edge_mask", 0.0)
    noise_sigma = regime_cfg.get("noise_sigma", 0.0)
    feature_drop = regime_cfg.get("feature_drop", 0.0)

    if edge_mask > 0.0:
        result = _mask_edges(result, edge_mask, rng)
    if noise_sigma > 0.0:
        result = _add_feature_noise(result, noise_sigma, rng)
    if feature_drop > 0.0:
        result = _drop_features(result, feature_drop, rng)

    return result


def _subsample_snapshots(
    snapshots: list[GraphSnapshot],
    fraction: float,
    rng: np.random.Generator,
) -> list[GraphSnapshot]:
    """Subsample snapshots to simulate low-data regime.

    Always keeps at least 1 snapshot. Maintains chronological order.
    """
    if fraction >= 1.0 or len(snapshots) <= 1:
        return snapshots
    n_keep = max(1, int(len(snapshots) * fraction))
    indices = sorted(rng.choice(len(snapshots), size=n_keep, replace=False))
    return [snapshots[i] for i in indices]


# ---------------------------------------------------------------------------
# B1-style metric evaluation
# ---------------------------------------------------------------------------


def _compute_b1_metrics(
    pred_snapshots: list[GraphSnapshot],
    gt_snapshots: list[GraphSnapshot],
    prev_pred_metrics: list[dict[str, float]] | None = None,
    prev_gt_metrics: list[dict[str, float]] | None = None,
) -> dict[str, float]:
    """Compute B1-style metrics (MAE, Rank Correlation, Trend Consistency).

    Args:
        pred_snapshots: Predicted snapshots (persistence, possibly degraded).
        gt_snapshots: Ground truth snapshots.
        prev_pred_metrics: Previous-step predicted metrics (for trend consistency).
        prev_gt_metrics: Previous-step ground truth metrics (for trend consistency).

    Returns:
        Dict with pagerank_mae, hhi_mae, density_mae, gini_mae,
        rank_correlation, trend_consistency.
    """
    if not pred_snapshots or not gt_snapshots:
        return {
            "pagerank_mae": float("nan"), "hhi_mae": float("nan"),
            "density_mae": float("nan"), "gini_mae": float("nan"),
            "rank_correlation": float("nan"), "trend_consistency": float("nan"),
        }

    n = min(len(pred_snapshots), len(gt_snapshots))

    pr_errors: list[float] = []
    hhi_errors: list[float] = []
    density_errors: list[float] = []
    gini_errors: list[float] = []

    # For rank correlation: collect per-node PageRank vectors
    pred_pr_vectors: list[dict[str, float]] = []
    gt_pr_vectors: list[dict[str, float]] = []

    # For trend consistency
    trend_correct = 0
    trend_total = 0

    for i in range(n):
        pred_metrics = compute_metrics(pred_snapshots[i])
        gt_metrics = compute_metrics(gt_snapshots[i])

        # MAE per metric
        pr_errors.append(abs(pred_metrics["M1"] - gt_metrics["M1"]))
        hhi_errors.append(abs(pred_metrics["M3"] - gt_metrics["M3"]))
        density_errors.append(abs(pred_metrics["M4"] - gt_metrics["M4"]))
        gini_errors.append(abs(pred_metrics["M7"] - gt_metrics["M7"]))

        # Collect node-level PageRank for rank correlation
        pred_nodes = list(pred_snapshots[i].nodes.keys())
        gt_nodes = list(gt_snapshots[i].nodes.keys())

        # Build adjacency for PageRank
        pred_adj: dict[str, dict[str, float]] = {n_id: {} for n_id in pred_nodes}
        for edge in pred_snapshots[i].edges:
            if edge.source in pred_adj:
                pred_adj[edge.source][edge.target] = (
                    pred_adj[edge.source].get(edge.target, 0.0) + edge.weight
                )
        pred_pr = _pagerank(pred_nodes, pred_adj) if pred_nodes else {}

        gt_adj: dict[str, dict[str, float]] = {n_id: {} for n_id in gt_nodes}
        for edge in gt_snapshots[i].edges:
            if edge.source in gt_adj:
                gt_adj[edge.source][edge.target] = (
                    gt_adj[edge.source].get(edge.target, 0.0) + edge.weight
                )
        gt_pr = _pagerank(gt_nodes, gt_adj) if gt_nodes else {}

        pred_pr_vectors.append(pred_pr)
        gt_pr_vectors.append(gt_pr)

        # Trend consistency: check if direction of change matches for each metric
        if prev_pred_metrics is not None and prev_gt_metrics is not None and i < len(prev_pred_metrics):
            for mid in METRIC_IDS:
                pred_delta = pred_metrics.get(mid, 0.0) - prev_pred_metrics[i].get(mid, 0.0)
                gt_delta = gt_metrics.get(mid, 0.0) - prev_gt_metrics[i].get(mid, 0.0)
                if (pred_delta >= 0) == (gt_delta >= 0):
                    trend_correct += 1
                trend_total += 1

    # Aggregate MAE
    pagerank_mae = float(np.mean(pr_errors)) if pr_errors else float("nan")
    hhi_mae = float(np.mean(hhi_errors)) if hhi_errors else float("nan")
    density_mae = float(np.mean(density_errors)) if density_errors else float("nan")
    gini_mae = float(np.mean(gini_errors)) if gini_errors else float("nan")

    # Rank Correlation: average Spearman rho across snapshots
    rank_corrs: list[float] = []
    for pred_pr, gt_pr in zip(pred_pr_vectors, gt_pr_vectors):
        common_nodes = sorted(set(pred_pr.keys()) & set(gt_pr.keys()))
        if len(common_nodes) >= 3:
            pred_vals = [pred_pr[n] for n in common_nodes]
            gt_vals = [gt_pr[n] for n in common_nodes]
            rho, _ = scipy_stats.spearmanr(pred_vals, gt_vals)
            if not np.isnan(rho):
                rank_corrs.append(float(rho))
    rank_correlation = float(np.mean(rank_corrs)) if rank_corrs else float("nan")

    # Trend consistency
    trend_consistency = trend_correct / trend_total if trend_total > 0 else float("nan")

    return {
        "pagerank_mae": pagerank_mae,
        "hhi_mae": hhi_mae,
        "density_mae": density_mae,
        "gini_mae": gini_mae,
        "rank_correlation": rank_correlation,
        "trend_consistency": trend_consistency,
    }


def _aggregate_metric(metrics: dict[str, float]) -> float:
    """Compute a single scalar from B1-style metrics for degradation comparison.

    Uses mean of the four MAE values (lower is better).
    """
    mae_vals = [
        metrics.get("pagerank_mae", float("nan")),
        metrics.get("hhi_mae", float("nan")),
        metrics.get("density_mae", float("nan")),
        metrics.get("gini_mae", float("nan")),
    ]
    valid = [v for v in mae_vals if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_b6(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    regimes: list[str] | None = None,
    horizon: int = FIXED_HORIZON,
    **kwargs,
) -> list[B6Result]:
    """Run B6 benchmark for a given method across all degradation regimes.

    Args:
        method_id: Forecasting method registered in experiments.methods.
        data_dir: Path to processed graph snapshots.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        regimes: Override default regimes list.
        horizon: Forecasting horizon in weeks (default: 4).
        **kwargs: Extra method-specific config.

    Returns:
        List of B6Result, one per regime.
    """
    if regimes is None:
        regimes = REGIMES

    results_dir = kwargs.pop("results_dir", "results/")
    log = ExpLogger("B6", method=method_id, results_dir=results_dir)

    log.info(
        f"B6 | method={method_id} | test_split={test_split} | "
        f"regimes={regimes} | horizon={horizon}"
    )

    rng = np.random.default_rng(seed=42)

    # --- Load test snapshots ---
    loader = SnapshotLoader(data_dir=data_dir)
    test_snapshots = loader.load(date_range=test_split)
    all_dates = loader.dates

    if len(test_snapshots) < 2:
        log.warning("B6: fewer than 2 test snapshots, returning NaN results")
        return [B6Result(method=method_id, regime=r, horizon=horizon, n_test_snapshots=0)
                for r in regimes]

    log.info(f"B6: loaded {len(test_snapshots)} test snapshots")

    # Build date -> snapshot index for GT lookups
    all_snaps = loader.load()
    date_to_snap: dict[str, GraphSnapshot] = {s.date: s for s in all_snaps}

    # --- Build paired (input, pred, gt) lists for clean evaluation ---
    # Keep input snapshots so we can degrade them before prediction in regimes
    from experiments.predict_helper import predict_graph

    clean_inputs: list[GraphSnapshot] = []  # original input snapshots
    clean_pred: list[GraphSnapshot] = []
    clean_gt: list[GraphSnapshot] = []

    for snap_t in test_snapshots:
        t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
        if t_idx < 0:
            continue
        future_idx = t_idx + horizon
        if future_idx >= len(all_dates):
            continue
        future_date = all_dates[future_idx]
        if future_date in date_to_snap:
            gt_graph = date_to_snap[future_date]
        else:
            try:
                gt_graph = loader.load_single(future_date)
            except KeyError:
                continue
        clean_inputs.append(snap_t)
        clean_pred.append(predict_graph(method_id, snap_t, horizon=horizon))
        clean_gt.append(gt_graph)

    if not clean_pred:
        log.warning("B6: no valid (pred, gt) pairs")
        return [B6Result(method=method_id, regime=r, horizon=horizon, n_test_snapshots=0)
                for r in regimes]

    n_pairs = len(clean_pred)
    log.info(f"B6: {n_pairs} valid (pred, gt) pairs for horizon={horizon}")

    # --- Step 2: Clean (reference) B1-style evaluation ---
    with log.timer("clean baseline evaluation"):
        clean_metrics = _compute_b1_metrics(clean_pred, clean_gt)
        clean_aggregate = _aggregate_metric(clean_metrics)

    log.info(
        f"B6 clean baseline: pr_mae={clean_metrics['pagerank_mae']:.4f} "
        f"hhi_mae={clean_metrics['hhi_mae']:.4f} "
        f"density_mae={clean_metrics['density_mae']:.4f} "
        f"gini_mae={clean_metrics['gini_mae']:.4f} "
        f"aggregate={clean_aggregate:.4f}"
    )

    # --- Step 3: Evaluate each regime ---
    results: list[B6Result] = []

    for regime in log.progress(regimes, desc="Regimes", total=len(regimes), unit="regime"):
        if regime not in REGIME_CONFIGS:
            log.warning(f"B6: unknown regime {regime}, skipping")
            results.append(B6Result(method=method_id, regime=regime, horizon=horizon, n_test_snapshots=0))
            continue

        regime_cfg = REGIME_CONFIGS[regime]
        log.info(f"B6: evaluating regime {regime} with config {regime_cfg}")

        train_fraction = regime_cfg.get("train_fraction", 1.0)

        if train_fraction < 1.0:
            # Low-data regime: subsample the (input, gt) pairs, then predict
            n_keep = max(1, int(len(clean_inputs) * train_fraction))
            indices = sorted(rng.choice(len(clean_inputs), size=n_keep, replace=False))
            degraded_pred = [predict_graph(method_id, clean_inputs[i], horizon=horizon)
                            for i in indices]
            degraded_gt = [clean_gt[i] for i in indices]
        else:
            # Per-snapshot degradation: degrade INPUT, then predict on degraded input
            # This correctly tests robustness — the model sees noisy/incomplete data
            degraded_inputs = [_apply_degradation(snap, regime_cfg, rng)
                               for snap in clean_inputs]
            degraded_pred = [predict_graph(method_id, deg_snap, horizon=horizon)
                            for deg_snap in degraded_inputs]
            degraded_gt = clean_gt  # GT is never degraded

        if not degraded_pred:
            results.append(B6Result(method=method_id, regime=regime, horizon=horizon, n_test_snapshots=0))
            continue

        # Compute B1-style metrics on degraded data
        degraded_metrics = _compute_b1_metrics(degraded_pred, degraded_gt)
        degraded_aggregate = _aggregate_metric(degraded_metrics)

        # Relative degradation vs clean
        if not np.isnan(clean_aggregate) and clean_aggregate > 0:
            rel_degrad = (degraded_aggregate - clean_aggregate) / clean_aggregate
        elif not np.isnan(degraded_aggregate):
            rel_degrad = float("inf") if degraded_aggregate > 0 else 0.0
        else:
            rel_degrad = float("nan")

        result = B6Result(
            method=method_id,
            regime=regime,
            horizon=horizon,
            pagerank_mae=degraded_metrics["pagerank_mae"],
            hhi_mae=degraded_metrics["hhi_mae"],
            density_mae=degraded_metrics["density_mae"],
            gini_mae=degraded_metrics["gini_mae"],
            rank_correlation=degraded_metrics["rank_correlation"],
            trend_consistency=degraded_metrics["trend_consistency"],
            relative_degradation=float(rel_degrad),
            n_test_snapshots=len(degraded_pred),
        )
        results.append(result)

        log.step(
            regime,
            pr_mae=degraded_metrics["pagerank_mae"],
            hhi_mae=degraded_metrics["hhi_mae"],
            rel_degrad=float(rel_degrad),
            n_snapshots=len(degraded_pred),
        )

    # --- Summary ---
    valid_results = [r for r in results if r.n_test_snapshots and r.n_test_snapshots > 0]
    if valid_results:
        log.summary({
            "n_regimes": len(valid_results),
            "clean_aggregate_mae": clean_aggregate,
            "mean_pagerank_mae": float(np.mean([r.pagerank_mae for r in valid_results])),
            "mean_hhi_mae": float(np.mean([r.hhi_mae for r in valid_results])),
            "mean_relative_degradation": float(np.mean([
                r.relative_degradation for r in valid_results
                if not np.isnan(r.relative_degradation) and r.relative_degradation != float("inf")
            ])) if any(
                not np.isnan(r.relative_degradation) and r.relative_degradation != float("inf")
                for r in valid_results
            ) else float("nan"),
        })
    else:
        log.summary({"n_regimes": 0})

    log.save_results([
        {
            "method": r.method, "regime": r.regime, "horizon": r.horizon,
            "pagerank_mae": r.pagerank_mae, "hhi_mae": r.hhi_mae,
            "density_mae": r.density_mae, "gini_mae": r.gini_mae,
            "rank_correlation": r.rank_correlation,
            "trend_consistency": r.trend_consistency,
            "relative_degradation": r.relative_degradation,
            "n_test_snapshots": r.n_test_snapshots,
        }
        for r in results
    ])

    log.info(f"B6 complete: {len(results)} regime results")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="B6: Robustness benchmark")
    parser.add_argument("--method", required=True, help="Method ID (e.g. C0, C4)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range YYYY-MM~YYYY-MM")
    parser.add_argument("--regimes", default=",".join(REGIMES),
                        help="Comma-separated regime names")
    parser.add_argument("--horizon", type=int, default=FIXED_HORIZON,
                        help="Forecasting horizon in weeks")
    args = parser.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",")]
    results = run_b6(
        args.method, args.data_dir, args.test_split,
        regimes=regimes, horizon=args.horizon,
    )
    for r in results:
        print(r)
