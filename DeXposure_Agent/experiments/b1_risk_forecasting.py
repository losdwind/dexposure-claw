#!/usr/bin/env python3
"""B1: Risk Forecasting (Section 3.2 of EXPERIMENT_PLAN)

Evaluates multi-horizon systemic risk metric forecasting across
prediction horizons h in {1, 4, 8, 12} weeks.

Metrics:
- PageRank MAE       -- weighted PageRank of exposure network
- HHI MAE            -- Herfindahl-Hirschman Index (concentration)
- Density MAE        -- edge density of the exposure graph
- Gini MAE           -- Gini coefficient of exposure weights
- Rank Correlation   -- Spearman rho between predicted and true protocol rankings
- Trend Consistency  -- fraction of correctly predicted directional changes
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from scipy.stats import spearmanr

# Ensure repo root is on sys.path so dexposure_agent is importable
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dexposure_agent.data_loader import SnapshotLoader, parse_date_range
from dexposure_agent.monitor import compute_metrics, _pagerank
from dexposure_agent.types import GraphSnapshot
from experiments.exp_logger import ExpLogger
from experiments.methods import METHOD_NAMES
from experiments.predict_helper import predict_graph


HORIZONS = [1, 4, 8, 12]  # weeks


@dataclass
class B1Result:
    method: str
    horizon: int
    pagerank_mae: float = float("nan")
    hhi_mae: float = float("nan")
    density_mae: float = float("nan")
    gini_mae: float = float("nan")
    rank_correlation: float = float("nan")   # Spearman rho
    trend_consistency: float = float("nan")  # fraction in [0, 1]
    n_test_snapshots: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"B1Result(method={self.method}, h={self.horizon}, "
            f"pr_mae={self.pagerank_mae:.4f}, hhi_mae={self.hhi_mae:.4f}, "
            f"density_mae={self.density_mae:.4f}, gini_mae={self.gini_mae:.4f}, "
            f"spearman={self.rank_correlation:.4f}, trend={self.trend_consistency:.4f})"
        )


# ---------------------------------------------------------------------------
# Helper: compute PageRank vector for a GraphSnapshot
# ---------------------------------------------------------------------------

def _compute_pagerank_vector(graph: GraphSnapshot) -> dict[str, float]:
    """Compute PageRank for all nodes in a graph snapshot.

    Uses the power-iteration implementation from the monitor module.
    Returns {node_id: pagerank_value}.
    """
    nodes = list(graph.nodes.keys())
    if not nodes:
        return {}

    # Build adjacency dict matching monitor._pagerank signature
    adjacency: dict[str, dict[str, float]] = {n: {} for n in nodes}
    for edge in graph.edges:
        if edge.source in adjacency:
            adjacency[edge.source][edge.target] = (
                adjacency[edge.source].get(edge.target, 0.0) + edge.weight
            )

    return _pagerank(nodes, adjacency)


# ---------------------------------------------------------------------------
# Helper: find the ground-truth snapshot at t + h weeks
# ---------------------------------------------------------------------------

def _find_future_snapshot(
    date_str: str,
    horizon_weeks: int,
    date_to_snapshot: dict[str, GraphSnapshot],
    sorted_dates: list[str],
    tolerance_days: int = 3,
) -> GraphSnapshot | None:
    """Find the snapshot closest to (date + horizon_weeks * 7 days).

    Searches within +/- tolerance_days of the target date.
    Returns None if no snapshot is found within the tolerance window.
    """
    base_dt = datetime.strptime(date_str, "%Y-%m-%d")
    target_dt = base_dt + timedelta(weeks=horizon_weeks)

    best_snap = None
    best_delta = timedelta(days=tolerance_days + 1)

    for d in sorted_dates:
        d_dt = datetime.strptime(d, "%Y-%m-%d")
        delta = abs(d_dt - target_dt)
        if delta <= timedelta(days=tolerance_days) and delta < best_delta:
            best_delta = delta
            best_snap = date_to_snapshot[d]

    return best_snap


# ---------------------------------------------------------------------------
# Helper: compute per-node PageRank MAE
# ---------------------------------------------------------------------------

def _pagerank_mae(
    pred_pr: dict[str, float],
    actual_pr: dict[str, float],
) -> float:
    """Mean |predicted_PR(node) - actual_PR(node)| across the union of nodes.

    Missing nodes are treated as having PageRank 0.
    """
    all_nodes = set(pred_pr.keys()) | set(actual_pr.keys())
    if not all_nodes:
        return 0.0

    total_ae = 0.0
    for node in all_nodes:
        total_ae += abs(pred_pr.get(node, 0.0) - actual_pr.get(node, 0.0))

    return total_ae / len(all_nodes)


# ---------------------------------------------------------------------------
# Helper: Spearman rank correlation on PageRank vectors
# ---------------------------------------------------------------------------

def _rank_correlation(
    pred_pr: dict[str, float],
    actual_pr: dict[str, float],
) -> float:
    """Spearman rank correlation between predicted and actual PageRank.

    Nodes missing from one vector get PageRank 0.
    Returns 0.0 if fewer than 3 nodes or constant values.
    """
    all_nodes = sorted(set(pred_pr.keys()) | set(actual_pr.keys()))
    if len(all_nodes) < 3:
        return 0.0

    pred_vals = np.array([pred_pr.get(n, 0.0) for n in all_nodes])
    actual_vals = np.array([actual_pr.get(n, 0.0) for n in all_nodes])

    # Handle constant arrays (spearmanr returns nan for them)
    if np.std(pred_vals) == 0.0 or np.std(actual_vals) == 0.0:
        return 0.0

    rho, _ = spearmanr(pred_vals, actual_vals)
    return float(rho) if not math.isnan(rho) else 0.0


# ---------------------------------------------------------------------------
# Helper: trend consistency
# ---------------------------------------------------------------------------

def _trend_consistency(
    metrics_t: dict[str, float],
    metrics_pred: dict[str, float],
    metrics_actual: dict[str, float],
) -> float:
    """Fraction of metrics where the predicted direction matches reality.

    For each metric m:
      predicted_trend  = sign(metrics_pred[m]   - metrics_t[m])
      actual_trend     = sign(metrics_actual[m]  - metrics_t[m])
      match = (predicted_trend == actual_trend)

    Returns the fraction of matches across all common metrics.
    """
    common_keys = set(metrics_t.keys()) & set(metrics_pred.keys()) & set(metrics_actual.keys())
    if not common_keys:
        return 0.0

    matches = 0
    for k in common_keys:
        pred_sign = np.sign(metrics_pred[k] - metrics_t[k])
        actual_sign = np.sign(metrics_actual[k] - metrics_t[k])
        if pred_sign == actual_sign:
            matches += 1

    return matches / len(common_keys)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_b1(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    horizons: list[int] | None = None,
    **kwargs,
) -> list[B1Result]:
    """Run B1 benchmark for a given method across all horizons.

    Args:
        method_id: Forecasting method ID registered in experiments.methods.
        data_dir: Path to processed graph snapshots.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        horizons: Override default horizons (default: [1, 4, 8, 12]).
        **kwargs: Extra method-specific config passed through.

    Returns:
        List of B1Result, one per horizon.
    """
    if horizons is None:
        horizons = HORIZONS

    results_dir = kwargs.pop("results_dir", "results/")
    log = ExpLogger("B1", method=method_id, results_dir=results_dir)

    log.info(
        "B1 | method={} ({}) | test_split={} | horizons={}",
        method_id, METHOD_NAMES.get(method_id, "?"), test_split, horizons,
    )

    # ------------------------------------------------------------------
    # Step 1: Load ALL snapshots (need future ground truth beyond test range)
    # ------------------------------------------------------------------
    loader = SnapshotLoader(data_dir=data_dir)
    all_snapshots = loader.load()  # no date filter -> load everything
    log.info("B1 | loaded {} total snapshots", len(all_snapshots))

    if not all_snapshots:
        log.error("B1 | no snapshots found in {} -- aborting", data_dir)
        return [B1Result(method=method_id, horizon=h, n_test_snapshots=0) for h in horizons]

    # Build lookup structures
    date_to_snapshot: dict[str, GraphSnapshot] = {s.date: s for s in all_snapshots}
    sorted_dates = sorted(date_to_snapshot.keys())
    log.info("B1 | date range: {} ~ {}", sorted_dates[0], sorted_dates[-1])

    # ------------------------------------------------------------------
    # Step 2: Identify test snapshots within test_split
    # ------------------------------------------------------------------
    dt_start, dt_end = parse_date_range(test_split)
    test_snapshots = [
        s for s in all_snapshots
        if dt_start <= datetime.strptime(s.date, "%Y-%m-%d") <= dt_end
    ]
    test_snapshots.sort(key=lambda s: s.date)
    log.info("B1 | {} test snapshots in {}", len(test_snapshots), test_split)

    if not test_snapshots:
        log.error("B1 | no test snapshots in {} -- aborting", test_split)
        return [B1Result(method=method_id, horizon=h, n_test_snapshots=0) for h in horizons]

    # ------------------------------------------------------------------
    # Step 3: Pre-compute metrics and PageRank for all snapshots (cache)
    # ------------------------------------------------------------------
    metrics_cache: dict[str, dict[str, float]] = {}
    pagerank_cache: dict[str, dict[str, float]] = {}

    for snap in log.progress(all_snapshots, desc="Pre-computing metrics", unit="snap"):
        metrics_cache[snap.date] = compute_metrics(snap)
        pagerank_cache[snap.date] = _compute_pagerank_vector(snap)

    log.info("B1 | metrics/PageRank cache built for {} dates", len(metrics_cache))

    # ------------------------------------------------------------------
    # Step 4: Evaluate per horizon
    # ------------------------------------------------------------------
    results: list[B1Result] = []

    for h in horizons:
        log.info("B1 | evaluating horizon h={} weeks ...", h)

        pr_mae_list: list[float] = []
        hhi_mae_list: list[float] = []
        density_mae_list: list[float] = []
        gini_mae_list: list[float] = []
        rank_corr_list: list[float] = []
        trend_list: list[float] = []
        n_evaluated = 0

        for snap_t in log.progress(test_snapshots, desc=f"Horizon h={h}", unit="snap"):
            # Find ground-truth snapshot at t + h weeks
            snap_actual = _find_future_snapshot(
                snap_t.date, h, date_to_snapshot, sorted_dates,
            )
            if snap_actual is None:
                log.step(f"h={h}", date=snap_t.date, status="skipped_no_gt")
                continue

            # Generate predicted graph
            snap_pred = predict_graph(method_id, snap_t, h, config=kwargs.get("config"))

            # Compute metrics for predicted graph
            # Always compute fresh: FM prediction has same date but different structure
            if snap_pred is snap_t:
                # Persistence: reuse cached metrics
                metrics_pred = metrics_cache[snap_t.date]
                pr_pred = pagerank_cache[snap_t.date]
            else:
                # FM or other model: compute on the predicted graph
                metrics_pred = compute_metrics(snap_pred)
                pr_pred = _compute_pagerank_vector(snap_pred)

            # Ground-truth metrics (always from cache)
            metrics_actual = metrics_cache[snap_actual.date]
            pr_actual = pagerank_cache[snap_actual.date]

            # Current-time metrics (for trend consistency)
            metrics_t = metrics_cache[snap_t.date]

            # -- PageRank MAE --
            pr_mae_val = _pagerank_mae(pr_pred, pr_actual)
            pr_mae_list.append(pr_mae_val)

            # -- HHI MAE (M3) --
            hhi_mae_val = abs(metrics_pred.get("M3", 0.0) - metrics_actual.get("M3", 0.0))
            hhi_mae_list.append(hhi_mae_val)

            # -- Density MAE (M4) --
            density_mae_list.append(abs(metrics_pred.get("M4", 0.0) - metrics_actual.get("M4", 0.0)))

            # -- Gini MAE (M7: Gini of weighted degree) --
            gini_mae_list.append(abs(metrics_pred.get("M7", 0.0) - metrics_actual.get("M7", 0.0)))

            # -- Spearman rank correlation --
            rank_corr_list.append(_rank_correlation(pr_pred, pr_actual))

            # -- Trend consistency --
            trend_list.append(_trend_consistency(metrics_t, metrics_pred, metrics_actual))

            n_evaluated += 1

            # Log per-step metrics
            log.step(f"h={h}", date=snap_t.date, pr_mae=pr_mae_val, hhi_mae=hhi_mae_val)

        # Aggregate results for this horizon
        if n_evaluated == 0:
            log.warning("B1 | h={} | no valid test pairs found", h)
            results.append(B1Result(method=method_id, horizon=h, n_test_snapshots=0))
            continue

        result = B1Result(
            method=method_id,
            horizon=h,
            pagerank_mae=float(np.mean(pr_mae_list)),
            hhi_mae=float(np.mean(hhi_mae_list)),
            density_mae=float(np.mean(density_mae_list)),
            gini_mae=float(np.mean(gini_mae_list)),
            rank_correlation=float(np.mean(rank_corr_list)),
            trend_consistency=float(np.mean(trend_list)),
            n_test_snapshots=n_evaluated,
        )
        log.info("B1 | {}", result)
        results.append(result)

    # ------------------------------------------------------------------
    # Step 5: Summary and save
    # ------------------------------------------------------------------
    summary_metrics = {}
    for r in results:
        if r.n_test_snapshots and r.n_test_snapshots > 0:
            summary_metrics[f"h{r.horizon}_pr_mae"] = r.pagerank_mae
            summary_metrics[f"h{r.horizon}_hhi_mae"] = r.hhi_mae
            summary_metrics[f"h{r.horizon}_density_mae"] = r.density_mae
            summary_metrics[f"h{r.horizon}_spearman"] = r.rank_correlation
            summary_metrics[f"h{r.horizon}_trend"] = r.trend_consistency
    log.summary(summary_metrics)

    # Save structured results
    serialized = []
    for r in results:
        serialized.append({
            "method": r.method,
            "horizon": r.horizon,
            "pagerank_mae": r.pagerank_mae,
            "hhi_mae": r.hhi_mae,
            "density_mae": r.density_mae,
            "gini_mae": r.gini_mae,
            "rank_correlation": r.rank_correlation,
            "trend_consistency": r.trend_consistency,
            "n_test_snapshots": r.n_test_snapshots,
        })
    log.save_results(serialized)

    log.info(
        "B1 | method={} completed: {} horizons, {} total results",
        method_id, len(horizons), len(results),
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json as _json
    from datetime import datetime as _dt

    parser = argparse.ArgumentParser(description="B1: Risk Forecasting benchmark")
    parser.add_argument("--method", required=True, help="Method ID (e.g. C0, C2, C4)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range YYYY-MM~YYYY-MM")
    parser.add_argument("--horizons", default="1,4,8,12",
                        help="Comma-separated horizon values in weeks")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path (optional)")
    parser.add_argument("--log-file", default=None,
                        help="Log file path (optional, logs to stderr by default)")
    args = parser.parse_args()

    # Configure logging
    if args.log_file:
        logger.add(
            args.log_file,
            rotation="50 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
        logger.info("B1 log file: {}", args.log_file)

    horizons = [int(h) for h in args.horizons.split(",")]
    results = run_b1(
        args.method,
        args.data_dir,
        args.test_split,
        horizons=horizons,
    )

    # Print results to stdout
    for r in results:
        print(r)

    # Optionally save to JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = []
        for r in results:
            serialized.append({
                "method": r.method,
                "horizon": r.horizon,
                "pagerank_mae": r.pagerank_mae,
                "hhi_mae": r.hhi_mae,
                "density_mae": r.density_mae,
                "gini_mae": r.gini_mae,
                "rank_correlation": r.rank_correlation,
                "trend_consistency": r.trend_consistency,
                "n_test_snapshots": r.n_test_snapshots,
            })
        payload = {
            "benchmark": "B1",
            "method": args.method,
            "method_name": METHOD_NAMES.get(args.method, "?"),
            "test_split": args.test_split,
            "horizons": horizons,
            "timestamp": _dt.now().isoformat(),
            "results": serialized,
        }
        out_path.write_text(_json.dumps(payload, indent=2))
        logger.info("B1 | results saved to {}", out_path)
