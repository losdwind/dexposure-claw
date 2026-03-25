#!/usr/bin/env python3
"""B3: Uncertainty Calibration (Section 3.4 of EXPERIMENT_PLAN)

Evaluates the quality of predictive uncertainty estimates.
Methods that do not produce distributional outputs are not applicable (see APPLICABILITY).

Metrics:
- ECE              -- Expected Calibration Error (lower is better)
- PI Coverage      -- fraction of true values inside 90% prediction interval (target: 0.90)
- PI Width         -- mean width of 90% prediction interval (lower is better)
- CRPS             -- Continuous Ranked Probability Score (lower is better)
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
from dexposure_agent.data_loader import SnapshotLoader
from dexposure_agent.monitor import compute_metrics, METRIC_NAMES
from dexposure_agent.types import Edge, GraphSnapshot


TARGET_COVERAGE = 0.90  # nominal coverage of prediction intervals

METRIC_IDS = list(METRIC_NAMES.keys())  # M1, M3, M4, M6, M7

MC_NOISE_SIGMA = 0.1  # standard deviation for persistence MC noise


@dataclass
class B3Result:
    method: str
    ece: float = float("nan")
    pi_coverage: float = float("nan")   # fraction, target = TARGET_COVERAGE
    pi_width: float = float("nan")      # mean width in original units
    crps: float = float("nan")
    n_predictions: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"B3Result(method={self.method}, ECE={self.ece:.4f}, "
            f"PI_cov={self.pi_coverage:.3f} (target={TARGET_COVERAGE}), "
            f"PI_width={self.pi_width:.4f}, CRPS={self.crps:.4f})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_mc_samples(
    graph: GraphSnapshot,
    n_samples: int,
    sigma: float = MC_NOISE_SIGMA,
    rng: np.random.Generator | None = None,
) -> list[GraphSnapshot]:
    """Generate MC samples by adding N(0, sigma * w) noise to edge weights.

    This implements persistence prediction with Gaussian perturbation:
    for each MC draw, every edge weight w_ij is replaced by
    max(0, w_ij + N(0, sigma * w_ij)).
    """
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


def _compute_ece(predicted_probs: list[float], actual_inside: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Bins predicted confidence levels and computes the weighted-average gap
    between predicted probability and observed frequency within each bin.

    Here 'predicted_probs' are the fractional coverages from MC quantiles for
    each prediction, and 'actual_inside' indicates whether the actual fell inside.
    """
    if not predicted_probs:
        return 0.0

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for prob, inside in zip(predicted_probs, actual_inside):
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        bins[bin_idx].append((prob, inside))

    total = len(predicted_probs)
    ece = 0.0
    for bin_entries in bins:
        if not bin_entries:
            continue
        avg_prob = np.mean([p for p, _ in bin_entries])
        avg_actual = np.mean([float(a) for _, a in bin_entries])
        ece += len(bin_entries) / total * abs(avg_prob - avg_actual)

    return float(ece)


def _compute_crps_sample(samples: np.ndarray, actual: float) -> float:
    """Approximate CRPS using the energy form for a set of samples.

    CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are iid draws from the predictive distribution.
    """
    n = len(samples)
    if n == 0:
        return 0.0

    # E|X - y|
    term1 = float(np.mean(np.abs(samples - actual)))

    # E|X - X'| via double-sum (O(n^2) but MC samples are small)
    if n > 1:
        sorted_s = np.sort(samples)
        # Efficient formula: E|X-X'| = (2/(n^2)) * sum_i (2*i - n) * sorted_s[i]
        indices = np.arange(n)
        term2 = float(np.sum((2.0 * indices - n) * sorted_s) * 2.0 / (n * n))
    else:
        term2 = 0.0

    return term1 - 0.5 * abs(term2)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_b3(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    target_coverage: float = TARGET_COVERAGE,
    **kwargs,
) -> list[B3Result]:
    """Run B3 benchmark for a given method.

    Args:
        method_id: One of C0, C1, C4 (methods with uncertainty output).
        data_dir: Path to processed graph snapshots.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        target_coverage: Nominal PI coverage level (default 0.90).
        **kwargs: Extra method-specific config.

    Returns:
        List containing a single B3Result.
    """
    results_dir = kwargs.pop("results_dir", "results/")
    log = ExpLogger("B3", method=method_id, results_dir=results_dir)

    log.info(
        f"B3 | method={method_id} | test_split={test_split} | "
        f"target_coverage={target_coverage}"
    )

    config = AgentConfig(**{k: v for k, v in kwargs.items() if k in AgentConfig.model_fields})
    mc_count = config.mc_samples
    rng = np.random.default_rng(seed=42)

    # --- Load test snapshots ---
    loader = SnapshotLoader(data_dir=data_dir)
    test_snapshots = loader.load(date_range=test_split)
    all_dates = loader.dates

    if len(test_snapshots) < 2:
        log.warning("B3: fewer than 2 test snapshots, returning NaN result")
        return [B3Result(method=method_id, n_predictions=0)]

    log.info(f"B3: loaded {len(test_snapshots)} test snapshots")

    # Build a date -> snapshot index for horizon lookup
    date_to_snap: dict[str, GraphSnapshot] = {s.date: s for s in test_snapshots}

    # Use horizon = 1 (next week) as default for calibration evaluation
    horizon = kwargs.get("horizon", 1)
    alpha_lo = (1.0 - target_coverage) / 2.0
    alpha_hi = 1.0 - alpha_lo

    # --- Accumulators ---
    all_coverages: list[bool] = []       # was actual inside PI?
    all_pi_widths: list[float] = []      # width of PI
    all_crps: list[float] = []           # CRPS per prediction
    all_predicted_probs: list[float] = []  # for ECE: predicted coverage fraction
    all_actual_inside: list[bool] = []     # for ECE

    n_predictions = 0

    # --- Iterate over test snapshots ---
    for i, snap_t in log.progress(
        enumerate(test_snapshots), desc="Test snapshots", total=len(test_snapshots), unit="snap"
    ):
        # Find ground truth snapshot at t+h
        t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
        if t_idx < 0:
            continue
        future_idx = t_idx + horizon
        if future_idx >= len(all_dates):
            continue
        future_date = all_dates[future_idx]

        # Load ground truth (may need to load individually if not in test set)
        if future_date in date_to_snap:
            gt_graph = date_to_snap[future_date]
        else:
            try:
                gt_graph = loader.load_single(future_date)
            except KeyError:
                log.info(f"B3: no ground truth for {future_date}, skipping")
                continue

        # Generate prediction (FM for C0/C4, persistence for C2/others)
        from experiments.predict_helper import predict_graph
        pred_graph = predict_graph(method_id, snap_t, horizon=horizon)

        # Generate MC samples with noise
        mc_samples = _generate_mc_samples(pred_graph, mc_count, sigma=MC_NOISE_SIGMA, rng=rng)

        # Compute ground truth metrics
        gt_metrics = compute_metrics(gt_graph)

        # Compute metrics on each MC sample
        sample_metrics: dict[str, list[float]] = {mid: [] for mid in METRIC_IDS}
        for sample in mc_samples:
            sm = compute_metrics(sample)
            for mid in METRIC_IDS:
                sample_metrics[mid].append(sm.get(mid, 0.0))

        # For each metric: compute PI coverage, width, CRPS
        for mid in METRIC_IDS:
            actual_val = gt_metrics.get(mid, 0.0)
            samples_arr = np.array(sample_metrics[mid])

            if len(samples_arr) == 0:
                continue

            # 90% prediction interval from MC quantiles
            lo = float(np.quantile(samples_arr, alpha_lo))
            hi = float(np.quantile(samples_arr, alpha_hi))

            inside = lo <= actual_val <= hi
            width = hi - lo

            all_coverages.append(inside)
            all_pi_widths.append(width)

            # CRPS for this metric-snapshot pair
            crps_val = _compute_crps_sample(samples_arr, actual_val)
            all_crps.append(crps_val)

            # For ECE: predicted probability = target_coverage (since we always build
            # a target_coverage PI), actual = whether it was inside
            all_predicted_probs.append(target_coverage)
            all_actual_inside.append(inside)

            n_predictions += 1

        log.step(
            f"snapshot {i+1}",
            date=snap_t.date, n_preds_so_far=n_predictions,
        )

    if n_predictions == 0:
        log.warning("B3: no valid predictions computed")
        return [B3Result(method=method_id, n_predictions=0)]

    # --- Aggregate metrics ---
    pi_coverage = float(np.mean(all_coverages))
    pi_width = float(np.mean(all_pi_widths))
    crps = float(np.mean(all_crps))

    # ECE: bin the predicted probabilities (all are target_coverage here,
    # so ECE reduces to |target_coverage - actual_coverage|)
    ece = _compute_ece(all_predicted_probs, all_actual_inside, n_bins=10)

    result = B3Result(
        method=method_id,
        ece=ece,
        pi_coverage=pi_coverage,
        pi_width=pi_width,
        crps=crps,
        n_predictions=n_predictions,
    )

    log.summary({
        "ECE": ece,
        "PI_coverage": pi_coverage,
        "PI_width": pi_width,
        "CRPS": crps,
        "n_predictions": n_predictions,
    })

    log.save_results([{
        "method": method_id,
        "ece": ece,
        "pi_coverage": pi_coverage,
        "pi_width": pi_width,
        "crps": crps,
        "n_predictions": n_predictions,
    }])

    log.info(f"B3 complete: {result}")
    return [result]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="B3: Uncertainty Calibration benchmark")
    parser.add_argument("--method", required=True, help="Method ID (e.g. C0, C1, C4)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range YYYY-MM~YYYY-MM")
    parser.add_argument("--target-coverage", type=float, default=TARGET_COVERAGE,
                        help="Nominal PI coverage level")
    args = parser.parse_args()

    results = run_b3(
        args.method, args.data_dir, args.test_split,
        target_coverage=args.target_coverage,
    )
    for r in results:
        print(r)
