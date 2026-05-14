#!/usr/bin/env python3
"""b4_stress: Stress Test (Section 3.5 of EXPERIMENT_PLAN)

Evaluates accuracy of simulated contagion outcomes under five stylised stress
scenarios S1-S5 applied to the exposure network.

Metrics:
- Loss MAE               -- MAE of total exposure loss
- Distressed Count MAE   -- MAE of number of distressed protocols
- Propagation Depth MAE  -- MAE of contagion propagation depth (hops)
- Target Overlap@K       -- Jaccard overlap of predicted vs true top-K affected

Scenarios:
- S1: Single large-protocol failure (top-5 by TVL, one at a time)
- S2: Sector-wide shock (all lending protocols -50% TVL)
- S3: Stablecoin depeg cascade (stablecoin protocols fail sequentially)
- S4: Liquidity crunch (all edge weights halved simultaneously)
- S5: Combined shock (S2 + S4 simultaneously)
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
from dexposure_agent.scenario import (
    SCENARIO_LIBRARY,
    apply_shock,
    compute_contagion_loss,
)
from dexposure_agent.types import GraphSnapshot


SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
DEFAULT_TOP_K = 10  # for Target Overlap@K


@dataclass
class StressResult:
    method: str
    scenario: str
    loss_mae: float = float("nan")
    distressed_count_mae: float = float("nan")
    propagation_depth_mae: float = float("nan")
    target_overlap_at_k: float = float("nan")
    top_k: int = DEFAULT_TOP_K
    n_simulations: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"StressResult(method={self.method}, scenario={self.scenario}, "
            f"loss_mae={self.loss_mae:.4f}, dist_count_mae={self.distressed_count_mae:.4f}, "
            f"prop_depth_mae={self.propagation_depth_mae:.4f}, "
            f"overlap@{self.top_k}={self.target_overlap_at_k:.4f})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets. Returns 0.0 if both empty."""
    if not set_a and not set_b:
        return 1.0  # both empty => perfect agreement
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_b4(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    scenarios: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    **kwargs,
) -> list[StressResult]:
    """Run b4_stress benchmark for a given method across all stress scenarios.

    Args:
        method_id: Scenario-capable forecasting method registered in experiments.methods.
        data_dir: Path to processed graph snapshots and scenario configs.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        scenarios: Override default scenarios (default: S1-S5).
        top_k: K for Target Overlap@K metric.
        **kwargs: Extra method-specific config.

    Returns:
        List of StressResult, one per scenario.
    """
    if scenarios is None:
        scenarios = SCENARIOS

    results_dir = kwargs.pop("results_dir", "results/")
    log = ExpLogger("b4_stress", method=method_id, results_dir=results_dir)

    log.info(
        f"b4_stress | method={method_id} | test_split={test_split} | "
        f"scenarios={scenarios} | top_k={top_k}"
    )

    config = AgentConfig(**{k: v for k, v in kwargs.items() if k in AgentConfig.model_fields})

    # --- Load test snapshots ---
    loader = SnapshotLoader(data_dir=data_dir)
    test_snapshots = loader.load(date_range=test_split)
    all_dates = loader.dates

    if len(test_snapshots) < 2:
        log.warning("b4_stress: fewer than 2 test snapshots, returning NaN results")
        return [StressResult(method=method_id, scenario=sid, top_k=top_k, n_simulations=0)
                for sid in scenarios]

    log.info(f"b4_stress: loaded {len(test_snapshots)} test snapshots")

    # Build date -> snapshot index
    date_to_snap: dict[str, GraphSnapshot] = {s.date: s for s in test_snapshots}

    # Default horizon for stress test comparison
    horizon = kwargs.get("horizon", 4)

    results: list[StressResult] = []

    for sid in log.progress(scenarios, desc="Scenarios", total=len(scenarios), unit="scenario"):
        if sid not in SCENARIO_LIBRARY:
            log.warning(f"b4_stress: scenario {sid} not in SCENARIO_LIBRARY, skipping")
            results.append(StressResult(method=method_id, scenario=sid, top_k=top_k, n_simulations=0))
            continue

        spec = SCENARIO_LIBRARY[sid]
        log.info(f"b4_stress: evaluating scenario {sid} ({spec.get('name', '')})")

        loss_errors: list[float] = []
        distressed_errors: list[float] = []
        depth_errors: list[float] = []
        overlaps: list[float] = []
        n_sims = 0

        for snap_t in log.progress(
            test_snapshots, desc=f"Scenario {sid} snapshots",
            total=len(test_snapshots), unit="snap"
        ):
            # Find ground truth snapshot at t+h
            t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
            if t_idx < 0:
                continue
            future_idx = t_idx + horizon
            if future_idx >= len(all_dates):
                continue
            future_date = all_dates[future_idx]

            # Load ground truth
            if future_date in date_to_snap:
                gt_graph = date_to_snap[future_date]
            else:
                try:
                    gt_graph = loader.load_single(future_date)
                except KeyError:
                    log.info(f"b4_stress: no ground truth for {future_date}, skipping")
                    continue

            # --- Predicted path: shock on predicted graph ---
            from experiments.predict_helper import predict_graph
            pred_graph = predict_graph(method_id, snap_t, horizon=horizon)
            pred_shocked = apply_shock(pred_graph, spec)
            pred_loss = compute_contagion_loss(
                pred_graph, pred_shocked,
                scenario_id=sid, scenario_name=spec.get("name", sid),
                horizon=horizon,
            )

            # --- Ground truth path: shock on actual G_{t+h} ---
            gt_shocked = apply_shock(gt_graph, spec)
            gt_loss = compute_contagion_loss(
                gt_graph, gt_shocked,
                scenario_id=sid, scenario_name=spec.get("name", sid),
                horizon=horizon,
            )

            # --- Compute errors ---
            loss_errors.append(abs(pred_loss.expected_loss - gt_loss.expected_loss))
            distressed_errors.append(abs(pred_loss.distressed_count - gt_loss.distressed_count))
            depth_errors.append(abs(pred_loss.propagation_depth - gt_loss.propagation_depth))

            # Target Overlap@K: Jaccard of top-K targets
            pred_top_k = set(pred_loss.top_targets[:top_k])
            gt_top_k = set(gt_loss.top_targets[:top_k])
            overlaps.append(_jaccard(pred_top_k, gt_top_k))

            n_sims += 1

            log.step(
                f"{sid}/{snap_t.date}",
                loss_err=loss_errors[-1], dist_err=distressed_errors[-1],
                depth_err=depth_errors[-1], overlap=overlaps[-1],
            )

        if n_sims == 0:
            log.warning(f"b4_stress: no valid simulations for scenario {sid}")
            results.append(StressResult(method=method_id, scenario=sid, top_k=top_k, n_simulations=0))
            continue

        result = StressResult(
            method=method_id,
            scenario=sid,
            loss_mae=float(np.mean(loss_errors)),
            distressed_count_mae=float(np.mean(distressed_errors)),
            propagation_depth_mae=float(np.mean(depth_errors)),
            target_overlap_at_k=float(np.mean(overlaps)),
            top_k=top_k,
            n_simulations=n_sims,
        )
        results.append(result)
        log.info(f"b4_stress scenario {sid}: {result}")

    # --- Summary ---
    valid_results = [r for r in results if r.n_simulations and r.n_simulations > 0]
    if valid_results:
        log.summary({
            "n_scenarios": len(valid_results),
            "mean_loss_mae": float(np.mean([r.loss_mae for r in valid_results])),
            "mean_distressed_mae": float(np.mean([r.distressed_count_mae for r in valid_results])),
            "mean_depth_mae": float(np.mean([r.propagation_depth_mae for r in valid_results])),
            "mean_overlap_at_k": float(np.mean([r.target_overlap_at_k for r in valid_results])),
        })
    else:
        log.summary({"n_scenarios": 0})

    log.save_results([
        {
            "method": r.method, "scenario": r.scenario,
            "loss_mae": r.loss_mae, "distressed_count_mae": r.distressed_count_mae,
            "propagation_depth_mae": r.propagation_depth_mae,
            "target_overlap_at_k": r.target_overlap_at_k,
            "top_k": r.top_k, "n_simulations": r.n_simulations,
        }
        for r in results
    ])

    log.info(f"b4_stress complete: {len(results)} scenario results")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="b4_stress: Stress Test benchmark")
    parser.add_argument("--method", required=True, help="Method ID (e.g. m5_fm_rules, m4_fm_only)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range YYYY-MM~YYYY-MM")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS),
                        help="Comma-separated scenario IDs")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="K for Target Overlap@K")
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",")]
    results = run_b4(
        args.method, args.data_dir, args.test_split,
        scenarios=scenarios, top_k=args.top_k,
    )
    for r in results:
        print(r)
