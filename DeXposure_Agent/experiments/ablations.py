#!/usr/bin/env python3
"""Ablation studies for DeXposure-Agent.

Each ablation disables or degrades one component of DeXposure-Agent (C0) and
evaluates on the appropriate benchmarks, measuring the performance drop.

Ablation configs (applied as overrides to the C0 default config):
- A1: tau_data=0.0         -- disable data-health gating
- A2: tau_conf=0.0         -- disable confidence-based intervention control
- A3: skip_scenario=True   -- skip scenario engine (no stress simulation)
- A6: horizons=[4]         -- single-horizon forecasting only
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "A1": {"tau_data": 0.0},
    "A2": {"tau_conf": 0.0},
    "A3": {"skip_scenario": True},
    "A6": {"horizons": [4]},
}

ABLATION_DESCRIPTIONS = {
    "A1": "Disable data-health gating (tau_data=0.0)",
    "A2": "Disable confidence-based intervention control (tau_conf=0.0)",
    "A3": "Skip scenario engine",
    "A6": "Single-horizon forecasting (horizons=[4])",
}

# Which benchmarks each ablation should run
ABLATION_BENCHMARKS: dict[str, list[str]] = {
    "A1": ["B5"],        # data-health gate affects decision quality
    "A2": ["B5"],        # confidence gate affects decision quality
    "A3": ["B4", "B5"],  # scenario engine affects stress + decision
    "A6": ["B1", "B5"],  # horizons affect forecasting + decision
}


@dataclass
class AblationResult:
    ablation_id: str
    description: str
    config_override: dict[str, Any]
    benchmarks_run: list[str] = field(default_factory=list)
    # B1 metrics (only for A6)
    b1_rank_correlation: float = float("nan")
    b1_trend_consistency: float = float("nan")
    b1_hhi_mae: float = float("nan")
    # B4 metrics (only for A3)
    b4_loss_mae: float = float("nan")
    b4_overlap_at_10: float = float("nan")
    # B5 metrics (all ablations)
    b5_ticket_precision: float = float("nan")
    b5_target_stability: float = float("nan")
    b5_false_intervention_rate: float = float("nan")
    b5_suppression_rate: float = float("nan")
    # Relative change vs C0 full model (positive = worse)
    relative_drop: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"AblationResult({self.ablation_id}: {self.description}, "
            f"benchmarks={self.benchmarks_run}, "
            f"b5_prec={self.b5_ticket_precision:.4f}, "
            f"b5_fir={self.b5_false_intervention_rate:.4f}, "
            f"rel_drop={self.relative_drop})"
        )


def run_ablation(
    ablation_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    results_dir: str = "",
    c0_baseline: Optional[dict[str, Any]] = None,
    **kwargs,
) -> AblationResult:
    """Run a single ablation study.

    Runs C0 with the ablation's config override applied, then evaluates on
    the benchmarks specified for that ablation.

    Args:
        ablation_id: One of 'A1', 'A2', 'A3', 'A6'.
        data_dir: Path to processed graph snapshots.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        results_dir: Directory for saving results.
        c0_baseline: Optional dict of C0 full-model metrics for computing relative drop.
        **kwargs: Extra config overrides.

    Returns:
        AblationResult with metrics for this ablation variant.
    """
    if ablation_id not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation ID: {ablation_id}. "
            f"Must be one of {list(ABLATION_CONFIGS)}"
        )

    config_override = {**ABLATION_CONFIGS[ablation_id], **kwargs}
    description = ABLATION_DESCRIPTIONS[ablation_id]
    benchmarks = ABLATION_BENCHMARKS[ablation_id]

    logger.info(
        f"Ablation {ablation_id} | {description} | "
        f"override={config_override} | benchmarks={benchmarks}"
    )

    result = AblationResult(
        ablation_id=ablation_id,
        description=description,
        config_override=config_override,
        benchmarks_run=benchmarks,
    )

    # Run B1 if needed (A6)
    if "B1" in benchmarks:
        from experiments.b1_risk_forecasting import run_b1
        b1_results = run_b1(
            method_id="C0",
            data_dir=data_dir,
            test_split=test_split,
            results_dir=results_dir,
            **config_override,
        )
        if b1_results:
            # B1 returns a list with per-horizon results; use h=4 for comparison
            h4_results = [r for r in b1_results if r.horizon == 4]
            if h4_results:
                r = h4_results[0]
                result.b1_rank_correlation = r.rank_correlation
                result.b1_trend_consistency = r.trend_consistency
                result.b1_hhi_mae = r.hhi_mae

    # Run B4 if needed (A3)
    if "B4" in benchmarks:
        from experiments.b4_stress_test import run_b4
        b4_results = run_b4(
            method_id="C0",
            data_dir=data_dir,
            test_split=test_split,
            results_dir=results_dir,
            **config_override,
        )
        if b4_results:
            # Aggregate across scenarios
            loss_maes = [r.loss_mae for r in b4_results if not _isnan(r.loss_mae)]
            overlaps = [r.overlap_at_k for r in b4_results if not _isnan(r.overlap_at_k)]
            result.b4_loss_mae = float(sum(loss_maes) / len(loss_maes)) if loss_maes else float("nan")
            result.b4_overlap_at_10 = float(sum(overlaps) / len(overlaps)) if overlaps else float("nan")

    # Run B5 (all ablations)
    if "B5" in benchmarks:
        from experiments.b5_decision_quality import run_b5
        b5_results = run_b5(
            method_id="C0",
            data_dir=data_dir,
            test_split=test_split,
            results_dir=results_dir,
            **config_override,
        )
        if b5_results:
            r = b5_results[0]
            result.b5_ticket_precision = r.ticket_precision
            result.b5_target_stability = r.target_stability
            result.b5_false_intervention_rate = r.false_intervention_rate
            result.b5_suppression_rate = r.suppression_rate

    # Compute relative drop vs C0 baseline
    if c0_baseline:
        result.relative_drop = _compute_relative_drop(result, c0_baseline)

    logger.info(f"Ablation {ablation_id} complete: {result}")
    return result


def _isnan(v: float) -> bool:
    import math
    return math.isnan(v)


def _compute_relative_drop(
    ablation: AblationResult,
    baseline: dict[str, Any],
) -> dict[str, float]:
    """Compute relative performance change vs C0 baseline.

    Positive values = worse than baseline.
    For metrics where lower is better (MAE, FIR), sign is inverted.
    """
    drops = {}

    # B5 metrics (higher is better for precision/stability, lower for FIR)
    if not _isnan(ablation.b5_ticket_precision) and "b5_ticket_precision" in baseline:
        base = baseline["b5_ticket_precision"]
        if base > 0:
            drops["b5_precision"] = (base - ablation.b5_ticket_precision) / base

    if not _isnan(ablation.b5_false_intervention_rate) and "b5_false_intervention_rate" in baseline:
        base = baseline["b5_false_intervention_rate"]
        if base > 0:
            drops["b5_fir"] = (ablation.b5_false_intervention_rate - base) / base
        elif ablation.b5_false_intervention_rate > 0:
            drops["b5_fir"] = float("inf")

    if not _isnan(ablation.b5_target_stability) and "b5_target_stability" in baseline:
        base = baseline["b5_target_stability"]
        if base > 0:
            drops["b5_stability"] = (base - ablation.b5_target_stability) / base

    # B1 metrics
    if not _isnan(ablation.b1_rank_correlation) and "b1_rank_correlation" in baseline:
        base = baseline["b1_rank_correlation"]
        if base > 0:
            drops["b1_rank_corr"] = (base - ablation.b1_rank_correlation) / base

    return drops


def run_all_ablations(
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    results_dir: str = "",
    ablation_ids: list[str] | None = None,
) -> list[AblationResult]:
    """Run all (or a subset of) ablation studies.

    First runs C0 full-model as baseline, then each ablation variant.
    """
    if ablation_ids is None:
        ablation_ids = list(ABLATION_CONFIGS.keys())

    logger.info(f"Running ablations: {ablation_ids}")

    # Step 1: Run C0 baseline for reference
    logger.info("Running C0 baseline for ablation reference...")
    c0_baseline = _run_c0_baseline(data_dir, test_split, results_dir)
    logger.info(f"C0 baseline: {c0_baseline}")

    # Step 2: Run each ablation
    results = []
    for aid in ablation_ids:
        try:
            result = run_ablation(
                aid,
                data_dir=data_dir,
                test_split=test_split,
                results_dir=results_dir,
                c0_baseline=c0_baseline,
            )
            results.append(result)
        except Exception as exc:
            logger.error(f"Ablation {aid} failed: {exc}")
            results.append(
                AblationResult(
                    ablation_id=aid,
                    description=ABLATION_DESCRIPTIONS.get(aid, ""),
                    config_override=ABLATION_CONFIGS.get(aid, {}),
                )
            )

    # Save summary
    if results_dir:
        _save_ablation_summary(results, c0_baseline, results_dir)

    return results


def _run_c0_baseline(
    data_dir: str,
    test_split: str,
    results_dir: str,
) -> dict[str, Any]:
    """Run C0 with default config to get baseline metrics."""
    baseline = {}

    from experiments.b1_risk_forecasting import run_b1
    b1_results = run_b1("C0", data_dir, test_split, results_dir=results_dir)
    h4 = [r for r in b1_results if r.horizon == 4]
    if h4:
        baseline["b1_rank_correlation"] = h4[0].rank_correlation
        baseline["b1_trend_consistency"] = h4[0].trend_consistency
        baseline["b1_hhi_mae"] = h4[0].hhi_mae

    from experiments.b5_decision_quality import run_b5
    b5_results = run_b5("C0", data_dir, test_split, results_dir=results_dir)
    if b5_results:
        baseline["b5_ticket_precision"] = b5_results[0].ticket_precision
        baseline["b5_target_stability"] = b5_results[0].target_stability
        baseline["b5_false_intervention_rate"] = b5_results[0].false_intervention_rate
        baseline["b5_suppression_rate"] = b5_results[0].suppression_rate

    return baseline


def _save_ablation_summary(
    results: list[AblationResult],
    baseline: dict[str, Any],
    results_dir: str,
) -> None:
    """Save ablation results to JSON."""
    out_path = Path(results_dir) / "ablation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "c0_baseline": baseline,
        "ablations": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Ablation summary saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation studies A1,A2,A3,A6")
    parser.add_argument(
        "--ablations",
        default=",".join(ABLATION_CONFIGS.keys()),
        help="Comma-separated ablation IDs (e.g. A1,A2)",
    )
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument(
        "--test-split",
        default="2025-01~2025-08",
        help="Test split range YYYY-MM~YYYY-MM",
    )
    parser.add_argument("--results-dir", default="results/", help="Results directory")
    args = parser.parse_args()

    ablation_ids = [a.strip() for a in args.ablations.split(",")]
    results = run_all_ablations(
        data_dir=args.data_dir,
        test_split=args.test_split,
        results_dir=args.results_dir,
        ablation_ids=ablation_ids,
    )
    for r in results:
        print(r)
