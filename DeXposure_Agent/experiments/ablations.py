#!/usr/bin/env python3
"""Ablation studies A1-A8 for DeXposure-Agent (Section 4.4 of EXPERIMENT_PLAN)

Each ablation disables or degrades one component of DeXposure-Agent (C0) and
evaluates on B1 (h=4) unless otherwise noted, measuring the performance drop.

Ablation configs (applied as overrides to the C0 default config):
- A1: tau_data=0.0         -- disable data-health gating
- A2: tau_conf=0.0         -- disable confidence-weighted aggregation
- A3: skip_scenario=True   -- skip scenario engine (no stress simulation)
- A4: top_k=0              -- no attribution (disable GNNExplainer / gradient masking)
- A5: rolling_window=999999-- fixed (non-rolling) training window
- A6: horizons=[1]         -- single-horizon forecasting only
- A7: mc_samples=1         -- no Monte Carlo sampling (point estimate only)
- A8: unconstrained_actions=True -- all actions always feasible (no playbook constraints)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from loguru import logger


ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "A1": {"tau_data": 0.0},
    "A2": {"tau_conf": 0.0},
    "A3": {"skip_scenario": True},
    "A4": {"top_k": 0},
    "A5": {"rolling_window": 999_999},
    "A6": {"horizons": [1]},
    "A7": {"mc_samples": 1},
    "A8": {"unconstrained_actions": True},
}

ABLATION_DESCRIPTIONS = {
    "A1": "Disable data-health gating (tau_data=0.0)",
    "A2": "Disable confidence weighting (tau_conf=0.0)",
    "A3": "Skip scenario engine",
    "A4": "No attribution (top_k=0)",
    "A5": "Fixed training window (rolling_window=999999)",
    "A6": "Single-horizon forecasting (horizons=[1])",
    "A7": "No Monte Carlo sampling (mc_samples=1)",
    "A8": "Unconstrained actions (no playbook constraints)",
}


@dataclass
class AblationResult:
    ablation_id: str
    description: str
    config_override: dict[str, Any]
    # B1@h=4 metrics (primary evaluation surface)
    pagerank_mae: float = float("nan")
    hhi_mae: float = float("nan")
    density_mae: float = float("nan")
    gini_mae: float = float("nan")
    rank_correlation: float = float("nan")
    trend_consistency: float = float("nan")
    # relative drop vs C0 full model (positive = worse)
    relative_drop: float = float("nan")

    def __str__(self) -> str:
        return (
            f"AblationResult({self.ablation_id}: {self.description}, "
            f"pr_mae={self.pagerank_mae:.4f}, spearman={self.rank_correlation:.4f}, "
            f"rel_drop={self.relative_drop:+.3f})"
        )


def run_ablation(
    ablation_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    horizon: int = 4,
    **kwargs,
) -> AblationResult:
    """Run a single ablation study.

    Args:
        ablation_id: One of 'A1' through 'A8'.
        data_dir: Path to processed graph snapshots.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        horizon: Forecasting horizon for B1 evaluation (default: 4 weeks).
        **kwargs: Extra config overrides passed to the runner.

    Returns:
        AblationResult with metrics for this ablation variant.
    """
    if ablation_id not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation ID: {ablation_id}. Must be one of {list(ABLATION_CONFIGS)}")

    config_override = {**ABLATION_CONFIGS[ablation_id], **kwargs}
    description = ABLATION_DESCRIPTIONS[ablation_id]

    logger.info(
        f"Ablation {ablation_id} | {description} | "
        f"override={config_override} | test_split={test_split}"
    )
    # TODO: construct C0 (DeXposure-Agent) with config_override applied
    # TODO: run B1 at the specified horizon
    # TODO: compute relative_drop vs C0 full-model baseline stored in results/
    raise NotImplementedError(f"Ablation {ablation_id} not yet implemented")


def run_all_ablations(
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    horizon: int = 4,
    ablation_ids: list[str] | None = None,
) -> list[AblationResult]:
    """Run all (or a subset of) ablation studies A1-A8.

    Args:
        data_dir: Path to processed graph snapshots.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        horizon: Forecasting horizon for evaluation (default: 4 weeks).
        ablation_ids: Override list of ablation IDs to run. Default: A1-A8.

    Returns:
        List of AblationResult objects.
    """
    if ablation_ids is None:
        ablation_ids = list(ABLATION_CONFIGS.keys())

    logger.info(f"Running ablations: {ablation_ids}")
    results = []
    for aid in ablation_ids:
        try:
            result = run_ablation(aid, data_dir, test_split, horizon)
            results.append(result)
        except NotImplementedError as exc:
            logger.warning(f"Ablation {aid} not yet implemented: {exc}")
            results.append(
                AblationResult(
                    ablation_id=aid,
                    description=ABLATION_DESCRIPTIONS.get(aid, ""),
                    config_override=ABLATION_CONFIGS.get(aid, {}),
                )
            )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation studies A1-A8")
    parser.add_argument("--ablations", default=",".join(ABLATION_CONFIGS.keys()),
                        help="Comma-separated ablation IDs (e.g. A1,A2)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range YYYY-MM~YYYY-MM")
    parser.add_argument("--horizon", type=int, default=4,
                        help="Forecasting horizon in weeks")
    args = parser.parse_args()

    ablation_ids = [a.strip() for a in args.ablations.split(",")]
    results = run_all_ablations(
        data_dir=args.data_dir,
        test_split=args.test_split,
        horizon=args.horizon,
        ablation_ids=ablation_ids,
    )
    for r in results:
        print(r)
