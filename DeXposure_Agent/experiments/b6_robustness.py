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

from dataclasses import dataclass
from typing import Optional
from loguru import logger


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
        method_id: One of C0, C1, C3, C4, C5, C6, C7, C8, C9.
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

    logger.info(
        f"B6 | method={method_id} | test_split={test_split} | "
        f"regimes={regimes} | horizon={horizon}"
    )
    # TODO: for each regime:
    #   1. apply REGIME_CONFIGS[regime] to data loading / feature pipeline
    #   2. run method at the specified horizon
    #   3. compute B1-style metrics
    #   4. compute relative_degradation vs clean run
    raise NotImplementedError(f"B6 not yet implemented for {method_id}")


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
