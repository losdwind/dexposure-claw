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

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


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


def run_b1(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    horizons: list[int] | None = None,
    **kwargs,
) -> list[B1Result]:
    """Run B1 benchmark for a given method across all horizons.

    Args:
        method_id: One of C0-C10.
        data_dir: Path to processed graph snapshots.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        horizons: Override default horizons (default: [1, 4, 8, 12]).
        **kwargs: Extra method-specific config passed through.

    Returns:
        List of B1Result, one per horizon.
    """
    if horizons is None:
        horizons = HORIZONS

    logger.info(f"B1 | method={method_id} | test_split={test_split} | horizons={horizons}")
    # TODO: load test snapshots from data_dir filtered by test_split
    # TODO: run method_id forward pass to get predicted risk metrics
    # TODO: compute MAE, Spearman rho, trend consistency per horizon
    raise NotImplementedError(f"B1 not yet implemented for {method_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="B1: Risk Forecasting benchmark")
    parser.add_argument("--method", required=True, help="Method ID (e.g. C0, C4)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range YYYY-MM~YYYY-MM")
    parser.add_argument("--horizons", default="1,4,8,12",
                        help="Comma-separated horizon values in weeks")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    results = run_b1(args.method, args.data_dir, args.test_split, horizons=horizons)
    for r in results:
        print(r)
