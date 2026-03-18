#!/usr/bin/env python3
"""B4: Stress Test (Section 3.5 of EXPERIMENT_PLAN)

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

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
DEFAULT_TOP_K = 10  # for Target Overlap@K


@dataclass
class B4Result:
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
            f"B4Result(method={self.method}, scenario={self.scenario}, "
            f"loss_mae={self.loss_mae:.4f}, dist_count_mae={self.distressed_count_mae:.4f}, "
            f"prop_depth_mae={self.propagation_depth_mae:.4f}, "
            f"overlap@{self.top_k}={self.target_overlap_at_k:.4f})"
        )


def run_b4(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    scenarios: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    **kwargs,
) -> list[B4Result]:
    """Run B4 benchmark for a given method across all stress scenarios.

    Args:
        method_id: One of C0, C1, C2, C4, C5, C6, C7, C8, C9, C10.
        data_dir: Path to processed graph snapshots and scenario configs.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        scenarios: Override default scenarios (default: S1-S5).
        top_k: K for Target Overlap@K metric.
        **kwargs: Extra method-specific config.

    Returns:
        List of B4Result, one per scenario.
    """
    if scenarios is None:
        scenarios = SCENARIOS

    logger.info(
        f"B4 | method={method_id} | test_split={test_split} | "
        f"scenarios={scenarios} | top_k={top_k}"
    )
    # TODO: for each scenario:
    #   1. load baseline graph snapshot from test_split start
    #   2. apply scenario shock to the graph
    #   3. run method's contagion propagation / scenario engine
    #   4. compare against ground-truth simulated outcomes (oracle simulator)
    #   5. compute Loss MAE, Distressed Count MAE, Propagation Depth MAE, Target Overlap@K
    raise NotImplementedError(f"B4 not yet implemented for {method_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="B4: Stress Test benchmark")
    parser.add_argument("--method", required=True, help="Method ID (e.g. C0, C4)")
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
