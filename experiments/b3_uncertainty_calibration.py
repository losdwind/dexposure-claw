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

from dataclasses import dataclass
from typing import Optional
from loguru import logger


TARGET_COVERAGE = 0.90  # nominal coverage of prediction intervals


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
    logger.info(
        f"B3 | method={method_id} | test_split={test_split} | "
        f"target_coverage={target_coverage}"
    )
    # TODO: load test snapshots
    # TODO: run method to get predictive distributions (mean + std or samples)
    # TODO: compute ECE via reliability diagram binning
    # TODO: compute PI coverage and width at target_coverage level
    # TODO: compute CRPS via energy score or analytical formula
    raise NotImplementedError(f"B3 not yet implemented for {method_id}")


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
