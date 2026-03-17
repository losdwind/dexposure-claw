#!/usr/bin/env python3
"""B2: Early Warning Detection (Section 3.3 of EXPERIMENT_PLAN)

Evaluates the ability to issue timely risk warnings before known stress events.

Metrics:
- Lead Time (weeks)   -- how many weeks before event peak an alert is issued
- Precision@budget    -- precision when limited to `budget` alerts per window
- Recall@budget       -- recall when limited to `budget` alerts per window
- Alert Stability     -- 1 - (# alert flips / # periods), measures consistency
- F1-warning          -- harmonic mean of Precision@budget and Recall@budget

Alert budgets: {5, 10, 20} protocols per week.

Stress event windows (hardcoded from historical record):
- Terra/Luna collapse  pre=2022-03-28~2022-05-02, event=2022-05-02~2022-05-23,
                       post=2022-05-23~2022-06-20
- FTX collapse         pre=2022-09-26~2022-10-31, event=2022-10-31~2022-11-21,
                       post=2022-11-21~2022-12-19
- SVB/USDC depeg       pre=2023-02-06~2023-03-06, event=2023-03-06~2023-03-27,
                       post=2023-03-27~2023-04-24
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from loguru import logger


# Alert budgets (max protocols flagged per evaluation window)
ALERT_BUDGETS = [5, 10, 20]

# Stress event windows: name -> {phase: (start_date, end_date)}
STRESS_EVENTS = {
    "terra_luna": {
        "pre":   ("2022-03-28", "2022-05-02"),
        "event": ("2022-05-02", "2022-05-23"),
        "post":  ("2022-05-23", "2022-06-20"),
    },
    "ftx": {
        "pre":   ("2022-09-26", "2022-10-31"),
        "event": ("2022-10-31", "2022-11-21"),
        "post":  ("2022-11-21", "2022-12-19"),
    },
    "svb_usdc": {
        "pre":   ("2023-02-06", "2023-03-06"),
        "event": ("2023-03-06", "2023-03-27"),
        "post":  ("2023-03-27", "2023-04-24"),
    },
}


@dataclass
class B2Result:
    method: str
    stress_event: str               # 'terra_luna', 'ftx', 'svb_usdc'
    alert_budget: int
    lead_time_weeks: float = float("nan")
    precision_at_budget: float = float("nan")
    recall_at_budget: float = float("nan")
    alert_stability: float = float("nan")
    f1_warning: float = float("nan")
    n_alerted_protocols: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"B2Result(method={self.method}, event={self.stress_event}, "
            f"budget={self.alert_budget}, lead={self.lead_time_weeks:.1f}w, "
            f"P={self.precision_at_budget:.3f}, R={self.recall_at_budget:.3f}, "
            f"F1={self.f1_warning:.3f}, stability={self.alert_stability:.3f})"
        )


def run_b2(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    alert_budgets: list[int] | None = None,
    stress_events: dict | None = None,
    **kwargs,
) -> list[B2Result]:
    """Run B2 benchmark for a given method across all stress events and budgets.

    Args:
        method_id: One of C0, C1, C2, C3 (see APPLICABILITY).
        data_dir: Path to processed graph snapshots.
        test_split: Date range string -- NOTE: B2 also uses historical windows
                    pre-2025 (Terra/Luna, FTX, SVB); test_split gates C0 eval.
        alert_budgets: Override default budgets {5, 10, 20}.
        stress_events: Override default STRESS_EVENTS dict.
        **kwargs: Extra method-specific config.

    Returns:
        List of B2Result, one per (stress_event, alert_budget) combination.
    """
    if alert_budgets is None:
        alert_budgets = ALERT_BUDGETS
    if stress_events is None:
        stress_events = STRESS_EVENTS

    logger.info(
        f"B2 | method={method_id} | events={list(stress_events.keys())} | "
        f"budgets={alert_budgets}"
    )
    # TODO: for each stress event:
    #   1. load graph snapshots covering pre-window
    #   2. run method to generate per-protocol risk scores in pre-window
    #   3. threshold at alert_budget, measure lead time, P/R/F1, stability
    raise NotImplementedError(f"B2 not yet implemented for {method_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="B2: Early Warning benchmark")
    parser.add_argument("--method", required=True, help="Method ID (e.g. C0, C1)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range YYYY-MM~YYYY-MM")
    parser.add_argument("--budgets", default="5,10,20",
                        help="Comma-separated alert budget values")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    results = run_b2(args.method, args.data_dir, args.test_split, alert_budgets=budgets)
    for r in results:
        print(r)
