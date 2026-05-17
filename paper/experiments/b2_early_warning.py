#!/usr/bin/env python3
"""b2_warning: Early Warning Detection (Section 3.3 of EXPERIMENT_PLAN)

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

import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from experiments.exp_logger import ExpLogger


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

# b2_warning is a shared weighted-degree alert heuristic, not a model comparison.
APPLICABLE_METHODS = {"h1_weighted_degree"}

# Ground truth loss threshold: nodes that lost >30% edge weight are "stressed"
GROUND_TRUTH_LOSS_THRESHOLD = 0.30

# Baseline window length in weeks (approximately 6 months)
BASELINE_WINDOW_WEEKS = 26


@dataclass
class WarningResult:
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
            f"WarningResult(method={self.method}, event={self.stress_event}, "
            f"budget={self.alert_budget}, lead={self.lead_time_weeks:.1f}w, "
            f"P={self.precision_at_budget:.3f}, R={self.recall_at_budget:.3f}, "
            f"F1={self.f1_warning:.3f}, stability={self.alert_stability:.3f})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD string into datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def _node_weighted_degree(graph) -> dict[str, float]:
    """Compute weighted out-degree for every node in a GraphSnapshot.

    This serves as the per-node risk score used for ranking which protocols
    to include in the alert set (proxy for systemic importance / PageRank).
    """
    wd: dict[str, float] = {node_id: 0.0 for node_id in graph.nodes}
    for edge in graph.edges:
        if edge.source in wd:
            wd[edge.source] += edge.weight
        # Also count in-degree weight for a more comprehensive risk measure
        if edge.target in wd:
            wd[edge.target] += edge.weight
    return wd


def _compute_ground_truth_stressed(
    pre_window_end_snap,
    event_snapshots: list,
) -> set[str]:
    """Identify protocols that lost >30% total edge weight during event vs pre-window end.

    A node is "stressed" if its weighted degree at the worst point during the
    event window dropped by more than GROUND_TRUTH_LOSS_THRESHOLD relative to
    its weighted degree at the end of the pre-window.

    Returns:
        Set of node IDs that are ground-truth stressed.
    """
    wd_pre = _node_weighted_degree(pre_window_end_snap)
    if not wd_pre:
        return set()

    # Compute minimum weighted degree during the event window for each node
    wd_event_min: dict[str, float] = {}
    for snap in event_snapshots:
        wd_event = _node_weighted_degree(snap)
        for node_id in wd_pre:
            event_val = wd_event.get(node_id, 0.0)
            if node_id not in wd_event_min:
                wd_event_min[node_id] = event_val
            else:
                wd_event_min[node_id] = min(wd_event_min[node_id], event_val)

    stressed: set[str] = set()
    for node_id, pre_val in wd_pre.items():
        if pre_val <= 0.0:
            continue
        event_val = wd_event_min.get(node_id, 0.0)
        loss_frac = (pre_val - event_val) / pre_val
        if loss_frac > GROUND_TRUTH_LOSS_THRESHOLD:
            stressed.add(node_id)

    return stressed


def _rank_nodes_by_risk(graph, baseline_history: list[dict[str, float]],
                        budget: int) -> set[str]:
    """Rank nodes by weighted degree and return the top-budget node IDs as alerts.

    Uses a simple strategy: compute weighted degree for each node in the current
    graph, rank descending, and flag the top `budget` nodes.
    """
    wd = _node_weighted_degree(graph)
    if not wd:
        return set()
    # Sort by weighted degree descending (higher = more systemic importance = more risk)
    ranked = sorted(wd.items(), key=lambda x: x[1], reverse=True)
    return {node_id for node_id, _ in ranked[:budget]}


def _compute_lead_time(
    pre_snapshots: list,
    event_start_date: str,
    ground_truth_stressed: set[str],
    budget: int,
    baseline_history: list[dict[str, float]],
) -> float:
    """Compute lead time: weeks before event_start that the first alert fires
    for a truly-stressed protocol.

    Iterates through pre-window snapshots in chronological order. For each,
    rank nodes and check if any alerted node is in ground_truth_stressed.
    The lead time is the time from the first such alert to the event start.

    Returns:
        Lead time in weeks, or 0.0 if no alert fires before event.
    """
    event_start = _parse_date(event_start_date)
    first_alert_date: datetime | None = None

    for snap in pre_snapshots:
        alerted = _rank_nodes_by_risk(snap, baseline_history, budget)
        overlap = alerted & ground_truth_stressed
        if overlap:
            snap_date = _parse_date(snap.date)
            if first_alert_date is None or snap_date < first_alert_date:
                first_alert_date = snap_date
            break  # First chronological alert found

    if first_alert_date is None:
        return 0.0

    delta = event_start - first_alert_date
    return max(0.0, delta.days / 7.0)


def _compute_alert_stability(
    pre_snapshots: list,
    budget: int,
    baseline_history: list[dict[str, float]],
) -> float:
    """Compute alert stability: 1 - (number of flips / number of periods).

    A "flip" occurs when a node enters or leaves the alert set between
    consecutive periods. Stability of 1.0 means the alert set never changes.

    Returns:
        Stability score in [0, 1].
    """
    if len(pre_snapshots) < 2:
        return 1.0

    total_flips = 0
    prev_alerted: set[str] | None = None

    for snap in pre_snapshots:
        current_alerted = _rank_nodes_by_risk(snap, baseline_history, budget)
        if prev_alerted is not None:
            # Flips = symmetric difference (nodes that entered or left)
            flips = len(current_alerted.symmetric_difference(prev_alerted))
            total_flips += flips
        prev_alerted = current_alerted

    n_transitions = len(pre_snapshots) - 1
    # Normalize by max possible flips per transition (2 * budget)
    max_flips_per_transition = 2 * budget
    max_total_flips = max_flips_per_transition * n_transitions

    if max_total_flips == 0:
        return 1.0

    stability = 1.0 - (total_flips / max_total_flips)
    return max(0.0, min(1.0, stability))


def _f1_score(precision: float, recall: float) -> float:
    """Compute harmonic mean of precision and recall."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_b2(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    alert_budgets: list[int] | None = None,
    stress_events: dict | None = None,
    **kwargs,
) -> list[WarningResult]:
    """Run b2_warning benchmark for a given method across all stress events and budgets.

    Args:
        method_id: h1_weighted_degree. b2_warning is method-agnostic and should
                   not be reported as a model-vs-model comparison.
        data_dir: Path to processed graph snapshots.
        test_split: Date range string -- NOTE: b2_warning also uses historical
                    windows pre-2025 (Terra/Luna, FTX, SVB); test_split gates
                    FM-method evaluation.
        alert_budgets: Override default budgets {5, 10, 20}.
        stress_events: Override default STRESS_EVENTS dict.
        **kwargs: Extra method-specific config.

    Returns:
        List of WarningResult, one per (stress_event, alert_budget) combination.
    """
    if alert_budgets is None:
        alert_budgets = ALERT_BUDGETS
    if stress_events is None:
        stress_events = STRESS_EVENTS

    results_dir = kwargs.pop("results_dir", "results/")
    log = ExpLogger("b2_warning", method=method_id, results_dir=results_dir)

    log.info(
        f"b2_warning |method={method_id} | events={list(stress_events.keys())} | "
        f"budgets={alert_budgets}"
    )

    # Validate method applicability
    if method_id not in APPLICABLE_METHODS:
        raise ValueError(
            "b2_warning currently evaluates a shared weighted-degree heuristic, "
            "not a model-specific method. Use method_id='h1_weighted_degree'. "
            f"Got {method_id!r}."
        )

    # Ensure repo root is on sys.path for imports
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from dexposure_agent.data_loader import SnapshotLoader
    loader = SnapshotLoader(data_dir=data_dir)
    results: list[WarningResult] = []

    for event_name, windows in log.progress(
        stress_events.items(), desc="Stress events", total=len(stress_events), unit="event"
    ):
        pre_start, pre_end = windows["pre"]
        event_start, event_end = windows["event"]

        log.info(
            f"b2_warning |{event_name} | pre={pre_start}~{pre_end} | "
            f"event={event_start}~{event_end}"
        )

        # --- Load pre-window snapshots ---
        pre_snapshots = loader.load(start=pre_start, end=pre_end)
        if not pre_snapshots:
            log.warning(
                f"b2_warning |{event_name} | No snapshots in pre-window "
                f"{pre_start}~{pre_end}, skipping"
            )
            for budget in alert_budgets:
                results.append(WarningResult(
                    method=method_id,
                    stress_event=event_name,
                    alert_budget=budget,
                ))
            continue

        log.info(
            f"b2_warning |{event_name} | Loaded {len(pre_snapshots)} pre-window snapshots"
        )

        # --- Load event-window snapshots (for ground truth) ---
        event_snapshots = loader.load(start=event_start, end=event_end)
        if not event_snapshots:
            log.warning(
                f"b2_warning |{event_name} | No snapshots in event-window "
                f"{event_start}~{event_end}, skipping ground truth"
            )
            for budget in alert_budgets:
                results.append(WarningResult(
                    method=method_id,
                    stress_event=event_name,
                    alert_budget=budget,
                ))
            continue

        log.info(
            f"b2_warning |{event_name} | Loaded {len(event_snapshots)} event-window snapshots"
        )

        # --- Build rolling baseline from snapshots before the pre-window ---
        # Use up to BASELINE_WINDOW_WEEKS snapshots ending just before pre_start
        baseline_history = loader.build_baseline(
            before=pre_start, window=BASELINE_WINDOW_WEEKS
        )
        log.info(
            f"b2_warning |{event_name} | Baseline: {len(baseline_history)} snapshots "
            f"before {pre_start}"
        )

        # --- Determine ground truth stressed protocols ---
        # Compare pre-window end snapshot to worst point in event window
        pre_window_end_snap = pre_snapshots[-1]
        ground_truth_stressed = _compute_ground_truth_stressed(
            pre_window_end_snap, event_snapshots
        )
        log.info(
            f"b2_warning |{event_name} | Ground truth: {len(ground_truth_stressed)} "
            f"stressed protocols (>{GROUND_TRUTH_LOSS_THRESHOLD:.0%} loss)"
        )

        if not ground_truth_stressed:
            log.warning(
                f"b2_warning |{event_name} | No protocols exceeded "
                f"{GROUND_TRUTH_LOSS_THRESHOLD:.0%} loss threshold; "
                f"recall will be undefined"
            )

        # --- Evaluate each alert budget ---
        for budget in log.progress(
            alert_budgets, desc=f"{event_name} budgets", total=len(alert_budgets), unit="budget"
        ):
            log.info(f"b2_warning |{event_name} | budget={budget}")

            # Collect per-snapshot alert sets for the pre-window
            per_snapshot_alerted: list[set[str]] = []
            for snap in pre_snapshots:
                alerted = _rank_nodes_by_risk(snap, baseline_history, budget)
                per_snapshot_alerted.append(alerted)

            # Union of all alerted protocols across pre-window
            all_alerted = set()
            for a in per_snapshot_alerted:
                all_alerted.update(a)

            # Use the final pre-window snapshot's alert set for P/R/F1
            final_alerted = per_snapshot_alerted[-1] if per_snapshot_alerted else set()

            # --- Precision@budget ---
            if final_alerted:
                true_positives = final_alerted & ground_truth_stressed
                precision = len(true_positives) / len(final_alerted)
            else:
                precision = 0.0

            # --- Recall@budget ---
            if ground_truth_stressed:
                # Use union of all alerts across pre-window for recall
                # (a protocol counts as recalled if it was ever alerted)
                recall_positives = all_alerted & ground_truth_stressed
                recall = len(recall_positives) / len(ground_truth_stressed)
            else:
                recall = 0.0

            # --- F1-warning ---
            f1 = _f1_score(precision, recall)

            # --- Lead time ---
            lead_time = _compute_lead_time(
                pre_snapshots, event_start,
                ground_truth_stressed, budget, baseline_history,
            )

            # --- Alert stability ---
            stability = _compute_alert_stability(
                pre_snapshots, budget, baseline_history,
            )

            result = WarningResult(
                method=method_id,
                stress_event=event_name,
                alert_budget=budget,
                lead_time_weeks=lead_time,
                precision_at_budget=precision,
                recall_at_budget=recall,
                alert_stability=stability,
                f1_warning=f1,
                n_alerted_protocols=len(all_alerted),
            )
            log.step(
                f"{event_name}/budget={budget}",
                lead_time=lead_time, precision=precision, recall=recall,
                f1=f1, stability=stability,
            )
            results.append(result)

    # --- Summary ---
    def _safe_mean(vals):
        valid = [v for v in vals if not math.isnan(v)]
        return sum(valid) / len(valid) if valid else float("nan")

    if results:
        avg_f1 = _safe_mean([r.f1_warning for r in results])
        avg_lead = _safe_mean([r.lead_time_weeks for r in results])
        avg_precision = _safe_mean([r.precision_at_budget for r in results])
        avg_recall = _safe_mean([r.recall_at_budget for r in results])
        avg_stability = _safe_mean([r.alert_stability for r in results])
        log.summary({
            "n_results": len(results),
            "mean_f1_warning": avg_f1,
            "mean_lead_time_weeks": avg_lead,
            "mean_precision": avg_precision,
            "mean_recall": avg_recall,
            "mean_stability": avg_stability,
        })
    else:
        log.summary({"n_results": 0})

    log.save_results([
        {
            "method": r.method, "stress_event": r.stress_event,
            "alert_budget": r.alert_budget, "lead_time_weeks": r.lead_time_weeks,
            "precision": r.precision_at_budget, "recall": r.recall_at_budget,
            "f1_warning": r.f1_warning, "stability": r.alert_stability,
            "n_alerted": r.n_alerted_protocols,
        }
        for r in results
    ])

    log.info(f"b2_warning |Completed: {len(results)} results for method={method_id}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="b2_warning: Early Warning benchmark")
    parser.add_argument(
        "--method",
        default="h1_weighted_degree",
        help="Use h1_weighted_degree for the shared heuristic",
    )
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
