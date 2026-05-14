#!/usr/bin/env python3
"""Alert Timeline Analysis: WHEN do FM vs Persistence alerts fire relative to crises?

For each crisis event, outputs per-week:
- Date, phase (pre/during/post), n_alerts, n_true_positive, precision
- Cumulative alerts before event date = "early warning score"

This answers: Is the FM a forecaster or a historian?

Usage:
    DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 python3 scripts/run_alert_timeline.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from loguru import logger

_SCRIPT_DIR = Path(__file__).resolve().parent
_AGENT_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _AGENT_DIR.parent
sys.path.insert(0, str(_AGENT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from dexposure_agent.config import AgentConfig
from dexposure_agent.data_health import compute_data_health
from dexposure_agent.data_loader import SnapshotLoader
from dexposure_agent.decision import generate_tickets
from dexposure_agent.monitor import compute_metrics, detect_alerts, _compute_rolling_baseline
from dexposure_agent.scenario import run_scenarios
from dexposure_agent.types import Edge, GraphSnapshot
from dexposure_agent.agent_loop import _aggregate_scenarios
from experiments.predict_helper import predict_graph

DATA_DIR = str(_REPO_ROOT / "data")
RESULTS_DIR = str(_AGENT_DIR / "results" / "run_alert_timeline")
MC_NOISE_SIGMA = 2.0
HORIZONS = [1, 4, 8, 12]

# Crisis event dates (the actual crash/peak date)
CRISIS_EVENTS = {
    "terra_luna": {
        "event_date": "2022-05-09",
        "test_split": "2022-03~2022-08",  # wider window to capture pre/post
        "description": "Terra/Luna collapse",
    },
    "ftx": {
        "event_date": "2022-11-07",
        "test_split": "2022-09~2023-02",
        "description": "FTX collapse",
    },
}

METHODS = ["m5_fm_rules", "m1_persistence_rules"]


def _node_weights(g):
    t = {n: 0.0 for n in g.nodes}
    for e in g.edges:
        t[e.source] = t.get(e.source, 0) + e.weight
        t[e.target] = t.get(e.target, 0) + e.weight
    return t


def _stressed(snap_t, snap_f, thresh=0.20):
    wt, wf = _node_weights(snap_t), _node_weights(snap_f)
    return {n for n, w in wt.items() if w > 0 and (w - wf.get(n, 0)) / w > thresh}


def _mc_samples(graph, n, sigma, rng):
    samples = []
    for _ in range(n):
        edges = []
        for e in graph.edges:
            noise = rng.normal(0, sigma * abs(e.weight)) if e.weight else 0
            edges.append(Edge(source=e.source, target=e.target, weight=max(0, e.weight + noise)))
        samples.append(GraphSnapshot(date=graph.date, nodes=graph.nodes, edges=edges))
    return samples


def classify_phase(week_date: str, event_date: str) -> str:
    """Classify a week as pre-event, event-week, or post-event."""
    wd = datetime.fromisoformat(week_date)
    ed = datetime.fromisoformat(event_date)
    delta_days = (wd - ed).days
    if delta_days < -7:
        return "pre"
    elif delta_days <= 21:  # event week + 3 weeks aftermath
        return "during"
    else:
        return "post"


def weeks_before_event(week_date: str, event_date: str) -> float:
    """How many weeks before the event date. Negative = after event."""
    wd = datetime.fromisoformat(week_date)
    ed = datetime.fromisoformat(event_date)
    return -(wd - ed).days / 7.0


@dataclass
class WeekResult:
    date: str
    phase: str  # pre / during / post
    weeks_to_event: float
    method: str
    n_alerts: int = 0
    alerts_per_horizon: dict[int, int] = field(default_factory=dict)
    n_true_positive_targets: int = 0
    n_total_targets: int = 0
    precision: float = float("nan")
    truly_stressed_protocols: int = 0
    flagged_protocols: list[str] = field(default_factory=list)
    ticket_actions: list[str] = field(default_factory=list)


def run_timeline(crisis_name: str, crisis_info: dict) -> list[WeekResult]:
    """Run per-week alert analysis for one crisis event."""
    event_date = crisis_info["event_date"]
    test_split = crisis_info["test_split"]

    loader = SnapshotLoader(data_dir=DATA_DIR)
    all_dates = loader.dates
    all_snaps = loader.load()
    date_to_snap = {s.date: s for s in all_snaps}
    rng = np.random.default_rng(42)
    config = AgentConfig(horizons=HORIZONS)

    results = []

    test_pairs = list(loader.iter_test_with_baselines(test_split, baseline_window=config.rolling_window))
    logger.info(f"Timeline {crisis_name}: {len(test_pairs)} weeks, event={event_date}")

    for method in METHODS:
        logger.info(f"  Method: {method}")
        rng = np.random.default_rng(42)  # reset per method for consistency

        for snap_t, baseline_history in test_pairs:
            t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
            if t_idx < 0:
                continue
            # Use h=4 for ground truth lookahead
            future_idx = t_idx + 4
            if future_idx >= len(all_dates):
                continue
            gt = date_to_snap.get(all_dates[future_idx])
            if gt is None:
                continue

            phase = classify_phase(snap_t.date, event_date)
            wte = weeks_before_event(snap_t.date, event_date)

            dh = compute_data_health(snap_t, config)

            # Multi-horizon alert aggregation
            all_alerts = []
            all_scenario_losses = []
            alerts_per_h = {h: 0 for h in HORIZONS}

            for h in HORIZONS:
                pred = predict_graph(method, snap_t, horizon=h)
                metrics = compute_metrics(pred)
                baseline = _compute_rolling_baseline(baseline_history, config.rolling_window)
                alerts_h = detect_alerts(metrics, baseline, horizon=h, config=config)
                all_alerts.extend(alerts_h)
                alerts_per_h[h] += len(alerts_h)

                mc = _mc_samples(pred, config.mc_samples, MC_NOISE_SIGMA, rng)
                scen = run_scenarios(pred, mc, config, horizon=h)
                all_scenario_losses.extend(scen.ranked_losses)

            scenario_summary = _aggregate_scenarios(all_scenario_losses)
            decision = generate_tickets(all_alerts, scenario_summary, dh, config)

            # Collect targets
            targets = set()
            actions = []
            for ticket in decision.tickets:
                targets.update(ticket.targets)
                actions.append(ticket.action)

            # Ground truth
            stressed = _stressed(snap_t, gt)
            tp = len(targets & stressed)

            wr = WeekResult(
                date=snap_t.date,
                phase=phase,
                weeks_to_event=round(wte, 1),
                method=method,
                n_alerts=len(all_alerts),
                alerts_per_horizon=alerts_per_h,
                n_true_positive_targets=tp,
                n_total_targets=len(targets),
                precision=tp / len(targets) if targets else float("nan"),
                truly_stressed_protocols=len(stressed),
                flagged_protocols=sorted(targets)[:10],
                ticket_actions=actions,
            )
            results.append(wr)

            flag = "***" if phase == "pre" and len(all_alerts) > 0 else ""
            logger.info(
                f"    {snap_t.date} [{phase:6s}] wte={wte:+5.1f}w | "
                f"alerts={len(all_alerts):2d} targets={len(targets):2d} "
                f"TP={tp:2d} stressed={len(stressed):3d} "
                f"prec={wr.precision:.2f} {flag}"
            )

    return results


def main():
    logger.info("=" * 65)
    logger.info("ALERT TIMELINE ANALYSIS: When do alerts fire vs crisis events?")
    logger.info("=" * 65)

    all_results = {}

    for crisis_name, crisis_info in CRISIS_EVENTS.items():
        logger.info(f"\n{'='*65}")
        logger.info(f"Crisis: {crisis_name} ({crisis_info['description']})")
        logger.info(f"Event date: {crisis_info['event_date']}")
        logger.info(f"{'='*65}")

        results = run_timeline(crisis_name, crisis_info)
        all_results[crisis_name] = results

    # Save
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for crisis_name, results in all_results.items():
        data = [asdict(r) for r in results]
        (out_dir / f"timeline_{crisis_name}.json").write_text(
            json.dumps(data, indent=2, default=str)
        )

    # Print summary tables
    for crisis_name, results in all_results.items():
        event_date = CRISIS_EVENTS[crisis_name]["event_date"]
        print(f"\n{'='*75}")
        print(f"TIMELINE: {crisis_name} (event={event_date})")
        print(f"{'='*75}")

        for method in METHODS:
            method_results = [r for r in results if r.method == method]
            pre = [r for r in method_results if r.phase == "pre"]
            during = [r for r in method_results if r.phase == "during"]
            post = [r for r in method_results if r.phase == "post"]

            pre_alerts = sum(r.n_alerts for r in pre)
            pre_tp = sum(r.n_true_positive_targets for r in pre)
            pre_total = sum(r.n_total_targets for r in pre)
            during_alerts = sum(r.n_alerts for r in during)
            post_alerts = sum(r.n_alerts for r in post)
            total_alerts = pre_alerts + during_alerts + post_alerts

            # First alert week
            first_alert = next((r for r in method_results if r.n_alerts > 0), None)
            first_tp = next((r for r in method_results if r.n_true_positive_targets > 0), None)

            print(f"\n  {method}:")
            print(f"    Pre-event alerts:    {pre_alerts:3d} / {total_alerts} ({pre_alerts/max(total_alerts,1)*100:.0f}%)")
            print(f"    During-event alerts: {during_alerts:3d} / {total_alerts} ({during_alerts/max(total_alerts,1)*100:.0f}%)")
            print(f"    Post-event alerts:   {post_alerts:3d} / {total_alerts} ({post_alerts/max(total_alerts,1)*100:.0f}%)")
            print(f"    Pre-event TP targets: {pre_tp} / {pre_total}")
            if first_alert:
                print(f"    First alert:     {first_alert.date} ({first_alert.weeks_to_event:+.1f} weeks)")
            if first_tp:
                print(f"    First TP alert:  {first_tp.date} ({first_tp.weeks_to_event:+.1f} weeks)")

            print(f"\n    {'Date':<12} {'Phase':<8} {'Wks':>5} {'Alerts':>6} {'Targets':>7} {'TP':>4} {'Stressed':>8} {'Prec':>6}")
            print(f"    {'-'*62}")
            for r in method_results:
                marker = " <-- EVENT" if abs(r.weeks_to_event) < 1.5 else ""
                print(f"    {r.date:<12} {r.phase:<8} {r.weeks_to_event:>+5.1f} {r.n_alerts:>6} "
                      f"{r.n_total_targets:>7} {r.n_true_positive_targets:>4} {r.truly_stressed_protocols:>8} "
                      f"{r.precision:>6.2f}{marker}")

    logger.info("\nALL DONE")


if __name__ == "__main__":
    main()
