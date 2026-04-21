#!/usr/bin/env python3
"""A1-isolated: Test data-health gate with confidence gate DISABLED.

The original A1 test showed no effect because:
1. tau_conf=0.6 blocks interventions before data-health gate matters
2. DH_t never drops below 0.7 with standard degradation

This script isolates A1 by:
- Setting tau_conf=0.0 (confidence gate off) so interventions CAN happen
- Using extreme degradation (95% edges + 90% features) to trigger safe_mode
- Testing on BOTH calm (2025) and crisis (2022) periods

Expected result: with tau_data=0.7, safe_mode blocks garbage interventions.
With tau_data=0.0, garbage data produces false interventions -> FIR rises.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
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
from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures

DATA_DIR = str(_REPO_ROOT / "DeXposure" / "data")
RESULTS_DIR = str(_AGENT_DIR / "results" / "run_a1_isolated")
STRESS_LOOKAHEAD = 4
MC_NOISE_SIGMA = 2.0  # from calibration


def _mask_edges(graph, mask_frac, rng):
    if mask_frac <= 0 or not graph.edges:
        return graph
    n_keep = max(1, int(len(graph.edges) * (1.0 - mask_frac)))
    idx = rng.choice(len(graph.edges), size=n_keep, replace=False)
    return GraphSnapshot(date=graph.date, nodes=graph.nodes, edges=[graph.edges[i] for i in sorted(idx)])


def _drop_features(graph, drop_frac, rng):
    if drop_frac <= 0:
        return graph
    new_nodes = {}
    for nid, f in graph.nodes.items():
        vals = {"log_size": f.log_size, "num_tokens": f.num_tokens,
                "max_share": f.max_share, "entropy": f.entropy, "category": f.category}
        for attr in ["log_size", "num_tokens", "max_share", "entropy"]:
            if rng.random() < drop_frac:
                vals[attr] = 0 if attr == "num_tokens" else 0.0
        new_nodes[nid] = NodeFeatures(**vals)
    return GraphSnapshot(date=graph.date, nodes=new_nodes, edges=graph.edges)


def _mc_samples(graph, n, sigma, rng):
    samples = []
    for _ in range(n):
        edges = []
        for e in graph.edges:
            noise = rng.normal(0, sigma * abs(e.weight)) if e.weight else 0
            edges.append(Edge(source=e.source, target=e.target, weight=max(0, e.weight + noise)))
        samples.append(GraphSnapshot(date=graph.date, nodes=graph.nodes, edges=edges))
    return samples


def _node_weights(g):
    t = {n: 0.0 for n in g.nodes}
    for e in g.edges:
        t[e.source] = t.get(e.source, 0) + e.weight
        t[e.target] = t.get(e.target, 0) + e.weight
    return t


def _stressed(snap_t, snap_f, thresh=0.20):
    wt, wf = _node_weights(snap_t), _node_weights(snap_f)
    return {n for n, w in wt.items() if w > 0 and (w - wf.get(n, 0)) / w > thresh}


# Degradation regimes that WILL trigger safe_mode
REGIMES = {
    "extreme":       {"feature_drop": 0.90, "edge_mask": 0.95},
    "extreme_topo":  {"feature_drop": 0.00, "edge_mask": 0.98},
    "extreme_feat":  {"feature_drop": 0.95, "edge_mask": 0.00},
    "severe":        {"feature_drop": 0.80, "edge_mask": 0.85},
}

# Test conditions
CONDITIONS = [
    # (tau_data, tau_conf, label)
    (0.85, 0.0, "gate_on_strict"),   # data-health gate ON (strict), confidence OFF
    (0.7,  0.0, "gate_on"),          # data-health gate ON (default), confidence OFF
    (0.0,  0.0, "gate_off"),         # both gates OFF
]


@dataclass
class Result:
    regime: str
    label: str
    tau_data: float
    tau_conf: float
    test_split: str
    n_weeks: int = 0
    mean_dh: float = 0.0
    safe_mode_pct: float = 0.0
    prec: float = 0.0
    fir: float = 0.0
    n_tickets: int = 0
    n_intervention: int = 0


def run_one(regime_name, regime_cfg, tau_data, tau_conf, label, test_split, loader, all_dates, date_to_snap, rng):
    config = AgentConfig(tau_data=tau_data, tau_conf=tau_conf)
    from experiments.predict_helper import predict_graph

    r = Result(regime=regime_name, label=label, tau_data=tau_data, tau_conf=tau_conf, test_split=test_split)
    dh_scores, safe_modes, ticket_ok, false_intv = [], [], [], []

    test_pairs = list(loader.iter_test_with_baselines(test_split, baseline_window=config.rolling_window))
    for snap_t, baseline_history in test_pairs:
        t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
        if t_idx < 0: continue
        fi = t_idx + STRESS_LOOKAHEAD
        if fi >= len(all_dates): continue
        gt = date_to_snap.get(all_dates[fi])
        if gt is None: continue

        deg = _drop_features(_mask_edges(snap_t, regime_cfg["edge_mask"], rng), regime_cfg["feature_drop"], rng)
        dh = compute_data_health(deg, config)
        dh_scores.append(dh.score)
        safe_modes.append(dh.safe_mode)

        pred = predict_graph("C0", deg, horizon=STRESS_LOOKAHEAD)
        metrics = compute_metrics(pred)
        baseline = _compute_rolling_baseline(baseline_history, config.rolling_window)
        alerts = detect_alerts(metrics, baseline, horizon=STRESS_LOOKAHEAD, config=config)

        mc = _mc_samples(pred, config.mc_samples, MC_NOISE_SIGMA, rng)
        scen = run_scenarios(pred, mc, config, horizon=STRESS_LOOKAHEAD)
        decision = generate_tickets(alerts, scen, dh, config)

        targets, intv_targets = set(), set()
        for t in decision.tickets:
            targets.update(t.targets)
            r.n_tickets += 1
            if t.action in ("Recommend-Reduce", "Contingency"):
                intv_targets.update(t.targets)
                r.n_intervention += 1

        stressed = _stressed(snap_t, gt)
        for t in targets: ticket_ok.append(t in stressed)
        stable = set(snap_t.nodes.keys()) - stressed
        for t in intv_targets: false_intv.append(t in stable)
        r.n_weeks += 1

    r.mean_dh = float(np.mean(dh_scores)) if dh_scores else 0
    r.safe_mode_pct = float(np.mean(safe_modes)) if safe_modes else 0
    r.prec = float(np.mean(ticket_ok)) if ticket_ok else 0
    r.fir = float(np.mean(false_intv)) if false_intv else 0
    return r


def main():
    logger.info("=" * 60)
    logger.info("A1-ISOLATED: Data-health gate with confidence gate DISABLED")
    logger.info("=" * 60)

    loader = SnapshotLoader(data_dir=DATA_DIR)
    all_dates = loader.dates
    all_snaps = loader.load()
    date_to_snap = {s.date: s for s in all_snaps}
    rng = np.random.default_rng(42)

    test_splits = [
        "2025-01~2025-08",   # calm period
        "2022-04~2022-07",   # Terra/Luna crisis
    ]

    results = []
    for split in test_splits:
        for regime_name, regime_cfg in REGIMES.items():
            for tau_data, tau_conf, label in CONDITIONS:
                logger.info(f"Running: {regime_name} | {label} | split={split}")
                t0 = time.time()
                r = run_one(regime_name, regime_cfg, tau_data, tau_conf, label, split, loader, all_dates, date_to_snap, rng)
                elapsed = time.time() - t0
                logger.info(
                    f"  {regime_name}/{label}/{split}: DH={r.mean_dh:.3f} safe={r.safe_mode_pct:.0%} "
                    f"prec={r.prec:.3f} FIR={r.fir:.3f} #intv={r.n_intervention} ({elapsed:.0f}s)"
                )
                results.append(r)

    # Save
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    (out_dir / "a1_isolated.json").write_text(json.dumps(data, indent=2))

    # Print summary table
    print("\n=== A1-ISOLATED RESULTS ===")
    print(f"{'Regime':<16} {'Label':<18} {'Split':<20} {'DH':>5} {'Safe%':>5} {'Prec':>5} {'FIR':>5} {'#Intv':>5}")
    print("-" * 85)
    for r in results:
        print(f"{r.regime:<16} {r.label:<18} {r.test_split:<20} {r.mean_dh:>5.3f} {r.safe_mode_pct:>5.0%} {r.prec:>5.3f} {r.fir:>5.3f} {r.n_intervention:>5}")

    logger.info("ALL DONE")


if __name__ == "__main__":
    main()
