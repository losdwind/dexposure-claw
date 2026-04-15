#!/usr/bin/env python3
"""Run a single autoresearch trial: evaluate one AgentConfig on the validation split.

Runs B1 (rank correlation @ h=4), B2 (F1-warning), and B5 (ticket precision)
on the validation period (2024-07 ~ 2024-12).  Outputs a single composite
score and per-benchmark breakdown in a deterministic format that the
autoresearch loop can parse.

Usage (on GPU server):
    python3 run_trial.py --config '{"pi_min": 0.3, "z_threshold": 1.5}'
    python3 run_trial.py --config-file trial_config.json
    python3 run_trial.py   # runs with default AgentConfig
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from loguru import logger

# Ensure repo root importable
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dexposure_agent.config import AgentConfig

# ---------------------------------------------------------------------------
# Validation split (strict: no overlap with test 2025-01 ~ 2025-08)
# ---------------------------------------------------------------------------
VAL_SPLIT = "2024-07~2024-12"
DATA_DIR = "/root/graph-dexposure/DeXposure/data/"
RESULTS_DIR = "/tmp/autoresearch_trial/"
METHOD = "C0"


# ---------------------------------------------------------------------------
# B1: Rank Correlation @ h=4 (primary forecasting quality)
# ---------------------------------------------------------------------------
def eval_b1(config: AgentConfig) -> dict:
    """Run B1 on validation split, return mean Spearman rho and trend consistency."""
    from experiments import b1_risk_forecasting
    from experiments.b1_risk_forecasting import run_b1

    # Reset the FM predictor singleton so it picks up the trial's pi_min
    b1_risk_forecasting._FM_PREDICTOR = None
    from dexposure_agent.fm_predictor import FMPredictor
    b1_risk_forecasting._FM_PREDICTOR = FMPredictor(pi_min=config.pi_min)

    results = run_b1(
        method_id=METHOD,
        data_dir=DATA_DIR,
        test_split=VAL_SPLIT,
        horizons=[4],  # single horizon for speed
        results_dir=RESULTS_DIR,
    )
    if not results:
        return {"rank_corr": 0.0, "trend": 0.0}

    r = results[0]
    return {
        "rank_corr": float(r.rank_correlation) if not np.isnan(r.rank_correlation) else 0.0,
        "trend": float(r.trend_consistency) if not np.isnan(r.trend_consistency) else 0.0,
    }


# ---------------------------------------------------------------------------
# B2: F1-Warning (early warning quality)
# ---------------------------------------------------------------------------
def eval_b2(config: AgentConfig) -> dict:
    """Run B2 early warning, return mean F1 score across events.

    NOTE: B2 evaluates on fixed historical stress events (Terra/Luna, FTX, SVB)
    with its own hardcoded baseline window.  It does NOT use the val split or
    most AgentConfig params.  Include it for completeness but it won't vary
    much across configs.
    """
    from experiments.b2_early_warning import run_b2

    results = run_b2(
        method_id=METHOD,
        data_dir=DATA_DIR,
        test_split=VAL_SPLIT,
        results_dir=RESULTS_DIR,
    )
    if not results:
        return {"f1_warning": 0.0}

    f1_scores = [r.f1_warning for r in results if not np.isnan(r.f1_warning)]
    return {"f1_warning": float(np.mean(f1_scores)) if f1_scores else 0.0}


# ---------------------------------------------------------------------------
# B5: Ticket Precision (decision quality)
# ---------------------------------------------------------------------------
def eval_b5(config: AgentConfig) -> dict:
    """Run B5 on validation split, return ticket precision."""
    from experiments.b5_decision_quality import run_b5

    results = run_b5(
        method_id=METHOD,
        data_dir=DATA_DIR,
        test_split=VAL_SPLIT,
        results_dir=RESULTS_DIR,
        **{k: v for k, v in config.model_dump().items() if k in AgentConfig.model_fields},
    )
    if not results:
        return {"ticket_prec": 0.0}

    r = results[0]
    prec = r.ticket_precision if not np.isnan(r.ticket_precision) else 0.0
    return {"ticket_prec": float(prec)}


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------
def compute_composite(metrics: dict) -> float:
    """Weighted composite: 0.4*rank_corr + 0.2*trend + 0.2*f1_warning + 0.2*ticket_prec.

    All components in [0, 1] range. Higher is better.
    """
    return (
        0.4 * metrics.get("rank_corr", 0.0)
        + 0.2 * metrics.get("trend", 0.0)
        + 0.2 * metrics.get("f1_warning", 0.0)
        + 0.2 * metrics.get("ticket_prec", 0.0)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run one autoresearch trial")
    parser.add_argument("--config", type=str, default="{}", help="JSON config overrides")
    parser.add_argument("--config-file", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--benchmarks", type=str, default="B1,B5",
                        help="Comma-separated benchmarks to run (B1,B2,B5). "
                             "Default: B1,B5 (B2 uses fixed historical events, "
                             "not val split, so less useful for tuning)")
    args = parser.parse_args()

    # Parse config
    if args.config_file:
        overrides = json.loads(Path(args.config_file).read_text())
    else:
        overrides = json.loads(args.config)

    config = AgentConfig(**overrides)
    benchmarks = [b.strip().upper() for b in args.benchmarks.split(",")]

    logger.info(f"Trial config: {config.model_dump()}")
    logger.info(f"Benchmarks: {benchmarks}")
    logger.info(f"Val split: {VAL_SPLIT}")

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    metrics = {}

    # Run selected benchmarks
    for bm in benchmarks:
        try:
            if bm == "B1":
                metrics.update(eval_b1(config))
            elif bm == "B2":
                metrics.update(eval_b2(config))
            elif bm == "B5":
                metrics.update(eval_b5(config))
            else:
                logger.warning(f"Unknown benchmark {bm}, skipping")
        except Exception:
            logger.error(f"Benchmark {bm} failed:\n{traceback.format_exc()}")

    elapsed = time.time() - t0
    composite = compute_composite(metrics)

    # Print in deterministic parseable format (like autoresearch)
    print("---")
    print(f"composite_score:  {composite:.6f}")
    print(f"rank_corr:        {metrics.get('rank_corr', 0.0):.6f}")
    print(f"trend:            {metrics.get('trend', 0.0):.6f}")
    print(f"f1_warning:       {metrics.get('f1_warning', 0.0):.6f}")
    print(f"ticket_prec:      {metrics.get('ticket_prec', 0.0):.6f}")
    print(f"elapsed_seconds:  {elapsed:.1f}")
    print(f"config:           {json.dumps(overrides)}")
    print("---")

    # Also save to file for easy retrieval
    result = {
        "composite_score": composite,
        "metrics": metrics,
        "config": overrides,
        "elapsed_seconds": elapsed,
    }
    out = Path(RESULTS_DIR) / "last_trial.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info(f"Trial done in {elapsed:.1f}s | composite={composite:.4f}")


if __name__ == "__main__":
    main()
