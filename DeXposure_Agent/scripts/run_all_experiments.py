#!/usr/bin/env python3
"""Run all supplementary experiments for the paper.

Execution order (designed for continuous GPU utilization):
  1. Train EvolveGCN (m3_evolvegcn) for h={1,4,8,12}          ~60 min
  2. Run m3_evolvegcn on b1_forecast + b4_stress + b6_robustness                          ~15 min
  3. Re-run b3_calibration with conformal calibration fix         ~10 min
  4. Run 4 ablation experiments (A1,A2,A3,A6)         ~30 min
  5. Re-run full b1_forecast-b6_robustness for m5_fm_rules (with all fixes)        ~40 min

Total estimated GPU time: ~2.5 hours

Usage:
    ssh gpu-server
    cd /root/graph-dexposure/DeXposure_Agent
    DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
      nohup python3 -u scripts/run_all_experiments.py \
      > results/all_experiments_stdout.log 2>&1 &
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT.parent))

from loguru import logger

# ── Configuration ──────────────────────────────────────────────────────

DATA_DIR = str(_REPO_ROOT.parent / "data")
TEST_SPLIT = "2025-01~2025-08"
TRAIN_SPLIT = "2020-03~2024-06"
VAL_SPLIT = "2024-07~2024-12"
EVOLVEGCN_CKPT_DIR = str(_REPO_ROOT.parent / "checkpoints" / "evolvegcn")

RESULTS_ROOT = _REPO_ROOT / "results"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = RESULTS_ROOT / f"run_{RUN_TIMESTAMP}_supplementary"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logger.add(RESULTS_DIR / "experiment_run.log", level="INFO")

overall_status: dict[str, dict] = {}


def _record(step: str, status: str, duration: float, details: dict = None):
    """Record step result."""
    overall_status[step] = {
        "status": status,
        "duration_seconds": round(duration, 1),
        "details": details or {},
    }
    logger.info(f"[{step}] {status} ({duration:.1f}s)")


# ── Step 1: Train EvolveGCN ───────────────────────────────────────────

def step1_train_evolvegcn():
    logger.info("=" * 60)
    logger.info("STEP 1: Train EvolveGCN (m3_evolvegcn) for all horizons")
    logger.info("=" * 60)

    from experiments.competitors.evolvegcn import train_evolvegcn

    t0 = time.time()
    for h in [1, 4, 8, 12]:
        logger.info(f"Training EvolveGCN h={h}...")
        try:
            ckpt_path = train_evolvegcn(
                data_dir=DATA_DIR,
                train_split=TRAIN_SPLIT,
                val_split=VAL_SPLIT,
                horizon=h,
                checkpoint_dir=EVOLVEGCN_CKPT_DIR,
            )
            logger.info(f"  h={h} saved to {ckpt_path}")
        except Exception as e:
            logger.error(f"  h={h} FAILED: {e}")

    _record("step1_train_evolvegcn", "done", time.time() - t0)


# ── Step 2: Run m3_evolvegcn benchmarks ────────────────────────────────────────

def step2_evolvegcn_benchmarks():
    logger.info("=" * 60)
    logger.info("STEP 2: Run EvolveGCN (m3_evolvegcn) on b1_forecast + b4_stress + b6_robustness")
    logger.info("=" * 60)

    t0 = time.time()
    results_dir = str(RESULTS_DIR)

    # b1_forecast
    try:
        from experiments.b1_risk_forecasting import run_b1
        b1 = run_b1("m3_evolvegcn", DATA_DIR, TEST_SPLIT, results_dir=results_dir)
        logger.info(f"  b1_forecast m3_evolvegcn: {len(b1)} results")
    except Exception as e:
        logger.error(f"  b1_forecast m3_evolvegcn FAILED: {e}")

    # b4_stress
    try:
        from experiments.b4_stress_test import run_b4
        b4 = run_b4("m3_evolvegcn", DATA_DIR, TEST_SPLIT, results_dir=results_dir)
        logger.info(f"  b4_stress m3_evolvegcn: {len(b4)} results")
    except Exception as e:
        logger.error(f"  b4_stress m3_evolvegcn FAILED: {e}")

    # b6_robustness
    try:
        from experiments.b6_robustness import run_b6
        b6 = run_b6("m3_evolvegcn", DATA_DIR, TEST_SPLIT, results_dir=results_dir)
        logger.info(f"  b6_robustness m3_evolvegcn: {len(b6)} results")
    except Exception as e:
        logger.error(f"  b6_robustness m3_evolvegcn FAILED: {e}")

    _record("step2_evolvegcn_benchmarks", "done", time.time() - t0)


# ── Step 3: Re-run b3_calibration with conformal calibration ─────────────────────

def step3_b3_conformal():
    logger.info("=" * 60)
    logger.info("STEP 3: Re-run b3_calibration with conformal calibration fix")
    logger.info("=" * 60)

    t0 = time.time()
    try:
        from experiments.b3_uncertainty_calibration import run_b3
        b3 = run_b3("m5_fm_rules", DATA_DIR, TEST_SPLIT, results_dir=str(RESULTS_DIR))
        if b3:
            r = b3[0]
            logger.info(
                f"  b3_calibration m5_fm_rules (conformal): ECE={r.ece:.4f} "
                f"PI_cov={r.pi_coverage:.3f} PI_width={r.pi_width:.4f} "
                f"CRPS={r.crps:.4f}"
            )
            _record("step3_b3_conformal", "done", time.time() - t0, {
                "ece": r.ece, "pi_coverage": r.pi_coverage,
                "pi_width": r.pi_width, "crps": r.crps,
            })
        else:
            _record("step3_b3_conformal", "no_results", time.time() - t0)
    except Exception as e:
        logger.error(f"  b3_calibration FAILED: {e}")
        _record("step3_b3_conformal", "failed", time.time() - t0, {"error": str(e)})


# ── Step 4: Ablation experiments ──────────────────────────────────────

def step4_ablations():
    logger.info("=" * 60)
    logger.info("STEP 4: Run ablation experiments (A1, A2, A3, A6)")
    logger.info("=" * 60)

    t0 = time.time()
    try:
        from experiments.ablations import run_all_ablations
        results = run_all_ablations(
            data_dir=DATA_DIR,
            test_split=TEST_SPLIT,
            results_dir=str(RESULTS_DIR),
            ablation_ids=["A1", "A2", "A3", "A6"],
        )
        for r in results:
            logger.info(f"  {r}")

        _record("step4_ablations", "done", time.time() - t0, {
            "n_ablations": len(results),
        })
    except Exception as e:
        logger.error(f"  Ablations FAILED: {e}")
        _record("step4_ablations", "failed", time.time() - t0, {"error": str(e)})


# ── Step 5: Re-run full m5_fm_rules+m1_persistence_rules suite ──────────────────────────────────

def step5_full_suite():
    logger.info("=" * 60)
    logger.info("STEP 5: Re-run full b1_forecast-b6_robustness for m5_fm_rules+m1_persistence_rules (with all fixes)")
    logger.info("=" * 60)

    t0 = time.time()
    results_dir = str(RESULTS_DIR)

    from experiments.b1_risk_forecasting import run_b1
    from experiments.b2_early_warning import run_b2
    from experiments.b3_uncertainty_calibration import run_b3
    from experiments.b4_stress_test import run_b4
    from experiments.b5_decision_quality import run_b5
    from experiments.b6_robustness import run_b6

    benchmarks = [
        ("b1_forecast", run_b1), ("b2_warning", run_b2), ("b3_calibration", run_b3),
        ("b4_stress", run_b4), ("b5_decision", run_b5), ("b6_robustness", run_b6),
    ]

    for bname, bfunc in benchmarks:
        methods = ["h1_weighted_degree"] if bname == "b2_warning" else ["m5_fm_rules", "m1_persistence_rules"]
        for method in methods:
            # b3_calibration only runs for m5_fm_rules (uncertainty method)
            if bname == "b3_calibration" and method == "m1_persistence_rules":
                continue
            try:
                logger.info(f"  Running {bname} {method}...")
                result = bfunc(method, DATA_DIR, TEST_SPLIT, results_dir=results_dir)
                logger.info(f"  {bname} {method}: {len(result)} result(s)")
            except Exception as e:
                logger.error(f"  {bname} {method} FAILED: {e}")

    _record("step5_full_suite", "done", time.time() - t0)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info(f"DeXposure-Agent Supplementary Experiments")
    logger.info(f"Results dir: {RESULTS_DIR}")
    logger.info(f"Data dir: {DATA_DIR}")
    logger.info(f"Test split: {TEST_SPLIT}")
    logger.info("=" * 60)

    t_total = time.time()

    step1_train_evolvegcn()
    step2_evolvegcn_benchmarks()
    step3_b3_conformal()
    step4_ablations()
    step5_full_suite()

    total_time = time.time() - t_total

    # Save overall summary
    summary = {
        "timestamp": RUN_TIMESTAMP,
        "total_time_seconds": round(total_time, 1),
        "total_time_human": f"{total_time/60:.1f} min",
        "steps": overall_status,
    }
    summary_path = RESULTS_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    logger.info("=" * 60)
    logger.info(f"ALL DONE in {total_time/60:.1f} min")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
