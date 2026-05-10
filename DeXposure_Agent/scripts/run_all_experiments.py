#!/usr/bin/env python3
"""Run all supplementary experiments for the paper.

Execution order (designed for continuous GPU utilization):
  1. Train EvolveGCN (C7) for h={1,4,8,12}          ~60 min
  2. Run C7 on B1 + B4 + B6                          ~15 min
  3. Re-run B3 with conformal calibration fix         ~10 min
  4. Run 4 ablation experiments (A1,A2,A3,A6)         ~30 min
  5. Re-run full B1-B6 for C0 (with all fixes)        ~40 min

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

DATA_DIR = str(_REPO_ROOT.parent / "DeXposure" / "data")
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
    logger.info("STEP 1: Train EvolveGCN (C7) for all horizons")
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


# ── Step 2: Run C7 benchmarks ────────────────────────────────────────

def step2_evolvegcn_benchmarks():
    logger.info("=" * 60)
    logger.info("STEP 2: Run EvolveGCN (C7) on B1 + B4 + B6")
    logger.info("=" * 60)

    t0 = time.time()
    results_dir = str(RESULTS_DIR)

    # B1
    try:
        from experiments.b1_risk_forecasting import run_b1
        b1 = run_b1("C7", DATA_DIR, TEST_SPLIT, results_dir=results_dir)
        logger.info(f"  B1 C7: {len(b1)} results")
    except Exception as e:
        logger.error(f"  B1 C7 FAILED: {e}")

    # B4
    try:
        from experiments.b4_stress_test import run_b4
        b4 = run_b4("C7", DATA_DIR, TEST_SPLIT, results_dir=results_dir)
        logger.info(f"  B4 C7: {len(b4)} results")
    except Exception as e:
        logger.error(f"  B4 C7 FAILED: {e}")

    # B6
    try:
        from experiments.b6_robustness import run_b6
        b6 = run_b6("C7", DATA_DIR, TEST_SPLIT, results_dir=results_dir)
        logger.info(f"  B6 C7: {len(b6)} results")
    except Exception as e:
        logger.error(f"  B6 C7 FAILED: {e}")

    _record("step2_evolvegcn_benchmarks", "done", time.time() - t0)


# ── Step 3: Re-run B3 with conformal calibration ─────────────────────

def step3_b3_conformal():
    logger.info("=" * 60)
    logger.info("STEP 3: Re-run B3 with conformal calibration fix")
    logger.info("=" * 60)

    t0 = time.time()
    try:
        from experiments.b3_uncertainty_calibration import run_b3
        b3 = run_b3("C0", DATA_DIR, TEST_SPLIT, results_dir=str(RESULTS_DIR))
        if b3:
            r = b3[0]
            logger.info(
                f"  B3 C0 (conformal): ECE={r.ece:.4f} "
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
        logger.error(f"  B3 FAILED: {e}")
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


# ── Step 5: Re-run full C0+C2 suite ──────────────────────────────────

def step5_full_suite():
    logger.info("=" * 60)
    logger.info("STEP 5: Re-run full B1-B6 for C0+C2 (with all fixes)")
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
        ("B1", run_b1), ("B2", run_b2), ("B3", run_b3),
        ("B4", run_b4), ("B5", run_b5), ("B6", run_b6),
    ]

    for bname, bfunc in benchmarks:
        methods = ["H0"] if bname == "B2" else ["C0", "C2"]
        for method in methods:
            # B3 only runs for C0 (uncertainty method)
            if bname == "B3" and method == "C2":
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
