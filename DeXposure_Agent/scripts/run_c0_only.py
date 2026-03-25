#!/usr/bin/env python3
"""Quick runner: only C0 (FM-Agent) benchmarks. Skips C2 (already done)."""
import importlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger

RESULTS_ROOT = Path(_REPO_ROOT) / "results"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = RESULTS_ROOT / f"run_{RUN_TIMESTAMP}_c0_only"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

log_file = RESULTS_DIR / "benchmark_run.log"
logger.add(str(log_file), rotation="100 MB", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}")

DATA_DIR = str(Path(_REPO_ROOT).parent / "DeXposure" / "data")
TEST_SPLIT = "2025-01~2025-08"


def _fmt(s):
    if s < 60: return f"{s:.1f}s"
    elif s < 3600: return f"{int(s//60)}m{int(s%60)}s"
    else: return f"{int(s//3600)}h{int(s%3600//60)}m"


def main():
    total_start = time.time()
    tasks = [
        ("B1", "experiments.b1_risk_forecasting", "run_b1"),
        ("B2", "experiments.b2_early_warning", "run_b2"),
        ("B3", "experiments.b3_uncertainty_calibration", "run_b3"),
        ("B4", "experiments.b4_stress_test", "run_b4"),
        ("B5", "experiments.b5_decision_quality", "run_b5"),
        ("B6", "experiments.b6_robustness", "run_b6"),
    ]

    logger.info(f"C0-only run | pi_min=0.5 | results={RESULTS_DIR}")
    logger.info(f"Data: {DATA_DIR} | Split: {TEST_SPLIT}")

    results = {}
    for i, (bench, mod_path, func_name) in enumerate(tasks):
        logger.info(f"\n[{i+1}/{len(tasks)}] {bench}/C0 starting...")
        t0 = time.time()
        try:
            module = importlib.import_module(mod_path)
            run_fn = getattr(module, func_name)
            res = run_fn(method_id="C0", data_dir=DATA_DIR, test_split=TEST_SPLIT, results_dir=str(RESULTS_DIR))
            elapsed = time.time() - t0
            for r in res:
                logger.info(f"  {r}")
            results[f"{bench}/C0"] = {"status": "OK", "elapsed": _fmt(elapsed), "n": len(res)}
            logger.info(f"[{i+1}/{len(tasks)}] {bench}/C0 DONE in {_fmt(elapsed)}")
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"{bench}/C0 FAILED: {e}")
            logger.error(traceback.format_exc())
            results[f"{bench}/C0"] = {"status": f"FAIL: {e}", "elapsed": _fmt(elapsed)}

    total = time.time() - total_start
    logger.info(f"\nALL DONE in {_fmt(total)}")
    for k, v in results.items():
        logger.info(f"  [{v['status'][:4]}] {k:<10} {v['elapsed']}")

    (RESULTS_DIR / "run_summary.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "total": _fmt(total),
        "pi_min": 0.5,
        "method": "C0",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
