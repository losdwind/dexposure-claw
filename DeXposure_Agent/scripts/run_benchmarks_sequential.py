#!/usr/bin/env python3
"""Sequential benchmark runner for B1-B6 experiments.

Runs each benchmark with available methods, saves results to results/ directory.
Designed to run on GPU server with full data in ../../DeXposure/data/.

Features:
- Overall progress bar showing benchmark completion
- Per-benchmark timing with ETA
- Structured logging to file + stderr
- Summary table at the end
"""
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Ensure repo root importable
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Setup - each run gets its own timestamped directory
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_ROOT = Path(_REPO_ROOT) / "results"
RESULTS_DIR = RESULTS_ROOT / f"run_{RUN_TIMESTAMP}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Symlink 'latest' to this run for convenience
latest_link = RESULTS_ROOT / "latest"
if latest_link.is_symlink():
    latest_link.unlink()
try:
    latest_link.symlink_to(RESULTS_DIR.name)
except OSError:
    pass

log_file = RESULTS_DIR / f"benchmark_run.log"
logger.add(
    str(log_file),
    rotation="100 MB",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
)

# Data directory
DATA_DIR = str(Path(_REPO_ROOT).parent / "DeXposure" / "data")
TEST_SPLIT = "2025-01~2025-08"


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    elif s < 3600:
        return f"{int(s//60)}m{int(s%60)}s"
    else:
        return f"{int(s//3600)}h{int(s%3600//60)}m{int(s%60)}s"


def _run_benchmark(bench_name: str, run_fn, method: str, results_tracker: dict):
    """Run a single benchmark-method pair with timing and error handling."""
    logger.info(f"{'='*60}")
    logger.info(f"{bench_name}/{method} | STARTING")
    logger.info(f"{'='*60}")
    t0 = time.time()
    try:
        results = run_fn(
            method_id=method,
            data_dir=DATA_DIR,
            test_split=TEST_SPLIT,
            results_dir=str(RESULTS_DIR),
        )
        elapsed = time.time() - t0
        logger.info(f"{bench_name}/{method} | DONE in {_fmt_time(elapsed)}")
        for r in results:
            logger.info(f"  {r}")
        results_tracker[f"{bench_name}/{method}"] = {
            "status": "OK",
            "elapsed": _fmt_time(elapsed),
            "n_results": len(results),
        }
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"{bench_name}/{method} | FAILED in {_fmt_time(elapsed)}: {e}")
        logger.error(traceback.format_exc())
        results_tracker[f"{bench_name}/{method}"] = {
            "status": f"FAILED: {e}",
            "elapsed": _fmt_time(elapsed),
        }


def main():
    total_start = time.time()

    # Define all benchmark-method pairs
    benchmark_plan = [
        ("B1", "experiments.b1_risk_forecasting", "run_b1", ["C2", "C0"]),
        ("B2", "experiments.b2_early_warning", "run_b2", ["C2", "C0"]),
        ("B3", "experiments.b3_uncertainty_calibration", "run_b3", ["C0"]),
        ("B4", "experiments.b4_stress_test", "run_b4", ["C2", "C0"]),
        ("B5", "experiments.b5_decision_quality", "run_b5", ["C2", "C0"]),
        ("B6", "experiments.b6_robustness", "run_b6", ["C2", "C0"]),
    ]

    # Count total tasks
    total_tasks = sum(len(methods) for _, _, _, methods in benchmark_plan)

    # Header
    logger.info("\n" + "#" * 70)
    logger.info("  DeXposure-Agent Benchmark Suite")
    logger.info(f"  Data dir:    {DATA_DIR}")
    logger.info(f"  Test split:  {TEST_SPLIT}")
    logger.info(f"  Results dir: {RESULTS_DIR}")
    logger.info(f"  Log file:    {log_file}")
    logger.info(f"  Tasks:       {total_tasks} benchmark-method pairs across {len(benchmark_plan)} benchmarks")
    logger.info(f"  Started:     {datetime.now().isoformat()}")
    logger.info("#" * 70 + "\n")

    results_tracker: dict = {}
    completed = 0

    for bench_name, module_path, func_name, methods in benchmark_plan:
        logger.info(f"\n{'#'*70}")
        logger.info(f"### {bench_name} ({len(methods)} methods: {', '.join(methods)}) ###")
        logger.info(f"{'#'*70}")

        # Import the benchmark module dynamically
        import importlib
        module = importlib.import_module(module_path)
        run_fn = getattr(module, func_name)

        bench_start = time.time()
        for method in methods:
            completed += 1
            overall_pct = completed / total_tasks * 100
            elapsed_total = time.time() - total_start
            if completed > 1:
                eta = elapsed_total / (completed - 1) * (total_tasks - completed)
                eta_str = f"ETA={_fmt_time(eta)}"
            else:
                eta_str = "ETA=calculating..."

            logger.info(
                f"\n[OVERALL {completed}/{total_tasks} ({overall_pct:.0f}%)] "
                f"elapsed={_fmt_time(elapsed_total)} {eta_str}"
            )

            _run_benchmark(bench_name, run_fn, method, results_tracker)

        bench_elapsed = time.time() - bench_start
        logger.info(f"### {bench_name} ALL METHODS COMPLETE in {_fmt_time(bench_elapsed)} ###\n")

    # Final summary
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 70)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total time: {_fmt_time(total_elapsed)}")
    logger.info(f"  Tasks: {total_tasks}")
    logger.info("")

    ok_count = 0
    fail_count = 0
    for task, info in results_tracker.items():
        status = info["status"]
        elapsed = info["elapsed"]
        if status == "OK":
            ok_count += 1
            logger.info(f"  [OK]   {task:<20} {elapsed:>10}  ({info.get('n_results', '?')} results)")
        else:
            fail_count += 1
            logger.info(f"  [FAIL] {task:<20} {elapsed:>10}  {status}")

    logger.info("")
    logger.info(f"  Passed: {ok_count}/{total_tasks}  Failed: {fail_count}/{total_tasks}")
    logger.info(f"  Results: {RESULTS_DIR}")
    logger.info(f"  Log: {log_file}")
    logger.info("=" * 70)

    # Save tracker to JSON
    tracker_path = RESULTS_DIR / "run_summary.json"
    tracker_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "total_elapsed_human": _fmt_time(total_elapsed),
        "tasks_ok": ok_count,
        "tasks_failed": fail_count,
        "tasks_total": total_tasks,
        "results": results_tracker,
    }, indent=2))
    logger.info(f"Run summary saved to {tracker_path}")


if __name__ == "__main__":
    main()
