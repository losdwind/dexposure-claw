#!/usr/bin/env python3
"""Master experiment runner for DeXposure-Agent paper.

Usage:
    python experiments/run_all.py --benchmarks B1,B2,B3,B4,B5,B6 --methods all
    python experiments/run_all.py --benchmarks B1 --methods C0,C4
    python experiments/run_all.py --ablations --output results/
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Competitor-benchmark applicability matrix (Section 4.3)
APPLICABILITY = {
    "C0":  {"B1": True, "B2": True, "B3": True, "B4": True, "B5": True, "B6": True},
    "C1":  {"B1": True, "B2": True, "B3": True, "B4": True, "B5": True, "B6": True},
    "C2":  {"B1": True, "B2": True, "B3": False,"B4": True, "B5": True, "B6": False},
    "C3":  {"B1": False,"B2": True, "B3": False,"B4": False,"B5": True, "B6": True},
    "C4":  {"B1": True, "B2": False,"B3": True, "B4": True, "B5": False,"B6": True},
    "C5":  {"B1": True, "B2": False,"B3": False,"B4": True, "B5": False,"B6": True},
    "C6":  {"B1": True, "B2": False,"B3": False,"B4": True, "B5": False,"B6": True},
    "C7":  {"B1": True, "B2": False,"B3": False,"B4": True, "B5": False,"B6": True},
    "C8":  {"B1": True, "B2": False,"B3": False,"B4": True, "B5": False,"B6": True},
    "C9":  {"B1": True, "B2": False,"B3": False,"B4": True, "B5": False,"B6": True},
    "C10": {"B1": True, "B2": False,"B3": False,"B4": True, "B5": False,"B6": False},
}

METHOD_NAMES = {
    "C0": "DeXposure-Agent", "C1": "ROLAND-Agent", "C2": "Persistence-Agent",
    "C3": "LLM-Agent", "C4": "DeXposure-FM", "C5": "ROLAND",
    "C6": "GraphPFN-Frozen", "C7": "EvolveGCN", "C8": "DyRep",
    "C9": "TGN", "C10": "Static GCN",
}

BENCHMARK_MODULES = {
    "B1": "experiments.b1_risk_forecasting",
    "B2": "experiments.b2_early_warning",
    "B3": "experiments.b3_uncertainty_calibration",
    "B4": "experiments.b4_stress_test",
    "B5": "experiments.b5_decision_quality",
    "B6": "experiments.b6_robustness",
}

BENCHMARK_FUNCS = {
    "B1": "run_b1",
    "B2": "run_b2",
    "B3": "run_b3",
    "B4": "run_b4",
    "B5": "run_b5",
    "B6": "run_b6",
}


def _run_benchmark(bench: str, method: str, data_dir: str, test_split: str) -> dict:
    """Dynamically dispatch to the appropriate benchmark runner."""
    import importlib
    import sys
    from pathlib import Path as _Path
    # Ensure repo root is on sys.path so 'experiments.*' is importable when
    # the script is run as  python experiments/run_all.py  from the repo root.
    _repo_root = str(_Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    module = importlib.import_module(BENCHMARK_MODULES[bench])
    func = getattr(module, BENCHMARK_FUNCS[bench])
    try:
        results = func(method_id=method, data_dir=data_dir, test_split=test_split)
        return {"status": "ok", "results": [str(r) for r in results]}
    except NotImplementedError as exc:
        logger.warning(f"{bench}/{method}: {exc}")
        return {"status": "not_implemented", "error": str(exc)}
    except Exception as exc:
        logger.error(f"{bench}/{method} failed: {exc}")
        return {"status": "error", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="DeXposure-Agent Experiment Runner")
    parser.add_argument("--benchmarks", default="B1,B2,B3,B4,B5,B6",
                        help="Comma-separated benchmarks (e.g. B1,B2)")
    parser.add_argument("--methods", default="all",
                        help="Comma-separated method IDs or 'all'")
    parser.add_argument("--ablations", action="store_true",
                        help="Run ablation studies A1-A8")
    parser.add_argument("--output", default="results/",
                        help="Output directory")
    parser.add_argument("--data-dir", default="data/",
                        help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range (YYYY-MM~YYYY-MM)")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log_file = out / f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger.add(str(log_file), rotation="100 MB", retention="30 days",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.info(f"Starting experiment run: benchmarks={args.benchmarks}, methods={args.methods}")
    logger.info(f"Output dir: {out.resolve()}")

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    methods = (list(METHOD_NAMES.keys())
               if args.methods == "all"
               else [m.strip() for m in args.methods.split(",")])

    unknown_benchmarks = [b for b in benchmarks if b not in BENCHMARK_MODULES]
    unknown_methods = [m for m in methods if m not in METHOD_NAMES]
    if unknown_benchmarks:
        logger.error(f"Unknown benchmarks: {unknown_benchmarks}")
        sys.exit(1)
    if unknown_methods:
        logger.error(f"Unknown methods: {unknown_methods}")
        sys.exit(1)

    results = {}
    for bench in benchmarks:
        for method in methods:
            if not APPLICABILITY.get(method, {}).get(bench, False):
                logger.info(f"Skipping {method} ({METHOD_NAMES[method]}) for {bench} (not applicable)")
                continue
            logger.info(f"Running {bench} for {method} ({METHOD_NAMES[method]})")
            results[(bench, method)] = _run_benchmark(
                bench, method, args.data_dir, args.test_split
            )

    if args.ablations:
        logger.info("Running ablation studies A1-A8")
        import importlib
        ablations_mod = importlib.import_module("experiments.ablations")
        try:
            ablation_results = ablations_mod.run_all_ablations(
                data_dir=args.data_dir, test_split=args.test_split
            )
            results[("ablations", "C0")] = {"status": "ok", "results": str(ablation_results)}
        except NotImplementedError as exc:
            logger.warning(f"Ablations not yet implemented: {exc}")
            results[("ablations", "C0")] = {"status": "not_implemented", "error": str(exc)}

    # Save results
    results_file = out / "results.json"
    serializable = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
    results_file.write_text(json.dumps(serializable, indent=2))
    logger.info(f"Results saved to {results_file}")

    n_ok = sum(1 for v in results.values() if v.get("status") == "ok")
    n_total = len(results)
    logger.info(f"Done. {n_ok}/{n_total} benchmark-method pairs completed successfully.")


if __name__ == "__main__":
    main()
