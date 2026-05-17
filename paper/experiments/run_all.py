#!/usr/bin/env python3
"""Reproducible master runner for DeXposure-Agent experiments.

This runner covers the graph/rules benchmarks b1_forecast..b6_robustness.
LLM decision-quality methods (m2_snapshot_llm, m6_fm_llm, m7_fm_llm_gated)
intentionally run through experiments/llm_eval_b5.py so API prompts, judge
artifacts, and costs stay separate from the FM benchmark suite.

Usage:
    python experiments/run_all.py --benchmarks b1_forecast,b2_warning,b3_calibration,b4_stress,b5_decision,b6_robustness --methods all
    python experiments/run_all.py --benchmarks b1_forecast --methods m5_fm_rules,m4_fm_only
    python experiments/run_all.py --ablations --output results/
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import logging
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ModuleNotFoundError:
    class _StdLogger:
        def __init__(self) -> None:
            self._logger = logging.getLogger("dexposure.run_all")
            self._logger.setLevel(logging.DEBUG)
            self._next_id = 0
            self._handlers: dict[int, logging.Handler] = {}

        def _render(self, msg: object, *args: object) -> str:
            text = str(msg)
            if not args:
                return text
            try:
                return text.format(*args)
            except Exception:
                try:
                    return text % args
                except Exception:
                    return " ".join([text, *(str(arg) for arg in args)])

        def add(self, sink: object, level: str = "INFO", **_: object) -> int:
            if hasattr(sink, "write"):
                handler: logging.Handler = logging.StreamHandler(sink)
            else:
                handler = logging.FileHandler(str(sink))
            handler.setLevel(getattr(logging, str(level).upper(), logging.INFO))
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self._logger.addHandler(handler)
            self._next_id += 1
            self._handlers[self._next_id] = handler
            return self._next_id

        def remove(self, handler_id: int | None = None) -> None:
            if handler_id is None:
                handler_ids = list(self._handlers)
            else:
                handler_ids = [handler_id]
            for hid in handler_ids:
                handler = self._handlers.pop(hid, None)
                if handler is not None:
                    self._logger.removeHandler(handler)
                    handler.close()

        def debug(self, msg: object, *args: object, **_: object) -> None:
            self._logger.debug(self._render(msg, *args))

        def info(self, msg: object, *args: object, **_: object) -> None:
            self._logger.info(self._render(msg, *args))

        def warning(self, msg: object, *args: object, **_: object) -> None:
            self._logger.warning(self._render(msg, *args))

        def error(self, msg: object, *args: object, **_: object) -> None:
            self._logger.error(self._render(msg, *args))

    logger = _StdLogger()
    sys.modules.setdefault("loguru", types.SimpleNamespace(logger=logger))


os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_DISABLE_GRAPHBOLT", "1")

AGENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AGENT_ROOT.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from experiments.methods import (  # noqa: E402
    METHOD_NAMES,
    METHODS,
    get_method,
)

from experiments.exceptions import PredictionUnavailable  # noqa: E402


BENCHMARK_MODULES = {
    "b1_forecast": "experiments.b1_risk_forecasting",
    "b2_warning": "experiments.b2_early_warning",
    "b3_calibration": "experiments.b3_uncertainty_calibration",
    "b4_stress": "experiments.b4_stress_test",
    "b5_decision": "experiments.b5_decision_quality",
    "b6_robustness": "experiments.b6_robustness",
}

BENCHMARK_FUNCS = {
    "b1_forecast": "run_b1",
    "b2_warning": "run_b2",
    "b3_calibration": "run_b3",
    "b4_stress": "run_b4",
    "b5_decision": "run_b5",
    "b6_robustness": "run_b6",
}

# Applicability is benchmark-first so b2_warning's heuristic-only contract is visible.
BENCHMARK_APPLICABILITY = {
    "b1_forecast": {"m5_fm_rules", "m1_persistence_rules", "m4_fm_only", "m3_evolvegcn"},
    "b2_warning": {"h1_weighted_degree"},
    "b3_calibration": {"m5_fm_rules", "m4_fm_only"},
    "b4_stress": {"m5_fm_rules", "m1_persistence_rules", "m4_fm_only", "m3_evolvegcn"},
    "b5_decision": {"m5_fm_rules", "m1_persistence_rules"},
    "b6_robustness": {"m5_fm_rules", "m1_persistence_rules", "m4_fm_only", "m3_evolvegcn"},
}

LLM_PIPELINE_METHODS = {"m2_snapshot_llm", "m6_fm_llm", "m7_fm_llm_gated"}
RUN_ALL_METHODS = [
    method_id for method_id in METHODS if method_id not in LLM_PIPELINE_METHODS
]
TEST_SPLIT_RE = re.compile(r"\d{4}-\d{2}~\d{4}-\d{2}")
MAX_REPR_CHARS = 500


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _setup_logger(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    log_file = out / f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    return log_file


def _run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_checkpoint_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    suffixes = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
    roots = [AGENT_ROOT / "checkpoints", PROJECT_ROOT / "checkpoints"]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                try:
                    hashes[str(path.relative_to(PROJECT_ROOT))] = _sha256_file(path)
                except Exception as exc:
                    hashes[str(path)] = f"hash_error: {exc}"
    return hashes


def _collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "dgl_backend": os.environ.get("DGLBACKEND"),
        "dgl_disable_graphbolt": os.environ.get("DGL_DISABLE_GRAPHBOLT"),
    }

    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device_count"] = int(torch.cuda.device_count())
        env["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        env["torch_error"] = repr(exc)

    try:
        import dgl

        env["dgl_version"] = dgl.__version__
    except Exception as exc:
        env["dgl_error"] = repr(exc)

    return env


def _set_seed(seed: int) -> list[str]:
    seeded = ["python.random"]
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        seeded.append("numpy")
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded.append("torch")
    except Exception:
        pass

    return seeded


def _jsonify(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonify(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _jsonify(value.model_dump())
    return _repr_limited(value)


def _repr_limited(value: Any) -> str:
    text = repr(value)
    if len(text) <= MAX_REPR_CHARS:
        return text
    return text[:MAX_REPR_CHARS - 3] + "..."


def _is_unavailable_exception(exc: Exception) -> bool:
    # Unavailable predictor exceptions are registered in experiments.exceptions
    # and raised by experiments.predict_helper.
    return isinstance(exc, PredictionUnavailable)


def _exception_payload(exc: Exception) -> dict[str, Any]:
    return {
        "error": str(exc),
        "exception_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
    }


def _write_state(results_file: Path, state: dict[str, Any]) -> None:
    payload = json.dumps(_jsonify(state), indent=2, sort_keys=True, allow_nan=False)
    tmp_file = results_file.with_suffix(results_file.suffix + ".tmp")
    tmp_file.write_text(payload + "\n")
    tmp_file.replace(results_file)


def _summarize(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results.values():
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "counts": counts,
        "n_total_records": len(results),
        "n_executed": sum(
            1 for result in results.values()
            if result.get("status") != "applicable_skip"
        ),
        "n_ok": counts.get("ok", 0),
    }


def _collect_metadata(args: argparse.Namespace, log_file: Path) -> dict[str, Any]:
    status_short = _run_git(["status", "--short"])
    return {
        "schema_version": 2,
        "runner": "experiments/run_all.py",
        "started_at": _utc_now(),
        "args": vars(args),
        "log_file": str(log_file.relative_to(PROJECT_ROOT)),
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status_short),
        "git_status_short": status_short.splitlines(),
        "environment": _collect_environment(),
        "seeded_libraries": _set_seed(args.seed),
        "methods": {
            method_id: dataclasses.asdict(spec) for method_id, spec in METHODS.items()
        },
        "benchmark_applicability": {
            benchmark: sorted(methods)
            for benchmark, methods in BENCHMARK_APPLICABILITY.items()
        },
        "checkpoint_sha256": _collect_checkpoint_hashes(),
    }


def _parse_request(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if not TEST_SPLIT_RE.fullmatch(args.test_split):
        raise ValueError(
            f"Invalid --test-split {args.test_split!r}; expected YYYY-MM~YYYY-MM"
        )

    benchmarks = _split_csv(args.benchmarks)
    unknown_benchmarks = [b for b in benchmarks if b not in BENCHMARK_MODULES]
    if unknown_benchmarks:
        raise ValueError(f"Unknown benchmarks: {unknown_benchmarks}")

    methods = RUN_ALL_METHODS if args.methods == "all" else _split_csv(args.methods)
    unknown_methods = [m for m in methods if m not in METHOD_NAMES]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")

    llm_methods = [m for m in methods if m in LLM_PIPELINE_METHODS]
    if llm_methods:
        raise ValueError(
            "LLM decision-quality methods are evaluated by "
            "experiments/llm_eval_b5.py, not run_all.py: "
            + ", ".join(llm_methods)
        )

    for method in methods:
        get_method(method)

    return benchmarks, methods


def _skip_result(bench: str, method: str) -> dict[str, Any]:
    return {
        "status": "applicable_skip",
        "benchmark": bench,
        "method": method,
        "method_label": METHOD_NAMES[method],
        "reason": f"{method} is not applicable to {bench}",
    }


def _run_benchmark(
    bench: str,
    method: str,
    data_dir: str,
    test_split: str,
    results_dir: Path,
) -> dict[str, Any]:
    started_at = _utc_now()
    start = time.perf_counter()
    module = importlib.import_module(BENCHMARK_MODULES[bench])
    func = getattr(module, BENCHMARK_FUNCS[bench])

    try:
        results = func(
            method_id=method,
            data_dir=data_dir,
            test_split=test_split,
            results_dir=str(results_dir),
        )
        status = "ok"
        payload: dict[str, Any] = {"results": _jsonify(results)}
    except NotImplementedError as exc:
        status = "not_implemented"
        payload = _exception_payload(exc)
        logger.warning(f"{bench}/{method}: {exc}")
    except Exception as exc:
        payload = _exception_payload(exc)
        if _is_unavailable_exception(exc):
            status = "unavailable"
            logger.warning(f"{bench}/{method} unavailable: {exc}")
        else:
            status = "runtime_error"
            logger.error(f"{bench}/{method} failed: {exc}")

    ended_at = _utc_now()
    return {
        "status": status,
        "benchmark": bench,
        "method": method,
        "method_label": METHOD_NAMES[method],
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": round(time.perf_counter() - start, 3),
        **payload,
    }


def _run_ablations(
    data_dir: str,
    test_split: str,
    results_dir: Path,
    ablation_ids: list[str] | None,
) -> dict[str, Any]:
    started_at = _utc_now()
    start = time.perf_counter()
    module = importlib.import_module("experiments.ablations")

    try:
        results = module.run_all_ablations(
            data_dir=data_dir,
            test_split=test_split,
            results_dir=str(results_dir),
            ablation_ids=ablation_ids,
        )
        status = "ok"
        payload: dict[str, Any] = {"results": _jsonify(results)}
    except NotImplementedError as exc:
        status = "not_implemented"
        payload = _exception_payload(exc)
    except Exception as exc:
        payload = _exception_payload(exc)
        if _is_unavailable_exception(exc):
            status = "unavailable"
            logger.warning(f"Ablations unavailable: {exc}")
        else:
            status = "runtime_error"
            logger.error(f"Ablations failed: {exc}")

    return {
        "status": status,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "wall_seconds": round(time.perf_counter() - start, 3),
        "ablation_ids": ablation_ids,
        **payload,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeXposure-Agent Experiment Runner")
    parser.add_argument(
        "--benchmarks",
        default="b1_forecast,b2_warning,b3_calibration,b4_stress,b5_decision,b6_robustness",
        help="Comma-separated benchmarks (e.g. b1_forecast,b2_warning)",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated method IDs or 'all'. run_all.py excludes "
            "m2_snapshot_llm/m6_fm_llm/m7_fm_llm_gated; "
            "use experiments/llm_eval_b5.py for those."
        ),
    )
    parser.add_argument("--ablations", action="store_true", help="Run ablations")
    parser.add_argument(
        "--ablation-ids",
        default="",
        help="Comma-separated ablation IDs; default is all implemented ablations",
    )
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument(
        "--test-split",
        default="2025-01~2025-08",
        help="Test split range (YYYY-MM~YYYY-MM)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after runtime errors; results.json still records failures",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        benchmarks, methods = _parse_request(args)
    except Exception as exc:
        parser.error(str(exc))

    out = Path(args.output)
    log_file = _setup_logger(out)
    results_file = out / "results.json"
    state: dict[str, Any] = {
        "metadata": _collect_metadata(args, log_file),
        "results": {},
        "ablations": None,
        "summary": {},
    }
    _write_state(results_file, state)

    logger.info(
        f"Starting experiment run: benchmarks={benchmarks}, methods={methods}, "
        f"test_split={args.test_split}, seed={args.seed}"
    )
    logger.info(f"Output dir: {out.resolve()}")

    exit_code = 0
    started = time.perf_counter()

    try:
        stop_requested = False
        for bench in benchmarks:
            for method in methods:
                key = f"{bench}__{method}"
                if method not in BENCHMARK_APPLICABILITY[bench]:
                    state["results"][key] = _skip_result(bench, method)
                    state["summary"] = _summarize(state["results"])
                    _write_state(results_file, state)
                    logger.info(
                        f"Skipping {method} ({METHOD_NAMES[method]}) for {bench}"
                    )
                    continue

                logger.info(f"Running {bench} for {method} ({METHOD_NAMES[method]})")
                result = _run_benchmark(
                    bench=bench,
                    method=method,
                    data_dir=args.data_dir,
                    test_split=args.test_split,
                    results_dir=out,
                )
                state["results"][key] = result
                state["summary"] = _summarize(state["results"])
                _write_state(results_file, state)

                if result["status"] == "runtime_error":
                    exit_code = 1
                    if not args.continue_on_error:
                        stop_requested = True
                        logger.error(
                            "Stopping after failed benchmark. Use "
                            "--continue-on-error to keep running remaining pairs."
                        )
                        break
                elif result["status"] in {"not_implemented", "unavailable"}:
                    exit_code = 1
            if stop_requested:
                break

        if not stop_requested and args.ablations:
            ablation_ids = _split_csv(args.ablation_ids) if args.ablation_ids else None
            logger.info("Running ablation studies")
            state["ablations"] = _run_ablations(
                data_dir=args.data_dir,
                test_split=args.test_split,
                results_dir=out,
                ablation_ids=ablation_ids,
            )
            if state["ablations"]["status"] != "ok":
                exit_code = 1
            _write_state(results_file, state)

    except KeyboardInterrupt:
        state["metadata"]["interrupted_at"] = _utc_now()
        exit_code = 130
        logger.warning("Interrupted; partial results have been saved.")
    finally:
        state["metadata"]["ended_at"] = _utc_now()
        state["metadata"]["total_wall_seconds"] = round(time.perf_counter() - started, 3)
        state["summary"] = _summarize(state["results"])
        _write_state(results_file, state)

    logger.info(f"Results saved to {results_file}")
    if exit_code == 0:
        logger.info(
            f"ALL DONE. {state['summary'].get('n_ok', 0)}/"
            f"{state['summary'].get('n_executed', 0)} executed records completed."
        )
    else:
        logger.error(
            f"Run ended with status {exit_code}. "
            f"Summary: {state['summary'].get('counts', {})}"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
