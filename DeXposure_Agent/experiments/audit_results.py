#!/usr/bin/env python3
"""Audit DeXposure-Agent result artifacts for known reproducibility hazards.

This script is intentionally dependency-free. It should run in a bare Python
environment before the full ML stack is installed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRIC_KEYS = ("precision", "completeness", "false_intervention_rate")
METADATA_KEYS = {
    "method",
    "timestamp",
    "elapsed_seconds",
    "log_file",
    "test_split",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _safe_mean(values: list[float]) -> float | None:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _raw_metric_means(raw_entries: list[dict[str, Any]]) -> dict[str, float]:
    means: dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = []
        for entry in raw_entries:
            assessment = entry.get("assessment", {})
            value = assessment.get(key)
            if isinstance(value, (int, float)):
                vals.append(float(value))
        mean = _safe_mean(vals)
        if mean is not None:
            means[key] = mean
    return means


def _checkpoint_metric_means(checkpoint: dict[str, Any]) -> dict[str, float]:
    week_results = checkpoint.get("week_results", [])
    means: dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = []
        for entry in week_results:
            value = entry.get(key)
            if isinstance(value, (int, float)):
                vals.append(float(value))
        mean = _safe_mean(vals)
        if mean is not None:
            means[key] = mean
    return means


def audit_llm_checkpoints(results_dir: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for run_dir in sorted(results_dir.glob("llm_eval_*")):
        if not run_dir.is_dir():
            continue
        for raw_path in sorted(run_dir.glob("raw_*.json")):
            method = raw_path.stem.removeprefix("raw_")
            checkpoint_path = run_dir / f"checkpoint_{method}.json"
            if not checkpoint_path.exists():
                continue
            raw_data = _load_json(raw_path)
            checkpoint_data = _load_json(checkpoint_path)
            if not isinstance(raw_data, list) or not isinstance(checkpoint_data, dict):
                continue

            raw_means = _raw_metric_means(raw_data)
            checkpoint_means = _checkpoint_metric_means(checkpoint_data)
            mismatches = {}
            for key, raw_value in raw_means.items():
                checkpoint_value = checkpoint_means.get(key)
                if checkpoint_value is None:
                    continue
                if abs(raw_value - checkpoint_value) > 1e-6:
                    mismatches[key] = {
                        "raw": round(raw_value, 6),
                        "checkpoint": round(checkpoint_value, 6),
                    }
            if mismatches:
                issues.append({
                    "code": "llm_checkpoint_mismatch",
                    "severity": "error",
                    "path": str(run_dir),
                    "message": (
                        f"{method} raw assessments and checkpoint week_results "
                        "do not agree."
                    ),
                    "details": mismatches,
                })
    return issues


def _scrub_for_method_compare(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, dict):
        return {
            key: _scrub_for_method_compare(val)
            for key, val in value.items()
            if key not in METADATA_KEYS
        }
    if isinstance(value, list):
        return [_scrub_for_method_compare(item) for item in value]
    return value


def audit_identical_method_results(results_dir: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    search_dirs = [results_dir] + [p for p in results_dir.iterdir() if p.is_dir()]
    for run_dir in sorted(search_dirs):
        for c7_path in sorted(run_dir.glob("B*_C7.json")):
            benchmark = c7_path.name.split("_", 1)[0]
            c2_path = run_dir / f"{benchmark}_C2.json"
            if not c2_path.exists():
                continue
            c7_data = _scrub_for_method_compare(_load_json(c7_path))
            c2_data = _scrub_for_method_compare(_load_json(c2_path))
            if c7_data == c2_data:
                issues.append({
                    "code": "identical_method_results",
                    "severity": "error",
                    "path": str(run_dir),
                    "message": (
                        f"{benchmark} C7 results are identical to C2 after "
                        "removing metadata; this usually means C7 fell back to "
                        "persistence."
                    ),
                    "details": {
                        "benchmark": benchmark,
                        "method_a": str(c7_path),
                        "method_b": str(c2_path),
                    },
                })
    return issues


def audit_results(results_dir: Path) -> dict[str, Any]:
    issues = []
    issues.extend(audit_llm_checkpoints(results_dir))
    issues.extend(audit_identical_method_results(results_dir))
    return {
        "results_dir": str(results_dir),
        "n_issues": len(issues),
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit DeXposure-Agent result artifacts")
    parser.add_argument("--results-dir", default="DeXposure_Agent/results")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--allow-issues",
        action="store_true",
        help="Exit 0 even when issues are found; useful for tests and inventories.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"results dir not found: {results_dir}")

    report = audit_results(results_dir)
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(f"Audited {report['results_dir']}: {report['n_issues']} issue(s)")
        for issue in report["issues"]:
            print(f"[{issue['severity']}] {issue['code']}: {issue['message']}")
            print(f"  path: {issue['path']}")

    if report["issues"] and not args.allow_issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
