"""Benchmark command builders for DeXposure-Bench."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_BENCHMARKS = (
    "b1_forecast",
    "b2_warning",
    "b3_calibration",
    "b4_stress",
    "b5_decision",
    "b6_robustness",
)


@dataclass(frozen=True)
class BenchmarkCommand:
    cwd: Path
    argv: list[str]


def build_run_all_command(
    agent_root: Path,
    output: Path,
    benchmarks: list[str] | None = None,
    methods: str = "all",
) -> BenchmarkCommand:
    selected = benchmarks or list(DEFAULT_BENCHMARKS)
    return BenchmarkCommand(
        cwd=agent_root,
        argv=[
            "python3",
            "experiments/run_all.py",
            "--benchmarks",
            ",".join(selected),
            "--methods",
            methods,
            "--output",
            str(output),
        ],
    )
