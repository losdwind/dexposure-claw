"""Orchestration placeholders for DeXposure benchmark workflows."""
from __future__ import annotations

from pathlib import Path

from dexposure_claw.tools.benchmark import BenchmarkCommand, build_run_all_command


def plan_full_suite(agent_root: Path, output: Path) -> list[BenchmarkCommand]:
    return [build_run_all_command(agent_root=agent_root, output=output)]
