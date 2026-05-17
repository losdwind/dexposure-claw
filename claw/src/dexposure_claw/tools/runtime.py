"""Runtime utility helpers."""
from __future__ import annotations

from pathlib import Path


def list_runs(results_root: Path) -> list[str]:
    if not results_root.exists():
        return []
    return sorted(path.name for path in results_root.iterdir() if path.is_dir())
