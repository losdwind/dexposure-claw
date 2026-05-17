"""Filesystem run store helpers."""
from __future__ import annotations

from pathlib import Path


def ensure_run_dir(root: Path, run_id: str) -> Path:
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
