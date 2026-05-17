"""Result summarization helpers for DeXposure Claw."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(results_file: Path) -> dict[str, Any]:
    return json.loads(results_file.read_text())


def summarize_counts(results_file: Path) -> dict[str, int]:
    payload = load_results(results_file)
    summary = payload.get("summary", {})
    counts = summary.get("counts", {})
    return {str(key): int(value) for key, value in counts.items()}
