"""Run manifest generation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def new_manifest(run_id: str, target: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "target": target,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
