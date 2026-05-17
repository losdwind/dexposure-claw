"""Append-only event payload helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def event(kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "payload": payload or {},
    }
