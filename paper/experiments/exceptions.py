"""Shared experiment exceptions with no heavyweight dependencies."""
from __future__ import annotations


class PredictionUnavailable(RuntimeError):
    """Raised when a requested predictor cannot be used honestly."""
