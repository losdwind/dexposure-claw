"""Shared prediction helper for B1-B6 benchmarks.

Routes method_id to the appropriate prediction strategy:
- C0/C4: DeXposure-FM backbone (if available)
- C2: Persistence (G_{t+h} = G_t)
- Others: Persistence proxy with warning
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dexposure_agent.types import GraphSnapshot

logger = logging.getLogger(__name__)

_FM_PREDICTOR = None


def _get_fm():
    global _FM_PREDICTOR
    if _FM_PREDICTOR is None:
        try:
            from dexposure_agent.fm_predictor import FMPredictor
            _FM_PREDICTOR = FMPredictor()
            if _FM_PREDICTOR.available:
                logger.info("FM predictor ready")
            else:
                logger.warning("FM predictor: no checkpoints found")
        except Exception as e:
            logger.warning("FM predictor init failed: %s", e)
            _FM_PREDICTOR = None
    return _FM_PREDICTOR


def predict_graph(
    method_id: str,
    current_snapshot: GraphSnapshot,
    horizon: int,
) -> GraphSnapshot:
    """Generate predicted graph G_{t+h} for any method.

    Returns a NEW GraphSnapshot for FM methods, or the same object for persistence.
    Use `is` check to determine if prediction differs from input.
    """
    # C2: persistence baseline
    if method_id == "C2":
        return current_snapshot

    # C0/C4: FM backbone
    if method_id in ("C0", "C4"):
        fm = _get_fm()
        if fm is not None and fm.available:
            return fm.predict(current_snapshot, horizon)
        logger.debug("FM not available for %s, using persistence", method_id)
        return current_snapshot

    # All others: persistence proxy
    return current_snapshot
