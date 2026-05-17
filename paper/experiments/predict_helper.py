"""Shared prediction helper for b1_forecast/b3_calibration/b4_stress/b5_decision/b6_robustness.

Routes method IDs through the canonical registry and fails closed when a
requested predictor is unavailable. This prevents unimplemented baselines from
silently producing persistence results.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dexposure_agent.config import AgentConfig
from dexposure_agent.types import GraphSnapshot
from experiments.exceptions import PredictionUnavailable
from experiments.methods import require_implemented

logger = logging.getLogger(__name__)

_FM_PREDICTOR = None
_EVOLVEGCN_PREDICTOR = None


def _get_fm(config: AgentConfig | None = None):
    global _FM_PREDICTOR
    if _FM_PREDICTOR is None:
        try:
            from dexposure_agent.fm_predictor import FMPredictor
            cfg = config or AgentConfig()
            _FM_PREDICTOR = FMPredictor(pi_min=cfg.pi_min)
            if _FM_PREDICTOR.available:
                logger.info("FM predictor ready")
            else:
                logger.warning("FM predictor: no checkpoints found")
        except Exception as e:
            logger.warning("FM predictor init failed: %s", e)
            _FM_PREDICTOR = None
    return _FM_PREDICTOR


def _get_evolvegcn():
    global _EVOLVEGCN_PREDICTOR
    if _EVOLVEGCN_PREDICTOR is None:
        try:
            from experiments.competitors.evolvegcn import EvolveGCNPredictor
            _EVOLVEGCN_PREDICTOR = EvolveGCNPredictor()
            if _EVOLVEGCN_PREDICTOR.available:
                logger.info("EvolveGCN predictor ready")
            else:
                logger.warning("EvolveGCN predictor: no checkpoints found")
        except Exception as e:
            logger.warning("EvolveGCN predictor init failed: %s", e)
            _EVOLVEGCN_PREDICTOR = None
    return _EVOLVEGCN_PREDICTOR


def predict_graph(
    method_id: str,
    current_snapshot: GraphSnapshot,
    horizon: int,
    config: AgentConfig | None = None,
) -> GraphSnapshot:
    """Generate predicted graph G_{t+h} for a supported method.

    Returns a NEW GraphSnapshot for FM/GNN methods, or the same object for persistence.
    Raises PredictionUnavailable instead of substituting another method.
    """
    spec = require_implemented(method_id)

    if spec.predictor == "persistence":
        return current_snapshot

    if spec.predictor == "fm":
        fm = _get_fm(config)
        if fm is not None and fm.available:
            return fm.predict(current_snapshot, horizon)
        raise PredictionUnavailable(
            f"{method_id} ({spec.label}) requires FM checkpoints, but FM is unavailable"
        )

    if spec.predictor == "evolvegcn":
        egcn = _get_evolvegcn()
        if egcn is not None and egcn.available:
            return egcn.predict(current_snapshot, horizon)
        raise PredictionUnavailable(
            f"{method_id} ({spec.label}) requires trained EvolveGCN checkpoints"
        )

    if spec.predictor == "current":
        raise PredictionUnavailable(
            f"{method_id} ({spec.label}) is not a graph-forecasting method"
        )

    raise PredictionUnavailable(
        f"{method_id} ({spec.label}) has unsupported predictor={spec.predictor!r}"
    )
