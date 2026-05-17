"""DataHealth Gate for the DeXposure-Agent pipeline.

Implements Section 2.4 of the DeXposure-Agent paper. Runs four deterministic
quality checks on a DeFi credit-exposure GraphSnapshot and returns a composite
score DH_t in [0,1] plus a SAFE_MODE flag.

Checks:
    freshness     - date field is parseable and non-empty
    missingness   - fraction of non-zero numeric features across all nodes
    topology      - edge density relative to node count
    discontinuity - always 1.0 for a single snapshot (multi-snapshot comparison
                    would detect temporal gaps in production)
"""
from __future__ import annotations

import logging
from datetime import datetime

from dexposure_agent.config import AgentConfig
from dexposure_agent.types import DataHealthResult, GraphSnapshot

logger = logging.getLogger(__name__)

# Numeric node feature names checked for missingness
_NUMERIC_FEATURES = ("log_size", "num_tokens", "max_share", "entropy")


def _check_freshness(graph: GraphSnapshot) -> float:
    """Return 1.0 if date is non-empty and ISO-parseable, else 0.0."""
    if not graph.date:
        return 0.0
    try:
        datetime.fromisoformat(graph.date)
        return 1.0
    except ValueError:
        logger.warning("DataHealth: unparseable date '%s'", graph.date)
        return 0.0


def _check_missingness(graph: GraphSnapshot) -> float:
    """Return fraction of non-zero numeric features across all nodes.

    An empty graph (no nodes) has no features to check; treat as fully missing
    so the score is 0.0.
    """
    nodes = graph.nodes
    if not nodes:
        return 0.0

    total = len(nodes) * len(_NUMERIC_FEATURES)
    non_zero = sum(
        1
        for features in nodes.values()
        for attr in _NUMERIC_FEATURES
        if getattr(features, attr) != 0
    )
    return non_zero / total


def _check_topology(graph: GraphSnapshot) -> float:
    """Return min(1.0, edge_count / max(node_count, 1)).

    Well-connected graphs score closer to 1.0; isolated or empty graphs
    score lower.
    """
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    return min(1.0, edge_count / max(node_count, 1))


def _check_discontinuity(_graph: GraphSnapshot) -> float:
    """Return 1.0 (single-snapshot baseline; production would compare to prior)."""
    return 1.0


def compute_data_health(graph: GraphSnapshot, config: AgentConfig) -> DataHealthResult:
    """Compute the DataHealth gate score DH_t for a single GraphSnapshot.

    Args:
        graph:  The current DeFi credit-exposure graph snapshot.
        config: Agent hyperparameters; uses config.tau_data as the safe-mode
                threshold.

    Returns:
        DataHealthResult with score DH_t in [0,1], safe_mode flag, and a
        per-check breakdown dict.
    """
    checks: dict[str, float] = {
        "freshness": _check_freshness(graph),
        "missingness": _check_missingness(graph),
        "topology": _check_topology(graph),
        "discontinuity": _check_discontinuity(graph),
    }

    dh_t = sum(checks.values()) / len(checks)
    safe_mode = dh_t < config.tau_data

    logger.info(
        "DataHealth | date=%s nodes=%d edges=%d DH_t=%.4f safe_mode=%s checks=%s",
        graph.date,
        len(graph.nodes),
        len(graph.edges),
        dh_t,
        safe_mode,
        checks,
    )

    return DataHealthResult(score=dh_t, safe_mode=safe_mode, checks=checks)
