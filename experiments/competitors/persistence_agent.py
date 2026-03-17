#!/usr/bin/env python3
"""C2: Persistence-Agent competitor wrapper.

Naive baseline: the predicted graph at horizon h is identical to the
most recently observed graph snapshot, i.e. G_{t+h} = G_t.

Applicable to: B1, B2, B4, B5 (not B3 -- no uncertainty; not B6 -- no training).

This wrapper computes risk metrics from the persisted snapshot and
generates null / no-change action recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from loguru import logger


@dataclass
class PersistenceAgentConfig:
    horizon: int = 4   # retained for API parity, but ignored in prediction
    rolling_window: int = 1   # only uses the most recent snapshot


@dataclass
class PersistenceAgentPrediction:
    """Output format shared across all agent-level competitors."""
    method_id: str = "C2"
    horizon: int = 4
    pagerank_pred: dict[str, float] = field(default_factory=dict)
    hhi_pred: float = float("nan")
    density_pred: float = float("nan")
    gini_pred: float = float("nan")
    risk_scores: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)   # empty -- no uncertainty
    recommended_actions: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"PersistenceAgentPrediction(h={self.horizon}, "
            f"n_protocols={len(self.risk_scores)})"
        )


def run_persistence_agent(
    graph: Any,
    config: PersistenceAgentConfig | None = None,
    **kwargs,
) -> PersistenceAgentPrediction:
    """Run the Persistence-Agent: copy the latest snapshot forward.

    Args:
        graph: Temporal graph object. Only the last snapshot is used.
               Expected to support .snapshots, .node_features, .edge_index.
        config: PersistenceAgentConfig. Uses defaults if None.
        **kwargs: Extra config key-value pairs that override config fields.

    Returns:
        PersistenceAgentPrediction where predictions == current observation.
    """
    if config is None:
        config = PersistenceAgentConfig()
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    logger.info(f"Persistence-Agent (C2) | horizon={config.horizon} (ignored)")
    # TODO: extract latest snapshot from graph
    # TODO: compute network risk metrics (PageRank, HHI, density, Gini)
    #       directly from the current snapshot (no forecasting)
    # TODO: generate null action recommendations (no change predicted)
    raise NotImplementedError("Persistence-Agent (C2) not yet implemented")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="C2: Persistence-Agent competitor")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08")
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()

    cfg = PersistenceAgentConfig(horizon=args.horizon)
    # TODO: load graph from args.data_dir
    pred = run_persistence_agent(graph=None, config=cfg)
    print(pred)
