#!/usr/bin/env python3
"""C4-C10: Non-agent baseline wrappers.

These baselines produce link-level and node-level predictions (not
action recommendations) and are evaluated primarily on B1, B4, B6.

C4:  DeXposure-FM        -- fine-tuned GraphPFN (our FM without the agent)
C5:  ROLAND              -- GCN+GRU from scratch (no agent layer)
C6:  GraphPFN-Frozen     -- GraphPFN encoder frozen, only scorer trained
C7:  EvolveGCN           -- EvolveGCN-O/H (Pareja et al., 2020)
C8:  DyRep               -- DyRep (Trivedi et al., 2019)
C9:  TGN                 -- Temporal Graph Networks (Rossi et al., 2020)
C10: Static GCN          -- single-snapshot GCN, no temporal modelling
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Shared prediction container for non-agent baselines
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselinePrediction:
    """Prediction output for C4-C10 (edge + node level, no actions)."""
    method_id: str
    horizon: int = 4
    edge_exist_prob: dict[tuple[str, str], float] = field(default_factory=dict)
    edge_weight_pred: dict[tuple[str, str], float] = field(default_factory=dict)
    node_delta_pred: dict[str, float] = field(default_factory=dict)  # TVL log-change
    # Derived risk metrics (computed from edge/node predictions)
    pagerank_pred: dict[str, float] = field(default_factory=dict)
    hhi_pred: float = float("nan")
    density_pred: float = float("nan")
    gini_pred: float = float("nan")

    def __str__(self) -> str:
        return (
            f"BaselinePrediction(method={self.method_id}, h={self.horizon}, "
            f"n_edges={len(self.edge_exist_prob)}, n_nodes={len(self.node_delta_pred)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Individual baseline runners
# ──────────────────────────────────────────────────────────────────────────────

def run_dexposure_fm(graph: Any, horizon: int = 4, checkpoint_path: Optional[str] = None,
                     **kwargs) -> BaselinePrediction:
    """C4: DeXposure-FM -- fine-tuned GraphPFN without the agent layer."""
    logger.info(f"DeXposure-FM (C4) | horizon={horizon}")
    # TODO: load fine-tuned GraphPFN from checkpoint_path
    # TODO: run encoder + link scorer + node head forward pass
    # TODO: decode edge probabilities and node delta predictions
    # TODO: compute derived risk metrics
    raise NotImplementedError("DeXposure-FM (C4) not yet implemented")


def run_roland(graph: Any, horizon: int = 4, checkpoint_path: Optional[str] = None,
               hidden_dim: int = 256, num_layers: int = 3, **kwargs) -> BaselinePrediction:
    """C5: ROLAND -- GCN+GRU trained from scratch without the agent layer."""
    logger.info(f"ROLAND (C5) | horizon={horizon} | hidden_dim={hidden_dim}")
    # TODO: instantiate ROLAND backbone
    # TODO: load checkpoint if provided
    # TODO: run temporal forward pass over snapshot sequence
    # TODO: decode predictions
    raise NotImplementedError("ROLAND (C5) not yet implemented")


def run_graphpfn_frozen(graph: Any, horizon: int = 4, checkpoint_path: Optional[str] = None,
                        **kwargs) -> BaselinePrediction:
    """C6: GraphPFN-Frozen -- encoder weights frozen, only scorer is trained."""
    logger.info(f"GraphPFN-Frozen (C6) | horizon={horizon}")
    # TODO: load GraphPFN with encoder frozen
    # TODO: run scorer-only forward pass
    raise NotImplementedError("GraphPFN-Frozen (C6) not yet implemented")


def run_evolvegcn(graph: Any, horizon: int = 4, checkpoint_path: Optional[str] = None,
                  variant: str = "O", **kwargs) -> BaselinePrediction:
    """C7: EvolveGCN-O/H (Pareja et al., KDD 2020).

    Args:
        variant: 'O' (EvolveGCN-O, ODE-based) or 'H' (EvolveGCN-H, hypernetwork).
    """
    logger.info(f"EvolveGCN-{variant} (C7) | horizon={horizon}")
    # TODO: instantiate EvolveGCN with specified variant
    # TODO: run temporal forward pass
    raise NotImplementedError(f"EvolveGCN-{variant} (C7) not yet implemented")


def run_dyrep(graph: Any, horizon: int = 4, checkpoint_path: Optional[str] = None,
              **kwargs) -> BaselinePrediction:
    """C8: DyRep (Trivedi et al., ICLR 2019) -- continuous-time dynamic graph."""
    logger.info(f"DyRep (C8) | horizon={horizon}")
    # TODO: instantiate DyRep with node association + communication processes
    # TODO: run event-driven temporal forward pass
    raise NotImplementedError("DyRep (C8) not yet implemented")


def run_tgn(graph: Any, horizon: int = 4, checkpoint_path: Optional[str] = None,
            memory_dim: int = 100, **kwargs) -> BaselinePrediction:
    """C9: TGN -- Temporal Graph Networks (Rossi et al., NeurIPS 2020)."""
    logger.info(f"TGN (C9) | horizon={horizon} | memory_dim={memory_dim}")
    # TODO: instantiate TGN with memory module and graph attention aggregator
    # TODO: run message-passing + memory update forward pass
    raise NotImplementedError("TGN (C9) not yet implemented")


def run_static_gcn(graph: Any, horizon: int = 4, checkpoint_path: Optional[str] = None,
                   **kwargs) -> BaselinePrediction:
    """C10: Static GCN -- single-snapshot GCN, no temporal modelling."""
    logger.info(f"Static GCN (C10) | horizon={horizon}")
    # TODO: stack all snapshots as a single static graph or use only latest
    # TODO: run two-layer GCN forward pass
    raise NotImplementedError("Static GCN (C10) not yet implemented")


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch function
# ──────────────────────────────────────────────────────────────────────────────

_BASELINE_REGISTRY = {
    "C4":  run_dexposure_fm,
    "C5":  run_roland,
    "C6":  run_graphpfn_frozen,
    "C7":  run_evolvegcn,
    "C8":  run_dyrep,
    "C9":  run_tgn,
    "C10": run_static_gcn,
}


def run_baseline(
    method_id: str,
    graph: Any,
    horizon: int = 4,
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> BaselinePrediction:
    """Dispatch to the appropriate C4-C10 baseline runner.

    Args:
        method_id: One of 'C4' through 'C10'.
        graph: Temporal graph object.
        horizon: Forecasting horizon in weeks.
        checkpoint_path: Path to saved model checkpoint (optional).
        **kwargs: Extra method-specific config.

    Returns:
        BaselinePrediction with edge/node predictions and derived risk metrics.

    Raises:
        ValueError: If method_id is not a known baseline.
        NotImplementedError: Until the baseline is implemented.
    """
    if method_id not in _BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline method: {method_id!r}. "
            f"Must be one of: {list(_BASELINE_REGISTRY)}"
        )
    runner = _BASELINE_REGISTRY[method_id]
    return runner(graph=graph, horizon=horizon, checkpoint_path=checkpoint_path, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="C4-C10: Baseline competitors")
    parser.add_argument("--method", required=True,
                        choices=list(_BASELINE_REGISTRY),
                        help="Baseline method ID (C4-C10)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    args = parser.parse_args()

    # TODO: load graph from args.data_dir filtered by args.test_split
    pred = run_baseline(
        method_id=args.method,
        graph=None,
        horizon=args.horizon,
        checkpoint_path=args.checkpoint,
    )
    print(pred)
