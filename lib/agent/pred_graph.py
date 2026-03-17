"""PredGraph Builder and Monte Carlo Sampler.

Builds a predicted graph G_hat from prediction distributions returned by the
GPU server, and draws Monte Carlo samples for uncertainty quantification.

Maps to Algorithm 1 Step 3 in the DeXposure-Agent pipeline.
"""
from __future__ import annotations

import numpy as np

from lib.agent.config import AgentConfig
from lib.agent.types import Edge, GraphSnapshot, NodeFeatures


def _stub_node_features() -> NodeFeatures:
    """Return a zero-feature NodeFeatures stub.

    Real features come from the original observed graph; these stubs are used
    for predicted graph nodes where feature values are not yet known.
    """
    return NodeFeatures(
        log_size=0.0,
        num_tokens=0,
        max_share=0.0,
        entropy=0.0,
        category="unknown",
    )


def build_pred_graph(
    prediction: dict,
    config: AgentConfig,
    date: str = "",
) -> GraphSnapshot:
    """Build a deterministic predicted graph G_hat from FM output distributions.

    Edges are included when their existence probability meets the threshold
    `config.pi_min`. The expected weight (mean) is used as the edge weight.

    Args:
        prediction: Dict with keys:
            - edge_probs: dict[(src, tgt) -> float]  existence probability
            - edge_weights: dict[(src, tgt) -> float]  predicted mean weight
            - weight_stds: dict[(src, tgt) -> float]   predicted std of weight
            - node_ids: list[str]  all node identifiers in the prediction
        config: AgentConfig with pi_min threshold.
        date: Optional date string for the snapshot.

    Returns:
        GraphSnapshot with filtered edges at their expected weights.
    """
    edge_probs: dict = prediction.get("edge_probs", {})
    edge_weights: dict = prediction.get("edge_weights", {})
    node_ids: list[str] = prediction.get("node_ids", [])

    edges: list[Edge] = []
    included_nodes: set[str] = set()

    for (src, tgt), prob in edge_probs.items():
        if prob >= config.pi_min:
            weight = edge_weights.get((src, tgt), 0.0)
            edges.append(Edge(source=src, target=tgt, weight=weight))
            included_nodes.add(src)
            included_nodes.add(tgt)

    # Only include nodes that appear in at least one accepted edge.
    # node_ids from the prediction is used as the universe, but we only
    # materialise stubs for nodes actually connected in G_hat.
    nodes: dict[str, NodeFeatures] = {
        nid: _stub_node_features() for nid in node_ids if nid in included_nodes
    }

    return GraphSnapshot(date=date, nodes=nodes, edges=edges)


def mc_sample(
    prediction: dict,
    config: AgentConfig,
    date: str = "",
    rng: np.random.Generator | None = None,
) -> list[GraphSnapshot]:
    """Draw Monte Carlo samples from the predicted graph distribution.

    For each sample:
      - Each edge is included via a Bernoulli draw with its existence probability.
      - If included, its weight is drawn from N(mean, std) and clamped to >= 0.

    Args:
        prediction: Same dict format as `build_pred_graph`.
        config: AgentConfig with mc_samples and pi_min.
        date: Optional date string propagated to each GraphSnapshot.
        rng: Optional numpy Generator for reproducibility. If None, uses the
             global numpy random state (respects np.random.seed()).

    Returns:
        List of `config.mc_samples` GraphSnapshot objects.
    """
    edge_probs: dict = prediction.get("edge_probs", {})
    edge_weights: dict = prediction.get("edge_weights", {})
    weight_stds: dict = prediction.get("weight_stds", {})
    node_ids: list[str] = prediction.get("node_ids", [])

    edge_list = list(edge_probs.keys())
    probs = np.array([edge_probs[e] for e in edge_list], dtype=float)
    means = np.array([edge_weights.get(e, 0.0) for e in edge_list], dtype=float)
    stds = np.array([weight_stds.get(e, 0.0) for e in edge_list], dtype=float)

    samples: list[GraphSnapshot] = []

    for _ in range(config.mc_samples):
        if rng is not None:
            bernoulli_draws = rng.random(len(edge_list)) < probs
            noise = rng.standard_normal(len(edge_list))
        else:
            bernoulli_draws = np.random.random(len(edge_list)) < probs
            noise = np.random.standard_normal(len(edge_list))

        sampled_weights = np.maximum(means + noise * stds, 0.0)

        edges: list[Edge] = []
        included_nodes: set[str] = set()

        for i, (src, tgt) in enumerate(edge_list):
            if bernoulli_draws[i]:
                edges.append(Edge(source=src, target=tgt, weight=float(sampled_weights[i])))
                included_nodes.add(src)
                included_nodes.add(tgt)

        nodes: dict[str, NodeFeatures] = {
            nid: _stub_node_features() for nid in node_ids if nid in included_nodes
        }

        samples.append(GraphSnapshot(date=date, nodes=nodes, edges=edges))

    return samples
