"""Scenario Engine for the DeXposure-Agent pipeline.

Implements Section 2.6 of the paper: apply stress shocks S1-S5 to predicted
graphs, compute contagion losses, rank scenarios by expected loss.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from dexposure_agent.config import AgentConfig
from dexposure_agent.types import GraphSnapshot, Edge, ScenarioLoss, ScenarioSummary

# ---------------------------------------------------------------------------
# Scenario definitions (S1-S5)
# ---------------------------------------------------------------------------

SCENARIO_LIBRARY: dict[str, dict[str, Any]] = {
    "S1": {
        "name": "Single protocol failure",
        "type": "top_node",
        "shock_pct": 1.0,
        "count": 1,
    },
    "S2": {
        "name": "Bridge cluster failure",
        "type": "category",
        "categories": ["Bridge", "Cross Chain"],
        "shock_pct": 1.0,
    },
    "S3": {
        "name": "Stablecoin de-peg",
        "type": "category",
        "categories": ["Algo-Stables", "Decentralized Stablecoin", "CDP"],
        "shock_pct": 0.5,
    },
    "S4": {
        "name": "Sector-wide shock",
        "type": "category",
        "categories": ["Lending", "Uncollateralized Lending", "RWA Lending", "NFT Lending"],
        "shock_pct": 0.3,
    },
    "S5": {
        "name": "Correlated stress",
        "type": "top_nodes",
        "shock_pct": 0.2,
        "count": 10,
    },
}


# ---------------------------------------------------------------------------
# Helper: compute per-node total edge weight
# ---------------------------------------------------------------------------

def _node_total_weights(graph: GraphSnapshot) -> dict[str, float]:
    """Sum of all edge weights touching each node (source + target)."""
    totals: dict[str, float] = defaultdict(float)
    for edge in graph.edges:
        totals[edge.source] += edge.weight
        totals[edge.target] += edge.weight
    return dict(totals)


# ---------------------------------------------------------------------------
# apply_shock
# ---------------------------------------------------------------------------

def apply_shock(graph: GraphSnapshot, scenario_spec: dict[str, Any]) -> GraphSnapshot:
    """Return a new GraphSnapshot with edge weights modified per *scenario_spec*.

    Shock types
    -----------
    top_node   : zero all edges of the single node with the highest total weight.
    top_nodes  : reduce edges of the top-N nodes by shock_pct.
    category   : reduce edges of all nodes whose category matches by shock_pct.
    """
    shock_type = scenario_spec["type"]
    shock_pct = scenario_spec["shock_pct"]

    node_weights = _node_total_weights(graph)

    if shock_type == "top_node":
        count = scenario_spec.get("count", 1)
        shocked_nodes = {
            n for n, _ in sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:count]
        }
    elif shock_type == "top_nodes":
        count = scenario_spec.get("count", 10)
        shocked_nodes = {
            n for n, _ in sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:count]
        }
    elif shock_type == "category":
        # Support both singular "category" and plural "categories" (list)
        target_cats = scenario_spec.get("categories", [])
        if not target_cats:
            target_cats = [scenario_spec["category"]]
        target_cats_lower = {c.lower() for c in target_cats}
        shocked_nodes = {
            name
            for name, features in graph.nodes.items()
            if features.category.lower() in target_cats_lower
        }
    else:
        raise ValueError(f"Unknown scenario type: {shock_type!r}")

    # Build new edge list
    new_edges: list[Edge] = []
    for edge in graph.edges:
        if edge.source in shocked_nodes or edge.target in shocked_nodes:
            new_weight = edge.weight * (1.0 - shock_pct)
            if new_weight > 0.0:
                new_edges.append(Edge(source=edge.source, target=edge.target, weight=new_weight))
            # else: drop the edge (weight <= 0)
        else:
            new_edges.append(edge)

    return GraphSnapshot(date=graph.date, nodes=graph.nodes, edges=new_edges)


# ---------------------------------------------------------------------------
# compute_contagion_loss
# ---------------------------------------------------------------------------

def compute_contagion_loss(
    original_graph: GraphSnapshot,
    shocked_graph: GraphSnapshot,
    scenario_id: str,
    scenario_name: str,
    horizon: int,
) -> ScenarioLoss:
    """Compute loss metrics comparing *original_graph* to *shocked_graph*.

    Metrics
    -------
    expected_loss     : (sum_orig - sum_shocked) / sum_orig  (0 if sum_orig == 0)
    distressed_count  : nodes that lost > 50 % of their total edge weight
    propagation_depth : floor(log2(affected_nodes + 1)) + 1
    top_targets       : nodes with largest absolute weight loss (descending)
    """
    orig_weights = _node_total_weights(original_graph)
    shocked_weights = _node_total_weights(shocked_graph)

    sum_orig = sum(orig_weights.values())
    sum_shocked = sum(shocked_weights.values())

    if sum_orig > 0.0:
        expected_loss = (sum_orig - sum_shocked) / sum_orig
    else:
        expected_loss = 0.0

    # Distressed: lost > 50 % of original weight
    distressed_count = 0
    loss_per_node: dict[str, float] = {}
    for node, orig_w in orig_weights.items():
        shocked_w = shocked_weights.get(node, 0.0)
        delta = orig_w - shocked_w
        loss_per_node[node] = delta
        if orig_w > 0.0 and delta / orig_w > 0.5:
            distressed_count += 1

    # Propagation depth
    affected_nodes = sum(1 for d in loss_per_node.values() if d > 0.0)
    propagation_depth = int(math.floor(math.log2(affected_nodes + 1))) + 1 if affected_nodes > 0 else 0

    # Top targets
    top_targets = [
        node
        for node, _ in sorted(loss_per_node.items(), key=lambda x: x[1], reverse=True)
        if loss_per_node[node] > 0.0
    ]

    return ScenarioLoss(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        horizon=horizon,
        expected_loss=max(expected_loss, 0.0),
        cvar_loss=max(expected_loss, 0.0),  # single-sample CVaR equals the loss itself
        distressed_count=distressed_count,
        propagation_depth=propagation_depth,
        top_targets=top_targets,
    )


# ---------------------------------------------------------------------------
# run_scenarios
# ---------------------------------------------------------------------------

def run_scenarios(
    pred_graph: GraphSnapshot,
    mc_samples: list[GraphSnapshot],
    config: AgentConfig,
    horizon: int,
) -> ScenarioSummary:
    """Run all 5 scenarios on *pred_graph*, using *mc_samples* for CVaR.

    For each scenario
    -----------------
    1. Apply shock to pred_graph, compute contagion loss.
    2. Apply shock to every MC sample, collect expected_loss values.
    3. CVaR = average of the worst lambda_tail fraction of MC losses
       (falls back to the pred_graph loss when mc_samples is empty).

    Returns a ScenarioSummary ranked by expected_loss (descending).
    """
    lambda_tail = config.lambda_tail
    losses: list[ScenarioLoss] = []

    for sid, spec in SCENARIO_LIBRARY.items():
        # --- primary prediction ---
        shocked = apply_shock(pred_graph, spec)
        loss = compute_contagion_loss(
            pred_graph, shocked,
            scenario_id=sid,
            scenario_name=spec["name"],
            horizon=horizon,
        )

        # --- MC CVaR ---
        if mc_samples:
            mc_losses: list[float] = []
            for sample in mc_samples:
                s_shocked = apply_shock(sample, spec)
                s_loss = compute_contagion_loss(
                    sample, s_shocked,
                    scenario_id=sid,
                    scenario_name=spec["name"],
                    horizon=horizon,
                )
                mc_losses.append(s_loss.expected_loss)

            mc_losses_sorted = sorted(mc_losses, reverse=True)
            tail_n = max(1, int(math.ceil(len(mc_losses_sorted) * lambda_tail)))
            cvar = sum(mc_losses_sorted[:tail_n]) / tail_n
        else:
            cvar = loss.expected_loss

        losses.append(
            ScenarioLoss(
                scenario_id=loss.scenario_id,
                scenario_name=loss.scenario_name,
                horizon=loss.horizon,
                expected_loss=loss.expected_loss,
                cvar_loss=cvar,
                distressed_count=loss.distressed_count,
                propagation_depth=loss.propagation_depth,
                top_targets=loss.top_targets,
            )
        )

    # Rank by expected_loss descending
    ranked = sorted(losses, key=lambda sl: sl.expected_loss, reverse=True)

    worst = ranked[0] if ranked else None

    return ScenarioSummary(
        ranked_losses=ranked,
        worst_scenario=worst.scenario_id if worst else "",
        worst_horizon=worst.horizon if worst else 0,
    )
