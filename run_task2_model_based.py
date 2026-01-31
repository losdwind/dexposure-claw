#!/usr/bin/env python3
"""
Task II: Model-based Forward-looking Financial Stability Analysis

This script implements the IMPROVED Task II experiments that demonstrate
the Foundation Model's predictive capabilities for risk assessment.

Core Experiments:
1. Forward-looking Risk Metric Prediction
   - Use DeXposure-FM to predict network at t+h
   - Compute risk metrics on predicted network
   - Compare with actual risk metrics at t+h

2. Predictive Contagion Simulation
   - Run contagion simulation on predicted networks
   - Compare: observed(t) vs predicted(t->t+h) vs actual(t+h)
   - Show forward-looking stress testing capability

3. Shock Event Early Warning
   - Predict network structure before Terra/Luna, FTX events
   - Check if predicted risk metrics rise before actual shocks
   - Demonstrate early warning capability

Key Insight: Previous experiments only used descriptive statistics on OBSERVED
networks. This version shows how the trained model enables FORWARD-LOOKING
risk assessment - the core value proposition of a Foundation Model.

Usage:
    # Run on GPU server
    ssh gpu-server
    cd /path/to/graph-dexposure
    python run_task2_model_based.py --experiment all
    python run_task2_model_based.py --experiment forward_risk
    python run_task2_model_based.py --experiment predictive_contagion
    python run_task2_model_based.py --experiment early_warning
"""

import argparse
import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import torch

# Figure configuration for publication quality
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add project paths
GRAPHPFN_ROOT = Path(__file__).parent
sys.path.insert(0, str(GRAPHPFN_ROOT))
sys.path.insert(0, str(GRAPHPFN_ROOT / "src"))

# Import from main experiment script
from run_full_experiment import (
    GRAPHPFN_AVAILABLE,
    ExperimentConfig,
    GraphPFNLinkPredictor,
    WeekPair,
    build_snapshot,
    build_week_pairs,
    get_single_split,
    load_graphpfn_encoder,
    load_metadata,
    load_network_data,
    predict_graphpfn,
    set_seed,
    train_graphpfn_epoch,
)


# ============== Snapshot Format Adapter ==============


def array_snap_to_dict_snap(
    snap: Dict[str, Any],
    edge_weight_is_log: bool = True,
) -> Dict[str, Any]:
    """
    Convert array-format snapshot to dict-format snapshot for contagion/SIS functions.

    Array format (from reconstruct_network_from_predictions):
        - node_ids: List[str]
        - sizes: np.ndarray (TVL values)
        - edge_src: np.ndarray (source indices)
        - edge_dst: np.ndarray (destination indices)
        - edge_weight: np.ndarray (log1p or linear weights)
        - categories: List[str]

    Dict format (expected by simulate_contagion, compute_systemic_importance_score):
        - nodes: Dict[str, {"tvlUsd": float, "category": str}]
        - edges: List[{"source": str, "target": str, "weight": float}]
    """
    # Check if already in dict format
    if "nodes" in snap and isinstance(snap.get("nodes"), dict):
        return snap

    node_ids = snap.get("node_ids", [])
    sizes = snap.get("sizes", [])
    categories = snap.get("categories", [])
    edge_src = snap.get("edge_src", [])
    edge_dst = snap.get("edge_dst", [])
    edge_weight = snap.get("edge_weight", [])

    # Build nodes dict
    nodes = {}
    for i, nid in enumerate(node_ids):
        tvl = float(sizes[i]) if i < len(sizes) else 0.0
        cat = categories[i] if i < len(categories) else "Unknown"
        nodes[nid] = {"tvlUsd": tvl, "category": cat}

    # Build edges list
    edges = []
    for i in range(len(edge_src)):
        src_idx = int(edge_src[i])
        dst_idx = int(edge_dst[i])
        if src_idx < len(node_ids) and dst_idx < len(node_ids):
            w = float(edge_weight[i]) if i < len(edge_weight) else 0.0
            # Convert from log scale if needed
            if edge_weight_is_log:
                w = float(np.expm1(w))
                w = max(0.0, w)
            edges.append({
                "source": node_ids[src_idx],
                "target": node_ids[dst_idx],
                "weight": w,
            })

    return {"nodes": nodes, "edges": edges, "date": snap.get("date", "")}


# ============== Risk Analysis Functions (Task II specific) ==============


def gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient of a distribution."""
    values = np.array(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def compute_risk_metrics_from_arrays(
    node_ids: List[str],
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_weights: np.ndarray,
    node_tvls: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive systemic risk metrics from array inputs using NetworkX.

    This function is designed for analyzing predicted networks where you have
    raw arrays of edges and node properties.

    Returns:
        Dictionary with risk metrics: PageRank, density, degree distribution, HHI, etc.
    """
    n_nodes = len(node_ids)
    if n_nodes == 0 or len(edge_src) == 0:
        return {"error": "Empty network"}

    # Build NetworkX graph
    G = nx.DiGraph()
    for i, node_id in enumerate(node_ids):
        G.add_node(i, tvl=float(node_tvls[i]) if i < len(node_tvls) else 0)

    for src, dst, w in zip(edge_src, edge_dst, edge_weights):
        if src < n_nodes and dst < n_nodes:
            # Convert from log1p-scale
            G.add_edge(int(src), int(dst), weight=float(np.expm1(w)))

    metrics = {}

    # 1. PageRank (weighted by edge weight)
    try:
        pagerank = nx.pagerank(G, weight="weight", max_iter=100)
        metrics["pagerank_mean"] = float(np.mean(list(pagerank.values())))
        metrics["pagerank_max"] = float(np.max(list(pagerank.values())))
        metrics["pagerank_gini"] = gini_coefficient(list(pagerank.values()))
    except Exception:
        metrics["pagerank_mean"] = 0.0
        metrics["pagerank_max"] = 0.0
        metrics["pagerank_gini"] = 0.0

    # 2. Network density
    metrics["density"] = nx.density(G) if G.number_of_nodes() > 1 else 0.0
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()

    # 3. Degree distribution
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    if in_degrees:
        metrics["in_degree_mean"] = float(np.mean(in_degrees))
        metrics["in_degree_max"] = float(np.max(in_degrees))
        metrics["in_degree_gini"] = gini_coefficient(in_degrees)
    if out_degrees:
        metrics["out_degree_mean"] = float(np.mean(out_degrees))
        metrics["out_degree_max"] = float(np.max(out_degrees))

    # 4. Edge weight concentration (HHI)
    if len(edge_weights) > 0:
        weights_linear = np.expm1(edge_weights)
        weights_linear = np.maximum(weights_linear, 0)
        total_weight = np.sum(weights_linear)
        if total_weight > 0:
            shares = weights_linear / total_weight
            metrics["edge_weight_hhi"] = float(np.sum(shares**2))
            # Top-5 concentration
            sorted_shares = np.sort(shares)[::-1]
            metrics["top5_edge_concentration"] = float(np.sum(sorted_shares[:5]))
        else:
            metrics["edge_weight_hhi"] = 0.0
            metrics["top5_edge_concentration"] = 0.0

    # 5. TVL concentration
    if len(node_tvls) > 0:
        total_tvl = np.sum(node_tvls)
        if total_tvl > 0:
            tvl_shares = node_tvls / total_tvl
            metrics["tvl_hhi"] = float(np.sum(tvl_shares**2))
            sorted_tvl = np.sort(tvl_shares)[::-1]
            metrics["top10_tvl_concentration"] = float(np.sum(sorted_tvl[:10]))
        metrics["total_tvl"] = float(total_tvl)
        metrics["tvl_gini"] = gini_coefficient(node_tvls.tolist())

    return metrics


def compute_systemic_importance_score(
    snap: Dict,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> Dict[str, float]:
    """
    Compute systemic importance score for each node.

    Score = alpha * (TVL share) + beta * (in-degree centrality) + gamma * (out-degree centrality)
    """
    nodes = snap.get("nodes", {})
    edges = snap.get("edges", [])

    if not nodes:
        return {}

    # TVL share
    tvl_values = {n: data.get("tvlUsd", 0) for n, data in nodes.items()}
    total_tvl = sum(tvl_values.values())
    tvl_share = {n: v / total_tvl if total_tvl > 0 else 0 for n, v in tvl_values.items()}

    # Degree centrality
    in_degree = {n: 0 for n in nodes}
    out_degree = {n: 0 for n in nodes}
    for edge in edges:
        src, dst = edge.get("source"), edge.get("target")
        if src in out_degree:
            out_degree[src] += 1
        if dst in in_degree:
            in_degree[dst] += 1

    max_degree = len(nodes) - 1 if len(nodes) > 1 else 1
    in_centrality = {n: d / max_degree for n, d in in_degree.items()}
    out_centrality = {n: d / max_degree for n, d in out_degree.items()}

    # Combined score
    scores = {}
    for n in nodes:
        scores[n] = (
            alpha * tvl_share.get(n, 0) +
            beta * in_centrality.get(n, 0) +
            gamma * out_centrality.get(n, 0)
        )

    return scores


def compute_sector_spillover_index(
    snap: Dict,
    sector_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute sector-level spillover index based on cross-sector edge weights.
    """
    nodes = snap.get("nodes", {})
    edges = snap.get("edges", [])

    # Default sector mapping based on protocol category
    if sector_map is None:
        sector_map = {}
        for node_id, data in nodes.items():
            category = data.get("category", "other")
            sector_map[node_id] = category

    # Aggregate cross-sector exposures
    sector_exposure = {}
    for edge in edges:
        src, dst = edge.get("source"), edge.get("target")
        weight = edge.get("weight", 0)

        src_sector = sector_map.get(src, "other")
        dst_sector = sector_map.get(dst, "other")

        if src_sector != dst_sector:
            if src_sector not in sector_exposure:
                sector_exposure[src_sector] = {}
            if dst_sector not in sector_exposure[src_sector]:
                sector_exposure[src_sector][dst_sector] = 0
            sector_exposure[src_sector][dst_sector] += weight

    return sector_exposure


def simulate_contagion(
    snap: Dict,
    shocked_nodes: List[str],
    shock_fraction: float = 1.0,
    distress_threshold: float = 0.1,
    max_rounds: int = 10,
) -> Dict[str, Any]:
    """
    Simulate DebtRank-style contagion from shocked nodes.

    Args:
        snap: Network snapshot with nodes and edges
        shocked_nodes: List of initially shocked node IDs
        shock_fraction: Fraction of TVL lost by shocked nodes (0-1)
        distress_threshold: Loss/TVL ratio that triggers distress
        max_rounds: Maximum propagation rounds

    Returns:
        Dictionary with contagion results
    """
    nodes = snap.get("nodes", {})
    edges = snap.get("edges", [])

    # Initialize TVL and losses
    tvl = {n: data.get("tvlUsd", 0) for n, data in nodes.items()}
    losses = {n: 0.0 for n in nodes}

    # Build exposure index for propagation.
    # We treat an edge `creditor -> debtor` as "creditor has exposure to debtor".
    # When debtor is distressed, losses propagate to its creditors (incoming neighbors).
    exposures_in = {}
    for edge in edges:
        src, dst = edge.get("source"), edge.get("target")
        weight = float(edge.get("weight", 0) or 0.0)
        if src is None or dst is None:
            continue
        if dst not in exposures_in:
            exposures_in[dst] = {}
        exposures_in[dst][src] = exposures_in[dst].get(src, 0.0) + weight

    # Initial shock
    distressed = set()
    for node in shocked_nodes:
        if node in tvl:
            losses[node] = shock_fraction * tvl[node]
            distressed.add(node)

    # Propagation rounds
    affected_history = [len(distressed)]
    for round_num in range(max_rounds):
        new_distressed = set()

        for node in distressed:
            # Propagate debtor losses to its creditors
            for creditor, exposure in exposures_in.get(node, {}).items():
                if creditor not in distressed:
                    # Loss proportional to exposure share
                    loss_share = exposure / tvl[node] if tvl[node] > 0 else 0
                    propagated_loss = losses[node] * loss_share
                    losses[creditor] += propagated_loss

                    # Check if creditor becomes distressed
                    if losses[creditor] > distress_threshold * tvl.get(creditor, 0):
                        new_distressed.add(creditor)

        if not new_distressed:
            break

        distressed = distressed.union(new_distressed)
        affected_history.append(len(distressed))

    # Compute aggregate statistics
    total_tvl = sum(tvl.values())
    total_loss = sum(losses.values())

    return {
        "shocked_nodes": shocked_nodes,
        "shock_fraction": shock_fraction,
        "total_loss": total_loss,
        "total_loss_pct": 100.0 * total_loss / total_tvl if total_tvl > 0 else 0,
        "loss_fraction": total_loss / total_tvl if total_tvl > 0 else 0,
        "affected_count": len(distressed),
        "distressed_count": len(distressed),
        "distressed_nodes": list(distressed),
        "propagation_rounds": len(affected_history) - 1,
        "affected_history": affected_history,
    }

# ============== Configuration ==============

OUTPUT_DIR = Path("output/task2_model_based")
MODEL_CACHE_DIR = Path("output/model_cache")

SHOCK_EVENTS = {
    "terra_luna": {
        "name": "Terra/Luna Collapse",
        "event_date": "2022-05-09",
        "pre_window": ("2022-03-28", "2022-05-02"),  # 5 weeks before
        "event_window": ("2022-05-02", "2022-05-23"),
        "description": "UST algorithmic stablecoin depeg and death spiral",
    },
    "ftx": {
        "name": "FTX Collapse",
        "event_date": "2022-11-07",
        "pre_window": ("2022-09-26", "2022-10-31"),  # 5 weeks before
        "event_window": ("2022-10-31", "2022-11-21"),
        "description": "Exchange failure and FTT token collapse",
    },
}


# ============== Visualization Functions ==============

COLORS = {
    "predicted": "#2E86AB",  # Blue
    "actual": "#E94F37",     # Red
    "observed": "#7B8D8E",   # Gray
    "event": "#F39C12",      # Orange
}


def plot_early_warning_timeseries(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Plot Early Warning time series for Terra/Luna and FTX events.

    Creates a 2-row figure showing:
    - Risk metrics (HHI, Density) over time
    - Predicted vs Actual comparison
    - Event date markers
    """
    events_data = results.get("events", {})
    if not events_data:
        logger.warning("No early warning data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Early Warning Analysis: Model Predictions Before Major Events", fontsize=14, fontweight="bold")

    for row_idx, (event_id, event_data) in enumerate(events_data.items()):
        if row_idx >= 2:
            break

        event_info = event_data.get("event_info", {})
        predictions = event_data.get("predictions", [])
        baseline = event_data.get("baseline_metrics", {})

        if not predictions:
            continue

        event_name = event_info.get("name", event_id)
        event_date = event_info.get("event_date", "")

        # Extract time series data
        dates = [p["target_date"] for p in predictions]
        pred_hhi = [p["predicted_metrics"].get("tvl_hhi", np.nan) for p in predictions]
        actual_hhi = [p["actual_metrics"].get("tvl_hhi", np.nan) for p in predictions]
        pred_contagion = [p.get("predicted_contagion_loss", np.nan) for p in predictions]
        actual_contagion = [p.get("actual_contagion_loss", np.nan) for p in predictions]

        baseline_hhi = baseline.get("tvl_hhi", np.nan)

        # Plot HHI (left column)
        ax1 = axes[row_idx, 0]
        x_pos = np.arange(len(dates))

        ax1.plot(x_pos, pred_hhi, "o-", color=COLORS["predicted"], label="Predicted", linewidth=2, markersize=6)
        ax1.plot(x_pos, actual_hhi, "s--", color=COLORS["actual"], label="Actual", linewidth=2, markersize=6)
        ax1.axhline(y=baseline_hhi, color=COLORS["observed"], linestyle=":", linewidth=1.5, label=f"Baseline (pre-event)")

        # Mark event window
        ax1.axvspan(-0.5, len(dates)-0.5, alpha=0.1, color=COLORS["event"], label="Event window")

        ax1.set_xlabel("Weeks into event period")
        ax1.set_ylabel("TVL HHI (Concentration)")
        ax1.set_title(f"{event_name}: Risk Concentration")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"W{i+1}" for i in range(len(dates))], rotation=45)
        ax1.legend(loc="best", framealpha=0.9)
        ax1.set_ylim(bottom=0)

        # Plot Contagion Loss (right column)
        ax2 = axes[row_idx, 1]

        valid_pred = [v if not np.isnan(v) else 0 for v in pred_contagion]
        valid_actual = [v if not np.isnan(v) else 0 for v in actual_contagion]

        width = 0.35
        ax2.bar(x_pos - width/2, valid_pred, width, color=COLORS["predicted"], label="Predicted", alpha=0.8)
        ax2.bar(x_pos + width/2, valid_actual, width, color=COLORS["actual"], label="Actual", alpha=0.8)

        ax2.set_xlabel("Weeks into event period")
        ax2.set_ylabel("Contagion Loss (%)")
        ax2.set_title(f"{event_name}: Stress Test Loss")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"W{i+1}" for i in range(len(dates))], rotation=45)
        ax2.legend(loc="best", framealpha=0.9)
        ax2.set_ylim(bottom=0)

        # Add annotation for prediction accuracy
        if valid_pred and valid_actual:
            mae = np.mean(np.abs(np.array(valid_pred) - np.array(valid_actual)))
            ax2.text(0.95, 0.95, f"MAE: {mae:.2f}%", transform=ax2.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    fig_path = output_dir / "figures" / "fig_early_warning.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix(".png"))
    plt.close()

    logger.info(f"Saved Early Warning figure to {fig_path}")


def plot_contagion_comparison(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Plot Contagion simulation comparison: Observed(t) vs Predicted(t+h) vs Actual(t+h).

    Creates grouped bar chart showing system loss across different shock scenarios.
    """
    horizons_data = results.get("horizons", {})
    if not horizons_data:
        logger.warning("No contagion data to plot")
        return

    # Collect data across horizons
    scenarios = ["Top Protocol Shock", "Top 5 Protocols Shock", "Bridge Sector Shock"]

    fig, axes = plt.subplots(1, len(horizons_data), figsize=(4 * len(horizons_data), 5), sharey=True)
    if len(horizons_data) == 1:
        axes = [axes]

    fig.suptitle("Predictive Contagion: Forward-looking Stress Testing", fontsize=14, fontweight="bold")

    for ax_idx, (horizon_key, horizon_data) in enumerate(horizons_data.items()):
        ax = axes[ax_idx]
        samples = horizon_data.get("samples", [])

        if not samples:
            continue

        # Aggregate across samples for each scenario
        scenario_results = {s: {"observed": [], "predicted": [], "actual": []} for s in scenarios}

        for sample in samples:
            for scenario_name, scenario_data in sample.get("scenarios", {}).items():
                if scenario_name in scenario_results:
                    scenario_results[scenario_name]["observed"].append(
                        scenario_data.get("observed_t", {}).get("total_loss_pct", 0)
                    )
                    scenario_results[scenario_name]["predicted"].append(
                        scenario_data.get("predicted_t_h", {}).get("total_loss_pct", 0)
                    )
                    scenario_results[scenario_name]["actual"].append(
                        scenario_data.get("actual_t_h", {}).get("total_loss_pct", 0)
                    )

        # Plot grouped bars
        x = np.arange(len(scenarios))
        width = 0.25

        observed_means = [np.mean(scenario_results[s]["observed"]) if scenario_results[s]["observed"] else 0 for s in scenarios]
        predicted_means = [np.mean(scenario_results[s]["predicted"]) if scenario_results[s]["predicted"] else 0 for s in scenarios]
        actual_means = [np.mean(scenario_results[s]["actual"]) if scenario_results[s]["actual"] else 0 for s in scenarios]

        observed_stds = [np.std(scenario_results[s]["observed"]) if len(scenario_results[s]["observed"]) > 1 else 0 for s in scenarios]
        predicted_stds = [np.std(scenario_results[s]["predicted"]) if len(scenario_results[s]["predicted"]) > 1 else 0 for s in scenarios]
        actual_stds = [np.std(scenario_results[s]["actual"]) if len(scenario_results[s]["actual"]) > 1 else 0 for s in scenarios]

        bars1 = ax.bar(x - width, observed_means, width, yerr=observed_stds,
                       label="Observed (t)", color=COLORS["observed"], alpha=0.8, capsize=3)
        bars2 = ax.bar(x, predicted_means, width, yerr=predicted_stds,
                       label="Predicted (t+h)", color=COLORS["predicted"], alpha=0.8, capsize=3)
        bars3 = ax.bar(x + width, actual_means, width, yerr=actual_stds,
                       label="Actual (t+h)", color=COLORS["actual"], alpha=0.8, capsize=3)

        ax.set_xlabel("Shock Scenario")
        ax.set_ylabel("System Loss (%)" if ax_idx == 0 else "")
        ax.set_title(f"Horizon {horizon_key}")
        ax.set_xticks(x)
        ax.set_xticklabels(["Top-1", "Top-5", "Bridge"], rotation=0)

        if ax_idx == 0:
            ax.legend(loc="upper left", framealpha=0.9)

        # Add prediction accuracy annotation
        pred_vs_actual_mae = np.mean(np.abs(np.array(predicted_means) - np.array(actual_means)))
        ax.text(0.95, 0.95, f"Pred vs Actual\nMAE: {pred_vs_actual_mae:.2f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    fig_path = output_dir / "figures" / "fig_contagion_comparison.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix(".png"))
    plt.close()

    logger.info(f"Saved Contagion Comparison figure to {fig_path}")


def plot_forward_risk_scatter(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Plot Forward Risk Prediction scatter plots: Predicted vs Actual for key metrics.

    Creates 2x2 panel showing correlation between predicted and actual values
    for HHI, Density, SIS, and Spillover metrics across horizons.
    """
    horizons_data = results.get("horizons", {})
    if not horizons_data:
        logger.warning("No forward risk data to plot")
        return

    metrics_to_plot = [
        ("tvl_hhi", "TVL Concentration (HHI)"),
        ("density", "Network Density"),
        ("mean_sis", "Mean Systemic Importance"),
        ("degree_gini", "Degree Inequality (Gini)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Forward-looking Risk Metric Prediction Accuracy", fontsize=14, fontweight="bold")

    horizon_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(horizons_data)))

    for ax_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[ax_idx // 2, ax_idx % 2]

        all_pred = []
        all_actual = []

        for h_idx, (horizon_key, horizon_data) in enumerate(horizons_data.items()):
            pred_metrics = horizon_data.get("predicted_metrics", [])
            actual_metrics = horizon_data.get("actual_metrics", [])

            pred_vals = [m.get(metric_key, np.nan) for m in pred_metrics]
            actual_vals = [m.get(metric_key, np.nan) for m in actual_metrics]

            # Filter valid pairs
            valid_mask = ~(np.isnan(pred_vals) | np.isnan(actual_vals))
            pred_vals = np.array(pred_vals)[valid_mask]
            actual_vals = np.array(actual_vals)[valid_mask]

            if len(pred_vals) > 0:
                ax.scatter(actual_vals, pred_vals, c=[horizon_colors[h_idx]],
                          label=horizon_key, alpha=0.6, s=30, edgecolors="white", linewidth=0.5)
                all_pred.extend(pred_vals)
                all_actual.extend(actual_vals)

        if all_pred and all_actual:
            # Add 45-degree reference line
            all_vals = np.concatenate([all_pred, all_actual])
            min_val, max_val = np.min(all_vals), np.max(all_vals)
            margin = (max_val - min_val) * 0.1
            line_range = [min_val - margin, max_val + margin]
            ax.plot(line_range, line_range, "k--", alpha=0.5, linewidth=1, label="Perfect prediction")

            # Compute R²
            corr = np.corrcoef(all_pred, all_actual)[0, 1]
            r_squared = corr ** 2
            ax.text(0.05, 0.95, f"$R^2$ = {r_squared:.3f}", transform=ax.transAxes,
                   ha="left", va="top", fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlim(line_range)
            ax.set_ylim(line_range)

        ax.set_xlabel(f"Actual {metric_label}")
        ax.set_ylabel(f"Predicted {metric_label}")
        ax.set_title(metric_label)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    fig_path = output_dir / "figures" / "fig_forward_risk_scatter.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix(".png"))
    plt.close()

    logger.info(f"Saved Forward Risk Scatter figure to {fig_path}")


def plot_all_figures(
    forward_risk_results: Optional[Dict] = None,
    contagion_results: Optional[Dict] = None,
    early_warning_results: Optional[Dict] = None,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate all Task II figures."""
    logger.info("Generating Task II visualization figures...")

    if forward_risk_results:
        plot_forward_risk_scatter(forward_risk_results, output_dir)

    if contagion_results:
        plot_contagion_comparison(contagion_results, output_dir)

    if early_warning_results:
        plot_early_warning_timeseries(early_warning_results, output_dir)

    logger.info(f"All figures saved to {output_dir / 'figures'}")


# ============== Model Loading ==============

def load_or_train_model(
    config: ExperimentConfig,
    train_pairs: List[WeekPair],
    force_retrain: bool = False,
) -> GraphPFNLinkPredictor:
    """Load cached DeXposure-FM model or train if not exists."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MODEL_CACHE_DIR / "dexposure_fm.pt"
    device = torch.device(config.device)

    encoder = load_graphpfn_encoder(config.checkpoint_path, device)
    embed_dim = encoder.tfm.embed_dim
    model = GraphPFNLinkPredictor(encoder, embed_dim, config.hidden_dim).to(device)

    if cache_path.exists() and not force_retrain:
        logger.info(f"Loading cached model from {cache_path}")
        state = torch.load(cache_path, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        return model

    logger.info("Training DeXposure-FM (fine-tuned)...")
    for p in model.encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_state = None
    best_loss = float("inf")

    for epoch in range(config.epochs):
        losses, _ = train_graphpfn_epoch(
            model, train_pairs, optimizer, config,
            finetune_encoder=True, prev_embeddings=None
        )
        total_loss = losses["exist_loss"] + losses["weight_loss"]

        if total_loss < best_loss:
            best_loss = total_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch+1}: exist={losses['exist_loss']:.4f}, "
                       f"weight={losses['weight_loss']:.4f}, node={losses['node_loss']:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    torch.save({"model": model.state_dict()}, cache_path)
    logger.info(f"Model saved to {cache_path}")

    return model


# ============== Network Reconstruction from Predictions ==============

def reconstruct_network_from_predictions(
    pred: Dict[str, Any],
    pair: WeekPair,
    edge_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Reconstruct a network snapshot from model predictions.

    Args:
        pred: Model prediction output
        pair: Original WeekPair with node info
        edge_threshold: Probability threshold for edge existence

    Returns:
        Reconstructed snapshot dictionary compatible with risk analysis functions
    """
    # Edge predictions
    exist_prob = 1 / (1 + np.exp(-pred["exist_logits"]))
    edge_mask = exist_prob > edge_threshold

    pred_src = pair.pair_src[edge_mask]
    pred_dst = pair.pair_dst[edge_mask]
    pred_weights = pred["weight_pred"][edge_mask]

    # Node predictions (TVL change)
    sizes_t = pair.sizes_t
    log_size_t = np.log1p(np.maximum(sizes_t, 0))
    pred_log_tvl = log_size_t + pred["node_pred"]
    pred_sizes = np.expm1(np.maximum(pred_log_tvl, 0))

    return {
        "date": pred.get("time_t1", "predicted"),
        "node_ids": pair.node_ids,
        "categories": pair.categories,
        "sizes": pred_sizes,
        "edge_src": pred_src,
        "edge_dst": pred_dst,
        "edge_weight": pred_weights,
        "features": pair.features_t,  # Use original features
    }


def reconstruct_actual_network(
    pred: Dict[str, Any],
    pair: WeekPair,
) -> Dict[str, Any]:
    """Reconstruct actual network at t+h from ground truth."""
    edge_mask = pair.y_exist > 0.5

    sizes_t = pair.sizes_t
    log_size_t = np.log1p(np.maximum(sizes_t, 0))
    actual_log_tvl = log_size_t + pair.y_node
    actual_sizes = np.expm1(np.maximum(actual_log_tvl, 0))

    return {
        "date": pred.get("time_t1", "actual"),
        "node_ids": pair.node_ids,
        "categories": pair.categories,
        "sizes": actual_sizes,
        "edge_src": pair.pair_src[edge_mask],
        "edge_dst": pair.pair_dst[edge_mask],
        "edge_weight": pair.y_weight[edge_mask],
        "features": pair.features_t,
    }


# ============== Risk Metrics Computation ==============

def compute_network_risk_metrics(
    snap: Dict[str, Any],
    meta_category: Dict,
    edge_weight_is_log: bool = True,
) -> Dict[str, float]:
    """
    Compute comprehensive risk metrics on a network snapshot.

    Args:
        snap: Network snapshot in array format (node_ids, sizes, edge_src, edge_dst, edge_weight)
        meta_category: Protocol category mapping
        edge_weight_is_log: If True, edge_weight is log1p-scaled; if False, linear

    Returns metrics that are relevant for financial stability assessment.
    """
    node_ids = snap["node_ids"]
    sizes = np.array(snap["sizes"], dtype=np.float64)
    edge_src = snap["edge_src"]
    edge_dst = snap["edge_dst"]
    edge_weight = snap["edge_weight"]

    n_nodes = len(node_ids)
    n_edges = len(edge_src)

    metrics = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "total_tvl": float(np.sum(sizes)),
    }

    if n_nodes == 0:
        return metrics

    # 1. TVL concentration (HHI)
    total_tvl = np.sum(sizes)
    if total_tvl > 0:
        tvl_shares = sizes / total_tvl
        metrics["tvl_hhi"] = float(np.sum(tvl_shares ** 2))
        sorted_shares = np.sort(tvl_shares)[::-1]
        metrics["top10_tvl_share"] = float(np.sum(sorted_shares[:10]))
        metrics["tvl_gini"] = gini_coefficient(sizes)
    else:
        metrics["tvl_hhi"] = 0.0
        metrics["top10_tvl_share"] = 0.0
        metrics["tvl_gini"] = 0.0

    # 2. Edge weight concentration
    if len(edge_weight) > 0:
        # Convert to linear scale if needed
        if edge_weight_is_log:
            weights_linear = np.expm1(np.array(edge_weight))
        else:
            weights_linear = np.array(edge_weight)
        weights_linear = np.maximum(weights_linear, 0)
        total_weight = np.sum(weights_linear)
        if total_weight > 0:
            weight_shares = weights_linear / total_weight
            metrics["edge_hhi"] = float(np.sum(weight_shares ** 2))
            sorted_weights = np.sort(weight_shares)[::-1]
            metrics["top10_edge_share"] = float(np.sum(sorted_weights[:10]))
        else:
            metrics["edge_hhi"] = 0.0
            metrics["top10_edge_share"] = 0.0
    else:
        metrics["edge_hhi"] = 0.0
        metrics["top10_edge_share"] = 0.0

    # 3. Network density
    max_edges = n_nodes * (n_nodes - 1)
    metrics["density"] = n_edges / max_edges if max_edges > 0 else 0.0

    # 4. Degree distribution
    in_degree = np.zeros(n_nodes)
    out_degree = np.zeros(n_nodes)
    for src, dst in zip(edge_src, edge_dst):
        if src < n_nodes:
            out_degree[src] += 1
        if dst < n_nodes:
            in_degree[dst] += 1

    metrics["mean_in_degree"] = float(np.mean(in_degree))
    metrics["max_in_degree"] = float(np.max(in_degree)) if n_nodes > 0 else 0
    metrics["degree_gini"] = gini_coefficient(in_degree + out_degree)

    # 5. SIS scores (if available)
    # Need to convert array format to dict format for compute_systemic_importance_score
    try:
        dict_snap = array_snap_to_dict_snap(snap, edge_weight_is_log=edge_weight_is_log)
        sis = compute_systemic_importance_score(dict_snap)
        if sis:
            sis_values = list(sis.values())
            metrics["mean_sis"] = float(np.mean(sis_values))
            metrics["max_sis"] = float(np.max(sis_values))
            metrics["sis_gini"] = gini_coefficient(sis_values)
    except Exception as e:
        logger.warning(f"Could not compute SIS: {e}")

    # 6. Sector spillover (if meta_category available)
    # Note: compute_sector_spillover_index returns a dict of sector->sector->weight
    # We compute a scalar spillover index as the total cross-sector exposure
    try:
        dict_snap = array_snap_to_dict_snap(snap, edge_weight_is_log=edge_weight_is_log)
        sector_exposure = compute_sector_spillover_index(dict_snap, meta_category)
        total_spillover = sum(
            sum(targets.values()) for targets in sector_exposure.values()
        )
        metrics["spillover_index"] = float(total_spillover)
    except Exception as e:
        logger.warning(f"Could not compute spillover: {e}")

    return metrics


# ============== Experiment 1: Forward-looking Risk Metric Prediction ==============

def run_forward_risk_prediction(
    config: ExperimentConfig,
    model: GraphPFNLinkPredictor,
    test_snapshots: List[Dict],
    meta_category: Dict,
    horizons: List[int] = [1, 4, 8, 12],
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Experiment 1: Forward-looking Risk Metric Prediction

    Demonstrates that the model can predict future network risk metrics.

    For each horizon h:
    1. Use model to predict network at t+h
    2. Compute risk metrics on predicted network
    3. Compare with actual risk metrics at t+h
    4. Report prediction accuracy for each risk metric
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Forward-looking Risk Metric Prediction")
    logger.info("=" * 70)
    logger.info("Goal: Show model can predict future network risk metrics")

    results = {"horizons": {}, "summary": {}}

    for horizon in horizons:
        logger.info(f"\n--- Horizon h={horizon} weeks ---")

        # Build test pairs for this horizon
        test_pairs = build_week_pairs(test_snapshots, config.neg_ratio, config.seed, horizon=horizon)
        if not test_pairs:
            logger.warning(f"No test pairs for horizon {horizon}")
            continue

        # Get predictions
        preds = predict_graphpfn(model, test_pairs, config)

        horizon_results = {
            "n_samples": len(test_pairs),
            "predicted_metrics": [],
            "actual_metrics": [],
            "metric_errors": {},
        }

        for pair, pred in zip(test_pairs, preds):
            # Reconstruct predicted and actual networks
            pred_net = reconstruct_network_from_predictions(pred, pair)
            actual_net = reconstruct_actual_network(pred, pair)

            # Compute risk metrics
            pred_metrics = compute_network_risk_metrics(pred_net, meta_category)
            actual_metrics = compute_network_risk_metrics(actual_net, meta_category)

            horizon_results["predicted_metrics"].append(pred_metrics)
            horizon_results["actual_metrics"].append(actual_metrics)

        # Compute errors for each metric
        key_metrics = [
            "tvl_hhi", "top10_tvl_share", "tvl_gini",
            "edge_hhi", "density", "degree_gini",
            "mean_sis", "spillover_index", "total_tvl"
        ]

        for key in key_metrics:
            pred_vals = [m.get(key, np.nan) for m in horizon_results["predicted_metrics"]]
            actual_vals = [m.get(key, np.nan) for m in horizon_results["actual_metrics"]]

            # Filter out NaN
            valid_mask = ~(np.isnan(pred_vals) | np.isnan(actual_vals))
            pred_vals = np.array(pred_vals)[valid_mask]
            actual_vals = np.array(actual_vals)[valid_mask]

            if len(pred_vals) > 0:
                mae = float(np.mean(np.abs(pred_vals - actual_vals)))
                rmse = float(np.sqrt(np.mean((pred_vals - actual_vals) ** 2)))
                mean_actual = float(np.mean(actual_vals))
                mean_pred = float(np.mean(pred_vals))
                corr = float(np.corrcoef(pred_vals, actual_vals)[0, 1]) if len(pred_vals) > 1 else np.nan

                horizon_results["metric_errors"][key] = {
                    "mae": mae,
                    "rmse": rmse,
                    "mean_actual": mean_actual,
                    "mean_predicted": mean_pred,
                    "relative_error": mae / (abs(mean_actual) + 1e-10),
                    "correlation": corr,
                }

        results["horizons"][f"h={horizon}"] = horizon_results

        # Print summary for this horizon
        logger.info(f"\n  Risk Metric Prediction Accuracy (h={horizon}):")
        logger.info(f"  {'Metric':<20} {'MAE':<12} {'Corr':<12} {'Rel.Err':<12}")
        logger.info(f"  {'-'*56}")
        for key in ["tvl_hhi", "edge_hhi", "density", "mean_sis"]:
            if key in horizon_results["metric_errors"]:
                err = horizon_results["metric_errors"][key]
                logger.info(f"  {key:<20} {err['mae']:<12.4f} {err['correlation']:<12.3f} "
                           f"{err['relative_error']:<12.2%}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp1_forward_risk.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nResults saved to {output_dir / 'exp1_forward_risk.json'}")

    return results


# ============== Experiment 2: Predictive Contagion Simulation ==============

def run_predictive_contagion(
    config: ExperimentConfig,
    model: GraphPFNLinkPredictor,
    test_snapshots: List[Dict],
    meta_category: Dict,
    horizons: List[int] = [1, 4, 8],
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Experiment 2: Predictive Contagion Simulation

    Demonstrates forward-looking stress testing capability.

    For each test sample:
    1. Run contagion on observed network at t
    2. Run contagion on predicted network (t -> t+h)
    3. Run contagion on actual network at t+h
    4. Compare: Does predicted contagion match actual future contagion?
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: Predictive Contagion Simulation")
    logger.info("=" * 70)
    logger.info("Goal: Show model enables forward-looking stress testing")

    # Define shock scenarios
    scenarios = [
        {"name": "Top Protocol Shock", "top_n": 1, "shock_ratio": 0.5},
        {"name": "Top 5 Protocols Shock", "top_n": 5, "shock_ratio": 0.3},
        {"name": "Bridge Sector Shock", "sector": "Bridge", "shock_ratio": 1.0},
    ]

    results = {"horizons": {}, "scenarios": scenarios}

    for horizon in horizons:
        logger.info(f"\n--- Horizon h={horizon} weeks ---")

        test_pairs = build_week_pairs(test_snapshots, config.neg_ratio, config.seed, horizon=horizon)
        if not test_pairs or len(test_pairs) < 3:
            logger.warning(f"Insufficient test pairs for horizon {horizon}")
            continue

        # Sample a few pairs for detailed analysis
        sample_pairs = test_pairs[:min(10, len(test_pairs))]
        preds = predict_graphpfn(model, sample_pairs, config)

        horizon_results = {"samples": []}

        for pair, pred in zip(sample_pairs, preds):
            sample_result = {
                "time_t": pair.time_t,
                "time_t1": pair.time_t1,
                "scenarios": {},
            }

            # Reconstruct networks
            pred_net = reconstruct_network_from_predictions(pred, pair)
            actual_net = reconstruct_actual_network(pred, pair)

            # Get observed network at t (from pair)
            observed_net = {
                "date": pair.time_t,
                "node_ids": pair.node_ids,
                "categories": pair.categories,
                "sizes": pair.sizes_t,
                "edge_src": pair.edge_src_t,
                "edge_dst": pair.edge_dst_t,
                "edge_weight": pair.edge_weight_t,
            }

            for scenario in scenarios:
                # Determine shocked nodes
                if "top_n" in scenario:
                    # Shock top N protocols by TVL
                    tvl_order = np.argsort(pred_net["sizes"])[::-1]
                    shocked_indices = tvl_order[:scenario["top_n"]]
                    shocked_nodes = [pred_net["node_ids"][i] for i in shocked_indices if i < len(pred_net["node_ids"])]
                elif "sector" in scenario:
                    # Shock all protocols in a sector
                    shocked_nodes = [
                        nid for nid in pred_net["node_ids"]
                        if scenario["sector"].lower() in meta_category.get(nid, "").lower()
                    ]
                else:
                    shocked_nodes = []

                if not shocked_nodes:
                    continue

                shock_ratio = scenario["shock_ratio"]

                # Run contagion on all three networks
                # Convert array format to dict format for simulate_contagion
                try:
                    observed_dict = array_snap_to_dict_snap(observed_net, edge_weight_is_log=True)
                    predicted_dict = array_snap_to_dict_snap(pred_net, edge_weight_is_log=True)
                    actual_dict = array_snap_to_dict_snap(actual_net, edge_weight_is_log=True)

                    contagion_observed = simulate_contagion(observed_dict, shocked_nodes, shock_ratio)
                    contagion_predicted = simulate_contagion(predicted_dict, shocked_nodes, shock_ratio)
                    contagion_actual = simulate_contagion(actual_dict, shocked_nodes, shock_ratio)

                    sample_result["scenarios"][scenario["name"]] = {
                        "shocked_nodes_count": len(shocked_nodes),
                        "observed_t": {
                            "total_loss_pct": contagion_observed["total_loss_pct"],
                            "affected_count": contagion_observed["affected_count"],
                            "depth": contagion_observed["propagation_rounds"],
                        },
                        "predicted_t_h": {
                            "total_loss_pct": contagion_predicted["total_loss_pct"],
                            "affected_count": contagion_predicted["affected_count"],
                            "depth": contagion_predicted["propagation_rounds"],
                        },
                        "actual_t_h": {
                            "total_loss_pct": contagion_actual["total_loss_pct"],
                            "affected_count": contagion_actual["affected_count"],
                            "depth": contagion_actual["propagation_rounds"],
                        },
                    }
                except Exception as e:
                    logger.warning(f"Contagion simulation failed: {e}")

            horizon_results["samples"].append(sample_result)

        # Compute aggregate accuracy
        for scenario in scenarios:
            predicted_losses = []
            actual_losses = []

            for sample in horizon_results["samples"]:
                if scenario["name"] in sample["scenarios"]:
                    s = sample["scenarios"][scenario["name"]]
                    predicted_losses.append(s["predicted_t_h"]["total_loss_pct"])
                    actual_losses.append(s["actual_t_h"]["total_loss_pct"])

            if predicted_losses:
                mae = np.mean(np.abs(np.array(predicted_losses) - np.array(actual_losses)))
                corr = np.corrcoef(predicted_losses, actual_losses)[0, 1] if len(predicted_losses) > 1 else np.nan

                horizon_results[f"{scenario['name']}_accuracy"] = {
                    "mae_loss_pct": float(mae),
                    "correlation": float(corr),
                    "n_samples": len(predicted_losses),
                }

                logger.info(f"  {scenario['name']}: MAE={mae:.2f}%, Corr={corr:.3f}")

        results["horizons"][f"h={horizon}"] = horizon_results

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp2_predictive_contagion.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nResults saved to {output_dir / 'exp2_predictive_contagion.json'}")

    return results


# ============== Experiment 3: Shock Early Warning ==============

def run_early_warning_analysis(
    config: ExperimentConfig,
    model: GraphPFNLinkPredictor,
    all_snapshots: List[Dict],
    date_to_snap: Dict[str, Dict],
    meta_category: Dict,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Experiment 3: Shock Early Warning Analysis

    Demonstrates the model's ability to predict risk before major events.

    For each shock event (Terra/Luna, FTX):
    1. Get network snapshots from pre-event period
    2. Use model to predict risk metrics for event period
    3. Compare predicted vs actual risk trajectories
    4. Assess early warning capability
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: Shock Early Warning Analysis")
    logger.info("=" * 70)
    logger.info("Goal: Show model can predict elevated risk before major shocks")

    all_dates = sorted([s["date"] for s in all_snapshots])
    results = {"events": {}}

    for event_id, event_info in SHOCK_EVENTS.items():
        logger.info(f"\n--- Analyzing {event_info['name']} ---")
        logger.info(f"    Event date: {event_info['event_date']}")

        pre_start, pre_end = event_info["pre_window"]
        event_start, event_end = event_info["event_window"]

        # Get pre-event and event-period snapshots
        event_dates = [d for d in all_dates if event_start <= d <= event_end]
        if not event_dates:
            logger.warning(f"Insufficient data for {event_info['name']}")
            continue

        # Use the last available snapshot strictly before the first event week as anchor.
        first_event_idx = all_dates.index(event_dates[0])
        if first_event_idx <= 0:
            logger.warning(f"Insufficient pre-event data for {event_info['name']}")
            continue

        anchor_date = all_dates[first_event_idx - 1]
        pre_dates = [d for d in all_dates if pre_start <= d <= anchor_date]

        if len(pre_dates) < 2:
            logger.warning(f"Insufficient data for {event_info['name']}")
            continue

        logger.info(f"    Pre-event weeks: {len(pre_dates)}")
        logger.info(f"    Event weeks: {len(event_dates)}")

        pre_snapshots = [date_to_snap[d] for d in pre_dates if d in date_to_snap]

        event_results = {
            "event_info": event_info,
            "pre_dates": pre_dates,
            "event_dates": event_dates,
            "predictions": [],
            "actual_event_metrics": [],
        }

        # Anchor is the last observed week before the event window.
        last_pre_snap = date_to_snap.get(anchor_date, pre_snapshots[-1])

        # For early warning, use 1-step-ahead predictions:
        # predict each event-week network from the immediately preceding observed week.
        for week_idx, event_date in enumerate(event_dates):
            if event_date not in date_to_snap:
                continue

            event_snap = date_to_snap[event_date]

            prev_idx = all_dates.index(event_date) - 1
            if prev_idx < 0:
                continue
            prev_date = all_dates[prev_idx]
            if prev_date not in date_to_snap:
                continue
            prev_snap = date_to_snap[prev_date]

            # Build prediction pair with h=1 (one-step ahead, truly consecutive in time)
            test_pairs = build_week_pairs(
                [prev_snap, event_snap],
                config.neg_ratio,
                config.seed,
                horizon=1  # Always use h=1 for early warning
            )

            if not test_pairs:
                logger.warning(f"No pairs for {event_date}")
                continue

            preds = predict_graphpfn(model, test_pairs[:1], config)
            pred = preds[0]
            pair = test_pairs[0]

            # Reconstruct networks
            pred_net = reconstruct_network_from_predictions(pred, pair)
            actual_net = reconstruct_actual_network(pred, pair)

            # Compute risk metrics
            pred_metrics = compute_network_risk_metrics(pred_net, meta_category)
            actual_metrics = compute_network_risk_metrics(actual_net, meta_category)

            # Run contagion simulation (convert to dict format first)
            try:
                # Shock top 5 protocols
                tvl_order = np.argsort(actual_net["sizes"])[::-1]
                shocked_nodes = [actual_net["node_ids"][i] for i in tvl_order[:5] if i < len(actual_net["node_ids"])]

                pred_dict = array_snap_to_dict_snap(pred_net, edge_weight_is_log=True)
                actual_dict = array_snap_to_dict_snap(actual_net, edge_weight_is_log=True)

                contagion_pred = simulate_contagion(pred_dict, shocked_nodes, 0.5)
                contagion_actual = simulate_contagion(actual_dict, shocked_nodes, 0.5)
            except Exception as e:
                logger.warning(f"Contagion failed: {e}")
                contagion_pred = {"total_loss_pct": np.nan}
                contagion_actual = {"total_loss_pct": np.nan}

            week_result = {
                "weeks_ahead": week_idx + 1,
                "target_date": event_date,
                "predicted_metrics": pred_metrics,
                "actual_metrics": actual_metrics,
                "predicted_contagion_loss": contagion_pred["total_loss_pct"],
                "actual_contagion_loss": contagion_actual["total_loss_pct"],
            }

            event_results["predictions"].append(week_result)

            # Print progress
            logger.info(f"    week {week_idx+1} ({event_date}): "
                       f"Pred HHI={pred_metrics.get('tvl_hhi', 0):.4f}, "
                       f"Actual HHI={actual_metrics.get('tvl_hhi', 0):.4f}, "
                       f"Pred Contagion={contagion_pred['total_loss_pct']:.2f}%")

        # Also compute metrics for observed pre-event period (baseline)
        # Note: last_pre_snap has LINEAR edge weights (from build_snapshot)
        baseline_metrics = compute_network_risk_metrics(
            last_pre_snap, meta_category, edge_weight_is_log=False
        )
        event_results["baseline_metrics"] = baseline_metrics

        # Compute summary statistics
        if event_results["predictions"]:
            pred_hhi = [p["predicted_metrics"].get("tvl_hhi", np.nan) for p in event_results["predictions"]]
            actual_hhi = [p["actual_metrics"].get("tvl_hhi", np.nan) for p in event_results["predictions"]]

            event_results["summary"] = {
                "baseline_hhi": baseline_metrics.get("tvl_hhi", np.nan),
                "mean_predicted_hhi": float(np.nanmean(pred_hhi)),
                "mean_actual_hhi": float(np.nanmean(actual_hhi)),
                "hhi_prediction_mae": float(np.nanmean(np.abs(np.array(pred_hhi) - np.array(actual_hhi)))),
                "hhi_change_predicted": float(np.nanmean(pred_hhi) - baseline_metrics.get("tvl_hhi", 0)),
                "hhi_change_actual": float(np.nanmean(actual_hhi) - baseline_metrics.get("tvl_hhi", 0)),
            }

            logger.info(f"\n    Summary for {event_info['name']}:")
            logger.info(f"    Baseline HHI: {event_results['summary']['baseline_hhi']:.4f}")
            logger.info(f"    Predicted HHI during event: {event_results['summary']['mean_predicted_hhi']:.4f}")
            logger.info(f"    Actual HHI during event: {event_results['summary']['mean_actual_hhi']:.4f}")
            logger.info(f"    HHI Change (Predicted): {event_results['summary']['hhi_change_predicted']:+.4f}")
            logger.info(f"    HHI Change (Actual): {event_results['summary']['hhi_change_actual']:+.4f}")

        results["events"][event_id] = event_results

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "exp3_early_warning.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nResults saved to {output_dir / 'exp3_early_warning.json'}")

    return results


# ============== Main ==============

def main():
    default_device = ExperimentConfig().device
    parser = argparse.ArgumentParser(description="Task II: Model-based Financial Stability Analysis")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "forward_risk", "predictive_contagion", "early_warning"],
        help="Which experiment to run",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs if model needs training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help=f"Device to use (default: {default_device})",
    )
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--output-dir", type=str, default="output/task2_model_based")

    args = parser.parse_args()
    if args.device.startswith("cuda") and default_device == "cpu":
        logger.warning(
            "CUDA requested but DGL CUDA is not available; falling back to CPU."
        )
        args.device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler for logging
    file_handler = logging.FileHandler(output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("Task II: Model-based Forward-looking Financial Stability Analysis")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {output_dir}")

    if not GRAPHPFN_AVAILABLE:
        logger.error("GraphPFN not available. Please install required dependencies.")
        return

    # Setup config
    config = ExperimentConfig(
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    set_seed(config.seed)

    # Load data
    logger.info("\nLoading data...")
    meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
    network_data = load_network_data(config.data_path)
    all_dates = sorted(network_data.keys())

    logger.info(f"Loaded {len(all_dates)} weekly snapshots")

    # Build snapshots
    snapshots = [
        build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
        for date in all_dates
    ]
    date_to_snap = {s["date"]: s for s in snapshots}

    # Get train/test split
    date_splits = get_single_split(all_dates)
    train_snapshots = [date_to_snap[d] for d in date_splits["train"] if d in date_to_snap]
    test_snapshots = [date_to_snap[d] for d in date_splits["test"] if d in date_to_snap]

    logger.info(f"Train snapshots: {len(train_snapshots)}")
    logger.info(f"Test snapshots: {len(test_snapshots)}")

    # Build training pairs and load/train model
    logger.info("\nPreparing model...")
    train_pairs = build_week_pairs(train_snapshots, config.neg_ratio, config.seed, horizon=1)
    model = load_or_train_model(config, train_pairs, force_retrain=args.force_retrain)

    # Run experiments
    results = {}

    if args.experiment in ["all", "forward_risk"]:
        results["forward_risk"] = run_forward_risk_prediction(
            config, model, test_snapshots, meta_category, output_dir=output_dir
        )

    if args.experiment in ["all", "predictive_contagion"]:
        results["predictive_contagion"] = run_predictive_contagion(
            config, model, test_snapshots, meta_category, output_dir=output_dir
        )

    if args.experiment in ["all", "early_warning"]:
        results["early_warning"] = run_early_warning_analysis(
            config, model, snapshots, date_to_snap, meta_category, output_dir=output_dir
        )

    # Save combined results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Generate visualization figures
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING VISUALIZATION FIGURES")
    logger.info("=" * 70)

    plot_all_figures(
        forward_risk_results=results.get("forward_risk"),
        contagion_results=results.get("predictive_contagion"),
        early_warning_results=results.get("early_warning"),
        output_dir=output_dir,
    )

    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Figures saved to: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
