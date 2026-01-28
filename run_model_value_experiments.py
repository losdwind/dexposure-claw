#!/usr/bin/env python3
"""
Model Value Experiments: Demonstrate the trained model's value beyond forecasting.

This script implements three key experiments:
1. Imputation Comparison: Compare DeXposure-FM vs GraphPFN-frozen vs ROLAND
2. Predicted Network Risk Analysis: Compute risk metrics on predicted networks
3. Shock Early Warning: Model's ability to predict structural changes before shocks

These experiments address the gap where systemic risk measurement and imputation
tasks did not fully leverage the trained model's capabilities.

KEY FEATURE: Model caching - trains once, reuses for all experiments.
"""

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

# Import from main experiment script
from run_full_experiment import (
    GRAPHPFN_AVAILABLE,
    ExperimentConfig,
    GraphPFNLinkPredictor,
    LinkScorer,
    NodeHead,
    ROLANDBaseline,
    WeekPair,
    build_snapshot,
    build_week_pairs,
    evaluate_imputation_pairs,
    get_single_split,
    graphpfn_encode,
    load_graphpfn_encoder,
    load_metadata,
    load_network_data,
    log_info,
    predict_graphpfn,
    set_seed,
    train_graphpfn_epoch,
    train_roland_epoch,
)

# ============== Model Caching ==============

MODEL_CACHE_DIR = Path("output/model_cache")


def get_model_cache_path(model_name: str) -> Path:
    """Get cache path for a model."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_CACHE_DIR / f"{model_name}.pt"


def load_or_train_dexposure_fm(
    config: ExperimentConfig,
    train_pairs: List[WeekPair],
    val_pairs: Optional[List[WeekPair]] = None,
    force_retrain: bool = False,
) -> GraphPFNLinkPredictor:
    """Load cached DeXposure-FM model or train if not exists."""
    cache_path = get_model_cache_path("dexposure_fm")
    device = torch.device(config.device)

    encoder = load_graphpfn_encoder(config.checkpoint_path, device)
    embed_dim = encoder.tfm.embed_dim
    model = GraphPFNLinkPredictor(encoder, embed_dim, config.hidden_dim).to(device)

    if cache_path.exists() and not force_retrain:
        log_info(f"Loading cached DeXposure-FM from {cache_path}")
        state = torch.load(cache_path, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        return model

    log_info("Training DeXposure-FM (fine-tuned)...")
    for p in model.encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_model_state = None
    best_metric = float("inf")
    prev_emb = None

    for epoch in range(config.epochs):
        losses, prev_emb = train_graphpfn_epoch(
            model, train_pairs, optimizer, config,
            finetune_encoder=True, prev_embeddings=prev_emb
        )

        # Simple validation: use training loss as proxy if no val_pairs
        current_metric = losses["exist_loss"] + losses["weight_loss"]
        if current_metric < best_metric:
            best_metric = current_metric
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0:
            log_info(f"  Epoch {epoch+1}: exist={losses['exist_loss']:.4f}, weight={losses['weight_loss']:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    # Save to cache
    torch.save({"model": model.state_dict()}, cache_path)
    log_info(f"Saved DeXposure-FM to {cache_path}")

    return model


def load_or_train_graphpfn_frozen(
    config: ExperimentConfig,
    train_pairs: List[WeekPair],
    force_retrain: bool = False,
) -> GraphPFNLinkPredictor:
    """Load cached GraphPFN-frozen model or train if not exists."""
    cache_path = get_model_cache_path("graphpfn_frozen")
    device = torch.device(config.device)

    encoder = load_graphpfn_encoder(config.checkpoint_path, device)
    embed_dim = encoder.tfm.embed_dim
    model = GraphPFNLinkPredictor(encoder, embed_dim, config.hidden_dim).to(device)

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    if cache_path.exists() and not force_retrain:
        log_info(f"Loading cached GraphPFN-frozen from {cache_path}")
        state = torch.load(cache_path, map_location=device)
        # Only load head weights (encoder is frozen)
        model.link_scorer.load_state_dict(state["link_scorer"], strict=True)
        model.node_head.load_state_dict(state["node_head"], strict=True)
        return model

    log_info("Training GraphPFN-frozen (linear probe)...")
    head_params = list(model.link_scorer.parameters()) + list(model.node_head.parameters())
    optimizer = torch.optim.Adam(head_params, lr=config.lr)
    best_model_state = None
    best_metric = float("inf")
    prev_emb = None

    for epoch in range(config.epochs):
        losses, prev_emb = train_graphpfn_epoch(
            model, train_pairs, optimizer, config,
            finetune_encoder=False, prev_embeddings=prev_emb
        )

        current_metric = losses["exist_loss"] + losses["weight_loss"]
        if current_metric < best_metric:
            best_metric = current_metric
            best_model_state = {
                "link_scorer": copy.deepcopy(model.link_scorer.state_dict()),
                "node_head": copy.deepcopy(model.node_head.state_dict()),
            }

        if (epoch + 1) % 5 == 0:
            log_info(f"  Epoch {epoch+1}: exist={losses['exist_loss']:.4f}, weight={losses['weight_loss']:.4f}")

    if best_model_state:
        model.link_scorer.load_state_dict(best_model_state["link_scorer"])
        model.node_head.load_state_dict(best_model_state["node_head"])

    # Save to cache
    torch.save({
        "link_scorer": model.link_scorer.state_dict(),
        "node_head": model.node_head.state_dict(),
    }, cache_path)
    log_info(f"Saved GraphPFN-frozen to {cache_path}")

    return model


def load_or_train_roland(
    config: ExperimentConfig,
    train_pairs: List[WeekPair],
    input_dim: int,
    force_retrain: bool = False,
) -> ROLANDBaseline:
    """Load cached ROLAND model or train if not exists."""
    cache_path = get_model_cache_path("roland")
    device = torch.device(config.device)

    model = ROLANDBaseline(input_dim=input_dim, hidden_dim=64, out_dim=32).to(device)

    if cache_path.exists() and not force_retrain:
        log_info(f"Loading cached ROLAND from {cache_path}")
        state = torch.load(cache_path, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        return model

    log_info("Training ROLAND (from scratch)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_model_state = None
    best_metric = float("inf")

    for epoch in range(config.epochs):
        losses = train_roland_epoch(model, train_pairs, optimizer, config)

        current_metric = losses["exist_loss"] + losses["weight_loss"]
        if current_metric < best_metric:
            best_metric = current_metric
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0:
            log_info(f"  Epoch {epoch+1}: exist={losses['exist_loss']:.4f}, weight={losses['weight_loss']:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    # Save to cache
    torch.save({"model": model.state_dict()}, cache_path)
    log_info(f"Saved ROLAND to {cache_path}")

    return model

# Shock event definitions
SHOCK_EVENTS = {
    "terra_luna": {
        "name": "Terra/Luna Collapse",
        "event_date": "2022-05-09",
        "pre_start": "2022-04-11",  # 4 weeks before
        "pre_end": "2022-05-02",
        "event_start": "2022-05-02",
        "event_end": "2022-05-16",
        "post_end": "2022-05-30",
    },
    "ftx": {
        "name": "FTX Collapse",
        "event_date": "2022-11-07",
        "pre_start": "2022-10-10",  # 4 weeks before
        "pre_end": "2022-10-31",
        "event_start": "2022-10-31",
        "event_end": "2022-11-14",
        "post_end": "2022-11-28",
    },
}


# ============== Experiment 1: Imputation Comparison ==============


def run_imputation_comparison(
    config: ExperimentConfig,
    mask_ratios: List[float] = None,
    output_dir: Optional[Path] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Compare imputation performance across three models:
    - DeXposure-FM (fine-tuned)
    - GraphPFN-frozen (linear probe)
    - ROLAND (trained from scratch)

    Uses cached models if available (saves training time/cost).

    Returns comparison table suitable for paper.
    """
    log_info("\n" + "=" * 70)
    log_info("EXPERIMENT 1: Imputation Model Comparison")
    log_info("=" * 70)

    if not GRAPHPFN_AVAILABLE:
        log_info("GraphPFN not available, skipping...")
        return {}

    if mask_ratios is None:
        mask_ratios = [0.1, 0.2, 0.3]

    set_seed(config.seed)
    device = torch.device(config.device)
    rng = np.random.default_rng(config.seed)

    # Load data
    meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
    network_data = load_network_data(config.data_path)
    all_dates = sorted(network_data.keys())
    snapshots = [
        build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
        for date in all_dates
    ]
    date_splits = get_single_split(all_dates)
    date_to_snap = {s["date"]: s for s in snapshots}

    train_snapshots = [date_to_snap[d] for d in date_splits["train"] if d in date_to_snap]
    val_snapshots = [date_to_snap[d] for d in date_splits["val"] if d in date_to_snap]
    test_snapshots = [date_to_snap[d] for d in date_splits["test"] if d in date_to_snap]

    log_info(f"Train: {len(train_snapshots)}, Val: {len(val_snapshots)}, Test: {len(test_snapshots)}")

    # Build pairs
    train_pairs = build_week_pairs(train_snapshots, config.neg_ratio, config.seed, horizon=1)
    test_pairs = build_week_pairs(test_snapshots, config.neg_ratio, config.seed, horizon=1)

    results = {"mask_ratios": mask_ratios, "models": {}}
    input_dim = train_snapshots[0]["features"].shape[1]

    # ---- Model 1: DeXposure-FM (fine-tuned) ----
    log_info("\n--- DeXposure-FM (fine-tuned) ---")
    model_fm = load_or_train_dexposure_fm(config, train_pairs, force_retrain=force_retrain)
    preds_fm = predict_graphpfn(model_fm, test_pairs, config)
    results["models"]["DeXposure-FM"] = {}
    for ratio in mask_ratios:
        results["models"]["DeXposure-FM"][f"{int(ratio*100)}%"] = evaluate_imputation_pairs(
            test_pairs, preds_fm, ratio, rng
        )
    log_info("DeXposure-FM imputation completed.")

    # ---- Model 2: GraphPFN-frozen (linear probe) ----
    log_info("\n--- GraphPFN-frozen (linear probe) ---")
    model_frozen = load_or_train_graphpfn_frozen(config, train_pairs, force_retrain=force_retrain)
    preds_frozen = predict_graphpfn(model_frozen, test_pairs, config)
    results["models"]["GraphPFN-frozen"] = {}
    for ratio in mask_ratios:
        results["models"]["GraphPFN-frozen"][f"{int(ratio*100)}%"] = evaluate_imputation_pairs(
            test_pairs, preds_frozen, ratio, rng
        )
    log_info("GraphPFN-frozen imputation completed.")

    # ---- Model 3: ROLAND (from scratch) ----
    log_info("\n--- ROLAND (from scratch) ---")
    model_roland = load_or_train_roland(config, train_pairs, input_dim, force_retrain=force_retrain)
    preds_roland = predict_roland(model_roland, test_pairs, config)
    results["models"]["ROLAND"] = {}
    for ratio in mask_ratios:
        results["models"]["ROLAND"][f"{int(ratio*100)}%"] = evaluate_imputation_pairs(
            test_pairs, preds_roland, ratio, rng
        )
    log_info("ROLAND imputation completed.")

    # Print comparison table
    print_imputation_comparison_table(results)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "imputation_comparison.json", "w") as f:
            json.dump(results, f, indent=2, default=float)

    return results


def predict_roland(
    model: ROLANDBaseline,
    pairs: List[WeekPair],
    config: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """Generate predictions using ROLAND model."""
    model.eval()
    device = torch.device(config.device)
    predictions = []

    with torch.no_grad():
        for pair in pairs:
            x = torch.tensor(pair.features_t, dtype=torch.float32, device=device)
            edge_index = torch.tensor(
                np.stack([pair.edge_src_t, pair.edge_dst_t]), dtype=torch.long, device=device
            )
            edge_label_index = torch.tensor(
                np.stack([pair.pair_src, pair.pair_dst]), dtype=torch.long, device=device
            )

            pred_logits, weight_pred, _, _, node_embed = model(x, edge_index, edge_label_index)

            # Node prediction
            node_pred = model.node_head(node_embed).squeeze(-1)

            predictions.append({
                "exist_logits": pred_logits.cpu().numpy(),
                "weight_pred": weight_pred.cpu().numpy(),
                "node_pred": node_pred.cpu().numpy(),
                "y_exist": pair.y_exist,
                "y_weight": pair.y_weight,
                "y_node": pair.y_node,
                "node_mask": pair.node_mask if hasattr(pair, "node_mask") else np.ones(len(pair.node_ids), dtype=bool),
            })

    return predictions


def print_imputation_comparison_table(results: Dict[str, Any]) -> None:
    """Print formatted comparison table."""
    log_info("\n" + "=" * 80)
    log_info("IMPUTATION COMPARISON RESULTS")
    log_info("=" * 80)

    header = f"{'Model':<20} {'Mask %':<10} {'Edge Recall':<15} {'Weight MAE':<15} {'Node MAE':<15}"
    log_info(header)
    log_info("-" * 80)

    for model_name, model_results in results.get("models", {}).items():
        for mask_key, metrics in model_results.items():
            recall = metrics.get("edge_exist_recall_mean", float("nan"))
            weight_mae = metrics.get("edge_weight_mae_mean", float("nan"))
            node_mae = metrics.get("node_size_mae_mean", float("nan"))
            log_info(f"{model_name:<20} {mask_key:<10} {recall:<15.4f} {weight_mae:<15.4f} {node_mae:<15.4f}")


# ============== Experiment 2: Predicted Network Risk Analysis ==============


def compute_network_risk_metrics(
    node_ids: List[str],
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_weights: np.ndarray,
    node_tvls: np.ndarray,
    node_categories: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute systemic risk metrics on a network.

    Returns:
        Dictionary with risk metrics: SIS scores, spillover index, concentration, etc.
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
            G.add_edge(int(src), int(dst), weight=float(np.exp(w) - 1))  # Convert from log-scale

    metrics = {}

    # 1. PageRank
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
        weights_linear = np.exp(edge_weights) - 1
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


def run_predicted_network_risk_analysis(
    config: ExperimentConfig,
    horizons: List[int] = None,
    output_dir: Optional[Path] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Compare risk metrics computed on predicted networks vs actual networks.

    This demonstrates the model's value for forward-looking risk assessment:
    - Use model to predict network at t+h
    - Compute risk metrics on predicted network
    - Compare with actual risk metrics at t+h

    Uses cached DeXposure-FM model if available.
    """
    log_info("\n" + "=" * 70)
    log_info("EXPERIMENT 2: Predicted Network Risk Analysis")
    log_info("=" * 70)

    if not GRAPHPFN_AVAILABLE:
        log_info("GraphPFN not available, skipping...")
        return {}

    if horizons is None:
        horizons = [1, 3, 7]

    set_seed(config.seed)
    device = torch.device(config.device)

    # Load data
    meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
    network_data = load_network_data(config.data_path)
    all_dates = sorted(network_data.keys())
    snapshots = [
        build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
        for date in all_dates
    ]
    date_splits = get_single_split(all_dates)
    date_to_snap = {s["date"]: s for s in snapshots}

    train_snapshots = [date_to_snap[d] for d in date_splits["train"] if d in date_to_snap]
    test_snapshots = [date_to_snap[d] for d in date_splits["test"] if d in date_to_snap]

    # Load cached model or train
    train_pairs = build_week_pairs(train_snapshots, config.neg_ratio, config.seed, horizon=1)
    model = load_or_train_dexposure_fm(config, train_pairs, force_retrain=force_retrain)

    log_info("Model ready. Computing risk metrics on predicted networks...")

    results = {"horizons": {}}

    for horizon in horizons:
        log_info(f"\n--- Horizon h={horizon} weeks ---")

        test_pairs = build_week_pairs(test_snapshots, config.neg_ratio, config.seed, horizon=horizon)
        if not test_pairs:
            log_info(f"No test pairs for horizon {horizon}")
            continue

        preds = predict_graphpfn(model, test_pairs, config)

        horizon_results = {
            "predicted_metrics": [],
            "actual_metrics": [],
            "metric_errors": {},
        }

        for pair, pred in zip(test_pairs, preds):
            # Predicted network: use predicted edge probabilities and weights
            exist_prob = 1 / (1 + np.exp(-pred["exist_logits"]))
            predicted_edges_mask = exist_prob > 0.5

            pred_src = pair.pair_src[predicted_edges_mask]
            pred_dst = pair.pair_dst[predicted_edges_mask]
            pred_weights = pred["weight_pred"][predicted_edges_mask]

            # Predicted node TVLs
            sizes_t = pair.sizes_t
            log_size_t = np.log1p(np.maximum(sizes_t, 0))
            pred_log_tvl = log_size_t + pred["node_pred"]
            pred_tvl = np.expm1(np.maximum(pred_log_tvl, 0))

            pred_metrics = compute_network_risk_metrics(
                pair.node_ids, pred_src, pred_dst, pred_weights, pred_tvl
            )
            horizon_results["predicted_metrics"].append(pred_metrics)

            # Actual network at t+h
            actual_edges_mask = pair.y_exist > 0.5
            actual_src = pair.pair_src[actual_edges_mask]
            actual_dst = pair.pair_dst[actual_edges_mask]
            actual_weights = pair.y_weight[actual_edges_mask]

            actual_log_tvl = log_size_t + pair.y_node
            actual_tvl = np.expm1(np.maximum(actual_log_tvl, 0))

            actual_metrics = compute_network_risk_metrics(
                pair.node_ids, actual_src, actual_dst, actual_weights, actual_tvl
            )
            horizon_results["actual_metrics"].append(actual_metrics)

        # Compute average metric errors
        metric_keys = [
            "pagerank_mean", "pagerank_gini", "density", "num_edges",
            "edge_weight_hhi", "top5_edge_concentration", "tvl_hhi", "tvl_gini"
        ]

        for key in metric_keys:
            pred_vals = [m.get(key, 0) for m in horizon_results["predicted_metrics"]]
            actual_vals = [m.get(key, 0) for m in horizon_results["actual_metrics"]]

            if pred_vals and actual_vals:
                mae = float(np.mean(np.abs(np.array(pred_vals) - np.array(actual_vals))))
                rmse = float(np.sqrt(np.mean((np.array(pred_vals) - np.array(actual_vals))**2)))
                mean_actual = float(np.mean(actual_vals))
                mean_pred = float(np.mean(pred_vals))

                horizon_results["metric_errors"][key] = {
                    "mae": mae,
                    "rmse": rmse,
                    "mean_actual": mean_actual,
                    "mean_predicted": mean_pred,
                    "relative_error": mae / max(mean_actual, 1e-6),
                }

        results["horizons"][f"h={horizon}"] = horizon_results

        # Print summary
        log_info(f"  Risk Metric Prediction Accuracy (h={horizon}):")
        for key, errors in horizon_results["metric_errors"].items():
            log_info(
                f"    {key}: MAE={errors['mae']:.4f}, "
                f"Actual={errors['mean_actual']:.4f}, Pred={errors['mean_predicted']:.4f}"
            )

    # Print final comparison table
    print_risk_analysis_table(results)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Convert to JSON-serializable format
        serializable = serialize_results(results)
        with open(output_dir / "predicted_network_risk.json", "w") as f:
            json.dump(serializable, f, indent=2)

    return results


def serialize_results(obj: Any) -> Any:
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: serialize_results(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_results(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def print_risk_analysis_table(results: Dict[str, Any]) -> None:
    """Print formatted risk analysis comparison."""
    log_info("\n" + "=" * 90)
    log_info("PREDICTED NETWORK RISK METRICS vs ACTUAL")
    log_info("=" * 90)

    header = f"{'Metric':<25} {'Horizon':<10} {'Actual':<15} {'Predicted':<15} {'MAE':<15} {'Rel.Err':<10}"
    log_info(header)
    log_info("-" * 90)

    key_metrics = ["pagerank_gini", "edge_weight_hhi", "tvl_hhi", "density"]

    for horizon_key, horizon_data in results.get("horizons", {}).items():
        for metric in key_metrics:
            if metric in horizon_data.get("metric_errors", {}):
                errors = horizon_data["metric_errors"][metric]
                log_info(
                    f"{metric:<25} {horizon_key:<10} {errors['mean_actual']:<15.4f} "
                    f"{errors['mean_predicted']:<15.4f} {errors['mae']:<15.4f} "
                    f"{errors['relative_error']:<10.2%}"
                )


# ============== Experiment 3: Shock Early Warning ==============


def run_shock_early_warning_analysis(
    config: ExperimentConfig,
    output_dir: Optional[Path] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Model's ability to predict network changes before shock events.

    For each shock (Terra/Luna, FTX):
    1. Use pre-shock data to train/update model
    2. Predict network structure during shock period
    3. Compare predicted vs actual changes
    4. Assess early warning capability

    Note: For shock analysis, we use the main cached model trained on holdout split.
    This simulates realistic usage where a pre-trained model is applied to new events.
    """
    log_info("\n" + "=" * 70)
    log_info("EXPERIMENT 3: Shock Early Warning Analysis")
    log_info("=" * 70)

    if not GRAPHPFN_AVAILABLE:
        log_info("GraphPFN not available, skipping...")
        return {}

    set_seed(config.seed)
    device = torch.device(config.device)

    # Load data
    meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
    network_data = load_network_data(config.data_path)
    all_dates = sorted(network_data.keys())
    snapshots = [
        build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
        for date in all_dates
    ]
    date_to_snap = {s["date"]: s for s in snapshots}
    date_splits = get_single_split(all_dates)

    # Load cached model (trained on holdout training data)
    train_snapshots_main = [date_to_snap[d] for d in date_splits["train"] if d in date_to_snap]
    train_pairs_main = build_week_pairs(train_snapshots_main, config.neg_ratio, config.seed, horizon=1)
    model = load_or_train_dexposure_fm(config, train_pairs_main, force_retrain=force_retrain)

    results = {"shocks": {}}

    for shock_id, shock_info in SHOCK_EVENTS.items():
        log_info(f"\n--- Analyzing {shock_info['name']} ---")

        # Get relevant snapshots
        pre_dates = [d for d in all_dates if shock_info["pre_start"] <= d < shock_info["event_start"]]
        event_dates = [d for d in all_dates if shock_info["event_start"] <= d < shock_info["event_end"]]
        post_dates = [d for d in all_dates if shock_info["event_end"] <= d <= shock_info["post_end"]]

        if not pre_dates or not event_dates:
            log_info(f"Insufficient data for {shock_info['name']}")
            continue

        log_info(f"  Pre-shock weeks: {len(pre_dates)}, Event weeks: {len(event_dates)}, Post weeks: {len(post_dates)}")

        pre_snapshots = [date_to_snap[d] for d in pre_dates]
        event_snapshots = [date_to_snap[d] for d in event_dates]

        # Predict from last pre-shock week to event weeks
        shock_results = {
            "name": shock_info["name"],
            "event_date": shock_info["event_date"],
            "predictions": [],
            "actual_changes": [],
            "early_warning_metrics": {},
        }

        # Use last pre-shock snapshot to predict event period
        last_pre_snap = pre_snapshots[-1]

        for h, event_snap in enumerate(event_snapshots, start=1):
            # Build prediction pair
            test_pairs = build_week_pairs([last_pre_snap, event_snap], config.neg_ratio, config.seed, horizon=h)
            if not test_pairs:
                continue

            preds = predict_graphpfn(model, test_pairs[:1], config)
            pred = preds[0]

            # Compute changes
            exist_prob = 1 / (1 + np.exp(-pred["exist_logits"]))
            y_exist = pred["y_exist"]

            # Predicted vs actual edge changes
            pred_new_edges = np.sum((exist_prob > 0.5) & (y_exist < 0.5))  # FP: predicted but do not exist
            pred_lost_edges = np.sum((exist_prob < 0.5) & (y_exist > 0.5))  # FN: exist but not predicted
            actual_edges = np.sum(y_exist > 0.5)

            # TVL change prediction
            pred_tvl_change = pred["node_pred"]
            actual_tvl_change = pred["y_node"]
            tvl_mae = float(np.mean(np.abs(pred_tvl_change - actual_tvl_change)))

            # Large TVL drops (>20%)
            large_drop_mask = actual_tvl_change < -0.2
            if np.any(large_drop_mask):
                pred_large_drops = pred_tvl_change[large_drop_mask]
                large_drop_recall = float(np.mean(pred_large_drops < -0.1))  # Did we predict >10% drop?
            else:
                large_drop_recall = float("nan")

            week_result = {
                "weeks_ahead": h,
                "pred_lost_edges": int(pred_lost_edges),
                "actual_edges": int(actual_edges),
                "tvl_mae": tvl_mae,
                "large_drop_recall": large_drop_recall,
                "edge_exist_auprc": compute_auprc(y_exist, exist_prob),
            }
            shock_results["predictions"].append(week_result)

            recall_str = f"{large_drop_recall:.2%}" if not np.isnan(large_drop_recall) else "N/A"
            log_info(
                f"    h={h}: Lost edges pred={pred_lost_edges}, TVL MAE={tvl_mae:.4f}, "
                f"Large drop recall={recall_str}"
            )

        # Compute early warning metrics
        if shock_results["predictions"]:
            shock_results["early_warning_metrics"] = {
                "avg_tvl_mae": float(np.mean([p["tvl_mae"] for p in shock_results["predictions"]])),
                "avg_auprc": float(np.mean([p["edge_exist_auprc"] for p in shock_results["predictions"]])),
            }

        results["shocks"][shock_id] = shock_results

    # Print summary
    print_early_warning_table(results)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        serializable = serialize_results(results)
        with open(output_dir / "shock_early_warning.json", "w") as f:
            json.dump(serializable, f, indent=2)

    return results


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve."""
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def print_early_warning_table(results: Dict[str, Any]) -> None:
    """Print early warning analysis summary."""
    log_info("\n" + "=" * 80)
    log_info("SHOCK EARLY WARNING ANALYSIS SUMMARY")
    log_info("=" * 80)

    for shock_id, shock_data in results.get("shocks", {}).items():
        log_info(f"\n{shock_data.get('name', shock_id)}:")
        log_info(f"  Event date: {shock_data.get('event_date', 'N/A')}")

        metrics = shock_data.get("early_warning_metrics", {})
        log_info(f"  Average TVL MAE: {metrics.get('avg_tvl_mae', float('nan')):.4f}")
        log_info(f"  Average Edge AUPRC: {metrics.get('avg_auprc', float('nan')):.4f}")

        log_info("\n  Week-by-week predictions:")
        for pred in shock_data.get("predictions", []):
            log_info(
                f"    h={pred['weeks_ahead']}: AUPRC={pred['edge_exist_auprc']:.4f}, "
                f"TVL MAE={pred['tvl_mae']:.4f}"
            )


# ============== Main ==============


def main():
    parser = argparse.ArgumentParser(description="Model Value Experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "imputation", "risk", "shock"],
        help="Which experiment to run",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs (only used if no cache)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="output/model_value", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if cached models exist",
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_info(f"Running Model Value Experiments")
    log_info(f"Output directory: {output_dir}")
    log_info(f"Model cache: {MODEL_CACHE_DIR}")
    log_info(f"Device: {args.device}")
    log_info(f"Experiment: {args.experiment}")
    log_info(f"Force retrain: {args.force_retrain}")

    results = {}

    if args.experiment in ["all", "imputation"]:
        results["imputation"] = run_imputation_comparison(
            config, output_dir=output_dir, force_retrain=args.force_retrain
        )

    if args.experiment in ["all", "risk"]:
        results["risk"] = run_predicted_network_risk_analysis(
            config, output_dir=output_dir, force_retrain=args.force_retrain
        )

    if args.experiment in ["all", "shock"]:
        results["shock"] = run_shock_early_warning_analysis(
            config, output_dir=output_dir, force_retrain=args.force_retrain
        )

    log_info("\n" + "=" * 70)
    log_info("ALL EXPERIMENTS COMPLETE")
    log_info("=" * 70)

    return results


if __name__ == "__main__":
    main()
