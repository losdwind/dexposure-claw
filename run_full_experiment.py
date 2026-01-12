#!/usr/bin/env python3
"""
DeXposure-FM Full Experiment Suite

This script implements the complete evaluation framework as specified in the paper:

Task I: Multi-step Forecasting
- Horizon h ∈ {1, 3, 7} weeks ahead prediction
- Metrics: AUPRC, AUROC for link existence; MAE, RMSE for edge weight

Task II: Policy-relevant Scenario Analysis (Shock Analysis)
- Terra/Luna collapse (2022-05-09): UST depeg and death spiral
- FTX collapse (2022-11-07): Exchange failure and contagion
- Network structure changes: TVL, edges, Gini, HHI
- Model performance degradation during crisis

Task III: Imputation
- Random edge masking (10%, 20%, 30%)
- Node size masking
- Evaluate reconstruction accuracy (MAE, recall, correlation)

Rolling Walk-forward Evaluation (Expanding Window)
- Expanding train window with fixed val/test windows
- No look-ahead bias in evaluation

Model Comparison
- GraphPFN (Frozen encoder): Linear probe on pretrained embeddings
- GraphPFN (Finetuned): End-to-end fine-tuning
- ROLAND baseline: Temporal GNN (GCN + GRU)

Network Statistics
- Gini coefficient (degree centralization)
- HHI (Herfindahl-Hirschman Index)
- Network density, entropy, assortativity
- TVL concentration metrics

Usage:
    python run_full_experiment.py --mode all
    python run_full_experiment.py --mode frozen
    python run_full_experiment.py --mode finetuned
    python run_full_experiment.py --mode roland
    python run_full_experiment.py --mode stats
    python run_full_experiment.py --mode shock
    python run_full_experiment.py --mode shock --shock-model roland
    python run_full_experiment.py --mode impute
"""

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error

# Add project paths
GRAPHPFN_ROOT = Path(__file__).parent
sys.path.insert(0, str(GRAPHPFN_ROOT))
sys.path.insert(0, str(GRAPHPFN_ROOT / "src"))

# Import GraphPFN components
try:
    from lib.graphpfn.model import GraphPFN, GraphPFNLayerWrapper
    from lib.limix.model.layer import MultiheadAttention as MHA
    GRAPHPFN_AVAILABLE = True
except ImportError:
    GRAPHPFN_AVAILABLE = False
    print("Warning: GraphPFN not available, will skip GraphPFN experiments")

# Import network statistics
from src.network_statistics import (
    compute_all_network_statistics,
    compute_rolling_statistics,
    gini_coefficient,
    herfindahl_hirschman_index
)

# ============== Configuration ==============

@dataclass
class ExperimentConfig:
    # Data paths
    data_path: str = "data/historical-network_week_2020-03-30.json"
    meta_path: str = "data/meta_df.csv"
    output_dir: str = "output/full_experiment"
    checkpoint_path: str = "checkpoints/graphpfn-v1.ckpt"

    # Experiment settings
    seed: int = 42
    neg_ratio: int = 5
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20

    # Model settings
    hidden_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    edge_batch_size: int = 20000

    # Multi-step forecasting
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 3, 7])

    # Loss weights
    exist_loss_weight: float = 1.0
    weight_loss_weight: float = 1.0
    node_loss_weight: float = 0.5

    # Random seeds for multiple runs
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 2024])

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


EPS = 1e-12


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============== Data Loading ==============

def load_network_data(path: str) -> Dict:
    """Load network JSON data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Loading {path} ({size_mb:.1f} MB)")

    with path.open("rb") as f:
        try:
            import ijson
            data = {k: v for k, v in ijson.kvitems(f, "data")}
            return data
        except Exception:
            f.seek(0)
            payload = json.load(f)
            return payload["data"]


def load_metadata(path: str) -> Tuple[Dict, List[str], Dict]:
    """Load protocol metadata."""
    meta_df = pd.read_csv(path)
    meta_df["id"] = meta_df["id"].astype(str)

    category_list = sorted(meta_df["category"].dropna().unique().tolist())
    if "Unknown" not in category_list:
        category_list.append("Unknown")

    category_to_idx = {c: i for i, c in enumerate(category_list)}
    meta_category = meta_df.set_index("id")["category"].to_dict()

    return meta_category, category_list, category_to_idx


def node_features(node: Dict, meta_category: Dict, category_to_idx: Dict, category_list: List) -> Tuple[np.ndarray, float, str]:
    """Extract node features."""
    node_id = str(node.get("id"))
    size = float(node.get("size", 0.0))
    comp = node.get("composition", {}) or {}

    log_size = math.log1p(max(size, 0.0))
    num_tokens = float(len(comp))

    if size > 0 and comp:
        values = np.array(list(comp.values()), dtype=np.float64)
        values = np.maximum(values, 0.0)
        total = values.sum() + EPS
        shares = values / total
        max_share = float(shares.max())
        entropy = float(-(shares * np.log(shares + EPS)).sum())
    else:
        max_share = 0.0
        entropy = 0.0

    category = meta_category.get(node_id, "Unknown")
    idx = category_to_idx.get(category, category_to_idx["Unknown"])
    cat_vec = np.zeros(len(category_list), dtype=np.float32)
    cat_vec[idx] = 1.0

    feats = np.array([log_size, num_tokens, max_share, entropy], dtype=np.float32)
    feats = np.concatenate([feats, cat_vec], axis=0)

    return feats, size, category


def build_snapshot(date: str, snapshot: Dict, meta_category: Dict, category_to_idx: Dict, category_list: List) -> Dict:
    """Build a single snapshot."""
    nodes = snapshot.get("nodes", [])
    links = snapshot.get("links", [])

    node_ids, features, sizes, categories = [], [], [], []

    for node in nodes:
        node_id = node.get("id")
        if node_id is None:
            continue
        feats, size, category = node_features(node, meta_category, category_to_idx, category_list)
        node_ids.append(str(node_id))
        features.append(feats)
        sizes.append(size)
        categories.append(category)

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    src, dst, weights = [], [], []
    for link in links:
        source = link.get("source")
        target = link.get("target")
        if source is None or target is None:
            continue
        source, target = str(source), str(target)
        if source not in id_to_idx or target not in id_to_idx:
            continue
        src.append(id_to_idx[source])
        dst.append(id_to_idx[target])
        weights.append(float(link.get("size", 0.0)))

    return {
        "date": date,
        "node_ids": node_ids,
        "features": np.array(features, dtype=np.float32) if features else np.zeros((0, 4 + len(category_list)), dtype=np.float32),
        "sizes": np.array(sizes, dtype=np.float32),
        "categories": categories,
        "edge_src": np.array(src, dtype=np.int64),
        "edge_dst": np.array(dst, dtype=np.int64),
        "edge_weight": np.array(weights, dtype=np.float32),
    }


# ============== Strict Temporal Split ==============

def expanding_window_split(
    all_dates: List[str],
    holdout_start: str = "2025-01-01",
    min_train_weeks: int = 104,  # 最少2年训练数据
    val_weeks: int = 12,         # 验证集: 12周 (约3个月)
    test_weeks: int = 8,         # 每fold测试: 8周 (约2个月)
    step_weeks: int = 8,         # 每次向前滚动8周
) -> Dict[str, Any]:
    """
    Expanding Window Walk-Forward Validation (金融惯例).

    特点:
    1. 训练窗口随时间扩展 (Expanding Window)
    2. 验证集固定大小，紧跟训练集
    3. 测试集固定大小，紧跟验证集
    4. Hold-out 2025 数据用于最终评估

    Returns:
        {
            "folds": [
                {"train": [...], "val": [...], "test": [...]},
                ...
            ],
            "holdout": {
                "train": [...],  # 所有 < holdout_start 的数据
                "val": [...],    # 最后 val_weeks
                "test": [...]    # >= holdout_start 的数据 (NEVER seen)
            }
        }

    Timeline:
    ────────────────────────────────────────────────────────────────
    Fold 1: [==== Train (104w) ====][Val 12w][Test 8w]
    Fold 2: [====== Train (112w) ======][Val 12w][Test 8w]
    Fold 3: [======== Train (120w) ========][Val 12w][Test 8w]
    ...
    ────────────────────────────────────────────────────────────────
    Holdout: [============ All pre-2025 Train ============][Val][ 2025 Test ]
    """
    sorted_dates = sorted(all_dates)

    # 分离 holdout 数据
    pre_holdout = [d for d in sorted_dates if d < holdout_start]
    holdout_dates = [d for d in sorted_dates if d >= holdout_start]

    # 生成 rolling folds
    folds = []
    fold_idx = 0

    while True:
        # 训练集终点 (expanding)
        train_end_idx = min_train_weeks + fold_idx * step_weeks

        # 验证集范围
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_weeks

        # 测试集范围
        test_start_idx = val_end_idx
        test_end_idx = test_start_idx + test_weeks

        # 检查是否超出 pre_holdout 范围
        if test_end_idx > len(pre_holdout):
            break

        fold = {
            "fold_id": fold_idx + 1,
            "train": pre_holdout[:train_end_idx],
            "val": pre_holdout[val_start_idx:val_end_idx],
            "test": pre_holdout[test_start_idx:test_end_idx],
        }
        folds.append(fold)
        fold_idx += 1

    # Holdout split: 用所有 pre-2025 训练，2025 测试
    if len(pre_holdout) > val_weeks:
        holdout_train = pre_holdout[:-val_weeks]
        holdout_val = pre_holdout[-val_weeks:]
    else:
        holdout_train = pre_holdout
        holdout_val = []

    return {
        "folds": folds,
        "n_folds": len(folds),
        "holdout": {
            "train": holdout_train,
            "val": holdout_val,
            "test": holdout_dates,
        },
        "config": {
            "min_train_weeks": min_train_weeks,
            "val_weeks": val_weeks,
            "test_weeks": test_weeks,
            "step_weeks": step_weeks,
            "holdout_start": holdout_start,
        }
    }


def get_single_split(
    all_dates: List[str],
    holdout_start: str = "2025-01-01",
    val_weeks: int = 12,
) -> Dict[str, List[str]]:
    """
    简化版: 仅返回最终 holdout 划分 (用于快速实验).

    Train: 所有 < holdout_start - val_weeks
    Val:   holdout_start 前 val_weeks
    Test:  所有 >= holdout_start (2025 hold-out)
    """
    sorted_dates = sorted(all_dates)
    pre_holdout = [d for d in sorted_dates if d < holdout_start]
    holdout_dates = [d for d in sorted_dates if d >= holdout_start]

    if len(pre_holdout) > val_weeks:
        train_dates = pre_holdout[:-val_weeks]
        val_dates = pre_holdout[-val_weeks:]
    else:
        train_dates = pre_holdout
        val_dates = []

    return {
        "train": train_dates,
        "val": val_dates,
        "test": holdout_dates,
    }


# ============== Data Quality Statistics ==============

def compute_data_quality(snapshots: List[Dict], raw_network_data: Dict) -> Dict[str, Any]:
    """Compute data quality statistics for each snapshot."""
    weekly_stats = []

    for i, snap in enumerate(snapshots):
        date = snap["date"]
        raw_snap = raw_network_data.get(date, {})
        raw_nodes = raw_snap.get("nodes", [])
        raw_links = raw_snap.get("links", [])

        # Count dropped edges
        n_target_null = sum(1 for link in raw_links if link.get("target") is None)
        n_endpoint_missing = len(raw_links) - n_target_null - len(snap["edge_src"])

        # Compute overlap with next snapshot
        overlap_ratio = 0.0
        if i < len(snapshots) - 1:
            next_ids = set(snapshots[i + 1]["node_ids"])
            current_ids = set(snap["node_ids"])
            if len(current_ids) > 0:
                overlap_ratio = len(current_ids & next_ids) / len(current_ids)

        # Count unknown categories
        n_unknown = sum(1 for cat in snap["categories"] if cat == "Unknown")

        weekly_stats.append({
            "date": date,
            "N_nodes": len(snap["node_ids"]),
            "N_edges": len(snap["edge_src"]),
            "pct_target_null_dropped": n_target_null / max(len(raw_links), 1),
            "pct_endpoint_missing_dropped": n_endpoint_missing / max(len(raw_links), 1),
            "overlap_ratio_next_week": overlap_ratio,
            "pct_category_unknown": n_unknown / max(len(snap["node_ids"]), 1),
        })

    # Summary statistics
    summary = {
        "mean_nodes_per_week": float(np.mean([s["N_nodes"] for s in weekly_stats])),
        "mean_edges_per_week": float(np.mean([s["N_edges"] for s in weekly_stats])),
        "std_nodes_per_week": float(np.std([s["N_nodes"] for s in weekly_stats])),
        "std_edges_per_week": float(np.std([s["N_edges"] for s in weekly_stats])),
        "mean_pct_target_null_dropped": float(np.mean([s["pct_target_null_dropped"] for s in weekly_stats])),
        "mean_pct_endpoint_missing_dropped": float(np.mean([s["pct_endpoint_missing_dropped"] for s in weekly_stats])),
        "mean_overlap_ratio": float(np.mean([s["overlap_ratio_next_week"] for s in weekly_stats[:-1]])) if len(weekly_stats) > 1 else 0.0,
        "mean_pct_category_unknown": float(np.mean([s["pct_category_unknown"] for s in weekly_stats])),
    }

    summary["pct_target_null_dropped"] = summary["mean_pct_target_null_dropped"]
    summary["pct_endpoint_missing_dropped"] = summary["mean_pct_endpoint_missing_dropped"]

    return {"summary": summary, "weekly": weekly_stats}


# ============== Predictions Output ==============

def save_predictions_csv(
    preds: List[Dict],
    output_dir: Path,
    model_name: str,
    horizon: int,
    fold_id: Optional[int] = None,
    append: bool = False
) -> Tuple[Path, Path]:
    """Save predictions to CSV files."""
    edges_rows = []
    nodes_rows = []

    for item in preds:
        time_t = item["time_t"]
        time_t1 = item["time_t1"]
        node_ids = item["node_ids"]
        categories = item["categories"]
        sizes_t = item["sizes_t"]
        exist_prob = 1 / (1 + np.exp(-item["exist_logits"]))

        # Edge predictions
        for i in range(len(item["y_exist"])):
            u_idx = int(item["pair_src"][i])
            v_idx = int(item["pair_dst"][i])
            edges_rows.append({
                "time_t": time_t,
                "time_t1": time_t1,
                "horizon": horizon,
                "fold_id": fold_id,
                "u_id": node_ids[u_idx],
                "v_id": node_ids[v_idx],
                "y_exist_true": int(item["y_exist"][i]),
                "y_exist_pred": float(exist_prob[i]),
                "y_w_true": float(item["y_weight"][i]),
                "y_w_pred": float(item["weight_pred"][i]),
                "is_positive": bool(item["weight_mask"][i] > 0.5),
            })

        # Node predictions
        node_mask = item["node_mask"]
        for i in range(len(node_mask)):
            if node_mask[i]:
                nodes_rows.append({
                    "time_t": time_t,
                    "time_t1": time_t1,
                    "horizon": horizon,
                    "fold_id": fold_id,
                    "node_id": node_ids[i],
                    "y_node_true": float(item["y_node"][i]),
                    "y_node_pred": float(item["node_pred"][i]),
                    "size_t": float(sizes_t[i]),
                    "category": categories[i],
                })

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_path = output_dir / "predictions_edges_test.csv"
    nodes_path = output_dir / "predictions_nodes_test.csv"

    if edges_rows:
        edges_df = pd.DataFrame(edges_rows)
        edges_df.to_csv(edges_path, index=False, mode="a" if append else "w", header=not (append and edges_path.exists()))
    if nodes_rows:
        nodes_df = pd.DataFrame(nodes_rows)
        nodes_df.to_csv(nodes_path, index=False, mode="a" if append else "w", header=not (append and nodes_path.exists()))

    return edges_path, nodes_path


def save_metrics_json(result: Dict[str, Any], output_dir: Path) -> Path:
    """Save metrics.json in the experiment plan format."""
    payload = {"model": result.get("model", "unknown")}
    results = result.get("results", {})
    for key in sorted(results.keys()):
        payload[key] = results[key]

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(payload, f, indent=2)

    return metrics_path


# ============== Week Pair Construction ==============

@dataclass
class WeekPair:
    time_t: str
    time_t1: str
    node_ids: List[str]
    categories: List[str]
    sizes_t: np.ndarray
    features_t: np.ndarray
    edge_src_t: np.ndarray
    edge_dst_t: np.ndarray
    pair_src: np.ndarray
    pair_dst: np.ndarray
    y_exist: np.ndarray
    y_weight: np.ndarray
    weight_mask: np.ndarray
    y_node: np.ndarray
    node_mask: np.ndarray
    pos_edge_count: int
    neg_edge_count: int


def sample_negatives(num_nodes: int, pos_set: set, num_neg: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample negative edges."""
    neg_src, neg_dst = [], []
    seen = set(pos_set)
    attempts = 0
    max_attempts = num_neg * 10

    while len(neg_src) < num_neg and attempts < max_attempts:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u != v and (u, v) not in seen:
            seen.add((u, v))
            neg_src.append(u)
            neg_dst.append(v)
        attempts += 1

    return np.array(neg_src, dtype=np.int64), np.array(neg_dst, dtype=np.int64)


def build_week_pairs(snapshots: List[Dict], neg_ratio: int, seed: int, horizon: int = 1) -> List[WeekPair]:
    """
    Build week pairs for training/evaluation.

    Args:
        snapshots: List of snapshot dictionaries
        neg_ratio: Negative sampling ratio
        seed: Random seed
        horizon: Prediction horizon (default 1 = next week)
    """
    rng = np.random.default_rng(seed)
    pairs = []

    for t in range(len(snapshots) - horizon):
        snap_t = snapshots[t]
        snap_t1 = snapshots[t + horizon]

        id_to_idx = {nid: i for i, nid in enumerate(snap_t["node_ids"])}
        size_t1_map = {nid: size for nid, size in zip(snap_t1["node_ids"], snap_t1["sizes"])}

        # Build positive edges (edges that exist at t+h)
        pos_src, pos_dst, pos_w, pos_set = [], [], [], set()

        for src_idx, dst_idx, w in zip(snap_t1["edge_src"], snap_t1["edge_dst"], snap_t1["edge_weight"]):
            src_id = snap_t1["node_ids"][int(src_idx)]
            dst_id = snap_t1["node_ids"][int(dst_idx)]
            if src_id not in id_to_idx or dst_id not in id_to_idx:
                continue
            u, v = id_to_idx[src_id], id_to_idx[dst_id]
            pos_src.append(u)
            pos_dst.append(v)
            pos_w.append(math.log1p(max(w, 0.0)))
            pos_set.add((u, v))

        pos_src = np.array(pos_src, dtype=np.int64)
        pos_dst = np.array(pos_dst, dtype=np.int64)
        pos_w = np.array(pos_w, dtype=np.float32)

        num_pos = len(pos_src)
        if num_pos == 0:
            continue

        # Sample negatives
        num_neg = num_pos * neg_ratio
        neg_src, neg_dst = sample_negatives(len(snap_t["node_ids"]), pos_set, num_neg, rng)

        # Combine and shuffle
        pair_src = np.concatenate([pos_src, neg_src])
        pair_dst = np.concatenate([pos_dst, neg_dst])
        y_exist = np.concatenate([np.ones(num_pos, dtype=np.float32), np.zeros(len(neg_src), dtype=np.float32)])
        y_weight = np.concatenate([pos_w, np.zeros(len(neg_src), dtype=np.float32)])
        weight_mask = np.concatenate([np.ones(num_pos, dtype=np.float32), np.zeros(len(neg_src), dtype=np.float32)])

        order = rng.permutation(len(pair_src))
        pair_src, pair_dst = pair_src[order], pair_dst[order]
        y_exist, y_weight, weight_mask = y_exist[order], y_weight[order], weight_mask[order]

        # Node-level labels
        y_node = np.zeros(len(snap_t["node_ids"]), dtype=np.float32)
        node_mask = np.zeros(len(snap_t["node_ids"]), dtype=bool)
        for i, nid in enumerate(snap_t["node_ids"]):
            if nid in size_t1_map:
                y_node[i] = math.log1p(max(size_t1_map[nid], 0.0)) - math.log1p(max(snap_t["sizes"][i], 0.0))
                node_mask[i] = True

        pairs.append(WeekPair(
            time_t=snap_t["date"],
            time_t1=snap_t1["date"],
            node_ids=snap_t["node_ids"],
            categories=snap_t["categories"],
            sizes_t=snap_t["sizes"],
            features_t=snap_t["features"],
            edge_src_t=snap_t["edge_src"],
            edge_dst_t=snap_t["edge_dst"],
            pair_src=pair_src,
            pair_dst=pair_dst,
            y_exist=y_exist,
            y_weight=y_weight,
            weight_mask=weight_mask,
            y_node=y_node,
            node_mask=node_mask,
            pos_edge_count=num_pos,
            neg_edge_count=len(neg_src),
        ))

    return pairs


# ============== Model Definitions ==============

class LinkScorer(nn.Module):
    """Link prediction head with exist and weight outputs."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = 4 * embed_dim
        self.exist_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
        h_u, h_v = h[src], h[dst]
        z = torch.cat([h_u, h_v, h_u * h_v, (h_u - h_v).abs()], dim=-1)
        return self.exist_head(z).squeeze(-1), self.weight_head(z).squeeze(-1)


class NodeHead(nn.Module):
    """Node-level prediction head."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


# ============== GraphPFN Encoder ==============

if GRAPHPFN_AVAILABLE:
    def graphpfn_encode(model: GraphPFN, graph: dgl.DGLGraph, features: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Encode graph using GraphPFN."""
        num_nodes = graph.num_nodes()

        train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        train_mask[-1] = False  # At least 1 test node required

        n_train = int(train_mask.sum().item())
        y_train = torch.zeros(n_train, device=device)

        features = features.to(device)
        graph = graph.to(device)

        tfm_features = torch.cat([features[train_mask], features[~train_mask]], dim=0)

        for module in model.modules():
            if isinstance(module, GraphPFNLayerWrapper):
                module.train_mask = train_mask
                module.graph = graph
            if isinstance(module, MHA):
                module.batched = False

        if device.type == "cpu":
            sdpa_backends = [torch.nn.attention.SDPBackend.MATH]
        else:
            sdpa_backends = [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ]

        with torch.nn.attention.sdpa_kernel(sdpa_backends):
            out = model.tfm.forward(
                x=tfm_features.unsqueeze(0),
                y=y_train.unsqueeze(0),
                eval_pos=n_train,
                task_type="reg",
                checkpointing=True,
            )

        inv_order = torch.argsort((~train_mask).float(), stable=True)
        order = torch.argsort(inv_order, stable=True)
        return out["encoder_embed"].squeeze(0)[order]


    class GraphPFNLinkPredictor(nn.Module):
        """GraphPFN-based link predictor."""

        def __init__(self, encoder: GraphPFN, embed_dim: int, hidden_dim: int):
            super().__init__()
            self.encoder = encoder
            self.link_scorer = LinkScorer(embed_dim, hidden_dim)
            self.node_head = NodeHead(embed_dim, hidden_dim)

        def encode(self, graph: dgl.DGLGraph, features: torch.Tensor, device: torch.device) -> torch.Tensor:
            return graphpfn_encode(self.encoder, graph, features, device)


    def load_graphpfn_encoder(checkpoint_path: str, device: torch.device) -> GraphPFN:
        """Load pretrained GraphPFN encoder."""
        model = GraphPFN(edge_head=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device)
        return model


# ============== ROLAND Baseline ==============

class ROLANDBaseline(nn.Module):
    """
    ROLAND baseline model (adapted from DeXposure).
    Temporal GNN with GCN + GRU for link prediction.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, out_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

        self.gru1 = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru2 = nn.GRUCell(out_dim, out_dim)

        self.link_decoder = nn.Linear(out_dim, 1)
        self.dropout = dropout

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor,
                h1: Optional[torch.Tensor] = None,
                h2: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Returns:
            pred: Link predictions
            h1_new: Updated hidden state 1
            h2_new: Updated hidden state 2
        """
        # Preprocess
        h = self.preprocess(x)

        # GCN layer 1
        h = self.conv1(h, edge_index)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GRU update 1
        if h1 is not None:
            h = self.gru1(h, h1)
        h1_new = h.clone()

        # GCN layer 2
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GRU update 2
        if h2 is not None:
            h = self.gru2(h, h2)
        h2_new = h.clone()

        # Link prediction (Hadamard product)
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = h_src * h_dst
        pred = self.link_decoder(h_hadamard).squeeze(-1)

        return pred, h1_new, h2_new


# ============== Training Functions ==============

def train_graphpfn_epoch(model, pairs, optimizer, config, finetune_encoder: bool):
    """Train GraphPFN model for one epoch."""
    model.train()
    if not finetune_encoder:
        model.encoder.eval()

    device = torch.device(config.device)
    total_exist, total_weight, total_node, total_samples = 0.0, 0.0, 0.0, 0

    for sample in pairs:
        graph = dgl.graph((sample.edge_src_t, sample.edge_dst_t), num_nodes=len(sample.node_ids))
        features = torch.tensor(sample.features_t, dtype=torch.float32)

        if finetune_encoder:
            h = model.encode(graph, features, device)
        else:
            with torch.no_grad():
                h = model.encode(graph, features, device)
            h = h.detach()

        # Node loss
        node_true = torch.tensor(sample.y_node, dtype=torch.float32, device=device)
        node_mask = torch.tensor(sample.node_mask, dtype=torch.bool, device=device)
        node_pred = model.node_head(h)
        node_loss = F.smooth_l1_loss(node_pred[node_mask], node_true[node_mask]) if node_mask.any() else torch.tensor(0.0, device=device)

        # Edge prediction
        src = torch.tensor(sample.pair_src, dtype=torch.long, device=device)
        dst = torch.tensor(sample.pair_dst, dtype=torch.long, device=device)
        y_exist = torch.tensor(sample.y_exist, dtype=torch.float32, device=device)
        y_weight = torch.tensor(sample.y_weight, dtype=torch.float32, device=device)
        weight_mask = torch.tensor(sample.weight_mask, dtype=torch.float32, device=device)

        optimizer.zero_grad()

        logits, w_pred = model.link_scorer(h, src, dst)
        exist_loss = F.binary_cross_entropy_with_logits(logits, y_exist)

        mask = weight_mask > 0.5
        weight_loss = F.smooth_l1_loss(w_pred[mask], y_weight[mask]) if mask.any() else torch.tensor(0.0, device=device)

        loss = (config.exist_loss_weight * exist_loss +
                config.weight_loss_weight * weight_loss +
                config.node_loss_weight * node_loss)

        loss.backward()
        optimizer.step()

        total_exist += exist_loss.item()
        total_weight += weight_loss.item()
        total_node += node_loss.item()
        total_samples += 1

    return {
        "exist_loss": total_exist / max(total_samples, 1),
        "weight_loss": total_weight / max(total_samples, 1),
        "node_loss": total_node / max(total_samples, 1),
    }


def train_roland_epoch(model, pairs, optimizer, config, h1=None, h2=None):
    """Train ROLAND model for one epoch."""
    model.train()
    device = torch.device(config.device)
    total_loss, total_samples = 0.0, 0

    for sample in pairs:
        # Build PyG edge index
        edge_index = torch.tensor(
            np.stack([sample.edge_src_t, sample.edge_dst_t]),
            dtype=torch.long, device=device
        )
        x = torch.tensor(sample.features_t, dtype=torch.float32, device=device)

        edge_label_index = torch.tensor(
            np.stack([sample.pair_src, sample.pair_dst]),
            dtype=torch.long, device=device
        )
        y_exist = torch.tensor(sample.y_exist, dtype=torch.float32, device=device)

        optimizer.zero_grad()

        pred, h1, h2 = model(x, edge_index, edge_label_index, h1, h2)
        h1, h2 = h1.detach(), h2.detach()

        loss = F.binary_cross_entropy_with_logits(pred, y_exist)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += 1

    return {"loss": total_loss / max(total_samples, 1)}, h1, h2


# ============== Evaluation Functions ==============

def compute_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_values: List[int] = [100, 500, 1000]) -> Dict[str, float]:
    """Compute Recall@K for top-K predictions."""
    results = {}
    n_positive = int(y_true.sum())

    if n_positive == 0:
        return {f"recall@{k}": float("nan") for k in k_values}

    # Get indices of top-K predictions
    top_indices = np.argsort(y_score)[::-1]

    for k in k_values:
        if k > len(y_score):
            k = len(y_score)
        top_k_indices = top_indices[:k]
        true_positives_in_top_k = y_true[top_k_indices].sum()
        results[f"recall@{k}"] = float(true_positives_in_top_k / n_positive)

    return results


def compute_weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Compute Weighted MAE (weighted by true edge weights)."""
    if len(y_true) == 0 or weights.sum() == 0:
        return float("nan")

    # Normalize weights
    normalized_weights = weights / weights.sum()
    weighted_errors = normalized_weights * np.abs(y_true - y_pred)
    return float(weighted_errors.sum())


def evaluate_predictions(preds: List[Dict]) -> Dict[str, Any]:
    """Evaluate predictions and compute metrics."""
    exist_true, exist_score = [], []
    weight_true, weight_pred_list = [], []
    node_true, node_pred_list = [], []

    for item in preds:
        exist_prob = 1 / (1 + np.exp(-item["exist_logits"]))
        exist_true.append(item["y_exist"])
        exist_score.append(exist_prob)

        if item["weight_mask"].sum() > 0:
            mask = item["weight_mask"] > 0.5
            weight_true.append(item["y_weight"][mask])
            weight_pred_list.append(item["weight_pred"][mask])

        if item["node_mask"].any():
            node_true.append(item["y_node"][item["node_mask"]])
            node_pred_list.append(item["node_pred"][item["node_mask"]])

    exist_true = np.concatenate(exist_true)
    exist_score = np.concatenate(exist_score)

    # Exist metrics
    exist_metrics = {}
    if len(np.unique(exist_true)) > 1:
        exist_metrics["auprc"] = float(average_precision_score(exist_true, exist_score))
        exist_metrics["auroc"] = float(roc_auc_score(exist_true, exist_score))
        # Add Recall@K
        recall_at_k = compute_recall_at_k(exist_true, exist_score)
        exist_metrics.update(recall_at_k)
    else:
        exist_metrics["auprc"] = float("nan")
        exist_metrics["auroc"] = float("nan")
        exist_metrics["recall@100"] = float("nan")
        exist_metrics["recall@500"] = float("nan")
        exist_metrics["recall@1000"] = float("nan")

    # Weight metrics
    weight_metrics = {"mae": float("nan"), "rmse": float("nan"), "weighted_mae": float("nan")}
    if weight_true:
        wt = np.concatenate(weight_true)
        wp = np.concatenate(weight_pred_list)
        weight_metrics["mae"] = float(mean_absolute_error(wt, wp))
        weight_metrics["rmse"] = float(np.sqrt(mean_squared_error(wt, wp)))
        # Weighted MAE (weighted by true edge weights)
        weight_metrics["weighted_mae"] = compute_weighted_mae(wt, wp, np.exp(wt))  # exp to get original scale

    # Node metrics
    node_metrics = {"mae": float("nan"), "rmse": float("nan")}
    if node_true:
        nt = np.concatenate(node_true)
        np_ = np.concatenate(node_pred_list)
        node_metrics["mae"] = float(mean_absolute_error(nt, np_))
        node_metrics["rmse"] = float(np.sqrt(mean_squared_error(nt, np_)))

    return {"exist": exist_metrics, "weight": weight_metrics, "node": node_metrics}


def predict_graphpfn(model, pairs, config):
    """Generate predictions with GraphPFN model."""
    model.eval()
    device = torch.device(config.device)
    outputs = []

    with torch.no_grad():
        for sample in pairs:
            graph = dgl.graph((sample.edge_src_t, sample.edge_dst_t), num_nodes=len(sample.node_ids))
            features = torch.tensor(sample.features_t, dtype=torch.float32)
            h = model.encode(graph, features, device)
            node_pred = model.node_head(h).cpu().numpy()

            src = torch.tensor(sample.pair_src, dtype=torch.long, device=device)
            dst = torch.tensor(sample.pair_dst, dtype=torch.long, device=device)
            logits, w_pred = model.link_scorer(h, src, dst)

            outputs.append({
                "time_t": sample.time_t,
                "time_t1": sample.time_t1,
                "node_ids": sample.node_ids,
                "categories": sample.categories,
                "sizes_t": sample.sizes_t,
                "pair_src": sample.pair_src,
                "pair_dst": sample.pair_dst,
                "y_exist": sample.y_exist,
                "y_weight": sample.y_weight,
                "weight_mask": sample.weight_mask,
                "y_node": sample.y_node,
                "node_mask": sample.node_mask,
                "exist_logits": logits.cpu().numpy(),
                "weight_pred": w_pred.cpu().numpy(),
                "node_pred": node_pred,
            })

    return outputs


def predict_roland(model, pairs, config, h1=None, h2=None):
    """Generate predictions with ROLAND model."""
    model.eval()
    device = torch.device(config.device)
    outputs = []

    with torch.no_grad():
        for sample in pairs:
            edge_index = torch.tensor(
                np.stack([sample.edge_src_t, sample.edge_dst_t]),
                dtype=torch.long, device=device
            )
            x = torch.tensor(sample.features_t, dtype=torch.float32, device=device)
            edge_label_index = torch.tensor(
                np.stack([sample.pair_src, sample.pair_dst]),
                dtype=torch.long, device=device
            )

            pred, h1, h2 = model(x, edge_index, edge_label_index, h1, h2)

            outputs.append({
                "time_t": sample.time_t,
                "time_t1": sample.time_t1,
                "node_ids": sample.node_ids,
                "categories": sample.categories,
                "sizes_t": sample.sizes_t,
                "pair_src": sample.pair_src,
                "pair_dst": sample.pair_dst,
                "y_exist": sample.y_exist,
                "y_weight": sample.y_weight,
                "weight_mask": sample.weight_mask,
                "y_node": sample.y_node,
                "node_mask": sample.node_mask,
                "exist_logits": pred.cpu().numpy(),
                "weight_pred": np.zeros_like(sample.y_weight),  # ROLAND doesn't predict weight
                "node_pred": np.zeros(len(sample.node_ids)),  # ROLAND doesn't predict node
            })

    return outputs, h1, h2


def _summarize_metric_group(metrics_list: List[Dict[str, Any]], group: str, keys: List[str]) -> Dict[str, Dict[str, float]]:
    """Summarize mean/std for metric group across folds."""
    summary = {}
    for key in keys:
        values = []
        for metrics in metrics_list:
            value = metrics.get(group, {}).get(key, float("nan"))
            if isinstance(value, float) and math.isnan(value):
                continue
            values.append(value)
        if values:
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        else:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
    return summary


def aggregate_fold_metrics(fold_results: List[Dict[str, Any]], horizons: List[int]) -> Dict[str, Any]:
    """Aggregate fold metrics into mean/std summaries per horizon."""
    summary = {}
    for horizon in horizons:
        metrics_list = []
        for result in fold_results:
            if not result or "results" not in result:
                continue
            metrics = result["results"].get(f"h{horizon}")
            if metrics:
                metrics_list.append(metrics)
        if not metrics_list:
            continue
        summary[f"h{horizon}"] = {
            "n_folds": len(metrics_list),
            "exist": _summarize_metric_group(metrics_list, "exist", ["auprc", "auroc", "recall@100", "recall@500", "recall@1000"]),
            "weight": _summarize_metric_group(metrics_list, "weight", ["mae", "rmse", "weighted_mae"]),
            "node": _summarize_metric_group(metrics_list, "node", ["mae", "rmse"]),
        }
    return summary


def run_expanding_window_evaluation(
    config: ExperimentConfig,
    split_result: Dict[str, Any],
    date_to_snap: Dict[str, Dict],
    run_frozen: bool,
    run_finetuned: bool,
    run_roland: bool,
    save_predictions: bool,
    output_dir: Path
) -> Dict[str, Any]:
    """Run expanding-window walk-forward evaluation across folds."""
    folds = split_result.get("folds", [])
    if not folds:
        return {"method": "expanding_window_walk_forward", "folds": [], "summary": {}}

    print(f"\n{'='*60}")
    print("EXPANDING WINDOW WALK-FORWARD EVALUATION")
    print(f"{'='*60}")
    print(f"  Folds: {len(folds)}")

    fold_entries = []
    model_fold_results = {
        "graphpfn_frozen": [],
        "graphpfn_finetuned": [],
        "roland": [],
    }

    for fold in folds:
        fold_id = fold["fold_id"]
        train_snaps = [date_to_snap[d] for d in fold["train"] if d in date_to_snap]
        val_snaps = [date_to_snap[d] for d in fold["val"] if d in date_to_snap]
        test_snaps = [date_to_snap[d] for d in fold["test"] if d in date_to_snap]

        print(f"\n--- Fold {fold_id} ---")
        print(f"  Train: {fold['train'][0]} ~ {fold['train'][-1]} ({len(fold['train'])}w)")
        print(f"  Val:   {fold['val'][0]} ~ {fold['val'][-1]} ({len(fold['val'])}w)")
        print(f"  Test:  {fold['test'][0]} ~ {fold['test'][-1]} ({len(fold['test'])}w)")

        fold_entry = {
            "fold_id": fold_id,
            "train_range": f"{fold['train'][0]} ~ {fold['train'][-1]}",
            "val_range": f"{fold['val'][0]} ~ {fold['val'][-1]}",
            "test_range": f"{fold['test'][0]} ~ {fold['test'][-1]}",
        }

        if run_frozen:
            result = run_graphpfn_experiment(
                config,
                finetune=False,
                train_snaps=train_snaps,
                val_snaps=val_snaps,
                test_snaps=test_snaps,
                save_predictions=save_predictions,
                output_dir=output_dir,
                fold_id=fold_id
            )
            if result:
                fold_entry["graphpfn_frozen"] = result
                model_fold_results["graphpfn_frozen"].append(result)

        if run_finetuned:
            result = run_graphpfn_experiment(
                config,
                finetune=True,
                train_snaps=train_snaps,
                val_snaps=val_snaps,
                test_snaps=test_snaps,
                save_predictions=save_predictions,
                output_dir=output_dir,
                fold_id=fold_id
            )
            if result:
                fold_entry["graphpfn_finetuned"] = result
                model_fold_results["graphpfn_finetuned"].append(result)

        if run_roland:
            result = run_roland_experiment(
                config,
                train_snaps=train_snaps,
                val_snaps=val_snaps,
                test_snaps=test_snaps,
                save_predictions=save_predictions,
                output_dir=output_dir,
                fold_id=fold_id
            )
            if result:
                fold_entry["roland"] = result
                model_fold_results["roland"].append(result)

        fold_entries.append(fold_entry)

    summary = {}
    if model_fold_results["graphpfn_frozen"]:
        summary["graphpfn_frozen"] = aggregate_fold_metrics(model_fold_results["graphpfn_frozen"], config.forecast_horizons)
    if model_fold_results["graphpfn_finetuned"]:
        summary["graphpfn_finetuned"] = aggregate_fold_metrics(model_fold_results["graphpfn_finetuned"], config.forecast_horizons)
    if model_fold_results["roland"]:
        summary["roland"] = aggregate_fold_metrics(model_fold_results["roland"], config.forecast_horizons)

    return {
        "method": "expanding_window_walk_forward",
        "n_folds": len(fold_entries),
        "folds": fold_entries,
        "summary": summary,
    }


# ============== Main Experiment Functions ==============

def run_graphpfn_experiment(
    config: ExperimentConfig,
    finetune: bool = True,
    train_snaps: Optional[List[Dict]] = None,
    val_snaps: Optional[List[Dict]] = None,
    test_snaps: Optional[List[Dict]] = None,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None,
    fold_id: Optional[int] = None
):
    """
    Run GraphPFN experiment (frozen or finetuned).

    Args:
        config: Experiment configuration
        finetune: Whether to fine-tune encoder (True) or freeze (False)
        train_snaps: Pre-loaded training snapshots (if None, will load from config)
        val_snaps: Pre-loaded validation snapshots
        test_snaps: Pre-loaded test snapshots
        save_predictions: Whether to save predictions to CSV
        output_dir: Output directory for predictions
    """
    if not GRAPHPFN_AVAILABLE:
        print("GraphPFN not available, skipping...")
        return None

    mode = "Finetuned" if finetune else "Frozen"
    model_name = "graphpfn_finetuned" if finetune else "graphpfn_frozen"
    model_output_dir = None
    append_predictions = False
    if output_dir:
        model_output_dir = output_dir / model_name
        if fold_id is not None:
            model_output_dir = output_dir / f"{model_name}_fold{fold_id}"
    print(f"\n{'='*60}")
    print(f"GraphPFN Experiment ({mode})")
    print(f"{'='*60}")

    set_seed(config.seed)
    device = torch.device(config.device)

    # Load data if not provided
    if train_snaps is None:
        meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
        network_data = load_network_data(config.data_path)
        all_dates = sorted(network_data.keys())

        snapshots = [
            build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
            for date in all_dates
        ]

        # Split data by ratio (legacy mode)
        n = len(snapshots)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))

        train_snaps = snapshots[:train_end]
        val_snaps = snapshots[train_end:val_end]
        test_snaps = snapshots[val_end:]
        print("  Using ratio-based split (legacy mode)")
    else:
        print("  Using strict temporal split (recommended)")

    print(f"  Train: {len(train_snaps)}, Val: {len(val_snaps) if val_snaps else 0}, Test: {len(test_snaps) if test_snaps else 0}")

    # Load encoder
    encoder = load_graphpfn_encoder(config.checkpoint_path, device)
    embed_dim = encoder.tfm.embed_dim

    # Create model
    model = GraphPFNLinkPredictor(encoder, embed_dim, config.hidden_dim).to(device)

    # Set encoder trainability
    for p in model.encoder.parameters():
        p.requires_grad = finetune

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay
    )

    # Build pairs for each horizon
    results = {}
    for horizon in config.forecast_horizons:
        print(f"\n--- Horizon h={horizon} ---")

        train_pairs = build_week_pairs(train_snaps, config.neg_ratio, config.seed, horizon)
        val_pairs = build_week_pairs(val_snaps, config.neg_ratio, config.seed, horizon) if val_snaps else []
        test_pairs = build_week_pairs(test_snaps, config.neg_ratio, config.seed, horizon) if test_snaps else []

        print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

        # Train
        for epoch in range(config.epochs):
            losses = train_graphpfn_epoch(model, train_pairs, optimizer, config, finetune)
            if (epoch + 1) % 2 == 0:
                val_preds = predict_graphpfn(model, val_pairs, config)
                val_metrics = evaluate_predictions(val_preds)
                print(f"  Epoch {epoch+1}: loss={losses}, val_auprc={val_metrics['exist']['auprc']:.4f}")

        # Test
        test_preds = predict_graphpfn(model, test_pairs, config)
        test_metrics = evaluate_predictions(test_preds)
        results[f"h{horizon}"] = test_metrics

        print(f"  Test AUPRC: {test_metrics['exist']['auprc']:.4f}")
        print(f"  Test AUROC: {test_metrics['exist']['auroc']:.4f}")
        print(f"  Test Recall@100: {test_metrics['exist'].get('recall@100', 'N/A')}")
        print(f"  Test Weight MAE: {test_metrics['weight']['mae']:.4f}")
        print(f"  Test Weighted MAE: {test_metrics['weight']['weighted_mae']:.4f}")

        # Save predictions if requested
        if save_predictions and model_output_dir:
            edges_path, nodes_path = save_predictions_csv(
                test_preds,
                model_output_dir,
                model_name,
                horizon,
                fold_id,
                append=append_predictions
            )
            append_predictions = True
            print(f"  Predictions saved to: {edges_path.name}, {nodes_path.name}")

    result = {
        "model": f"GraphPFN ({mode})",
        "results": results,
        "config": {
            "finetune": finetune,
            "epochs": config.epochs,
            "seed": config.seed,
        }
    }

    if model_output_dir:
        metrics_path = save_metrics_json(result, model_output_dir)
        print(f"  Metrics saved to: {metrics_path.name}")

    return result


def run_roland_experiment(
    config: ExperimentConfig,
    train_snaps: Optional[List[Dict]] = None,
    val_snaps: Optional[List[Dict]] = None,
    test_snaps: Optional[List[Dict]] = None,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None,
    fold_id: Optional[int] = None
):
    """Run ROLAND baseline experiment."""
    print(f"\n{'='*60}")
    print("ROLAND Baseline Experiment")

    model_name = "roland"
    model_output_dir = None
    append_predictions = False
    if output_dir:
        model_output_dir = output_dir / model_name
        if fold_id is not None:
            model_output_dir = output_dir / f"{model_name}_fold{fold_id}"
    print(f"{'='*60}")

    set_seed(config.seed)
    device = torch.device(config.device)

    # Load data if not provided
    if train_snaps is None:
        meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
        network_data = load_network_data(config.data_path)
        all_dates = sorted(network_data.keys())

        snapshots = [
            build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
            for date in all_dates
        ]

        # Split data by ratio (legacy mode)
        n = len(snapshots)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))

        train_snaps = snapshots[:train_end]
        val_snaps = snapshots[train_end:val_end]
        test_snaps = snapshots[val_end:]
        print("  Using ratio-based split (legacy mode)")
    else:
        print("  Using strict temporal split (recommended)")

    # Get input dimension
    input_dim = train_snaps[0]["features"].shape[1]

    print(f"  Train: {len(train_snaps)}, Val: {len(val_snaps) if val_snaps else 0}, Test: {len(test_snaps) if test_snaps else 0}")

    results = {}
    for horizon in config.forecast_horizons:
        print(f"\n--- Horizon h={horizon} ---")

        # Create model
        model = ROLANDBaseline(input_dim, hidden_dim=64, out_dim=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        train_pairs = build_week_pairs(train_snaps, config.neg_ratio, config.seed, horizon)
        val_pairs = build_week_pairs(val_snaps, config.neg_ratio, config.seed, horizon) if val_snaps else []
        test_pairs = build_week_pairs(test_snaps, config.neg_ratio, config.seed, horizon) if test_snaps else []

        print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

        # Train
        h1, h2 = None, None
        for epoch in range(config.epochs):
            losses, h1, h2 = train_roland_epoch(model, train_pairs, optimizer, config, h1, h2)
            if (epoch + 1) % 2 == 0:
                val_preds, _, _ = predict_roland(model, val_pairs, config, h1, h2)
                val_metrics = evaluate_predictions(val_preds)
                print(f"  Epoch {epoch+1}: loss={losses['loss']:.4f}, val_auprc={val_metrics['exist']['auprc']:.4f}")

        # Test
        test_preds, _, _ = predict_roland(model, test_pairs, config, h1, h2)
        test_metrics = evaluate_predictions(test_preds)
        results[f"h{horizon}"] = test_metrics

        print(f"  Test AUPRC: {test_metrics['exist']['auprc']:.4f}")
        print(f"  Test AUROC: {test_metrics['exist']['auroc']:.4f}")

        if save_predictions and model_output_dir:
            edges_path, nodes_path = save_predictions_csv(
                test_preds,
                model_output_dir,
                model_name,
                horizon,
                fold_id,
                append=append_predictions
            )
            append_predictions = True
            print(f"  Predictions saved to: {edges_path.name}, {nodes_path.name}")

    result = {
        "model": "ROLAND",
        "results": results,
        "config": {"epochs": config.epochs, "seed": config.seed}
    }

    if model_output_dir:
        metrics_path = save_metrics_json(result, model_output_dir)
        print(f"  Metrics saved to: {metrics_path.name}")

    return result


def run_network_statistics(config: ExperimentConfig):
    """Compute network-level statistics for all snapshots."""
    print(f"\n{'='*60}")
    print("Computing Network Statistics")
    print(f"{'='*60}")

    meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
    network_data = load_network_data(config.data_path)
    all_dates = sorted(network_data.keys())

    # Build snapshots
    snapshots = []
    for date in all_dates:
        snap = build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
        snapshots.append({
            "date": date,
            "edge_index": np.stack([snap["edge_src"], snap["edge_dst"]]) if len(snap["edge_src"]) > 0 else np.zeros((2, 0), dtype=np.int64),
            "num_nodes": len(snap["node_ids"]),
            "edge_weights": snap["edge_weight"],
            "node_sizes": snap["sizes"],
            "node_sectors": snap["categories"],
        })

    # Compute statistics
    all_stats = compute_rolling_statistics(snapshots, category_list)

    # Convert to DataFrame
    df = pd.DataFrame(all_stats)

    # Save
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "network_statistics.csv", index=False)

    # Summary
    print(f"\nComputed statistics for {len(all_stats)} snapshots")
    print(f"Date range: {all_stats[0]['date']} to {all_stats[-1]['date']}")
    print(f"\nSummary statistics:")
    for col in ["density", "degree_gini", "tvl_hhi", "top_10_concentration"]:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    return {"statistics": all_stats, "summary": df.describe().to_dict()}


# ============== Shock Analysis (Task II: Policy-relevant Scenario Analysis) ==============

@dataclass
class ShockEvent:
    """Represents a market shock event for analysis."""
    name: str
    event_date: str  # YYYY-MM-DD
    pre_window_start: str  # Start of pre-event window
    post_window_end: str   # End of post-event window
    description: str = ""


# Key DeFi shock events (from Section 5.2 of the paper)
SHOCK_EVENTS = [
    ShockEvent(
        name="Terra/Luna Collapse",
        event_date="2022-05-09",
        pre_window_start="2022-04-25",  # 2 weeks before
        post_window_end="2022-05-23",   # 2 weeks after
        description="UST depeg and LUNA death spiral"
    ),
    ShockEvent(
        name="FTX Collapse",
        event_date="2022-11-07",
        pre_window_start="2022-10-31",  # 1 week before
        post_window_end="2022-11-21",   # 2 weeks after
        description="FTX exchange collapse and contagion"
    ),
]


def find_nearest_date(target: str, dates: List[str]) -> Optional[str]:
    """Find the nearest available date to the target date."""
    if target in dates:
        return target

    sorted_dates = sorted(dates)

    # Find closest date
    for date in sorted_dates:
        if date >= target:
            return date

    # Return last date if target is after all dates
    return sorted_dates[-1] if sorted_dates else None


def analyze_shock_network_changes(
    snapshots: List[Dict],
    event: ShockEvent,
    config: ExperimentConfig
) -> Dict[str, Any]:
    """
    Analyze network structure changes around a shock event.

    Computes:
    - Network statistics before/during/after the event
    - Changes in concentration metrics
    - Edge creation/deletion rates
    - Node size volatility
    """
    dates = [s["date"] for s in snapshots]

    # Find date indices
    pre_start_date = find_nearest_date(event.pre_window_start, dates)
    event_date = find_nearest_date(event.event_date, dates)
    post_end_date = find_nearest_date(event.post_window_end, dates)

    if not all([pre_start_date, event_date, post_end_date]):
        return {"error": f"Could not find dates for event {event.name}"}

    pre_start_idx = dates.index(pre_start_date)
    event_idx = dates.index(event_date)
    post_end_idx = dates.index(post_end_date)

    print(f"\n  Event: {event.name}")
    print(f"    Pre-window: {pre_start_date} (idx={pre_start_idx})")
    print(f"    Event date: {event_date} (idx={event_idx})")
    print(f"    Post-window: {post_end_date} (idx={post_end_idx})")

    # Split into periods
    pre_snapshots = snapshots[pre_start_idx:event_idx]
    event_snapshots = snapshots[event_idx:event_idx+1]  # Event week only
    post_snapshots = snapshots[event_idx+1:post_end_idx+1]

    results = {
        "event_name": event.name,
        "event_date": event_date,
        "pre_window": {"start": pre_start_date, "end": event_date, "num_weeks": len(pre_snapshots)},
        "post_window": {"start": event_date, "end": post_end_date, "num_weeks": len(post_snapshots)},
    }

    # Compute statistics for each period
    def compute_period_stats(snaps: List[Dict], period_name: str) -> Dict[str, float]:
        if not snaps:
            return {}

        all_stats = []
        for snap in snaps:
            edge_index = np.stack([snap["edge_src"], snap["edge_dst"]]) if len(snap["edge_src"]) > 0 else np.zeros((2, 0), dtype=np.int64)
            stats = compute_all_network_statistics(
                edge_index=edge_index,
                num_nodes=len(snap["node_ids"]),
                edge_weights=snap["edge_weight"],
                node_sizes=snap["sizes"],
            )
            all_stats.append(stats)

        # Aggregate
        agg_stats = {}
        for key in all_stats[0].keys():
            if key not in ["date"]:
                values = [s[key] for s in all_stats]
                agg_stats[f"{period_name}_{key}_mean"] = float(np.mean(values))
                agg_stats[f"{period_name}_{key}_std"] = float(np.std(values))

        return agg_stats

    results["pre_event_stats"] = compute_period_stats(pre_snapshots, "pre")
    results["event_stats"] = compute_period_stats(event_snapshots, "event")
    results["post_event_stats"] = compute_period_stats(post_snapshots, "post")

    # Compute change metrics
    if pre_snapshots and post_snapshots:
        # TVL change
        pre_tvl = np.mean([np.sum(s["sizes"]) for s in pre_snapshots])
        post_tvl = np.mean([np.sum(s["sizes"]) for s in post_snapshots])
        results["tvl_change_pct"] = float((post_tvl - pre_tvl) / (pre_tvl + EPS) * 100)

        # Edge count change
        pre_edges = np.mean([len(s["edge_src"]) for s in pre_snapshots])
        post_edges = np.mean([len(s["edge_src"]) for s in post_snapshots])
        results["edge_count_change_pct"] = float((post_edges - pre_edges) / (pre_edges + EPS) * 100)

        # Concentration change (Gini)
        pre_gini = np.mean([gini_coefficient(s["sizes"]) for s in pre_snapshots])
        post_gini = np.mean([gini_coefficient(s["sizes"]) for s in post_snapshots])
        results["gini_change"] = float(post_gini - pre_gini)

        # HHI change
        pre_hhi = np.mean([herfindahl_hirschman_index(s["sizes"]) for s in pre_snapshots])
        post_hhi = np.mean([herfindahl_hirschman_index(s["sizes"]) for s in post_snapshots])
        results["hhi_change"] = float(post_hhi - pre_hhi)

        print(f"    TVL change: {results['tvl_change_pct']:.2f}%")
        print(f"    Edge count change: {results['edge_count_change_pct']:.2f}%")
        print(f"    Gini change: {results['gini_change']:.4f}")
        print(f"    HHI change: {results['hhi_change']:.4f}")

    return results


def evaluate_model_during_shock(
    model,
    predict_fn,
    snapshots: List[Dict],
    event: ShockEvent,
    config: ExperimentConfig,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluate model prediction performance during a shock event.

    Compares:
    - Pre-event prediction accuracy
    - During-event prediction accuracy
    - Post-event prediction accuracy
    """
    dates = [s["date"] for s in snapshots]

    # Find date indices
    pre_start_date = find_nearest_date(event.pre_window_start, dates)
    event_date = find_nearest_date(event.event_date, dates)
    post_end_date = find_nearest_date(event.post_window_end, dates)

    if not all([pre_start_date, event_date, post_end_date]):
        return {"error": f"Could not find dates for event {event.name}"}

    pre_start_idx = dates.index(pre_start_date)
    event_idx = dates.index(event_date)
    post_end_idx = dates.index(post_end_date)

    results = {
        "event_name": event.name,
        "model_name": model_name,
    }

    # Evaluate on different periods
    def evaluate_period(snap_indices: List[int], period_name: str) -> Dict:
        if len(snap_indices) < 2:
            return {"error": "Not enough snapshots"}

        period_snaps = [snapshots[i] for i in snap_indices]
        pairs = build_week_pairs(period_snaps, config.neg_ratio, config.seed, horizon=1)

        if not pairs:
            return {"error": "No pairs built"}

        preds = predict_fn(model, pairs, config)
        metrics = evaluate_predictions(preds)

        return {
            f"{period_name}_auprc": metrics["exist"]["auprc"],
            f"{period_name}_auroc": metrics["exist"]["auroc"],
            f"{period_name}_weight_mae": metrics["weight"]["mae"],
            f"{period_name}_num_pairs": len(pairs),
        }

    # Pre-event period
    pre_indices = list(range(max(0, pre_start_idx - 4), pre_start_idx))
    if len(pre_indices) >= 2:
        results.update(evaluate_period(pre_indices, "pre"))

    # During-event period (event week and surrounding)
    event_indices = list(range(pre_start_idx, min(event_idx + 2, len(snapshots))))
    if len(event_indices) >= 2:
        results.update(evaluate_period(event_indices, "event"))

    # Post-event period
    post_indices = list(range(event_idx + 1, min(post_end_idx + 4, len(snapshots))))
    if len(post_indices) >= 2:
        results.update(evaluate_period(post_indices, "post"))

    # Compute performance degradation
    if "pre_auprc" in results and "event_auprc" in results:
        results["auprc_degradation"] = float(results["pre_auprc"] - results["event_auprc"])
        print(f"    AUPRC degradation during {event.name}: {results['auprc_degradation']:.4f}")

    return results


def run_shock_analysis(config: ExperimentConfig, model=None, model_type: str = "graphpfn"):
    """
    Run complete shock analysis for all predefined events.

    Args:
        config: Experiment configuration
        model: Optional pretrained model (if None, will train fresh)
        model_type: Type of model ("graphpfn" or "roland")
    """
    print(f"\n{'='*60}")
    print("Shock Analysis (Task II: Policy-relevant Scenario Analysis)")
    print(f"{'='*60}")

    # Load data
    meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
    network_data = load_network_data(config.data_path)
    all_dates = sorted(network_data.keys())

    # Build snapshots
    snapshots = [
        build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
        for date in all_dates
    ]

    print(f"Loaded {len(snapshots)} snapshots from {all_dates[0]} to {all_dates[-1]}")

    # Analyze network changes for each event
    network_change_results = {}
    for event in SHOCK_EVENTS:
        # Check if event is in our date range
        if event.event_date < all_dates[0] or event.event_date > all_dates[-1]:
            print(f"\n  Skipping {event.name}: event date {event.event_date} outside data range")
            continue

        results = analyze_shock_network_changes(snapshots, event, config)
        network_change_results[event.name] = results

    # Model performance evaluation (if model provided or can be trained)
    model_performance_results = {}

    if GRAPHPFN_AVAILABLE and model_type == "graphpfn":
        print("\n--- Training GraphPFN for shock analysis ---")
        device = torch.device(config.device)

        # Train model on data before first event
        first_event_date = min(e.pre_window_start for e in SHOCK_EVENTS)
        train_end_idx = 0
        for i, date in enumerate(all_dates):
            if date >= first_event_date:
                train_end_idx = i
                break

        if train_end_idx > 10:  # Need enough data to train
            train_snapshots = snapshots[:train_end_idx]

            # Load and setup model
            encoder = load_graphpfn_encoder(config.checkpoint_path, device)
            embed_dim = encoder.tfm.embed_dim
            model = GraphPFNLinkPredictor(encoder, embed_dim, config.hidden_dim).to(device)

            # Fine-tune on training data
            for p in model.encoder.parameters():
                p.requires_grad = True

            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            train_pairs = build_week_pairs(train_snapshots, config.neg_ratio, config.seed, horizon=1)

            print(f"  Training on {len(train_pairs)} pairs (before {first_event_date})")
            for epoch in range(config.epochs):
                train_graphpfn_epoch(model, train_pairs, optimizer, config, finetune_encoder=True)

            # Evaluate during each shock event
            for event in SHOCK_EVENTS:
                if event.event_date < all_dates[0] or event.event_date > all_dates[-1]:
                    continue

                results = evaluate_model_during_shock(
                    model,
                    predict_fn=predict_graphpfn,
                    snapshots=snapshots,
                    event=event,
                    config=config,
                    model_name="GraphPFN (Finetuned)"
                )
                model_performance_results[f"graphpfn_{event.name}"] = results

    elif model_type == "roland":
        print("\n--- Training ROLAND for shock analysis ---")
        device = torch.device(config.device)
        input_dim = snapshots[0]["features"].shape[1]

        # Train model
        first_event_date = min(e.pre_window_start for e in SHOCK_EVENTS)
        train_end_idx = next((i for i, d in enumerate(all_dates) if d >= first_event_date), len(all_dates)//2)

        if train_end_idx > 10:
            train_snapshots = snapshots[:train_end_idx]

            model = ROLANDBaseline(input_dim, hidden_dim=64, out_dim=32).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            train_pairs = build_week_pairs(train_snapshots, config.neg_ratio, config.seed, horizon=1)

            h1, h2 = None, None
            for epoch in range(config.epochs):
                _, h1, h2 = train_roland_epoch(model, train_pairs, optimizer, config, h1, h2)

            # Evaluate during each shock event
            for event in SHOCK_EVENTS:
                if event.event_date < all_dates[0] or event.event_date > all_dates[-1]:
                    continue

                def roland_predict_wrapper(model, pairs, config):
                    preds, _, _ = predict_roland(model, pairs, config, h1, h2)
                    return preds

                results = evaluate_model_during_shock(
                    model,
                    predict_fn=roland_predict_wrapper,
                    snapshots=snapshots,
                    event=event,
                    config=config,
                    model_name="ROLAND"
                )
                model_performance_results[f"roland_{event.name}"] = results

    # Compile all results
    all_results = {
        "network_changes": network_change_results,
        "model_performance": model_performance_results,
        "events_analyzed": [e.name for e in SHOCK_EVENTS if e.event_date >= all_dates[0] and e.event_date <= all_dates[-1]],
    }

    # Print summary table
    print(f"\n{'='*80}")
    print("SHOCK ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Event':<25} {'TVL Change':<15} {'Edge Change':<15} {'Gini Δ':<12} {'HHI Δ':<12}")
    print("-" * 80)

    for event_name, results in network_change_results.items():
        if "error" not in results:
            tvl_chg = f"{results.get('tvl_change_pct', 0):.1f}%"
            edge_chg = f"{results.get('edge_count_change_pct', 0):.1f}%"
            gini_chg = f"{results.get('gini_change', 0):.4f}"
            hhi_chg = f"{results.get('hhi_change', 0):.4f}"
            print(f"{event_name:<25} {tvl_chg:<15} {edge_chg:<15} {gini_chg:<12} {hhi_chg:<12}")

    if model_performance_results:
        print(f"\n{'Model + Event':<35} {'Pre AUPRC':<12} {'Event AUPRC':<12} {'Degradation':<12}")
        print("-" * 80)
        for key, results in model_performance_results.items():
            if "error" not in results:
                pre = f"{results.get('pre_auprc', float('nan')):.4f}"
                event = f"{results.get('event_auprc', float('nan')):.4f}"
                deg = f"{results.get('auprc_degradation', float('nan')):.4f}"
                print(f"{key:<35} {pre:<12} {event:<12} {deg:<12}")

    return all_results


# ============== Task III: Imputation ==============

def evaluate_imputation_pairs(
    pairs: List[WeekPair],
    preds: List[Dict[str, Any]],
    mask_ratio: float,
    rng: np.random.Generator
) -> Dict[str, float]:
    """Evaluate imputation performance on masked positives and nodes."""
    edge_recall = []
    edge_prob = []
    edge_mae = []
    edge_rmse = []
    edge_corr = []
    node_mae = []
    node_rmse = []

    for pair, pred in zip(pairs, preds):
        y_exist = pred["y_exist"]
        exist_prob = 1 / (1 + np.exp(-pred["exist_logits"]))

        pos_idx = np.where(y_exist > 0.5)[0]
        if len(pos_idx) > 0:
            num_mask = max(1, int(len(pos_idx) * mask_ratio))
            mask_idx = rng.choice(pos_idx, size=num_mask, replace=False)

            masked_probs = exist_prob[mask_idx]
            edge_recall.append(float(np.mean(masked_probs > 0.5)))
            edge_prob.append(float(np.mean(masked_probs)))

            true_w = pred["y_weight"][mask_idx]
            pred_w = pred["weight_pred"][mask_idx]
            edge_mae.append(float(np.mean(np.abs(true_w - pred_w))))
            edge_rmse.append(float(np.sqrt(np.mean((true_w - pred_w) ** 2))))
            if len(true_w) > 1 and np.std(true_w) > 0 and np.std(pred_w) > 0:
                edge_corr.append(float(np.corrcoef(true_w, pred_w)[0, 1]))

        node_mask = pred["node_mask"]
        if node_mask.any():
            node_indices = np.where(node_mask)[0]
            num_mask = max(1, int(len(node_indices) * mask_ratio))
            mask_nodes = rng.choice(node_indices, size=num_mask, replace=False)

            size_t = pair.sizes_t[mask_nodes]
            log_size_t = np.log1p(np.maximum(size_t, 0))
            true_log_t1 = log_size_t + pred["y_node"][mask_nodes]
            pred_log_t1 = log_size_t + pred["node_pred"][mask_nodes]

            node_mae.append(float(np.mean(np.abs(true_log_t1 - pred_log_t1))))
            node_rmse.append(float(np.sqrt(np.mean((true_log_t1 - pred_log_t1) ** 2))))

    results: Dict[str, float] = {}

    def add_stats(key: str, values: List[float]) -> None:
        if values:
            results[f"{key}_mean"] = float(np.mean(values))
            results[f"{key}_std"] = float(np.std(values))

    add_stats("edge_exist_recall", edge_recall)
    add_stats("edge_exist_avg_prob", edge_prob)
    add_stats("edge_weight_mae", edge_mae)
    add_stats("edge_weight_rmse", edge_rmse)
    add_stats("edge_weight_corr", edge_corr)
    add_stats("node_size_mae", node_mae)
    add_stats("node_size_rmse", node_rmse)

    return results


def run_imputation_experiment(
    config: ExperimentConfig,
    mask_ratios: List[float] = None,
    snapshots: Optional[List[Dict]] = None,
    date_splits: Optional[Dict[str, List[str]]] = None
):
    """
    Run Task III: Imputation experiment.

    Tests model's ability to recover masked edges and node attributes.

    Args:
        config: Experiment configuration
        mask_ratios: List of mask ratios to test (default: [0.1, 0.2, 0.3])
        snapshots: Optional pre-built snapshots
        date_splits: Optional train/test split (uses holdout if provided)
    """
    print(f"\n{'='*60}")
    print("Task III: Imputation Experiment")
    print(f"{'='*60}")

    if not GRAPHPFN_AVAILABLE:
        print("GraphPFN not available, skipping imputation experiment...")
        return None

    if mask_ratios is None:
        mask_ratios = [0.1, 0.2, 0.3]

    set_seed(config.seed)
    device = torch.device(config.device)
    rng = np.random.default_rng(config.seed)

    # Load data
    if snapshots is None:
        meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
        network_data = load_network_data(config.data_path)
        all_dates = sorted(network_data.keys())
        snapshots = [
            build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
            for date in all_dates
        ]
    else:
        all_dates = [snap["date"] for snap in snapshots]

    if date_splits is None:
        date_splits = get_single_split(all_dates)

    print(f"Loaded {len(snapshots)} snapshots")

    date_to_snap = {s["date"]: s for s in snapshots}
    train_snapshots = [date_to_snap[d] for d in date_splits["train"] if d in date_to_snap]
    test_snapshots = [date_to_snap[d] for d in date_splits["test"] if d in date_to_snap]

    # Load and train model
    encoder = load_graphpfn_encoder(config.checkpoint_path, device)
    embed_dim = encoder.tfm.embed_dim
    model = GraphPFNLinkPredictor(encoder, embed_dim, config.hidden_dim).to(device)

    for p in model.encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_pairs = build_week_pairs(train_snapshots, config.neg_ratio, config.seed, horizon=1)

    print(f"Training on {len(train_pairs)} pairs...")
    for epoch in range(config.epochs):
        train_graphpfn_epoch(model, train_pairs, optimizer, config, finetune_encoder=True)

    test_pairs = build_week_pairs(test_snapshots, config.neg_ratio, config.seed, horizon=1)
    if not test_pairs:
        print("No test pairs available for imputation evaluation.")
        return None

    preds = predict_graphpfn(model, test_pairs, config)

    results_by_ratio = {}

    for mask_ratio in mask_ratios:
        print(f"\n--- Mask ratio: {mask_ratio*100:.0f}% ---")

        ratio_results = {"mask_ratio": mask_ratio}
        ratio_results.update(evaluate_imputation_pairs(test_pairs, preds, mask_ratio, rng))
        results_by_ratio[f"ratio_{int(mask_ratio*100)}"] = ratio_results

        print(f"  Edge exist recall: {ratio_results.get('edge_exist_recall_mean', float('nan')):.4f}")
        print(f"  Edge weight MAE: {ratio_results.get('edge_weight_mae_mean', float('nan')):.4f}")
        print(f"  Node size MAE: {ratio_results.get('node_size_mae_mean', float('nan')):.4f}")

    print(f"\n{'='*80}")
    print("IMPUTATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Mask %':<12} {'Edge Recall':<15} {'Edge MAE':<15} {'Node MAE':<15}")
    print("-" * 60)

    for ratio_key, results in results_by_ratio.items():
        ratio = results.get("mask_ratio", 0) * 100
        edge_recall = results.get("edge_exist_recall_mean", float("nan"))
        edge_mae = results.get("edge_weight_mae_mean", float("nan"))
        node_mae = results.get("node_size_mae_mean", float("nan"))
        print(f"{ratio:<12.0f} {edge_recall:<15.4f} {edge_mae:<15.4f} {node_mae:<15.4f}")

    return {
        "model": "GraphPFN (Finetuned)",
        "results": results_by_ratio,
        "mask_ratios_tested": mask_ratios,
        "num_test_snapshots": len(test_snapshots),
    }

# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="DeXposure-FM Full Experiment Suite")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "frozen", "finetuned", "roland", "stats", "shock", "impute", "compare"],
                       help="Experiment mode: all, frozen, finetuned, roland, stats, shock, impute, compare")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shock-model", type=str, default="graphpfn",
                       choices=["graphpfn", "roland"],
                       help="Model type for shock analysis")
    # Temporal split arguments (金融惯例: Expanding Window Walk-Forward)
    parser.add_argument("--holdout-start", type=str, default="2025-01-01",
                       help="Hold-out test set start date (YYYY-MM-DD)")
    parser.add_argument("--min-train-weeks", type=int, default=104,
                       help="Minimum training weeks (default: 104 = 2 years)")
    parser.add_argument("--val-weeks", type=int, default=12,
                       help="Validation window size in weeks (default: 12)")
    parser.add_argument("--test-weeks", type=int, default=8,
                       help="Test window size per fold in weeks (default: 8)")
    parser.add_argument("--step-weeks", type=int, default=8,
                       help="Step size for expanding window folds in weeks (default: 8)")
    parser.add_argument("--rolling", action="store_true",
                       help="Run expanding window walk-forward evaluation (slower but more robust)")
    parser.add_argument("--save-predictions", action="store_true",
                       help="Save predictions to CSV files")
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.output_dir:
        config.output_dir = args.output_dir
    config.epochs = args.epochs
    config.seed = args.seed

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    # Load data once for all experiments
    print("Loading data...")
    meta_category, category_list, category_to_idx = load_metadata(config.meta_path)
    network_data = load_network_data(config.data_path)
    all_dates = sorted(network_data.keys())
    print(f"Loaded {len(all_dates)} snapshots: {all_dates[0]} to {all_dates[-1]}")

    # Build all snapshots
    print("Building snapshots...")
    all_snapshots = [
        build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
        for date in all_dates
    ]

    # Expanding Window Walk-Forward Split (金融惯例)
    split_result = expanding_window_split(
        all_dates,
        holdout_start=args.holdout_start,
        min_train_weeks=args.min_train_weeks,
        val_weeks=args.val_weeks,
        test_weeks=args.test_weeks,
        step_weeks=args.step_weeks,
    )

    print(f"\n{'='*60}")
    print("EXPANDING WINDOW WALK-FORWARD SPLIT (金融惯例)")
    print(f"{'='*60}")
    print(f"  Rolling Folds: {split_result['n_folds']}")
    print(f"  Min Train: {args.min_train_weeks} weeks | Val: {args.val_weeks} weeks | Test: {args.test_weeks} weeks")
    print(f"  Step Size: {args.step_weeks} weeks")
    if split_result['folds']:
        print(f"\n  Fold 1:  Train {split_result['folds'][0]['train'][0]} ~ {split_result['folds'][0]['train'][-1]} ({len(split_result['folds'][0]['train'])}w)")
        print(f"           Val   {split_result['folds'][0]['val'][0]} ~ {split_result['folds'][0]['val'][-1]} ({len(split_result['folds'][0]['val'])}w)")
        print(f"           Test  {split_result['folds'][0]['test'][0]} ~ {split_result['folds'][0]['test'][-1]} ({len(split_result['folds'][0]['test'])}w)")
        if len(split_result['folds']) > 1:
            last = split_result['folds'][-1]
            print(f"  Fold {last['fold_id']}: Train ... ~ {last['train'][-1]} ({len(last['train'])}w)")
            print(f"           Val   {last['val'][0]} ~ {last['val'][-1]} ({len(last['val'])}w)")
            print(f"           Test  {last['test'][0]} ~ {last['test'][-1]} ({len(last['test'])}w)")

    holdout = split_result['holdout']
    print(f"\n  Hold-out (Final Evaluation):")
    print(f"    Train: {holdout['train'][0] if holdout['train'] else 'N/A'} ~ {holdout['train'][-1] if holdout['train'] else 'N/A'} ({len(holdout['train'])} weeks)")
    print(f"    Val:   {holdout['val'][0] if holdout['val'] else 'N/A'} ~ {holdout['val'][-1] if holdout['val'] else 'N/A'} ({len(holdout['val'])} weeks)")
    print(f"    Test:  {holdout['test'][0] if holdout['test'] else 'N/A'} ~ {holdout['test'][-1] if holdout['test'] else 'N/A'} ({len(holdout['test'])} weeks)")
    print(f"    ⚠️  Hold-out test data is NEVER seen during training!")

    # 使用 holdout split 作为默认 (用于快速实验)
    date_splits = holdout

    # Compute and save data quality
    print("\nComputing data quality statistics...")
    data_quality = compute_data_quality(all_snapshots, network_data)
    with open(output_dir / "data_quality.json", "w") as f:
        json.dump(data_quality, f, indent=2)
    print(f"  Data quality saved to: {output_dir / 'data_quality.json'}")
    print(f"  Mean nodes/week: {data_quality['summary']['mean_nodes_per_week']:.1f}")
    print(f"  Mean edges/week: {data_quality['summary']['mean_edges_per_week']:.1f}")

    # Get snapshots by split
    date_to_snap = {s["date"]: s for s in all_snapshots}
    train_snaps = [date_to_snap[d] for d in date_splits["train"] if d in date_to_snap]
    val_snaps = [date_to_snap[d] for d in date_splits["val"] if d in date_to_snap]
    test_snaps = [date_to_snap[d] for d in date_splits["test"] if d in date_to_snap]

    run_frozen = args.mode in ["all", "frozen"]
    run_finetuned = args.mode in ["all", "finetuned"]
    run_roland = args.mode in ["all", "roland"]
    run_models = run_frozen or run_finetuned or run_roland

    rolling_results = None
    if args.rolling and run_models:
        rolling_results = run_expanding_window_evaluation(
            config=config,
            split_result=split_result,
            date_to_snap=date_to_snap,
            run_frozen=run_frozen,
            run_finetuned=run_finetuned,
            run_roland=run_roland,
            save_predictions=args.save_predictions,
            output_dir=output_dir,
        )

    all_results = {
        "temporal_split": {
            "method": "expanding_window_walk_forward",
            "n_folds": split_result["n_folds"],
            "holdout": {
                "train_weeks": len(date_splits["train"]),
                "val_weeks": len(date_splits["val"]),
                "test_weeks": len(date_splits["test"]),
                "train_range": f"{date_splits['train'][0]} ~ {date_splits['train'][-1]}" if date_splits['train'] else "N/A",
                "val_range": f"{date_splits['val'][0]} ~ {date_splits['val'][-1]}" if date_splits['val'] else "N/A",
                "test_range": f"{date_splits['test'][0]} ~ {date_splits['test'][-1]}" if date_splits['test'] else "N/A",
            },
            "config": split_result["config"],
        },
        "data_quality_summary": data_quality["summary"],
    }
    if rolling_results:
        all_results["rolling_folds"] = rolling_results

    # Run experiments based on mode
    if args.mode in ["all", "frozen"]:
        result = run_graphpfn_experiment(
            config,
            finetune=False,
            train_snaps=train_snaps,
            val_snaps=val_snaps,
            test_snaps=test_snaps,
            save_predictions=args.save_predictions,
            output_dir=output_dir
        )
        if result:
            all_results["graphpfn_frozen"] = result

    if args.mode in ["all", "finetuned"]:
        result = run_graphpfn_experiment(
            config,
            finetune=True,
            train_snaps=train_snaps,
            val_snaps=val_snaps,
            test_snaps=test_snaps,
            save_predictions=args.save_predictions,
            output_dir=output_dir
        )
        if result:
            all_results["graphpfn_finetuned"] = result

    if args.mode in ["all", "roland"]:
        result = run_roland_experiment(
            config,
            train_snaps=train_snaps,
            val_snaps=val_snaps,
            test_snaps=test_snaps,
            save_predictions=args.save_predictions,
            output_dir=output_dir
        )
        all_results["roland"] = result

    if args.mode in ["all", "stats"]:
        result = run_network_statistics(config)
        all_results["network_stats"] = result

    if args.mode in ["all", "shock"]:
        result = run_shock_analysis(config, model_type=args.shock_model)
        all_results["shock_analysis"] = result

    if args.mode in ["all", "impute"]:
        result = run_imputation_experiment(config, snapshots=all_snapshots, date_splits=date_splits)
        if result:
            all_results["imputation"] = result

    # Save all results
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    if args.mode in ["all", "compare"]:
        print(f"\n{'='*80}")
        print("COMPARISON TABLE - Multi-step Forecasting Results")
        print(f"{'='*80}")
        print(f"{'Model':<30} {'h=1 AUPRC':<15} {'h=3 AUPRC':<15} {'h=7 AUPRC':<15}")
        print("-" * 80)

        for name, result in all_results.items():
            if "results" in result:
                h1 = result["results"].get("h1", {}).get("exist", {}).get("auprc", float("nan"))
                h3 = result["results"].get("h3", {}).get("exist", {}).get("auprc", float("nan"))
                h7 = result["results"].get("h7", {}).get("exist", {}).get("auprc", float("nan"))
                print(f"{result.get('model', name):<30} {h1:<15.4f} {h3:<15.4f} {h7:<15.4f}")

    print(f"\nResults saved to: {output_dir / 'experiment_results.json'}")
    print(f"\nExperiment completed. Available modes:")
    print(f"  --mode frozen      : GraphPFN with frozen encoder")
    print(f"  --mode finetuned   : GraphPFN with fine-tuned encoder")
    print(f"  --mode roland      : ROLAND baseline")
    print(f"  --mode stats       : Network statistics computation")
    print(f"  --mode shock       : Shock analysis (Terra/FTX events)")
    print(f"  --mode impute      : Imputation experiment (Task III)")
    print(f"  --mode all         : Run all experiments")
    print(f"  --mode compare     : Print comparison table")


if __name__ == "__main__":
    main()
