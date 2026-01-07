#!/usr/bin/env python3
"""
DeXposure GraphPFN Link Prediction Experiment
按照 notebooks/dexposure_experiment_plan.ipynb 规范运行
"""
import os
import sys
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error

# 添加项目路径
GRAPHPFN_ROOT = Path(__file__).parent
sys.path.insert(0, str(GRAPHPFN_ROOT))

from lib.graphpfn.model import GraphPFN, GraphPFNLayerWrapper
from lib.limix.model.layer import MultiheadAttention as MHA
from lib.util import TaskType

# ============== 配置 ==============
DATA_PATH = "data/historical-network_week_2025-07-01.json"
META_PATH = "data/meta_df.csv"
OUTPUT_DIR = "output/dexposure_graphpfn_link"
CHECKPOINT_PATH = "checkpoints/graphpfn-v1.ckpt"

NEG_RATIO = 5
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

EDGE_BATCH_SIZE = 20000
HIDDEN_DIM = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 5

EXIST_LOSS_WEIGHT = 1.0
WEIGHT_LOSS_WEIGHT = 1.0
NODE_LOSS_WEIGHT = 0.5

FINETUNE_FULL = False
MAX_WEEKS = None  # None for all data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="DeXposure GraphPFN Experiment")
    parser.add_argument("--finetune", action="store_true", help="Finetune encoder (default: frozen)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=EDGE_BATCH_SIZE, help="Edge batch size")
    return parser.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============== 数据加载 ==============
def load_network_data(path: str) -> Dict:
    path = Path(path)
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


def node_features(node: Dict, meta_category: Dict, category_to_idx: Dict, category_list: List) -> Tuple[np.ndarray, float, str]:
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
    nodes = snapshot.get("nodes", [])
    links = snapshot.get("links", [])

    node_ids = []
    features = []
    sizes = []
    categories = []

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

    null_target = 0
    missing_endpoint = 0
    src = []
    dst = []
    weights = []

    for link in links:
        source = link.get("source")
        target = link.get("target")

        if target is None:
            null_target += 1
            continue
        if source is None:
            missing_endpoint += 1
            continue

        source = str(source)
        target = str(target)

        if source not in id_to_idx or target not in id_to_idx:
            missing_endpoint += 1
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


# ============== Week Pairs ==============
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
    neg_src = []
    neg_dst = []
    seen = set(pos_set)
    while len(neg_src) < num_neg:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if (u, v) in seen:
            continue
        seen.add((u, v))
        neg_src.append(u)
        neg_dst.append(v)
    return np.array(neg_src, dtype=np.int64), np.array(neg_dst, dtype=np.int64)


def build_week_pairs(snapshots: List[Dict], neg_ratio: int, seed: int) -> List[WeekPair]:
    rng = np.random.default_rng(seed)
    pairs = []

    for t in range(len(snapshots) - 1):
        snap_t = snapshots[t]
        snap_t1 = snapshots[t + 1]

        id_to_idx = {nid: i for i, nid in enumerate(snap_t["node_ids"])}
        size_t1_map = {nid: size for nid, size in zip(snap_t1["node_ids"], snap_t1["sizes"])}

        pos_src = []
        pos_dst = []
        pos_w = []
        pos_set = set()

        for src_idx, dst_idx, w in zip(snap_t1["edge_src"], snap_t1["edge_dst"], snap_t1["edge_weight"]):
            src_id = snap_t1["node_ids"][int(src_idx)]
            dst_id = snap_t1["node_ids"][int(dst_idx)]
            if src_id not in id_to_idx or dst_id not in id_to_idx:
                continue
            u = id_to_idx[src_id]
            v = id_to_idx[dst_id]
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

        num_neg = num_pos * neg_ratio
        neg_src, neg_dst = sample_negatives(len(snap_t["node_ids"]), pos_set, num_neg, rng)

        pair_src = np.concatenate([pos_src, neg_src])
        pair_dst = np.concatenate([pos_dst, neg_dst])
        y_exist = np.concatenate([np.ones(num_pos, dtype=np.float32), np.zeros(num_neg, dtype=np.float32)])
        y_weight = np.concatenate([pos_w, np.zeros(num_neg, dtype=np.float32)])
        weight_mask = np.concatenate([np.ones(num_pos, dtype=np.float32), np.zeros(num_neg, dtype=np.float32)])

        order = rng.permutation(len(pair_src))
        pair_src = pair_src[order]
        pair_dst = pair_dst[order]
        y_exist = y_exist[order]
        y_weight = y_weight[order]
        weight_mask = weight_mask[order]

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
            neg_edge_count=num_neg,
        ))

    return pairs


# ============== Model ==============
def graphpfn_encode(model: GraphPFN, graph: dgl.DGLGraph, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    num_nodes = graph.num_nodes()

    # GraphPFN requires at least 1 test node (eval_pos < x.shape[1])
    # We set the last node as "test" node with dummy label
    train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    train_mask[-1] = False  # At least 1 test node required

    n_train = int(train_mask.sum().item())
    y_train = torch.zeros(n_train, device=device)

    features = features.to(device)
    graph = graph.to(device)

    tfm_features = torch.cat([features[train_mask, ...], features[~train_mask, ...]], dim=0)

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
            eval_pos=n_train,  # Must be < num_nodes
            task_type="reg",
            checkpointing=True,
        )

    inv_order = torch.argsort((~train_mask).float(), stable=True)
    order = torch.argsort(inv_order, stable=True)
    encoder_embed = out["encoder_embed"].squeeze(0)[order, ...]
    return encoder_embed


class LinkScorer(nn.Module):
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
        h_u = h[src]
        h_v = h[dst]
        z = torch.cat([h_u, h_v, h_u * h_v, (h_u - h_v).abs()], dim=-1)
        exist_logits = self.exist_head(z).squeeze(-1)
        weight_pred = self.weight_head(z).squeeze(-1)
        return exist_logits, weight_pred


class NodeHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class GraphPFNLinkPredictor(nn.Module):
    def __init__(self, encoder: GraphPFN, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.link_scorer = LinkScorer(embed_dim, hidden_dim)
        self.node_head = NodeHead(embed_dim, hidden_dim)

    def encode(self, graph: dgl.DGLGraph, features: torch.Tensor, device: torch.device) -> torch.Tensor:
        return graphpfn_encode(self.encoder, graph, features, device=device)


def load_graphpfn_encoder(checkpoint_path: str, device: torch.device) -> GraphPFN:
    model = GraphPFN(edge_head=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    return model


def set_encoder_trainable(model: GraphPFN, trainable: bool):
    for p in model.parameters():
        p.requires_grad = trainable


# ============== Training ==============
def iter_edge_batches(num_pairs: int, batch_size: Optional[int], rng=None, shuffle: bool = True):
    idx = np.arange(num_pairs)
    if shuffle and rng is not None:
        rng.shuffle(idx)
    if batch_size is None:
        yield idx
    else:
        for start in range(0, num_pairs, batch_size):
            yield idx[start:start + batch_size]


def train_one_epoch(model, pairs, optimizer, device, edge_batch_size, finetune_encoder):
    model.train()
    if finetune_encoder:
        model.encoder.train()
    else:
        model.encoder.eval()

    rng = np.random.default_rng(SEED)
    total_exist = 0.0
    total_weight = 0.0
    total_node = 0.0
    total_samples = 0

    for sample in pairs:
        graph = dgl.graph((sample.edge_src_t, sample.edge_dst_t), num_nodes=len(sample.node_ids))
        features = torch.tensor(sample.features_t, dtype=torch.float32)

        if finetune_encoder:
            h = model.encode(graph, features, device=device)
        else:
            with torch.no_grad():
                h = model.encode(graph, features, device=device)
            h = h.detach()

        node_true = torch.tensor(sample.y_node, dtype=torch.float32, device=device)
        node_mask = torch.tensor(sample.node_mask, dtype=torch.bool, device=device)
        node_pred = model.node_head(h)
        if node_mask.any():
            node_loss = F.smooth_l1_loss(node_pred[node_mask], node_true[node_mask])
        else:
            node_loss = torch.tensor(0.0, device=device)

        src = torch.tensor(sample.pair_src, dtype=torch.long, device=device)
        dst = torch.tensor(sample.pair_dst, dtype=torch.long, device=device)
        y_exist = torch.tensor(sample.y_exist, dtype=torch.float32, device=device)
        y_weight = torch.tensor(sample.y_weight, dtype=torch.float32, device=device)
        weight_mask = torch.tensor(sample.weight_mask, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        batches = list(iter_edge_batches(len(src), edge_batch_size, rng=rng, shuffle=True))
        num_batches = len(batches)

        # Gradient accumulation: 每个 batch 单独 backward，梯度累积
        for b, batch_idx in enumerate(batches):
            logits, w_pred = model.link_scorer(h, src[batch_idx], dst[batch_idx])
            exist_loss = F.binary_cross_entropy_with_logits(logits, y_exist[batch_idx])

            mask = weight_mask[batch_idx] > 0.5
            if mask.any():
                weight_loss = F.smooth_l1_loss(w_pred[mask], y_weight[batch_idx][mask])
            else:
                weight_loss = torch.tensor(0.0, device=device)

            batch_loss = (EXIST_LOSS_WEIGHT * exist_loss) + (WEIGHT_LOSS_WEIGHT * weight_loss)

            # 除以 batch 数归一化梯度
            scaled_loss = batch_loss / num_batches

            # 最后一个 batch 不需要 retain_graph
            is_last_batch = (b == num_batches - 1) and (NODE_LOSS_WEIGHT <= 0)
            scaled_loss.backward(retain_graph=not is_last_batch)

            total_exist += exist_loss.item()
            total_weight += weight_loss.item()

        if NODE_LOSS_WEIGHT > 0:
            node_loss_scaled = (NODE_LOSS_WEIGHT * node_loss) / num_batches
            node_loss_scaled.backward()

        total_node += node_loss.item()
        total_samples += 1
        optimizer.step()

    return {
        "exist_loss": total_exist / max(total_samples, 1),
        "weight_loss": total_weight / max(total_samples, 1),
        "node_loss": total_node / max(total_samples, 1),
    }


def predict_samples(model, pairs, device, edge_batch_size):
    model.eval()
    outputs = []

    with torch.no_grad():
        for sample in pairs:
            graph = dgl.graph((sample.edge_src_t, sample.edge_dst_t), num_nodes=len(sample.node_ids))
            features = torch.tensor(sample.features_t, dtype=torch.float32)
            h = model.encode(graph, features, device=device)
            node_pred = model.node_head(h).cpu().numpy()

            src = torch.tensor(sample.pair_src, dtype=torch.long, device=device)
            dst = torch.tensor(sample.pair_dst, dtype=torch.long, device=device)

            all_logits = []
            all_weight = []

            if edge_batch_size is None:
                logits, w_pred = model.link_scorer(h, src, dst)
                all_logits.append(logits.cpu())
                all_weight.append(w_pred.cpu())
            else:
                for batch_idx in iter_edge_batches(len(src), edge_batch_size, rng=None, shuffle=False):
                    logits, w_pred = model.link_scorer(h, src[batch_idx], dst[batch_idx])
                    all_logits.append(logits.cpu())
                    all_weight.append(w_pred.cpu())

            logits = torch.cat(all_logits).numpy()
            weight_pred = torch.cat(all_weight).numpy()

            outputs.append({
                "time_t": sample.time_t,
                "time_t1": sample.time_t1,
                "pair_src": sample.pair_src,
                "pair_dst": sample.pair_dst,
                "y_exist": sample.y_exist,
                "y_weight": sample.y_weight,
                "weight_mask": sample.weight_mask,
                "node_ids": sample.node_ids,
                "categories": sample.categories,
                "sizes_t": sample.sizes_t,
                "y_node": sample.y_node,
                "node_mask": sample.node_mask,
                "exist_logits": logits,
                "weight_pred": weight_pred,
                "node_pred": node_pred,
            })

    return outputs


def evaluate_predictions(preds, k=100):
    exist_true = []
    exist_score = []
    weight_true = []
    weight_pred_list = []
    node_true = []
    node_pred_list = []

    for item in preds:
        y_exist = item["y_exist"]
        y_weight = item["y_weight"]
        weight_mask = item["weight_mask"]
        exist_prob = 1 / (1 + np.exp(-item["exist_logits"]))

        exist_true.append(y_exist)
        exist_score.append(exist_prob)

        if weight_mask.sum() > 0:
            weight_true.append(y_weight[weight_mask > 0.5])
            weight_pred_list.append(item["weight_pred"][weight_mask > 0.5])

        node_mask = item["node_mask"]
        if node_mask.any():
            node_true.append(item["y_node"][node_mask])
            node_pred_list.append(item["node_pred"][node_mask])

    exist_true = np.concatenate(exist_true)
    exist_score = np.concatenate(exist_score)

    exist_metrics = {}
    if len(np.unique(exist_true)) > 1:
        exist_metrics["auprc"] = float(average_precision_score(exist_true, exist_score))
        exist_metrics["auroc"] = float(roc_auc_score(exist_true, exist_score))
    else:
        exist_metrics["auprc"] = float("nan")
        exist_metrics["auroc"] = float("nan")

    weight_metrics = {"mae": float("nan"), "rmse": float("nan")}
    if weight_true:
        wt = np.concatenate(weight_true)
        wp = np.concatenate(weight_pred_list)
        weight_metrics["mae"] = float(mean_absolute_error(wt, wp))
        weight_metrics["rmse"] = float(np.sqrt(mean_squared_error(wt, wp)))

    node_metrics = {"mae": float("nan"), "rmse": float("nan")}
    if node_true:
        nt = np.concatenate(node_true)
        np_ = np.concatenate(node_pred_list)
        node_metrics["mae"] = float(mean_absolute_error(nt, np_))
        node_metrics["rmse"] = float(np.sqrt(mean_squared_error(nt, np_)))

    return {
        "exist": exist_metrics,
        "weight": weight_metrics,
        "node": node_metrics,
    }


# ============== Main ==============
def main():
    args = parse_args()

    # 根据参数设置
    finetune_full = args.finetune
    output_dir = args.output_dir or (
        "output/dexposure_graphpfn_finetuned" if finetune_full
        else "output/dexposure_graphpfn_frozen"
    )
    epochs = args.epochs
    edge_batch_size = args.batch_size

    mode_str = "Finetuned" if finetune_full else "Frozen Probing"
    print("=" * 60)
    print(f"DeXposure GraphPFN Link Prediction Experiment ({mode_str})")
    print("=" * 60)

    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    print("\n[1/7] Loading metadata...")
    meta_df = pd.read_csv(META_PATH)
    meta_df["id"] = meta_df["id"].astype(str)
    category_list = sorted(meta_df["category"].dropna().unique().tolist())
    if "Unknown" not in category_list:
        category_list.append("Unknown")
    category_to_idx = {c: i for i, c in enumerate(category_list)}
    meta_category = meta_df.set_index("id")["category"].to_dict()
    print(f"  Categories: {len(category_list)}")

    # Load network data
    print("\n[2/7] Loading network data...")
    network_data = load_network_data(DATA_PATH)
    all_dates = sorted(network_data.keys())
    if MAX_WEEKS is not None:
        all_dates = all_dates[:MAX_WEEKS]
    print(f"  Snapshots: {len(all_dates)}")

    # Build snapshots
    print("\n[3/7] Building snapshots...")
    snapshots = []
    for date in all_dates:
        snap = build_snapshot(date, network_data[date], meta_category, category_to_idx, category_list)
        snapshots.append(snap)
        print(f"  {date}: {len(snap['node_ids'])} nodes, {len(snap['edge_src'])} edges")

    # Build week pairs
    print("\n[4/7] Building week pairs...")
    week_pairs = build_week_pairs(snapshots, NEG_RATIO, SEED)
    print(f"  Week pairs: {len(week_pairs)}")

    if not week_pairs:
        print("ERROR: No week pairs created!")
        return

    # Split data
    n = len(week_pairs)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_pairs = week_pairs[:train_end]
    val_pairs = week_pairs[train_end:val_end]
    test_pairs = week_pairs[val_end:]
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # Load encoder
    print("\n[5/7] Loading GraphPFN encoder...")
    encoder = load_graphpfn_encoder(CHECKPOINT_PATH, DEVICE)
    embed_dim = encoder.tfm.embed_dim
    print(f"  Embed dim: {embed_dim}")

    # Create model
    print("\n[6/7] Creating model...")
    model = GraphPFNLinkPredictor(encoder=encoder, embed_dim=embed_dim, hidden_dim=HIDDEN_DIM).to(DEVICE)
    set_encoder_trainable(model.encoder, trainable=finetune_full)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Training
    print(f"\n[7/7] Training ({epochs} epochs, finetune={finetune_full})...")
    best_val_auprc = -float("inf")
    best_model_path = Path(output_dir) / "best_model.ckpt"

    for epoch in range(1, epochs + 1):
        train_losses = train_one_epoch(model, train_pairs, optimizer, DEVICE, edge_batch_size, finetune_full)

        # Validation
        model.eval()
        val_preds = predict_samples(model, val_pairs, DEVICE, edge_batch_size)
        val_metrics = evaluate_predictions(val_preds)

        val_auprc = val_metrics["exist"].get("auprc", 0.0)
        if isinstance(val_auprc, float) and not math.isnan(val_auprc):
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                torch.save(model.state_dict(), best_model_path)
                print(f"Epoch {epoch}: train_loss={train_losses} | val_auprc={val_auprc:.4f} *BEST*")
            else:
                print(f"Epoch {epoch}: train_loss={train_losses} | val_auprc={val_auprc:.4f}")
        else:
            print(f"Epoch {epoch}: train_loss={train_losses} | val_auprc=N/A")

    # Load best model
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    model.eval()
    train_preds = predict_samples(model, train_pairs, DEVICE, edge_batch_size)
    val_preds = predict_samples(model, val_pairs, DEVICE, edge_batch_size)
    test_preds = predict_samples(model, test_pairs, DEVICE, edge_batch_size)

    train_metrics = evaluate_predictions(train_preds)
    val_metrics = evaluate_predictions(val_preds)
    test_metrics = evaluate_predictions(test_preds)

    print(f"\nTrain: {train_metrics}")
    print(f"Val:   {val_metrics}")
    print(f"Test:  {test_metrics}")

    # Save results
    metrics_payload = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "config": {
            "neg_ratio": NEG_RATIO,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "hidden_dim": HIDDEN_DIM,
            "epochs": epochs,
            "seed": SEED,
            "finetune_full": finetune_full,
        },
        "generated_at": datetime.now().isoformat(),
    }

    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
