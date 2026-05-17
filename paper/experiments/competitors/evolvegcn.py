#!/usr/bin/env python3
"""m3_evolvegcn: EvolveGCN baseline (Pareja et al., KDD 2020).

Trains an EvolveGCN-O model on the DeXposure weekly graph sequence and
produces link-weight predictions for the risk forecasting benchmarks.

This baseline is a non-agent method: it produces predicted graphs but
does not generate alerts, scenarios, or decision tickets.  It is
assessed on b1_forecast (risk forecasting), b4_stress (stress test),
and b6_robustness (robustness).

Architecture:
    - EvolveGCN-O uses an LSTM to update GCN weight matrices over time.
    - Input: sliding window of W graphs (default W=8).
    - Output: edge weight predictions for horizon h.
    - Separate model trained per horizon.

Dependencies:
    pip install torch-geometric-temporal
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dexposure_agent.data_loader import SnapshotLoader, parse_date_range
from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────

WINDOW_SIZE = 8       # input sequence length (weeks)
HIDDEN_DIM = 64       # GCN hidden dimension
NUM_GCN_LAYERS = 2    # number of EvolveGCN layers
EPOCHS = 100          # training epochs
LR = 1e-3
PATIENCE = 10         # early stopping patience
BATCH_SIZE = 1        # one graph sequence per batch


class EvolveGCNO(nn.Module):
    """EvolveGCN-O: GCN with LSTM-updated weight matrices.

    Sparse adjacency variant; scales to graphs with tens of thousands of nodes.
    """

    def __init__(self, n_features: int, hidden_dim: int, n_nodes_max: int):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_nodes_max = n_nodes_max

        # GCN layer weights updated by LSTM
        self.lstm1 = nn.LSTMCell(
            input_size=n_features * hidden_dim,
            hidden_size=n_features * hidden_dim,
        )
        self.lstm2 = nn.LSTMCell(
            input_size=hidden_dim * hidden_dim,
            hidden_size=hidden_dim * hidden_dim,
        )

        # Initial GCN weights
        self.w1_init = nn.Parameter(torch.randn(n_features, hidden_dim) * 0.01)
        self.w2_init = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

        # Edge predictor: bilinear on node embeddings
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        adj_sequence: list[torch.Tensor],
        feat_sequence: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass over a sequence of graphs.

        Args:
            adj_sequence: List of [N, N] adjacency matrices (normalized).
            feat_sequence: List of [N, F] node feature matrices.

        Returns:
            Node embeddings [N, hidden_dim] from the last timestep.
        """
        h1 = self.w1_init.flatten().unsqueeze(0)  # [1, F*H]
        c1 = torch.zeros_like(h1)
        h2 = self.w2_init.flatten().unsqueeze(0)  # [1, H*H]
        c2 = torch.zeros_like(h2)

        node_embed = None

        for adj, feat in zip(adj_sequence, feat_sequence):
            # Update GCN weights via LSTM
            h1, c1 = self.lstm1(h1, (h1, c1))
            w1 = h1.view(self.n_features, self.hidden_dim)

            h2, c2 = self.lstm2(h2, (h2, c2))
            w2 = h2.view(self.hidden_dim, self.hidden_dim)

            # GCN forward: H = ReLU(A * X * W1) * W2
            # adj is a sparse COO tensor; use torch.sparse.mm for sparse @ dense.
            x = torch.sparse.mm(adj, feat)       # [N, F]
            x = F.relu(torch.mm(x, w1))          # [N, H]
            x = torch.sparse.mm(adj, x)          # [N, H]
            node_embed = torch.mm(x, w2)         # [N, H]

        return node_embed

    def predict_edges(
        self,
        node_embed: torch.Tensor,
        edge_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict edge weights for given (src, tgt) pairs.

        Args:
            node_embed: [N, H] node embeddings.
            edge_indices: [E, 2] tensor of (src, tgt) index pairs.

        Returns:
            [E] predicted edge weights (non-negative).
        """
        src_embed = node_embed[edge_indices[:, 0]]  # [E, H]
        tgt_embed = node_embed[edge_indices[:, 1]]  # [E, H]
        pair_feat = torch.cat([src_embed, tgt_embed], dim=-1)  # [E, 2H]
        pred = self.edge_predictor(pair_feat).squeeze(-1)  # [E]
        return F.softplus(pred)  # ensure non-negative


# ──────────────────────────────────────────────────────────────────────────
# Data conversion
# ──────────────────────────────────────────────────────────────────────────

def _build_node_index(snapshots: list[GraphSnapshot]) -> dict[str, int]:
    """Build a global node-to-index mapping from all snapshots."""
    all_nodes: set[str] = set()
    for snap in snapshots:
        all_nodes.update(snap.nodes.keys())
        for e in snap.edges:
            all_nodes.add(e.source)
            all_nodes.add(e.target)
    return {node: i for i, node in enumerate(sorted(all_nodes))}


def _snap_to_tensors(
    snap: GraphSnapshot,
    node_index: dict[str, int],
    n_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[int, int], float]]:
    """Convert a GraphSnapshot to a sparse normalized adjacency + dense features.

    Returns:
        adj: sparse COO [N, N] = D^-1/2 (A + I) D^-1/2.
        feat: [N, F] node features (F=5: log_size, num_tokens, max_share, entropy, degree).
        edge_weights: dict mapping (src_idx, tgt_idx) -> raw weight (for the loss target).
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    deg_counter: dict[int, int] = {}
    edge_weights: dict[tuple[int, int], float] = {}

    for e in snap.edges:
        if e.source in node_index and e.target in node_index:
            si = node_index[e.source]
            ti = node_index[e.target]
            rows.append(si)
            cols.append(ti)
            vals.append(1.0)
            edge_weights[(si, ti)] = e.weight
            deg_counter[si] = deg_counter.get(si, 0) + 1
            deg_counter[ti] = deg_counter.get(ti, 0) + 1

    # Self-loops (A + I)
    for i in range(n_nodes):
        rows.append(i)
        cols.append(i)
        vals.append(1.0)

    rows_t = torch.tensor(rows, dtype=torch.long, device=device)
    cols_t = torch.tensor(cols, dtype=torch.long, device=device)
    vals_t = torch.tensor(vals, dtype=torch.float32, device=device)

    # Degree (row sum) -- include self-loops
    deg = torch.zeros(n_nodes, device=device)
    deg.index_add_(0, rows_t, vals_t)
    deg_inv_sqrt = deg.clamp(min=1e-8).pow(-0.5)

    # Symmetric normalization: w_ij / sqrt(d_i * d_j)
    norm_vals = vals_t * deg_inv_sqrt[rows_t] * deg_inv_sqrt[cols_t]

    indices = torch.stack([rows_t, cols_t], dim=0)
    adj = torch.sparse_coo_tensor(
        indices, norm_vals, (n_nodes, n_nodes), device=device
    ).coalesce()

    # Node features (dense, mostly zero rows for inactive nodes)
    feat = torch.zeros(n_nodes, 5, device=device)
    for nid, nf in snap.nodes.items():
        if nid in node_index:
            idx = node_index[nid]
            feat[idx, 0] = nf.log_size
            feat[idx, 1] = float(nf.num_tokens)
            feat[idx, 2] = nf.max_share
            feat[idx, 3] = nf.entropy
            feat[idx, 4] = float(deg_counter.get(idx, 0))  # incident-edge count

    return adj, feat, edge_weights


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────

def train_evolvegcn(
    data_dir: str,
    train_split: str = "2020-03~2024-06",
    val_split: str = "2024-07~2024-12",
    horizon: int = 4,
    checkpoint_dir: str = "checkpoints/evolvegcn",
    device_str: str = "auto",
) -> Path:
    """Train EvolveGCN-O for a given horizon.

    Args:
        data_dir: Path to graph snapshots.
        train_split: Training period.
        val_split: Validation period for early stopping.
        horizon: Prediction horizon in weeks.
        checkpoint_dir: Where to save the trained model.
        device_str: 'cuda', 'cpu', or 'auto'.

    Returns:
        Path to saved checkpoint.
    """
    device = torch.device(
        "cuda" if device_str == "auto" and torch.cuda.is_available()
        else device_str if device_str != "auto" else "cpu"
    )
    logger.info(f"EvolveGCN training | h={horizon} | device={device}")

    loader = SnapshotLoader(data_dir=data_dir)
    train_snaps = loader.load(date_range=train_split)
    val_snaps = loader.load(date_range=val_split)
    all_snaps = train_snaps + val_snaps

    node_index = _build_node_index(all_snaps)
    n_nodes = len(node_index)
    n_features = 5

    logger.info(f"Global node set: {n_nodes} nodes")

    # Convert all snapshots to tensors
    tensor_cache: dict[str, tuple[torch.Tensor, torch.Tensor, dict]] = {}
    for snap in all_snaps:
        adj, feat, ew = _snap_to_tensors(snap, node_index, n_nodes, device)
        tensor_cache[snap.date] = (adj, feat, ew)

    # Build training sequences: (window of graphs) -> target graph at t+h
    all_dates = loader.dates
    _, train_end_dt = parse_date_range(train_split)

    train_date_set = {s.date for s in train_snaps}
    val_date_set = {s.date for s in val_snaps}

    def _build_sequences(date_set: set[str]) -> list[tuple[list[str], str]]:
        """Build (input_dates, target_date) pairs."""
        sequences = []
        for date in all_dates:
            if date not in date_set:
                continue
            date_idx = all_dates.index(date)
            start_idx = date_idx - WINDOW_SIZE + 1
            target_idx = date_idx + horizon
            if start_idx < 0 or target_idx >= len(all_dates):
                continue
            input_dates = all_dates[start_idx:date_idx + 1]
            target_date = all_dates[target_idx]
            if len(input_dates) == WINDOW_SIZE and target_date in tensor_cache:
                sequences.append((input_dates, target_date))
        return sequences

    train_seqs = _build_sequences(train_date_set)
    val_seqs = _build_sequences(val_date_set)

    logger.info(f"Training sequences: {len(train_seqs)}, Validation: {len(val_seqs)}")

    if not train_seqs:
        logger.error("No training sequences! Check data availability.")
        raise RuntimeError("No training sequences for EvolveGCN")

    # Initialize model
    model = EvolveGCNO(n_features, HIDDEN_DIM, n_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for input_dates, target_date in train_seqs:
            adj_seq = [tensor_cache[d][0] for d in input_dates]
            feat_seq = [tensor_cache[d][1] for d in input_dates]
            target_edges = tensor_cache[target_date][2]

            if not target_edges:
                continue

            node_embed = model(adj_seq, feat_seq)

            # Build edge indices and targets
            edge_idx = torch.tensor(
                list(target_edges.keys()), dtype=torch.long, device=device
            )
            edge_targets = torch.tensor(
                list(target_edges.values()), dtype=torch.float32, device=device
            )
            # Log-scale targets for stability
            edge_targets_log = torch.log1p(edge_targets)

            pred_weights = model.predict_edges(node_embed, edge_idx)
            pred_weights_log = torch.log1p(pred_weights)

            loss = F.mse_loss(pred_weights_log, edge_targets_log)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_seqs), 1)

        # Validation
        model.train(False)
        val_loss = 0.0
        with torch.no_grad():
            for input_dates, target_date in val_seqs:
                adj_seq = [tensor_cache[d][0] for d in input_dates]
                feat_seq = [tensor_cache[d][1] for d in input_dates]
                target_edges = tensor_cache[target_date][2]

                if not target_edges:
                    continue

                node_embed = model(adj_seq, feat_seq)
                edge_idx = torch.tensor(
                    list(target_edges.keys()), dtype=torch.long, device=device
                )
                edge_targets = torch.tensor(
                    list(target_edges.values()), dtype=torch.float32, device=device
                )

                pred = model.predict_edges(node_embed, edge_idx)
                loss = F.mse_loss(torch.log1p(pred), torch.log1p(edge_targets))
                val_loss += loss.item()

        val_loss /= max(len(val_seqs), 1)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            logger.info(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            ckpt_path = Path(checkpoint_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            save_path = ckpt_path / f"evolvegcn_h{horizon}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "node_index": node_index,
                "n_nodes": n_nodes,
                "n_features": n_features,
                "hidden_dim": HIDDEN_DIM,
                "horizon": horizon,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    save_path = Path(checkpoint_dir) / f"evolvegcn_h{horizon}.pt"
    logger.info(f"Best val_loss={best_val_loss:.6f}, saved to {save_path}")
    return save_path


# ──────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────

class EvolveGCNPredictor:
    """Loads trained EvolveGCN and predicts future graphs."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/evolvegcn",
        data_dir: str = "data/",
        device_str: str = "auto",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(
            "cuda" if device_str == "auto" and torch.cuda.is_available()
            else device_str if device_str != "auto" else "cpu"
        )
        self.models: dict[int, EvolveGCNO] = {}
        self.meta: dict[int, dict] = {}
        self._loader = SnapshotLoader(data_dir=data_dir)
        self.available = self._load_models()

    def _load_models(self) -> bool:
        """Load all available horizon-specific checkpoints."""
        if not self.checkpoint_dir.exists():
            logger.warning(f"EvolveGCN checkpoint dir not found: {self.checkpoint_dir}")
            return False

        loaded = False
        for h in [1, 4, 8, 12]:
            ckpt_path = self.checkpoint_dir / f"evolvegcn_h{h}.pt"
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    model = EvolveGCNO(
                        ckpt["n_features"],
                        ckpt["hidden_dim"],
                        ckpt["n_nodes"],
                    ).to(self.device)
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.train(False)
                    self.models[h] = model
                    self.meta[h] = ckpt
                    loaded = True
                    logger.info(f"EvolveGCN h={h} loaded (val_loss={ckpt['best_val_loss']:.6f})")
                except Exception as e:
                    logger.warning(f"Failed to load EvolveGCN h={h}: {e}")

        return loaded

    def predict(
        self,
        current_snapshot: GraphSnapshot,
        horizon: int,
    ) -> GraphSnapshot:
        """Predict the graph at t+horizon using EvolveGCN.

        Raises RuntimeError if no trained model or input history is available.
        """
        # Find closest available horizon model
        h = horizon
        if h not in self.models:
            available_h = sorted(self.models.keys())
            if not available_h:
                raise RuntimeError("No trained EvolveGCN models available")
            h = min(available_h, key=lambda x: abs(x - horizon))
            logger.info(f"EvolveGCN: no model for h={horizon}, using h={h}")

        model = self.models[h]
        meta = self.meta[h]
        node_index = meta["node_index"]
        n_nodes = meta["n_nodes"]

        # Build input sequence from recent snapshots
        all_dates = self._loader.dates
        if current_snapshot.date not in all_dates:
            raise RuntimeError(
                f"Current snapshot date {current_snapshot.date} is not available "
                "in the EvolveGCN loader history"
            )

        t_idx = all_dates.index(current_snapshot.date)
        start_idx = max(0, t_idx - WINDOW_SIZE + 1)
        input_dates = all_dates[start_idx:t_idx + 1]

        # Pad if we don't have enough history
        while len(input_dates) < WINDOW_SIZE:
            input_dates = [input_dates[0]] + input_dates

        # Convert to tensors
        adj_seq = []
        feat_seq = []
        for d in input_dates:
            try:
                snap = self._loader.load_single(d)
            except (KeyError, IndexError):
                snap = current_snapshot
            adj, feat, _ = _snap_to_tensors(snap, node_index, n_nodes, self.device)
            adj_seq.append(adj)
            feat_seq.append(feat)

        # Forward pass + batched edge weight prediction.
        # Each test snapshot may carry tens of thousands of edges; predicting
        # them one-by-one with .item() syncs serialises the GPU. Batch the
        # whole edge set into a single forward + a single host transfer.
        with torch.no_grad():
            node_embed = model(adj_seq, feat_seq)

            valid_edges: list[Edge] = []
            src_idx: list[int] = []
            tgt_idx: list[int] = []
            for e in current_snapshot.edges:
                if e.source in node_index and e.target in node_index:
                    valid_edges.append(e)
                    src_idx.append(node_index[e.source])
                    tgt_idx.append(node_index[e.target])

            if valid_edges:
                edge_idx = torch.stack(
                    [
                        torch.tensor(src_idx, dtype=torch.long, device=self.device),
                        torch.tensor(tgt_idx, dtype=torch.long, device=self.device),
                    ],
                    dim=1,
                )
                pred_weights = model.predict_edges(node_embed, edge_idx).cpu().numpy()
                edges_out = [
                    Edge(source=e.source, target=e.target, weight=float(pred_weights[i]))
                    for i, e in enumerate(valid_edges)
                ]
            else:
                edges_out = []

        return GraphSnapshot(
            date=current_snapshot.date,
            nodes=current_snapshot.nodes,
            edges=edges_out,
        )


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="m3_evolvegcn: EvolveGCN baseline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/evolvegcn")
    parser.add_argument("--train-split", default="2020-03~2024-06")
    parser.add_argument("--val-split", default="2024-07~2024-12")
    args = parser.parse_args()

    if args.mode == "train":
        for h in [1, 4, 8, 12]:
            logger.info(f"Training EvolveGCN for horizon h={h}")
            train_evolvegcn(
                data_dir=args.data_dir,
                train_split=args.train_split,
                val_split=args.val_split,
                horizon=h,
                checkpoint_dir=args.checkpoint_dir,
            )
    else:
        predictor = EvolveGCNPredictor(
            checkpoint_dir=args.checkpoint_dir,
            data_dir=args.data_dir,
        )
        print(f"EvolveGCN available: {predictor.available}")
        print(f"Loaded horizons: {list(predictor.models.keys())}")
