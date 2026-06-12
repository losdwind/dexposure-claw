"""DeXposure-FM model predictor for the agent pipeline.

Loads fine-tuned GraphPFN checkpoints and generates graph predictions
(edge existence probabilities + edge weight predictions) for use in
b1_forecast..b6_robustness benchmarks.

Checkpoints:
    dexposure-fm-h1.pt   -> horizon 1 week
    dexposure-fm-h4.pt   -> horizon 4 weeks
    dexposure-fm-h8-h12.pt -> horizons 8 and 12 weeks

Each checkpoint contains a full state_dict with keys:
    encoder.tfm.*       -> GraphPFN transformer encoder
    link_scorer.*       -> edge existence + weight prediction heads
    node_head.*         -> node TVL change prediction head
    scenario_head.*     -> scenario classification head
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Add project root for lib imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_DISABLE_GRAPHBOLT", "1")

from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures

logger = logging.getLogger(__name__)

# Checkpoint file mapping
CHECKPOINT_MAP = {
    1: "dexposure-fm-h1.pt",
    4: "dexposure-fm-h4.pt",
    8: "dexposure-fm-h8-h12.pt",
    12: "dexposure-fm-h8-h12.pt",
}

# Default checkpoint directory (override with DEXPOSURE_FM_CKPT_DIR, e.g. to
# point the whole pipeline at the pre-2022 retrained checkpoints)
DEFAULT_CKPT_DIR = os.environ.get(
    "DEXPOSURE_FM_CKPT_DIR",
    os.path.join(_PROJECT_ROOT, "checkpoints", "dexposure-fm-release"),
)


class FMPredictor:
    """Loads DeXposure-FM and generates graph predictions.

    Usage:
        predictor = FMPredictor(checkpoint_dir="checkpoints/dexposure-fm-release/")
        predicted_graph = predictor.predict(current_graph, horizon=4)
    """

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CKPT_DIR,
        device: str | None = None,
        pi_min: float = 0.5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pi_min = pi_min

        # Lazy-loaded models per horizon
        self._models: dict[int, nn.Module] = {}
        self._link_scorers: dict[int, nn.Module] = {}
        self._node_heads: dict[int, nn.Module] = {}

        # Check availability
        self._available = self.checkpoint_dir.exists()
        if not self._available:
            logger.warning("FM checkpoint dir not found: %s", self.checkpoint_dir)

    @property
    def available(self) -> bool:
        return self._available

    def _get_horizon_key(self, horizon: int) -> int:
        """Map horizon to checkpoint key."""
        if horizon <= 1:
            return 1
        elif horizon <= 4:
            return 4
        else:
            return 8  # h8-h12 checkpoint covers both

    def _load_model(self, horizon: int):
        """Load encoder + link_scorer + node_head for a given horizon."""
        h_key = self._get_horizon_key(horizon)
        if h_key in self._models:
            return

        ckpt_name = CHECKPOINT_MAP.get(h_key)
        if not ckpt_name:
            raise ValueError(f"No checkpoint for horizon {horizon}")

        ckpt_path = self.checkpoint_dir / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logger.info("Loading FM checkpoint: %s (horizon=%d)", ckpt_name, horizon)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt["model"]

        # Import model classes
        from lib.graphpfn.model import GraphPFN

        # 1. Load encoder
        encoder = GraphPFN(edge_head=False)
        encoder_params = {
            k.replace("encoder.tfm.", ""): v
            for k, v in sd.items()
            if k.startswith("encoder.tfm.")
        }
        encoder.load_state_dict(encoder_params, strict=False)
        encoder = encoder.to(self.device)
        encoder.requires_grad_(False)

        # 2. Determine embed_dim and hidden_dim from checkpoint weights
        # link_scorer.exist_head.0.weight shape is [hidden_dim, 4*embed_dim]
        exist_w = sd.get("link_scorer.exist_head.0.weight")
        if exist_w is not None:
            hidden_dim = exist_w.shape[0]
            embed_dim = exist_w.shape[1] // 4
        else:
            embed_dim = 192
            hidden_dim = 256

        # 3. Build and load LinkScorer
        link_scorer = _LinkScorer(embed_dim, hidden_dim)
        ls_params = {
            k.replace("link_scorer.", ""): v
            for k, v in sd.items()
            if k.startswith("link_scorer.")
        }
        link_scorer.load_state_dict(ls_params, strict=True)
        link_scorer = link_scorer.to(self.device)
        link_scorer.requires_grad_(False)

        # 4. Build NodeHead dynamically from checkpoint structure
        nh_params = {
            k.replace("node_head.", ""): v
            for k, v in sd.items()
            if k.startswith("node_head.")
        }
        node_head = _build_node_head_from_params(nh_params)
        node_head = node_head.to(self.device)
        node_head.requires_grad_(False)

        self._models[h_key] = encoder
        self._link_scorers[h_key] = link_scorer
        self._node_heads[h_key] = node_head

        n_params = sum(p.numel() for p in encoder.parameters())
        logger.info(
            "FM h=%d loaded: encoder=%d params, embed_dim=%d, device=%s",
            horizon, n_params, embed_dim, self.device,
        )

    def predict(
        self,
        graph: GraphSnapshot,
        horizon: int,
    ) -> GraphSnapshot:
        """Generate a predicted GraphSnapshot using the FM model.

        Args:
            graph: Current observation GraphSnapshot.
            horizon: Forecast horizon in weeks.

        Returns:
            Predicted GraphSnapshot with FM-generated edges and weights.
        """
        if not self._available:
            logger.warning("FM not available, falling back to persistence")
            return graph

        h_key = self._get_horizon_key(horizon)
        self._load_model(horizon)

        encoder = self._models[h_key]
        link_scorer = self._link_scorers[h_key]

        # Convert GraphSnapshot to tensor format
        node_ids = list(graph.nodes.keys())
        n = len(node_ids)
        if n < 2:
            return graph

        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Build node features tensor [N, F]
        features = self._build_features(graph, node_ids)

        # Build edge index for DGL graph
        src_list, dst_list, weight_list = [], [], []
        for edge in graph.edges:
            if edge.source in node_to_idx and edge.target in node_to_idx:
                src_list.append(node_to_idx[edge.source])
                dst_list.append(node_to_idx[edge.target])
                weight_list.append(edge.weight)

        if not src_list:
            return graph

        import dgl
        g = dgl.graph(
            (torch.tensor(src_list), torch.tensor(dst_list)),
            num_nodes=n,
        )

        # Encode with GraphPFN
        with torch.no_grad():
            h = self._encode(encoder, g, features)

            # Generate candidate pairs (existing edges + 2-hop neighbors)
            src_idx, dst_idx = self._get_candidate_pairs(
                src_list, dst_list, n, max_pairs=min(n * n, 50000)
            )

            src_t = torch.tensor(src_idx, dtype=torch.long, device=self.device)
            dst_t = torch.tensor(dst_idx, dtype=torch.long, device=self.device)

            logits, w_pred = link_scorer(h, src_t, dst_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            weights = w_pred.cpu().numpy()

        # Build predicted graph: FM-driven strategy
        # Follows DeXposure-FM paper Eq.(9): ŵ_{τ+h} = w̃_τ + r̂_{pq,τ,h}
        # Use existence probability to decide IF an edge exists (binary),
        # and the weight residual to decide HOW MUCH it weighs (regression).
        # These are separate prediction heads trained with separate losses.
        edges: list[Edge] = []
        included_nodes: set[str] = set()

        # Index FM predictions by (src_idx, dst_idx)
        fm_pred: dict[tuple[int, int], tuple[float, float]] = {}
        for i in range(len(src_idx)):
            fm_pred[(src_idx[i], dst_idx[i])] = (float(probs[i]), float(weights[i]))

        # Current edge lookup
        current_edges: dict[tuple[str, str], float] = {}
        for edge in graph.edges:
            current_edges[(edge.source, edge.target)] = edge.weight

        # Step 1: Existing edges — use FM existence prob RANKING to remove
        #         bottom fraction, use full residual for weight adjustment.
        #
        # FM existence probs are not calibrated (trained with 5:1 neg sampling),
        # but RANKING is excellent (AUROC=0.995). So we remove the bottom
        # `removal_pct` of edges by probability — the ones FM is most confident
        # will disappear.
        removal_pct = 0.05  # remove bottom 5% (tunable on validation set)

        # Collect probs for existing edges to compute percentile threshold
        existing_probs: list[float] = []
        for edge in graph.edges:
            s_id, t_id = edge.source, edge.target
            if s_id in node_to_idx and t_id in node_to_idx:
                si, ti = node_to_idx[s_id], node_to_idx[t_id]
                if (si, ti) in fm_pred:
                    existing_probs.append(fm_pred[(si, ti)][0])

        if existing_probs:
            import numpy as _np
            removal_threshold = float(_np.percentile(existing_probs, removal_pct * 100))
        else:
            removal_threshold = 0.0

        n_removed = 0
        for edge in graph.edges:
            s_id, t_id = edge.source, edge.target
            if s_id not in node_to_idx or t_id not in node_to_idx:
                edges.append(edge)
                included_nodes.add(s_id)
                included_nodes.add(t_id)
                continue

            si, ti = node_to_idx[s_id], node_to_idx[t_id]
            if (si, ti) in fm_pred:
                prob, w_residual = fm_pred[(si, ti)]
                # Existence decision: remove if in bottom removal_pct by prob
                if prob < removal_threshold:
                    n_removed += 1
                    continue
                # Weight update: baseline + full residual (Eq.9 from FM paper)
                new_w = max(edge.weight + w_residual, 0.0)
                if new_w > 0:
                    edges.append(Edge(source=s_id, target=t_id, weight=new_w))
                else:
                    edges.append(Edge(source=s_id, target=t_id, weight=edge.weight * 0.01))
            else:
                # No FM prediction for this edge — keep as-is
                edges.append(edge)
            included_nodes.add(s_id)
            included_nodes.add(t_id)

        # Step 2: Add NEW edges with high FM confidence
        n_new = 0
        for (si, ti), (prob, w_residual) in fm_pred.items():
            s_id = node_ids[si]
            t_id = node_ids[ti]
            if (s_id, t_id) not in current_edges and prob >= self.pi_min:
                w = max(w_residual, 0.01)  # new edge, no baseline
                edges.append(Edge(source=s_id, target=t_id, weight=w))
                included_nodes.add(s_id)
                included_nodes.add(t_id)
                n_new += 1

        # Build nodes dict — include all nodes that have edges
        nodes: dict[str, NodeFeatures] = {
            nid: graph.nodes[nid]
            for nid in node_ids
            if nid in included_nodes and nid in graph.nodes
        }

        logger.info(
            "FM predict h=%d: %d nodes, %d edges (kept=%d, removed=%d, new=%d, candidates=%d, pi_min=%.2f)",
            horizon, len(nodes), len(edges), len(edges) - n_new, n_removed, n_new,
            len(src_idx), self.pi_min,
        )

        return GraphSnapshot(date=graph.date, nodes=nodes, edges=edges)

    def _build_features(
        self, graph: GraphSnapshot, node_ids: list[str]
    ) -> torch.Tensor:
        """Build feature tensor [N, F] from GraphSnapshot node features."""
        features = []
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Compute degree stats first
        in_deg = [0.0] * len(node_ids)
        out_deg = [0.0] * len(node_ids)
        for edge in graph.edges:
            if edge.source in node_to_idx:
                out_deg[node_to_idx[edge.source]] += 1
            if edge.target in node_to_idx:
                in_deg[node_to_idx[edge.target]] += 1
        max_deg = max(max(in_deg, default=1), max(out_deg, default=1), 1)

        for i, nid in enumerate(node_ids):
            nf = graph.nodes[nid]
            features.append([
                nf.log_size,
                float(nf.num_tokens),
                nf.max_share,
                nf.entropy,
                float(hash(nf.category) % 15),  # category hash
                0.0,  # tvl_change placeholder
                in_deg[i] / max_deg,
                out_deg[i] / max_deg,
            ])

        return torch.tensor(features, dtype=torch.float32)

    def _encode(
        self, encoder, graph, features: torch.Tensor
    ) -> torch.Tensor:
        """Encode graph using GraphPFN with proper setup."""
        from lib.graphpfn.model import GraphPFNLayerWrapper
        from lib.limix.model.layer import MultiheadAttention as MHA

        num_nodes = graph.num_nodes()
        device = self.device

        train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        train_mask[-1] = False  # At least 1 test node required

        n_train = int(train_mask.sum().item())
        y_train = torch.zeros(n_train, device=device)

        features = features.to(device)
        graph = graph.to(device)

        tfm_features = torch.cat([features[train_mask], features[~train_mask]], dim=0)

        for module in encoder.modules():
            if isinstance(module, GraphPFNLayerWrapper):
                module.train_mask = train_mask
                module.graph = graph
            if isinstance(module, MHA):
                module.batched = False

        # Use appropriate SDPA backend
        if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
            if device.type == "cpu":
                backends = [torch.nn.attention.SDPBackend.MATH]
            else:
                backends = [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                ]
            with torch.nn.attention.sdpa_kernel(backends):
                out = encoder.tfm.forward(
                    x=tfm_features.unsqueeze(0),
                    y=y_train.unsqueeze(0),
                    eval_pos=n_train,
                    task_type="reg",
                    checkpointing=False,
                )
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                out = encoder.tfm.forward(
                    x=tfm_features.unsqueeze(0),
                    y=y_train.unsqueeze(0),
                    eval_pos=n_train,
                    task_type="reg",
                    checkpointing=False,
                )

        inv_order = torch.argsort((~train_mask).float(), stable=True)
        order = torch.argsort(inv_order, stable=True)

        if "encoder_out_full" in out:
            feature_embeds = out["encoder_out_full"][:, :, :-1, :]
            node_embeds = feature_embeds.mean(dim=2).squeeze(0)[order]
        else:
            node_embeds = out["encoder_embed"].squeeze(0)[order]

        return node_embeds

    def _get_candidate_pairs(
        self,
        src_list: list[int],
        dst_list: list[int],
        n: int,
        max_pairs: int = 50000,
    ) -> tuple[list[int], list[int]]:
        """Get candidate node pairs for prediction.

        Includes all existing edges + 2-hop neighbors + random negatives.
        """
        pairs = set()
        # Include all existing edges
        for s, d in zip(src_list, dst_list):
            pairs.add((s, d))

        # Add 2-hop neighbors
        adj: dict[int, set[int]] = {}
        for s, d in zip(src_list, dst_list):
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)

        for node in list(adj.keys()):
            for neighbor in list(adj.get(node, [])):
                for hop2 in adj.get(neighbor, []):
                    if hop2 != node:
                        pairs.add((node, hop2))
                        if len(pairs) >= max_pairs:
                            break
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break

        # If still under budget, add random samples
        rng = np.random.default_rng(42)
        attempts = 0
        while len(pairs) < min(max_pairs, n * (n - 1)) and attempts < max_pairs * 2:
            s = int(rng.integers(0, n))
            d = int(rng.integers(0, n))
            if s != d:
                pairs.add((s, d))
            attempts += 1

        pair_list = list(pairs)
        return [p[0] for p in pair_list], [p[1] for p in pair_list]


def _build_node_head_from_params(params: dict) -> nn.Module:
    """Build a NodeHead nn.Sequential that matches the checkpoint param keys.

    h1 checkpoint uses: net.0 -> ReLU -> Dropout -> net.3 -> ReLU -> net.5
    h4/h8 checkpoints use: net.0 -> ReLU -> net.2
    We detect which variant by checking which keys exist.
    """
    # Find all linear layer indices
    weight_keys = sorted(k for k in params if k.endswith(".weight"))
    layers = []
    for wk in weight_keys:
        idx = wk.split(".")[1]  # e.g. "0", "2", "3", "5"
        w = params[wk]
        b = params.get(wk.replace(".weight", ".bias"))
        linear = nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data = w
        if b is not None:
            linear.bias.data = b
        # Add ReLU + Dropout between layers (except after last)
        if layers:
            layers.append(nn.ReLU())
            if int(idx) - int(weight_keys[len(layers)//2 - 1].split(".")[1]) > 2:
                layers.append(nn.Dropout(0.1))
        layers.append(linear)

    module = nn.Sequential(*layers)
    # Wrap in a module that handles the inflow/outflow aggregation
    return _NodeHeadWrapper(module)


# ---- Model component replicas (must match checkpoint structure) ----


class _LinkScorer(nn.Module):
    """Replica of LinkScorer from run_full_experiment.py."""

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


class _NodeHeadWrapper(nn.Module):
    """Wraps a sequential net with inflow/outflow neighbor aggregation."""

    def __init__(self, net: nn.Sequential):
        super().__init__()
        self.net = net

    def forward(self, h, edge_src=None, edge_dst=None, edge_weight=None):
        # Check if the net expects 3*embed_dim (neighbor-aware) or just embed_dim
        first_layer = self.net[0]
        expects_neighbors = first_layer.in_features > h.shape[-1]

        if expects_neighbors and edge_src is not None and edge_dst is not None and edge_weight is not None:
            w = edge_weight.float()
            inflow = torch.zeros_like(h)
            inflow.index_add_(0, edge_dst, h[edge_src] * w.unsqueeze(-1))
            outflow = torch.zeros_like(h)
            outflow.index_add_(0, edge_src, h[edge_dst] * w.unsqueeze(-1))
            x = torch.cat([h, inflow, outflow], dim=-1)
        elif expects_neighbors:
            x = torch.cat([h, torch.zeros_like(h), torch.zeros_like(h)], dim=-1)
        else:
            x = h

        return self.net(x).squeeze(-1)
