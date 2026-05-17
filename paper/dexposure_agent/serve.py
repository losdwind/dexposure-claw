"""FastAPI server wrapping DeXposure-FM inference on GPU.

This server exposes the FM model as an HTTP API so that the LLM agent
can run on a separate (non-GPU) machine and call FM predictions remotely.

Endpoints:
    GET  /health           - Server status and model availability
    POST /forecast         - Load snapshot by date, run FM predict, return predicted graph
    POST /predict-graph    - Accept a full GraphSnapshot JSON, run FM predict, return predicted graph
    POST /batch-forecast   - Forecast across multiple horizons

Run on GPU server:
    cd /root/graph-dexposure/paper
    DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
        uvicorn dexposure_agent.serve:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_DISABLE_GRAPHBOLT", "1")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    date: str
    horizon: int = 4

class BatchForecastRequest(BaseModel):
    date: str
    horizons: list[int] = [1, 4, 8, 12]

class GraphNodeFeatures(BaseModel):
    log_size: float
    num_tokens: int
    max_share: float
    entropy: float
    category: str

class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float

class GraphSnapshotPayload(BaseModel):
    date: str
    nodes: dict[str, GraphNodeFeatures]
    edges: list[GraphEdge]

class PredictGraphRequest(BaseModel):
    graph: GraphSnapshotPayload
    horizon: int = 4

class PredictGraphResponse(BaseModel):
    date: str
    nodes: dict[str, GraphNodeFeatures]
    edges: list[GraphEdge]
    n_nodes: int
    n_edges: int
    horizon: int
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Global state (lazy-loaded)
# ---------------------------------------------------------------------------

_fm_predictor = None
_snapshot_loader = None

def _get_fm():
    global _fm_predictor
    if _fm_predictor is None:
        from dexposure_agent.fm_predictor import FMPredictor
        _fm_predictor = FMPredictor()
        logger.info("FMPredictor initialized: available=%s", _fm_predictor.available)
    return _fm_predictor

def _get_loader():
    global _snapshot_loader
    if _snapshot_loader is None:
        from dexposure_agent.data_loader import SnapshotLoader
        data_dir = os.environ.get(
            "DEXPOSURE_DATA_DIR",
            str(Path(_PROJECT_ROOT) / "data"),
        )
        _snapshot_loader = SnapshotLoader(data_dir=data_dir)
        logger.info("SnapshotLoader initialized: %d dates", len(_snapshot_loader.dates))
    return _snapshot_loader

def _to_internal_graph(payload: GraphSnapshotPayload):
    """Convert API payload to internal GraphSnapshot."""
    from dexposure_agent.types import GraphSnapshot, NodeFeatures, Edge
    nodes = {
        nid: NodeFeatures(
            log_size=nf.log_size, num_tokens=nf.num_tokens,
            max_share=nf.max_share, entropy=nf.entropy, category=nf.category,
        )
        for nid, nf in payload.nodes.items()
    }
    edges = [
        Edge(source=e.source, target=e.target, weight=e.weight)
        for e in payload.edges
    ]
    return GraphSnapshot(date=payload.date, nodes=nodes, edges=edges)

def _to_response(graph, horizon: int, elapsed_ms: float) -> PredictGraphResponse:
    """Convert internal GraphSnapshot to API response."""
    nodes = {
        nid: GraphNodeFeatures(
            log_size=nf.log_size, num_tokens=nf.num_tokens,
            max_share=nf.max_share, entropy=nf.entropy, category=nf.category,
        )
        for nid, nf in graph.nodes.items()
    }
    edges = [
        GraphEdge(source=e.source, target=e.target, weight=e.weight)
        for e in graph.edges
    ]
    return PredictGraphResponse(
        date=graph.date, nodes=nodes, edges=edges,
        n_nodes=len(nodes), n_edges=len(edges),
        horizon=horizon, elapsed_ms=round(elapsed_ms, 1),
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="DeXposure-FM API", version="0.2.0")

    @app.get("/health")
    async def health():
        fm = _get_fm()
        return {
            "status": "ok",
            "fm_available": fm.available,
            "timestamp": time.time(),
        }

    @app.post("/forecast", response_model=PredictGraphResponse)
    async def forecast(req: ForecastRequest):
        """Load snapshot by date from local data, run FM prediction."""
        fm = _get_fm()
        if not fm.available:
            raise HTTPException(503, "FM model not available")

        loader = _get_loader()
        try:
            snap = loader.load_single(req.date)
        except (KeyError, IndexError) as e:
            raise HTTPException(404, f"Snapshot not found for date {req.date}: {e}")

        t0 = time.time()
        predicted = fm.predict(snap, req.horizon)
        elapsed_ms = (time.time() - t0) * 1000

        logger.info(
            "forecast | date=%s h=%d | %d→%d nodes, %d→%d edges | %.0fms",
            req.date, req.horizon,
            len(snap.nodes), len(predicted.nodes),
            len(snap.edges), len(predicted.edges),
            elapsed_ms,
        )

        return _to_response(predicted, req.horizon, elapsed_ms)

    @app.post("/predict-graph", response_model=PredictGraphResponse)
    async def predict_graph(req: PredictGraphRequest):
        """Accept a full GraphSnapshot, run FM prediction, return predicted graph."""
        fm = _get_fm()
        if not fm.available:
            raise HTTPException(503, "FM model not available")

        snap = _to_internal_graph(req.graph)

        t0 = time.time()
        predicted = fm.predict(snap, req.horizon)
        elapsed_ms = (time.time() - t0) * 1000

        logger.info(
            "predict-graph | date=%s h=%d | %d→%d nodes, %d→%d edges | %.0fms",
            req.graph.date, req.horizon,
            len(snap.nodes), len(predicted.nodes),
            len(snap.edges), len(predicted.edges),
            elapsed_ms,
        )

        return _to_response(predicted, req.horizon, elapsed_ms)

    @app.post("/batch-forecast")
    async def batch_forecast(req: BatchForecastRequest):
        """Forecast across multiple horizons for a single date."""
        fm = _get_fm()
        if not fm.available:
            raise HTTPException(503, "FM model not available")

        loader = _get_loader()
        try:
            snap = loader.load_single(req.date)
        except (KeyError, IndexError) as e:
            raise HTTPException(404, f"Snapshot not found: {e}")

        results = []
        for h in req.horizons:
            t0 = time.time()
            predicted = fm.predict(snap, h)
            elapsed_ms = (time.time() - t0) * 1000
            results.append(_to_response(predicted, h, elapsed_ms))

        return results

    @app.get("/dates")
    async def list_dates():
        """List available snapshot dates."""
        loader = _get_loader()
        return {"dates": loader.dates, "count": len(loader.dates)}

    @app.get("/snapshot")
    async def get_snapshot(date: str):
        """Return raw snapshot (no FM prediction) for a given date.

        Used by the LLM agent for ground truth comparison and
        m2_snapshot_llm (pure LLM) prompts that need the current network state.
        """
        loader = _get_loader()
        try:
            snap = loader.load_single(date)
        except (KeyError, IndexError) as e:
            raise HTTPException(404, f"Snapshot not found for date {date}: {e}")

        nodes = {
            nid: GraphNodeFeatures(
                log_size=nf.log_size, num_tokens=nf.num_tokens,
                max_share=nf.max_share, entropy=nf.entropy, category=nf.category,
            )
            for nid, nf in snap.nodes.items()
        }
        edges = [
            GraphEdge(source=e.source, target=e.target, weight=e.weight)
            for e in snap.edges
        ]
        return {
            "date": snap.date,
            "nodes": {nid: nf.model_dump() for nid, nf in nodes.items()},
            "edges": [e.model_dump() for e in edges],
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        }

    return app


# Default app instance
app = create_app()
