"""FastAPI server wrapping DeXposure-FM inference.

Run with: uvicorn lib.agent.serve:app --host 0.0.0.0 --port 8000
For testing: create_app(mock_mode=True)
"""
from __future__ import annotations
import time
import random
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


class ForecastRequest(BaseModel):
    date: str
    horizon: int = 4


class BatchForecastRequest(BaseModel):
    date: str
    horizons: list[int] = [1, 4, 8, 12]


class RunEpochRequest(BaseModel):
    date: str


# ---- Mock model for testing ----

MOCK_PROTOCOLS = ["aave-v3", "lido", "uniswap-v3", "compound", "maker", "curve", "wbtc-bridge"]


def _build_mock_graph(date: str):
    """Build a realistic mock GraphSnapshot for the given date."""
    from lib.agent.types import GraphSnapshot, NodeFeatures, Edge

    rng = random.Random(hash(date))
    categories = ["lending", "dex", "liquid-staking", "bridge", "lending", "dex", "bridge"]
    nodes = {p: NodeFeatures(
        log_size=round(rng.uniform(6.0, 10.0), 1),
        num_tokens=rng.randint(1, 5),
        max_share=round(rng.random(), 2),
        entropy=round(rng.uniform(0.1, 2.5), 1),
        category=categories[i % len(categories)],
    ) for i, p in enumerate(MOCK_PROTOCOLS)}
    edges = []
    for i, src in enumerate(MOCK_PROTOCOLS):
        for j, tgt in enumerate(MOCK_PROTOCOLS):
            if i != j and rng.random() > 0.4:
                edges.append(Edge(source=src, target=tgt, weight=round(rng.uniform(4.0, 9.0), 1)))
    return GraphSnapshot(date=date, nodes=nodes, edges=edges)


def _mock_forecast(date: str, horizon: int) -> dict[str, Any]:
    """Return a plausible mock prediction."""
    rng = random.Random(hash((date, horizon)))
    nodes = rng.sample(MOCK_PROTOCOLS, k=min(5, len(MOCK_PROTOCOLS)))
    edge_probs = {}
    edge_weights = {}
    weight_stds = {}
    for i, src in enumerate(nodes):
        for j, tgt in enumerate(nodes):
            if i != j and rng.random() > 0.4:
                pair_key = f"{src}|{tgt}"  # JSON keys must be strings
                edge_probs[pair_key] = round(rng.random(), 3)
                edge_weights[pair_key] = round(rng.uniform(3.0, 10.0), 2)
                weight_stds[pair_key] = round(rng.uniform(0.1, 2.0), 2)
    return {
        "edge_probs": edge_probs,
        "edge_weights": edge_weights,
        "weight_stds": weight_stds,
        "node_ids": nodes,
    }


def create_app(mock_mode: bool = False) -> FastAPI:
    app = FastAPI(title="DeXposure-Agent API", version="0.1.0")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "mock_mode": mock_mode,
            "models": ["dexposure-fm-mock"] if mock_mode else [],
            "timestamp": time.time(),
        }

    @app.get("/models")
    async def models():
        loaded = ["dexposure-fm-mock"] if mock_mode else []
        return {"loaded": loaded, "mock_mode": mock_mode}

    @app.post("/forecast")
    async def forecast(req: ForecastRequest):
        if mock_mode:
            return _mock_forecast(req.date, req.horizon)
        # TODO: Real model inference
        raise NotImplementedError("Real model inference not yet implemented")

    @app.post("/batch-forecast")
    async def batch_forecast(req: BatchForecastRequest):
        results = []
        for h in req.horizons:
            if mock_mode:
                results.append(_mock_forecast(req.date, h))
            else:
                raise NotImplementedError("Real model inference not yet implemented")
        return results

    @app.post("/run-epoch")
    async def run_epoch_endpoint(req: RunEpochRequest):
        """Run the full agent loop for one epoch."""
        from lib.agent.agent_loop import run_epoch
        from lib.agent.config import AgentConfig

        if mock_mode:
            # Build a mock graph from mock protocols
            graph = _build_mock_graph(req.date)

            # Inject mock forecast directly — avoids HTTP self-call
            async def mock_forecast_fn(g, horizon, cfg):
                return _mock_forecast(g.date, horizon)

            config = AgentConfig(mc_samples=10)  # Fewer samples for speed
            output = await run_epoch(
                graph=graph,
                baseline_history=[],
                config=config,
                forecast_fn=mock_forecast_fn,
            )
        else:
            raise NotImplementedError("Real data loading not yet implemented")

        return output.model_dump()

    return app


# Default app instance for uvicorn
app = create_app(mock_mode=False)
