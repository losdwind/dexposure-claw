---
name: DeXposure-FM Forecast
description: >
  This skill should be used when the user asks to "forecast network",
  "predict exposure graph", "run DeXposure-FM", "what will the network look like",
  or requests multi-horizon credit exposure predictions.
version: 0.1.0
---

# DeXposure-FM Forecast

DeXposure-FM is the foundation model at the core of the DeXposure-Agent. It takes a historical sequence of DeFi credit-exposure graphs and outputs probabilistic predictions of the future graph structure at multiple forecast horizons.

## Forecast Horizons

Four horizons are supported, measured in weekly snapshots:

| Horizon | Look-ahead | Use case |
|---|---|---|
| h=1 | 1 week | Short-term position review |
| h=4 | 1 month | Monthly risk committee reporting |
| h=8 | 2 months | Quarterly stress planning |
| h=12 | 3 months | Regulatory capital planning |

Specify the horizon in the API request. If omitted, defaults to h=4.

## What the Prediction Contains

Each forecast response includes three core outputs:

**edge_probs** — A dictionary mapping `(source_protocol, target_protocol)` pairs to a probability in [0,1] that an exposure edge exists at horizon h. An edge represents a meaningful credit exposure relationship.

**edge_weights** — Expected weight (log-TVL scaled) of each predicted edge, conditional on the edge existing. High weight = large exposure volume.

**weight_stds** — Standard deviation of the edge weight predictions, derived from Monte Carlo sampling over the model's stochastic latent space. Use this to assess prediction confidence per edge.

Example response structure:

```json
{
  "horizon": 4,
  "date": "2025-01-15",
  "edge_probs": {
    "aave_compound": 0.91,
    "curve_aave": 0.73
  },
  "edge_weights": {
    "aave_compound": 2.14,
    "curve_aave": 1.87
  },
  "weight_stds": {
    "aave_compound": 0.18,
    "curve_aave": 0.31
  }
}
```

## Building the Predicted Graph (PredGraph)

From the raw probabilistic output, a discrete PredGraph is constructed using a probability threshold pi_min (default: 0.5). Only edges with `edge_prob >= pi_min` are included in PredGraph.

This threshold controls the precision-recall trade-off:
- Higher pi_min: fewer edges, higher precision, may miss emerging risks
- Lower pi_min: more edges, higher recall, more false positives in stress tests

For risk monitoring purposes, the default pi_min=0.5 is recommended. For conservative stress-testing, lower to 0.35.

## Monte Carlo Uncertainty Quantification

The model uses MC sampling (default N=50 forward passes with dropout enabled) to produce the weight_stds. This reflects aleatoric uncertainty in the credit exposure magnitudes.

Edges with high weight_std relative to edge_weight (coefficient of variation > 0.3) should be treated with caution — the model is uncertain about the magnitude of that exposure relationship.

## How to Invoke

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" predict --date 2025-01-15 --horizon 4 --compact
```

For full JSON output:

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" predict --date 2025-01-15 --horizon 4 --compact --output json
```

The API endpoint is `POST /forecast` with body `{"date": "YYYY-MM-DD", "horizon": 4}`.
