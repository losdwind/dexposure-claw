---
name: Data Health Assessment
description: >
  This skill should be used when the user asks to "check data quality",
  "assess data health", "is the data fresh", "run data health gate",
  "check if data is safe", or mentions DeFi graph data quality concerns.
version: 0.1.0
---

# Data Health Assessment

Before any forecast or risk analysis can be trusted, the underlying DeFi graph data must pass a health gate. This skill describes what the DataHealth module checks, how to invoke it, and how to interpret results.

## What DataHealth Checks

The DataHealth module evaluates four dimensions of graph data quality:

**Freshness** — Verify that the latest snapshot timestamp falls within an acceptable window (default: <= 24 hours old). Stale data is the most common failure mode after on-chain indexer outages.

**Missingness** — Count missing node features (TVL, sector, protocol name) and missing edge weights. If more than 10% of nodes or 15% of edges have missing attributes, the score is penalized.

**Topology consistency** — Check that the graph is a valid DAG-like structure with no isolated components that should be connected. Detects cases where indexer bugs produce orphaned subgraphs.

**Discontinuity detection** — Compare edge-count and total-TVL against a rolling 7-day baseline. Sharp drops (>40%) indicate data pipeline failures rather than real market events.

## How to Run

Use the API directly via the call-api.py helper:

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" health
```

This calls `GET /health` on the DeXposure-Agent server. A full data health check is also embedded at the start of every `/run-epoch` call.

For a standalone structured JSON result:

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" health --output json
```

## Interpreting Results

The health endpoint returns a JSON object with the following key fields:

```json
{
  "score": 0.84,
  "safe_mode": false,
  "checks": {
    "freshness": true,
    "missingness": 0.03,
    "topology": true,
    "discontinuity": false
  },
  "timestamp": "2025-01-15T12:00:00Z"
}
```

**score >= 0.7**: Data is healthy. Full pipeline proceeds normally including interventions.

**score < 0.7**: Data enters SAFE_MODE. Forecasts and monitoring still run, but recommendation tickets that require SAFE_MODE=0 (Recommend-Reduce, Contingency) are suppressed. This prevents false alarms from triggering real-world actions on bad data.

## Common Causes of Safe Mode

| Cause | Symptom | Resolution |
|---|---|---|
| Indexer lag | Low freshness, stale timestamp | Wait for indexer sync or trigger manual re-pull |
| API rate limit | High missingness on specific protocols | Check API credentials, retry with backoff |
| Chain reorg | Discontinuity flag true | Wait 2-3 blocks, re-fetch snapshot |
| Protocol sunset | Sudden node disappearance | Manually verify protocol status |

## Safe Mode Behavior

When SAFE_MODE=1, the system:
- Still generates forecast predictions
- Still generates monitoring alerts
- Still runs stress test scenarios
- Does NOT generate Recommend-Reduce or Contingency tickets
- Appends a `suppressed: true` flag to AgentOutput

This ensures the agent remains informative during data outages without triggering irreversible actions.
