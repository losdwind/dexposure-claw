---
name: Run Full Analysis Epoch
description: >
  This skill should be used when the user asks to "run full analysis",
  "analyze this epoch", "run the agent", "execute Algorithm 1",
  "run DeXposure-Agent", or wants end-to-end risk analysis for a date.
version: 0.1.0
---

# Run Full Analysis Epoch

A full analysis epoch executes the complete DeXposure-Agent pipeline (Algorithm 1) for a given date. This is the primary entry point for end-to-end risk analysis. It chains DataHealth -> Forecast -> Monitor -> Scenario -> Decision into a single API call.

## Algorithm 1 Walkthrough

### Step 1: DataHealth Gate

First, assess data quality for the target date (see Data Health skill). If score >= 0.7, proceed normally. If score < 0.7, set SAFE_MODE=1 and continue with reduced output.

### Step 2: Forecast

Run DeXposure-FM for all four horizons (h=1,4,8,12) to get the predicted graph ensemble. Build PredGraph using the default pi_min=0.5 threshold. The h=4 (1-month) PredGraph is the primary graph used for monitoring and scenarios.

### Step 3: Monitor

Compute M1-M7 metrics on PredGraph. Compare against rolling 30-day baselines. Generate alerts for any metric with |z| >= 2.0. Score alert confidence using Eq. 6. Filter to confidence >= 0.5.

### Step 4: Scenario

Run all five stress scenarios (S1-S5) on PredGraph. Compute contagion losses, identify at-risk protocols, calculate CVaR-95 across scenarios.

### Step 5: Decision

Generate action tickets from qualifying alerts and stress test results. Apply safe mode suppression if active. Sort by priority score descending.

## How to Invoke

```
python plugin/dexposure-agent/scripts/call-api.py run-epoch --date 2025-01-15
```

For full JSON output:

```
python plugin/dexposure-agent/scripts/call-api.py run-epoch --date 2025-01-15 --output json
```

API endpoint: `POST /run-epoch` with body `{"date": "YYYY-MM-DD"}`.

## AgentOutput JSON Structure

A complete epoch run returns the following structure:

```json
{
  "date": "2025-01-15",
  "data_health": {
    "score": 0.84,
    "safe_mode": false
  },
  "forecasts": {
    "h1": { "edge_probs": {...}, "edge_weights": {...}, "weight_stds": {...} },
    "h4": { "edge_probs": {...}, "edge_weights": {...}, "weight_stds": {...} },
    "h8": { "edge_probs": {...}, "edge_weights": {...}, "weight_stds": {...} },
    "h12": { "edge_probs": {...}, "edge_weights": {...}, "weight_stds": {...} }
  },
  "metrics": {
    "M1": 0.21, "M2": 0.38, "M3": 0.17,
    "M4": 0.08, "M5": 0.09, "M6": 0.54, "M7": 0.48
  },
  "alerts": [
    {
      "metric": "M1",
      "z_score": 2.8,
      "severity": "HIGH",
      "confidence": 0.79,
      "top_contributors": ["aave", "compound"]
    }
  ],
  "stress_tests": [
    {
      "scenario": "S2",
      "total_loss_fraction": 0.12,
      "cvar_95": 0.18,
      "at_risk_protocols": ["compound", "maker"]
    }
  ],
  "tickets": [
    {
      "action": "Recommend-Reduce",
      "protocol": "aave",
      "severity": "High",
      "priority_score": 3.41,
      "suppressed": false
    }
  ],
  "suppressed": false,
  "duration_seconds": 12.4
}
```

## Interpreting the Output

1. Check `data_health.safe_mode` first — if true, all High/Critical tickets are suppressed.
2. Scan `alerts` sorted by severity and confidence.
3. Review `stress_tests` for scenarios with `cvar_95 > 0.10`.
4. Act on `tickets` in priority order, respecting suppression flags.
5. Use the multi-horizon forecasts to distinguish short-term vs long-term risk trajectories.

## Running for Historical Dates

The run-epoch command accepts any historical date for backtesting and review:

```
python plugin/dexposure-agent/scripts/call-api.py run-epoch --date 2024-03-15 --output json
```

Historical epochs are read from the stored snapshot database and do not re-download on-chain data.
