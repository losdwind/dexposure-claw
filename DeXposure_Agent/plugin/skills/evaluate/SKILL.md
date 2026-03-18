---
name: Self-Evaluation
description: >
  This skill should be used when the user asks to "evaluate performance",
  "how accurate is the agent", "run benchmarks", "check calibration",
  "test the model", or wants to assess agent quality on their data.
version: 0.1.0
---

# Self-Evaluation

The DeXposure-Agent includes three benchmarks for assessing model quality. These can be run on historical data to verify that the agent performs as expected on a given deployment's DeFi protocol universe.

## Available Benchmarks

### B1: Risk Forecasting Accuracy

Tests whether the predicted graph (PredGraph at h=4) correctly anticipates which protocols will show elevated risk in the actual future snapshot.

**Metric:** AUROC for alert prediction — does a high predicted M1/M5 at time t correctly rank protocols that actually trigger alerts at time t+4?

**Passing threshold:** AUROC >= 0.70

**How to run:** Supply a date range with available ground truth:
```
POST /evaluate/b1
{
  "start_date": "2024-01-01",
  "end_date": "2024-06-30"
}
```

### B3: Uncertainty Calibration

Tests whether the model's weight_stds are calibrated — i.e., does a predicted uncertainty of sigma=0.2 actually correspond to 68% of observations falling within one sigma of the mean?

**Metric:** Expected Calibration Error (ECE). Lower is better.

**Passing threshold:** ECE <= 0.10

**How to run:**
```
POST /evaluate/b3
{
  "start_date": "2024-01-01",
  "end_date": "2024-06-30"
}
```

### B4: Stress-test Accuracy

Tests whether the stress-test module correctly identifies which protocols suffer the largest real losses when a major DeFi event occurs (using historical events as ground truth).

**Metric:** Spearman rank correlation between predicted loss ranking (from stress test) and actual loss ranking during historical stress events.

**Passing threshold:** Spearman rho >= 0.60

**How to run:**
```
POST /evaluate/b4
{
  "event_dates": ["2022-05-09", "2022-11-08", "2023-03-10"]
}
```

## Running on Custom Data

To evaluate on a new protocol universe or deployment environment:

1. Ensure historical snapshots are loaded into the server's snapshot database for at least 90 days prior to the evaluation window.
2. Call the relevant benchmark endpoint with the desired date range.
3. The server will replay the agent pipeline for each historical date and compare predictions against realized outcomes.

## Interpreting Results

Benchmark results are returned as:

```json
{
  "benchmark": "B1",
  "auroc": 0.74,
  "pass": true,
  "n_epochs_evaluated": 26,
  "worst_epoch": "2024-03-15",
  "notes": "Performance degraded in March 2024 — consider retraining on post-March data"
}
```

If a benchmark fails (falls below passing threshold):
- B1 fail: Model may need fine-tuning on local protocol universe; check if protocol taxonomy matches training data
- B3 fail: MC sampling count N may need to increase; check temperature scaling calibration
- B4 fail: Stress scenario parameters (shock fractions) may need recalibration for local market conditions

## Note on B2

Benchmark B2 (structural graph reconstruction) is a held-out test used during model development and is not exposed via the production API. Contact the DeXposure team for B2 evaluation if needed for academic benchmarking.
