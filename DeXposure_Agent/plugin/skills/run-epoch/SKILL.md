---
name: Run Full Analysis Epoch
description: >
  This skill should be used when the user asks to "run full analysis",
  "analyze this epoch", "run the agent", "execute Algorithm 1",
  "run DeXposure-Agent", or wants end-to-end risk analysis for a date.
version: 0.2.0
---

# Run Full Analysis Epoch

A full analysis epoch executes the complete DeXposure-Agent pipeline (Algorithm 1) for a given date. Claude Code orchestrates the pipeline: FM prediction (via HTTP API), metrics computation, stress testing, then LLM reasoning for decisions.

## Pipeline Steps

### Step 1: Check Server & Get Dates

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" health
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" dates
```

### Step 2: FM Prediction (compact summary)

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" predict --date YYYY-MM-DD --horizon 4 --compact
```

### Step 3: Network Risk Metrics

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" metrics --date YYYY-MM-DD --horizon 4
```

Metrics: M1 (max PageRank), M3 (HHI concentration), M4 (density), M6 (PageRank Gini), M7 (degree Gini)

### Step 4: Stress Test Scenarios

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" stress --date YYYY-MM-DD --horizon 4
```

Runs S1-S5: top protocol failure, bridge cluster, stablecoin de-peg, lending shock, correlated stress.

### Step 5: Multi-Horizon (optional, for --full)

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" batch --date YYYY-MM-DD
# Then metrics for each horizon:
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" metrics --date YYYY-MM-DD --horizon 1
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" metrics --date YYYY-MM-DD --horizon 8
python "${CLAUDE_PLUGIN_ROOT}/scripts/call-api.py" metrics --date YYYY-MM-DD --horizon 12
```

### Step 6: LLM Reasoning (Claude Code does this)

After collecting all data, Claude Code (you) performs the reasoning:
- Compare metrics to known crisis signatures (Terra/Luna, FTX, SVB)
- Assess which stress scenarios pose systemic risk (>15% loss)
- Generate decision tickets: Monitor / Investigate / Recommend-Reduce / Contingency
- Provide auditable rationale citing specific numbers

## Interpreting Metrics

| Metric | Meaning | Alert if |
|--------|---------|----------|
| M1 (max PageRank) | Systemic importance of top protocol | > 0.05 |
| M3 (HHI) | Exposure concentration | > 0.1 |
| M4 (density) | Network connectivity | Sudden drop |
| M6 (PageRank Gini) | PageRank inequality | > 0.8 |
| M7 (degree Gini) | Degree inequality | > 0.95 |

## Stress Severity Thresholds

| System Loss | Assessment |
|-------------|------------|
| < 5% | Contained |
| 5-15% | Significant |
| > 15% | Systemic — escalate |
