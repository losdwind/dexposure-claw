---
name: Stress Test Scenarios
description: >
  This skill should be used when the user asks to "run stress test",
  "what if protocol fails", "simulate contagion", "scenario analysis",
  "stress test the network", or wants to understand DeFi contagion risk.
version: 0.1.0
---

# Stress Test Scenarios

Stress testing simulates how shocks to individual protocols or sectors propagate through the credit exposure network. The DeXposure-Agent supports five built-in scenarios (S1-S5) covering common DeFi failure modes. See `references/scenario-library.md` for full scenario specifications.

## How Stress Testing Works

1. **Apply shock** — Select a scenario and identify the affected protocols. Apply the specified TVL loss (e.g., 100% for failure, 50% for de-peg).

2. **Compute contagion** — Use the DebtRank algorithm on the PredGraph to propagate losses from shocked nodes to their creditors and counterparties. Each round of propagation represents one transmission step.

3. **Rank losses** — Sort all protocols by their simulated loss amount. Protocols in the top quartile of losses are flagged as "at risk".

4. **Compute CVaR** — Compute the Conditional Value-at-Risk at the 95th percentile across all scenarios to get a worst-case aggregate loss estimate. CVaR is the expected loss given that we are in the worst 5% of outcomes.

## Scenario Selection

- For known failure modes (bridge hack, stablecoin de-peg), use the corresponding specific scenario (S2, S3).
- For unknown/general stress, run all five scenarios and compare aggregate CVaR.
- For regulatory reporting, S5 (correlated stress on top-10) is the standard scenario.

## Interpreting Results

Each stress test returns:

```json
{
  "scenario": "S2",
  "total_loss_fraction": 0.12,
  "at_risk_protocols": ["compound", "maker", "balancer"],
  "propagation_rounds": 3,
  "cvar_95": 0.18
}
```

**total_loss_fraction**: Fraction of network TVL lost. > 0.10 is considered systemic.

**at_risk_protocols**: Protocols absorbing > 1% of total TVL in losses.

**propagation_rounds**: Number of contagion hops before losses stabilize. More rounds = more interconnected failure path.

**cvar_95**: Expected loss in worst-case scenario tail. Primary number for capital planning.

## Running via API

Stress tests are embedded in the `/run-epoch` call automatically using the default scenario set (all five). To run a specific scenario interactively:

```
POST /stress-test
{
  "date": "2025-01-15",
  "scenario": "S2"
}
```

## Reference

See `references/scenario-library.md` for full S1-S5 specifications, shock parameters, and target protocol selection logic.
