---
description: Run the full six-axis DeXposure-Bench suite through DeXposure Claw.
---

# Run Full Suite

Use this skill when the user asks to run, reproduce, audit, or summarize the
complete DeXposure-Bench evaluation.

## Workflow

1. Check environment health with `dexposure-claw health`.
2. Confirm whether LLM-costing methods should run before starting LLM eval.
3. Use the DeXposure MCP tools or CLI to run the suite.
4. Save results under `results/agent_runs/<run_id>/`.
5. Generate an auditable report with benchmark IDs and method names expanded.

## Benchmark Names

- B1 Forecast: temporal risk-metric forecasting.
- B2 Warning: streaming early-warning quality.
- B3 Calibration: uncertainty calibration.
- B4 Stress: what-if scenario fidelity.
- B5 Decision: supervisory ticket quality.
- B6 Robustness: data-quality sensitivity.
