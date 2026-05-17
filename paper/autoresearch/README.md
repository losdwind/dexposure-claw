# DeXposure-Agent Autoresearch

Autonomous hyperparameter optimization for DeXposure-Agent, adapted from
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) pattern.

## How it works

Instead of an AI agent modifying LLM training code, we have it modify
AgentConfig hyperparameters, evaluate on a validation split (2024-07 ~ 2024-12),
and keep or discard based on a composite metric.

## Files

- `program.md` -- instructions for the autonomous agent loop (read this first)
- `run_trial.py` -- GPU-side script that evaluates one config on the validation split
- `results.tsv` -- experiment log (tab-separated)
- `best_config.json` -- current best config found

## Quick start

```bash
# In a Claude Code session:
# 1. Read program.md and kick off the setup
# 2. The agent runs autonomously, modifying configs and evaluating

# Or manually run a single trial:
ssh gpu-server "cd /root/graph-dexposure/paper && \
  DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
  python3 -u autoresearch/run_trial.py \
    --config '{\"pi_min\": 0.3}' \
    --benchmarks b1_forecast,b5_decision"
```

## Metric

composite_score = 0.4*rank_corr + 0.2*trend + 0.2*f1_warning + 0.2*ticket_prec

Higher is better. All components in [0, 1].
