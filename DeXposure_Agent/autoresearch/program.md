# DeXposure-Agent Autoresearch

Autonomous hyperparameter optimization for the DeXposure-Agent systemic risk
monitoring pipeline.  Inspired by Karpathy's autoresearch pattern: propose a
change, run a fixed-budget experiment, measure the metric, keep or discard,
repeat forever.

## Context

DeXposure-Agent is a multi-module pipeline (data-health -> FM prediction ->
monitoring -> scenario engine -> decision) with 8 tunable hyperparameters
defined in `dexposure_agent/config.py` (AgentConfig).  The agent currently
uses hand-tuned defaults.  Your job is to find the configuration that
maximizes a composite score on the **validation split** (2024-07 ~ 2024-12).

## Setup

To set up a new autoresearch run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `tune-mar26`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files** for full context:
   - This file (`autoresearch/program.md`) -- your instructions.
   - `dexposure_agent/config.py` -- the parameter space you are optimizing.
   - `autoresearch/run_trial.py` -- the evaluation script (runs on GPU server).
   - `experiments/tune_agent.py` -- the original search grid for reference.
   - `CLAUDE.md` -- GPU server SOP (SSH, rsync, env vars).
4. **Verify GPU server is reachable**: `ssh gpu-server "nvidia-smi"`.
5. **Sync code to GPU server** (use the rsync commands from CLAUDE.md, plus):
   ```bash
   rsync -avz --delete -e "ssh -p 35417 -i ~/.ssh/github_ed25519" \
     DeXposure_Agent/autoresearch/ root@ssh9.vast.ai:/root/graph-dexposure/DeXposure_Agent/autoresearch/
   ```
6. **Verify the trial runner loads**:
   ```bash
   ssh gpu-server "cd /root/graph-dexposure/DeXposure_Agent && \
     DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
     python3 -c 'from autoresearch.run_trial import main; print(\"OK\")'"
   ```
7. **Initialize results.tsv**: Create `autoresearch/results.tsv` with header row.
8. **Run the baseline trial** (default config, no overrides):
   ```bash
   ssh gpu-server "cd /root/graph-dexposure/DeXposure_Agent && \
     DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
     python3 -u autoresearch/run_trial.py > /tmp/autoresearch_run.log 2>&1"
   ```
   Parse the output with:
   ```bash
   ssh gpu-server "grep '^composite_score:\|^rank_corr:\|^trend:\|^f1_warning:\|^ticket_prec:\|^elapsed' /tmp/autoresearch_run.log"
   ```
9. **Record baseline** in results.tsv and **confirm with user** before entering the loop.

## The Parameter Space

These are the parameters you optimize (from AgentConfig):

| Parameter       | Default | Search Range         | Role                           |
|-----------------|---------|----------------------|--------------------------------|
| pi_min          | 0.2     | [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] | Edge existence threshold for FM prediction |
| z_threshold     | 2.0     | [1.0, 1.5, 2.0, 2.5, 3.0]       | Alert z-score cutoff           |
| rolling_window  | 26      | [13, 26, 39, 52]                 | Baseline window (weeks)        |
| tau_data        | 0.7     | [0.4, 0.5, 0.6, 0.7, 0.8]       | Safe-mode threshold            |
| tau_conf        | 0.6     | [0.3, 0.4, 0.5, 0.6, 0.7]       | Confidence gate for decisions  |
| lambda_tail     | 0.5     | [0.0, 0.25, 0.5, 0.75, 1.0]     | CVaR tail weight               |
| mc_samples      | 50      | [10, 20, 50, 100]                | MC samples for uncertainty     |
| top_k           | 10      | [3, 5, 10, 15, 20]               | Attribution node count         |

**Total search space**: ~150,000 combinations.  Grid search is infeasible.
Use intelligent exploration: change 1-2 parameters at a time, keep what works,
build intuition from the results.

## The Metric

**composite_score** (higher is better): weighted average of
- 0.4 * rank_corr (B1 Spearman rho @ h=4)
- 0.2 * trend (B1 trend consistency)
- 0.2 * f1_warning (B2 early warning F1)
- 0.2 * ticket_prec (B5 ticket precision)

All components in [0, 1] range.

## The Experiment Loop

Each experiment modifies the config, runs on the GPU server, and takes ~3-10
minutes depending on benchmarks selected.

LOOP FOREVER:

1. **Propose a hypothesis**: Based on prior results, pick 1-2 parameters to change.
   Write your reasoning as the experiment description.

2. **Prepare the config**: Create a JSON config override:
   ```json
   {"pi_min": 0.3, "z_threshold": 1.5}
   ```

3. **Sync code** (only needed if you changed Python files, not for config-only changes):
   ```bash
   rsync -avz --delete -e "ssh -p 35417 -i ~/.ssh/github_ed25519" \
     DeXposure_Agent/autoresearch/ root@ssh9.vast.ai:/root/graph-dexposure/DeXposure_Agent/autoresearch/
   ```

4. **Run the trial on GPU**:
   ```bash
   ssh gpu-server "cd /root/graph-dexposure/DeXposure_Agent && \
     DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
     python3 -u autoresearch/run_trial.py \
       --config '{\"pi_min\": 0.3, \"z_threshold\": 1.5}' \
       > /tmp/autoresearch_run.log 2>&1"
   ```

5. **Read results**:
   ```bash
   ssh gpu-server "grep '^composite_score:\|^rank_corr:\|^trend:\|^f1_warning:\|^ticket_prec:\|^elapsed' /tmp/autoresearch_run.log"
   ```
   If grep returns empty, the trial crashed:
   ```bash
   ssh gpu-server "tail -50 /tmp/autoresearch_run.log"
   ```
   Diagnose and fix if it's a simple bug; otherwise skip and move on.

6. **Log results** to `autoresearch/results.tsv`.

7. **Keep or discard**:
   - If composite_score improved: KEEP. Update `best_config.json` with the new best.
   - If composite_score is equal or worse: DISCARD. Revert to previous best config.
   - Record the decision in results.tsv.

8. **Repeat**: Go to step 1. Think about what to try next based on what you've learned.

## Logging Results

Append each trial to `autoresearch/results.tsv` (tab-separated):

```
trial	composite	rank_corr	trend	f1_warning	ticket_prec	elapsed_s	status	config	description
```

Example:
```
trial	composite	rank_corr	trend	f1_warning	ticket_prec	elapsed_s	status	config	description
0	0.3210	0.5646	0.5862	0.0000	0.0000	180.5	baseline	{}	Default AgentConfig
1	0.3410	0.5800	0.6100	0.0000	0.0200	195.2	keep	{"pi_min": 0.3}	Increase pi_min: stricter new-edge threshold
2	0.3150	0.5500	0.5700	0.0000	0.0100	188.0	discard	{"z_threshold": 1.5}	Lower alert threshold: more sensitive alerts
```

## Research Strategy

Start with the parameters most likely to have the biggest impact:

1. **Phase 1 -- FM Prediction** (trials 1-5): Sweep `pi_min` first.
   This controls how aggressively the FM model adds new edges, which affects
   all downstream metrics.

2. **Phase 2 -- Alert Sensitivity** (trials 6-10): Sweep `z_threshold` and
   `rolling_window`. These control when alerts fire, affecting B2 and B5.

3. **Phase 3 -- Decision Quality** (trials 11-15): Sweep `tau_conf`,
   `lambda_tail`, `top_k`. These affect ticket generation (B5).

4. **Phase 4 -- Interactions** (trials 16+): Combine the best values found
   in phases 1-3. Try joint changes. Search around the current best config.

5. **Phase 5 -- Fine-tuning**: Small adjustments around the best config.
   Try intermediate values not in the original grid.

After ~20-30 trials you should have a strong config.  Keep going if there's
still signal (each trial improves or nearly improves).

## Output

When you've found a good config (or when manually stopped), save the best to:
- `autoresearch/best_config.json` -- the best AgentConfig overrides found
- `autoresearch/results.tsv` -- the full experiment log

## Rules

**What you CAN do:**
- Modify config parameters within the ranges above.
- Try values outside the predefined ranges if you have a good hypothesis.
- Change which benchmarks to run (--benchmarks flag) for faster iteration.
  B1 alone is fastest (~3 min).  Add B2 and B5 for full composite.
- Edit `run_trial.py` if you discover bugs or need adjustments.

**What you CANNOT do:**
- Modify the benchmark implementations (B1, B2, B5) -- they define the metric.
- Use the test split (2025-01 ~ 2025-08) -- that's for final evaluation only.
- Skip logging results.

**NEVER STOP**: Once the loop begins, do NOT pause to ask if you should
continue.  The user might be asleep.  Run autonomously until manually stopped.
If you run out of ideas, re-read results.tsv for patterns, try combinations
of winners, try values between grid points, or try more radical changes.
Each trial is ~5 minutes, so you can run ~12/hour, ~100 overnight.

## Timeout

Each trial should complete in under 15 minutes.  If it exceeds that, kill it:
```bash
ssh gpu-server "pkill -f 'run_trial.py'"
```
Log as "crash" and move on.
