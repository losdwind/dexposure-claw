# DeXposure-Claw Paper Workspace

## Folder Structure

The installable Claw agent extension lives at repository root under `claw/`.
This `DeXposure_Agent/` directory contains the paper, deterministic pipeline,
benchmarks, and experiment artifacts.

    DeXposure_Agent/
    |
    |-- dexposure_agent/          # Python library: FM inference + deterministic pipeline
    |   |                         # Runs on GPU server. No LLM logic here.
    |   |-- serve.py              #   FastAPI server -- FM prediction HTTP API
    |   |-- fm_predictor.py       #   GraphPFN model loading + inference
    |   |-- agent_loop.py         #   Algorithm 1 deterministic pipeline
    |   |-- monitor.py            #   Network metrics N1..N5 + z-score alerts
    |   |-- scenario.py           #   Stress scenarios S1-S5
    |   |-- decision.py           #   Rule-based decision engine (benchmark baseline)
    |   |-- data_health.py        #   Data quality gate
    |   |-- data_loader.py        #   Snapshot data loading
    |   |-- pred_graph.py         #   Predicted graph builder + MC sampling
    |   |-- config.py             #   Hyperparameter configuration
    |   +-- types.py              #   Pydantic data models
    |
    |-- experiments/              # b1_forecast-b6_robustness benchmark implementations
    |   |                         # b1_forecast-b6_robustness run on GPU server; llm_eval runs LOCALLY.
    |   |-- b1_risk_forecasting.py - b6_robustness.py
    |   |-- llm_eval_b5.py        #   LLM evaluation pipeline (Layer 2+3, runs locally)
    |   |-- predict_helper.py     #   FM prediction router (m5_fm_rules=FM, m1_persistence_rules=persistence)
    |   |-- competitors/
    |   |   |-- llm_agent.py      #   m2_snapshot_llm: Pure LLM competitor (no FM, calls Claude API)
    |   |   +-- evolvegcn.py      #   m3_evolvegcn: EvolveGCN baseline
    |   |-- exp_logger.py         #   Structured experiment logging
    |   +-- tune_agent.py         #   Hyperparameter tuning
    |
    |-- scripts/                  # GPU server operations + local LLM eval
    |   |-- run_benchmarks_sequential.py  # Run full b1_forecast-b6_robustness suite (~40 min)
    |   |-- run_fm_rules_only.py        #   Run m5_fm_rules-only benchmarks (~8 min)
    |   |-- run_llm_eval.sh       #   Run LLM evaluation locally (Layer 2+3)
    |   |-- start_server.sh       #   Start FM API server
    |   +-- sync_and_serve.sh     #   Sync code from local + start server
    |
    |-- results/                  # Benchmark result JSONs (timestamped run directories)
    |-- sections/                 # LaTeX paper sections
    |-- tests/                    # Unit tests
    +-- autoresearch/             # Automated hyperparameter search

Key principles:
- dexposure_agent/ = GPU server only, no LLM SDK, pure FM + deterministic compute
- ../claw/ = installable DeXposure Claw package, MCP server, skills, and runtime adapters
- experiments/ b1_forecast-b6_robustness run on GPU server; llm_eval_b5.py runs LOCALLY
- FM API (serve.py) is the bridge: GPU server hosts it, scripts call it via SSH tunnel

## LLM Evaluation Pipeline (Layer 2 + Layer 3)

Evaluates LLM decision quality for the paper's core claim: FM+LLM > FM+Rules > Pure LLM.
Runs LOCALLY (not on GPU server). Calls FM API via SSH tunnel + Claude API directly.

### Methods Compared

| ID     | Name           | Prediction     | Decision   | Runs where |
|--------|----------------|----------------|------------|------------|
| m1_persistence_rules     | Persist+Rules  | G_{t+h}=G_t   | Rule engine| GPU (b5_decision)   |
| m5_fm_rules     | FM+Rules       | FM backbone    | Rule engine| GPU (b5_decision)   |
| m2_snapshot_llm     | NoFM+LLM      | None (raw text)| Claude     | LOCAL      |
| m6_fm_llm | FM+LLM        | FM backbone    | Claude     | LOCAL      |

### Running

    # 1. Ensure SSH tunnel + FM API running
    # `ssh gpu-server` already forwards 8080 (see ~/.ssh/config). For a no-shell tunnel:
    ssh -f -N gpu-server
    curl localhost:8080/health

    # 2. Export API key
    export ANTHROPIC_API_KEY=sk-ant-...

    # 3. Run (auto-sets up tunnel if needed)
    bash DeXposure_Agent/scripts/run_llm_eval.sh

    # Options:
    bash DeXposure_Agent/scripts/run_llm_eval.sh --method m6_fm_llm  # single method
    bash DeXposure_Agent/scripts/run_llm_eval.sh --resume          # resume from checkpoint
    bash DeXposure_Agent/scripts/run_llm_eval.sh --no-judge        # skip explanation quality
    bash DeXposure_Agent/scripts/run_llm_eval.sh --model claude-haiku-4-5  # cheaper model

### Metrics

Layer 2 (LLM reasoning quality):
- Grounding Score: fraction of cited values traceable to input data
- Consistency: Jaccard across 3 repeated runs (temperature=0)

Layer 3 (end-to-end decision quality):
- Ticket Precision: flagged protocols that were truly stressed (>20% weight drop)
- Audit Completeness: truly stressed protocols that were flagged
- Target Stability: Jaccard between consecutive weeks
- Severity Correlation: Spearman rho between recommended severity vs actual loss
- False Intervention Rate: high-severity tickets targeting stable protocols
- Explanation Quality: LLM-as-Judge score (1-5)

### Output

Results go to results/llm_eval_{timestamp}/:
- summary_C0-LLM.json, summary_C3.json: per-method aggregate metrics
- raw_C0-LLM.json, raw_C3.json: full prompts + LLM responses for audit
- comparison.json: side-by-side comparison table
- checkpoint_*.json: for resume support
- eval.log: detailed execution log

### Cost Estimate

~29 test weeks x 2 methods x 3 consistency runs x ~$0.03/call = ~$5-10 total
LLM-as-Judge adds ~$0.50 (uses Haiku)

## Experiment SOP

Standard operating procedure for running the b1_forecast-b6_robustness benchmark experiments.

## GPU Server (vast.ai)

    ssh gpu-server
    # Host: 103.177.249.208:43526, user root, key=~/.ssh/id_ed25519
    # Alias defined in ~/.ssh/config; LocalForward 8080 -> localhost:8080 for FM API
    # Hardware: RTX 4090 24 GB, driver 555.52.04, CUDA 12.5
    # OS: Ubuntu 24.04.4 LTS, kernel 6.5.0-41

Conda/venv (the bashrc auto-activates `/venv/main`):

    Python 3.12.13 (in /venv/main)
    torch 2.11.0+cu126  (already installed)
    dgl, torch_geometric, loguru, etc. -- NOT installed yet, must `pip install` before first run

Server paths (fresh instance — nothing pulled yet):

    /workspace/                            # 16 GB persistent volume (recommended for code)
    └── graph-dexposure/                   # ← clone or rsync your repo here
        ├── data/                          # Weekly graph snapshots (1.2GB JSON + meta_df.csv)
        ├── DeXposure_Agent/               # Agent code + experiments
        │   ├── dexposure_agent/           # Core modules
        │   ├── experiments/               # b1_forecast..b6_robustness benchmark implementations
        │   ├── scripts/                   # Runners
        │   └── results/run_YYYYMMDD_*/    # Timestamped results (never overwritten)
        ├── checkpoints/dexposure-fm-release/  # FM model weights (h1, h4, h8-h12)
        └── lib/                           # GraphPFN + LiMiX encoder libraries

## Before Running Experiments

### 0. First-time bootstrap on a fresh vast.ai instance

The image only ships `torch 2.11.0+cu126`. The rest must be installed once per instance:

    ssh gpu-server "source /venv/main/bin/activate && \
      pip install --quiet \
        dgl-cu126 -f https://data.dgl.ai/wheels/cu126/repo.html  # or matching wheel
      pip install --quiet \
        torch-geometric loguru pydantic tqdm scipy numpy pandas \
        scikit-learn optuna huggingface_hub fastapi uvicorn xgboost delu rtdl"

Also create the workspace directory:

    ssh gpu-server "mkdir -p /workspace/graph-dexposure"

### 1. Sync local code + data to server

    # code (no need for --delete on data; keep --delete only on code)
    rsync -avz --delete -e ssh \
      DeXposure_Agent/dexposure_agent/ gpu-server:/workspace/graph-dexposure/DeXposure_Agent/dexposure_agent/
    rsync -avz --delete -e ssh \
      DeXposure_Agent/experiments/ gpu-server:/workspace/graph-dexposure/DeXposure_Agent/experiments/
    rsync -avz -e ssh \
      DeXposure_Agent/scripts/ gpu-server:/workspace/graph-dexposure/DeXposure_Agent/scripts/

    # data (large, ~1.2 GB; only resync when meta_df.csv or week JSONs change)
    rsync -avz -e ssh \
      data/ gpu-server:/workspace/graph-dexposure/data/

    # checkpoints (~few hundred MB total)
    rsync -avz -e ssh \
      checkpoints/dexposure-fm-release/ gpu-server:/workspace/graph-dexposure/checkpoints/dexposure-fm-release/

### 2. Verify FM model loads

    ssh gpu-server 'cd /workspace/graph-dexposure/DeXposure_Agent && \
      source /venv/main/bin/activate && \
      DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 python -c "
    import sys; sys.path.insert(0, \".\"); sys.path.insert(0, \"..\")
    from dexposure_agent.fm_predictor import FMPredictor
    fm = FMPredictor()
    print(f\"FM available: {fm.available}\")
    "'

Must print "FM available: True".

### 3. Verify data

    ssh gpu-server "ls -lh /workspace/graph-dexposure/data/historical-network_week_*.json /workspace/graph-dexposure/data/meta_df.csv"

Need two JSON files (~1.1GB + 76MB) and meta_df.csv (128K).

## Running Experiments

### Full suite (m5_fm_rules + m1_persistence_rules, all benchmarks, ~40 min)

    ssh -f gpu-server "cd /workspace/graph-dexposure/DeXposure_Agent && \
      source /venv/main/bin/activate && \
      DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
      nohup python -u scripts/run_benchmarks_sequential.py > results/benchmark_stdout.log 2>&1 &"

### m5_fm_rules-only (skip m1_persistence_rules persistence, ~8 min)

    ssh -f gpu-server "cd /workspace/graph-dexposure/DeXposure_Agent && \
      source /venv/main/bin/activate && \
      DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
      nohup python -u scripts/run_fm_rules_only.py > results/fm_rules_only_stdout.log 2>&1 &"

### Monitor progress

    ssh gpu-server "tail -f /workspace/graph-dexposure/DeXposure_Agent/results/benchmark_stdout.log"
    # or for m5_fm_rules-only:
    ssh gpu-server "tail -f /workspace/graph-dexposure/DeXposure_Agent/results/fm_rules_only_stdout.log"

Progress shows: tqdm bars with ETA, [OVERALL x/11] tracking, per-benchmark timing.

### Check completion

    ssh gpu-server "grep 'ALL DONE' /workspace/graph-dexposure/DeXposure_Agent/results/benchmark_stdout.log"

## Pulling Results

Results go into timestamped directories (e.g. results/run_20260325_181310/).

    # Find latest run directory
    ssh gpu-server "ls -td /workspace/graph-dexposure/DeXposure_Agent/results/run_*/ | head -1"

    # Pull JSON results only (skip logs)
    mkdir -p DeXposure_Agent/results/run_LABEL
    rsync -avz --include="*.json" --exclude="*.log" -e ssh \
      gpu-server:/workspace/graph-dexposure/DeXposure_Agent/results/run_YYYYMMDD_*/ \
      DeXposure_Agent/results/run_LABEL/

## Benchmarks

| ID | Name | What it tests | Time |
|----|------|---------------|------|
| b1_forecast | Risk Forecasting | HHI/Gini/PageRank MAE + Spearman + trend at h={1,4,8,12} | ~3 min |
| b2_warning | Early Warning | Alert lead time, precision, recall for Terra/Luna, FTX, SVB | ~30 sec |
| b3_calibration | Uncertainty Calibration | ECE, PI coverage/width, CRPS from MC samples | ~9 min |
| b4_stress | Stress Test | Contagion loss MAE, overlap@10 for scenarios S1-S5 | ~3 min |
| b5_decision | Decision Quality | Ticket precision, risk reduction, target stability | ~11 min |
| b6_robustness | Robustness | b1_forecast metrics under 5 degradation regimes | ~2 min |

## Methods

| ID | Name | Prediction | Agent pipeline? |
|----|------|-----------|-----------------|
| m5_fm_rules | DeXposure-Agent | FM backbone (GraphPFN) | Yes |
| m1_persistence_rules | Persistence-Agent | G_{t+h} = G_t | Yes |
| m4_fm_only | DeXposure-FM | FM backbone | No (model only) |

## FM Prediction Strategy

The FM predictor uses a hybrid approach (fm_predictor.py):

1. KEEP all existing edges from G_t
2. REWEIGHT them: new_weight = old_weight + prob * FM_residual
3. ADD new edges only if FM existence probability >= pi_min (default 0.5)

This preserves network structure while letting FM adjust edge weights and predict new connections.

## Key Parameters

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| pi_min | fm_predictor.py FMPredictor.__init__ | 0.5 | Threshold for adding NEW edges |
| test_split | run scripts | 2025-01~2025-08 | 33 weeks test period |
| mc_samples | config.py AgentConfig | 50 | MC samples for b3_calibration uncertainty |
| z_threshold | config.py AgentConfig | 2.0 | Alert detection threshold |

## Environment Variables

Always set before running on GPU server:

    DGLBACKEND=pytorch
    DGL_DISABLE_GRAPHBOLT=1

## Results Directory Structure

    results/
    ├── run_20260325_persistence_baseline/   # m5_fm_rules=m1_persistence_rules (before FM integration)
    ├── run_20260325_fm_integrated/          # FM pi_min=0.2 (first FM run)
    ├── run_20260325_fm_pi05_c0only/         # FM pi_min=0.5 hard threshold
    ├── run_20260325_fm_hybrid/              # FM hybrid (keep edges + reweight)
    └── run_YYYYMMDD_HHMMSS/                # Future runs auto-timestamped
        ├── b1_forecast__m5_fm_rules.json                       # Per-benchmark-method results
        ├── b1_forecast__m1_persistence_rules.json
        ├── ...
        ├── run_summary.json                 # Overall pass/fail + timing
        ├── b1_forecast__m5_fm_rules__20260325_*.log             # Per-benchmark detailed log
        └── benchmark_run.log                # Master run log

## Known Issues

- meta_df.csv path: SnapshotLoader resolves from data_dir; if categories show "Unknown",
  check that meta_df.csv is in the same directory as the JSON files
- b3_calibration PI coverage was very low on the early runs (~1-2%); the conformal
  calibration fix (split conformal on the val split + data-driven MC sigma) brought
  it to ~0.91-0.93 in gpu_run_20260510_181509. If you ever see <0.85 again, check
  _calibrate_mc_sigma() in experiments/b3_uncertainty_calibration.py.
- b4_stress S2-S4 show perfect scores when meta_df.csv is missing (no categories = no nodes
  to shock for bridge/stablecoin/lending scenarios)
- SSH port forward warning "Address already in use" is cosmetic; ignore it

## Troubleshooting

Model not loading:

    # Check deps
    ssh gpu-server "source /venv/main/bin/activate && pip list | grep -iE 'dgl|torch|ogb|xgboost|delu|rtdl|optuna|huggingface'"

    # Check checkpoints exist
    ssh gpu-server "ls -lh /workspace/graph-dexposure/checkpoints/dexposure-fm-release/*.pt"

Process seems stuck:

    # Check if still running
    ssh gpu-server "pgrep -a python3"

    # Check GPU usage
    ssh gpu-server "nvidia-smi"

    # Kill and restart
    ssh gpu-server "pkill -f python3"
