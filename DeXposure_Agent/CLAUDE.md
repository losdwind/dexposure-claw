# DeXposure-Agent Experiment SOP

Standard operating procedure for running the B1-B6 benchmark experiments.

## GPU Server

    ssh gpu-server
    # Config: ssh9.vast.ai:35417, key=~/.ssh/github_ed25519
    # RTX 4090 24GB, Python 3.12, PyTorch 2.6.0+cu124, PyG 2.7.0, DGL 2.5.0

Server paths:

    /root/graph-dexposure/
    ├── DeXposure/data/                    # Weekly graph snapshots (1.2GB JSON + meta_df.csv)
    ├── DeXposure_Agent/                   # Agent code + experiments
    │   ├── dexposure_agent/               # Core modules
    │   ├── experiments/                   # B1-B6 benchmark implementations
    │   ├── scripts/                       # Runners
    │   └── results/run_YYYYMMDD_*/        # Timestamped results (never overwritten)
    ├── checkpoints/dexposure-fm-release/  # FM model weights (h1, h4, h8-h12)
    └── lib/                               # GraphPFN + LiMiX encoder libraries

## Before Running Experiments

### 1. Sync local code to server

    rsync -avz --delete -e "ssh -p 35417 -i ~/.ssh/github_ed25519" \
      DeXposure_Agent/dexposure_agent/ root@ssh9.vast.ai:/root/graph-dexposure/DeXposure_Agent/dexposure_agent/
    rsync -avz --delete -e "ssh -p 35417 -i ~/.ssh/github_ed25519" \
      DeXposure_Agent/experiments/ root@ssh9.vast.ai:/root/graph-dexposure/DeXposure_Agent/experiments/
    rsync -avz -e "ssh -p 35417 -i ~/.ssh/github_ed25519" \
      DeXposure_Agent/scripts/ root@ssh9.vast.ai:/root/graph-dexposure/DeXposure_Agent/scripts/

### 2. Verify FM model loads

    ssh gpu-server 'cd /root/graph-dexposure/DeXposure_Agent && DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 python3 -c "
    import sys; sys.path.insert(0, \".\"); sys.path.insert(0, \"..\")
    from dexposure_agent.fm_predictor import FMPredictor
    fm = FMPredictor()
    print(f\"FM available: {fm.available}\")
    "'

Must print "FM available: True".

### 3. Verify data

    ssh gpu-server "ls -lh /root/graph-dexposure/DeXposure/data/historical-network_week_*.json /root/graph-dexposure/DeXposure/data/meta_df.csv"

Need two JSON files (~1.1GB + 76MB) and meta_df.csv (128K).

## Running Experiments

### Full suite (C0 + C2, all benchmarks, ~40 min)

    ssh -f gpu-server "cd /root/graph-dexposure/DeXposure_Agent && \
      DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
      nohup python3 -u scripts/run_benchmarks_sequential.py > results/benchmark_stdout.log 2>&1 &"

### C0-only (skip C2 persistence, ~8 min)

    ssh -f gpu-server "cd /root/graph-dexposure/DeXposure_Agent && \
      DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
      nohup python3 -u scripts/run_c0_only.py > results/c0_stdout.log 2>&1 &"

### Monitor progress

    ssh gpu-server "tail -f /root/graph-dexposure/DeXposure_Agent/results/benchmark_stdout.log"
    # or for C0-only:
    ssh gpu-server "tail -f /root/graph-dexposure/DeXposure_Agent/results/c0_stdout.log"

Progress shows: tqdm bars with ETA, [OVERALL x/11] tracking, per-benchmark timing.

### Check completion

    ssh gpu-server "grep 'ALL DONE' /root/graph-dexposure/DeXposure_Agent/results/benchmark_stdout.log"

## Pulling Results

Results go into timestamped directories (e.g. results/run_20260325_181310/).

    # Find latest run directory
    ssh gpu-server "ls -td /root/graph-dexposure/DeXposure_Agent/results/run_*/ | head -1"

    # Pull JSON results only (skip logs)
    mkdir -p DeXposure_Agent/results/run_LABEL
    rsync -avz --include="*.json" --exclude="*.log" \
      -e "ssh -p 35417 -i ~/.ssh/github_ed25519" \
      root@ssh9.vast.ai:/root/graph-dexposure/DeXposure_Agent/results/run_YYYYMMDD_*/ \
      DeXposure_Agent/results/run_LABEL/

## Benchmarks

| ID | Name | What it tests | Time |
|----|------|---------------|------|
| B1 | Risk Forecasting | HHI/Gini/PageRank MAE + Spearman + trend at h={1,4,8,12} | ~3 min |
| B2 | Early Warning | Alert lead time, precision, recall for Terra/Luna, FTX, SVB | ~30 sec |
| B3 | Uncertainty Calibration | ECE, PI coverage/width, CRPS from MC samples | ~9 min |
| B4 | Stress Test | Contagion loss MAE, overlap@10 for scenarios S1-S5 | ~3 min |
| B5 | Decision Quality | Ticket precision, risk reduction, target stability | ~11 min |
| B6 | Robustness | B1 metrics under 5 degradation regimes | ~2 min |

## Methods

| ID | Name | Prediction | Agent pipeline? |
|----|------|-----------|-----------------|
| C0 | DeXposure-Agent | FM backbone (GraphPFN) | Yes |
| C2 | Persistence-Agent | G_{t+h} = G_t | Yes |
| C4 | DeXposure-FM | FM backbone | No (model only) |

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
| mc_samples | config.py AgentConfig | 50 | MC samples for B3 uncertainty |
| z_threshold | config.py AgentConfig | 2.0 | Alert detection threshold |

## Environment Variables

Always set before running on GPU server:

    DGLBACKEND=pytorch
    DGL_DISABLE_GRAPHBOLT=1

## Results Directory Structure

    results/
    ├── run_20260325_persistence_baseline/   # C0=C2 (before FM integration)
    ├── run_20260325_fm_integrated/          # FM pi_min=0.2 (first FM run)
    ├── run_20260325_fm_pi05_c0only/         # FM pi_min=0.5 hard threshold
    ├── run_20260325_fm_hybrid/              # FM hybrid (keep edges + reweight)
    └── run_YYYYMMDD_HHMMSS/                # Future runs auto-timestamped
        ├── B1_C0.json                       # Per-benchmark-method results
        ├── B1_C2.json
        ├── ...
        ├── run_summary.json                 # Overall pass/fail + timing
        ├── B1_C0_20260325_*.log             # Per-benchmark detailed log
        └── benchmark_run.log                # Master run log

## Known Issues

- meta_df.csv path: SnapshotLoader resolves from data_dir; if categories show "Unknown",
  check that meta_df.csv is in the same directory as the JSON files
- B3 PI coverage is very low (~1-2% vs target 90%): MC noise sigma=0.1 is too small
  relative to actual week-over-week metric changes; needs calibration
- B4 S2-S4 show perfect scores when meta_df.csv is missing (no categories = no nodes
  to shock for bridge/stablecoin/lending scenarios)
- SSH port forward warning "Address already in use" is cosmetic; ignore it

## Troubleshooting

Model not loading:

    # Check deps
    ssh gpu-server "pip list | grep -iE 'dgl|torch|ogb|xgboost|delu|rtdl|optuna|huggingface'"

    # Check checkpoints exist
    ssh gpu-server "ls -lh /root/graph-dexposure/checkpoints/dexposure-fm-release/*.pt"

Process seems stuck:

    # Check if still running
    ssh gpu-server "pgrep -a python3"

    # Check GPU usage
    ssh gpu-server "nvidia-smi"

    # Kill and restart
    ssh gpu-server "pkill -f python3"
