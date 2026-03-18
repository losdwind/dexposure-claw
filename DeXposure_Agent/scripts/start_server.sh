#!/usr/bin/env bash
# Start the DeXposure-Agent API server on GPU server.
# Usage: ssh gpu-server 'bash -s' < scripts/start_server.sh
set -euo pipefail

cd ~/CodeProjects/graph-dexposure
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export DEXPOSURE_CHECKPOINT_DIR=./checkpoints

# Ensure log directory exists
mkdir -p logs

echo "[$(date)] Starting DeXposure-Agent API server..."
python -m uvicorn lib.agent.serve:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --timeout-keep-alive 300 \
    2>&1 | tee logs/serve_$(date +%Y%m%d_%H%M%S).log
