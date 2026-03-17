#!/usr/bin/env bash
# Sync local code to GPU server and (re)start API server.
# Usage: bash scripts/sync_and_serve.sh [gpu-server-host]
set -euo pipefail

GPU_HOST="${1:-gpu-server}"
REMOTE_DIR="~/CodeProjects/graph-dexposure"

echo "[sync] Pushing latest code to ${GPU_HOST}..."
rsync -avz --exclude='.venv' --exclude='data/' --exclude='checkpoints/' \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    ./ "${GPU_HOST}:${REMOTE_DIR}/"

echo "[serve] Stopping any existing server..."
ssh "${GPU_HOST}" "pkill -f 'uvicorn lib.agent.serve' || true"

echo "[serve] Starting API server..."
ssh "${GPU_HOST}" "cd ${REMOTE_DIR} && nohup bash scripts/start_server.sh > /dev/null 2>&1 &"

# Wait a moment and check health
sleep 3
echo "[check] Verifying server health..."
if ssh "${GPU_HOST}" "curl -sf http://localhost:8000/health" > /dev/null 2>&1; then
    echo "[done] Server running at ${GPU_HOST}:8000"
else
    echo "[warn] Server may still be starting (model loading takes time). Check with:"
    echo "  ssh ${GPU_HOST} 'curl http://localhost:8000/health'"
fi
