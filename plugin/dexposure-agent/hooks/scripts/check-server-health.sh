#!/usr/bin/env bash
# Check if the DeXposure-Agent API server is reachable.
GPU_API="${DEXPOSURE_API_URL:-http://gpu-server:8000}"
if curl -sf "${GPU_API}/health" > /dev/null 2>&1; then
    echo "[dexposure-agent] GPU server is reachable at ${GPU_API}"
else
    echo "[dexposure-agent] WARNING: GPU server not reachable at ${GPU_API}. Run scripts/sync_and_serve.sh to start it."
fi
