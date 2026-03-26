#!/usr/bin/env bash
# Check if the DeXposure-FM API server is reachable.
FM_API="${FM_API_URL:-http://localhost:8000}"
if curl -sf "${FM_API}/health" > /dev/null 2>&1; then
    echo "[dexposure-agent] FM API server is reachable at ${FM_API}"
else
    echo "[dexposure-agent] WARNING: FM API server not reachable at ${FM_API}. Start the server on GPU or set FM_API_URL env var. If using SSH tunnel: ssh -f -N -L 8000:localhost:8000 gpu-server"
fi
