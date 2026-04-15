#!/bin/bash
# Run LLM evaluation pipeline locally.
#
# This script:
#   1. Sets up SSH tunnel to GPU server (if not already running)
#   2. Verifies FM API is reachable
#   3. Runs llm_eval_b5.py with specified methods
#
# Prerequisites:
#   - ANTHROPIC_API_KEY set in environment
#   - GPU server accessible via ssh gpu-server
#   - FM API running on GPU server (port 8000)
#
# Usage:
#   bash DeXposure_Agent/scripts/run_llm_eval.sh
#   bash DeXposure_Agent/scripts/run_llm_eval.sh --method C0-LLM
#   bash DeXposure_Agent/scripts/run_llm_eval.sh --resume
#   bash DeXposure_Agent/scripts/run_llm_eval.sh --model claude-haiku-4-5 --no-judge

set -euo pipefail
cd "$(dirname "$0")/.."

# Check API key (OpenRouter or Anthropic)
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: No LLM API key set"
    echo "  export OPENROUTER_API_KEY=sk-or-..."
    echo "  or export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

# Set up SSH tunnel if not already running
if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "Setting up SSH tunnel to GPU server..."
    ssh -f -N -L 8000:localhost:8000 gpu-server 2>/dev/null || true
    sleep 2

    if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "ERROR: Cannot reach FM API at localhost:8000"
        echo "  1. Ensure GPU server is running: ssh gpu-server"
        echo "  2. Start FM API if needed (see CLAUDE.md)"
        echo "  3. Set up tunnel: ssh -f -N -L 8000:localhost:8000 gpu-server"
        exit 1
    fi
fi

echo "FM API: $(curl -sf http://localhost:8000/health)"

# Check Python deps
python3 -c "import scipy; import numpy; import loguru" 2>/dev/null || {
    echo "Installing missing Python dependencies..."
    pip install scipy numpy loguru
}

# Run the evaluation
echo "Starting LLM evaluation pipeline..."
echo "  Decision model: ${LLM_EVAL_MODEL:-claude-sonnet-4-6}"
echo "  Judge model: ${LLM_JUDGE_MODEL:-claude-haiku-4-5}"
echo ""

python3 experiments/llm_eval_b5.py "$@"
