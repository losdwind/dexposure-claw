#!/usr/bin/env bash
# The ONLY sanctioned way to launch GPU training on Vast.ai.
# Wraps `sky launch` with autodown: the instance destroys itself after the
# job finishes + $IDLE minutes, even if this laptop goes offline.
#
# Usage:
#   cloud/train.sh python lib/train.py --epochs 10
#   GPU=A100:1 IDLE=5 CLUSTER=exp2 cloud/train.sh python paper/experiments/foo.py
#
# Afterwards:
#   sky status          # what is running (run this if ever unsure)
#   sky logs train      # stream logs
#   sky down train      # destroy early by hand
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
  echo "usage: cloud/train.sh <training command...>" >&2
  exit 1
fi

CLUSTER="${CLUSTER:-train}"
IDLE="${IDLE:-15}"   # minutes to keep the instance after the job ends (for sky logs / ssh), then auto-destroy
GPU="${GPU:-}"       # e.g. RTX4090:1, RTX3090:1, A100:1 -- empty = default in cloud/train.yaml

ARGS=(-y -d -c "$CLUSTER" -i "$IDLE" --down)
if [ -n "$GPU" ]; then
  ARGS+=(--gpus "$GPU")
fi

# HF token for checkpoint upload (see run step in cloud/train.yaml).
# --secret is redacted in sky logs/YAML, unlike --env.
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [ -f "$HF_TOKEN_FILE" ]; then
  ARGS+=(--secret "HF_TOKEN=$(cat "$HF_TOKEN_FILE")")
else
  echo "WARNING: $HF_TOKEN_FILE not found -- training results will not be uploaded!" >&2
fi

exec sky launch "${ARGS[@]}" --env TRAIN_CMD="$*" cloud/train.yaml
