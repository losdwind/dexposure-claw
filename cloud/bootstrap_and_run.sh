#!/usr/bin/env bash
# Provision a Vast instance via SkyPilot, work around its rsync issues,
# and submit the pre-2022 run. Idempotent: re-run after instance loss.
#
# Workarounds baked in (learned 2026-06-11):
#   - Vast images ship without rsync -> install it before anything else.
#   - SkyPilot's internal rsync stalls on the Vast SSH proxy -> sync the
#     workdir + data with single-stream tar over ssh instead, and launch
#     with cloud/train_nosync.yaml (no workdir/file_mounts).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:$HOME/.local/bin"

CLUSTER="${CLUSTER:-train}"
IDLE="${IDLE:-15}"
REGION="${REGION:-}"   # e.g. "iceland, is, eu" -- pin away from flaky hosts
GPU="${GPU:-}"         # e.g. H100:1 -- override the YAML default (RTX4090)

echo "=== [1/5] Provision (idle ${IDLE}m, autodown) ==="
LAUNCH_ARGS=(-y -d -c "$CLUSTER" -i "$IDLE" --down)
[ -n "$REGION" ] && LAUNCH_ARGS+=(--infra "vast/$REGION")
[ -n "$GPU" ] && LAUNCH_ARGS+=(--gpus "$GPU")
sky launch "${LAUNCH_ARGS[@]}" \
  --secret "HF_TOKEN=$(cat ~/.cache/huggingface/token)" \
  --env TRAIN_CMD="true" cloud/train_nosync.yaml
sky status | grep "$CLUSTER.*UP" || { echo "FATAL: cluster not UP"; exit 1; }

echo "=== [2/5] Ensure rsync exists on the instance ==="
ssh "$CLUSTER" "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)" 2>/dev/null

echo "=== [3/5] Sync code (tar single-stream) ==="
tar --exclude='./data' --exclude='./checkpoints' --exclude='./archive' \
    --exclude='./docs' --exclude='./icml2026' --exclude='./paper-emnlp-industry' \
    --exclude='./claw/dist' --exclude='./claw/pack' --exclude='./.git' \
    --exclude='./.venv' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='*.pdf' --exclude='./tmp' --exclude='./cloud/upload' \
    --exclude='node_modules' -czf - . \
  | ssh "$CLUSTER" "mkdir -p sky_workdir && tar -xzf - -C sky_workdir" 2>/dev/null

echo "=== [4/5] Sync data subset + base ckpt ==="
tar -C cloud/upload -czf - data graphpfn-v1.ckpt \
  | ssh "$CLUSTER" "cd sky_workdir && tar -xzf - && mkdir -p checkpoints && mv -f graphpfn-v1.ckpt checkpoints/" 2>/dev/null
ssh "$CLUSTER" "du -sh sky_workdir sky_workdir/data" 2>/dev/null

if [ "${SUBMIT:-1}" = "1" ]; then
  echo "=== [5/5] Submit the real job ==="
  sky exec "$CLUSTER" -d --env TRAIN_CMD="bash cloud/pre2022_run.sh" cloud/train_nosync.yaml
  sky queue "$CLUSTER"
else
  echo "=== [5/5] SKIPPED (SUBMIT=0) -- run manually via ssh $CLUSTER ==="
fi
echo "BOOTSTRAP DONE -- monitor with: sky logs $CLUSTER"
