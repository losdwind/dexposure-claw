#!/usr/bin/env bash
# Launch a Vast instance via SkyPilot and VERIFY the actual GPU before use.
#
# Why: Vast hosts can hand back a different GPU than advertised (observed:
# "H100" -> RTX 5090, "A100" -> RTX PRO 6000 Blackwell). Blackwell (sm_120)
# cannot run our locked torch 2.2.1+cu121 / dgl 2.1.0 stack; the resulting
# CUDA error made five training runs die silently (job exits -> idle ->
# autodown). Fix: after each launch, poll until UP, check nvidia-smi; if the
# card is not on the allowlist, tear down and try the next GPU type.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:$HOME/.local/bin"

CLUSTER="${CLUSTER:-train}"
IDLE="${IDLE:-120}"
REGION="${REGION:-}"   # e.g. "iceland, is, eu" -- pin to a known datacenter host
GPUS="${GPUS:-RTX5880-Ada:1 H100:1 RTX4090:1 RTX3090:1}"
# Cards our torch 2.2.1+cu121 stack supports (sm_70..sm_90).
ALLOW_RE="${ALLOW_RE:-A100|H100|H200|RTX 4090|RTX 3090|6000 Ada|5880|L40|V100}"
DENY_RE="5090|PRO 6000|Blackwell"

for GPU in $GPUS; do
  echo "=== trying $GPU ==="
  LAUNCH_ARGS=(-y -d -c "$CLUSTER" -i "$IDLE" --down --gpus "$GPU")
  [ -n "$REGION" ] && LAUNCH_ARGS+=(--infra "vast/$REGION")
  # `sky launch` blocks until provisioning+setup are done (-d only skips
  # streaming job logs), so its exit code is the readiness signal. Do NOT
  # poll `sky status` here -- a buggy grep on it once tore down a healthy
  # cluster (2026-06-12).
  if ! sky launch "${LAUNCH_ARGS[@]}" \
      --secret "HF_TOKEN=$(cat ~/.cache/huggingface/token)" \
      --env TRAIN_CMD="true" cloud/train_nosync.yaml; then
    echo "launch failed for $GPU -- tearing down"
    sky down "$CLUSTER" -y 2>/dev/null
    continue
  fi
  CARD=""
  for _ in $(seq 1 12); do
    CARD=$(ssh -o ConnectTimeout=10 "$CLUSTER" \
      "nvidia-smi --query-gpu=name --format=csv,noheader | head -1" 2>/dev/null)
    [ -n "$CARD" ] && break
    sleep 10
  done
  if [ -z "$CARD" ]; then
    echo "ssh/nvidia-smi unreachable for $GPU -- tearing down"
    sky down "$CLUSTER" -y 2>/dev/null
    continue
  fi
  echo "got card: '$CARD'"
  if echo "$CARD" | grep -qE "$DENY_RE"; then
    echo "MISMATCH (Blackwell: $CARD) -- tearing down"
    sky down "$CLUSTER" -y
    continue
  fi
  if echo "$CARD" | grep -qE "$ALLOW_RE"; then
    echo "VERIFIED: $CARD"
    exit 0
  fi
  echo "UNRECOGNISED ($CARD) -- tearing down"
  sky down "$CLUSTER" -y
done
echo "FATAL: no allowed GPU across types: $GPUS" >&2
exit 1
