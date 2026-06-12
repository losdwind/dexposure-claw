#!/usr/bin/env bash
# Pre-Terra FM retrain + crisis-window backtest + LLM prompt export.
# Runs ON the Vast instance via:  cloud/train.sh "bash cloud/pre2022_run.sh"
#
# Sequence (planned as one job so the instance is never left idle):
#   1. Fine-tune DeXposure-FM on pre-2022-04 data only (Terra/FTX/SVB unseen).
#   2. Stage checkpoints in the fm_predictor layout.
#   3. m5_fm_rules vs m1_persistence_rules crisis backtest (rules stack, no LLM).
#   4. Export m6/m7 evidence-bundle prompts for the three crisis windows;
#      the LLM decision layer + judge are replayed LOCALLY from these.
# Artifacts land in checkpoints/ and logs/, which train.yaml uploads to HF.
set -euo pipefail
ROOT="$PWD"   # ~/sky_workdir
PY="$ROOT/.venv/bin/python"
# Self-bootstrap the venv: the provision job's setup may have run before the
# code was tar-synced (no pyproject yet), so don't rely on it.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
test -x "$PY" || (cd "$ROOT" && (uv sync --frozen || uv sync))
test -x "$PY" || { echo "FATAL: venv bootstrap failed" >&2; exit 1; }
export DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1
export PYTHONPATH="$ROOT:$ROOT/paper:$ROOT/cloud/train_src"
# DGL graphbolt dlopens libnvrtc.so.12, which lives inside torch's pip-shipped
# nvidia libs and is not on the default loader path on Vast images.
export LD_LIBRARY_PATH="$(echo "$ROOT"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"

echo "=== [1/4] Train pre-Terra FM (holdout >= 2022-04-01) ==="
"$PY" cloud/train_src/run_full_experiment.py \
  --mode dexposure-fm --holdout-start 2022-04-01 \
  --val-weeks 12 --epochs 20 --seed 42 --horizons 1,4,8,12 \
  --output-dir checkpoints/pre2022_train

echo "=== [2/4] Stage checkpoints in fm_predictor layout ==="
mkdir -p checkpoints/pre2022
for h in 1 4 8; do
  src=$(find checkpoints/pre2022_train -name "best_model_h${h}.pt" | head -1)
  test -n "$src" || { echo "FATAL: best_model_h${h}.pt not found" >&2; exit 1; }
  case "$h" in
    8) cp "$src" checkpoints/pre2022/dexposure-fm-h8-h12.pt ;;  # h12 served by h8 weights (release layout)
    *) cp "$src" "checkpoints/pre2022/dexposure-fm-h${h}.pt" ;;
  esac
done
ls -lh checkpoints/pre2022/

# Hosts can die mid-run (machine 16132 vanished 10 min after boot on
# 2026-06-12) -- push the trained weights to HF before the backtest/export
# stages so a host failure cannot destroy them.
if [ -n "${HF_TOKEN:-}" ]; then
  EARLY_TAG="pre2022_$(date +%Y%m%d_%H%M%S)_weights"
  uvx --from 'huggingface_hub[cli]' hf upload losdwind/graph-dexposure-ckpt \
    checkpoints/pre2022 "runs/$EARLY_TAG/checkpoints_pre2022" --repo-type model \
    --commit-message "early weight push: $EARLY_TAG" \
    || echo "WARN: early HF push failed -- continuing" >&2
fi

echo "=== [3/4] Crisis backtest: m5_fm_rules vs m1_persistence_rules ==="
export DEXPOSURE_FM_CKPT_DIR="$ROOT/checkpoints/pre2022"
export DEXPOSURE_DATA_DIR="$ROOT/data"
cd "$ROOT/paper"
CRISIS_SPLIT_KEYS=terra_luna,ftx,svb "$PY" scripts/run_fm_vs_persistence_crisis.py

echo "=== [4/4] Export m6/m7 evidence prompts for local LLM replay ==="
"$PY" -m uvicorn dexposure_agent.serve:app --host 127.0.0.1 --port 8000 \
  > "$ROOT/logs/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT
ok=0
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
test "$ok" = 1 || { echo "FATAL: FM API failed to start"; tail -50 "$ROOT/logs/serve.log"; exit 1; }
curl -fsS http://127.0.0.1:8000/health; echo

for WIN in "2022-04~2022-07" "2022-10~2023-01" "2023-01~2023-05"; do
  "$PY" experiments/llm_eval_b5.py --export-prompts-only \
    --method m6_fm_llm --method m7_fm_llm_gated --test-split "$WIN"
done

echo "=== Collect artifacts for upload ==="
cd "$ROOT"
mkdir -p logs/crisis_backtest
cp -r paper/results/run_fm_vs_persistence_crisis logs/crisis_backtest/
for d in paper/results/llm_eval_*; do
  test -d "$d" && cp -r "$d" logs/crisis_backtest/
done
rm -rf checkpoints/pre2022_train   # keep the HF upload lean
rm -f  checkpoints/graphpfn-v1.ckpt
echo "PRE2022 RUN DONE"
