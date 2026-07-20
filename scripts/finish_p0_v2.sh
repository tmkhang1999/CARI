#!/usr/bin/env bash
# Idempotent P0 finisher v2. Fixes the v1 concurrency bug (wait_gpu gated on memory
# dropping below 3.5 GB, which a running 5 GB job never does -> effectively serial per GPU).
#
# v2 uses FIXED SLOTS: SLOTS_LIGHT jobs per GPU for the small models (v17 ~2.5 GB, so 3
# fit in 7.5 GB with room to spare), and Marigolds run ONE per GPU, LAST (12 GB diffusion).
# MID is partly CPU/IO-bound (25 EXR loads per scene), so concurrent jobs fill the GPU
# bubbles a single stream leaves — real throughput gain despite 100% coarse util.
#
# ORDER (per user 2026-07-14): ARAP accuracy (157 imgs, fast) FIRST, then MID (heavy).
# Idempotent: skips any model whose output already exists, so the two in-flight MID jobs
# (v17_29/v17_33) finishing on their own are simply skipped when their JSONs land.
set -u
ROOT=/home/khang/IR-IID
PY=/home/khang/miniconda3/envs/IR/bin/python
CK=$ROOT/checkpoints
MID_OUT=$ROOT/tests/visualizations/rerun_mid_perscene
AACC_OUT=$ROOT/tests/visualizations/rerun_arap_accuracy157
mkdir -p "$MID_OUT" "$AACC_OUT"
cd "$ROOT"
SLOTS_LIGHT=2    # 2/GPU: ARAP peaks ~9GB/job, so 2x=18GB fits; 3x OOMd

LIGHT=(
  v17_20:$CK/v17_20/checkpoint_iter_60000.pth:17
  v17_23:$CK/v17_23/checkpoint_iter_60000.pth:17
  v17_29:$CK/v17_29/checkpoint_iter_60000.pth:17
  v17_33:$CK/v17_33/checkpoint_iter_60000.pth:17
  v17_34:$CK/v17_34/checkpoint_iter_60000.pth:17
  v17_44:$CK/v17_44/checkpoint_iter_40000.pth:17
  v17_41:$CK/v17_41/checkpoint_iter_40000.pth:17
  v17_42:$CK/v17_42/checkpoint_iter_40000.pth:17
  v17_43:$CK/v17_43/checkpoint_iter_40000.pth:17
  CRefNet:$CK/CRefNet/final_real.pt:crefnet
  Ordinal:$ROOT/ordinal-hub-weights:ordinal
)
HEAVY=(
  Marigold-App:$CK/marigold-iid-appearance-v1-1:marigold-appearance
  Marigold-Light:$CK/marigold-iid-lighting-v1-1:marigold-lighting
)

run_arap () { # spec gpu
  IFS=: read -r L P T <<< "$1"
  [ -f "$AACC_OUT/${L}__wb.done" ] && { echo "[arap skip] $L"; return; }
  CUDA_VISIBLE_DEVICES=$2 $PY -u tests/eval/eval_arap.py \
    --checkpoint "$P" --model_version "$T" --white_balance --scene_filter all --max_size 1280 \
    > "$AACC_OUT/${L}__wb.log" 2>&1
  if grep -q Traceback "$AACC_OUT/${L}__wb.log"; then echo "[arap FAIL] $L"
  else touch "$AACC_OUT/${L}__wb.done"; echo "[arap ok] $L (gpu$2)"; fi
}
run_mid () { # spec gpu
  IFS=: read -r L P T <<< "$1"
  [ -f "$MID_OUT/mid_${L}.json" ] && { echo "[mid skip] $L"; return; }
  CUDA_VISIBLE_DEVICES=$2 $PY -u tests/eval/eval_mid_constancy.py \
    --ckpts "${L}=${P}=${T}" --split test --infer-max-size 1280 \
    --save-json "$MID_OUT/mid_${L}.json" > "$MID_OUT/${L}.log" 2>&1
  grep -q Traceback "$MID_OUT/${L}.log" && echo "[mid FAIL] $L" || echo "[mid ok] $L (gpu$2)"
}

# dispatch a job list across both GPUs with a per-GPU slot cap
dispatch () { # funcname slotcap spec...
  local fn=$1 cap=$2; shift 2
  local specs=("$@")
  declare -A n=( [0]=0 [1]=0 )
  local pids0=() pids1=()
  reap () { # drop finished pids, free slots
    local live=(); for p in "${pids0[@]}"; do kill -0 "$p" 2>/dev/null && live+=("$p"); done
    n[0]=${#live[@]}; pids0=("${live[@]}")
    live=(); for p in "${pids1[@]}"; do kill -0 "$p" 2>/dev/null && live+=("$p"); done
    n[1]=${#live[@]}; pids1=("${live[@]}")
  }
  for spec in "${specs[@]}"; do
    while :; do
      reap
      if   [ "${n[0]}" -lt "$cap" ]; then $fn "$spec" 0 & pids0+=($!); break
      elif [ "${n[1]}" -lt "$cap" ]; then $fn "$spec" 1 & pids1+=($!); break
      else sleep 5; fi
    done
    sleep 3
  done
  wait
}

# Wait for any pre-existing eval jobs (e.g. the last in-flight MID) to drain, so Phase 1
# does not stack ARAP onto an already-occupied card and re-trigger the OOM.
while pgrep -f "python.*eval_mid_constancy" >/dev/null || pgrep -f "python.*eval_arap\\.py" >/dev/null; do
  echo "  (waiting for in-flight eval jobs to drain...)"; sleep 20
done
sleep 3
echo "== PHASE 1: ARAP accuracy (157 imgs), ${SLOTS_LIGHT}/GPU, Marigolds last =="
dispatch run_arap "$SLOTS_LIGHT" "${LIGHT[@]}"
run_arap "${HEAVY[0]}" 0 & run_arap "${HEAVY[1]}" 1 & wait
echo "== ARAP-157 COMPLETE =="

echo "== PHASE 2: MID per-scene, ${SLOTS_LIGHT}/GPU, Marigolds last =="
dispatch run_mid "$SLOTS_LIGHT" "${LIGHT[@]}"
run_mid "${HEAVY[0]}" 0 & run_mid "${HEAVY[1]}" 1 & wait
echo "ALL P0 FINISHER v2 COMPLETE"
