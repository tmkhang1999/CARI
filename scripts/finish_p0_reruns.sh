#!/usr/bin/env bash
# Idempotent finisher for the full 13-model P0 rerun. Computes remaining work from disk
# (skips any model whose output already exists), so it is safe to launch while other jobs
# are still finishing and safe to re-run.
#
# Honours two standing instructions:
#   * MID first, flooding BOTH GPUs (user priority — MID is the heavy benchmark).
#   * Marigolds ALWAYS LAST and one-per-GPU (they are ~12 GB diffusion models; two on one
#     card OOMs, as Marigold-Light already did).
#
# Three evals make up the 13-model roster:
#   MID per-scene            (constancy + Chroma_err + bootstrap inputs)
#   ARAP constancy, 37 grp   (C_arap / Cast_RMS — raw + white-balanced)
#   ARAP accuracy, 157 img   (LMSE/RMSE/SSIM vs published Ordinal — needs NO --constancy;
#                             this is the full 51-scene/157-image protocol the SOTA table
#                             must use, distinct from the multi-light constancy subset)
set -u
ROOT=/home/khang/IR-IID
PY=/home/khang/miniconda3/envs/IR/bin/python
CK=$ROOT/checkpoints
MID_OUT=$ROOT/tests/visualizations/rerun_mid_perscene
AWB_OUT=$ROOT/tests/visualizations/rerun_arap_wb
AACC_OUT=$ROOT/tests/visualizations/rerun_arap_accuracy157
mkdir -p "$MID_OUT" "$AWB_OUT" "$AACC_OUT"
cd "$ROOT"

# label|path|type  — non-Marigold first, Marigolds LAST (heavy)
LIGHT=(
  "v17_20|$CK/v17_20/checkpoint_iter_60000.pth|17"
  "v17_23|$CK/v17_23/checkpoint_iter_60000.pth|17"
  "v17_29|$CK/v17_29/checkpoint_iter_60000.pth|17"
  "v17_33|$CK/v17_33/checkpoint_iter_60000.pth|17"
  "v17_34|$CK/v17_34/checkpoint_iter_60000.pth|17"
  "v17_44|$CK/v17_44/checkpoint_iter_40000.pth|17"
  "v17_41|$CK/v17_41/checkpoint_iter_40000.pth|17"
  "v17_42|$CK/v17_42/checkpoint_iter_40000.pth|17"
  "v17_43|$CK/v17_43/checkpoint_iter_40000.pth|17"
  "CRefNet|$CK/CRefNet/final_real.pt|crefnet"
  "Ordinal|$ROOT/ordinal-hub-weights|ordinal"
)
HEAVY=(
  "Marigold-App|$CK/marigold-iid-appearance-v1-1|marigold-appearance"
  "Marigold-Light|$CK/marigold-iid-lighting-v1-1|marigold-lighting"
)

gpu_free () { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")" -lt 3500 ]; }
wait_gpu () { while ! gpu_free "$1"; do sleep 20; done; }

run_mid () { # spec gpu
  IFS='|' read -r L P T <<< "$1"
  [ -f "$MID_OUT/mid_${L}.json" ] && { echo "[mid skip] $L"; return; }
  CUDA_VISIBLE_DEVICES=$2 $PY -u tests/eval/eval_mid_constancy.py \
    --ckpts "${L}=${P}=${T}" --split test --infer-max-size 1280 \
    --save-json "$MID_OUT/mid_${L}.json" > "$MID_OUT/${L}.log" 2>&1
  grep -q Traceback "$MID_OUT/${L}.log" && echo "[mid FAIL] $L" || echo "[mid ok] $L"
}
run_arap_acc () { # spec gpu — full 157-image accuracy protocol, WHITE-BALANCED, no --constancy
  IFS='|' read -r L P T <<< "$1"
  [ -f "$AACC_OUT/${L}__wb.done" ] && { echo "[arap-acc skip] $L"; return; }
  CUDA_VISIBLE_DEVICES=$2 $PY -u tests/eval/eval_arap.py \
    --checkpoint "$P" --model_version "$T" \
    --white_balance --scene_filter all --max_size 1280 \
    > "$AACC_OUT/${L}__wb.log" 2>&1
  if grep -q Traceback "$AACC_OUT/${L}__wb.log"; then echo "[arap-acc FAIL] $L"
  else touch "$AACC_OUT/${L}__wb.done"; echo "[arap-acc ok] $L"; fi
}

# ── PHASE 1: MID, flood both GPUs (light models 2-per-GPU) ────────────────────
echo "== PHASE 1: MID (both GPUs) =="
i=0
for spec in "${LIGHT[@]}"; do
  L="${spec%%|*}"; [ -f "$MID_OUT/mid_${L}.json" ] && continue
  g=$(( i % 2 )); i=$((i+1))
  wait_gpu "$g"; run_mid "$spec" "$g" &
  sleep 8   # stagger so the two on a card don't both peak on model load
  # keep at most 2 per GPU => at most 4 concurrent
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 10; done
done
wait
# Marigolds last, one per GPU, solo
run_mid "${HEAVY[0]}" 0 &
run_mid "${HEAVY[1]}" 1 &
wait
echo "== MID COMPLETE =="

# ── PHASE 2: ARAP accuracy 157 (both GPUs), Marigolds last ────────────────────
echo "== PHASE 2: ARAP accuracy (157 imgs) =="
i=0
for spec in "${LIGHT[@]}"; do
  L="${spec%%|*}"; [ -f "$AACC_OUT/${L}__wb.done" ] && continue
  g=$(( i % 2 )); i=$((i+1))
  wait_gpu "$g"; run_arap_acc "$spec" "$g" &
  sleep 4
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 8; done
done
wait
run_arap_acc "${HEAVY[0]}" 0 &
run_arap_acc "${HEAVY[1]}" 1 &
wait
echo "ALL P0 FINISHER WORK COMPLETE"
