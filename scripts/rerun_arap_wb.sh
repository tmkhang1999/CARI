#!/usr/bin/env bash
# ARAP standard-protocol rerun with TRUE white balance (P0 #1 from THESIS_REVIEW_2026-07-14).
#
# The old --white_balance path called _hdr_norm (one scalar multiply) and provably left
# r/g, b/g UNCHANGED — verified: kitchen's illuminant read r/g=1.67 both before and after.
# _white_balance_gt now reconstructs I_wb = A* * luminance(I / A*), driving the implied
# illuminant to exactly 1.000/1.000. Every published-comparison ARAP number must be redone.
#
# GPU0 only: the MID per-scene rerun owns GPU1.
set -u
ROOT=/home/khang/IR-IID
PY=/home/khang/miniconda3/envs/IR/bin/python
CK=$ROOT/checkpoints
OUT=$ROOT/tests/visualizations/rerun_arap_wb
mkdir -p "$OUT"
cd "$ROOT"

ROSTER=(
  "v17_34|$CK/v17_34/checkpoint_iter_60000.pth|17"
  "v17_44|$CK/v17_44/checkpoint_iter_40000.pth|17"
  "v17_20|$CK/v17_20/checkpoint_iter_60000.pth|17"
  "v17_23|$CK/v17_23/checkpoint_iter_60000.pth|17"
  "v17_29|$CK/v17_29/checkpoint_iter_60000.pth|17"
  "v17_33|$CK/v17_33/checkpoint_iter_60000.pth|17"
  "CRefNet|$CK/CRefNet/final_real.pt|crefnet"
  "Ordinal|$ROOT/ordinal-hub-weights|ordinal"
  "Marigold-App|$CK/marigold-iid-appearance-v1-1|marigold-appearance"
  "Marigold-Light|$CK/marigold-iid-lighting-v1-1|marigold-lighting"
)

run_one () {
  IFS='|' read -r L P T <<< "$1"
  CUDA_VISIBLE_DEVICES=0 $PY -u tests/eval/eval_arap.py \
    --checkpoint "$P" --model_version "$T" --constancy \
    --white_balance --scene_filter all --max_size 1280 \
    > "$OUT/${L}__wb.log" 2>&1
  if grep -q "Traceback" "$OUT/${L}__wb.log"; then
    echo "[FAIL] $L"
  else
    echo "[ok] $L"
  fi
}
export -f run_one
export ROOT PY OUT

printf '%s\n' "${ROSTER[@]}" | xargs -P 2 -I{} bash -c 'run_one "$@"' _ {}
echo "ARAP-WB RERUN COMPLETE"
