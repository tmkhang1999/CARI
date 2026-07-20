#!/usr/bin/env bash
# SUPPLEMENT: Table-A rows 1-3 (v17_41/42/43), missing from the first P0 roster.
#
# WHY THIS WAS NEEDED (caught 2026-07-14): the first rerun covered the Table-B five +
# v17_44 + the four SOTA = 10 models. But Table A (tab:ablation) Part 2a carries its own
# "ARAP (white-balanced)" LMSE/RMSE/SSIM columns for rows 1-4 = v17_41/42/43/44, and those
# were produced by the SAME broken _hdr_norm path. Only v17_44 was in the roster, so rows
# 1-3 would have kept their invalid numbers.
#
# MID per-scene is needed for the same three, because the paired bootstrap test of the CARI
# claim ("Cast_rel improves in both architectures", rows 1->2 and 3->4) cannot be run
# without per-scene vectors for rows 1-3.
#
# Full corrected roster is therefore 13 models, not 10.
set -u
ROOT=/home/khang/IR-IID
PY=/home/khang/miniconda3/envs/IR/bin/python
CK=$ROOT/checkpoints
A_OUT=$ROOT/tests/visualizations/rerun_arap_wb
M_OUT=$ROOT/tests/visualizations/rerun_mid_perscene
mkdir -p "$A_OUT" "$M_OUT"
cd "$ROOT"

EXTRA=(
  "v17_41|$CK/v17_41/checkpoint_iter_40000.pth|17"
  "v17_42|$CK/v17_42/checkpoint_iter_40000.pth|17"
  "v17_43|$CK/v17_43/checkpoint_iter_40000.pth|17"
)

# Wait for the two in-flight jobs to release their cards before adding load.
while pgrep -f "python.*eval_arap\.py" > /dev/null || \
      pgrep -f "python.*eval_mid_constancy" > /dev/null; do sleep 30; done
sleep 5

arap_one () {
  IFS='|' read -r L P T <<< "$1"
  CUDA_VISIBLE_DEVICES=0 $PY -u tests/eval/eval_arap.py \
    --checkpoint "$P" --model_version "$T" --constancy \
    --white_balance --scene_filter all --max_size 1280 \
    > "$A_OUT/${L}__wb.log" 2>&1
  grep -q Traceback "$A_OUT/${L}__wb.log" && echo "[FAIL arap] $L" || echo "[ok arap] $L"
}
mid_one () {
  IFS='|' read -r L P T <<< "$1"
  CUDA_VISIBLE_DEVICES=1 $PY -u tests/eval/eval_mid_constancy.py \
    --ckpts "${L}=${P}=${T}" --split test --infer-max-size 1280 \
    --save-json "$M_OUT/mid_${L}.json" \
    > "$M_OUT/${L}.log" 2>&1
  grep -q Traceback "$M_OUT/${L}.log" && echo "[FAIL mid] $L" || echo "[ok mid] $L"
}
export -f arap_one mid_one
export ROOT PY A_OUT M_OUT

printf '%s\n' "${EXTRA[@]}" | xargs -P 2 -I{} bash -c 'arap_one "$@"' _ {} &
Q0=$!
printf '%s\n' "${EXTRA[@]}" | xargs -P 2 -I{} bash -c 'mid_one "$@"' _ {} &
Q1=$!
wait $Q0 $Q1
echo "TABLE-A SUPPLEMENT COMPLETE (13-model roster now whole)"
