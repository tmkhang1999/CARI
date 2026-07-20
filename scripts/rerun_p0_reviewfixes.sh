#!/usr/bin/env bash
# P0 reruns from THESIS_REVIEW_2026-07-14 (+ debate section).
#
#  #1 ARAP standard protocol — the `--white_balance` path used to call `_hdr_norm`, a single
#     scalar multiply that provably leaves r/g and b/g UNCHANGED (verified: kitchen's
#     illuminant stayed at r/g=1.67 before and after). Every published-comparison ARAP number
#     was therefore computed on a raw COLOURED input. `_white_balance_gt` now reconstructs
#     I_wb = A* * luminance(I / A*), which drives the implied illuminant to exactly 1.000/1.000.
#
#  #3 MID with per-scene retention — the old evaluator kept only np.nanmean, which hid that
#     Chroma_fid=1.008 is a mean-of-ratios (ratio-of-aggregates is 0.941) and made bootstrap
#     CIs impossible. Now also emits Chroma_err (the correct hue-calibration guard) per scene.
#
# Both cards are used. Marigolds are the long pole (~1 s/img diffusion), so one per GPU.
set -u
ROOT=/home/khang/IR-IID
PY=/home/khang/miniconda3/envs/IR/bin/python
CK=$ROOT/checkpoints
A_OUT=$ROOT/tests/visualizations/rerun_arap_wb
M_OUT=$ROOT/tests/visualizations/rerun_mid_perscene
mkdir -p "$A_OUT" "$M_OUT"
cd "$ROOT"

# label|path|type
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

run_arap () {  # $1=spec  $2=gpu
  IFS='|' read -r L P T <<< "$1"
  CUDA_VISIBLE_DEVICES=$2 $PY -u tests/eval/eval_arap.py \
    --checkpoint "$P" --model_version "$T" --constancy \
    --white_balance --scene_filter all --max_size 1280 \
    > "$A_OUT/${L}__wb.log" 2>&1
  echo "[arap-wb done] $L (gpu $2)"
}
run_mid () {   # $1=spec  $2=gpu
  IFS='|' read -r L P T <<< "$1"
  CUDA_VISIBLE_DEVICES=$2 $PY -u tests/eval/eval_mid_constancy.py \
    --ckpts "${L}=${P}=${T}" --split test --infer-max-size 1280 \
    --save-json "$M_OUT/mid_${L}.json" \
    > "$M_OUT/${L}.log" 2>&1
  echo "[mid done] $L (gpu $2)"
}
export -f run_arap run_mid
export ROOT PY A_OUT M_OUT

# GPU0 takes ARAP, GPU1 takes MID; 2 concurrent each, Marigolds land last on both lists.
printf '%s\n' "${ROSTER[@]}" | xargs -P 2 -I{} bash -c 'run_arap "$@"' _ {} 0 &
P0=$!
printf '%s\n' "${ROSTER[@]}" | xargs -P 2 -I{} bash -c 'run_mid "$@"' _ {} 1 &
P1=$!
wait $P0 $P1
echo "ALL P0 RERUNS COMPLETE"
