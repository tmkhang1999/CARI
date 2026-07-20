#!/usr/bin/env bash
# Corrected-cast MID rerun over the full roster (Table-B five + four SOTA).
#
# The published Cast_RMS pools chroma across materials and so rewards models that
# collapse material colour (it ranked the visibly colour-collapsed v17_42 best).
# eval_mid_constancy.py now also emits Cast_within / Cast_between / Cast_rel and the
# Chroma_fid / Sat_ratio fidelity guard. Every model in every Ch5 table must be scored
# with it, or rows are not comparable.
#
# Two jobs per GPU; one Marigold per GPU (~12 GB each) so neither card OOMs.
set -u

ROOT=/home/khang/IR-IID
OUT=$ROOT/tests/visualizations/roster_corrected_cast
PY=/home/khang/miniconda3/envs/IR/bin/python
CK=$ROOT/checkpoints
mkdir -p "$OUT"

# label=path=type
GPU0_JOBS=(
  "v17_20=$CK/v17_20/checkpoint_iter_60000.pth=17"
  "v17_23=$CK/v17_23/checkpoint_iter_60000.pth=17"
  "v17_34=$CK/v17_34/checkpoint_iter_60000.pth=17"
  "CRefNet=$CK/CRefNet/final_real.pt=crefnet"
  "Marigold-App=$CK/marigold-iid-appearance-v1-1=marigold-appearance"
)
GPU1_JOBS=(
  "v17_29=$CK/v17_29/checkpoint_iter_60000.pth=17"
  "v17_33=$CK/v17_33/checkpoint_iter_60000.pth=17"
  "Ordinal=$ROOT/ordinal-hub-weights=ordinal"
  "Marigold-Light=$CK/marigold-iid-lighting-v1-1=marigold-lighting"
)

run_one() {
  local spec="$1" gpu="$2"
  local label="${spec%%=*}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/tests/eval/eval_mid_constancy.py" \
    --ckpts "$spec" \
    --split test --infer-max-size 1280 \
    --save-json "$OUT/mid_${label}.json" \
    > "$OUT/log_${label}.txt" 2>&1
  echo "[done] $label (gpu $gpu)"
}
export -f run_one
export ROOT OUT PY

# 2 concurrent per GPU
printf '%s\n' "${GPU0_JOBS[@]}" | xargs -P 2 -I{} bash -c 'run_one "$@"' _ {} 0 &
P0=$!
printf '%s\n' "${GPU1_JOBS[@]}" | xargs -P 2 -I{} bash -c 'run_one "$@"' _ {} 1 &
P1=$!
wait $P0 $P1
echo "ALL ROSTER JOBS COMPLETE"
