#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/miniconda3/envs/IR/bin/python}"
RUN_BASE="${RUN_BASE:-tests/visualizations/v17_newtrain_subset25}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUT="$RUN_BASE/$RUN_NAME"

mkdir -p "$OUT"/{mid_vis,maw_pred,maw_vis,arap_vis,iiw_vis}
ln -sfn "$RUN_NAME" "$RUN_BASE/latest"

exec > >(tee -a "$OUT/run.log") 2>&1

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1
export PATH="$HOME/miniconda3/envs/IR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CUDA_INDEX="${CUDA_INDEX:-0}"
DEVICE="${DEVICE:-cuda}"

LABELS=(
  "B_combo_60k"
  "B_combo_70k"
  "B_refiner_65k"
  "A4_fullCARI_50k"
)

CKPTS=(
  "checkpoints/v17_23/checkpoint_iter_60000.pth"
  "checkpoints/v17_23/checkpoint_iter_70000.pth"
  "checkpoints/v17_27/checkpoint_iter_65000.pth"
  "checkpoints/v17_44/checkpoint_iter_50000.pth"
)

CKPT_SPECS=()
for i in "${!LABELS[@]}"; do
  if [[ ! -f "${CKPTS[$i]}" ]]; then
    echo "Missing checkpoint: ${CKPTS[$i]}" >&2
    exit 2
  fi
  CKPT_SPECS+=("${LABELS[$i]}=${CKPTS[$i]}")
done

echo "Output directory: $OUT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  CUDA_INDEX=$CUDA_INDEX  DEVICE=$DEVICE"
echo "Checkpoints:"
for i in "${!LABELS[@]}"; do
  echo "  ${LABELS[$i]} -> ${CKPTS[$i]}"
done

echo "########## [1/5] MID, 25 scenes, max side 1280 ##########"
"$PYTHON" tests/eval/eval_mid_constancy.py \
  --ckpts "${CKPT_SPECS[@]}" \
  --max-scenes 25 \
  --infer-max-size 1280 \
  --device "$DEVICE" \
  --save-json "$OUT/mid.json" \
  --save-vis "$OUT/mid_vis" \
  --vis-scenes 6

echo "########## [2/5] MAW1, 25 images, max side 1280 ##########"
"$PYTHON" tests/eval/eval_maw.py \
  --ckpts "${CKPT_SPECS[@]}" \
  --dataset maw1 \
  --max-scenes 25 \
  --infer-max-size 1280 \
  --infer-min-size 1024 \
  --device "$DEVICE" \
  --cuda-index "$CUDA_INDEX" \
  --save-json "$OUT/maw.json" \
  --pred-dir "$OUT/maw_pred" \
  --save-vis "$OUT/maw_vis"

echo "########## [3/5] ARAP indoor constancy, limit_groups 25 ##########"
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  ckpt="${CKPTS[$i]}"
  echo "===== ARAP: $label ====="
  "$PYTHON" tests/eval/eval_arap.py \
    --constancy \
    --checkpoint "$ckpt" \
    --model_version auto \
    --label "$label" \
    --scene_filter indoor \
    --limit_groups 25 \
    --max_size 1280 \
    --min_size 1024 \
    --device "$DEVICE" \
    --cuda_index "$CUDA_INDEX" \
    --save_json "$OUT/arap.json" \
    --save_dir "$OUT/arap_vis/$label"
done

echo "########## [4/5] IIW, 25 images ##########"
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  ckpt="${CKPTS[$i]}"
  echo "===== IIW: $label ====="
  "$PYTHON" tests/eval/eval_iiw.py \
    --checkpoint "$ckpt" \
    --model_version auto \
    --device "$DEVICE" \
    --cuda_index "$CUDA_INDEX" \
    --max_size 1280 \
    --min_size 1024 \
    --max_images 25 \
    --save_dir "$OUT/iiw_vis/$label" \
    |& tee "$OUT/iiw_${label}.log"
done

echo "########## [5/5] Summarize ##########"
"$PYTHON" tests/eval/summarize_v17_newtrain_subset25.py --run-dir "$OUT"

echo "DONE: $OUT"
