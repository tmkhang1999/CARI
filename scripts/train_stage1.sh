#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Train Stage 1  —  works for ALL versions (V1 / V2 / V3 / V4)
#
# Usage:
#   bash scripts/train_stage1.sh                                 # V1 (default), CUDA auto
#   bash scripts/train_stage1.sh --version 2                    # V2
#   bash scripts/train_stage1.sh --version 3 --cuda 1           # use GPU 1 (CUDA_VISIBLE_DEVICES=1)
#   bash scripts/train_stage1.sh --version 4 --device cpu       # force CPU
#   bash scripts/train_stage1.sh --version 2 --cuda 0,1         # expose multiple GPUs
#   bash scripts/train_stage1.sh --config src/configs/v2.yaml   # explicit config
#   bash scripts/train_stage1.sh --version 3 --resume checkpoints/v3/checkpoint_iter_20000.pth
#   bash scripts/train_stage1.sh --version 3 --auto-resume
#
# All extra flags are forwarded directly to train_stage1.py (e.g. --device cpu).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default to version 1 if no --version or --config flag is given
VERSION=1
RUN_DEVICE="cuda"          # forwarded to train_stage1.py --device
CUDA_IDS=""                # if set, exported as CUDA_VISIBLE_DEVICES
EXTRA_ARGS=()
RESUME_MODE=""            # forwarded as --resume <path|latest>
AUTO_RESUME=0             # forwarded as --auto-resume

# Parse script-owned flags; pass unknown flags through unchanged.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --cuda|--gpus|--cuda-visible-devices)
            CUDA_IDS="$2"
            shift 2
            ;;
        --device)
            RUN_DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME_MODE="$2"
            shift 2
            ;;
        --auto-resume)
            AUTO_RESUME=1
            shift 1
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

CONFIG="${ROOT_DIR}/src/configs/v${VERSION}.yaml"
# Handle versions like 2.5 which are stored as v2_5.yaml
if [[ ! -f "$CONFIG" ]]; then
    ALT_CONFIG="${ROOT_DIR}/src/configs/v${VERSION//./_}.yaml"
    if [[ -f "$ALT_CONFIG" ]]; then
        CONFIG="$ALT_CONFIG"
    fi
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    echo "Available configs: $(ls ${ROOT_DIR}/src/configs/v*.yaml 2>/dev/null | xargs -I{} basename {})"
    exit 1
fi

# Respect explicit CUDA selection unless running on CPU.
if [[ -n "$CUDA_IDS" && "$RUN_DEVICE" != "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
fi

if [[ "$AUTO_RESUME" -eq 1 && -n "$RESUME_MODE" ]]; then
    echo "ERROR: Use either --resume or --auto-resume, not both."
    exit 1
fi

echo "========================================"
echo "  Stage 1  |  Version ${VERSION}"
echo "  Config:  ${CONFIG}"
echo "  Device:  ${RUN_DEVICE}"
if [[ -n "$CUDA_IDS" ]]; then
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_IDS}"
fi
if [[ "$AUTO_RESUME" -eq 1 ]]; then
    echo "  Resume:  auto"
elif [[ -n "$RESUME_MODE" ]]; then
    echo "  Resume:  ${RESUME_MODE}"
fi
echo "========================================"

CMD=(
    python "${ROOT_DIR}/src/train_stage1.py"
    --version "${VERSION}"
    --config "${CONFIG}"
    --device "${RUN_DEVICE}"
)

if [[ "$AUTO_RESUME" -eq 1 ]]; then
    CMD+=(--auto-resume)
elif [[ -n "$RESUME_MODE" ]]; then
    CMD+=(--resume "$RESUME_MODE")
fi

CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
