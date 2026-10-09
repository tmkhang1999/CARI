#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Train a model from src/configs.
#
#   bash scripts/train.sh --config stage_a --cuda 0         # Stage A (stop at 19k)
#   bash scripts/train.sh --config ciai --cuda 0 --skip-optimizer   # the reported model
#   bash scripts/train.sh --config ciai --auto-resume
#   bash scripts/train.sh --config trifactor --cuda 0       # next model
#
# --config NAME uses src/configs/NAME.yaml. "trifactor" runs src/train_trifactor.py, every
# other config runs src/train.py. Unknown flags are passed through.
# ------------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_NAME=ciai
RUN_DEVICE="cuda"          # forwarded as --device
CUDA_IDS=""                # if set, exported as CUDA_VISIBLE_DEVICES
EXTRA_ARGS=()
RESUME_MODE=""             # forwarded as --resume <path|latest>
AUTO_RESUME=0              # forwarded as --auto-resume

require_value() {
    local flag="$1"
    if [[ $# -lt 2 || -z "${2:-}" || "${2}" == --* ]]; then
        echo "ERROR: ${flag} requires a value."
        exit 2
    fi
}

# Parse script-owned flags; pass unknown flags through unchanged.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            require_value "$1" "${2:-}"
            CONFIG_NAME="$2"
            shift 2
            ;;
        --cuda|--gpus|--cuda-visible-devices)
            require_value "$1" "${2:-}"
            CUDA_IDS="$2"
            shift 2
            ;;
        --device)
            require_value "$1" "${2:-}"
            RUN_DEVICE="$2"
            shift 2
            ;;
        --resume)
            # Allow bare --resume to mean --resume latest.
            if [[ $# -ge 2 && -n "${2:-}" && "${2}" != --* ]]; then
                RESUME_MODE="$2"
                shift 2
            else
                RESUME_MODE="latest"
                shift 1
            fi
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

CONFIG="${ROOT_DIR}/src/configs/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG" || "$CONFIG_NAME" == "base" ]]; then
    echo "ERROR: config not found or not runnable: $CONFIG"
    echo "Available configs: $(ls ${ROOT_DIR}/src/configs/*.yaml | xargs -n1 basename | sed 's/.yaml//' | grep -v '^base$' | tr '\n' ' ')"
    exit 1
fi
if [[ "$CONFIG_NAME" == "trifactor" ]]; then
    TRAIN_SCRIPT="${ROOT_DIR}/src/train_trifactor.py"
    CONFIG_ARG="$CONFIG"
else
    TRAIN_SCRIPT="${ROOT_DIR}/src/train.py"
    CONFIG_ARG="$CONFIG_NAME"     # train.py names the run directory after the config
fi

# Respect explicit CUDA selection unless running on CPU.
if [[ -n "$CUDA_IDS" && "$RUN_DEVICE" != "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
fi

# Reduce CUDA allocator fragmentation (two forward passes per step on a 24 GB card).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$AUTO_RESUME" -eq 1 && -n "$RESUME_MODE" ]]; then
    echo "ERROR: Use either --resume or --auto-resume, not both."
    exit 1
fi

echo "========================================"
echo "  Config ${CONFIG_NAME}  |  ${TRAIN_SCRIPT##*/}"
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
    python "${TRAIN_SCRIPT}"
    --config "${CONFIG_ARG}"
    --device "${RUN_DEVICE}"
)

if [[ "$AUTO_RESUME" -eq 1 ]]; then
    CMD+=(--auto-resume)
elif [[ -n "$RESUME_MODE" ]]; then
    CMD+=(--resume "$RESUME_MODE")
fi

CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
