#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Train —  supported versions: V9 / V10 / V11
#
# Usage:
#   bash scripts/train.sh                                  # V10 (default), CUDA auto
#   bash scripts/train.sh --version 11 --cuda 1           # use GPU 1
#   bash scripts/train.sh --version 9 --device cpu        # force CPU
#   bash scripts/train.sh --version 10 --resume checkpoints/v10/checkpoint_latest.pth
#   bash scripts/train.sh --version 9 --auto-resume
#
# All extra flags are forwarded directly to train_stage1.py (e.g. --device cpu).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default to version 10 if no --version or --config flag is given
VERSION=10
RUN_DEVICE="cuda"          # forwarded to train_stage1.py --device
CUDA_IDS=""                # if set, exported as CUDA_VISIBLE_DEVICES
EXTRA_ARGS=()
RESUME_MODE=""            # forwarded as --resume <path|latest>
AUTO_RESUME=0             # forwarded as --auto-resume
MODE="single"             # Default mode for V11

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
        --version)
            require_value "$1" "${2:-}"
            VERSION="$2"
            shift 2
            ;;
        --mode)
            require_value "$1" "${2:-}"
            MODE="$2"
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

if [[ "${VERSION}" != "9" && "${VERSION}" != "10" && "${VERSION}" != "11" ]]; then
    echo "ERROR: Unsupported version '${VERSION}'. Supported versions: 9, 10, 11"
    exit 1
fi

# Resolve config path and train script. For v11 we use mode flag (single or mix).
if [[ "${VERSION}" == "11" ]]; then
    if [[ "${MODE}" == "mix" ]]; then
        CONFIG="${ROOT_DIR}/src/configs/v11_mix.yaml"
        TRAIN_SCRIPT="${ROOT_DIR}/src/train_mix.py"
    else
        CONFIG="${ROOT_DIR}/src/configs/v11_single.yaml"
        TRAIN_SCRIPT="${ROOT_DIR}/src/train_single.py"
    fi
else
    CONFIG="${ROOT_DIR}/src/configs/v${VERSION}.yaml"
    TRAIN_SCRIPT="${ROOT_DIR}/src/train_stage1.py"  # legacy
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
echo "  Stage 1  |  Version ${VERSION}  |  Mode: ${MODE}"
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
