#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Train a model from src/configs.
#
#   bash scripts/train.sh --version 17_44 --cuda 0          # base CIAI model (V17)
#   bash scripts/train.sh --version 17_62 --cuda 0 --seed 43
#   bash scripts/train.sh --version 17_44 --auto-resume
#   bash scripts/train.sh --version 21 --cuda 0             # next model (V21)
#
# --version 17_xx runs src/train_v17.py with src/configs/v17_xx.yaml; --version 21 runs
# src/train_v21.py with src/configs/v21.yaml. Unknown flags are passed through.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION=17_44
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
        --version)
            require_value "$1" "${2:-}"
            VERSION="$2"
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

VERSION="${VERSION//./_}"
if [[ "${VERSION}" == 17_* ]]; then
    TRAIN_SCRIPT="${ROOT_DIR}/src/train_v17.py"
elif [[ "${VERSION}" == 21* ]]; then
    TRAIN_SCRIPT="${ROOT_DIR}/src/train_v21.py"
else
    echo "ERROR: unsupported version '${VERSION}'. Use 17_xx (V17 configs) or 21 (V21)."
    exit 1
fi
CONFIG="${ROOT_DIR}/src/configs/v${VERSION}.yaml"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    echo "Available configs: $(ls ${ROOT_DIR}/src/configs/v*.yaml 2>/dev/null | xargs -I{} basename {})"
    exit 1
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
echo "  Version ${VERSION}  |  ${TRAIN_SCRIPT##*/}"
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
