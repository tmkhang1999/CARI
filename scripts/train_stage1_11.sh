#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_DEVICE="cuda"
CUDA_IDS=""
EXTRA_ARGS=()
RESUME_MODE=""
AUTO_RESUME=0

require_value() {
    local flag="$1"
    if [[ $# -lt 2 || -z "${2:-}" || "${2}" == --* ]]; then
        echo "ERROR: ${flag} requires a value."
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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

CONFIG="${ROOT_DIR}/src/configs/v11.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

if [[ -n "$CUDA_IDS" && "$RUN_DEVICE" != "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
fi

if [[ "$AUTO_RESUME" -eq 1 && -n "$RESUME_MODE" ]]; then
    echo "ERROR: Use either --resume or --auto-resume, not both."
    exit 1
fi

echo "========================================"
echo "  Stage 1 V11"
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
    python "${ROOT_DIR}/src/train_stage1_11.py"
    --version "11"
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
