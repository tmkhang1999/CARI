#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:-"${ROOT_DIR}/src/configs/base.yaml"}

python "${ROOT_DIR}/src/train_stage2.py" --config "${CONFIG}" "$@"

