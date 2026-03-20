#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT=${1:-"${ROOT_DIR}/checkpoints/checkpoint_epoch_0.pth"}

python "${ROOT_DIR}/src/eval_hypersim.py" --checkpoint "${CHECKPOINT}" "$@"

