#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/precompute_midintrinsic_geometry.sh \
#     --mid_root ../datasets/MIDIntrinsics \
#     --output_root ../datasets/MIDIntrinsics/geometry_midintrinsic \
#     --split all --device cuda:0
#
# Note:
#   Geometry is now always computed from each scene's thumb.jpg.

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[error] Could not find conda.sh (checked ~/miniconda3 and ~/anaconda3)." >&2
  exit 1
fi

conda activate IR

python preprocessor/precompute_midintrinsic_geometry.py "$@"
