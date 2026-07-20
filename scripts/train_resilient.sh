#!/usr/bin/env bash
# Resilient train wrapper: restart a version after OOM/kill and resume latest.
#
# Required:
#   VERSION=17.44 CUDA=1 bash scripts/train_resilient.sh
#
# First launch from an external checkpoint, only used when no checkpoint exists yet:
#   VERSION=17.44 CUDA=1 \
#   INITIAL_RESUME=checkpoints/v17_old/checkpoint_iter_19000.pth \
#   INITIAL_SKIP_OPTIMIZER=1 \
#   bash scripts/train_resilient.sh
#
# Normal restart after at least one checkpoint exists uses:
#   bash scripts/train.sh --version $VERSION --cuda $CUDA --auto-resume
#
# Optional:
#   MAX_RESTARTS=20 SLEEP_SEC=90 EXTRA_ARGS="--reset-lr" bash scripts/train_resilient.sh
set -uo pipefail
cd "$(dirname "$0")/.."

VERSION="${VERSION:?Set VERSION, e.g. VERSION=17.44}"
CUDA="${CUDA:-0}"
INITIAL_RESUME="${INITIAL_RESUME:-}"
INITIAL_SKIP_OPTIMIZER="${INITIAL_SKIP_OPTIMIZER:-0}"
MAX_RESTARTS="${MAX_RESTARTS:-9999}"
SLEEP_SEC="${SLEEP_SEC:-60}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

VERSION_PATH="${VERSION//./_}"
CKPT_DIR="checkpoints/v${VERSION_PATH}"
mkdir -p "$CKPT_DIR" logs/resilient

run_id="v${VERSION_PATH}_cuda${CUDA}_$(date +%Y%m%d_%H%M%S)"
wrapper_log="logs/resilient/${run_id}.log"

has_checkpoint() {
  [[ -f "$CKPT_DIR/checkpoint_latest.pth" ]] && return 0
  compgen -G "$CKPT_DIR/checkpoint_iter_*.pth" >/dev/null && return 0
  return 1
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$now] attempt=$attempt version=$VERSION cuda=$CUDA ckpt_dir=$CKPT_DIR" | tee -a "$wrapper_log"

  cmd=(bash scripts/train.sh --version "$VERSION" --cuda "$CUDA")
  if has_checkpoint; then
    cmd+=(--auto-resume)
  else
    if [[ -z "$INITIAL_RESUME" ]]; then
      echo "ERROR: no checkpoint found in $CKPT_DIR and INITIAL_RESUME is not set" | tee -a "$wrapper_log"
      exit 2
    fi
    cmd+=(--resume "$INITIAL_RESUME")
    if [[ "$INITIAL_SKIP_OPTIMIZER" == "1" ]]; then
      cmd+=(--skip-optimizer)
    fi
  fi

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra=( $EXTRA_ARGS )
    cmd+=("${extra[@]}")
  fi

  echo "+ ${cmd[*]}" | tee -a "$wrapper_log"
  "${cmd[@]}" 2>&1 | tee -a "$wrapper_log"
  status=${PIPESTATUS[0]}

  now="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ "$status" -eq 0 ]]; then
    echo "[$now] training exited cleanly; stopping resilient wrapper" | tee -a "$wrapper_log"
    exit 0
  fi

  echo "[$now] training exited status=$status" | tee -a "$wrapper_log"
  if [[ "$attempt" -ge "$MAX_RESTARTS" ]]; then
    echo "[$now] reached MAX_RESTARTS=$MAX_RESTARTS; giving up" | tee -a "$wrapper_log"
    exit "$status"
  fi

  echo "[$now] sleeping ${SLEEP_SEC}s before retry" | tee -a "$wrapper_log"
  sleep "$SLEEP_SEC"
done
