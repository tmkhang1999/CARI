#!/usr/bin/env bash
# Resilient Stage-B launcher: fresh Adam on the FIRST launch, own optimizer state on
# every automatic restart.
#
# WHY THE TWO CASES DIFFER
# ------------------------
# Stage A (phase 1 + 2) trains on hypersim/interiorverse with no MID pairs, so the
# cross-render losses never fire. Stage B (phase 3) changes the data mix AND activates
# L_inv / L_explain / L_chr_explain. Adam's moment estimates carried over from Stage A
# are estimated for a different objective, so the FIRST launch into Stage B passes
# --skip-optimizer and starts Adam clean.
#
# An interruption mid-Stage-B is a different situation: the objective has not changed,
# and resetting Adam every time the job is killed would repeatedly re-disrupt training.
# So restarts DO load the run's own optimizer state (no --skip-optimizer), which is why
# the row configs keep `skip_optimizer: false` -- train_v17.py:1711 ORs the CLI flag with
# the config value, so config false + CLI flag on first launch gives exactly this.
#
# WHY THIS MATTERS FOR THE ABLATION
# ---------------------------------
# The original Table A was invalidated because rows differed in optimizer treatment: a
# silent fresh-Adam fallback (train_v17.py:299-307 catches a param-group mismatch, warns,
# and continues with a fresh optimizer) gave the colour-ON rows ~1.85x the effective LR
# of the colour-OFF rows. Every row must therefore take the SAME path. This script makes
# that path explicit, and greps each launch for the silent-fallback warning so a row that
# quietly reset its optimizer cannot pass unnoticed.
#
# skip_optimizer does NOT affect the learning rate. A fresh optimizer takes initial_lr
# from the config and CosineAnnealingLR recomputes the schedule from
# last_epoch=completed_opt_steps-1 (train_v17.py:1749-1755); a restart restores the saved
# initial_lr. Both give the same LR at the same step. Only Adam's moments differ.
#
# USAGE
#   INITIAL_RESUME=<fork ckpt> VERSION=17.61 CUDA=0 bash scripts/train_resilient.sh
#
# ENV
#   VERSION          required, e.g. 17.61 (dots are converted to underscores)
#   INITIAL_RESUME   checkpoint the FIRST launch resumes from (the Stage-A fork)
#   CUDA             GPU index (default 0)
#   MAX_RETRIES      restart attempts after a non-zero exit (default 20)
#   SLEEP_SEC        pause between restarts (default 60)
#   LOG_DIR          where to write the run log (default /tmp/cari_runs)
#   SKIP_INITIAL     set to 0 to NOT pass --skip-optimizer on the first launch
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="${VERSION:?set VERSION, e.g. VERSION=17.61}"
TAG="${VERSION//./_}"
CUDA="${CUDA:-0}"
MAX_RETRIES="${MAX_RETRIES:-20}"
SLEEP_SEC="${SLEEP_SEC:-60}"
LOG_DIR="${LOG_DIR:-/tmp/cari_runs}"
SKIP_INITIAL="${SKIP_INITIAL:-1}"
INITIAL_RESUME="${INITIAL_RESUME:-}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_v${TAG}.log"
PY="${PY:-/home/khang/miniconda3/envs/IR/bin/python}"
CFG="$ROOT/src/configs/v${TAG}.yaml"
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG"; exit 2; }

echo "===== v$TAG | GPU $CUDA | log $LOG ====="

attempt=0
while :; do
  if [ "$attempt" -eq 0 ]; then
    # FIRST launch: resume model weights from the Stage-A fork, fresh Adam.
    [ -n "$INITIAL_RESUME" ] || { echo "ERROR: INITIAL_RESUME required for the first launch"; exit 2; }
    [ -f "$INITIAL_RESUME" ] || { echo "ERROR: fork checkpoint missing: $INITIAL_RESUME"; exit 2; }
    ARGS=(--version "$TAG" --config "$CFG" --device cuda --resume "$INITIAL_RESUME")
    [ "$SKIP_INITIAL" = "1" ] && ARGS+=(--skip-optimizer)
    echo "--- launch 0 (fresh Adam=$SKIP_INITIAL) from $INITIAL_RESUME  $(date +%H:%M) ---"
  else
    # RESTART: continue this row, keeping its own optimizer state.
    ARGS=(--version "$TAG" --config "$CFG" --device cuda --auto-resume)
    echo "--- restart $attempt (own optimizer state)  $(date +%H:%M) ---"
  fi

  CUDA_VISIBLE_DEVICES="$CUDA" "$PY" "$ROOT/src/train_v17.py" "${ARGS[@]}" >>"$LOG" 2>&1
  rc=$?

  # A quietly-reset optimizer would silently confound this row against the others.
  if grep -q "optimizer state not loaded" "$LOG"; then
    echo "!!! SILENT FRESH-ADAM FALLBACK DETECTED in $LOG -- this row is not comparable."
    echo "!!! Stopping rather than producing a confounded result."
    exit 3
  fi

  if [ $rc -eq 0 ]; then
    echo "===== v$TAG COMPLETE $(date +%H:%M) ====="
    break
  fi
  attempt=$((attempt+1))
  if [ "$attempt" -gt "$MAX_RETRIES" ]; then
    echo "===== v$TAG GAVE UP after $MAX_RETRIES retries (last rc=$rc) ====="
    exit $rc
  fi
  echo "--- exit $rc; retrying in ${SLEEP_SEC}s ---"
  sleep "$SLEEP_SEC"
done

tr '\r' '\n' <"$LOG" | grep -oE "[0-9]+/[0-9]+ \[[^]]*\]" | tail -1
