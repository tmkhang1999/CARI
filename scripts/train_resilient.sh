#!/usr/bin/env bash
# Resilient Stage-B launcher: fresh Adam on the FIRST launch, own optimizer state on
# every automatic restart.
#
# WHY THE TWO CASES DIFFER
# ------------------------
# Stage A (phase 1 + 2) trains on hypersim/interiorverse with no MID pairs, so the
# CIAI pair losses never fire. Stage B (phase 3) changes the data mix AND activates
# L_inv / L_explain. Adam's moment estimates carried over from Stage A
# are estimated for a different objective, so the FIRST launch into Stage B passes
# --skip-optimizer and starts Adam clean.
#
# An interruption mid-Stage-B is a different situation: the objective has not changed,
# and resetting Adam every time the job is killed would repeatedly re-disrupt training.
# So restarts DO load the run's own optimizer state (no --skip-optimizer), which is why
# the row configs keep `skip_optimizer: false` -- train.py ORs the CLI flag with
# the config value, so config false + CLI flag on first launch gives exactly this.
#
# WHY THIS MATTERS FOR THE ABLATION
# ---------------------------------
# An earlier version of the ablation was invalidated because rows differed in optimizer treatment: a
# silent fresh-Adam fallback (load_checkpoint in train.py catches a param-group mismatch, warns,
# and continues with a fresh optimizer) gave the colour-ON rows ~1.85x the effective LR
# of the colour-OFF rows. Every row must therefore take the SAME path. This script makes
# that path explicit, and greps each launch for the silent-fallback warning so a row that
# quietly reset its optimizer cannot pass unnoticed.
#
# skip_optimizer does NOT affect the learning rate. A fresh optimizer takes initial_lr
# from the config and CosineAnnealingLR recomputes the schedule from
# last_epoch=completed_opt_steps-1 (train.py main); a restart restores the saved
# initial_lr. Both give the same LR at the same step. Only Adam's moments differ.
#
# HOW TO STOP A RUN  -- READ THIS BEFORE KILLING ANYTHING
# ------------------------------------------------------
# This script RESTARTS training whenever the python process exits non-zero. So
# `pkill -f train.py` does not stop a run -- it triggers a restart 60s later.
# That already caused one incident: two wrappers thought to be dead relaunched
# themselves onto both GPUs alongside a new pair of runs, putting two ~11 GiB
# processes on one 23.6 GiB card and OOM-ing all four.
#
# Kill the WRAPPER first, then the python:
#     pkill -f train_resilient.sh && sleep 3 && pkill -f train.py
# Or drop a stop file, which makes the wrapper exit cleanly after the current
# attempt instead of retrying:
#     touch /tmp/ciai_runs/STOP_ciai_s42
# Always confirm BOTH are gone before launching anything new:
#     ps aux | grep -c '[t]rain_resilient'   # must be 0
#     ps aux | grep -c '[t]rain.py'          # must be 0
#
# USAGE
#   INITIAL_RESUME=<fork ckpt> CONFIG=ciai CUDA=0 bash scripts/train_resilient.sh
#   SEED=43 INITIAL_RESUME=<fork ckpt> CONFIG=ablation_ciai CUDA=0 bash scripts/train_resilient.sh
#
# ENV
#   CONFIG           required, a config name under src/configs, e.g. ciai
#   INITIAL_RESUME   checkpoint the FIRST launch resumes from (the Stage-A fork)
#   CUDA             GPU index (default 0)
#   MAX_RETRIES      restart attempts after a non-zero exit (default 20)
#   SLEEP_SEC        pause between restarts (default 60)
#   LOG_DIR          where to write the run log (default /tmp/ciai_runs)
#   SKIP_INITIAL     set to 0 to NOT pass --skip-optimizer on the first launch
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG_TAG="${CONFIG:?set CONFIG, e.g. CONFIG=ciai}"
# SEED selects a replicate. The config is shared across replicates -- only --seed
# differs -- so the RUN tag carries the seed and each replicate gets its own
# checkpoint/log directory. Without this, two seeds of one config would write to the
# same checkpoint directory and --auto-resume would silently continue the WRONG run.
SEED="${SEED:-}"
if [ -n "$SEED" ]; then TAG="${CFG_TAG}_s${SEED}"; else TAG="$CFG_TAG"; fi
CUDA="${CUDA:-0}"
MAX_RETRIES="${MAX_RETRIES:-20}"
SLEEP_SEC="${SLEEP_SEC:-60}"
LOG_DIR="${LOG_DIR:-/tmp/ciai_runs}"
SKIP_INITIAL="${SKIP_INITIAL:-1}"
INITIAL_RESUME="${INITIAL_RESUME:-}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_${TAG}.log"
PY="${PY:-python}"
CFG="$ROOT/src/configs/${CFG_TAG}.yaml"
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG"; exit 2; }

# Where the run writes checkpoints (config paths.checkpoint_dir), read here so the
# archive step on success knows where to look. Falls back to the repo default.
CKPT_DIR="$("$PY" - "$CFG" <<'PYEOF' 2>/dev/null || echo ""
import sys, yaml, os
d = yaml.safe_load(open(sys.argv[1])) or {}
print((d.get('paths') or {}).get('checkpoint_dir', ''))
PYEOF
)"
[ -n "$CKPT_DIR" ] || CKPT_DIR="$ROOT/checkpoints"

echo "===== $TAG | cfg $CFG_TAG | seed ${SEED:-<unseeded>} | GPU $CUDA | log $LOG ====="

# PREFLIGHT: can this filesystem actually take a checkpoint?
#
# df cannot answer this on quota'd or session-capped filesystems: it can report hundreds
# of GB free while a 1.4 GB checkpoint save fails, which once killed runs at their first
# checkpoint with no traceback.
#
# So probe it for real: write a checkpoint-sized file and delete it. Costs seconds
# before a 10-hour run.
mkdir -p "$CKPT_DIR"
PROBE="$CKPT_DIR/.space_probe.$$"
if ! dd if=/dev/zero of="$PROBE" bs=1M count=3000 status=none 2>/dev/null; then
  rm -f "$PROBE"
  echo "ERROR: $CKPT_DIR cannot take a 3 GB write (a checkpoint is ~1.43 GB and a save"
  echo "       needs room for the transient .tmp alongside the retained copies)."
  echo "       df is NOT a valid check here -- quota and session caps are invisible to it."
  echo "       Free space before launching, e.g.:  du -sh $CKPT_DIR/*"
  exit 5
fi
rm -f "$PROBE"
echo "preflight: $CKPT_DIR accepted a 3 GB write"

attempt=0
while :; do
  if [ "$attempt" -eq 0 ]; then
    # FIRST launch: resume model weights from the Stage-A fork, fresh Adam.
    [ -n "$INITIAL_RESUME" ] || { echo "ERROR: INITIAL_RESUME required for the first launch"; exit 2; }
    [ -f "$INITIAL_RESUME" ] || { echo "ERROR: fork checkpoint missing: $INITIAL_RESUME"; exit 2; }
    ARGS=(--config "$CFG_TAG" --run-name "$TAG" --device cuda --resume "$INITIAL_RESUME")
    [ "$SKIP_INITIAL" = "1" ] && ARGS+=(--skip-optimizer)
    [ -n "$SEED" ] && ARGS+=(--seed "$SEED")
    echo "--- launch 0 (fresh Adam=$SKIP_INITIAL) from $INITIAL_RESUME  $(date +%H:%M) ---"
  else
    # RESTART: continue this row, keeping its own optimizer state.
    ARGS=(--config "$CFG_TAG" --run-name "$TAG" --device cuda --auto-resume)
    [ -n "$SEED" ] && ARGS+=(--seed "$SEED")
    echo "--- restart $attempt (own optimizer state)  $(date +%H:%M) ---"
  fi

  CUDA_VISIBLE_DEVICES="$CUDA" "$PY" "$ROOT/src/train.py" "${ARGS[@]}" >>"$LOG" 2>&1
  rc=$?

  # A quietly-reset optimizer would silently confound this row against the others.
  if grep -q "optimizer state not loaded" "$LOG"; then
    echo "!!! SILENT FRESH-ADAM FALLBACK DETECTED in $LOG -- this row is not comparable."
    echo "!!! Stopping rather than producing a confounded result."
    exit 3
  fi

  if [ $rc -eq 0 ]; then
    echo "===== $TAG COMPLETE $(date +%H:%M) ====="
    # Archiving is OPT-IN (ARCHIVE=1), because there is nowhere to put 18 checkpoints.
    # Measured headroom: home ~3 GB, scratchpad ~16 GB total. The study's 18 finals
    # would be ~26 GB, so keeping them all is not an option on this machine.
    #
    # The workflow is therefore EVALUATE-THEN-DISCARD: when a round finishes, evaluate
    # its rows, keep the metrics JSON (kilobytes), and free the checkpoints before the
    # next round. Only the two headline rows are worth archiving for figures, and home
    # has just about room for those.
    CKPT_SRC="$(ls -1t "$CKPT_DIR/$TAG"/checkpoint_iter_*.pth 2>/dev/null | head -1)"
    if [ -z "$CKPT_SRC" ]; then
      echo "!!! no checkpoint found in $CKPT_DIR/$TAG"
    elif [ "${ARCHIVE:-0}" = "1" ]; then
      mkdir -p "$ROOT/checkpoints/$TAG"
      if cp "$CKPT_SRC" "$ROOT/checkpoints/$TAG/"; then
        echo "archived $(basename "$CKPT_SRC") -> checkpoints/$TAG/"
      else
        echo "!!! ARCHIVE FAILED for $TAG (quota). Checkpoint is ONLY at $CKPT_SRC"
      fi
    else
      echo "final checkpoint: $CKPT_SRC"
      echo "  (not archived; set ARCHIVE=1 to copy to checkpoints/$TAG/ -- home has ~3 GB)"
      echo "  EVALUATE THIS ROW AND FREE IT before launching the next round."
    fi
    break
  fi

  # Deliberate stop: exit instead of restarting. Without this the only way to stop
  # a run is to kill the wrapper, and killing the python alone RESTARTS it.
  if [ -f "$LOG_DIR/STOP_$TAG" ]; then
    echo "===== $TAG STOPPED by $LOG_DIR/STOP_$TAG (rc=$rc) ====="
    exit 0
  fi

  # Do not retry into an OOM: a second process on the same GPU is the usual cause,
  # and retrying 20 times just fights whatever else is resident. Surface it instead.
  if tr '\r' '\n' <"$LOG" | tail -50 | grep -q "OutOfMemoryError"; then
    echo "!!! CUDA OOM in $LOG -- not retrying."
    echo "!!! Check for another process on GPU $CUDA:  nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    exit 4
  fi

  attempt=$((attempt+1))
  if [ "$attempt" -gt "$MAX_RETRIES" ]; then
    echo "===== $TAG GAVE UP after $MAX_RETRIES retries (last rc=$rc) ====="
    exit $rc
  fi
  echo "--- exit $rc; retrying in ${SLEEP_SEC}s ---"
  sleep "$SLEEP_SEC"
done

tr '\r' '\n' <"$LOG" | grep -oE "[0-9]+/[0-9]+ \[[^]]*\]" | tail -1
