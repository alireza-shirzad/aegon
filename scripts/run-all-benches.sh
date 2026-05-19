#!/usr/bin/env bash
# run-all-benches.sh — sequential driver for the local subset of the
# benchmark suite. Runs setup, publish, then lookup for the small
# regime, plus a per-shard SRS size measurement at the medium/large
# shared shard config (shard_log_capacity=27, kzh_k=9). Outputs land
# in bench-results/{setup,publish,lookup}/.
#
# Medium (2-shard) and large (128-shard) timing data come from the
# cluster — see scripts/run-cluster-suite.sh and
# scripts/run-medium-cluster.sh.
#
# Tunables (env): forwarded to the underlying drivers.
#   SKIP_SMALL=1        skip the small regime everywhere
#   SKIP_PER_SHARD=1    skip the per-shard SRS size measurement
#   SKIP_SETUP=1        skip setup-bench entirely
#   SKIP_PUBLISH=1      skip publish-bench entirely
#   SKIP_LOOKUP=1       skip lookup-bench entirely

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/bench-results"
LOG_FILE="$RESULTS_DIR/run-all.log"
mkdir -p "$RESULTS_DIR/setup" "$RESULTS_DIR/publish" "$RESULTS_DIR/lookup"

log() {
  local msg
  msg="[$(date +'%Y-%m-%dT%H:%M:%S%z')] $*"
  echo "$msg" | tee -a "$LOG_FILE"
}

run_phase() {
  local phase="$1"
  shift
  local t0
  t0="$(date +%s)"
  log "==== $phase START ===="
  if "$@"; then
    local t1; t1="$(date +%s)"
    local dur=$((t1 - t0))
    log "==== $phase OK in ${dur}s ===="
  else
    local rc=$?
    local t1; t1="$(date +%s)"
    local dur=$((t1 - t0))
    log "==== $phase FAILED rc=$rc after ${dur}s ===="
    return "$rc"
  fi
}

log "host: $(hostname) cores=$(nproc) mem_total=$(awk '/MemTotal/ {print $2}' /proc/meminfo) KB"
log "results dir: $RESULTS_DIR"

# --- Phase 1: setup-bench -----------------------------------------------
if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  run_phase "setup-bench" env \
    OUT_DIR="$RESULTS_DIR/setup" \
    SKIP_SMALL="${SKIP_SMALL:-0}" \
    SKIP_PER_SHARD="${SKIP_PER_SHARD:-0}" \
    "$REPO_ROOT/scripts/setup-bench.sh"
fi

# --- Phase 2: publish-bench --------------------------------------------
if [[ "${SKIP_PUBLISH:-0}" != "1" ]]; then
  run_phase "publish-bench" env \
    OUT_DIR="$RESULTS_DIR/publish" \
    SKIP_SMALL="${SKIP_SMALL:-0}" \
    "$REPO_ROOT/scripts/publish-bench.sh"
fi

# --- Phase 3: lookup-bench ---------------------------------------------
if [[ "${SKIP_LOOKUP:-0}" != "1" ]]; then
  run_phase "lookup-bench" env \
    OUT_DIR="$RESULTS_DIR/lookup" \
    SKIP_SMALL="${SKIP_SMALL:-0}" \
    "$REPO_ROOT/scripts/lookup-bench.sh"
fi

log "==== ALL DONE ===="
log "JSON outputs:"
find "$RESULTS_DIR" -type f -name '*.json' -printf '  %p (%s bytes)\n' | tee -a "$LOG_FILE"
