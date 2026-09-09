#!/usr/bin/env bash
# run-chain-small-medium-large.sh — wait for the in-flight small run,
# then run medium, then run large. One script so the whole sweep
# survives session disconnects under nohup.
#
# Usage:
#   PROJECT=jbonneau-mwalfish-9a0c \
#     nohup ./scripts/run-chain-small-medium-large.sh > /tmp/chain.log 2>&1 &
#   disown
#
# Env:
#   PROJECT       GCP project (required)
#   WAIT_FOR_PID  PID of an already-running small re-run to wait on
#                 (default: 1492610 — the one launched manually).
#                 Set to 0 to skip waiting and start small from scratch.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${PROJECT:-}" ]]; then
  echo "ERROR: PROJECT env var is required" >&2
  exit 2
fi
export PROJECT

WAIT_FOR_PID="${WAIT_FOR_PID:-1492610}"

log() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] CHAIN: $*"
}

# --- Stage 1: small ---
if [[ "$WAIT_FOR_PID" != "0" ]] && kill -0 "$WAIT_FOR_PID" 2>/dev/null; then
  log "waiting for in-flight small (pid=$WAIT_FOR_PID)"
  # `wait` only works on direct children; poll instead.
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 60
  done
  log "small (pid=$WAIT_FOR_PID) exited"
else
  log "no in-flight small detected (WAIT_FOR_PID=$WAIT_FOR_PID); launching fresh"
  "$REPO_ROOT/scripts/run-small-cluster.sh" \
    || log "WARN: small run failed (continuing to medium)"
fi

# Tiny pause so GCP backend settles after small's teardown.
sleep 30

# --- Stage 2: medium ---
log "starting medium"
if "$REPO_ROOT/scripts/run-medium-cluster.sh"; then
  log "medium OK"
else
  log "WARN: medium run failed (continuing to large)"
fi

sleep 30

# --- Stage 3: large ---
# N_MASKING_SERVERS=35 and BENCH_CLIENT_MACHINE_TYPE=n2-highmem-32 are
# set as defaults in run-cluster-suite.sh — no overrides needed here.
log "starting large"
if "$REPO_ROOT/scripts/run-cluster-suite.sh"; then
  log "large OK"
else
  log "WARN: large run failed"
fi

log "==== CHAIN DONE ===="
