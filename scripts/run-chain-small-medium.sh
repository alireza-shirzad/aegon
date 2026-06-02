#!/usr/bin/env bash
# run-chain-small-medium.sh — small (1 shard) → medium (2 shards),
# both with round-robin masking. Survives session disconnects under
# nohup. Use after the Rust round-robin fix is built.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${PROJECT:-}" ]]; then
  echo "ERROR: PROJECT env var is required" >&2
  exit 2
fi
export PROJECT

log() { echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] CHAIN: $*"; }

log "starting small (1 shard, round-robin masking)"
if "$REPO_ROOT/scripts/run-small-cluster.sh"; then
  log "small OK"
else
  log "WARN: small run failed (continuing to medium)"
fi

sleep 30

log "starting medium (2 shards, round-robin masking)"
if "$REPO_ROOT/scripts/run-medium-cluster.sh"; then
  log "medium OK"
else
  log "WARN: medium run failed"
fi

log "==== CHAIN DONE ===="
