#!/usr/bin/env bash
# run-medium-cluster.sh — drive the medium-regime cluster benchmark
# (2-shard cluster, same per-shard config as the large regime).
#
# Sequential to scripts/run-cluster-suite.sh — they share the
# bench-cluster.sh resource pool (VPC, VM names), so run them
# back-to-back, not in parallel.
#
# Medium regime sizing (matches sections/implementation.tex):
#   N_SHARDS=2, SHARD_LOG_CAPACITY=27, KZH_K=9
#   true dictionary capacity = 2 × 2^25 = 2^26 entries
#   total polynomial capacity = 2 × 2^27 = 2^28 slots
#   over-provisioning factor α = 4 (see akd/src/aegon/config.rs)
#   publish batch sizes: 64,128,256,512,1024,2048
#
# Outputs land at:
#   bench-results/setup/medium_cluster.json
#   bench-results/publish/medium_fill{0,30,60,90}.json
#   bench-results/lookup/medium_fill{0,30,60,90}.json
#
# Env vars: same as run-cluster-suite.sh — PROJECT is required,
# everything else has medium-appropriate defaults set below.

set -uo pipefail   # not -e: each phase should log on failure

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/bench-results"
LOG_FILE="$RESULTS_DIR/run-medium-cluster.log"
mkdir -p "$RESULTS_DIR/setup" "$RESULTS_DIR/publish" "$RESULTS_DIR/lookup"

if [[ -z "${PROJECT:-}" ]]; then
  echo "ERROR: PROJECT env var is required (export PROJECT=your-gcp-project-id)" >&2
  exit 2
fi

log() {
  local msg
  msg="[$(date +'%Y-%m-%dT%H:%M:%S%z')] $*"
  echo "$msg" | tee -a "$LOG_FILE"
}

run_phase() {
  local phase="$1"
  shift
  local t0; t0="$(date +%s)"
  log "==== $phase START ===="
  if "$@" >>"$LOG_FILE" 2>&1; then
    local t1; t1="$(date +%s)"
    local dur=$((t1 - t0))
    log "==== $phase OK in ${dur}s ===="
    return 0
  else
    local rc=$?
    local t1; t1="$(date +%s)"
    local dur=$((t1 - t0))
    log "==== $phase FAILED rc=$rc after ${dur}s ===="
    return "$rc"
  fi
}

# Medium-regime cluster config — passed through to bench-cluster.sh.
export PROJECT
export N_SHARDS="${N_SHARDS:-2}"
export SHARD_LOG_CAPACITY="${SHARD_LOG_CAPACITY:-27}"
export KZH_K="${KZH_K:-9}"
# Medium batch sizes; large uses 4096..131072.
export PUBLISH_BATCH_SIZES="${PUBLISH_BATCH_SIZES:-64,128,256,512,1024,2048}"
export LOOKUP_PUBLISH_BATCH_SIZES="${LOOKUP_PUBLISH_BATCH_SIZES:-${PUBLISH_BATCH_SIZES}}"

# Outputs — distinct filenames so medium and large don't overwrite
# each other in the same results dirs.
export LOCAL_SETUP_BENCH_OUT="$RESULTS_DIR/setup/medium_cluster.json"
export LOCAL_PUBLISH_BENCH_DIR="$RESULTS_DIR/publish"
export LOCAL_LOOKUP_BENCH_DIR="$RESULTS_DIR/lookup"
# bench-cluster.sh writes ${LOCAL_OUT_NAME_PREFIX}publish-${pct}pct.json and
# ${LOCAL_OUT_NAME_PREFIX}lookup-${pct}pct.json. We prefix with "medium-"
# so files don't collide with the large run.
export LOCAL_OUT_NAME_PREFIX="${LOCAL_OUT_NAME_PREFIX:-medium-}"

log "host: $(hostname)"
log "PROJECT=$PROJECT  N_SHARDS=$N_SHARDS  SHARD_LOG_CAPACITY=$SHARD_LOG_CAPACITY  KZH_K=$KZH_K"
log "PUBLISH_BATCH_SIZES=$PUBLISH_BATCH_SIZES"
log "results dir: $RESULTS_DIR"

# --- Phase 1: up ---
if [[ "${SKIP_UP:-0}" != "1" ]]; then
  run_phase "medium-up" "$REPO_ROOT/scripts/bench-cluster.sh" up || exit $?
fi

# --- Phase 2: deploy ---
if [[ "${SKIP_DEPLOY:-0}" != "1" ]]; then
  run_phase "medium-deploy" "$REPO_ROOT/scripts/bench-cluster.sh" deploy || exit $?
fi

# --- Phase 3: bootstrap ---
if [[ "${SKIP_BOOTSTRAP:-0}" != "1" ]]; then
  run_phase "medium-bootstrap" "$REPO_ROOT/scripts/bench-cluster.sh" bootstrap || exit $?
fi

# --- Phase 4: setup-bench ---
if [[ "${SKIP_SETUP_BENCH:-0}" != "1" ]]; then
  run_phase "medium-setup-bench" "$REPO_ROOT/scripts/bench-cluster.sh" setup-bench \
    || log "WARN: setup-bench failed; continuing with publish-bench"
fi

# --- Phase 5: publish-bench ---
if [[ "${SKIP_PUBLISH_BENCH:-0}" != "1" ]]; then
  run_phase "medium-publish-bench" "$REPO_ROOT/scripts/bench-cluster.sh" publish-bench \
    || log "WARN: publish-bench failed; continuing with lookup-bench"
fi

# --- Phase 6: lookup-bench ---
if [[ "${SKIP_LOOKUP_BENCH:-0}" != "1" ]]; then
  run_phase "medium-lookup-bench" "$REPO_ROOT/scripts/bench-cluster.sh" lookup-bench \
    || log "WARN: lookup-bench failed; tearing down anyway"
fi

# --- Phase 7: down ---
if [[ "${SKIP_DOWN:-0}" != "1" ]]; then
  run_phase "medium-down" "$REPO_ROOT/scripts/bench-cluster.sh" down \
    || log "WARN: down failed; some VMs may still be running — verify in console"
fi

log "==== MEDIUM CLUSTER DONE ===="
log "JSON outputs:"
find "$RESULTS_DIR" -name 'medium*.json' | sort | tee -a "$LOG_FILE"
