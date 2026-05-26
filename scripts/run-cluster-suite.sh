#!/usr/bin/env bash
# run-cluster-suite.sh — drive the full GCP cluster benchmark
# pipeline end-to-end:
#   1. provision 128 shard VMs + coordinator + Redis
#   2. build + deploy binaries
#   3. distributed SRS bootstrap (also exercises setup-bench)
#   4. setup-bench (gather distributed-setup metrics)
#   5. publish-bench (walks fill_percents)
#   6. lookup-bench (lookup + publish + audit, walks fill_percents)
#   7. tear down all VMs
#
# Outputs all JSONs into bench-results/{setup,publish,lookup}/ on the
# local host, with the cluster artifacts named *_cluster.json so they
# don't collide with the small/medium local files.
#
# Env vars:
#   PROJECT    GCP project ID (required)
#   ZONE       GCE zone (default us-central1-f)
#   N_SHARDS   shard count (default 128 — set by bench-cluster.sh)
#   *_SKIP=1   skip phase: SKIP_UP, SKIP_DEPLOY, SKIP_BOOTSTRAP,
#              SKIP_SETUP_BENCH, SKIP_PUBLISH_BENCH, SKIP_LOOKUP_BENCH,
#              SKIP_DOWN

set -uo pipefail   # not -e: we want each phase logged even on failure

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/bench-results"
LOG_FILE="$RESULTS_DIR/run-cluster.log"
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

log "host: $(hostname)"
log "PROJECT=$PROJECT"
log "results dir: $RESULTS_DIR"

# Each phase forwards PROJECT (and optionally PUBLISH_FILL_PERCENTS etc.)
# via env. bench-cluster.sh reads its env vars from `export` lines or
# the parent shell's environment.
export PROJECT
# Point cluster-driver outputs at our bench-results/{publish,lookup}
# directories so the JSONs land in the repo, not /tmp.
export LOCAL_SETUP_BENCH_OUT="$RESULTS_DIR/setup/large_cluster.json"
export LOCAL_PUBLISH_BENCH_DIR="$RESULTS_DIR/publish"
export LOCAL_LOOKUP_BENCH_DIR="$RESULTS_DIR/lookup"
# Prefix per-fill JSON filenames so they don't collide with the
# medium 2-shard cluster run (run-medium-cluster.sh uses "medium-").
export LOCAL_OUT_NAME_PREFIX="${LOCAL_OUT_NAME_PREFIX:-large-}"
# Large warmup batch — at 30% of 2^32 = ~1.3B entries, warmup
# needs ~10k publishes at 131072 to finish in reasonable time.
export PUBLISH_WARMUP_BATCH_SIZE="${PUBLISH_WARMUP_BATCH_SIZE:-131072}"

# --- Phase 1: up ---
if [[ "${SKIP_UP:-0}" != "1" ]]; then
  run_phase "cluster-up" "$REPO_ROOT/scripts/bench-cluster.sh" up || exit $?
fi

# --- Phase 2: deploy ---
if [[ "${SKIP_DEPLOY:-0}" != "1" ]]; then
  run_phase "cluster-deploy" "$REPO_ROOT/scripts/bench-cluster.sh" deploy || exit $?
fi

# --- Phase 3: setup-bench (centralized-SRS gen + broadcast + size metrics) ---
# This replaces the old distributed-SRS bootstrap. setup-bench:
#   1. Generates the SRS on shard-0.
#   2. scp's it to every other shard at $REMOTE_SRS_PATH.
#   3. Emits {gen_seconds, broadcast_seconds, srs_bytes} JSON.
# publish-bench + lookup-bench restart shards per-stage with
# --srs-path, so we don't need a separate "bootstrap"/"start" step.
if [[ "${SKIP_SETUP_BENCH:-0}" != "1" ]]; then
  run_phase "cluster-setup-bench" "$REPO_ROOT/scripts/bench-cluster.sh" setup-bench || exit $?
fi

# --- Phase 4a: start-masking (background masking server prebuilds packages) ---
# Must happen before start-shards: shards boot with --masking-addr and
# their first value-side opening will fetch_package from the masking
# server, so the masking server has to be listening already.
if [[ "${SKIP_START_MASKING:-0}" != "1" && "${ENABLE_MASKING_SERVER:-1}" == "1" ]]; then
  run_phase "cluster-start-masking" "$REPO_ROOT/scripts/bench-cluster.sh" start-masking || exit $?
fi

# --- Phase 4b: start-shards (once; bench binaries drive prefill via RPC) ---
if [[ "${SKIP_START_SHARDS:-0}" != "1" ]]; then
  run_phase "cluster-start-shards" "$REPO_ROOT/scripts/bench-cluster.sh" start-shards || exit $?
fi

# --- Phase 5: publish-bench (walks fills) ---
if [[ "${SKIP_PUBLISH_BENCH:-0}" != "1" ]]; then
  run_phase "cluster-publish-bench" "$REPO_ROOT/scripts/bench-cluster.sh" publish-bench \
    || log "WARN: publish-bench failed; continuing with lookup-bench"
fi

# --- Phase 6: lookup-bench (walks fills) ---
if [[ "${SKIP_LOOKUP_BENCH:-0}" != "1" ]]; then
  run_phase "cluster-lookup-bench" "$REPO_ROOT/scripts/bench-cluster.sh" lookup-bench \
    || log "WARN: lookup-bench failed; tearing down anyway"
fi

# --- Phase 7: down (always run unless explicitly skipped) ---
if [[ "${SKIP_DOWN:-0}" != "1" ]]; then
  run_phase "cluster-down" "$REPO_ROOT/scripts/bench-cluster.sh" down \
    || log "WARN: down failed; some VMs may still be running — verify in console"
fi

log "==== ALL DONE ===="
log "JSON outputs:"
find "$RESULTS_DIR" -name '*cluster*.json' -o -name 'lookup-*.json' -o -name 'publish-*.json' \
    | sort | tee -a "$LOG_FILE"
