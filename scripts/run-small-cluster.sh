#!/usr/bin/env bash
# run-small-cluster.sh — drive the small-regime cluster benchmark
# (1-shard cluster). Identical code path to run-medium-cluster.sh —
# only the per-shard capacity, kzh_k, batch sizes, and shard count
# differ. Same bench-cluster.sh phases, same real-publish prefill
# (no bulk-prefill shortcut), so history records are populated and
# the history-proof bytes are directly comparable to medium/large.
#
# Small regime sizing:
#   N_SHARDS=1, SHARD_LOG_CAPACITY=22, KZH_K=7
#   true dictionary capacity = 1 × 2^20 = 2^20 entries (~1M)
#   total polynomial capacity = 1 × 2^22 = 2^22 slots
#   over-provisioning factor α = 4 (see akd/src/aegon/config.rs)
#   publish batch sizes: 2,4,8,16,32,64
#
# Outputs land at:
#   bench-results/setup/small_cluster.json
#   bench-results/publish/small-publish-{1,30,60,90}pct.json
#   bench-results/lookup/small-combined.json
#
# Env vars: PROJECT is required, everything else has small defaults.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/bench-results"
LOG_FILE="$RESULTS_DIR/run-small-cluster.log"
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

# Small-regime cluster config — passed through to bench-cluster.sh.
export PROJECT
export N_SHARDS="${N_SHARDS:-1}"
export SHARD_LOG_CAPACITY="${SHARD_LOG_CAPACITY:-22}"
export KZH_K="${KZH_K:-7}"
# Small batch sizes — the per-batch MSM work is tiny at log_cap=22
# so the sweet spot is much smaller than medium's 64..2048.
export PUBLISH_BATCH_SIZES="${PUBLISH_BATCH_SIZES:-2,4,8,16,32,64}"
export LOOKUP_PUBLISH_BATCH_SIZES="${LOOKUP_PUBLISH_BATCH_SIZES:-${PUBLISH_BATCH_SIZES}}"
# Warmup batch sized for small: 1024 was the medium-mode batch_size
# in earlier local-mode small runs. Keeps cache-resident at log_cap=22.
export PUBLISH_WARMUP_BATCH_SIZE="${PUBLISH_WARMUP_BATCH_SIZE:-1024}"
# 4 masking servers so the throughput-sweep curve shows a real climb
# phase. One masking server (185 pkg/s ceiling at nv=22, k=7) is
# already saturated by a single-thread bench client (200 QPS), so the
# latency-knee plot just shows a vertical pile at the masking
# ceiling. With 4 masking servers (740 pkg/s ceiling) the bench can
# climb from conc=1 toward the next bottleneck (shard/coord CPU).
export N_MASKING_SERVERS="${N_MASKING_SERVERS:-4}"
# Throughput sweep at higher concurrencies to capture the post-climb
# saturation region with the new 4-masking-server headroom.
export LOOKUP_THROUGHPUT_CONCURRENCIES="${LOOKUP_THROUGHPUT_CONCURRENCIES:-1,4,16,64,256,512,1024}"

# Commodity hardware: all roles on n2-standard-16 (16 vCPU / 64 GB).
# Small-regime per-shard load tops out ~4M entries (fill=90% @ log_cap=22)
# which fits in <10 GB RSS — no need for the extra RAM that
# bench-cluster.sh sets for medium's 30%/60%/90% fills.
export SHARD_MACHINE_TYPE="${SHARD_MACHINE_TYPE:-n2-standard-16}"
export COORD_MACHINE_TYPE="${COORD_MACHINE_TYPE:-n2-standard-16}"
export BENCH_CLIENT_MACHINE_TYPE="${BENCH_CLIENT_MACHINE_TYPE:-n2-standard-16}"
export MASKING_MACHINE_TYPE="${MASKING_MACHINE_TYPE:-n2-standard-16}"

# Outputs — distinct filenames so small/medium/large don't collide.
export LOCAL_SETUP_BENCH_OUT="$RESULTS_DIR/setup/small_cluster.json"
export LOCAL_PUBLISH_BENCH_DIR="$RESULTS_DIR/publish"
export LOCAL_LOOKUP_BENCH_DIR="$RESULTS_DIR/lookup"
export LOCAL_OUT_NAME_PREFIX="${LOCAL_OUT_NAME_PREFIX:-small-}"

log "host: $(hostname)"
log "PROJECT=$PROJECT  N_SHARDS=$N_SHARDS  SHARD_LOG_CAPACITY=$SHARD_LOG_CAPACITY  KZH_K=$KZH_K"
log "PUBLISH_BATCH_SIZES=$PUBLISH_BATCH_SIZES"
log "results dir: $RESULTS_DIR"

# --- Phase 1: up ---
if [[ "${SKIP_UP:-0}" != "1" ]]; then
  run_phase "small-up" "$REPO_ROOT/scripts/bench-cluster.sh" up || exit $?
fi

# --- Phase 2: deploy ---
if [[ "${SKIP_DEPLOY:-0}" != "1" ]]; then
  run_phase "small-deploy" "$REPO_ROOT/scripts/bench-cluster.sh" deploy || exit $?
fi

# --- Phase 3: setup-bench (per-shard parallel SRS gen) ---
if [[ "${SKIP_SETUP_BENCH:-0}" != "1" ]]; then
  run_phase "small-setup-bench" "$REPO_ROOT/scripts/bench-cluster.sh" setup-bench || exit $?
fi

# --- Phase 4a: start-masking ---
if [[ "${SKIP_START_MASKING:-0}" != "1" && "${ENABLE_MASKING_SERVER:-1}" == "1" ]]; then
  run_phase "small-start-masking" "$REPO_ROOT/scripts/bench-cluster.sh" start-masking || exit $?
fi

# --- Phase 4b: start-shards ---
if [[ "${SKIP_START_SHARDS:-0}" != "1" ]]; then
  run_phase "small-start-shards" "$REPO_ROOT/scripts/bench-cluster.sh" start-shards || exit $?
fi

# --- Phase 5: combined lookup-bench (folds in publish-bench per fill) ---
if [[ "${SKIP_LOOKUP_BENCH:-0}" != "1" ]]; then
  run_phase "small-lookup-bench" "$REPO_ROOT/scripts/bench-cluster.sh" lookup-bench \
    || log "WARN: lookup-bench failed; tearing down anyway"
fi

# --- Phase 7: down ---
if [[ "${SKIP_DOWN:-0}" != "1" ]]; then
  run_phase "small-down" "$REPO_ROOT/scripts/bench-cluster.sh" down \
    || log "WARN: down failed; some VMs may still be running — verify in console"
fi

log "==== SMALL CLUSTER DONE ===="
log "JSON outputs:"
find "$RESULTS_DIR" -name 'small-*.json' -o -name 'small_cluster.json' | sort | tee -a "$LOG_FILE"
