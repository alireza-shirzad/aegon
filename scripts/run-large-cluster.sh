#!/usr/bin/env bash
# run-large-cluster.sh — drive the large-regime cluster benchmark
# (128-shard cluster, per-shard SHARD_LOG_CAPACITY=27, KZH_K=9).
#
# Modeled on run-medium-cluster.sh, with two differences:
#   1. Large defaults: 128 shards, true_log_capacity=32 (2^32 = 4.3B
#      entries dictionary), masking fleet bumped to 8 (medium's 4 was
#      adequate for 2-shard fan-out but undersized for 128).
#   2. Chains in migration-bench BEFORE the fill ladder so the warmup
#      K is freshly picked under the live large cluster, not inherited
#      from the medium sweep. The best-K file is read AFTER the
#      migration-bench phase succeeds.
#
# Large regime sizing (matches sections/implementation.tex):
#   N_SHARDS=128, SHARD_LOG_CAPACITY=27, KZH_K=9
#   per-shard polynomial = 2^27 slots (α=4 over-provision of 2^25 entries)
#   true dictionary capacity = 128 × 2^25 = 2^32 = ~4.3B entries
#   publish batch sizes: 4096,8192,16384,32768,65536,131072
#   (per-shard effective batch ≈ K/128, so the per-shard work matches
#    the medium publish sweep)
#
# Outputs land at:
#   bench-results/setup/large_cluster.json
#   bench-results/migration/large_K{K}.json, large_best_k.txt
#   bench-results/publish/large_fill{0,30,60,90}.json
#   bench-results/lookup/large_fill{0,30,60,90}.json
#
# Cost: ~$155/hr compute + ~$36/hr storage with the recommended disk
# overrides below (coord=500GB, bench-client=100GB). Plan ~12-15 hr
# for the full chain (migration-bench + ladder); ~$2.4k expected.
#
# Env vars: same as run-medium-cluster.sh — PROJECT is required,
# everything else has large-appropriate defaults set below.

set -uo pipefail   # not -e: each phase should log on failure

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/bench-results"
LOG_FILE="$RESULTS_DIR/run-large-cluster.log"
mkdir -p "$RESULTS_DIR/setup" "$RESULTS_DIR/publish" "$RESULTS_DIR/lookup" "$RESULTS_DIR/migration"

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

# Large-regime cluster config — passed through to bench-cluster.sh.
export PROJECT
export N_SHARDS="${N_SHARDS:-128}"
export SHARD_LOG_CAPACITY="${SHARD_LOG_CAPACITY:-27}"
export KZH_K="${KZH_K:-9}"
# Large batch sizes (per-shard effective = K/128, so K=131072 means
# ~1024 per shard — the same per-shard work as medium's K=2048).
export PUBLISH_BATCH_SIZES="${PUBLISH_BATCH_SIZES:-4096,8192,16384,32768,65536,131072}"
export LOOKUP_PUBLISH_BATCH_SIZES="${LOOKUP_PUBLISH_BATCH_SIZES:-${PUBLISH_BATCH_SIZES}}"
# Migration-bench will pick the best K and write it to
# bench-results/migration/large_best_k.txt — we read it AFTER the
# migration phase succeeds (below), not here, since the file doesn't
# exist on a cold-start run yet.
_LRG_BEST_K_FILE="$REPO_ROOT/bench-results/migration/large_best_k.txt"
# 8 masking servers (medium uses 4). With 128 shards fanning out
# publish work, the masking queue is 32x more loaded; doubling the
# fleet keeps the package-generation ceiling above the saturation
# point of the cluster's publish throughput. Override if you want a
# different headroom margin.
export N_MASKING_SERVERS="${N_MASKING_SERVERS:-8}"
export LOOKUP_THROUGHPUT_CONCURRENCIES="${LOOKUP_THROUGHPUT_CONCURRENCIES:-1,4,16,64,256,512,1024}"
# Fill percentages for the lookup ladder. Large uses 1..10 (NOT the
# 1,30,60,90 medium convention) because the absolute label counts at
# 30%/60%/90% of a 2^32 dictionary are infeasible to preload at large
# scale (each 1% step is 43M labels — 30% would be 1.3B). Confirmed
# from the prior pre-DB-sharding large-combined.json which used
# fill_percents=[1..10]. bench-cluster.sh's default is the medium one
# (1,30,60,90), so we MUST override here. Regression-history: the
# v1/v2/v3 large runs accidentally inherited 1,30,60,90 and would
# have spent ~7 days on fill=30 alone before producing useful data.
export PUBLISH_FILL_PERCENTS="${PUBLISH_FILL_PERCENTS:-1,2,3,4,5,6,7,8,9,10}"
export LOOKUP_FILL_PERCENTS="${LOOKUP_FILL_PERCENTS:-1,2,3,4,5,6,7,8,9,10}"
# Drive publishes + lookups through a remote aegon_coordinator_server
# (same as medium). Set USE_REMOTE_COORD=0 to fall back to legacy
# in-process coord; not recommended at large because the bench-client
# RSS at fill=1% alone exceeds 14 GB before VRF + sharded merkle
# overhead get added.
export USE_REMOTE_COORD="${USE_REMOTE_COORD:-1}"

# Default zone: us-central1-a (us-central1-f was stocked out during the
# most recent medium run, and 128-VM cluster requests are more likely to
# hit zonal capacity limits). Override via ZONE=... if needed.
export ZONE="${ZONE:-us-central1-a}"

# Pin the machine types explicitly, as the small and medium scripts do.
# This script used to set none of them and silently inherited whatever
# bench-cluster.sh defaulted to, which is how the large regime ended up
# on different hardware than the small and medium ones without anyone
# choosing that. Every regime now reports n2-standard-16.
export SHARD_MACHINE_TYPE="${SHARD_MACHINE_TYPE:-n2-standard-16}"
export COORD_MACHINE_TYPE="${COORD_MACHINE_TYPE:-n2-standard-16}"
export BENCH_CLIENT_MACHINE_TYPE="${BENCH_CLIENT_MACHINE_TYPE:-n2-standard-16}"
export MASKING_MACHINE_TYPE="${MASKING_MACHINE_TYPE:-n2-standard-16}"

# --- Disk overrides for the new architecture ----------------------
# Coord state in the per-shard-DB era is O(N_shards) (just
# coord:state + per-epoch commitment cache) — a few GB total, NOT
# the 10 TB the default was sized for in the OLD single-DB era.
# Drop to 500 GB which is still 100x what we actually need.
export COORD_BOOT_DISK_SIZE="${COORD_BOOT_DISK_SIZE:-500GB}"
# Bench-client in remote-coord mode is just an RPC client with ~200
# MB RSS — no in-process state, no RocksDB, no SRS. 100 GB covers
# the OS + binaries + tmpfile output JSON.
export BENCH_CLIENT_BOOT_DISK_SIZE="${BENCH_CLIENT_BOOT_DISK_SIZE:-100GB}"
# Shards default to 1 TB each in bench-cluster.sh, but at N_SHARDS=128
# that's 128 TB of pd-ssd — exceeds the default us-central1 project
# quota of SSD_TOTAL_STORAGE=80 TB (`compute.googleapis.com/ssd_total_storage`).
# Per-shard data at fill=90% is ~150 GB (30M entries × ~5 KB each
# including L0/L1 compaction churn), so 500 GB leaves a 3× safety
# margin and fits the cluster in ~65 TB total. Override only if the
# project quota has been bumped and you want the extra LSM headroom.
export SHARD_BOOT_DISK_SIZE="${SHARD_BOOT_DISK_SIZE:-500GB}"

# Outputs — distinct filenames so large doesn't overwrite medium.
export LOCAL_SETUP_BENCH_OUT="$RESULTS_DIR/setup/large_cluster.json"
export LOCAL_PUBLISH_BENCH_DIR="$RESULTS_DIR/publish"
export LOCAL_LOOKUP_BENCH_DIR="$RESULTS_DIR/lookup"
export LOCAL_MIGRATION_BENCH_DIR="$RESULTS_DIR/migration"
export LOCAL_OUT_NAME_PREFIX="${LOCAL_OUT_NAME_PREFIX:-large-}"
# bench-cluster.sh's cmd_migration_bench writes to
# ${LOCAL_MIGRATION_BENCH_DIR}/${MIGRATION_REGIME}_K{K}.json and
# ${LOCAL_MIGRATION_BENCH_DIR}/${MIGRATION_REGIME}_best_k.txt
# — pin MIGRATION_REGIME=large so the names match the existing
# medium/large pattern.
export MIGRATION_REGIME="${MIGRATION_REGIME:-large}"

log "host: $(hostname)"
log "PROJECT=$PROJECT  N_SHARDS=$N_SHARDS  SHARD_LOG_CAPACITY=$SHARD_LOG_CAPACITY  KZH_K=$KZH_K  ZONE=$ZONE"
log "PUBLISH_BATCH_SIZES=$PUBLISH_BATCH_SIZES"
log "MIGRATION_REGIME=$MIGRATION_REGIME"
log "results dir: $RESULTS_DIR"

# --- Phase 1: up ---
if [[ "${SKIP_UP:-0}" != "1" ]]; then
  run_phase "large-up" "$REPO_ROOT/scripts/bench-cluster.sh" up || exit $?
fi

# --- Phase 2: deploy ---
if [[ "${SKIP_DEPLOY:-0}" != "1" ]]; then
  run_phase "large-deploy" "$REPO_ROOT/scripts/bench-cluster.sh" deploy || exit $?
fi

# --- Phase 3: setup-bench (per-shard parallel SRS gen) ---
if [[ "${SKIP_SETUP_BENCH:-0}" != "1" ]]; then
  run_phase "large-setup-bench" "$REPO_ROOT/scripts/bench-cluster.sh" setup-bench || exit $?
fi

# --- Phase 4a: start-masking ---
if [[ "${SKIP_START_MASKING:-0}" != "1" && "${ENABLE_MASKING_SERVER:-1}" == "1" ]]; then
  run_phase "large-start-masking" "$REPO_ROOT/scripts/bench-cluster.sh" start-masking || exit $?
fi

# --- Phase 4b: start-shards ---
if [[ "${SKIP_START_SHARDS:-0}" != "1" ]]; then
  run_phase "large-start-shards" "$REPO_ROOT/scripts/bench-cluster.sh" start-shards || exit $?
fi

# --- Phase 4c: start-coord (only when USE_REMOTE_COORD=1) ---
if [[ "${SKIP_START_COORD:-0}" != "1" && "${USE_REMOTE_COORD:-1}" == "1" ]]; then
  run_phase "large-start-coord" "$REPO_ROOT/scripts/bench-cluster.sh" start-coord || exit $?
fi

# --- Phase 5: migration-bench (K-sweep, single-batch per K) ---
# Runs aegon_migration_bench against the live cluster, sweeping
# MIGRATION_CHUNK_SIZES (default 16384,65536,262144,1048576) and
# writing one JSON per K + a `large_best_k.txt` integer for the
# downstream lookup-bench to read as its --warmup-batch-size.
# Skipped when SKIP_MIGRATION_BENCH=1.
if [[ "${SKIP_MIGRATION_BENCH:-0}" != "1" ]]; then
  run_phase "large-migration-bench" "$REPO_ROOT/scripts/bench-cluster.sh" migration-bench \
    || log "WARN: migration-bench failed; lookup-bench will fall back to PUBLISH_WARMUP_BATCH_SIZE default"
fi

# Read the migration-bench's chosen K and export it for lookup-bench.
# The file is written by aegon_migration_bench's K-sweep post-pass.
# If the phase was skipped or failed, fall back to PUBLISH_WARMUP_BATCH_SIZE
# from the environment, or 131072 (the historical large default).
if [[ -z "${PUBLISH_WARMUP_BATCH_SIZE:-}" && -r "$_LRG_BEST_K_FILE" ]]; then
  _LRG_BEST_K="$(tr -dc '0-9' < "$_LRG_BEST_K_FILE")"
  if [[ -n "$_LRG_BEST_K" ]]; then
    PUBLISH_WARMUP_BATCH_SIZE="$_LRG_BEST_K"
    log "PUBLISH_WARMUP_BATCH_SIZE not set; using migration-best K=$_LRG_BEST_K from $_LRG_BEST_K_FILE"
  fi
fi
export PUBLISH_WARMUP_BATCH_SIZE="${PUBLISH_WARMUP_BATCH_SIZE:-131072}"
log "PUBLISH_WARMUP_BATCH_SIZE=$PUBLISH_WARMUP_BATCH_SIZE (used by lookup-bench for between-fill preload climbs)"

# --- Phase 5b: post-migration cluster reset ---
# migration-bench's K-sweep runs aegon_migration_bench against the
# shards via its OWN in-process ShardedAegon (separate state from the
# coord-server). `clear_dictionary` is called BETWEEN K-values but
# NOT after the last K, so the shards finish with state from the
# final K=1048576 publish (epoch=1, plus its labels in the DB). The
# long-running coord-server, meanwhile, was untouched by migration-
# bench and is still at epoch=0. When lookup-bench drives that
# coord-server, the first publish bumps shards from epoch=1 to 2
# (entries persisted with epoch=2) but the coord only advances to
# epoch=1 (epoch_commits.len()=2) — and lookup_history then OOB-
# indexes epoch_commits[entry.epoch] because shard.epoch > coord.epoch.
# Observed at 22:24:33 UTC 2026-06-18 (v4): `index out of bounds:
# the len is 42 but the index is 42` after fill=1% preload + 50
# lookup samples. Fix: wipe + restart shards AND coord so both
# start at epoch=0 for lookup-bench.
if [[ "${SKIP_POST_MIGRATION_RESET:-0}" != "1" ]]; then
  run_phase "large-restart-shards" "$REPO_ROOT/scripts/bench-cluster.sh" start-shards || exit $?
  if [[ "${USE_REMOTE_COORD:-1}" == "1" ]]; then
    run_phase "large-restart-coord" "$REPO_ROOT/scripts/bench-cluster.sh" start-coord || exit $?
  fi
fi

# --- Phase 6: combined lookup-bench (folds in publish-bench per fill) ---
WATCHDOG_LOG="$RESULTS_DIR/large-watchdog.log"
POSTMORTEM_DIR="$RESULTS_DIR/large-postmortem-$(date +%Y%m%dT%H%M%S)"
WATCHDOG_PID=""
if [[ "${SKIP_LOOKUP_BENCH:-0}" != "1" ]]; then
  : > "$WATCHDOG_LOG"
  log "starting watchdog loop (per-shard probe every 60s) -> $WATCHDOG_LOG"
  (
    while true; do
      echo "===== $(date -Is) =====" >> "$WATCHDOG_LOG"
      PROJECT="$PROJECT" N_SHARDS="$N_SHARDS" ZONE="$ZONE" \
        "$REPO_ROOT/scripts/bench-cluster.sh" watchdog >> "$WATCHDOG_LOG" 2>&1 || true
      sleep 60
    done
  ) &
  WATCHDOG_PID=$!
  log "watchdog PID=$WATCHDOG_PID"

  run_phase "large-lookup-bench" "$REPO_ROOT/scripts/bench-cluster.sh" lookup-bench \
    || log "WARN: lookup-bench failed; fetching postmortems before teardown"

  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
    log "watchdog stopped (PID=$WATCHDOG_PID)"
  fi

  # ALWAYS fetch shard/coord postmortems before `down` — see medium
  # script for rationale. At N_SHARDS=128 the per-shard fetch is
  # parallelized (background subshells) so the whole pass stays
  # under ~3 minutes instead of 128 × ~5s sequential.
  mkdir -p "$POSTMORTEM_DIR"
  log "fetching postmortems (parallel, 128 shards) -> $POSTMORTEM_DIR"
  for ((i = 0; i < N_SHARDS; i++)); do
    sname="aegon-bench-shard-$i"
    (
      timeout 120 gcloud compute ssh "$sname" --project="$PROJECT" --zone="$ZONE" \
        --tunnel-through-iap --quiet \
        --command="echo '=== /tmp/aegon-shard.log (last 2MB) ==='; tail -c 2000000 /tmp/aegon-shard.log 2>/dev/null; echo; echo '=== dmesg -T (last 500 lines) ==='; sudo dmesg -T 2>/dev/null | tail -n 500; echo; echo '=== free -m ==='; free -m; echo; echo '=== ps aux | grep aegon ==='; ps aux | grep -i aegon | grep -v grep" \
        > "$POSTMORTEM_DIR/shard-$i.log" 2>&1
    ) &
  done
  wait
  for cname in aegon-bench-coord aegon-bench-client; do
    (
      timeout 120 gcloud compute ssh "$cname" --project="$PROJECT" --zone="$ZONE" \
        --tunnel-through-iap --quiet \
        --command="echo '=== dmesg -T (last 500 lines) ==='; sudo dmesg -T 2>/dev/null | tail -n 500; echo; echo '=== free -m ==='; free -m; echo; echo '=== ps aux | grep aegon ==='; ps aux | grep -i aegon | grep -v grep; echo; echo '=== journalctl -u aegon-lookup-bench --no-pager | tail -n 200 ==='; sudo journalctl -u aegon-lookup-bench --no-pager 2>/dev/null | tail -n 200" \
        > "$POSTMORTEM_DIR/$cname.log" 2>&1
    ) || log "WARN: $cname postmortem fetch failed"
  done
  log "postmortems saved to $POSTMORTEM_DIR"
fi

# --- Phase 7: down ---
if [[ "${SKIP_DOWN:-0}" != "1" ]]; then
  run_phase "large-down" "$REPO_ROOT/scripts/bench-cluster.sh" down \
    || log "WARN: down failed; some VMs may still be running — verify in console"
fi

log "==== LARGE CLUSTER DONE ===="
log "JSON outputs:"
find "$RESULTS_DIR" -name 'large*.json' | sort | tee -a "$LOG_FILE"
