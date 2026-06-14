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
# PUBLISH_TRUE_LOG_CAP / LOOKUP_TRUE_LOG_CAP are now auto-derived
# by bench-cluster.sh from N_SHARDS and SHARD_LOG_CAPACITY (medium
# → 27 + 1 - 2 = 26). Override here only if you want a different
# addressable space than the cluster's physical one.
# Medium batch sizes; large uses 4096..131072.
export PUBLISH_BATCH_SIZES="${PUBLISH_BATCH_SIZES:-64,128,256,512,1024,2048}"
export LOOKUP_PUBLISH_BATCH_SIZES="${LOOKUP_PUBLISH_BATCH_SIZES:-${PUBLISH_BATCH_SIZES}}"
# Warmup batch size. Reads bench-results/migration/medium_best_k.txt
# (written by aegon_migration_bench's K-sweep) when present, falling
# back to 16384 — the historical empirical sweet spot from v6 on
# n2-standard-16 shards (v9 tested batch=65536 expecting MSM
# amortization; it was ~2x SLOWER per entry because the KZH-k FK
# acceleration tables overflowed L3 at the bigger batch). If you ran
# migration-bench at the current cluster hardware, the file's K is
# likely better tuned than the historical default; override via env
# if you want to force a specific value.
_MED_BEST_K_FILE="$REPO_ROOT/bench-results/migration/medium_best_k.txt"
if [[ -z "${PUBLISH_WARMUP_BATCH_SIZE:-}" && -r "$_MED_BEST_K_FILE" ]]; then
  _MED_BEST_K="$(tr -dc '0-9' < "$_MED_BEST_K_FILE")"
  if [[ -n "$_MED_BEST_K" ]]; then
    PUBLISH_WARMUP_BATCH_SIZE="$_MED_BEST_K"
    log "PUBLISH_WARMUP_BATCH_SIZE not set; using migration-best K=$_MED_BEST_K from $_MED_BEST_K_FILE"
  fi
fi
export PUBLISH_WARMUP_BATCH_SIZE="${PUBLISH_WARMUP_BATCH_SIZE:-16384}"
# 4 masking servers so the throughput-sweep curve shows a real climb
# phase. One masking server (120 pkg/s ceiling at nv=27, k=9) is
# already saturated by a single-thread bench client (~129 QPS), so
# the latency-knee plot just shows a vertical pile at the masking
# ceiling. With 4 masking servers (480 pkg/s ceiling) the bench can
# climb from conc=1 toward the next bottleneck.
export N_MASKING_SERVERS="${N_MASKING_SERVERS:-4}"
export LOOKUP_THROUGHPUT_CONCURRENCIES="${LOOKUP_THROUGHPUT_CONCURRENCIES:-1,4,16,64,256,512,1024}"
# Medium regime now drives publishes + lookups through a remote
# `aegon_coordinator_server` running on the coord VM (instead of
# bundling the coord state inside the bench-client binary).
# Architecturally matches what a production deployment would look like:
# bench-client → coord (over network) → shards (over network).
# Set USE_REMOTE_COORD=0 to fall back to the legacy in-process coord
# topology if you need to compare against pre-refactor bench numbers.
export USE_REMOTE_COORD="${USE_REMOTE_COORD:-1}"

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

# --- Phase 3: setup-bench (per-shard parallel SRS gen) ---
if [[ "${SKIP_SETUP_BENCH:-0}" != "1" ]]; then
  run_phase "medium-setup-bench" "$REPO_ROOT/scripts/bench-cluster.sh" setup-bench || exit $?
fi

# --- Phase 4a: start-masking (background masking server prebuilds packages) ---
if [[ "${SKIP_START_MASKING:-0}" != "1" && "${ENABLE_MASKING_SERVER:-1}" == "1" ]]; then
  run_phase "medium-start-masking" "$REPO_ROOT/scripts/bench-cluster.sh" start-masking || exit $?
fi

# --- Phase 4b: start-shards (once; bench binaries drive prefill via RPC) ---
if [[ "${SKIP_START_SHARDS:-0}" != "1" ]]; then
  run_phase "medium-start-shards" "$REPO_ROOT/scripts/bench-cluster.sh" start-shards || exit $?
fi

# --- Phase 4c: start-coord (only when USE_REMOTE_COORD=1) ---
# Launches `aegon_coordinator_server` on the coord VM, which talks to
# every shard via gRPC. The bench-client then connects only to the
# coord; without this phase, USE_REMOTE_COORD=1 fails fast in
# bench-cluster.sh because `coord_endpoint` can't resolve a listening
# port.
if [[ "${SKIP_START_COORD:-0}" != "1" && "${USE_REMOTE_COORD:-1}" == "1" ]]; then
  run_phase "medium-start-coord" "$REPO_ROOT/scripts/bench-cluster.sh" start-coord || exit $?
fi

# --- Phase 5: combined lookup-bench (folds in publish-bench per fill) ---
# Standalone publish-bench was removed: aegon_lookup_bench walks
# --fill-percents internally, climbing each fill via real publish
# (no shortcut) and running lookups + audit + publish-batch-sweep
# at every level. The warmup work is paid exactly once per fill.
#
# Wrap with a background watchdog loop (per-shard RSS + free mem + OOM
# count every 60s) and a postmortem fetch (last 2MB of each shard's
# /tmp/aegon-shard.log + last 500 lines of dmesg -T) so we can diagnose
# OOMs / panics that happened on a non-bench VM before `down` wipes the
# evidence.
WATCHDOG_LOG="$RESULTS_DIR/medium-watchdog.log"
POSTMORTEM_DIR="$RESULTS_DIR/medium-postmortem-$(date +%Y%m%dT%H%M%S)"
WATCHDOG_PID=""
if [[ "${SKIP_LOOKUP_BENCH:-0}" != "1" ]]; then
  : > "$WATCHDOG_LOG"
  log "starting watchdog loop (per-shard probe every 60s) -> $WATCHDOG_LOG"
  (
    while true; do
      echo "===== $(date -Is) =====" >> "$WATCHDOG_LOG"
      PROJECT="$PROJECT" N_SHARDS="$N_SHARDS" \
        "$REPO_ROOT/scripts/bench-cluster.sh" watchdog >> "$WATCHDOG_LOG" 2>&1 || true
      sleep 60
    done
  ) &
  WATCHDOG_PID=$!
  log "watchdog PID=$WATCHDOG_PID"

  run_phase "medium-lookup-bench" "$REPO_ROOT/scripts/bench-cluster.sh" lookup-bench \
    || log "WARN: lookup-bench failed; fetching postmortems before teardown"

  # Stop watchdog before its next iteration spams the log.
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
    log "watchdog stopped (PID=$WATCHDOG_PID)"
  fi

  # ALWAYS fetch shard/coord postmortems before `down` — this is the
  # only chance to recover the shard's stderr (panic traces, etc.) and
  # the kernel ring buffer (OOM-killer signatures). Fails-soft so a
  # dead VM doesn't block teardown.
  mkdir -p "$POSTMORTEM_DIR"
  log "fetching postmortems -> $POSTMORTEM_DIR"
  ZONE="${ZONE:-us-central1-f}"
  for ((i = 0; i < N_SHARDS; i++)); do
    sname="aegon-bench-shard-$i"
    (
      timeout 120 gcloud compute ssh "$sname" --project="$PROJECT" --zone="$ZONE" \
        --tunnel-through-iap --quiet \
        --command="echo '=== /tmp/aegon-shard.log (last 2MB) ==='; tail -c 2000000 /tmp/aegon-shard.log 2>/dev/null; echo; echo '=== dmesg -T (last 500 lines) ==='; sudo dmesg -T 2>/dev/null | tail -n 500; echo; echo '=== free -m ==='; free -m; echo; echo '=== ps aux | grep aegon ==='; ps aux | grep -i aegon | grep -v grep" \
        > "$POSTMORTEM_DIR/shard-$i.log" 2>&1
    ) || log "WARN: shard-$i postmortem fetch failed"
  done
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
  run_phase "medium-down" "$REPO_ROOT/scripts/bench-cluster.sh" down \
    || log "WARN: down failed; some VMs may still be running — verify in console"
fi

log "==== MEDIUM CLUSTER DONE ===="
log "JSON outputs:"
find "$RESULTS_DIR" -name 'medium*.json' | sort | tee -a "$LOG_FILE"
