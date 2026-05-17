#!/usr/bin/env bash
#
# bench-sweep.sh — 2D sweep over (prefill count, batch size).
#
# For each prefill level in {2^MIN_PREFILL_LOG2 .. 2^MAX_PREFILL_LOG2},
# redeploy the bench cluster (kills + restarts shards so they re-prefill),
# then run aegon_coordinator_bench with batch sizes {2^MIN_BATCH_LOG2 ..
# 2^MAX_BATCH_LOG2}. Saves one JSON per prefill level under $OUT_DIR.
#
# Defaults (chosen to match the request "prefill up to total_cap/8,
# batch up to min(2^17, total_cap/16)"):
#   N_SHARDS=4, SHARD_LOG_CAPACITY=20
#   → total_cap = N_SHARDS * 2^SHARD_LOG_CAPACITY = 2^22
#   → MAX_PREFILL_LOG2 = log2(total_cap/8) = 19
#   → MAX_BATCH_LOG2   = min(17, log2(total_cap/16)) = 17
#
# Override via env vars before invocation.
#
# Required: PROJECT (GCP project id).
# Assumes:  the cluster is already `up` (instances created). Will deploy
#           and re-deploy as it sweeps. Will not run `up` or `down` for
#           you — manage cluster lifecycle separately.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- knobs ------------------------------------------------------------
N_SHARDS="${N_SHARDS:-4}"
SHARD_LOG_CAPACITY="${SHARD_LOG_CAPACITY:-20}"
SHARD_MACHINE_TYPE="${SHARD_MACHINE_TYPE:-n2-standard-4}"
SAMPLES_PER_BATCH="${SAMPLES_PER_BATCH:-1}"
SETUP_SEED="${SETUP_SEED:-42}"
PREFILL_SEED="${PREFILL_SEED:-1}"
KZH_K="${KZH_K:-10}"

# Compute total cluster capacity in log2 (needs N_SHARDS to be a power of 2).
TOTAL_CAP_LOG2=$(python3 -c "
import math
n = ${N_SHARDS}
assert n > 0 and (n & (n-1)) == 0, f'N_SHARDS must be power of two, got {n}'
print(${SHARD_LOG_CAPACITY} + int(math.log2(n)))
")

MIN_PREFILL_LOG2="${MIN_PREFILL_LOG2:-0}"
MAX_PREFILL_LOG2="${MAX_PREFILL_LOG2:-$((TOTAL_CAP_LOG2 - 3))}"
MIN_BATCH_LOG2="${MIN_BATCH_LOG2:-0}"
MAX_BATCH_LOG2="${MAX_BATCH_LOG2:-$(python3 -c "print(min(17, ${TOTAL_CAP_LOG2} - 4))")}"

OUT_DIR="${OUT_DIR:-/tmp/aegon-sweep-$(date +%Y%m%d-%H%M%S)}"
ZONE="${ZONE:-us-central1-f}"

# ---- helpers ----------------------------------------------------------
die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

[[ -n "${PROJECT:-}" ]] || die "PROJECT env var is required (export PROJECT=...)"

mkdir -p "$OUT_DIR"

BATCH_SIZES_CSV=$(python3 -c "
sizes = [2**i for i in range($MIN_BATCH_LOG2, $MAX_BATCH_LOG2 + 1)]
print(','.join(str(s) for s in sizes))
")

# Wait for every shard's /tmp/aegon-shard.log to contain 'listening on'.
# Polls every 15s; gives up after 30 min per shard (well past the worst
# expected prefill time even at 2^17/shard).
wait_for_shards() {
  local n="$1"
  for i in $(seq 0 $((n - 1))); do
    local timeout=120  # 120 * 15s = 30 min
    local waited=0
    until gcloud compute ssh "aegon-bench-shard-$i" --zone="$ZONE" \
            --tunnel-through-iap --quiet \
            --command 'grep -q "listening on" /tmp/aegon-shard.log' 2>/dev/null; do
      sleep 15
      waited=$((waited + 1))
      if (( waited > timeout )); then
        die "shard $i never bound within 30 min — inspect /tmp/aegon-shard.log on it"
      fi
    done
  done
}

# ---- preamble ---------------------------------------------------------
log "================ bench-sweep ================"
log "project=$PROJECT  zone=$ZONE  out=$OUT_DIR"
log "N_SHARDS=$N_SHARDS  SHARD_LOG_CAPACITY=$SHARD_LOG_CAPACITY  (cluster cap = 2^$TOTAL_CAP_LOG2)"
log "prefill log2: $MIN_PREFILL_LOG2 .. $MAX_PREFILL_LOG2  ($((MAX_PREFILL_LOG2 - MIN_PREFILL_LOG2 + 1)) levels)"
log "batch   log2: $MIN_BATCH_LOG2 .. $MAX_BATCH_LOG2  ($((MAX_BATCH_LOG2 - MIN_BATCH_LOG2 + 1)) levels)"
log "batch sizes (csv): $BATCH_SIZES_CSV"
log "samples per batch: $SAMPLES_PER_BATCH"
log "============================================="

# Write the sweep configuration alongside the per-prefill results so the
# downstream plotter knows what each column means.
cat >"$OUT_DIR/sweep-config.json" <<EOF
{
  "n_shards": $N_SHARDS,
  "shard_log_capacity": $SHARD_LOG_CAPACITY,
  "total_cap_log2": $TOTAL_CAP_LOG2,
  "kzh_k": $KZH_K,
  "prefill_log2_min": $MIN_PREFILL_LOG2,
  "prefill_log2_max": $MAX_PREFILL_LOG2,
  "batch_log2_min": $MIN_BATCH_LOG2,
  "batch_log2_max": $MAX_BATCH_LOG2,
  "batch_sizes": [$BATCH_SIZES_CSV],
  "samples_per_batch": $SAMPLES_PER_BATCH,
  "setup_seed": $SETUP_SEED,
  "prefill_seed": $PREFILL_SEED
}
EOF

# ---- main loop --------------------------------------------------------
# First prefill level does a full `deploy` (cargo build + scp binaries +
# install/restart redis + start shards). Subsequent levels use
# `restart-shards`, which reuses the uploaded binaries and the on-disk
# SRS cache and just FLUSHALLs Redis + restarts shards with the new
# --prefill-count. That collapses per-level overhead from ~9 min to ~30 s.
first_iter=1
for prefill_log2 in $(seq "$MIN_PREFILL_LOG2" "$MAX_PREFILL_LOG2"); do
  prefill_count=$((1 << prefill_log2))
  per_shard=$(( prefill_count / N_SHARDS ))
  log "===== prefill = 2^$prefill_log2 = $prefill_count  ($per_shard / shard) ====="

  if (( first_iter )); then
    sub="deploy"
    first_iter=0
  else
    sub="restart-shards"
  fi

  # 1. (Re)start shards with the new prefill count. Each shard kills
  #    the existing server, then starts a fresh one with
  #    --prefill-count = $per_shard, which loads SRS from disk (or
  #    generates it once on first deploy) and prefills before binding
  #    the gRPC socket.
  log "  $sub: TOTAL_PRELOAD_LOG2=$prefill_log2"
  if ! PROJECT="$PROJECT" \
       N_SHARDS="$N_SHARDS" \
       SHARD_LOG_CAPACITY="$SHARD_LOG_CAPACITY" \
       TOTAL_PRELOAD_LOG2="$prefill_log2" \
       SHARD_MACHINE_TYPE="$SHARD_MACHINE_TYPE" \
       SETUP_SEED="$SETUP_SEED" \
       PREFILL_SEED="$PREFILL_SEED" \
       KZH_K="$KZH_K" \
       "$SCRIPT_DIR/bench-cluster.sh" "$sub" \
         >"$OUT_DIR/$sub-prefill$prefill_log2.log" 2>&1; then
    log "  WARN: $sub failed at prefill_log2=$prefill_log2 — see $OUT_DIR/$sub-prefill$prefill_log2.log; skipping"
    continue
  fi

  # 2. Wait for shards to actually bind. `deploy` returns once the
  #    background nohup is launched, not once the server is ready.
  log "  waiting for shards to bind..."
  if ! wait_for_shards "$N_SHARDS"; then
    log "  WARN: shards didn't bind; skipping"
    continue
  fi
  log "  all shards bound"

  # 3. Run the bench sweep at this prefill level.
  result_json="$OUT_DIR/bench-prefill$prefill_log2.json"
  log "  bench: batch sweep, samples=$SAMPLES_PER_BATCH → $result_json"
  if ! PROJECT="$PROJECT" \
       N_SHARDS="$N_SHARDS" \
       SHARD_LOG_CAPACITY="$SHARD_LOG_CAPACITY" \
       KZH_K="$KZH_K" \
       SETUP_SEED="$SETUP_SEED" \
       BATCH_SIZES="$BATCH_SIZES_CSV" \
       SAMPLES_PER_BATCH="$SAMPLES_PER_BATCH" \
       LOCAL_BENCH_OUT="$result_json" \
       "$SCRIPT_DIR/bench-cluster.sh" bench \
         >"$OUT_DIR/bench-prefill$prefill_log2.log" 2>&1; then
    log "  WARN: bench failed at prefill_log2=$prefill_log2 — see $OUT_DIR/bench-prefill$prefill_log2.log; continuing"
    continue
  fi
  log "  done"
done

log "============================================="
log "sweep complete. per-prefill JSONs in $OUT_DIR"
log "sweep-config.json captures the parameter ranges"
