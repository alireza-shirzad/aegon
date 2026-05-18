#!/usr/bin/env bash
# lookup-bench.sh — local lookup + publish combined bench
# driver for the small + medium regimes.
#
# At each fill_percent stage:
#   1. Top up the lookup-sampleable namespace to `floor(2^true_cap × pct/100)`
#      via publish (anonymous bulk filler is added once at startup via
#      --initial-prefill-count, so the dict already sits at the target
#      fill level when each stage begins — the lookup namespace is the
#      portion of the dict we know how to sample).
#   2. Run lookup-bench samples (server + client × {label, value,
#      value-history, label-history}) and record data/proof/total wire
#      sizes per the user's spec.
#   3. Run publish-bench (sweep batch sizes for that regime) on a
#      disjoint namespace so it doesn't pollute future-stage lookup
#      samples.
#
# Regime sizing mirrors publish-bench.sh / setup-bench.sh:
#
#   regime  | shard_log_cap | true_log_cap | publish batch sizes
#   --------|---------------|--------------|---------------------------
#   small   | 22            | 20           | 2,4,8,16,32,64
#   medium  | 28            | 26           | 64,128,512,1024,2048,4096
#
# Fill percentages (vs. true capacity): 0, 30, 60, 90 — walked in one
# process per regime. The initial bulk prefill brings the dict to the
# *highest* requested fill_pct minus a small headroom; subsequent
# stages publish only the small lookup-sampleable namespace and the
# publish-bench's own samples.
#
# Note: the planetary regime is the cluster path —
#   ./scripts/bench-cluster.sh lookup-bench
#
# Tunables (env):
#   OUT_DIR             output directory (default /tmp/aegon-lookup)
#   FILL_PERCENTS       comma-separated overrides (default 0,30,60,90)
#   LOOKUP_SAMPLES      lookup samples per stage (default 20)
#   PUBLISH_SAMPLES     publish bench samples per batch (default 3)
#   AUDIT_SAMPLES       per-stage `verify_sharded_invariance` samples
#                       (default 5 — needs ≥ N+1 epochs available)
#   SETUP_SEED          SRS RNG seed (default 42)
#   PREFILL_SEED        prefill RNG seed base (default 1)
#   COORD_LISTEN        loopback for in-process coordinator (default
#                       127.0.0.1:50190)
#   SKIP_SMALL          set to 1 to skip the small regime
#   SKIP_MEDIUM        set to 1 to skip the medium regime

set -euo pipefail

OUT_DIR="${OUT_DIR:-/tmp/aegon-lookup}"
FILL_PERCENTS="${FILL_PERCENTS:-0,30,60,90}"
LOOKUP_SAMPLES="${LOOKUP_SAMPLES:-20}"
PUBLISH_SAMPLES="${PUBLISH_SAMPLES:-3}"
AUDIT_SAMPLES="${AUDIT_SAMPLES:-5}"
SETUP_SEED="${SETUP_SEED:-42}"
PREFILL_SEED="${PREFILL_SEED:-1}"
COORD_LISTEN="${COORD_LISTEN:-127.0.0.1:50190}"
mkdir -p "$OUT_DIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "building aegon_lookup_bench (release)"
(cd "$REPO_ROOT" && cargo build --release -p akd --bin aegon_lookup_bench) >/dev/null
LB="$REPO_ROOT/target/release/aegon_lookup_bench"
[[ -x "$LB" ]] || { echo "binary missing: $LB" >&2; exit 1; }

# Compute the largest fill_pct we'll ever climb to. We bulk-prefill
# *up to* that target minus a per-stage lookup-namespace headroom, so
# each stage's lookup-sampleable count = target_total - initial_prefill
# is a manageable few thousand.
max_fill_pct() {
  local IFS=','
  local max=0
  for p in $1; do
    if (( p > max )); then max=$p; fi
  done
  echo "$max"
}
MAX_PCT="$(max_fill_pct "$FILL_PERCENTS")"

# Per-regime: lookup-namespace size at the *highest* fill_pct. Below
# that fill_pct, fewer labels are needed; the bench computes the
# sampleable count per stage as (target - initial_prefill).
#   small/medium: 1000 sampleable labels at MAX_PCT is comfortable
#                  for `--samples-per-level 20`.
LOOKUP_NAMESPACE_SIZE_SMALL=1000
LOOKUP_NAMESPACE_SIZE_MEDIUM=1000

run_one() {
  local label="$1"
  local shard_log_capacity="$2"
  local true_log_capacity="$3"
  local publish_batches="$4"
  local lookup_ns_size="$5"
  local out="$OUT_DIR/${label}.json"
  local true_cap=$((1 << true_log_capacity))
  # initial bulk prefill = floor(true_cap × MAX_PCT / 100) − lookup_ns_size,
  # clamped to ≥ 0. Below MAX_PCT, the bench publishes nothing because
  # initial_prefill already exceeds target (sampleable count clamped to 0
  # at those stages — they still run the publish bench).
  local max_target=$(( true_cap * MAX_PCT / 100 ))
  local prefill=$(( max_target - lookup_ns_size ))
  if (( prefill < 0 )); then prefill=0; fi
  log "$label: shard_log_capacity=$shard_log_capacity true_log_capacity=$true_log_capacity"
  log "$label:   true_capacity=2^$true_log_capacity=$true_cap"
  log "$label:   max_fill_pct=$MAX_PCT%  max_target=$max_target  initial_prefill=$prefill"
  log "$label:   publish_batches=$publish_batches"
  "$LB" \
    --shard-log-capacity "$shard_log_capacity" \
    --true-log-capacity "$true_log_capacity" \
    --n-shards 1 \
    --fill-percents "$FILL_PERCENTS" \
    --initial-prefill-count "$prefill" \
    --prefill-seed "$PREFILL_SEED" \
    --setup-seed "$SETUP_SEED" \
    --samples-per-level "$LOOKUP_SAMPLES" \
    --publish-batch-sizes "$publish_batches" \
    --publish-samples-per-batch "$PUBLISH_SAMPLES" \
    --audit-samples "$AUDIT_SAMPLES" \
    --coordinator-listen "$COORD_LISTEN" \
    --output "$out"
  log "$label: wrote $out"
}

if [[ "${SKIP_SMALL:-0}" != "1" ]]; then
  run_one "small"  22 20 "2,4,8,16,32,64"          "$LOOKUP_NAMESPACE_SIZE_SMALL"
else
  log "small: skipped via SKIP_SMALL=1"
fi

if [[ "${SKIP_MEDIUM:-0}" != "1" ]]; then
  run_one "medium" 28 26 "64,128,512,1024,2048,4096" "$LOOKUP_NAMESPACE_SIZE_MEDIUM"
else
  log "medium: skipped via SKIP_MEDIUM=1"
fi

log "done. JSON records in $OUT_DIR/"
log ""
log "for the planetary regime (2^32 dict / 32 shards / publish batches 4096..131072), run:"
log "    PROJECT=<gcp-project> ./scripts/bench-cluster.sh up"
log "    PROJECT=<gcp-project> ./scripts/bench-cluster.sh deploy"
log "    PROJECT=<gcp-project> ./scripts/bench-cluster.sh lookup-bench"
