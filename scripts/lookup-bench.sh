#!/usr/bin/env bash
# lookup-bench.sh — local lookup + publish + audit bench driver
# for the small regime only.
#
# Walks fill_percents (1,30,60,90 by default) by invoking the bench
# binary once per fill_percent. Each invocation:
#   1. Bulk-prefills the dict to the target fill via --initial-prefill-count
#      (anonymous entries — not lookup-sampleable).
#   2. Tops up a small lookup-sampleable namespace via publish.
#   3. Runs lookup samples (server + client × {label, value,
#      value-history, label-history}) and records data/proof/total wire
#      sizes per the user's spec.
#   4. Runs publish-bench (sweep batch sizes for that regime) on a
#      disjoint namespace so it doesn't pollute the lookup samples.
#   5. Runs the audit check `verify_sharded_invariance` for the most
#      recent N+1 epochs.
#
# Emits ${OUT_DIR}/${regime}_fill${pct}.json per stage. Per-stage
# invocation matches the cluster path (bench-cluster.sh
# cmd_lookup_bench), so plot scripts treat local and cluster data
# uniformly.
#
# Regime sizing mirrors publish-bench.sh / setup-bench.sh:
#
#   regime  | shard_log_cap | true_log_cap | publish batch sizes
#   --------|---------------|--------------|---------------------------
#   small   | 22            | 20           | 2,4,8,16,32,64
#
# Fill percentages (vs. true capacity): 1, 30, 60, 90 — set
# FILL_PERCENTS to override. True capacity for any regime is the
# shard polynomial size divided by `OVER_PROVISIONING_FACTOR` (= 4);
# see `akd/src/aegon/config.rs`.
#
# The medium (2-shard) and large (128-shard) regimes both run on a
# real cluster:
#   N_SHARDS=2   ./scripts/bench-cluster.sh lookup-bench   # medium
#   N_SHARDS=128 ./scripts/bench-cluster.sh lookup-bench   # large
#
# Tunables (env):
#   OUT_DIR             output directory (default /tmp/aegon-lookup)
#   FILL_PERCENTS       comma-separated overrides (default 1,30,60,90)
#   LOOKUP_NS_SIZE      lookup-sampleable namespace size at each
#                       stage > 0 (default 1000)
#   LOOKUP_SAMPLES      lookup samples per stage (default 20)
#   PUBLISH_SAMPLES     publish bench samples per batch (default 3)
#   AUDIT_SAMPLES       per-stage `verify_sharded_invariance` samples
#                       (default 5 — needs ≥ N+1 epochs available)
#   SETUP_SEED          SRS RNG seed (default 42)
#   PREFILL_SEED        prefill RNG seed base (default 1)
#   COORD_LISTEN        loopback for in-process coordinator (default
#                       127.0.0.1:50190)
#   SKIP_SMALL          set to 1 to skip the small regime

set -euo pipefail

OUT_DIR="${OUT_DIR:-/tmp/aegon-lookup}"
FILL_PERCENTS="${FILL_PERCENTS:-1,30,60,90}"
LOOKUP_NS_SIZE="${LOOKUP_NS_SIZE:-1000}"
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

run_one_fill() {
  local label="$1"
  local shard_log_capacity="$2"
  local true_log_capacity="$3"
  local publish_batches="$4"
  local fill_pct="$5"

  local true_cap=$((1 << true_log_capacity))
  local target=$(( true_cap * fill_pct / 100 ))
  local prefill=$(( target - LOOKUP_NS_SIZE ))
  if (( prefill < 0 )); then prefill=0; fi

  local out="$OUT_DIR/${label}_fill${fill_pct}.json"
  log "$label fill=${fill_pct}%: true_cap=$true_cap target=$target initial_prefill=$prefill lookup_ns=$LOOKUP_NS_SIZE"
  "$LB" \
    --shard-log-capacity "$shard_log_capacity" \
    --true-log-capacity "$true_log_capacity" \
    --n-shards 1 \
    --fill-percents "$fill_pct" \
    --initial-prefill-count "$prefill" \
    --prefill-seed "$PREFILL_SEED" \
    --setup-seed "$SETUP_SEED" \
    --samples-per-level "$LOOKUP_SAMPLES" \
    --publish-batch-sizes "$publish_batches" \
    --publish-samples-per-batch "$PUBLISH_SAMPLES" \
    --audit-samples "$AUDIT_SAMPLES" \
    --coordinator-listen "$COORD_LISTEN" \
    --private \
    --output "$out"
  log "$label fill=${fill_pct}%: wrote $out"
}

run_regime() {
  local label="$1"
  local shard_log_capacity="$2"
  local true_log_capacity="$3"
  local publish_batches="$4"
  IFS=',' read -ra fill_pcts <<< "$FILL_PERCENTS"
  for pct in "${fill_pcts[@]}"; do
    run_one_fill "$label" "$shard_log_capacity" "$true_log_capacity" "$publish_batches" "$pct"
  done
}

if [[ "${SKIP_SMALL:-0}" != "1" ]]; then
  run_regime "small"  22 20 "2,4,8,16,32,64"
else
  log "small: skipped via SKIP_SMALL=1"
fi

log "done. JSON records in $OUT_DIR/"
log ""
log "for the medium (2-shard) and large (128-shard) regimes, run:"
log "    PROJECT=<gcp-project> N_SHARDS=2   ./scripts/bench-cluster.sh up   # medium"
log "    PROJECT=<gcp-project> N_SHARDS=128 ./scripts/bench-cluster.sh up   # large"
log "and chain through deploy / lookup-bench / down."
