#!/usr/bin/env bash
# publish-bench.sh — local publish-time + commit-size benchmark
# driver for the small + medium regimes.
#
# Sweeps (fill_percent × batch_size) on a single in-process shard
# and emits one JSON per regime. The planetary regime needs the
# real cluster — drive it via
#   ./scripts/bench-cluster.sh publish-bench
# which writes a JSON in the same schema.
#
# Regime sizing matches `setup-bench.sh` and the project standard:
#
#   regime  | shard_log_cap | true_log_cap | kzh_k | n_shards | batch sizes
#   --------|---------------|--------------|-------|----------|---------------------------
#   small   | 22            | 20           | auto  | 1        | 2,4,8,16,32,64
#   medium  | 28            | 26           | auto  | 1        | 64,128,512,1024,2048,4096
#
# Fill percentages (vs. true capacity): 0, 30, 60, 90 — all four
# walked in one process per regime, with the in-process shard
# rebuilt fresh between stages so each stage starts from a clean
# epoch-0 state.
#
# Tunables (env):
#   OUT_DIR          output directory (default /tmp/aegon-publish)
#   FILL_PERCENTS    comma-separated overrides (default 0,30,60,90)
#   SAMPLES_PER_BATCH per batch_size (default 3)
#   SETUP_SEED       SRS RNG seed (default 42)
#   PREFILL_SEED     prefill RNG seed base (default 1)
#   SKIP_SMALL       set to 1 to skip the small regime
#   SKIP_MEDIUM      set to 1 to skip the medium regime
#
# Note on cost: medium's 90%-fill stage prefills ~60M random
# (slot, h_label, h_value) entries and commits the resulting
# polynomial — this takes time. End-to-end medium run (all four
# fills, six batches each) is typically tens of minutes; the
# small regime is single-digit minutes. Defer the planetary
# regime to the cluster path.

set -euo pipefail

OUT_DIR="${OUT_DIR:-/tmp/aegon-publish}"
FILL_PERCENTS="${FILL_PERCENTS:-0,30,60,90}"
SAMPLES_PER_BATCH="${SAMPLES_PER_BATCH:-3}"
SETUP_SEED="${SETUP_SEED:-42}"
PREFILL_SEED="${PREFILL_SEED:-1}"
mkdir -p "$OUT_DIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "building aegon_publish_bench (release)"
(cd "$REPO_ROOT" && cargo build --release -p akd --bin aegon_publish_bench) >/dev/null
PB="$REPO_ROOT/target/release/aegon_publish_bench"
[[ -x "$PB" ]] || { echo "binary missing: $PB" >&2; exit 1; }

run_one() {
  local label="$1"
  local shard_log_capacity="$2"
  local true_log_capacity="$3"
  local batch_sizes="$4"
  local out="$OUT_DIR/${label}.json"
  log "$label: shard_log_capacity=$shard_log_capacity true_log_capacity=$true_log_capacity batches=$batch_sizes"
  "$PB" \
    --shard-log-capacity "$shard_log_capacity" \
    --true-log-capacity "$true_log_capacity" \
    --n-shards 1 \
    --fill-percents "$FILL_PERCENTS" \
    --batch-sizes "$batch_sizes" \
    --samples-per-batch "$SAMPLES_PER_BATCH" \
    --setup-seed "$SETUP_SEED" \
    --prefill-seed "$PREFILL_SEED" \
    --out "$out"
  log "$label: wrote $out"
}

if [[ "${SKIP_SMALL:-0}" != "1" ]]; then
  run_one "small"  22 20 "2,4,8,16,32,64"
else
  log "small: skipped via SKIP_SMALL=1"
fi

if [[ "${SKIP_MEDIUM:-0}" != "1" ]]; then
  run_one "medium" 28 26 "64,128,512,1024,2048,4096"
else
  log "medium: skipped via SKIP_MEDIUM=1"
fi

log "done. JSON records in $OUT_DIR/"
log ""
log "for the planetary regime (2^32 dict / 32 shards / batches 4096..131072), run:"
log "    PROJECT=<gcp-project> ./scripts/bench-cluster.sh up"
log "    PROJECT=<gcp-project> ./scripts/bench-cluster.sh deploy"
log "    PROJECT=<gcp-project> ./scripts/bench-cluster.sh publish-bench"
log "(walks the same fill_percents = 0/30/60/90 by restarting shards"
log "between stages and emits one JSON per fill level.)"
