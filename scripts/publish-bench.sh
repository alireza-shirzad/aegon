#!/usr/bin/env bash
# publish-bench.sh — local publish-time + commit-size benchmark
# driver for the small regime only.
#
# Sweeps (fill_percent × batch_size) on a single in-process shard
# and emits one JSON. The medium and large regimes both require
# real clusters — drive them via
#   N_SHARDS=2   ./scripts/bench-cluster.sh publish-bench   # medium
#   N_SHARDS=128 ./scripts/bench-cluster.sh publish-bench   # large
# which write JSONs in the same schema.
#
# Regime sizing matches `setup-bench.sh` and the project standard:
#
#   regime  | shard_log_cap | true_log_cap | kzh_k | n_shards | batch sizes
#   --------|---------------|--------------|-------|----------|---------------------------
#   small   | 22            | 20           | auto  | 1        | 2,4,8,16,32,64
#
# Fill percentages (vs. true capacity): 0, 30, 60, 90 — all four
# walked in one process, with the in-process shard rebuilt fresh
# between stages so each stage starts from a clean epoch-0 state.
#
# Tunables (env):
#   OUT_DIR          output directory (default /tmp/aegon-publish)
#   FILL_PERCENTS    comma-separated overrides (default 1,30,60,90)
#   SAMPLES_PER_BATCH per batch_size (default 3)
#   SETUP_SEED       SRS RNG seed (default 42)
#   PREFILL_SEED     prefill RNG seed base (default 1)
#   SKIP_SMALL       set to 1 to skip the small regime
#   WARMUP_BATCH_SIZE_OVERRIDE
#                    explicit value for `--warmup-batch-size`. If unset
#                    the script reads `bench-results/migration/small_best_k.txt`
#                    (or the regime-specific equivalent for the medium/large
#                    cluster path) and falls back to 16384 if absent.
#
# True capacity for any regime is the shard polynomial size divided
# by `OVER_PROVISIONING_FACTOR` (= 4); see `akd/src/aegon/config.rs`.

set -euo pipefail

OUT_DIR="${OUT_DIR:-/tmp/aegon-publish}"
FILL_PERCENTS="${FILL_PERCENTS:-1,30,60,90}"
SAMPLES_PER_BATCH="${SAMPLES_PER_BATCH:-3}"
SETUP_SEED="${SETUP_SEED:-42}"
PREFILL_SEED="${PREFILL_SEED:-1}"
mkdir -p "$OUT_DIR"

# Per-regime best-K artifact written by `scripts/migration-bench.sh`.
# When present, downstream warmup uses it; otherwise fall back to 16384.
REPO_ROOT_FOR_K="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIGRATION_DIR="$REPO_ROOT_FOR_K/bench-results/migration"
read_best_k_or_default() {
  local regime="$1"
  local default_k="$2"
  if [[ -n "${WARMUP_BATCH_SIZE_OVERRIDE:-}" ]]; then
    echo "$WARMUP_BATCH_SIZE_OVERRIDE"
    return
  fi
  local f="$MIGRATION_DIR/${regime}_best_k.txt"
  if [[ -r "$f" ]]; then
    local k
    k="$(tr -dc '0-9' < "$f")"
    if [[ -n "$k" ]]; then
      echo "$k"
      return
    fi
  fi
  echo "$default_k"
}

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
  local warmup_k
  warmup_k="$(read_best_k_or_default "$label" 16384)"
  log "$label: shard_log_capacity=$shard_log_capacity true_log_capacity=$true_log_capacity batches=$batch_sizes warmup_batch_size=$warmup_k"
  "$PB" \
    --shard-log-capacity "$shard_log_capacity" \
    --true-log-capacity "$true_log_capacity" \
    --n-shards 1 \
    --fill-percents "$FILL_PERCENTS" \
    --batch-sizes "$batch_sizes" \
    --samples-per-batch "$SAMPLES_PER_BATCH" \
    --setup-seed "$SETUP_SEED" \
    --prefill-seed "$PREFILL_SEED" \
    --warmup-batch-size "$warmup_k" \
    --private \
    --out "$out"
  log "$label: wrote $out"
}

if [[ "${SKIP_SMALL:-0}" != "1" ]]; then
  run_one "small"  22 20 "2,4,8,16,32,64"
else
  log "small: skipped via SKIP_SMALL=1"
fi

log "done. JSON records in $OUT_DIR/"
log ""
log "for the medium (2-shard) and large (128-shard) regimes, run:"
log "    PROJECT=<gcp-project> N_SHARDS=2   ./scripts/bench-cluster.sh up   # medium"
log "    PROJECT=<gcp-project> N_SHARDS=128 ./scripts/bench-cluster.sh up   # large"
log "and chain through deploy / publish-bench / down. The cluster path"
log "walks the same fill_percents = 1/30/60/90 by restarting shards"
log "between stages and emits one JSON per fill level."
