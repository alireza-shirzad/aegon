#!/usr/bin/env bash
# migration-bench.sh — sweep --chunk-size K over a regime, time
# the migration from epoch 0 to a target fill, pick the best K,
# and write it to a per-regime artifact so the downstream
# publish/lookup benches read it as their `--warmup-batch-size`.
#
# Local in-process for the small regime; medium / large require
# real clusters — drive them via
#   N_SHARDS=2   ./scripts/bench-cluster.sh migration-bench   # medium
#   N_SHARDS=128 ./scripts/bench-cluster.sh migration-bench   # large
# (the cluster script needs a matching subcommand wired in — see
# `bench-cluster.sh` for the existing `publish-bench` / `lookup-bench`
# pattern to copy).
#
# Regime sizing matches `setup-bench.sh` and the publish bench:
#
#   regime  | shard_log_cap | true_log_cap | n_shards | K sweep
#   --------|---------------|--------------|----------|-------------------------
#   small   | 22            | 20           | 1        | 1024,4096,16384,65536
#   medium  | 27            | 26           | 2        | 4096,16384,65536,262144
#   large   | 27            | 32           | 128      | 16384,65536,262144,1048576
#
# Architecture: ONE `aegon_migration_bench` invocation per regime.
# The binary holds a single in-process `ShardedAegon` across the
# entire K-sweep — between K's it calls `clear_dictionary` to wipe
# the shards back to epoch-0 empty without rebuilding the SRS or
# tearing down the cluster. Per-K JSONs are written via the
# `--out-template` placeholder, then a `bash` post-pass picks the
# fastest K. (Old behaviour: one process per K, paying SRS regen
# every time. That cost dominated the small regime's total wall.)
#
# Tunables (env):
#   OUT_DIR             output directory (default bench-results/migration)
#   K_SWEEP             override comma-sep K list (default per regime)
#   TARGET_FILL_PERCENT default 90 — climb up to this fill before picking K
#   MILESTONE_FILLS     default 1,5,10,30,60,90 — recorded in the per-K JSON
#   SETUP_SEED          default 42
#   PREFILL_SEED        default 1
#   SKIP_SMALL          set to 1 to skip the small regime
#
# Output:
#   bench-results/migration/{regime}_K{K}.json   one per K
#   bench-results/migration/{regime}_best_k.txt  integer; lowest total
#                                                migration time wins.
#                                                Downstream benches read
#                                                this file (see
#                                                publish-bench.sh's
#                                                read_best_k_or_default).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/bench-results/migration}"
TARGET_FILL_PERCENT="${TARGET_FILL_PERCENT:-90}"
MILESTONE_FILLS="${MILESTONE_FILLS:-1,5,10,30,60,90}"
SETUP_SEED="${SETUP_SEED:-42}"
PREFILL_SEED="${PREFILL_SEED:-1}"
mkdir -p "$OUT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "building aegon_migration_bench (release)"
(cd "$REPO_ROOT" && cargo build --release -p akd --bin aegon_migration_bench) >/dev/null
MB="$REPO_ROOT/target/release/aegon_migration_bench"
[[ -x "$MB" ]] || { echo "binary missing: $MB" >&2; exit 1; }

# Pick the K with minimum total elapsed_ms across the per-K JSONs
# of one regime. Pure-bash JSON probe — we only read one number per
# file, so a `grep` + `sort` is sufficient and avoids a Python dep.
pick_best_k() {
  local regime="$1"
  local best_k="" best_ms=""
  shopt -s nullglob
  for f in "$OUT_DIR/${regime}_K"*.json; do
    local k ms
    # `chunk_size` and `elapsed_ms` are both unique in the JSON.
    k="$(grep -oE '"chunk_size":[[:space:]]*[0-9]+' "$f" \
         | head -1 | grep -oE '[0-9]+')"
    ms="$(grep -A2 '"total":' "$f" \
          | grep -oE '"elapsed_ms":[[:space:]]*[0-9.]+' \
          | head -1 | grep -oE '[0-9.]+')"
    if [[ -z "$k" || -z "$ms" ]]; then
      log "  warn: could not parse k/elapsed_ms from $f, skipping"
      continue
    fi
    if [[ -z "$best_ms" ]] || awk "BEGIN { exit !($ms < $best_ms) }"; then
      best_k="$k"
      best_ms="$ms"
    fi
  done
  shopt -u nullglob
  if [[ -z "$best_k" ]]; then
    log "  warn: no per-K JSONs found for regime=$regime, not writing best_k.txt"
    return
  fi
  echo "$best_k" > "$OUT_DIR/${regime}_best_k.txt"
  log "  picked K=$best_k (total elapsed=${best_ms} ms) for regime=$regime"
  log "  wrote $OUT_DIR/${regime}_best_k.txt"
}

run_regime() {
  local label="$1"
  local shard_log_capacity="$2"
  local true_log_capacity="$3"
  local default_k_sweep="$4"
  local k_sweep="${K_SWEEP:-$default_k_sweep}"
  log "$label: shard_log_capacity=$shard_log_capacity true_log_capacity=$true_log_capacity \
target_fill=${TARGET_FILL_PERCENT}% K-sweep=[$k_sweep]"
  # One process for the whole sweep: the binary clears the
  # dictionary between K's instead of paying SRS regen per K.
  "$MB" \
    --shard-log-capacity "$shard_log_capacity" \
    --true-log-capacity "$true_log_capacity" \
    --n-shards 1 \
    --target-fill-percent "$TARGET_FILL_PERCENT" \
    --chunk-sizes "$k_sweep" \
    --milestone-fills "$MILESTONE_FILLS" \
    --setup-seed "$SETUP_SEED" \
    --prefill-seed "$PREFILL_SEED" \
    --private \
    --out-template "$OUT_DIR/${label}_K{K}.json"
  pick_best_k "$label"
}

if [[ "${SKIP_SMALL:-0}" != "1" ]]; then
  run_regime "small" 22 20 "1024,4096,16384,65536"
else
  log "small: skipped via SKIP_SMALL=1"
fi

log "done. JSONs + best_k.txt in $OUT_DIR/"
log ""
log "for medium (2-shard) and large (128-shard) regimes, run them on a cluster"
log "via the 'migration-bench' subcommand on bench-cluster.sh:"
log "    PROJECT=<gcp-project> N_SHARDS=2   MIGRATION_REGIME=medium ./scripts/bench-cluster.sh up bootstrap migration-bench"
log "    PROJECT=<gcp-project> N_SHARDS=128 MIGRATION_REGIME=large  ./scripts/bench-cluster.sh up bootstrap migration-bench"
log "(per-K JSONs land in \$LOCAL_MIGRATION_BENCH_DIR; copy them into $OUT_DIR/"
log "alongside the small JSONs to feed the publish/lookup benches' warmup batch.)"
