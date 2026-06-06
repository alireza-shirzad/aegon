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
# Regime sizing matches `setup-bench.sh` and the publish bench. The
# two-layer rule (α = 0.5, see `shard_log_capacity_for_two_layer` in
# `akd/src/aegon/config.rs`) drops `shard_log_cap` by one bit vs. the
# old single-shard α = 0.25:
#
#   regime  | shard_log_cap | true_log_cap | n_shards | K sweep
#   --------|---------------|--------------|----------|-------------------------
#   small   | 21            | 20           | 1        | 1024,4096,16384,65536
#   medium  | 26            | 26           | 2        | 4096,16384,65536,262144
#   large   | 26            | 32           | 128      | 16384,65536,262144,1048576
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

# Print a full per-K inventory (K, elapsed, throughput) parsed
# straight from the per-K JSONs on disk. Always runs, even after a
# binary crash — that's the whole point: if the OOM-killer takes
# the binary out mid-sweep, the per-K JSONs the binary did manage
# to write (one atomic file per completed K) are the durable record.
# We sort numerically by K so the table reads top-to-bottom in
# sweep order.
print_k_inventory() {
  local regime="$1"
  shopt -s nullglob
  local files=( "$OUT_DIR/${regime}_K"*.json )
  shopt -u nullglob
  if (( ${#files[@]} == 0 )); then
    log "  inventory: no per-K JSONs on disk for regime=$regime"
    return
  fi
  # Build "K\tline" so a numeric sort respects K-order.
  log "  K-sweep inventory for regime=$regime:"
  for f in "${files[@]}"; do
    local k ms tp
    k="$(grep -oE '"chunk_size":[[:space:]]*[0-9]+' "$f" \
         | head -1 | grep -oE '[0-9]+')"
    ms="$(grep -A2 '"total":' "$f" \
          | grep -oE '"elapsed_ms":[[:space:]]*[0-9.]+' \
          | head -1 | grep -oE '[0-9.]+')"
    tp="$(grep -A3 '"total":' "$f" \
          | grep -oE '"throughput_users_per_sec":[[:space:]]*[0-9.]+' \
          | head -1 | grep -oE '[0-9.]+')"
    if [[ -z "$k" || -z "$tp" || -z "$ms" ]]; then
      log "    warn: could not parse K/elapsed/throughput from $f"
      continue
    fi
    printf "%s\t    K=%-8s elapsed=%8.1f s   throughput=%8.0f users/sec\n" \
      "$k" "$k" "$(awk "BEGIN { print $ms/1000 }")" "$tp"
  done | sort -n | cut -f2- | while IFS= read -r line; do log "$line"; done
}

# Pick the K with maximum throughput_users_per_sec across the per-K
# JSONs of one regime. Pure-bash JSON probe — we only read two
# numbers per file, so a `grep` + `awk` compare is sufficient and
# avoids a Python dep.
#
# We rank by THROUGHPUT, not min elapsed_ms — single-batch-per-K
# mode publishes a different number of users per K, so elapsed
# trivially scales with K and the only meaningful ordering is
# throughput. In full-climb mode every K hits the same target_count,
# so throughput and 1/elapsed agree, and ranking by either gives
# the same K.
pick_best_k() {
  local regime="$1"
  local best_k="" best_tp=""
  shopt -s nullglob
  for f in "$OUT_DIR/${regime}_K"*.json; do
    local k tp
    k="$(grep -oE '"chunk_size":[[:space:]]*[0-9]+' "$f" \
         | head -1 | grep -oE '[0-9]+')"
    tp="$(grep -A3 '"total":' "$f" \
          | grep -oE '"throughput_users_per_sec":[[:space:]]*[0-9.]+' \
          | head -1 | grep -oE '[0-9.]+')"
    if [[ -z "$k" || -z "$tp" ]]; then
      log "  warn: could not parse k/throughput from $f, skipping"
      continue
    fi
    if [[ -z "$best_tp" ]] || awk "BEGIN { exit !($tp > $best_tp) }"; then
      best_k="$k"
      best_tp="$tp"
    fi
  done
  shopt -u nullglob
  if [[ -z "$best_k" ]]; then
    log "  warn: no per-K JSONs found for regime=$regime, not writing best_k.txt"
    return
  fi
  echo "$best_k" > "$OUT_DIR/${regime}_best_k.txt"
  log "  picked K=$best_k (throughput=${best_tp} users/sec) for regime=$regime"
  log "  wrote $OUT_DIR/${regime}_best_k.txt"
}

# Classify a non-zero exit code from the bench binary. 128+N means
# "killed by signal N". 137 = SIGKILL, overwhelmingly the OOM-killer
# on Linux (the kernel sends SIGKILL when it OOMs a process). 143 =
# SIGTERM, usually a manual Ctrl-C-during-sleep or a CI timeout.
# Anything else: a real binary error.
classify_bench_exit() {
  local rc="$1"
  case "$rc" in
    0)   echo "ok" ;;
    137) echo "OOM-killed (SIGKILL, code 137) — almost certainly the kernel OOM-killer firing on the next K. The K's that completed before this point are still on disk." ;;
    143) echo "killed by SIGTERM (code 143) — manual kill or external timeout." ;;
    *)   echo "exited with code $rc — see the bench log for the failing K's error message." ;;
  esac
}

run_regime() {
  local label="$1"
  local shard_log_capacity="$2"
  local true_log_capacity="$3"
  local default_k_sweep="$4"
  local k_sweep="${K_SWEEP:-$default_k_sweep}"
  # SINGLE_BATCH=1 (default): one publish per K against the
  # cleared empty dictionary. Cheap — total wall = sum(K)/throughput,
  # which for the small sweep (1024+4096+16384+65536) is ~30s
  # vs ~25 min for the full 90% climb. Pick by max throughput.
  # SINGLE_BATCH=0 forces the full climb in case you suspect K
  # ordering flips across fill levels.
  local mode_flag=""
  local mode_summary=""
  if [[ "${SINGLE_BATCH:-1}" == "1" ]]; then
    mode_flag="--single-batch-per-k"
    mode_summary="sweep=single-batch-per-K"
  else
    mode_flag=""
    mode_summary="target_fill=${TARGET_FILL_PERCENT}%"
  fi
  log "$label: shard_log_capacity=$shard_log_capacity true_log_capacity=$true_log_capacity \
$mode_summary K-sweep=[$k_sweep]"
  # Disable `set -e` for the bench invocation only — OOM-killing a
  # K shouldn't abort the rest of the script. The per-K JSONs are
  # written atomically (Rust-side temp + rename), so anything that
  # completed before the crash is durable, and pick_best_k can rank
  # over what's actually on disk. After the call, always run the
  # inventory + picker.
  set +e
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
    $mode_flag \
    --out-template "$OUT_DIR/${label}_K{K}.json"
  local rc=$?
  set -e
  if (( rc != 0 )); then
    log "$label: WARN: $(classify_bench_exit "$rc")"
  fi
  print_k_inventory "$label"
  pick_best_k "$label"
}

if [[ "${SKIP_SMALL:-0}" != "1" ]]; then
  # Default sweep extends up to 524288 — the validated throughput
  # peak on a 64 GB box at shard_log_capacity=22. K=1048576 OOMs
  # under default RAM (the KZH-k commit's intermediate state for a
  # 1M-update batch on a 2^22 polynomial overflows ~60 GB). Override
  # via K_SWEEP if you want to chase the plateau on a bigger box.
  run_regime "small" 21 20 "1024,4096,16384,65536,131072,262144,524288"
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
