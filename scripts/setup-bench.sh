#!/usr/bin/env bash
# setup-bench.sh — local SRS setup benchmark driver (small + the
# per-shard SIZE measurement for the larger regimes).
#
# Regime sizing (matches the project's standard split). All shards
# across the medium and large regimes share an identical per-shard
# configuration — the only difference is N_SHARDS.
#
#   regime    | true dict   | shard polynomial | shard_log_cap | kzh_k | n_shards
#   ----------|-------------|------------------|---------------|-------|----------
#   small     | 2^20 entries| 2^21 slots/shard | 21            | auto  | 1
#   medium    | 2^26 entries| 2^26 slots/shard | 26 per shard  | 8     | 2
#   large     | 2^32 entries| 2^26 slots/shard | 26 per shard  | 8     | 128
#
# `auto` = optimal_kzh_k(shard_log_capacity). See akd::aegon::presets.
# The user-facing capacity for each regime is the shard polynomial size
# divided by `OVER_PROVISIONING_FACTOR` (= 2; the two-layer α = 0.5
# tuning); see `akd/src/aegon/config.rs`.
#
# What runs where:
#
#   * small: in-process via `aegon_setup_bench` on the local machine
#     (~512 MiB RAM).
#   * medium + large: a single per-shard pk/vk SIZE measurement runs
#     locally at log_cap=27 (~8-10 GiB RAM). The DISTRIBUTED setup
#     TIME for the medium and large regimes is measured on the real
#     cluster via `./scripts/bench-cluster.sh setup-bench`; that path
#     emits its own JSON in the same schema. Per-shard sizes are
#     identical across medium and large by construction, so we emit
#     one shared `per-shard.json` record.
#
# Skip individual regimes via env:
#     SKIP_SMALL=1 ./scripts/setup-bench.sh
#     SKIP_PER_SHARD=1 ./scripts/setup-bench.sh   # skip the size pass
#
# Output directory (default /tmp/aegon-setup):
#     OUT_DIR=./bench-out ./scripts/setup-bench.sh

set -euo pipefail

OUT_DIR="${OUT_DIR:-/tmp/aegon-setup}"
mkdir -p "$OUT_DIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Build the local binary release once. The dev build of the H_0 MSM
# at log_cap=28 is unbearably slow; release-mode parallel-feature is
# the only realistic mode for the medium regime.
log "building aegon_setup_bench (release)"
(cd "$REPO_ROOT" && cargo build --release -p akd --bin aegon_setup_bench) >/dev/null
SETUP_BENCH="$REPO_ROOT/target/release/aegon_setup_bench"
[[ -x "$SETUP_BENCH" ]] || { echo "binary missing: $SETUP_BENCH" >&2; exit 1; }

run_one() {
  local label="$1"
  local shard_log_capacity="$2"
  local kzh_k_arg="$3"  # either "" (use optimal_kzh_k) or "--kzh-k N"
  local out="$OUT_DIR/${label}.json"
  log "$label: shard_log_capacity=$shard_log_capacity $kzh_k_arg"
  # shellcheck disable=SC2086
  "$SETUP_BENCH" \
    --label "$label" \
    --shard-log-capacity "$shard_log_capacity" \
    $kzh_k_arg \
    --out "$out"
  log "$label: wrote $out"
}

if [[ "${SKIP_SMALL:-0}" != "1" ]]; then
  run_one "small" 21 ""
else
  log "small: skipped via SKIP_SMALL=1"
fi

if [[ "${SKIP_PER_SHARD:-0}" != "1" ]]; then
  # Per-shard size measurement at the shared medium/large config
  # (shard_log_cap=26, kzh_k=8). Time on a single-machine run is
  # meaningless — the real measurement is distributed across the
  # cluster — but the pk/vk sizes are identical to one shard of
  # either the medium or large cluster, so this file feeds the
  # "key sizes" plot for both regimes.
  run_one "per-shard" 26 "--kzh-k 8"
else
  log "per-shard: skipped via SKIP_PER_SHARD=1"
fi

log "all done. JSON records in $OUT_DIR/"
log ""
log "for the medium + large distributed-setup TIME measurements, run:"
log "    PROJECT=<gcp-project> N_SHARDS=2   ./scripts/bench-cluster.sh up"
log "    PROJECT=<gcp-project> N_SHARDS=128 ./scripts/bench-cluster.sh up"
log "and chain through deploy / setup-bench / publish-bench / lookup-bench / down."
