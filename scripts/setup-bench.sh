#!/usr/bin/env bash
# setup-bench.sh — three-regime SRS setup benchmark driver.
#
# Runs aegon_setup_bench three times for the small/medium/planetary
# regimes and collects the JSON records into one consolidated file.
#
# Regime sizing (matches the project's standard split):
#
#   regime       | dictionary | over-prov | shard_log_cap | kzh_k | n_shards
#   -------------|------------|-----------|---------------|-------|----------
#   small        | 2^20       | 2^22      | 22            | auto  | 1
#   medium       | 2^26       | 2^28      | 28            | auto  | 1
#   planetary    | 2^32       | 2^34      | 29 per shard  | 10    | 32
#
# `auto` = optimal_kzh_k(shard_log_capacity). See akd::aegon::presets.
#
# What runs where:
#
#   * small + medium: in-process via `aegon_setup_bench` on the local
#     machine. Small needs ~512 MiB RAM; medium needs ~30 GiB RAM
#     (the full H_0 lives in memory during gen).
#   * planetary: only the per-shard pk/vk SIZE measurement runs
#     locally (needs ~40 GiB RAM to hold one shard's SRS in memory at
#     log_cap=29). The DISTRIBUTED setup TIME for the planetary regime
#     is measured on the real cluster via
#     `./scripts/bench-cluster.sh setup-bench`; that path emits its
#     own JSON in the same schema.
#
# Skip individual regimes via env:
#     SKIP_SMALL=1 ./scripts/setup-bench.sh
#     SKIP_MEDIUM=1 ./scripts/setup-bench.sh
#     SKIP_PLANETARY=1 ./scripts/setup-bench.sh
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
  run_one "small" 22 ""
else
  log "small: skipped via SKIP_SMALL=1"
fi

if [[ "${SKIP_MEDIUM:-0}" != "1" ]]; then
  run_one "medium" 28 ""
else
  log "medium: skipped via SKIP_MEDIUM=1"
fi

if [[ "${SKIP_PLANETARY:-0}" != "1" ]]; then
  # Per-shard size measurement at the planetary parameters. Time on
  # this regime is meaningless (single-machine, while the real
  # measurement is distributed across 32 shards) — it's emitted so
  # downstream plotting can show the gap between sequential and
  # distributed gen.
  run_one "planetary-per-shard" 29 "--kzh-k 10"
else
  log "planetary-per-shard: skipped via SKIP_PLANETARY=1"
fi

log "all done. JSON records in $OUT_DIR/"
log ""
log "for the planetary DISTRIBUTED-SETUP TIME + comm-bytes measurement, run:"
log "    PROJECT=<gcp-project> ./scripts/bench-cluster.sh up"
log "    PROJECT=<gcp-project> ./scripts/bench-cluster.sh deploy"
log "    PROJECT=<gcp-project> ./scripts/bench-cluster.sh setup-bench"
log "(emits /tmp/aegon-setup-bench.json in the same schema as the local runs.)"
