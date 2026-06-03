#!/usr/bin/env bash
# bench-cluster.sh — spin up a production-scale Aegon shard cluster and
# run aegon_coordinator_bench against it.
#
# Subcommands:
#   up         Provision VPC + firewall + N high-memory shard VMs + coordinator VM
#   deploy     Build binaries locally (in a Docker linux/amd64 container on
#              macOS), scp aegon_shard_server to every shard VM, scp
#              aegon_coordinator_bench + aegon_srs_bootstrap to the
#              coordinator. Each shard binds its SrsService on :SRS_PORT
#              and parks in "awaiting-bootstrap" state — no SRS or
#              prefill work yet. Returns quickly.
#   bootstrap  Run aegon_srs_bootstrap on the coordinator: sample
#              trapdoors deterministically from --setup-seed, push them
#              to every shard, then poll WaitForReady until each shard
#              has run distributed SRS gen (or cache hit) + Aegon init
#              + prefill. First boot per (log_cap, k, seed) does the
#              full distributed exchange; subsequent boots hit the cache
#              and finish in seconds.
#   bench      Run aegon_coordinator_bench on the coordinator, retrieve
#              /tmp/aegon-bench.json from it
#   logs N     Tail the shard log on aegon-shard-N
#   down       Delete every instance + the VPC this script created
#
# Required env (or defaults):
#   PROJECT             GCP project ID                (no default; required)
#   ZONE                GCE zone                      (us-central1-f)
#   N_SHARDS            number of shards (power of 2) (128)
#   SHARD_LOG_CAPACITY  log_2 slots per shard         (27)
#   KZH_K               KZH-k block parameter         (9 — optimal_kzh_k(27))
#   TOTAL_PRELOAD_LOG2  log_2 of total prefilled users (8 → 2^8=256 split across shards)
#   BATCH_SIZES         comma-separated sweep sizes   (2,4,8,...,16384)
#   SAMPLES_PER_BATCH   timed publishes per batch     (1)
#   SETUP_SEED          deterministic SRS gen seed    (42)
#   PREFILL_SEED        deterministic prefill seed    (1)
#   SHARD_MACHINE_TYPE  GCE machine type for shards   (n2-standard-16 — 64 GB RAM)
#   COORD_MACHINE_TYPE  GCE machine type coordinator  (defaults to SHARD_MACHINE_TYPE = n2-standard-16)
#
# At the defaults above (N_SHARDS=128, SHARD_LOG_CAPACITY=27, KZH_K=9,
# PUBLISH_TRUE_LOG_CAP=32):
#   * Each shard owns one 2^27-slot polynomial (α=4 over-provisioning of
#     a 2^25-entry per-shard slice — total dictionary 2^32 entries).
#   * Per-shard KZH-k SRS at log_cap=27 / kzh_k=9: ~8.6 GiB on disk,
#     same in RAM during gen. Per-shard peak memory at the 90% fill
#     stage (~30 M entries/shard, the heaviest workload) lands around
#     ~50-55 GiB — fits in 64 GiB with margin.
#   * 128 × n2-standard-16 ≈ $100/hr on-demand; ~$30/hr with 3-year
#     committed-use. Tear down promptly when not benching.
#   * Previous defaults (32 shards × log_cap=29) exceeded n2-standard-16
#     RAM at 60%+ fill. The 128/27 split keeps all four fill stages
#     (1/30/60/90%) on commodity n2-standard-16 with no per-stage
#     reconfiguration.
#   * Redis is provisioned on its own small VM. In the current
#     architecture (coord-owns-everything), the shard's `--db-url` is
#     a no-op and the shard prefill writes nothing to Redis — the
#     coord's open-addressing probe falls back to a gRPC
#     `is_index_slot_occupied` against the owning shard on local-DB
#     miss. At the default light preload (256 entries / 2^34 capacity)
#     the collision rate is essentially zero, so the fallback rarely
#     fires. `deploy` still `FLUSHALL`s before starting to reset the
#     coord's own keyspace from prior runs.
#
# This script is a prototyping aid, not production infrastructure.

set -euo pipefail

PROJECT="${PROJECT:-}"
ZONE="${ZONE:-us-central1-f}"
N_SHARDS="${N_SHARDS:-128}"
SHARD_LOG_CAPACITY="${SHARD_LOG_CAPACITY:-27}"
KZH_K="${KZH_K:-9}"
# Default: 2^8 = 256 users preloaded total, evenly split across shards.
# This is the "light preload" baseline used by the 2^34-capacity bench.
TOTAL_PRELOAD_LOG2="${TOTAL_PRELOAD_LOG2:-8}"
# Default: power-of-two sweep from 2^1 to 2^14. Each batch is a single
# `ShardedAegon::publish` call that fans out across all 32 shards.
BATCH_SIZES="${BATCH_SIZES:-2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384}"
# Default: 1 sample per batch size. At log_cap=29 publish wall time
# climbs steeply with batch size; one sample per size keeps the whole
# sweep tractable. Bump to 3-5 for noise statistics.
SAMPLES_PER_BATCH="${SAMPLES_PER_BATCH:-1}"
SETUP_SEED="${SETUP_SEED:-42}"
PREFILL_SEED="${PREFILL_SEED:-1}"
SHARD_MACHINE_TYPE="${SHARD_MACHINE_TYPE:-n2-standard-16}"
# Coordinator uses the same machine type as the shards by default. The
# coordinator's in-memory footprint (open-addressing slot index over the
# 2^true_log_capacity keyspace + per-shard connection/batch buffers +
# RocksDB memtables) scales with keyspace size and shard count, so the
# small n2-standard-4 (16 GB) that sufficed for the medium regime OOM-kills
# at the large regime (2^32 keyspace, 128 shards). Matching the shard type
# keeps every node's CPU/RAM identical across the cluster; override with
# COORD_MACHINE_TYPE=... if you need a different size.
COORD_MACHINE_TYPE="${COORD_MACHINE_TYPE:-$SHARD_MACHINE_TYPE}"
# Coordinator boot disk is shared with the RocksDB store at
# $COORD_DB_PATH. Must fit the AKD history for the deepest fill we
# drive. Empirical: ~24KB/entry on disk (LSM overhead included).
# Sized for large at 10% fill (430M entries ≈ 10TB), with headroom
# for compaction churn: 12TB pd-ssd. For medium (60M @ 90%) the
# extra capacity is wasted but cheap relative to a re-run.
COORD_BOOT_DISK_SIZE="${COORD_BOOT_DISK_SIZE:-12TB}"
COORD_BOOT_DISK_TYPE="${COORD_BOOT_DISK_TYPE:-pd-ssd}"
# bench-client RocksDB lives on its own disk; the bench's in-process
# coord state climbs through fill levels via real publish, so this is
# the disk that actually fills up during lookup-bench. Sized larger
# than the coord disk to absorb RocksDB compaction spikes at the
# highest fills (~10% of 2^32 = ~10 TB steady state, +headroom).
BENCH_CLIENT_BOOT_DISK_SIZE="${BENCH_CLIENT_BOOT_DISK_SIZE:-20TB}"
BENCH_CLIENT_BOOT_DISK_TYPE="${BENCH_CLIENT_BOOT_DISK_TYPE:-pd-ssd}"
# Bench-client machine type. Distinct from $COORD_MACHINE_TYPE because
# the bench-client holds an in-process coord state that scales with
# fill level (baseline ~26 GB for the 2^32 open-addressing index + ~165
# bytes per filled entry), and the n2-standard-16 (64 GB) coord type
# OOM-killed the large run at ~5% fill. Override per-regime in
# run-*-cluster.sh; defaults to $COORD_MACHINE_TYPE so small/medium
# (which never approach the RAM ceiling) stay on the cheap node.
BENCH_CLIENT_MACHINE_TYPE="${BENCH_CLIENT_MACHINE_TYPE:-$COORD_MACHINE_TYPE}"

NETWORK="aegon-bench-vpc"
FIREWALL_GRPC="aegon-bench-grpc"
FIREWALL_SRS="aegon-bench-srs"
FIREWALL_SSH="aegon-bench-ssh"
SHARD_TAG="aegon-bench-shard"
COORD_TAG="aegon-bench-coord"
MASKING_TAG="aegon-bench-masking"
# Bench-client VM: runs aegon_lookup_bench. Separate from the coord so
# the bench's request-driving CPU doesn't contend with the coord's
# tonic worker pool, and lookup RPCs hit the coord's gRPC stack over
# the real intra-VPC network instead of localhost loopback. Latency
# is read from the response's `server_processing_micros` field so we
# never count the RTT we just inserted.
BENCH_CLIENT_TAG="aegon-bench-client"
FIREWALL_BENCH_CLIENT_GRPC="aegon-bench-client-grpc"
# Tag + label applied to every VM the script creates, signalling
# to the university SOC scanner that these are short-lived research
# benchmark workers. Apply BOTH a network tag (some scanners look at
# instance tags) and a label (most security policies filter on labels).
# Per the SOC ask: use exactly "temporary-worker-vm" as the marker.
SCANNER_TAG="temporary-worker-vm"
SCANNER_LABEL="temporary-worker-vm=true"
SHARD_PORT=50051
# Masking server(s): N_MASKING_SERVERS VMs per cluster. Each holds a
# queue of pre-built `KZHKMaskingPackage`s; background producers
# refill the queue continuously. Each shard's value-side opening
# fetches a package from a masking server instead of generating one
# inline — turns the per-lookup MSM into a sub-ms gRPC fetch.
#
# The shard server's `--masking-addr` takes a *list* (comma-separated
# or repeated) and dispatches package fetches across all of them
# round-robin via `MaskingClientPool`. Every shard gets the SAME
# endpoint list, so per-shard masking throughput becomes N × single-
# server rate regardless of shard count. This matters at small/medium
# (N_SHARDS ≤ 2) where the old per-shard pinning pinned all load to
# the first 1–2 masking servers and left the rest idle.
# Standalone masking throughput is ~120 pkg/sec for nv=27/k=9 and
# ~185 pkg/sec for nv=22/k=7. For 3 k QPS at large (nv=27/k=9), N≈25
# barely makes it; we run 35 for headroom. Small/medium use 4.
N_MASKING_SERVERS="${N_MASKING_SERVERS:-1}"
MASKING_PORT="${MASKING_PORT:-50061}"
MASKING_MACHINE_TYPE="${MASKING_MACHINE_TYPE:-n2-standard-16}"
MASKING_QUEUE_SIZE="${MASKING_QUEUE_SIZE:-512}"
# Producer count defaults to "all cores"; the binary picks
# `std::thread::available_parallelism()` if unset, but we pin it
# here so the masking server's behavior is explicit.
MASKING_PRODUCERS="${MASKING_PRODUCERS:-16}"
# Set ENABLE_MASKING_SERVER=0 to skip the masking VM entirely
# (shards fall back to inline package generation per opening).
ENABLE_MASKING_SERVER="${ENABLE_MASKING_SERVER:-1}"
# Coordinator-side RocksDB. The coordinator stores its open-
# addressing occupancy index and per-shard state checkpoints in
# this directory. Used by aegon_publish_bench, aegon_lookup_bench,
# and aegon_coordinator_bench via --db-path. Lives on the
# coordinator VM's local disk — no Redis VM, no network hop for DB
# ops, no MULTI/EXEC transaction-size limits. We wipe it between
# bench invocations so each run starts from a clean keyspace.
COORD_DB_PATH="/opt/aegon/coord-db"
# Per-shard local SRS deployment. Every shard generates its own
# SRS in parallel during setup-bench, using the same SETUP_SEED so
# they all converge on the same file (deterministic). The file
# lives on local disk at $REMOTE_SRS_PATH so subsequent restarts
# (publish-bench / lookup-bench cycles) just read it back — no
# NFS, no broadcast, no central gen, no cascading failure. Writer
# uses serialize_uncompressed so the read path skips the
# per-point sqrt and finishes in ~1 min instead of ~25.
SRS_FILENAME="cluster.srs"
REMOTE_SRS_PATH="${REMOTE_SRS_PATH:-/opt/aegon/srs/$SRS_FILENAME}"
REMOTE_SRS_DIR="/opt/aegon/srs"
# LOCAL_SRS_PATH is set further below, after REPO_ROOT.
ROUTER="aegon-bench-router"
NAT="aegon-bench-nat"
REGION="${ZONE%-*}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_SRS_PATH="${LOCAL_SRS_PATH:-$REPO_ROOT/bench-results/srs/aegon-srs-lc${SHARD_LOG_CAPACITY}-k${KZH_K}.bin}"
REMOTE_BIN_DIR="/opt/aegon/bin"
REMOTE_BENCH_OUT="/tmp/aegon-bench.json"
LOCAL_BENCH_OUT="${LOCAL_BENCH_OUT:-/tmp/aegon-bench.json}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

require_project() {
  if [[ -z "$PROJECT" ]]; then
    die "PROJECT env var is required (export PROJECT=your-gcp-project-id)"
  fi
  gcloud config set project "$PROJECT" >/dev/null 2>&1 || \
    die "could not set project to $PROJECT — is gcloud authenticated?"
}

require_power_of_two() {
  local n="$1"
  (( n > 0 && (n & (n - 1)) == 0 )) || die "N_SHARDS must be a power of two (got $n)"
}

# Prefill per shard = total_preload / N_SHARDS = 2^(TOTAL_PRELOAD_LOG2) / N_SHARDS.
prefill_per_shard() {
  python3 -c "print(2**${TOTAL_PRELOAD_LOG2} // ${N_SHARDS})"
}

wait_for_ssh() {
  # Probe SSH on `$1` until it succeeds or we hit the timeout.
  # cmd_deploy + setup-bench's first SSH calls can fire before
  # cloud-init's sshd is fully up (the IAP tunnel returns
  # "Failed to connect to port 22"), so retry instead of bailing.
  #
  # Each probe is wrapped in `timeout` because IAP's start-iap-tunnel
  # subprocess can hang silently (no timeout flag of its own) if the
  # tunnel handshake never completes — we've seen this strand the
  # whole deploy phase indefinitely.
  local instance="$1"
  local max_tries="${2:-30}"
  local per_try_secs="${3:-25}"
  local i
  for ((i = 1; i <= max_tries; i++)); do
    if timeout "$per_try_secs" gcloud compute ssh "$instance" \
         --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
         --ssh-flag="-o UserKnownHostsFile=/dev/null" \
         --ssh-flag="-o StrictHostKeyChecking=no" \
         --ssh-flag="-o LogLevel=ERROR" \
         --command="true" \
         >/dev/null 2>&1; then
      log "[$instance] SSH ready after $i attempt(s)"
      return 0
    fi
    sleep 5
  done
  die "[$instance] SSH not ready after $((max_tries * (per_try_secs + 5)))s"
}

remote() {
  local instance="$1"
  local cmd="$2"
  # VM names get reused across cluster create/destroy cycles, so new
  # VMs trip strict host-key checking against stale entries in
  # ~/.ssh/google_compute_known_hosts. IAP is the real auth boundary;
  # bypass the SSH-layer host-key check.
  local ssh_flags=(
    --ssh-flag="-o UserKnownHostsFile=/dev/null"
    --ssh-flag="-o StrictHostKeyChecking=no"
    --ssh-flag="-o LogLevel=ERROR"
  )
  if [[ "${3:-}" == "stream" ]]; then
    gcloud compute ssh "$instance" --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no "${ssh_flags[@]}" --command="$cmd"
  elif [[ "${3:-}" == "fire-and-forget" ]]; then
    # IAP-tunnel ssh teardown can hang for minutes after the remote
    # command exits. For start commands where the remote process is
    # already detached (setsid + nohup), we don't care whether the
    # gcloud client cleans up — kill it after a short grace window.
    # Portable timeout (macOS has neither `timeout` nor `gtimeout`
    # out of the box): background gcloud, sleep, send SIGTERM.
    #
    # 120s grace: IAP tunnel handshake alone can take 10-20s and on
    # cold sessions we've seen 30-60s. Cutting the SSH session before
    # the remote shell reaches `nohup ... &` leaves the shard server
    # un-launched and silently absent at bootstrap time.
    gcloud compute ssh "$instance" --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no \
      --quiet "${ssh_flags[@]}" --command="$cmd" &
    local _ssh_pid=$!
    ( sleep 120 && kill "$_ssh_pid" 2>/dev/null ) &
    local _watchdog_pid=$!
    wait "$_ssh_pid" 2>/dev/null || true
    kill "$_watchdog_pid" 2>/dev/null || true
  else
    gcloud compute ssh "$instance" --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet "${ssh_flags[@]}" --command="$cmd"
  fi
}

# Same rationale as remote(): bypass host-key checks for scp too.
SCP_HOSTKEY_FLAGS=(
  --scp-flag="-o UserKnownHostsFile=/dev/null"
  --scp-flag="-o StrictHostKeyChecking=no"
  --scp-flag="-o LogLevel=ERROR"
)

scp_to() {
  local instance="$1"; shift
  gcloud compute scp --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet "${SCP_HOSTKEY_FLAGS[@]}" "$@" "$instance:/tmp/"
}

scp_from() {
  local instance="$1"; shift
  local src="$1"; shift
  local dst="$1"; shift
  gcloud compute scp --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet "${SCP_HOSTKEY_FLAGS[@]}" "$instance:$src" "$dst"
}

shard_name() { echo "${SHARD_TAG}-$1"; }
coord_name() { echo "${COORD_TAG}"; }
# Indexed masking VM name. Each masking VM gets a sequence number 0..N-1
# (e.g. aegon-bench-masking-0, aegon-bench-masking-1, ...). The shared
# MASKING_TAG is still used for firewall source-tag matching, so every
# masking VM gets that tag at create time.
masking_name() { echo "${MASKING_TAG}-$1"; }
bench_client_name() { echo "${BENCH_CLIENT_TAG}"; }

coord_internal_ip() {
  gcloud compute instances describe "$(coord_name)" --zone="$ZONE" \
    --format='value(networkInterfaces[0].networkIP)'
}

shard_internal_ip() {
  gcloud compute instances describe "$(shard_name "$1")" --zone="$ZONE" \
    --format='value(networkInterfaces[0].networkIP)'
}

shard_endpoints_csv() {
  local out=()
  for ((i = 0; i < N_SHARDS; i++)); do
    out+=("http://$(shard_internal_ip "$i"):$SHARD_PORT")
  done
  local IFS=,
  echo "${out[*]}"
}

cmd_up() {
  require_project
  require_power_of_two "$N_SHARDS"

  local per_shard; per_shard="$(prefill_per_shard)"
  log "PROJECT=$PROJECT ZONE=$ZONE N_SHARDS=$N_SHARDS"
  log "shard_log_capacity=$SHARD_LOG_CAPACITY kzh_k=$KZH_K"
  log "total preload = 2^$TOTAL_PRELOAD_LOG2 → ${per_shard} entries per shard"
  log "shard machine type=$SHARD_MACHINE_TYPE, coord machine type=$COORD_MACHINE_TYPE"

  # ---- VPC ----
  if gcloud compute networks describe "$NETWORK" >/dev/null 2>&1; then
    log "VPC $NETWORK already exists, skipping create"
  else
    log "creating VPC $NETWORK"
    gcloud compute networks create "$NETWORK" --subnet-mode=auto >/dev/null
  fi

  # ---- Cloud NAT (outbound for VMs with no external IP) ----
  # Even without Redis on aegon-db, we still want NAT so the shard VMs
  # can pull a system Python/CA-cert update if they need to. Cheap and
  # subnet-wide.
  if gcloud compute routers describe "$ROUTER" --region="$REGION" >/dev/null 2>&1; then
    log "router $ROUTER exists"
  else
    log "creating cloud router $ROUTER ($REGION)"
    gcloud compute routers create "$ROUTER" \
      --network="$NETWORK" --region="$REGION" >/dev/null
  fi
  if gcloud compute routers nats describe "$NAT" --router="$ROUTER" --region="$REGION" >/dev/null 2>&1; then
    log "cloud nat $NAT exists"
  else
    log "creating cloud nat $NAT"
    gcloud compute routers nats create "$NAT" \
      --router="$ROUTER" --region="$REGION" \
      --auto-allocate-nat-external-ips \
      --nat-all-subnet-ip-ranges >/dev/null
  fi

  # ---- firewall: coordinator -> shards on gRPC port ----
  if gcloud compute firewall-rules describe "$FIREWALL_GRPC" >/dev/null 2>&1; then
    log "firewall $FIREWALL_GRPC exists"
  else
    log "creating firewall $FIREWALL_GRPC (coordinator -> shards:$SHARD_PORT)"
    gcloud compute firewall-rules create "$FIREWALL_GRPC" \
      --network="$NETWORK" \
      --allow="tcp:$SHARD_PORT" \
      --source-tags="$COORD_TAG" \
      --target-tags="$SHARD_TAG" >/dev/null
  fi

  # ---- firewall: bench-client -> shards on gRPC port ----
  # The bench-client VM runs aegon_lookup_bench, which brings up an
  # in-process coord state that fans out to the shards directly over
  # the VPC. So bench-client needs the same shard reachability the
  # coord has.
  if gcloud compute firewall-rules describe "$FIREWALL_BENCH_CLIENT_GRPC" >/dev/null 2>&1; then
    log "firewall $FIREWALL_BENCH_CLIENT_GRPC exists"
  else
    log "creating firewall $FIREWALL_BENCH_CLIENT_GRPC (bench-client -> shards:$SHARD_PORT)"
    gcloud compute firewall-rules create "$FIREWALL_BENCH_CLIENT_GRPC" \
      --network="$NETWORK" \
      --allow="tcp:$SHARD_PORT" \
      --source-tags="$BENCH_CLIENT_TAG" \
      --target-tags="$SHARD_TAG" >/dev/null
  fi

  # ---- firewall: shards -> masking server on masking port ----
  if [[ "$ENABLE_MASKING_SERVER" == "1" ]]; then
    local fw_masking="aegon-bench-masking"
    if gcloud compute firewall-rules describe "$fw_masking" >/dev/null 2>&1; then
      log "firewall $fw_masking exists"
    else
      log "creating firewall $fw_masking (shards -> masking:$MASKING_PORT)"
      gcloud compute firewall-rules create "$fw_masking" \
        --network="$NETWORK" \
        --allow="tcp:$MASKING_PORT" \
        --source-tags="$SHARD_TAG" \
        --target-tags="$MASKING_TAG" >/dev/null
    fi
  fi

  # ---- firewall: SSH via IAP tunnel only ----
  local ssh_want="35.235.240.0/20"
  if gcloud compute firewall-rules describe "$FIREWALL_SSH" >/dev/null 2>&1; then
    local ssh_have
    ssh_have="$(gcloud compute firewall-rules describe "$FIREWALL_SSH" \
      --format='value(sourceRanges.list())' 2>/dev/null)"
    if [[ "$ssh_have" != "$ssh_want" ]]; then
      log "firewall $FIREWALL_SSH: updating source range to $ssh_want"
      gcloud compute firewall-rules update "$FIREWALL_SSH" --source-ranges="$ssh_want" >/dev/null
    fi
  else
    log "creating firewall $FIREWALL_SSH (IAP -> instances:22)"
    gcloud compute firewall-rules create "$FIREWALL_SSH" \
      --network="$NETWORK" \
      --allow="tcp:22" \
      --source-ranges="$ssh_want" >/dev/null
  fi

  # ---- shard machines ----
  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    if gcloud compute instances describe "$name" --zone="$ZONE" >/dev/null 2>&1; then
      log "$name exists, skipping"
      continue
    fi
    log "creating $name ($SHARD_MACHINE_TYPE)"
    gcloud compute instances create "$name" \
      --zone="$ZONE" \
      --machine-type="$SHARD_MACHINE_TYPE" \
      --network="$NETWORK" \
      --no-address \
      --tags="$SHARD_TAG,$SCANNER_TAG" \
      --labels="$SCANNER_LABEL" \
      --image-family="ubuntu-2604-lts-amd64" --image-project="ubuntu-os-cloud" \
      --boot-disk-size=100GB >/dev/null
  done

  # ---- coordinator ----
  # Boot disk doubles as the RocksDB volume at $COORD_DB_PATH. Must hold
  # all of the AKD history table for the largest fill we drive — at
  # ~200KB/entry pre-compaction (observed 2.3GB for 12k entries in the
  # first ENOSPC failure), 90% of medium (60M entries) needs ~1-2TB
  # compacted with 2-3x headroom for WAL/L0 spikes during warmup.
  # 4TB pd-ssd gives that headroom and keeps I/O off the critical path.
  local cname; cname="$(coord_name)"
  if gcloud compute instances describe "$cname" --zone="$ZONE" >/dev/null 2>&1; then
    log "$cname exists, skipping"
  else
    log "creating $cname ($COORD_MACHINE_TYPE, boot=${COORD_BOOT_DISK_SIZE} ${COORD_BOOT_DISK_TYPE})"
    gcloud compute instances create "$cname" \
      --zone="$ZONE" \
      --machine-type="$COORD_MACHINE_TYPE" \
      --network="$NETWORK" \
      --no-address \
      --tags="$COORD_TAG,$SCANNER_TAG" \
      --labels="$SCANNER_LABEL" \
      --image-family="ubuntu-2604-lts-amd64" --image-project="ubuntu-os-cloud" \
      --boot-disk-size="$COORD_BOOT_DISK_SIZE" \
      --boot-disk-type="$COORD_BOOT_DISK_TYPE" >/dev/null
  fi

  # No DB VM: the coordinator stores its open-addressing index +
  # per-shard checkpoints in a local RocksDB at $COORD_DB_PATH on
  # its own disk. Cuts a VM, a firewall rule, and Redis's MULTI/EXEC
  # transaction-size ceiling out of the deployment.

  # ---- masking servers (optional, N_MASKING_SERVERS VMs per cluster) ----
  # Spun up in parallel — each is independent (no inter-server coord).
  if [[ "$ENABLE_MASKING_SERVER" == "1" ]]; then
    local -a masking_create_pids=()
    for ((mi = 0; mi < N_MASKING_SERVERS; mi++)); do
      local mname; mname="$(masking_name "$mi")"
      if gcloud compute instances describe "$mname" --zone="$ZONE" >/dev/null 2>&1; then
        log "$mname exists, skipping"
        continue
      fi
      log "creating $mname ($MASKING_MACHINE_TYPE)"
      (
        gcloud compute instances create "$mname" \
          --zone="$ZONE" \
          --machine-type="$MASKING_MACHINE_TYPE" \
          --network="$NETWORK" \
          --no-address \
          --tags="$MASKING_TAG,$SCANNER_TAG" \
          --labels="$SCANNER_LABEL" \
          --image-family="ubuntu-2604-lts-amd64" --image-project="ubuntu-os-cloud" \
          --boot-disk-size=100GB >/dev/null
      ) &
      masking_create_pids+=("$!")
    done
    if (( ${#masking_create_pids[@]} > 0 )); then
      log "waiting on ${#masking_create_pids[@]} masking VM create(s)..."
      for pid in "${masking_create_pids[@]}"; do
        wait "$pid" || log "WARN: a masking VM create failed (check above)"
      done
    fi
  fi

  # ---- bench-client (one VM per cluster, drives lookup-bench) ----
  # Separate VM for aegon_lookup_bench so the bench's request-driving
  # CPU and the coord's tonic worker pool don't share cores. Same
  # machine type + disk as the coord because the bench brings up its
  # own in-process coord state (writes RocksDB at $COORD_DB_PATH and
  # climbs through fill levels via real publish calls).
  local bname; bname="$(bench_client_name)"
  if gcloud compute instances describe "$bname" --zone="$ZONE" >/dev/null 2>&1; then
    log "$bname exists, skipping"
  else
    log "creating $bname ($BENCH_CLIENT_MACHINE_TYPE, boot=${BENCH_CLIENT_BOOT_DISK_SIZE} ${BENCH_CLIENT_BOOT_DISK_TYPE})"
    gcloud compute instances create "$bname" \
      --zone="$ZONE" \
      --machine-type="$BENCH_CLIENT_MACHINE_TYPE" \
      --network="$NETWORK" \
      --no-address \
      --tags="$BENCH_CLIENT_TAG,$SCANNER_TAG" \
      --labels="$SCANNER_LABEL" \
      --image-family="ubuntu-2604-lts-amd64" --image-project="ubuntu-os-cloud" \
      --boot-disk-size="$BENCH_CLIENT_BOOT_DISK_SIZE" \
      --boot-disk-type="$BENCH_CLIENT_BOOT_DISK_TYPE" >/dev/null
  fi

  log "instances up. waiting 30s for SSH to settle..."
  sleep 30
  log "ready. next: ./scripts/bench-cluster.sh deploy"
}

# Kill any running shard server on $name and start a fresh one with the
# given prefill count. Assumes the binary already exists at
# $REMOTE_BIN_DIR/aegon_shard_server (this is what makes `restart-shards`
# cheap relative to `deploy` — no rebuild, no scp).
#
# PID-file restart pattern (see cluster.sh for the rationale on avoiding
# `pkill -f`). We deliberately do NOT use `setsid` here: it forks when
# the caller is a session leader, so $! would point at a short-lived
# intermediate rather than the shard server, and the next restart's
# PID-file kill would no-op (leaving the old server holding port 50051).
# Plain `nohup ... &` keeps $! aligned with the actual shard server. The
# SSH-session hang that setsid was trying to solve is handled by
# `fire-and-forget` instead.
start_shard() {
  local name="$1"
  local i="$2"
  # Optional 3rd arg: pre-resolved masking endpoint(s) as a single
  # comma-separated string. Passed in by cmd_start_shards so we don't
  # fire one `gcloud describe` per shard at fan-out time (that was
  # tripping IAP throttling on 128-shard clusters). Empty string is
  # treated as "no masking". The shard's `--masking-addr` clap arg
  # has `value_delimiter=','` so the comma-joined string splits into
  # a Vec<String> on the shard, and the shard's `MaskingClientPool`
  # round-robins fetches across every endpoint in the list.
  local prefetched_masking_ep="${3:-}"
  log "[$name] starting shard server (shard_id=$i, --prefill-count 0)"
  # Each shard is launched exactly once per benchmark, with
  # --prefill-count 0. Per-fill_percent prefill is driven from the
  # coordinator via the gRPC ReconfigurePrefill RPC — the bench
  # binaries (aegon_publish_bench / aegon_lookup_bench) issue it
  # before every stage. That removes the kill-and-restart cycle
  # the old design needed between fills, and with it the long-
  # lived gcloud ssh sessions whose IAP-tunnel teardown latency
  # caused so many false-positive failures.
  #
  # No --db-url / --db-path: per the shard server's CLI docs
  # ("currently a no-op" at akd/src/bin/aegon_shard_server.rs:84),
  # the shard's DB connection is vestigial and skipped at runtime.
  # All persistent state lives on the coordinator's RocksDB.
  #
  # Two short ssh calls per shard, so each session is well under
  # any plausible IAP teardown lag:
  #   (1) spawn — fire-and-forget kill + nohup
  #   (2) wait_for_shard_ready — short-lived gcloud probes
  # Optional --masking-addr: caller passes the pre-resolved endpoint
  # to avoid one `gcloud describe` per shard at large scale (the
  # parallel fan-out was tripping IAP throttling). Empty string =
  # no masking endpoint (shards fall back to inline package
  # generation).
  local masking_flag=""
  if [[ "$ENABLE_MASKING_SERVER" == "1" && -n "$prefetched_masking_ep" ]]; then
    masking_flag="--masking-addr $prefetched_masking_ep"
  fi
  local spawn_cmd="if [ -f /tmp/aegon-shard.pid ]; then \
      kill \$(cat /tmp/aegon-shard.pid) 2>/dev/null || true; \
    fi; \
    pkill -x aegon_shard_ser 2>/dev/null || true; \
    sleep 2; \
    if [ ! -f $REMOTE_SRS_PATH ]; then \
      echo \"FAILED: SRS file not present at $REMOTE_SRS_PATH\"; exit 1; \
    fi; \
    mkdir -p \$HOME/aegon-run && \
    cd \$HOME/aegon-run && \
    nohup $REMOTE_BIN_DIR/aegon_shard_server \
      --bind 0.0.0.0:$SHARD_PORT \
      --srs-path $REMOTE_SRS_PATH \
      --shard-log-capacity $SHARD_LOG_CAPACITY \
      --kzh-k $KZH_K \
      --shard-id $i \
      --prefill-count 0 \
      --no-retain-epoch-polys \
      --private \
      $masking_flag \
      > /tmp/aegon-shard.log 2>&1 < /dev/null & \
    echo \$! > /tmp/aegon-shard.pid; \
    disown 2>/dev/null || true; \
    echo SPAWNED"
  # `|| spawn_rc=$?` shields the assignment from `set -e` — without
  # it, a non-zero exit from gcloud (timeout, IAP-tunnel failure,
  # etc.) terminates the subshell before we can inspect the output
  # or print a diagnostic, so the parent's `wait` reports a generic
  # failure with no log trail.
  # 300s budget: the remote bash work is ~5s (kill + nohup + echo
  # SPAWNED), but the IAP-tunnel setup + teardown can each take
  # tens of seconds. 120s wasn't enough headroom — slower IAP setup
  # could eat the entire budget before the remote bash even started.
  local spawn_out spawn_rc=0
  spawn_out="$(timeout 300 gcloud compute ssh "$name" --zone="$ZONE" \
    --tunnel-through-iap --quiet \
    --ssh-flag="-o UserKnownHostsFile=/dev/null" \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    --ssh-flag="-o LogLevel=ERROR" \
    --command="$spawn_cmd" 2>&1)" \
    || spawn_rc=$?
  if [[ "$spawn_out" == *"FAILED: SRS file not present"* ]]; then
    printf '%s\n' "$spawn_out"
    return 1
  fi
  if ! grep -qE '^SPAWNED$' <<<"$spawn_out"; then
    log "[$name] spawn ssh did not report SPAWNED (rc=$spawn_rc); output:"
    printf '%s\n' "$spawn_out"
    return "$spawn_rc"
  fi
  log "[$name] spawn OK; polling for :$SHARD_PORT"
  wait_for_shard_ready "$name"
}

# Probe the shard's listener via short-lived gcloud sessions. Returns
# 0 once `ss -tln` on the remote reports :$SHARD_PORT in LISTEN, or 1
# after the per-shard deadline. We keep each gcloud call ≤30s so any
# IAP teardown lag is bounded.
wait_for_shard_ready() {
  local name="$1"
  # At N_SHARDS=128 the start-shards parallel fan-out launches 128
  # of these probe loops simultaneously, each making a fresh gcloud
  # ssh-via-IAP call every poll interval. The IAP tunnel concurrency
  # ceiling means many probes time out even when the shard IS
  # listening (the probe SSH never reaches the remote in time).
  # Bumping the per-probe timeout + slowing the poll cadence + a
  # longer overall deadline gives the loop room to eventually
  # confirm readiness without falsely failing the start-shards phase.
  local deadline=$((SECONDS + ${WAIT_READY_SECONDS:-3600}))
  local probe_timeout="${WAIT_READY_PROBE_TIMEOUT:-90}"
  local probe_interval="${WAIT_READY_PROBE_INTERVAL:-30}"
  local probe_cmd="ss -tln 2>/dev/null | grep -qE ':$SHARD_PORT\\b' && echo READY || echo NOT_READY"
  while (( SECONDS < deadline )); do
    local out=""
    # `|| true` shields the assignment from `set -e` — a probe that
    # fails (timeout, transient IAP-tunnel hiccup) should let us
    # keep polling, not exit the subshell.
    out="$(timeout "$probe_timeout" gcloud compute ssh "$name" --zone="$ZONE" \
      --tunnel-through-iap --quiet \
      --ssh-flag="-o UserKnownHostsFile=/dev/null" \
      --ssh-flag="-o StrictHostKeyChecking=no" \
      --ssh-flag="-o LogLevel=ERROR" \
      --command="$probe_cmd" 2>/dev/null)" \
      || true
    if grep -qE '^READY$' <<<"$out"; then
      log "[$name] READY (listening on :$SHARD_PORT)"
      return 0
    fi
    sleep "$probe_interval"
  done
  log "[$name] still not listening after ${WAIT_READY_SECONDS:-3600}s; tailing remote log:"
  timeout 60 gcloud compute ssh "$name" --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no \
    --quiet \
    --ssh-flag="-o UserKnownHostsFile=/dev/null" \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    --ssh-flag="-o LogLevel=ERROR" \
    --command="tail -n 30 /tmp/aegon-shard.log 2>/dev/null" || true
  return 1
}

cmd_deploy() {
  require_project
  require_power_of_two "$N_SHARDS"

  # Optional: `TRACING=1` builds with the `tracing_instrument` feature
  # so the bins install a tracing-tree subscriber and emit phase spans
  # (`Aegon::PublishPhase1`, `KZH::FMAState`, `ShardedAegon::*`, ...).
  # Stderr only; the bench JSON is unaffected. Turn off for clean runs.
  # Cluster builds always opt into mimalloc — at log_capacity=27 the
  # shard's publish path makes many short-lived large allocations
  # (the prev/new state clones in v1 lived 4×1.2 GB at peak; the
  # refactor cut that, but the remaining churn still hands glibc more
  # than it returns to the OS). mimalloc trims aggressively.
  local cargo_features="--features mimalloc_alloc"
  if [[ "${TRACING:-0}" == "1" ]]; then
    cargo_features="$cargo_features --features tracing_instrument"
    log "TRACING=1: building with tracing_instrument feature (tracing-tree subscriber)"
  fi
  local local_bin_dir="$REPO_ROOT/target/release"
  local remote_bin_dir="$local_bin_dir"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    command -v docker >/dev/null || die "macOS host needs Docker (we build the Linux binaries inside a container)"
    docker info >/dev/null 2>&1  || die "Docker daemon unreachable — start Docker Desktop and retry"

    log "macOS host: building linux/amd64 binaries inside Docker"
    docker run --rm --platform linux/amd64 \
      -v "$REPO_ROOT:/workspace" -w /workspace \
      rust:slim-bookworm \
      bash -c "set -e; \
        apt-get update >/dev/null && \
        apt-get install -y --no-install-recommends protobuf-compiler ca-certificates >/dev/null && \
        cargo build --release -p akd $cargo_features --target x86_64-unknown-linux-gnu \
          --bin aegon_shard_server --bin aegon_coordinator_bench --bin aegon_srs_gen \
          --bin aegon_masking_server"
    remote_bin_dir="$REPO_ROOT/target/x86_64-unknown-linux-gnu/release"
  else
    log "building release binaries (aegon_shard_server, aegon_coordinator_bench, aegon_srs_gen, aegon_masking_server)"
    (cd "$REPO_ROOT" && cargo build --release -p akd $cargo_features \
      --bin aegon_shard_server --bin aegon_coordinator_bench --bin aegon_srs_gen \
      --bin aegon_masking_server) >/dev/null
  fi
  [[ -x "$remote_bin_dir/aegon_shard_server" ]]      || die "aegon_shard_server missing"
  [[ -x "$remote_bin_dir/aegon_coordinator_bench" ]] || die "aegon_coordinator_bench missing"
  [[ -x "$remote_bin_dir/aegon_srs_gen" ]]           || die "aegon_srs_gen missing"
  [[ -x "$remote_bin_dir/aegon_masking_server" ]]    || die "aegon_masking_server missing"

  local per_shard; per_shard="$(prefill_per_shard)"
  log "shards will prefill ${per_shard} entries each (total = 2^$TOTAL_PRELOAD_LOG2)"

  # ---- coordinator-side RocksDB directory ----
  # Make sure the coordinator has a writable $COORD_DB_PATH for the
  # bench binaries to point their --db-path at. The binaries
  # themselves create the directory on first open, but we mkdir
  # upfront so an early "permission denied" surfaces during deploy,
  # not 30 minutes into the publish-bench warmup.
  #
  # wait_for_ssh first — coord may still be booting (larger boot
  # disks like the 4TB pd-ssd take longer to ready than the default
  # 20GB pd-balanced).
  local cname; cname="$(coord_name)"
  wait_for_ssh "$cname"
  log "[$cname] preparing RocksDB directory $COORD_DB_PATH"
  remote "$cname" "sudo mkdir -p $COORD_DB_PATH && sudo chown \$(whoami) $COORD_DB_PATH"

  # ---- push to every shard + start, all in parallel ----
  # Each shard generates its own SRS in-process from --setup-seed
  # (no file transfer of a 30 GB SRS), then prefills with
  # --prefill-count entries (deterministic, seeded by PREFILL_SEED + i
  # so different shards fill different slot patterns).
  #
  # Each shard's upload+start runs in a background subshell because
  # gcloud's IAP-tunnel ssh has slow channel teardown (tens of seconds
  # per call), and we have N of them — serial would be O(N*teardown).
  local -a deploy_pids=()
  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    (
      wait_for_ssh "$name"
      log "[$name] uploading aegon_shard_server + aegon_srs_gen"
      scp_to "$name" "$remote_bin_dir/aegon_shard_server"
      scp_to "$name" "$remote_bin_dir/aegon_srs_gen"
      remote "$name" "sudo mkdir -p $REMOTE_BIN_DIR && \
        sudo mv /tmp/aegon_shard_server /tmp/aegon_srs_gen $REMOTE_BIN_DIR/ && \
        sudo chmod +x $REMOTE_BIN_DIR/aegon_shard_server $REMOTE_BIN_DIR/aegon_srs_gen && \
        sudo mkdir -p $REMOTE_SRS_DIR && \
        sudo chown \$(whoami) $REMOTE_SRS_DIR"
      # Per-shard local-SRS mode: don't start the shard server here.
      # setup-bench will generate the SRS on every shard in parallel
      # (same seed → identical files). publish-bench / lookup-bench
      # call restart_shard per stage anyway, so deferring the start
      # is correct.
      log "[$name] deploy done (binaries in place; SRS not yet generated)"
    ) &
    deploy_pids+=("$!")
  done
  log "waiting on ${#deploy_pids[@]} per-shard deploys (parallel)..."
  local failed=0
  for pid in "${deploy_pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
  done
  if (( failed > 0 )); then
    die "$failed shard deploy(s) failed — inspect output above and run 'logs <i>' to debug"
  fi

  # ---- push to coordinator ----
  local cname; cname="$(coord_name)"
  wait_for_ssh "$cname"
  log "[$cname] uploading aegon_coordinator_bench"
  scp_to "$cname" "$remote_bin_dir/aegon_coordinator_bench"
  remote "$cname" "sudo mkdir -p $REMOTE_BIN_DIR && \
    sudo mv /tmp/aegon_coordinator_bench $REMOTE_BIN_DIR/ && \
    sudo chmod +x $REMOTE_BIN_DIR/aegon_coordinator_bench"

  # Enable user-session lingering on the coord. Without this, the
  # transient systemd-run unit that wraps aegon_lookup_bench gets
  # placed in the user's slice (because we pass --uid=$me --gid=$me),
  # and systemd-logind tears down the whole user@.service after the
  # last SSH session that touched it is gone "for long enough" —
  # observed in v2 as SIGTERM (ExecMainStatus=15) at the ~15h mark,
  # mid-climb. Linger tells logind to keep the user manager running
  # indefinitely, so transient user-slice units survive arbitrarily
  # long disconnects.
  log "[$cname] enabling logind linger so the bench unit survives 15h+"
  remote "$cname" "sudo loginctl enable-linger \$(whoami) && loginctl show-user \$(whoami) --property=Linger --value"

  # ---- push to masking servers (optional, parallel) ----
  if [[ "$ENABLE_MASKING_SERVER" == "1" ]]; then
    local -a masking_deploy_pids=()
    for ((mi = 0; mi < N_MASKING_SERVERS; mi++)); do
      local mname; mname="$(masking_name "$mi")"
      (
        wait_for_ssh "$mname"
        log "[$mname] uploading aegon_masking_server + aegon_srs_gen"
        scp_to "$mname" "$remote_bin_dir/aegon_masking_server"
        scp_to "$mname" "$remote_bin_dir/aegon_srs_gen"
        remote "$mname" "sudo mkdir -p $REMOTE_BIN_DIR && \
          sudo mv /tmp/aegon_masking_server /tmp/aegon_srs_gen $REMOTE_BIN_DIR/ && \
          sudo chmod +x $REMOTE_BIN_DIR/aegon_masking_server $REMOTE_BIN_DIR/aegon_srs_gen && \
          sudo mkdir -p $REMOTE_SRS_DIR && \
          sudo chown \$(whoami) $REMOTE_SRS_DIR"
        log "[$mname] deploy done (binaries in place; SRS will be generated by start-masking)"
      ) &
      masking_deploy_pids+=("$!")
    done
    log "waiting on ${#masking_deploy_pids[@]} masking VM deploy(s)..."
    local m_failed=0
    for pid in "${masking_deploy_pids[@]}"; do
      wait "$pid" || m_failed=$((m_failed + 1))
    done
    if (( m_failed > 0 )); then
      die "$m_failed masking VM deploy(s) failed"
    fi
  fi

  # ---- prep the bench-client VM ----
  # aegon_lookup_bench is pushed here (the coord no longer runs it).
  # Same RocksDB directory layout as the coord — the bench's in-process
  # coord state writes its index + state checkpoints to $COORD_DB_PATH
  # on the bench-client's local disk.
  local bname; bname="$(bench_client_name)"
  wait_for_ssh "$bname"
  log "[$bname] preparing RocksDB directory $COORD_DB_PATH"
  remote "$bname" "sudo mkdir -p $COORD_DB_PATH && sudo chown \$(whoami) $COORD_DB_PATH"
  log "[$bname] enabling logind linger so the bench unit survives 15h+"
  remote "$bname" "sudo loginctl enable-linger \$(whoami) && loginctl show-user \$(whoami) --property=Linger --value"

  log "deploy done. SRS generation happens in setup-bench."
  log "next: ./scripts/bench-cluster.sh setup-bench"
}

# Restart every shard server with a fresh --prefill-count, reusing the
# already-uploaded binary and the on-disk SRS cache at
# $HOME/artifacts/srs/. Also FLUSHALLs Redis so the previous prefill's
# occupancy keys and shard checkpoints don't leak into the new run.
#
# Use this inside a (prefill, batch) sweep instead of `deploy`: it
# skips the cargo build, the scp uploads, and the redis re-install,
# collapsing per-prefill cycle from ~9 min to ~30 s. The cluster must
# already be `up` and `deploy`-ed once.
cmd_start_shards() {
  require_project
  require_power_of_two "$N_SHARDS"

  # Wipe the coordinator's RocksDB so each run starts from a clean
  # keyspace. Old open-addressing entries would otherwise be
  # observable to plan_phase_1 and the bench would see false
  # "slot occupied" hits.
  local cname; cname="$(coord_name)"
  log "[$cname] wiping $COORD_DB_PATH (clean RocksDB for the run)"
  remote "$cname" "sudo rm -rf $COORD_DB_PATH && sudo mkdir -p $COORD_DB_PATH && sudo chown \$(whoami) $COORD_DB_PATH"

  # Pre-resolve every masking server's endpoint ONCE before the per-
  # shard fan-out. Without this each of N_SHARDS background subshells
  # does its own `gcloud describe`, and at large scale (128 shards)
  # the resulting gcloud-API + IAP-tunnel pressure causes spawn_ssh
  # calls to time out with rc=255. Resolving here serializes a small
  # number of cheap calls (one per masking server, not one per shard).
  local -a MASKING_ENDPOINTS=()
  if [[ "$ENABLE_MASKING_SERVER" == "1" ]]; then
    log "pre-resolving $N_MASKING_SERVERS masking server endpoint(s)..."
    for ((mi = 0; mi < N_MASKING_SERVERS; mi++)); do
      local ep
      ep="$(masking_endpoint "$mi")" || die "could not resolve masking server $mi endpoint"
      MASKING_ENDPOINTS+=("$ep")
    done
    log "masking endpoints: ${MASKING_ENDPOINTS[*]}"
  fi

  # Comma-joined endpoint list for the shard's --masking-addr. The
  # shard server now accepts a list and dispatches package fetches
  # round-robin via `MaskingClientPool`, so EVERY shard gets EVERY
  # endpoint — not the old shard_id-mod-N per-shard pinning. This
  # matters at small/medium scale where N_SHARDS ≤ 2 and the per-shard
  # pinning left N-1 of the masking VMs idle; now every shard can drive
  # all N masking servers, so the masking ceiling scales with N
  # regardless of shard count.
  local masking_eps_joined=""
  if (( ${#MASKING_ENDPOINTS[@]} > 0 )); then
    masking_eps_joined="$(IFS=,; echo "${MASKING_ENDPOINTS[*]}")"
  fi

  local -a pids=()
  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    (
      start_shard "$name" "$i" "$masking_eps_joined"
      log "[$name] start done"
    ) &
    pids+=("$!")
  done
  log "waiting on ${#pids[@]} per-shard starts (parallel)..."
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
  done
  if (( failed > 0 )); then
    die "$failed shard start(s) failed — inspect output above and run 'logs <i>' to debug"
  fi
  log "all shards listening on :$SHARD_PORT (--prefill-count 0). Use the bench binaries to drive ReconfigurePrefill."
}

# Start the masking server on its own VM. If no SRS exists at
# $REMOTE_SRS_PATH on the masking VM, generates it first via
# aegon_srs_gen (same seed as the shards — deterministic identical
# file). Then launches aegon_masking_server with a stash queue and
# N background producers that refill the queue continuously.
#
# Idempotent — kills any existing aegon_masking_server first.
# Skips entirely when ENABLE_MASKING_SERVER=0.
start_one_masking() {
  local mname="$1"
  local log_n_shards="$2"
  wait_for_ssh "$mname"

  # SRS: generate on-demand if not already present. Same seed +
  # log_capacity + kzh_k as the shards → byte-identical SRS file →
  # consumers see the same hiding scalar `h`.
  log "[$mname] ensuring SRS at $REMOTE_SRS_PATH (generating if missing)"
  remote "$mname" "
    mkdir -p \$HOME/aegon-run $REMOTE_SRS_DIR
    if [ -f $REMOTE_SRS_PATH ]; then
      echo SRS already present
    else
      cd \$HOME/aegon-run && \
        $REMOTE_BIN_DIR/aegon_srs_gen \
          --shard-log-capacity $SHARD_LOG_CAPACITY \
          --kzh-k $KZH_K \
          --seed $SETUP_SEED \
          --log-n-shards $log_n_shards \
          --private \
          --out $REMOTE_SRS_PATH
    fi
  "

  log "[$mname] launching aegon_masking_server :$MASKING_PORT (queue=$MASKING_QUEUE_SIZE producers=$MASKING_PRODUCERS)"
  remote "$mname" "
    if [ -f /tmp/aegon-masking.pid ]; then
      kill \$(cat /tmp/aegon-masking.pid) 2>/dev/null || true
      sleep 1
    fi
    pkill -x aegon_masking_se 2>/dev/null || true
    sleep 1
    cd \$HOME/aegon-run && \
    setsid nohup $REMOTE_BIN_DIR/aegon_masking_server \
      --bind 0.0.0.0:$MASKING_PORT \
      --num-vars $SHARD_LOG_CAPACITY \
      --kzh-k $KZH_K \
      --srs-path $REMOTE_SRS_PATH \
      --queue-size $MASKING_QUEUE_SIZE \
      --producers $MASKING_PRODUCERS \
      > /tmp/aegon-masking.log 2>&1 < /dev/null &
    echo \$! > /tmp/aegon-masking.pid
    disown 2>/dev/null || true
    echo SPAWNED
  " fire-and-forget

  log "[$mname] waiting for masking server to bind :$MASKING_PORT"
  local ready=0
  for _ in $(seq 1 60); do
    if remote "$mname" "ss -tln | grep -q ':$MASKING_PORT '" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if (( ready == 0 )); then
    log "[$mname] masking server did not bind within 120s — check /tmp/aegon-masking.log"
    return 1
  fi
  log "[$mname] masking server ready on :$MASKING_PORT"
}

cmd_start_masking() {
  if [[ "$ENABLE_MASKING_SERVER" != "1" ]]; then
    log "ENABLE_MASKING_SERVER=0, skipping start-masking"
    return 0
  fi
  require_project
  require_power_of_two "$N_SHARDS"
  local log_n_shards
  log_n_shards="$(python3 -c "import math; print(int(math.log2($N_SHARDS)))")"

  # Launch every masking server in parallel — each is independent
  # (different VM, different gRPC endpoint, no coordination needed).
  log "starting $N_MASKING_SERVERS masking server(s) in parallel"
  local -a start_pids=()
  for ((mi = 0; mi < N_MASKING_SERVERS; mi++)); do
    local mname; mname="$(masking_name "$mi")"
    ( start_one_masking "$mname" "$log_n_shards" ) &
    start_pids+=("$!")
  done
  local failed=0
  for pid in "${start_pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
  done
  if (( failed > 0 )); then
    die "$failed masking server start(s) failed"
  fi
  log "all $N_MASKING_SERVERS masking server(s) listening on :$MASKING_PORT"
}

# Resolve the i-th masking server's internal IP and return its gRPC
# endpoint. With one argument, returns that index's endpoint; with no
# argument, returns the 0th (kept for compat with single-masking
# callers).
masking_endpoint() {
  if [[ "$ENABLE_MASKING_SERVER" != "1" ]]; then
    echo ""
    return 0
  fi
  local mi="${1:-0}"
  local mname; mname="$(masking_name "$mi")"
  local ip
  ip="$(gcloud compute instances describe "$mname" --zone="$ZONE" \
    --format='value(networkInterfaces[0].networkIP)' 2>/dev/null)"
  if [[ -z "$ip" ]]; then
    return 1
  fi
  echo "http://$ip:$MASKING_PORT"
}

# Run aegon_srs_bootstrap on the coordinator to drive the distributed
# SRS gen across all shards. Sends one BootstrapSrs RPC per shard to
# its SrsService port (sampling trapdoors from --setup-seed), then
# polls WaitForReady on every shard until they all transition through
# distributed compute + slab exchange + Aegon init + prefill.
#
# Behaviour:
#   * On cache hit (every shard has a `.cache` file under
#     $SRS_CACHE_DIR matching log_cap + k + seed-hash): runs in
#     seconds — the BootstrapSrs RPC returns cache_hit=true and
#     shards report Ready almost immediately.
#   * On cache miss: drives the full distributed-gen path. Each shard
#     computes its slab of every H_t in parallel, exchanges slabs
#     peer-to-peer over $SRS_PORT, assembles the SRS, writes the
#     cache, runs Aegon init + prefill, reports Ready.
#
# Run after `deploy` and before `bench` / `lookup-bench`.
cmd_bootstrap() {
  # In per-shard-local-SRS mode, "bootstrap" is just: run setup-bench
  # (each shard generates its own SRS in parallel), then start all
  # shards. The legacy distributed-gen path has been removed entirely.
  require_project
  require_power_of_two "$N_SHARDS"
  cmd_setup_bench
  log "starting all shards"
  cmd_start_shards
}


# Setup-time + comm-bytes benchmark for the large regime.
#
# Differs from `bootstrap` in three ways:
#   1. Forces shards to skip prefill (`PREFILL_COUNT=0`) so the
#      measured wall-clock isn't inflated by entry-population work
#      — pure "SRS gen + Aegon init". Requires the shards to have
#      been started with `prefill_count=0` (cmd_restart_shards before
#      calling this is the path).
#   2. Wipes any SRS cache files on every shard so the measurement
#      reflects the full distributed-gen path, not a cached re-run.
#   3. Asks the bootstrap actor to write its consolidated metrics
#      JSON (per-shard phase timestamps + inbound/outbound bytes + SRS
#      sizes) to /tmp/aegon-setup-bench.json on the coordinator, then
#      pulls it back to $LOCAL_SETUP_BENCH_OUT.
#
# Use this after `up + deploy` (shards bound on SRS_PORT, awaiting
# bootstrap). After it completes, the cluster is "Ready" but
# polynomials are empty — `restart-shards` with a non-zero
# TOTAL_PRELOAD_LOG2 puts it back into a benchable state, OR run
# `down` to tear it down once the large-regime numbers are in hand.
REMOTE_SETUP_BENCH_OUT="/tmp/aegon-setup-bench.json"
LOCAL_SETUP_BENCH_OUT="${LOCAL_SETUP_BENCH_OUT:-/tmp/aegon-setup-bench.json}"

# Per-shard local-SRS setup benchmark. Every shard generates its
# own SRS in parallel using the same SETUP_SEED, so they all
# converge on the same file (deterministic). No central gen, no
# broadcast, no NFS — each shard's file lives on local disk at
# $REMOTE_SRS_PATH and is read directly by aegon_shard_server at
# every subsequent restart.
#
# Reports:
#   * gen_seconds_per_shard — per-shard wall-clock for aegon_srs_gen
#   * gen_seconds_max       — slowest shard (effective cluster cost,
#                             since the benchmark blocks on the
#                             slowest one)
#   * gen_seconds_min/median/mean
#   * srs_bytes             — size of the SRS file on disk
#                             (identical across shards by construction)
#
# Prereq: cluster is `up + deploy`-ed (aegon_srs_gen + aegon_shard_server
# binaries already pushed to every shard).
cmd_setup_bench() {
  require_project
  require_power_of_two "$N_SHARDS"

  local log_n_shards
  log_n_shards="$(python3 -c "import math; print(int(math.log2($N_SHARDS)))")"

  # Per-shard parallel generation. Each shard runs aegon_srs_gen with
  # the same seed and writes to its local $REMOTE_SRS_PATH. We capture
  # each shard's gen wall-clock independently — written to a per-shard
  # tmpfile on the local box, then sucked into a JSON array.
  log "generating SRS on $N_SHARDS shards in parallel (seed=$SETUP_SEED, shard_log_capacity=$SHARD_LOG_CAPACITY, kzh_k=$KZH_K, log_n_shards=$log_n_shards)"
  local tmp_dir; tmp_dir="$(mktemp -d)"
  local i
  local -a gen_pids=()
  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    (
      local outfile="$tmp_dir/$i.sec"
      # `time -p` (POSIX) emits "real X.XX" on stderr, robust across
      # bash/dash. We grep "^real " out of stderr and round.
      local t0; t0="$(date +%s.%N)"
      timeout 3600 gcloud compute ssh "$name" --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no \
        --quiet \
        --ssh-flag="-o UserKnownHostsFile=/dev/null" \
        --ssh-flag="-o StrictHostKeyChecking=no" \
        --ssh-flag="-o LogLevel=ERROR" \
        --command "mkdir -p \$HOME/aegon-run \$HOME/artifacts/srs && cd \$HOME/aegon-run && \
          rm -f $REMOTE_SRS_PATH && \
          $REMOTE_BIN_DIR/aegon_srs_gen \
            --shard-log-capacity $SHARD_LOG_CAPACITY \
            --kzh-k $KZH_K \
            --seed $SETUP_SEED \
            --log-n-shards $log_n_shards \
            --private \
            --out $REMOTE_SRS_PATH" \
        >/dev/null 2>&1
      local rc=$?
      local t1; t1="$(date +%s.%N)"
      if (( rc == 0 )); then
        python3 -c "print(round($t1 - $t0, 3))" > "$outfile"
        echo "[$name] SRS gen OK in $(cat "$outfile")s"
      else
        echo "FAIL" > "$outfile"
        echo "[$name] SRS gen FAILED rc=$rc" >&2
      fi
    ) &
    gen_pids+=("$!")
  done
  log "waiting on ${#gen_pids[@]} parallel SRS generations..."
  local failed=0
  for pid in "${gen_pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
  done
  if (( failed > 0 )); then
    die "$failed shard(s) failed SRS generation; see logs above"
  fi

  # Gather per-shard timings.
  local -a per_shard_secs=()
  for ((i = 0; i < N_SHARDS; i++)); do
    local sec; sec="$(cat "$tmp_dir/$i.sec")"
    [[ "$sec" == "FAIL" ]] && die "shard $i SRS gen FAILED"
    per_shard_secs+=("$sec")
  done
  rm -rf "$tmp_dir"

  # Compute min/max/median/mean from per_shard_secs in python.
  local secs_csv; secs_csv="$(IFS=,; echo "${per_shard_secs[*]}")"
  local stats_json
  stats_json="$(python3 - "$secs_csv" <<'PY'
import sys, json, statistics
secs = [float(s) for s in sys.argv[1].split(',')]
print(json.dumps({
  "min":    round(min(secs), 3),
  "max":    round(max(secs), 3),
  "median": round(statistics.median(secs), 3),
  "mean":   round(statistics.mean(secs), 3),
  "all":    [round(s, 3) for s in secs],
}))
PY
)"
  log "SRS gen stats: $stats_json"

  # Capture SRS size from shard-0 (identical across shards by construction).
  local gen_host; gen_host="$(shard_name 0)"
  local srs_bytes; srs_bytes="$(remote "$gen_host" "stat -c '%s' $REMOTE_SRS_PATH" | tr -d '[:space:]')"
  log "[$gen_host] SRS size: $srs_bytes bytes"

  # Emit JSON. broadcast_seconds is kept (at 0) for plot compatibility
  # — the per-shard scheme has no broadcast phase.
  mkdir -p "$(dirname "$LOCAL_SETUP_BENCH_OUT")"
  python3 - <<PY > "$LOCAL_SETUP_BENCH_OUT"
import json
stats = json.loads('''$stats_json''')
payload = {
  "regime": "per-shard-local",
  "n_shards": $N_SHARDS,
  "shard_log_capacity": $SHARD_LOG_CAPACITY,
  "kzh_k": $KZH_K,
  "log_n_shards": $log_n_shards,
  "setup_seed": $SETUP_SEED,
  "gen_host_machine_type": "$SHARD_MACHINE_TYPE",
  "gen_host_note": "Per-shard parallel SRS generation; every shard runs aegon_srs_gen with the same seed, writing to local disk at $REMOTE_SRS_PATH. No NFS, no central gen, no broadcast.",
  "gen_seconds": stats["max"],
  "gen_seconds_max": stats["max"],
  "gen_seconds_min": stats["min"],
  "gen_seconds_median": stats["median"],
  "gen_seconds_mean": stats["mean"],
  "gen_seconds_per_shard": stats["all"],
  "broadcast_seconds": 0,
  "srs_bytes": $srs_bytes,
}
print(json.dumps(payload, indent=2))
PY
  log "setup-bench JSON written to $LOCAL_SETUP_BENCH_OUT"
}

# Publish-time + commit-size benchmark for the large regime.
#
# Walks PUBLISH_FILL_PERCENTS (default 1,30,60,90), restarting shards
# between stages with the per-shard prefill_count needed to hit that
# fill level vs. the cluster's TRUE log capacity (default 32, i.e.,
# 2^32 = ~4.3B entries total across 32 shards). For each stage, runs
# aegon_publish_bench in distributed mode against the live cluster
# and ferries one JSON record per fill level back to the local box.
#
# What this measures, per (fill_pct, batch_size):
#   - publish wall-clock (samples + fastest/slowest/median/mean ms)
#   - uncompressed bytes of the four published commits
#     (index, value, rand_index, rand_value) summed across all shards
#   - total ShardedEpochCommitment bytes (commits + Merkle root + epoch)
#
# Prereqs: cluster must be `up + deploy`-ed once, and `bootstrap` must
# have completed at least once (so the SRS cache file is on each
# shard's disk — subsequent bootstraps hit cache in seconds rather
# than re-running distributed gen). The script handles that
# automatically — it runs `bootstrap` between every restart-shards
# step, which is a no-op on cache hit.
PUBLISH_FILL_PERCENTS="${PUBLISH_FILL_PERCENTS:-1,30,60,90}"
PUBLISH_BATCH_SIZES="${PUBLISH_BATCH_SIZES:-4096,8192,16384,32768,65536,131072}"
PUBLISH_SAMPLES_PER_BATCH="${PUBLISH_SAMPLES_PER_BATCH:-3}"
# True (non-over-provisioned) total log capacity, derived from the
# cluster geometry:
#   total_log_slots = SHARD_LOG_CAPACITY + log2(N_SHARDS)
#   true_log_capacity = total_log_slots - LOG2_OVER_PROVISIONING_FACTOR(=2)
#
# Auto-deriving avoids a class of latent bugs we hit before, where a
# static default (32) carried over from the LARGE regime and made
# medium-cluster runs target 30% of 2^32 = 1.3B entries instead of
# 30% of 2^26 = 20M. Override only if you intentionally want to
# benchmark against a different addressable space than the cluster's
# physical one.
derive_true_log_cap() {
  python3 -c "
import math
n_shards = ${N_SHARDS}
shard_log_cap = ${SHARD_LOG_CAPACITY}
log2_alpha = 2  # mirrors LOG2_OVER_PROVISIONING_FACTOR in akd/src/aegon/config.rs
log2_n_shards = int(math.log2(n_shards))
print(shard_log_cap + log2_n_shards - log2_alpha)
"
}
PUBLISH_TRUE_LOG_CAP="${PUBLISH_TRUE_LOG_CAP:-$(derive_true_log_cap)}"
LOCAL_PUBLISH_BENCH_DIR="${LOCAL_PUBLISH_BENCH_DIR:-/tmp/aegon-publish-bench}"

# Print the resolved cluster + bench config. Called at the top of
# every bench command so a misconfiguration is visible in the log
# before any expensive work starts. Failures we hit in prior runs
# were almost always config-shape mismatches (wrong TRUE_LOG_CAP for
# the regime, etc.); making the resolved values appear up front
# turns "warmup ran for 12 hours then failed" into "we caught it in
# the first 2 seconds."
dump_bench_config() {
  local derived; derived="$(derive_true_log_cap)"
  log "==== resolved config ===="
  log "  N_SHARDS               = $N_SHARDS"
  log "  SHARD_LOG_CAPACITY     = $SHARD_LOG_CAPACITY (per shard 2^$SHARD_LOG_CAPACITY slots)"
  log "  KZH_K                  = $KZH_K"
  log "  PUBLISH_TRUE_LOG_CAP   = $PUBLISH_TRUE_LOG_CAP (derived = $derived; addressable = 2^$PUBLISH_TRUE_LOG_CAP entries)"
  log "  PUBLISH_FILL_PERCENTS  = $PUBLISH_FILL_PERCENTS"
  log "  PUBLISH_BATCH_SIZES    = $PUBLISH_BATCH_SIZES"
  log "  PUBLISH_WARMUP_BATCH   = ${PUBLISH_WARMUP_BATCH_SIZE:-16384}"
  log "  COORD_DB_PATH          = $COORD_DB_PATH (RocksDB on coordinator)"
  log "  COORD_BOOT_DISK        = $COORD_BOOT_DISK_SIZE $COORD_BOOT_DISK_TYPE"
  log "  SHARDS                 = $SHARD_MACHINE_TYPE (--no-retain-epoch-polys set)"
  if [[ "$PUBLISH_TRUE_LOG_CAP" != "$derived" ]]; then
    log "  NOTE: PUBLISH_TRUE_LOG_CAP overridden — derived value would be $derived"
  fi
  log "========================"
}

cmd_publish_bench() {
  require_project
  require_power_of_two "$N_SHARDS"
  mkdir -p "$LOCAL_PUBLISH_BENCH_DIR"
  dump_bench_config

  # Build + push the bench binary to the coordinator (idempotent —
  # cached cargo + scp-only-if-different is what `bench-cluster.sh
  # deploy` already does, but this subcommand may run on its own).
  local cargo_features=""
  if [[ "${TRACING:-0}" == "1" ]]; then
    cargo_features="--features tracing_instrument"
  fi
  local local_bin_dir="$REPO_ROOT/target/release"
  local remote_bin_dir="$local_bin_dir"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    command -v docker >/dev/null || die "macOS host needs Docker"
    docker info >/dev/null 2>&1  || die "Docker daemon unreachable"
    log "macOS host: building aegon_publish_bench inside Docker"
    docker run --rm --platform linux/amd64 \
      -v "$REPO_ROOT:/workspace" -w /workspace \
      rust:slim-bookworm \
      bash -c "set -e; \
        apt-get update >/dev/null && \
        apt-get install -y --no-install-recommends protobuf-compiler ca-certificates >/dev/null && \
        cargo build --release -p akd $cargo_features --target x86_64-unknown-linux-gnu \
          --bin aegon_publish_bench"
    remote_bin_dir="$REPO_ROOT/target/x86_64-unknown-linux-gnu/release"
  else
    log "building aegon_publish_bench (release)"
    (cd "$REPO_ROOT" && cargo build --release -p akd $cargo_features --bin aegon_publish_bench) >/dev/null
  fi
  [[ -x "$remote_bin_dir/aegon_publish_bench" ]] || die "aegon_publish_bench missing"

  local cname; cname="$(coord_name)"
  log "[$cname] uploading aegon_publish_bench"
  scp_to "$cname" "$remote_bin_dir/aegon_publish_bench"
  remote "$cname" "sudo mkdir -p $REMOTE_BIN_DIR && \
    sudo mv /tmp/aegon_publish_bench $REMOTE_BIN_DIR/ && \
    sudo chmod +x $REMOTE_BIN_DIR/aegon_publish_bench"

  local shard_csv; shard_csv="$(shard_endpoints_csv)"

  # One process now walks every fill percent — the bench binary
  # issues a ReconfigurePrefill RPC to each shard before each stage,
  # so the cluster never needs to kill and restart shards between
  # fills. Mirrors the local-mode publish-bench output: a single
  # JSON file with one record per fill_percent under "stages".
  # Prereq: shards already started (`bench-cluster.sh start-shards`).

  # Wipe the coordinator's RocksDB before publish-bench so we start
  # from a clean open-addressing keyspace. (cmd_start_shards also
  # wipes; this is a defensive second wipe in case publish-bench is
  # invoked directly without start-shards before it.)
  log "[$cname] wiping $COORD_DB_PATH"
  remote "$cname" "sudo rm -rf $COORD_DB_PATH && sudo mkdir -p $COORD_DB_PATH && sudo chown \$(whoami) $COORD_DB_PATH"

  local remote_out="/tmp/aegon-publish-bench.json"
  local remote_log="/tmp/aegon-publish-bench.log"
  local local_out="$LOCAL_PUBLISH_BENCH_DIR/${LOCAL_OUT_NAME_PREFIX:-}publish.json"
  local local_log="$LOCAL_PUBLISH_BENCH_DIR/${LOCAL_OUT_NAME_PREFIX:-}publish-bench.log"
  mkdir -p "$LOCAL_PUBLISH_BENCH_DIR"
  : > "$local_log"
  # PUBLISH_WARMUP_BATCH_SIZE controls how big the inter-stage
  # warmup publishes are. Pick a value at or above the high end of
  # PUBLISH_BATCH_SIZES so warmup time scales sensibly with
  # cluster size (e.g. 2048 for medium, 131072 for large).
  local warmup_batch="${PUBLISH_WARMUP_BATCH_SIZE:-16384}"

  # Launch the bench inside a systemd transient unit so it SURVIVES
  # SSH disconnects. The previous `remote ... stream` design held a
  # single IAP-tunneled SSH session open for the entire bench, which
  # at large scale (hours of warmup work) exceeded IAP's session
  # lifetime and exited with rc=255 mid-run. systemd-run reparents
  # the bench to systemd so SSH death can't reach it; we drive
  # log streaming + completion via short, recoverable SSH probes.
  # Mirrors the cmd_lookup_bench pattern (see below).
  local unit="aegon-publish-bench"
  local me; me="$(whoami)"
  log "[$cname] starting aegon_publish_bench as systemd unit '$unit' (distributed, fills=$PUBLISH_FILL_PERCENTS, warmup_batch=$warmup_batch)"
  remote "$cname" "
    sudo systemctl reset-failed $unit 2>/dev/null || true
    sudo systemctl stop $unit 2>/dev/null || true
    rm -f $remote_log $remote_out
    mkdir -p /home/$me/aegon-run
    sudo systemd-run \
      --unit=$unit \
      --description='Aegon publish-bench' \
      --uid=$me --gid=$me \
      --working-directory=/home/$me/aegon-run \
      --setenv=HOME=/home/$me \
      --setenv=AEGON_ROCKSDB_STATS_DUMP_SEC=${AEGON_ROCKSDB_STATS_DUMP_SEC:-60} \
      --setenv=AEGON_ROCKSDB_BLOCK_CACHE_GB=${AEGON_ROCKSDB_BLOCK_CACHE_GB:-16} \
      --setenv=AEGON_ROCKSDB_PARALLELISM=${AEGON_ROCKSDB_PARALLELISM:-16} \
      --property=LimitNOFILE=1048576 \
      bash -c '
        cd /home/$me/aegon-run
        $REMOTE_BIN_DIR/aegon_publish_bench \
          --shard-log-capacity $SHARD_LOG_CAPACITY \
          --true-log-capacity $PUBLISH_TRUE_LOG_CAP \
          --kzh-k $KZH_K \
          --n-shards $N_SHARDS \
          --endpoints $shard_csv \
          --fill-percents $PUBLISH_FILL_PERCENTS \
          --batch-sizes $PUBLISH_BATCH_SIZES \
          --warmup-batch-size $warmup_batch \
          --samples-per-batch $PUBLISH_SAMPLES_PER_BATCH \
          --setup-seed $SETUP_SEED \
          --prefill-seed $PREFILL_SEED \
          --db-path $COORD_DB_PATH \
          --private \
          --out $remote_out > $remote_log 2>&1
      '
    echo LAUNCHED
  "

  # Poll-loop on the LOCAL side. Each iteration is a short, recoverable
  # ssh command. When the systemd unit deactivates, we exit the loop.
  log "[$cname] polling $unit + streaming $remote_log -> $local_log (resilient to SSH death)"
  local printed_bytes=0
  local poll_rc=0
  while true; do
    local total_bytes
    total_bytes="$(timeout 60 gcloud compute ssh "$cname" \
      --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
      --ssh-flag="-o UserKnownHostsFile=/dev/null" \
      --ssh-flag="-o StrictHostKeyChecking=no" \
      --ssh-flag="-o LogLevel=ERROR" \
      --command="wc -c < $remote_log 2>/dev/null || echo 0" 2>/dev/null \
      | tr -d '[:space:]')"
    total_bytes="${total_bytes:-0}"
    if [[ "$total_bytes" =~ ^[0-9]+$ ]] && (( total_bytes > printed_bytes )); then
      timeout 60 gcloud compute ssh "$cname" \
        --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
        --ssh-flag="-o UserKnownHostsFile=/dev/null" \
        --ssh-flag="-o StrictHostKeyChecking=no" \
        --ssh-flag="-o LogLevel=ERROR" \
        --command="tail -c +$((printed_bytes + 1)) $remote_log 2>/dev/null" 2>/dev/null \
        | tee -a "$local_log" >&2
      printed_bytes=$total_bytes
    fi
    local active
    active="$(timeout 60 gcloud compute ssh "$cname" \
      --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
      --ssh-flag="-o UserKnownHostsFile=/dev/null" \
      --ssh-flag="-o StrictHostKeyChecking=no" \
      --ssh-flag="-o LogLevel=ERROR" \
      --command="systemctl is-active $unit 2>/dev/null || true" 2>/dev/null \
      | tr -d '[:space:]')"
    if [[ "$active" != "active" && "$active" != "activating" ]]; then
      log "[$cname] $unit final state: '$active' ($total_bytes bytes printed)"
      local exit_code
      exit_code="$(timeout 60 gcloud compute ssh "$cname" \
        --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
        --ssh-flag="-o UserKnownHostsFile=/dev/null" \
        --ssh-flag="-o StrictHostKeyChecking=no" \
        --ssh-flag="-o LogLevel=ERROR" \
        --command="systemctl show $unit --property=ExecMainStatus --value 2>/dev/null || echo 0" 2>/dev/null \
        | tr -d '[:space:]')"
      poll_rc="${exit_code:-1}"
      break
    fi
    sleep 60
  done

  # Flush any final bytes the last poll missed.
  local total_bytes_final
  total_bytes_final="$(timeout 60 gcloud compute ssh "$cname" \
    --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
    --ssh-flag="-o UserKnownHostsFile=/dev/null" \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    --ssh-flag="-o LogLevel=ERROR" \
    --command="wc -c < $remote_log 2>/dev/null || echo 0" 2>/dev/null \
    | tr -d '[:space:]')"
  total_bytes_final="${total_bytes_final:-0}"
  if [[ "$total_bytes_final" =~ ^[0-9]+$ ]] && (( total_bytes_final > printed_bytes )); then
    timeout 60 gcloud compute ssh "$cname" \
      --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
      --ssh-flag="-o UserKnownHostsFile=/dev/null" \
      --ssh-flag="-o StrictHostKeyChecking=no" \
      --ssh-flag="-o LogLevel=ERROR" \
      --command="tail -c +$((printed_bytes + 1)) $remote_log 2>/dev/null" 2>/dev/null \
      | tee -a "$local_log" >&2
  fi

  # Always try to fetch the JSON — even on failure, the bench may have
  # flushed partial data (one stage per completed fill_percent).
  log "retrieving $remote_out -> $local_out (regardless of exit status)"
  scp_from "$cname" "$remote_out" "$local_out" 2>/dev/null \
    || log "WARN: scp_from $remote_out failed (no output produced or coord unreachable)"
  if (( poll_rc != 0 )); then
    log "[$cname] aegon_publish_bench exited with status $poll_rc"
    return "$poll_rc"
  fi
  log "publish-bench complete: $local_out"
}

cmd_bench() {
  require_project
  require_power_of_two "$N_SHARDS"

  local cname; cname="$(coord_name)"
  local csv; csv="$(shard_endpoints_csv)"
  log "endpoints (first 2 shown): $(echo "$csv" | cut -d, -f1-2),..."
  log "db: $COORD_DB_PATH (RocksDB on coord)"
  log "[$cname] running aegon_coordinator_bench"
  remote "$cname" \
    "sudo prlimit --pid \$\$ --nofile=1048576:1048576 && \
     mkdir -p \$HOME/aegon-run \$HOME/artifacts/srs && \
     cd \$HOME/aegon-run && \
     $REMOTE_BIN_DIR/aegon_coordinator_bench \
       --shard-log-capacity $SHARD_LOG_CAPACITY \
       --kzh-k $KZH_K \
       --setup-seed $SETUP_SEED \
       --endpoints $csv \
       --batch-sizes $BATCH_SIZES \
       --samples-per-batch $SAMPLES_PER_BATCH \
       --db-path $COORD_DB_PATH \
       --output $REMOTE_BENCH_OUT" \
    stream
  log "retrieving $REMOTE_BENCH_OUT -> $LOCAL_BENCH_OUT"
  scp_from "$cname" "$REMOTE_BENCH_OUT" "$LOCAL_BENCH_OUT"
  log "bench JSON saved to $LOCAL_BENCH_OUT"
}

# Run aegon_lookup_bench on the coordinator. The lookup bench mirrors
# the publish-bench cluster flow: walks LOOKUP_FILL_PERCENTS (default
# 1,30,60,90), restarts shards with the per-shard anonymous prefill
# matching each fill level, bootstraps (cache hit -> fast), then runs
# aegon_lookup_bench against the live cluster with a small
# LOOKUP_PRELOAD_COUNT of sampleable labels published on top. At each
# stage the bench measures:
#   - lookup latencies (server + client × {label, value, value-history,
#     label-history})
#   - data + proof + total payload sizes per lookup type
#   - publish-bench sweep over LOOKUP_PUBLISH_BATCH_SIZES (default
#     matches PUBLISH_BATCH_SIZES — 4096..131072 for large)
#
# Why scp the binary fresh each time: the deploy step only pushes
# aegon_coordinator_bench, not aegon_lookup_bench. If you're running
# the lookup bench after a deploy, the binary isn't on the coord yet.
LOOKUP_PRELOAD_COUNT="${LOOKUP_PRELOAD_COUNT:-1000}"
LOOKUP_SAMPLES_PER_LEVEL="${LOOKUP_SAMPLES_PER_LEVEL:-20}"
LOOKUP_FILL_PERCENTS="${LOOKUP_FILL_PERCENTS:-${PUBLISH_FILL_PERCENTS}}"
LOOKUP_PUBLISH_BATCH_SIZES="${LOOKUP_PUBLISH_BATCH_SIZES:-${PUBLISH_BATCH_SIZES}}"
LOOKUP_PUBLISH_SAMPLES_PER_BATCH="${LOOKUP_PUBLISH_SAMPLES_PER_BATCH:-${PUBLISH_SAMPLES_PER_BATCH}}"
LOOKUP_TRUE_LOG_CAP="${LOOKUP_TRUE_LOG_CAP:-${PUBLISH_TRUE_LOG_CAP}}"
# Climb batch size for inter-fill real publishes. Matches the
# publish-bench warmup batch so the climb finishes in a sensible
# time at large fill levels (1024 was a small-bench default that
# would balloon medium's 20M-entry warmup to ~19k epochs).
LOOKUP_PUBLISH_BATCH_SIZE="${LOOKUP_PUBLISH_BATCH_SIZE:-${PUBLISH_WARMUP_BATCH_SIZE:-16384}}"
LOOKUP_AUDIT_SAMPLES="${LOOKUP_AUDIT_SAMPLES:-5}"
# Throughput sweep: comma-separated concurrency levels for the
# per-stage QPS-vs-N measurement. Empty = skip (preserves the old
# behavior). The sweep runs after lookup samples + audit, before
# publish_bench, so per-fill cluster state is exactly what audit saw.
LOOKUP_THROUGHPUT_CONCURRENCIES="${LOOKUP_THROUGHPUT_CONCURRENCIES:-}"
LOOKUP_THROUGHPUT_WINDOW_SECS="${LOOKUP_THROUGHPUT_WINDOW_SECS:-20}"
LOOKUP_THROUGHPUT_WARMUP_SECS="${LOOKUP_THROUGHPUT_WARMUP_SECS:-3}"
LOOKUP_THROUGHPUT_LOOKUP_KIND="${LOOKUP_THROUGHPUT_LOOKUP_KIND:-value}"
LOCAL_LOOKUP_BENCH_DIR="${LOCAL_LOOKUP_BENCH_DIR:-/tmp/aegon-lookup-bench}"
cmd_lookup_bench() {
  require_project
  require_power_of_two "$N_SHARDS"
  mkdir -p "$LOCAL_LOOKUP_BENCH_DIR"
  dump_bench_config
  log "  LOOKUP_FILL_PERCENTS   = $LOOKUP_FILL_PERCENTS"
  log "  LOOKUP_TRUE_LOG_CAP    = $LOOKUP_TRUE_LOG_CAP"
  log "  LOOKUP_PRELOAD_COUNT   = $LOOKUP_PRELOAD_COUNT"

  # Need the binary on the coord. The default deploy step doesn't push
  # it, so do that here (idempotent — same as cmd_deploy's coord push
  # but for the lookup binary).
  local cargo_features=""
  if [[ "${TRACING:-0}" == "1" ]]; then
    cargo_features="--features tracing_instrument"
  fi
  local local_bin_dir="$REPO_ROOT/target/release"
  local remote_bin_dir="$local_bin_dir"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    command -v docker >/dev/null || die "macOS host needs Docker (linux/amd64 build container)"
    docker info >/dev/null 2>&1  || die "Docker daemon unreachable"
    log "macOS host: building aegon_lookup_bench inside Docker"
    docker run --rm --platform linux/amd64 \
      -v "$REPO_ROOT:/workspace" -w /workspace \
      rust:slim-bookworm \
      bash -c "set -e; \
        apt-get update >/dev/null && \
        apt-get install -y --no-install-recommends protobuf-compiler ca-certificates >/dev/null && \
        cargo build --release -p akd $cargo_features --target x86_64-unknown-linux-gnu \
          --bin aegon_lookup_bench"
    remote_bin_dir="$REPO_ROOT/target/x86_64-unknown-linux-gnu/release"
  else
    log "building aegon_lookup_bench (release)"
    (cd "$REPO_ROOT" && cargo build --release -p akd $cargo_features --bin aegon_lookup_bench) >/dev/null
  fi
  [[ -x "$remote_bin_dir/aegon_lookup_bench" ]] || die "aegon_lookup_bench missing"

  # Bench now runs on a dedicated bench-client VM (not the coord) so
  # the bench's request-driving CPU doesn't contend with the coord's
  # tonic worker pool, and lookup RPCs hit the shard cluster over the
  # real VPC network. Per-request latency is read out of the
  # `server_processing_micros` field on each response, so RTT between
  # the bench-client and the in-process coord is excluded.
  local cname; cname="$(bench_client_name)"
  local shard_csv; shard_csv="$(shard_endpoints_csv)"
  log "[$cname] uploading aegon_lookup_bench"
  scp_to "$cname" "$remote_bin_dir/aegon_lookup_bench"
  remote "$cname" "sudo mkdir -p $REMOTE_BIN_DIR && \
    sudo mv /tmp/aegon_lookup_bench $REMOTE_BIN_DIR/ && \
    sudo chmod +x $REMOTE_BIN_DIR/aegon_lookup_bench"

  # Single combined invocation: aegon_lookup_bench walks --fill-percents
  # internally, climbing INCREMENTALLY between them via real publish
  # (no prefill_random shortcut). At each fill level the binary runs
  # lookups + audit + publish-batch-sweep in order, then advances to the
  # next fill — so the warmup work is paid exactly once per fill level
  # across the entire run (the publish-bench's work is folded in, not
  # duplicated). Prereq: shards already started + at epoch 0
  # (`bench-cluster.sh start-shards`).

  log "[$cname] wiping $COORD_DB_PATH (single wipe; no inter-fill resets)"
  remote "$cname" "sudo rm -rf $COORD_DB_PATH && sudo mkdir -p $COORD_DB_PATH && sudo chown \$(whoami) $COORD_DB_PATH"

  # Launch the bench inside a systemd transient unit so it SURVIVES
  # SSH disconnects. Earlier runs lost ~7h of climb work when the
  # local host suspended and the long-lived `gcloud ssh ... stream`
  # session broke, killing the bench process (which was a child of
  # that SSH command). With systemd-run --collect, the bench is
  # reparented to systemd, so SSH death doesn't reach it.
  local remote_out="/tmp/aegon-lookup-bench.json"
  local remote_log="/tmp/aegon-lookup-bench.log"
  local unit="aegon-lookup-bench"
  local me; me="$(whoami)"
  log "[$cname] starting aegon_lookup_bench as systemd unit '$unit'"
  remote "$cname" "
    sudo systemctl reset-failed $unit 2>/dev/null || true
    sudo systemctl stop $unit 2>/dev/null || true
    rm -f $remote_log $remote_out
    # Pre-create the working directory *before* systemd-run — systemd
    # chdir's there before spawning bash, so the inner 'mkdir -p' would
    # run too late and the unit would fail with status=200/CHDIR.
    mkdir -p /home/$me/aegon-run
    sudo systemd-run \
      --unit=$unit \
      --description='Aegon lookup-bench' \
      --uid=$me --gid=$me \
      --working-directory=/home/$me/aegon-run \
      --setenv=HOME=/home/$me \
      --setenv=AEGON_ROCKSDB_STATS_DUMP_SEC=${AEGON_ROCKSDB_STATS_DUMP_SEC:-60} \
      --setenv=AEGON_ROCKSDB_BLOCK_CACHE_GB=${AEGON_ROCKSDB_BLOCK_CACHE_GB:-16} \
      --setenv=AEGON_ROCKSDB_PARALLELISM=${AEGON_ROCKSDB_PARALLELISM:-16} \
      --property=LimitNOFILE=1048576 \
      bash -c '
        cd /home/$me/aegon-run
        $REMOTE_BIN_DIR/aegon_lookup_bench \
          --shard-log-capacity $SHARD_LOG_CAPACITY \
          --true-log-capacity $LOOKUP_TRUE_LOG_CAP \
          --kzh-k $KZH_K \
          --n-shards $N_SHARDS \
          --setup-seed $SETUP_SEED \
          --endpoints $shard_csv \
          --db-path $COORD_DB_PATH \
          --fill-percents $LOOKUP_FILL_PERCENTS \
          --prefill-seed $PREFILL_SEED \
          --samples-per-level $LOOKUP_SAMPLES_PER_LEVEL \
          --publish-batch-size $LOOKUP_PUBLISH_BATCH_SIZE \
          --publish-batch-sizes $LOOKUP_PUBLISH_BATCH_SIZES \
          --publish-samples-per-batch $LOOKUP_PUBLISH_SAMPLES_PER_BATCH \
          --audit-samples $LOOKUP_AUDIT_SAMPLES \
          ${LOOKUP_THROUGHPUT_CONCURRENCIES:+--throughput-concurrencies $LOOKUP_THROUGHPUT_CONCURRENCIES} \
          --throughput-window-secs $LOOKUP_THROUGHPUT_WINDOW_SECS \
          --throughput-warmup-secs $LOOKUP_THROUGHPUT_WARMUP_SECS \
          --throughput-lookup-kind $LOOKUP_THROUGHPUT_LOOKUP_KIND \
          --private \
          --output $remote_out > $remote_log 2>&1
      '
    echo LAUNCHED
  "

  # Poll-loop on the LOCAL side. Each iteration is a short, recoverable
  # ssh command (not a long-lived stream that can break). When the
  # systemd unit deactivates (success or failure), we exit the loop.
  local local_out="$LOCAL_LOOKUP_BENCH_DIR/${LOCAL_OUT_NAME_PREFIX:-}combined.json"
  local local_log="$LOCAL_LOOKUP_BENCH_DIR/${LOCAL_OUT_NAME_PREFIX:-}lookup-bench.log"
  mkdir -p "$LOCAL_LOOKUP_BENCH_DIR"
  : > "$local_log"   # truncate
  log "[$cname] polling $unit + streaming $remote_log -> $local_log (resilient to SSH death)"
  local printed_bytes=0
  local poll_rc=0
  while true; do
    # Pull new log bytes since last fetched offset. Use byte-offset
    # rather than line count so each poll is O(diff) not O(whole log).
    local total_bytes
    total_bytes="$(timeout 60 gcloud compute ssh "$cname" \
      --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
      --ssh-flag="-o UserKnownHostsFile=/dev/null" \
      --ssh-flag="-o StrictHostKeyChecking=no" \
      --ssh-flag="-o LogLevel=ERROR" \
      --command="wc -c < $remote_log 2>/dev/null || echo 0" 2>/dev/null \
      | tr -d '[:space:]')"
    total_bytes="${total_bytes:-0}"
    if [[ "$total_bytes" =~ ^[0-9]+$ ]] && (( total_bytes > printed_bytes )); then
      timeout 60 gcloud compute ssh "$cname" \
        --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
        --ssh-flag="-o UserKnownHostsFile=/dev/null" \
        --ssh-flag="-o StrictHostKeyChecking=no" \
        --ssh-flag="-o LogLevel=ERROR" \
        --command="tail -c +$((printed_bytes + 1)) $remote_log 2>/dev/null" 2>/dev/null \
        | tee -a "$local_log" >&2
      printed_bytes=$total_bytes
    fi

    # Check unit status. is-active prints active|inactive|failed.
    local active
    active="$(timeout 60 gcloud compute ssh "$cname" \
      --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
      --ssh-flag="-o UserKnownHostsFile=/dev/null" \
      --ssh-flag="-o StrictHostKeyChecking=no" \
      --ssh-flag="-o LogLevel=ERROR" \
      --command="systemctl is-active $unit 2>/dev/null || true" 2>/dev/null \
      | tr -d '[:space:]')"
    if [[ "$active" != "active" && "$active" != "activating" ]]; then
      log "[$cname] $unit final state: '$active' (started=$total_bytes bytes printed)"
      # Capture the unit's exit code.
      local exit_code
      exit_code="$(timeout 60 gcloud compute ssh "$cname" \
        --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
        --ssh-flag="-o UserKnownHostsFile=/dev/null" \
        --ssh-flag="-o StrictHostKeyChecking=no" \
        --ssh-flag="-o LogLevel=ERROR" \
        --command="systemctl show $unit --property=ExecMainStatus --value 2>/dev/null || echo 0" 2>/dev/null \
        | tr -d '[:space:]')"
      poll_rc="${exit_code:-1}"
      break
    fi
    sleep 60
  done

  # Flush any final bytes the last poll missed.
  local total_bytes_final
  total_bytes_final="$(timeout 60 gcloud compute ssh "$cname" \
    --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
    --ssh-flag="-o UserKnownHostsFile=/dev/null" \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    --ssh-flag="-o LogLevel=ERROR" \
    --command="wc -c < $remote_log 2>/dev/null || echo 0" 2>/dev/null \
    | tr -d '[:space:]')"
  total_bytes_final="${total_bytes_final:-0}"
  if [[ "$total_bytes_final" =~ ^[0-9]+$ ]] && (( total_bytes_final > printed_bytes )); then
    timeout 60 gcloud compute ssh "$cname" \
      --zone="$ZONE" --tunnel-through-iap --strict-host-key-checking=no --quiet \
      --ssh-flag="-o UserKnownHostsFile=/dev/null" \
      --ssh-flag="-o StrictHostKeyChecking=no" \
      --ssh-flag="-o LogLevel=ERROR" \
      --command="tail -c +$((printed_bytes + 1)) $remote_log 2>/dev/null" 2>/dev/null \
      | tee -a "$local_log" >&2
  fi

  # Always try to fetch the JSON — even on failure, the bench may have
  # flushed partial data (one entry per completed fill level). Losing
  # that data because we returned early on non-zero exit was the v2
  # failure mode: we got level 1+2 worth of flushed JSON but the
  # script bailed before scp_from, then teardown wiped the coord.
  #
  # The fetch can itself fail (coord dead, file missing), so swallow
  # its error and surface the original bench rc instead.
  log "retrieving $remote_out -> $local_out (regardless of exit status)"
  if scp_from "$cname" "$remote_out" "$local_out" 2>&1; then
    log "lookup-bench JSON fetched: $local_out"
  else
    log "WARN: failed to fetch $remote_out (coord may be down or file missing)"
  fi

  if [[ "$poll_rc" != "0" ]]; then
    log "[$cname] lookup-bench unit exited non-zero (status=$poll_rc); see $local_log"
    return "$poll_rc"
  fi

  log "lookup-bench complete: $local_out"
}

cmd_logs() {
  require_project
  local idx="${1:-0}"
  local lines="${2:-}"
  local name; name="$(shard_name "$idx")"
  if [[ -n "$lines" ]]; then
    # One-shot: print the last $lines and exit. Use this when you want
    # to grep/tail/pipe; `tail -f` would block downstream filters.
    log "[$name] tail -n $lines /tmp/aegon-shard.log"
    remote "$name" "tail -n $lines /tmp/aegon-shard.log"
  else
    log "[$name] tail -f /tmp/aegon-shard.log (Ctrl-C to stop)"
    remote "$name" "tail -f /tmp/aegon-shard.log" stream
  fi
}

probe_one() {
  # Single-host status probe. $1 = instance name, $2 = expected
  # process name pattern (truncated to 15 chars to match
  # /proc/<pid>/comm — pgrep -x is what we use). Writes one line to
  # stdout with the form:
  #   "[name] OK  pid=N rss=NMB freeMB=N oom=0 tail=..."
  # or
  #   "[name] BAD reason=... pid=... freeMB=... oom=...   tail=..."
  # Exits non-zero on BAD so the watchdog aggregator can count
  # failures via wait $pid && ... || failed=$((failed+1)).
  local name="$1"
  local proc="$2"
  local out
  out="$(remote "$name" "set +e; \
    pid=\$(pgrep -x ${proc} | head -1); \
    free=\$(free -m | awk '/^Mem:/ {print \$7}'); \
    oom=\$(sudo dmesg 2>/dev/null | grep -ciE 'out of memory|killed process|invoked oom-killer' || echo 0); \
    tail=\$(tail -n 1 /tmp/aegon-shard.log 2>/dev/null | head -c 80); \
    if [ -z \"\$pid\" ]; then \
      echo \"BAD reason=process-dead pid=- freeMB=\$free oom=\$oom tail=\$tail\"; \
      exit 1; \
    fi; \
    rss=\$(ps -o rss= -p \$pid 2>/dev/null | awk '{print int(\$1/1024)}'); \
    if [ \"\$oom\" -gt 0 ] 2>/dev/null; then \
      echo \"BAD reason=oom-in-dmesg pid=\$pid rssMB=\$rss freeMB=\$free oom=\$oom tail=\$tail\"; \
      exit 1; \
    fi; \
    echo \"OK  pid=\$pid rssMB=\$rss freeMB=\$free oom=0 tail=\$tail\"; \
    exit 0" 2>/dev/null)"
  local rc=$?
  if [[ -z "$out" ]]; then
    out="BAD reason=ssh-failed-or-no-output"
    rc=1
  fi
  printf "[%-28s] %s\n" "$name" "$out"
  return $rc
}

cmd_watchdog() {
  # Parallel health probe across every VM in the cluster. Use this:
  #   * once before a long phase (`watchdog`) to sanity-check no host
  #     died between deploy and bench
  #   * in a loop in the background during long phases:
  #       while true; do ./bench-cluster.sh watchdog; sleep 60; done \
  #           >> /tmp/aegon-watchdog.log 2>&1 &
  # Exits 0 only if every host probe came back OK. Lines beginning
  # with "BAD" are the ones to look at — they include the reason
  # (process-dead / oom-in-dmesg / ssh-failed), the PID it expected,
  # the available memory, and the last line of /tmp/aegon-shard.log
  # (often the panic backtrace or OOM message).
  #
  # NOTE: `sudo dmesg` is needed on recent Ubuntu (kernel.dmesg_restrict=1).
  # If the deploy account doesn't have passwordless sudo for dmesg, the
  # check silently becomes "oom=0" and only the process-dead check fires.
  # That's still useful — a real OOM kill takes the process down — but
  # consider patching sudoers for full coverage on long-running benches.
  require_project
  require_power_of_two "$N_SHARDS"

  log "watchdog: probing $((N_SHARDS + 2)) hosts in parallel"
  local tmpdir; tmpdir="$(mktemp -d)"
  local -a pids=()
  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    ( probe_one "$name" "aegon_shard_se" > "$tmpdir/shard-$i" ; echo $? > "$tmpdir/shard-$i.rc" ) &
    pids+=("$!")
  done
  local cname; cname="$(coord_name)"
  # Coord can be running either bench binary; check whichever exists.
  # We pass the longer match prefix that both bench bins share.
  ( probe_one "$cname" "aegon_coordinat" > "$tmpdir/coord" ; echo $? > "$tmpdir/coord.rc" ) &
  pids+=("$!")

  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  # Replay outputs in deterministic order (shard-0 .. shard-N, coord)
  # and tally failures.
  local failed=0
  for ((i = 0; i < N_SHARDS; i++)); do
    cat "$tmpdir/shard-$i"
    local rc; rc="$(cat "$tmpdir/shard-$i.rc" 2>/dev/null || echo 1)"
    (( rc != 0 )) && failed=$((failed+1))
  done
  cat "$tmpdir/coord"
  local rc; rc="$(cat "$tmpdir/coord.rc" 2>/dev/null || echo 1)"
  (( rc != 0 )) && failed=$((failed+1))
  rm -rf "$tmpdir"

  if (( failed > 0 )); then
    log "watchdog: $failed host(s) BAD"
    return 1
  fi
  log "watchdog: all $((N_SHARDS + 1)) hosts OK"
  return 0
}

cmd_down() {
  require_project
  log "tearing down bench cluster"

  # Collect all aegon-bench-* instances in one shot and delete them
  # in a single gcloud invocation (which deletes them in parallel
  # server-side). Serial per-VM delete via `gcloud delete` runs ~1
  # VM/minute and turns a 128-shard teardown into a 2-hour job.
  local insts
  insts="$(gcloud compute instances list \
    --filter="name~^aegon-bench-" \
    --zones="$ZONE" \
    --format='value(name)' 2>/dev/null | tr '\n' ' ')"
  if [[ -n "$insts" ]]; then
    log "deleting $(echo "$insts" | wc -w) instance(s) in parallel"
    gcloud compute instances delete $insts --zone="$ZONE" --quiet >/dev/null || \
      log "WARN: some instance deletions failed; check console"
  else
    log "no aegon-bench-* instances to delete"
  fi

  for fw in "$FIREWALL_GRPC" "$FIREWALL_BENCH_CLIENT_GRPC" "$FIREWALL_SRS" "$FIREWALL_SSH" "aegon-bench-masking"; do
    if gcloud compute firewall-rules describe "$fw" >/dev/null 2>&1; then
      log "deleting firewall $fw"
      gcloud compute firewall-rules delete "$fw" --quiet >/dev/null
    fi
  done

  if gcloud compute routers nats describe "$NAT" --router="$ROUTER" --region="$REGION" >/dev/null 2>&1; then
    log "deleting cloud nat $NAT"
    gcloud compute routers nats delete "$NAT" --router="$ROUTER" --region="$REGION" --quiet >/dev/null
  fi
  if gcloud compute routers describe "$ROUTER" --region="$REGION" >/dev/null 2>&1; then
    log "deleting cloud router $ROUTER"
    gcloud compute routers delete "$ROUTER" --region="$REGION" --quiet >/dev/null
  fi

  if gcloud compute networks describe "$NETWORK" >/dev/null 2>&1; then
    log "deleting VPC $NETWORK"
    gcloud compute networks delete "$NETWORK" --quiet >/dev/null
  fi
  log "teardown complete"
}

usage() {
  cat <<EOF
usage: $0 <subcommand>

  up               Provision VPC, firewall, $N_SHARDS shards + coordinator
  deploy           Build binaries, push to every node, start shard servers
                   in "awaiting-bootstrap" state. Shards bind their
                   SrsService on :$SRS_PORT and wait for the bootstrap
                   step before doing any SRS / prefill work.
  bootstrap        Run aegon_srs_bootstrap on the coordinator: sample
                   trapdoors from --setup-seed, push them to every
                   shard's SrsService, poll WaitForReady until all
                   shards finish distributed SRS gen + prefill. On
                   cache hit (re-runs with the same seed) the
                   bootstrap completes in seconds.
  setup-bench      Large-regime setup benchmark. Wipes every
                   shard's SRS cache, restarts shards with
                   prefill_count=0, then runs aegon_srs_bootstrap with
                   metrics gathering. Outputs one JSON record with
                   per-shard phase timestamps + inbound/outbound
                   slab bytes + pk/vk/universal sizes.
  publish-bench    Large-regime publish-time + commit-size bench.
                   Walks PUBLISH_FILL_PERCENTS (default 1,30,60,90),
                   restarting shards per stage with the per-shard
                   prefill count for that fill level. For each stage,
                   sweeps PUBLISH_BATCH_SIZES (default
                   4096..131072) and emits one JSON per fill level
                   under LOCAL_PUBLISH_BENCH_DIR (default
                   /tmp/aegon-publish-bench). Bootstrap between
                   stages should hit cache and be near-instant.
  start-shards     Start all shards once with --prefill-count 0 (per-fill
                   prefill is now driven via the ReconfigurePrefill RPC
                   issued by the bench binaries themselves). Run once
                   after setup-bench; bench binaries handle the rest.
  bench            Run aegon_coordinator_bench on the coordinator, fetch JSON
  lookup-bench     Large-regime combined lookup + publish bench.
                   Walks LOOKUP_FILL_PERCENTS (default mirrors
                   PUBLISH_FILL_PERCENTS = 1,30,60,90), restarting
                   shards per stage with the matching per-shard
                   anonymous prefill. At each stage, publishes
                   LOOKUP_PRELOAD_COUNT (default 1000) sampleable
                   labels and runs:
                   - lookup samples (server + client × {label, value,
                     value-history, label-history}) with data/proof/
                     total size reporting per the user's spec.
                   - publish-bench sweep over
                     LOOKUP_PUBLISH_BATCH_SIZES (default mirrors
                     PUBLISH_BATCH_SIZES).
                   - LOOKUP_AUDIT_SAMPLES (default 5) consecutive
                     epoch-transition audits via
                     `verify_sharded_invariance`, recording
                     per-epoch audit time + audit proof bytes.
                   Outputs one JSON per fill level under
                   LOCAL_LOOKUP_BENCH_DIR (default
                   /tmp/aegon-lookup-bench).
  watchdog         Parallel health probe across every VM (shards + coord +
                   db): checks process liveness, free memory, dmesg OOM
                   marks. Exits non-zero if any host is BAD. Wrap in a
                   loop during long phases.
  logs N           Tail the shard log on $SHARD_TAG-N
  down             Delete every instance + the VPC

Required env: PROJECT=<gcp-project-id>
Optional env: ZONE, N_SHARDS, SHARD_LOG_CAPACITY, KZH_K,
              TOTAL_PRELOAD_LOG2, BATCH_SIZES, SAMPLES_PER_BATCH,
              SETUP_SEED, PREFILL_SEED,
              SHARD_MACHINE_TYPE, COORD_MACHINE_TYPE,
              LOCAL_BENCH_OUT

At defaults (128 shards × n2-standard-16): ~\$100/hr list, ~\$30/hr with
3-year committed-use discount. Run \`down\` aggressively when idle.
EOF
}

main() {
  local sub="${1:-}"
  shift || true
  case "$sub" in
    up)             cmd_up ;;
    deploy)         cmd_deploy ;;
    bootstrap)      cmd_bootstrap ;;
    setup-bench)    cmd_setup_bench ;;
    publish-bench)  cmd_publish_bench ;;
    start-shards)   cmd_start_shards ;;
    start-masking)  cmd_start_masking ;;
    bench)          cmd_bench ;;
    lookup-bench)   cmd_lookup_bench ;;
    watchdog)       cmd_watchdog ;;
    logs)           cmd_logs "$@" ;;
    down)           cmd_down ;;
    ""|-h|--help|help) usage ;;
    *) usage; die "unknown subcommand: $sub" ;;
  esac
}

main "$@"
