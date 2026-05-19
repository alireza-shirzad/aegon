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
#   COORD_MACHINE_TYPE  GCE machine type coordinator  (n2-standard-4)
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
#     (0/30/60/90%) on commodity n2-standard-16 with no per-stage
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
COORD_MACHINE_TYPE="${COORD_MACHINE_TYPE:-n2-standard-4}"
REDIS_MACHINE_TYPE="${REDIS_MACHINE_TYPE:-n2-standard-2}"

NETWORK="aegon-bench-vpc"
FIREWALL_GRPC="aegon-bench-grpc"
FIREWALL_SRS="aegon-bench-srs"
FIREWALL_SSH="aegon-bench-ssh"
FIREWALL_REDIS="aegon-bench-redis"
SHARD_TAG="aegon-bench-shard"
COORD_TAG="aegon-bench-coord"
DB_TAG="aegon-bench-db"
SHARD_PORT=50051
# Distributed-SRS-bootstrap port. Each shard binds aegon_shard_server's
# SrsService here (separate listener from SHARD_PORT) so peers can pull
# H_t slabs from each other during distributed gen, and so the
# `aegon_srs_bootstrap` binary on the coordinator can push trapdoors +
# poll WaitForReady. Kept distinct from SHARD_PORT so the coordinator's
# normal shard client never accidentally hits the SRS-only listener.
SRS_PORT=50052
# On-shard cache directory for the assembled SRS. After the first
# `bootstrap`, every shard caches its SRS here, so a subsequent boot
# (e.g. via `restart-shards`) short-circuits the distributed exchange
# and reaches Ready in seconds rather than minutes.
SRS_CACHE_DIR='$HOME/artifacts/srs-cache'
REDIS_PORT=6379
ROUTER="aegon-bench-router"
NAT="aegon-bench-nat"
REGION="${ZONE%-*}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
         --zone="$ZONE" --tunnel-through-iap --quiet --command="true" \
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
  if [[ "${3:-}" == "stream" ]]; then
    gcloud compute ssh "$instance" --zone="$ZONE" --tunnel-through-iap --command="$cmd"
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
    gcloud compute ssh "$instance" --zone="$ZONE" --tunnel-through-iap \
      --quiet --command="$cmd" &
    local _ssh_pid=$!
    ( sleep 120 && kill "$_ssh_pid" 2>/dev/null ) &
    local _watchdog_pid=$!
    wait "$_ssh_pid" 2>/dev/null || true
    kill "$_watchdog_pid" 2>/dev/null || true
  else
    gcloud compute ssh "$instance" --zone="$ZONE" --tunnel-through-iap --quiet --command="$cmd"
  fi
}

scp_to() {
  local instance="$1"; shift
  gcloud compute scp --zone="$ZONE" --tunnel-through-iap --quiet "$@" "$instance:/tmp/"
}

scp_from() {
  local instance="$1"; shift
  local src="$1"; shift
  local dst="$1"; shift
  gcloud compute scp --zone="$ZONE" --tunnel-through-iap --quiet "$instance:$src" "$dst"
}

shard_name() { echo "${SHARD_TAG}-$1"; }
coord_name() { echo "${COORD_TAG}"; }
db_name()    { echo "${DB_TAG}"; }

shard_internal_ip() {
  gcloud compute instances describe "$(shard_name "$1")" --zone="$ZONE" \
    --format='value(networkInterfaces[0].networkIP)'
}

db_internal_ip() {
  gcloud compute instances describe "$(db_name)" --zone="$ZONE" \
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

# Same shape as `shard_endpoints_csv` but pointing at every shard's
# distributed-SRS-bootstrap port. Consumed by `cmd_bootstrap` to tell
# the bootstrap binary where each shard's `SrsService` listener lives
# (and pushed through to every shard so they know each other's
# GetSrsSlab URLs for the slab exchange phase).
srs_endpoints_csv() {
  local out=()
  for ((i = 0; i < N_SHARDS; i++)); do
    out+=("http://$(shard_internal_ip "$i"):$SRS_PORT")
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

  # ---- firewall: SRS-bootstrap port (shard <-> shard, coord -> shard) ----
  # Two source tags here: shards talk to each other on SRS_PORT for the
  # H_t slab exchange (every shard pulls (N-1)/N of every tensor from
  # its peers), and the coordinator talks to shards on SRS_PORT so the
  # `aegon_srs_bootstrap` binary running on the coord can push trapdoors
  # and poll WaitForReady.
  local srs_want="$SHARD_TAG,$COORD_TAG"
  if gcloud compute firewall-rules describe "$FIREWALL_SRS" >/dev/null 2>&1; then
    local srs_have
    srs_have="$(gcloud compute firewall-rules describe "$FIREWALL_SRS" \
      --format='value(sourceTags.list())' 2>/dev/null)"
    if [[ "$srs_have" != "$srs_want" ]]; then
      log "firewall $FIREWALL_SRS: updating source-tags to '$srs_want'"
      gcloud compute firewall-rules update "$FIREWALL_SRS" \
        --source-tags="$srs_want" >/dev/null
    else
      log "firewall $FIREWALL_SRS already correct"
    fi
  else
    log "creating firewall $FIREWALL_SRS (shard+coord -> shards:$SRS_PORT)"
    gcloud compute firewall-rules create "$FIREWALL_SRS" \
      --network="$NETWORK" \
      --allow="tcp:$SRS_PORT" \
      --source-tags="$srs_want" \
      --target-tags="$SHARD_TAG" >/dev/null
  fi

  # ---- firewall: coordinator + shards -> redis on the DB machine ----
  local redis_want="$COORD_TAG,$SHARD_TAG"
  if gcloud compute firewall-rules describe "$FIREWALL_REDIS" >/dev/null 2>&1; then
    local redis_have
    redis_have="$(gcloud compute firewall-rules describe "$FIREWALL_REDIS" \
      --format='value(sourceTags.list())' 2>/dev/null)"
    if [[ "$redis_have" != "$redis_want" ]]; then
      log "firewall $FIREWALL_REDIS: updating source-tags to '$redis_want'"
      gcloud compute firewall-rules update "$FIREWALL_REDIS" \
        --source-tags="$redis_want" >/dev/null
    else
      log "firewall $FIREWALL_REDIS already correct"
    fi
  else
    log "creating firewall $FIREWALL_REDIS (coord+shards -> db:$REDIS_PORT)"
    gcloud compute firewall-rules create "$FIREWALL_REDIS" \
      --network="$NETWORK" \
      --allow="tcp:$REDIS_PORT" \
      --source-tags="$redis_want" \
      --target-tags="$DB_TAG" >/dev/null
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
      --tags="$SHARD_TAG" \
      --image-family="ubuntu-2604-lts-amd64" --image-project="ubuntu-os-cloud" \
      --boot-disk-size=100GB >/dev/null
  done

  # ---- coordinator ----
  local cname; cname="$(coord_name)"
  if gcloud compute instances describe "$cname" --zone="$ZONE" >/dev/null 2>&1; then
    log "$cname exists, skipping"
  else
    log "creating $cname ($COORD_MACHINE_TYPE)"
    gcloud compute instances create "$cname" \
      --zone="$ZONE" \
      --machine-type="$COORD_MACHINE_TYPE" \
      --network="$NETWORK" \
      --no-address \
      --tags="$COORD_TAG" \
      --image-family="ubuntu-2604-lts-amd64" --image-project="ubuntu-os-cloud" \
      --boot-disk-size=20GB >/dev/null
  fi

  # ---- database (Redis) ----
  # Small VM — Redis is the open-addressing occupancy oracle, not a
  # heavyweight store. Disk is sized for the prefilled keyspace: at the
  # production target of 2^28 slot keys, each ~50 bytes serialised, the
  # working set is ~13 GiB. 20 GB boot disk is enough headroom and lets
  # Redis page out under memory pressure rather than OOM-killing.
  local dname; dname="$(db_name)"
  if gcloud compute instances describe "$dname" --zone="$ZONE" >/dev/null 2>&1; then
    log "$dname exists, skipping"
  else
    log "creating $dname ($REDIS_MACHINE_TYPE) — Redis on :$REDIS_PORT"
    gcloud compute instances create "$dname" \
      --zone="$ZONE" \
      --machine-type="$REDIS_MACHINE_TYPE" \
      --network="$NETWORK" \
      --no-address \
      --tags="$DB_TAG" \
      --image-family="ubuntu-2604-lts-amd64" --image-project="ubuntu-os-cloud" \
      --boot-disk-size=20GB >/dev/null
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
restart_shard() {
  local name="$1"
  local i="$2"
  local per_shard="$3"
  local db_ip="$4"
  local shard_prefill_seed=$((PREFILL_SEED + i))
  log "[$name] starting shard server (shard_id=$i, prefill_count=$per_shard, prefill_seed=$shard_prefill_seed)"
  # Distributed-SRS mode: pass `--srs-bind` + `--srs-cache-dir`. On first
  # boot every shard sits in "awaiting-bootstrap" on SRS_PORT until the
  # coordinator runs `bench-cluster.sh bootstrap`; on subsequent boots
  # (e.g. `restart-shards`) the cache hits and the shard reaches Ready
  # without needing the bootstrap actor.
  #
  # After spawning, we poll /dev/tcp/127.0.0.1/$SRS_PORT for up to 60s
  # to confirm the server is actually listening before declaring
  # restart_shard a success. Without this verification, a SIGTERM'd
  # ssh session (from fire-and-forget) can leave the shell killed
  # before `nohup &` runs, and we silently report "started" with no
  # process — which then makes bootstrap fail with mysterious
  # "transport error" cascades.
  #
  # NOTE: not fire-and-forget. We wait for either readiness or the
  # 60s budget to elapse; either way the remote shell exits and the
  # IAP tunnel tears down naturally.
  remote "$name" "if [ -f /tmp/aegon-shard.pid ]; then \
      kill \$(cat /tmp/aegon-shard.pid) 2>/dev/null || true; \
    fi; \
    pkill -x aegon_shard_ser 2>/dev/null || true; \
    sleep 2; \
    mkdir -p \$HOME/aegon-run $SRS_CACHE_DIR && \
    cd \$HOME/aegon-run && \
    nohup $REMOTE_BIN_DIR/aegon_shard_server \
      --bind 0.0.0.0:$SHARD_PORT \
      --srs-bind 0.0.0.0:$SRS_PORT \
      --srs-cache-dir $SRS_CACHE_DIR \
      --shard-log-capacity $SHARD_LOG_CAPACITY \
      --kzh-k $KZH_K \
      --setup-seed $SETUP_SEED \
      --shard-id $i \
      --prefill-count $per_shard \
      --prefill-seed $shard_prefill_seed \
      --db-url redis://$db_ip:$REDIS_PORT \
      > /tmp/aegon-shard.log 2>&1 < /dev/null & \
    echo \$! > /tmp/aegon-shard.pid; \
    disown 2>/dev/null || true; \
    for poll in \$(seq 1 60); do \
      if exec 9<>/dev/tcp/127.0.0.1/$SRS_PORT 2>/dev/null; then \
        exec 9<&-; exec 9>&-; \
        echo READY; exit 0; \
      fi; \
      sleep 1; \
    done; \
    echo FAILED; tail -n 30 /tmp/aegon-shard.log 2>/dev/null; exit 1"
}

cmd_deploy() {
  require_project
  require_power_of_two "$N_SHARDS"

  # Optional: `TRACING=1` builds with the `tracing_instrument` feature
  # so the bins install a tracing-tree subscriber and emit phase spans
  # (`Aegon::PublishPhase1`, `KZH::FMAState`, `ShardedAegon::*`, ...).
  # Stderr only; the bench JSON is unaffected. Turn off for clean runs.
  local cargo_features=""
  if [[ "${TRACING:-0}" == "1" ]]; then
    cargo_features="--features tracing_instrument"
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
          --bin aegon_shard_server --bin aegon_coordinator_bench --bin aegon_srs_bootstrap"
    remote_bin_dir="$REPO_ROOT/target/x86_64-unknown-linux-gnu/release"
  else
    log "building release binaries (aegon_shard_server, aegon_coordinator_bench, aegon_srs_bootstrap)"
    (cd "$REPO_ROOT" && cargo build --release -p akd $cargo_features \
      --bin aegon_shard_server --bin aegon_coordinator_bench --bin aegon_srs_bootstrap) >/dev/null
  fi
  [[ -x "$remote_bin_dir/aegon_shard_server" ]]      || die "aegon_shard_server missing"
  [[ -x "$remote_bin_dir/aegon_coordinator_bench" ]] || die "aegon_coordinator_bench missing"
  [[ -x "$remote_bin_dir/aegon_srs_bootstrap" ]]     || die "aegon_srs_bootstrap missing"

  local per_shard; per_shard="$(prefill_per_shard)"
  log "shards will prefill ${per_shard} entries each (total = 2^$TOTAL_PRELOAD_LOG2)"

  # ---- DB tier: install (idempotent) + FLUSHALL ----
  # Bring Redis up first because each shard PINGs it at startup (5s
  # timeout) and exits if it can't reach. FLUSHALL ensures every deploy
  # starts from a clean keyspace — otherwise a previous run's slot keys
  # would be visible to open-addressing and we'd see false "occupied"
  # for empty polynomial slots.
  local dname; dname="$(db_name)"
  wait_for_ssh "$dname"
  log "[$dname] installing + (re)starting redis on :$REDIS_PORT"
  # Wait for unattended-upgrades (which runs at boot on fresh Ubuntu
  # images) to release the apt lock. Without this, the install line
  # races with the daemon and fails with
  #   E: Could not get lock /var/lib/dpkg/lock-frontend.
  # 90 s is generous: cloud-init's unattended-upgrades pass typically
  # takes 30-60 s on a fresh n2-standard-2.
  # We hand apt itself a long lock-wait via DPkg::Lock::Timeout so
  # it blocks (instead of bailing) when cloud-init's
  # unattended-upgrades pass is still holding the dpkg lock. The
  # leading fuser poll is a fast-path; the timeout backstops it.
  remote "$dname" "set -e; \
    for i in \$(seq 1 90); do \
      if ! sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
         && ! sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1; then \
        break; \
      fi; \
      sleep 1; \
    done; \
    if ! dpkg -s redis-server >/dev/null 2>&1; then \
      sudo apt-get -o DPkg::Lock::Timeout=600 update -qq && \
      sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -qq redis-server; \
    fi; \
    sudo sed -i 's/^bind .*/bind 0.0.0.0/' /etc/redis/redis.conf; \
    sudo sed -i 's/^protected-mode .*/protected-mode no/' /etc/redis/redis.conf; \
    sudo systemctl restart redis-server; \
    sleep 1; \
    redis-cli -h 127.0.0.1 -p $REDIS_PORT FLUSHALL >/dev/null; \
    redis-cli -h 127.0.0.1 -p $REDIS_PORT PING"

  local db_ip; db_ip="$(db_internal_ip)"
  log "shards + coordinator will use redis://$db_ip:$REDIS_PORT"

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
      log "[$name] uploading aegon_shard_server"
      scp_to "$name" "$remote_bin_dir/aegon_shard_server"
      remote "$name" "sudo mkdir -p $REMOTE_BIN_DIR && \
        sudo mv /tmp/aegon_shard_server $REMOTE_BIN_DIR/ && \
        sudo chmod +x $REMOTE_BIN_DIR/aegon_shard_server"
      restart_shard "$name" "$i" "$per_shard" "$db_ip"
      log "[$name] deploy done"
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
  log "[$cname] uploading aegon_coordinator_bench + aegon_srs_bootstrap"
  scp_to "$cname" "$remote_bin_dir/aegon_coordinator_bench"
  scp_to "$cname" "$remote_bin_dir/aegon_srs_bootstrap"
  remote "$cname" "sudo mkdir -p $REMOTE_BIN_DIR && \
    sudo mv /tmp/aegon_coordinator_bench $REMOTE_BIN_DIR/ && \
    sudo mv /tmp/aegon_srs_bootstrap $REMOTE_BIN_DIR/ && \
    sudo chmod +x $REMOTE_BIN_DIR/aegon_coordinator_bench $REMOTE_BIN_DIR/aegon_srs_bootstrap"

  log "deploy done. shards are awaiting BootstrapSrs on :$SRS_PORT."
  log "next: ./scripts/bench-cluster.sh bootstrap"
  log "  (on cache hit, bootstrap completes in seconds; on cache miss it"
  log "  pushes trapdoors + waits for distributed gen + prefill across all shards.)"
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
cmd_restart_shards() {
  require_project
  require_power_of_two "$N_SHARDS"

  local per_shard; per_shard="$(prefill_per_shard)"
  log "shards will prefill ${per_shard} entries each (total = 2^$TOTAL_PRELOAD_LOG2)"

  local dname; dname="$(db_name)"
  log "[$dname] FLUSHALL"
  remote "$dname" "redis-cli -h 127.0.0.1 -p $REDIS_PORT FLUSHALL >/dev/null && \
                   redis-cli -h 127.0.0.1 -p $REDIS_PORT PING >/dev/null"

  local db_ip; db_ip="$(db_internal_ip)"

  local -a pids=()
  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    (
      restart_shard "$name" "$i" "$per_shard" "$db_ip"
      log "[$name] restart done"
    ) &
    pids+=("$!")
  done
  log "waiting on ${#pids[@]} per-shard restarts (parallel)..."
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
  done
  if (( failed > 0 )); then
    die "$failed shard restart(s) failed — inspect output above and run 'logs <i>' to debug"
  fi
  log "all shards restarted. wait for 'aegon_shard_server listening on ...' on each."
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
  require_project
  require_power_of_two "$N_SHARDS"

  # Pre-flight: verify every shard's SRS port is listening before we
  # invoke the bootstrap actor. The distributed-SRS protocol requires
  # all N_SHARDS to be reachable simultaneously (each shard peer-pulls
  # H_t slabs from every other shard); a single dead shard cascades
  # into all-shards-exit and a 30-min bootstrap timeout. We pay one
  # short probe pass up front and restart any dead shards before
  # touching the trapdoors.
  wait_for_shards_ready

  local cname; cname="$(coord_name)"
  local srs_csv; srs_csv="$(srs_endpoints_csv)"
  log "srs endpoints (first 2 shown): $(echo "$srs_csv" | cut -d, -f1-2),..."
  log "[$cname] running aegon_srs_bootstrap (seed=$SETUP_SEED)"
  remote "$cname" \
    "$REMOTE_BIN_DIR/aegon_srs_bootstrap \
       --shard-log-capacity $SHARD_LOG_CAPACITY \
       --kzh-k $KZH_K \
       --setup-seed $SETUP_SEED \
       --shard-endpoints $srs_csv" \
    stream
  log "bootstrap done. cluster ready for ./scripts/bench-cluster.sh bench"
}

# Probe every shard's SRS port from the coordinator (which can reach
# private IPs over the internal VPC). Restart any that aren't
# listening. Loops up to `max_passes` times so a flaky shard gets
# multiple chances. Aborts with `die` only if a shard fails to come
# up after all attempts — the right signal to surface a real fault
# (e.g. an OOM panic) rather than retrying forever.
wait_for_shards_ready() {
  local max_passes="${1:-3}"
  local pass
  local cname; cname="$(coord_name)"
  local per_shard; per_shard="$(prefill_per_shard)"
  local db_ip; db_ip="$(db_internal_ip)"
  for ((pass = 1; pass <= max_passes; pass++)); do
    log "[ready-barrier] pass $pass/$max_passes: probing all $N_SHARDS shards on :$SRS_PORT"
    # One single ssh-to-coordinator that runs a parallel probe of all shards.
    local dead_list
    dead_list="$(remote "$cname" "
      dead=''
      for i in \$(seq 0 $((N_SHARDS - 1))); do
        ip=\$(getent hosts ${SHARD_TAG}-\$i | awk '{print \$1}')
        if [ -z \"\$ip\" ] || ! timeout 2 bash -c \"</dev/tcp/\$ip/$SRS_PORT\" 2>/dev/null; then
          dead=\"\$dead \$i\"
        fi
      done
      echo \"DEAD:\$dead\"
    " 2>/dev/null | grep '^DEAD:' | sed 's/^DEAD://')"
    dead_list="$(echo "$dead_list" | xargs)"  # trim
    if [[ -z "$dead_list" ]]; then
      log "[ready-barrier] all $N_SHARDS shards listening on :$SRS_PORT ✓"
      return 0
    fi
    local dead_count; dead_count="$(echo "$dead_list" | wc -w)"
    log "[ready-barrier] $dead_count shard(s) not listening: $dead_list"
    log "[ready-barrier] restarting dead shards..."
    local -a restart_pids=()
    for i in $dead_list; do
      local name; name="$(shard_name "$i")"
      (
        restart_shard "$name" "$i" "$per_shard" "$db_ip"
      ) &
      restart_pids+=("$!")
    done
    local restart_failed=0
    for pid in "${restart_pids[@]}"; do
      wait "$pid" || restart_failed=$((restart_failed + 1))
    done
    if (( restart_failed > 0 )); then
      log "[ready-barrier] WARN: $restart_failed restart_shard call(s) returned non-zero; re-probing anyway"
    fi
  done
  die "[ready-barrier] shards still not ready after $max_passes passes"
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
cmd_setup_bench() {
  require_project
  require_power_of_two "$N_SHARDS"

  # Step 1: clear the SRS cache on every shard so we measure
  # distributed-gen wall-clock, not a cache-hit fast-path. The cache
  # dir is shell-expanded server-side, hence the literal $HOME below.
  log "wiping SRS cache on all shards ($SRS_CACHE_DIR)"
  local -a wipe_pids=()
  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    (
      remote "$name" "rm -f $SRS_CACHE_DIR/aegon-srs-*.cache 2>/dev/null || true"
    ) &
    wipe_pids+=("$!")
  done
  for pid in "${wipe_pids[@]}"; do wait "$pid"; done

  # Step 2: restart shards with prefill_count=0 so the setup-time
  # measurement is unpolluted by post-SRS prefill work.
  log "restarting shards with prefill_count=0 (pure setup-time measurement)"
  local db_ip; db_ip="$(db_internal_ip)"
  remote "$(db_name)" \
    "redis-cli -h 127.0.0.1 -p $REDIS_PORT FLUSHALL >/dev/null && \
     redis-cli -h 127.0.0.1 -p $REDIS_PORT PING >/dev/null"

  local -a restart_pids=()
  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    (
      restart_shard "$name" "$i" 0 "$db_ip"
    ) &
    restart_pids+=("$!")
  done
  local failed=0
  for pid in "${restart_pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
  done
  if (( failed > 0 )); then
    die "$failed shard restart(s) failed"
  fi

  # Step 3: run aegon_srs_bootstrap with --metrics-out so the
  # consolidated per-shard metrics land in one JSON file.
  local cname; cname="$(coord_name)"
  local srs_csv; srs_csv="$(srs_endpoints_csv)"
  log "[$cname] running aegon_srs_bootstrap with metrics gather"
  remote "$cname" \
    "$REMOTE_BIN_DIR/aegon_srs_bootstrap \
       --shard-log-capacity $SHARD_LOG_CAPACITY \
       --kzh-k $KZH_K \
       --setup-seed $SETUP_SEED \
       --shard-endpoints $srs_csv \
       --metrics-out $REMOTE_SETUP_BENCH_OUT" \
    stream

  log "retrieving $REMOTE_SETUP_BENCH_OUT -> $LOCAL_SETUP_BENCH_OUT"
  scp_from "$cname" "$REMOTE_SETUP_BENCH_OUT" "$LOCAL_SETUP_BENCH_OUT"
  log "setup-bench JSON saved to $LOCAL_SETUP_BENCH_OUT"
}

# Publish-time + commit-size benchmark for the large regime.
#
# Walks PUBLISH_FILL_PERCENTS (default 0,30,60,90), restarting shards
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
PUBLISH_FILL_PERCENTS="${PUBLISH_FILL_PERCENTS:-0,30,60,90}"
PUBLISH_BATCH_SIZES="${PUBLISH_BATCH_SIZES:-4096,8192,16384,32768,65536,131072}"
PUBLISH_SAMPLES_PER_BATCH="${PUBLISH_SAMPLES_PER_BATCH:-3}"
# True (non-over-provisioned) total log capacity. Default 32 (2^32
# entries). The per-shard polynomial sizes against SHARD_LOG_CAPACITY
# (default 29), which gives a 4x over-provisioning.
PUBLISH_TRUE_LOG_CAP="${PUBLISH_TRUE_LOG_CAP:-32}"
LOCAL_PUBLISH_BENCH_DIR="${LOCAL_PUBLISH_BENCH_DIR:-/tmp/aegon-publish-bench}"
cmd_publish_bench() {
  require_project
  require_power_of_two "$N_SHARDS"
  mkdir -p "$LOCAL_PUBLISH_BENCH_DIR"

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

  local db_ip; db_ip="$(db_internal_ip)"
  local shard_csv; shard_csv="$(shard_endpoints_csv)"
  local srs_csv; srs_csv="$(srs_endpoints_csv)"

  # Walk every fill percent. For each, compute per-shard prefill,
  # restart shards, bootstrap (cache hit -> fast), then run the
  # bench in distributed mode with --fill-percents <pct>.
  local total_capacity=$((1 << PUBLISH_TRUE_LOG_CAP))
  IFS=',' read -ra fill_pcts <<< "$PUBLISH_FILL_PERCENTS"
  for fill_pct in "${fill_pcts[@]}"; do
    log "==== publish-bench: fill_percent=$fill_pct ===="
    # python3 for the multiplication so the arithmetic survives at
    # PUBLISH_TRUE_LOG_CAP=32 (= ~4.3B, fits in u64 but easy to mis-
    # shift in bash).
    local per_shard
    per_shard="$(python3 -c "print((${total_capacity} * ${fill_pct}) // 100 // ${N_SHARDS})")"
    log "[$fill_pct%] per-shard prefill_count = $per_shard"

    # Stage 1: restart shards with the right prefill count.
    log "[$fill_pct%] restarting all shards with prefill_count=$per_shard"
    remote "$(db_name)" \
      "redis-cli -h 127.0.0.1 -p $REDIS_PORT FLUSHALL >/dev/null && \
       redis-cli -h 127.0.0.1 -p $REDIS_PORT PING >/dev/null"
    local -a restart_pids=()
    for ((i = 0; i < N_SHARDS; i++)); do
      local name; name="$(shard_name "$i")"
      (
        restart_shard "$name" "$i" "$per_shard" "$db_ip"
      ) &
      restart_pids+=("$!")
    done
    local failed=0
    for pid in "${restart_pids[@]}"; do
      wait "$pid" || failed=$((failed + 1))
    done
    if (( failed > 0 )); then
      die "$failed shard restart(s) failed at fill_pct=$fill_pct"
    fi

    # Stage 2: bootstrap. SRS cache should hit (assuming this isn't
    # the very first run), making this near-instant.
    log "[$fill_pct%] running bootstrap (cache hit expected)"
    remote "$cname" \
      "$REMOTE_BIN_DIR/aegon_srs_bootstrap \
         --shard-log-capacity $SHARD_LOG_CAPACITY \
         --kzh-k $KZH_K \
         --setup-seed $SETUP_SEED \
         --shard-endpoints $srs_csv" \
      stream

    # Stage 3: run the bench in distributed mode.
    local remote_out="/tmp/aegon-publish-bench-${fill_pct}.json"
    log "[$fill_pct%] running aegon_publish_bench (distributed)"
    remote "$cname" \
      "mkdir -p \$HOME/aegon-run && cd \$HOME/aegon-run && \
       $REMOTE_BIN_DIR/aegon_publish_bench \
         --shard-log-capacity $SHARD_LOG_CAPACITY \
         --true-log-capacity $PUBLISH_TRUE_LOG_CAP \
         --kzh-k $KZH_K \
         --n-shards $N_SHARDS \
         --endpoints $shard_csv \
         --fill-percents $fill_pct \
         --batch-sizes $PUBLISH_BATCH_SIZES \
         --samples-per-batch $PUBLISH_SAMPLES_PER_BATCH \
         --setup-seed $SETUP_SEED \
         --prefill-seed $PREFILL_SEED \
         --db-url redis://$db_ip:$REDIS_PORT \
         --out $remote_out" \
      stream

    local local_out="$LOCAL_PUBLISH_BENCH_DIR/${LOCAL_OUT_NAME_PREFIX:-}publish-${fill_pct}pct.json"
    log "[$fill_pct%] retrieving $remote_out -> $local_out"
    scp_from "$cname" "$remote_out" "$local_out"
    log "[$fill_pct%] done"
  done
  log "publish-bench complete. JSONs in $LOCAL_PUBLISH_BENCH_DIR/"
}

cmd_bench() {
  require_project
  require_power_of_two "$N_SHARDS"

  local cname; cname="$(coord_name)"
  local csv; csv="$(shard_endpoints_csv)"
  local db_ip; db_ip="$(db_internal_ip)"
  log "endpoints (first 2 shown): $(echo "$csv" | cut -d, -f1-2),..."
  log "db: redis://$db_ip:$REDIS_PORT"
  log "[$cname] running aegon_coordinator_bench"
  remote "$cname" \
    "mkdir -p \$HOME/aegon-run \$HOME/artifacts/srs && \
     cd \$HOME/aegon-run && \
     $REMOTE_BIN_DIR/aegon_coordinator_bench \
       --shard-log-capacity $SHARD_LOG_CAPACITY \
       --kzh-k $KZH_K \
       --setup-seed $SETUP_SEED \
       --endpoints $csv \
       --batch-sizes $BATCH_SIZES \
       --samples-per-batch $SAMPLES_PER_BATCH \
       --db-url redis://$db_ip:$REDIS_PORT \
       --output $REMOTE_BENCH_OUT" \
    stream
  log "retrieving $REMOTE_BENCH_OUT -> $LOCAL_BENCH_OUT"
  scp_from "$cname" "$REMOTE_BENCH_OUT" "$LOCAL_BENCH_OUT"
  log "bench JSON saved to $LOCAL_BENCH_OUT"
}

# Run aegon_lookup_bench on the coordinator. The lookup bench mirrors
# the publish-bench cluster flow: walks LOOKUP_FILL_PERCENTS (default
# 0,30,60,90), restarts shards with the per-shard anonymous prefill
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
LOOKUP_PUBLISH_BATCH_SIZE="${LOOKUP_PUBLISH_BATCH_SIZE:-1024}"
LOOKUP_AUDIT_SAMPLES="${LOOKUP_AUDIT_SAMPLES:-5}"
LOCAL_LOOKUP_BENCH_DIR="${LOCAL_LOOKUP_BENCH_DIR:-/tmp/aegon-lookup-bench}"
cmd_lookup_bench() {
  require_project
  require_power_of_two "$N_SHARDS"
  mkdir -p "$LOCAL_LOOKUP_BENCH_DIR"

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

  local cname; cname="$(coord_name)"
  local shard_csv; shard_csv="$(shard_endpoints_csv)"
  local srs_csv; srs_csv="$(srs_endpoints_csv)"
  local db_ip; db_ip="$(db_internal_ip)"
  log "[$cname] uploading aegon_lookup_bench"
  scp_to "$cname" "$remote_bin_dir/aegon_lookup_bench"
  remote "$cname" "sudo mkdir -p $REMOTE_BIN_DIR && \
    sudo mv /tmp/aegon_lookup_bench $REMOTE_BIN_DIR/ && \
    sudo chmod +x $REMOTE_BIN_DIR/aegon_lookup_bench"

  local total_capacity=$((1 << LOOKUP_TRUE_LOG_CAP))
  IFS=',' read -ra fill_pcts <<< "$LOOKUP_FILL_PERCENTS"
  for fill_pct in "${fill_pcts[@]}"; do
    log "==== lookup-bench: fill_percent=$fill_pct ===="
    local per_shard
    per_shard="$(python3 -c "print((${total_capacity} * ${fill_pct}) // 100 // ${N_SHARDS})")"
    log "[$fill_pct%] per-shard prefill_count = $per_shard"

    # Stage 1: restart shards with the right prefill count. Reuses
    # cmd_publish_bench's parallel-restart pattern.
    log "[$fill_pct%] restarting all shards with prefill_count=$per_shard"
    remote "$(db_name)" \
      "redis-cli -h 127.0.0.1 -p $REDIS_PORT FLUSHALL >/dev/null && \
       redis-cli -h 127.0.0.1 -p $REDIS_PORT PING >/dev/null"
    local -a restart_pids=()
    for ((i = 0; i < N_SHARDS; i++)); do
      local name; name="$(shard_name "$i")"
      (
        restart_shard "$name" "$i" "$per_shard" "$db_ip"
      ) &
      restart_pids+=("$!")
    done
    local failed=0
    for pid in "${restart_pids[@]}"; do
      wait "$pid" || failed=$((failed + 1))
    done
    if (( failed > 0 )); then
      die "$failed shard restart(s) failed at fill_pct=$fill_pct"
    fi

    # Stage 2: bootstrap. SRS cache should hit on re-runs.
    log "[$fill_pct%] running bootstrap (cache hit expected)"
    remote "$cname" \
      "$REMOTE_BIN_DIR/aegon_srs_bootstrap \
         --shard-log-capacity $SHARD_LOG_CAPACITY \
         --kzh-k $KZH_K \
         --setup-seed $SETUP_SEED \
         --shard-endpoints $srs_csv" \
      stream

    # Stage 3: run the lookup bench. --preload-counts is the
    # sampleable-namespace size *added on top* of the anonymous
    # per-shard prefill the cluster already has; --publish-batch-
    # sizes runs the per-stage publish bench using a disjoint
    # namespace.
    local remote_out="/tmp/aegon-lookup-bench-${fill_pct}.json"
    log "[$fill_pct%] running aegon_lookup_bench (preload=$LOOKUP_PRELOAD_COUNT, samples=$LOOKUP_SAMPLES_PER_LEVEL)"
    remote "$cname" \
      "mkdir -p \$HOME/aegon-run && \
       cd \$HOME/aegon-run && \
       $REMOTE_BIN_DIR/aegon_lookup_bench \
         --shard-log-capacity $SHARD_LOG_CAPACITY \
         --kzh-k $KZH_K \
         --n-shards $N_SHARDS \
         --setup-seed $SETUP_SEED \
         --endpoints $shard_csv \
         --db-url redis://$db_ip:$REDIS_PORT \
         --preload-counts $LOOKUP_PRELOAD_COUNT \
         --samples-per-level $LOOKUP_SAMPLES_PER_LEVEL \
         --publish-batch-size $LOOKUP_PUBLISH_BATCH_SIZE \
         --publish-batch-sizes $LOOKUP_PUBLISH_BATCH_SIZES \
         --publish-samples-per-batch $LOOKUP_PUBLISH_SAMPLES_PER_BATCH \
         --audit-samples $LOOKUP_AUDIT_SAMPLES \
         --output $remote_out" \
      stream

    local local_out="$LOCAL_LOOKUP_BENCH_DIR/${LOCAL_OUT_NAME_PREFIX:-}lookup-${fill_pct}pct.json"
    log "[$fill_pct%] retrieving $remote_out -> $local_out"
    scp_from "$cname" "$remote_out" "$local_out"
    log "[$fill_pct%] done"
  done
  log "lookup-bench complete. JSONs in $LOCAL_LOOKUP_BENCH_DIR/"
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
  local dname; dname="$(db_name)"
  ( probe_one "$dname" "redis-server"   > "$tmpdir/db"    ; echo $? > "$tmpdir/db.rc"    ) &
  pids+=("$!")

  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  # Replay outputs in deterministic order (shard-0 .. shard-N, coord, db)
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
  cat "$tmpdir/db"
  rc="$(cat "$tmpdir/db.rc" 2>/dev/null || echo 1)"
  (( rc != 0 )) && failed=$((failed+1))
  rm -rf "$tmpdir"

  if (( failed > 0 )); then
    log "watchdog: $failed host(s) BAD"
    return 1
  fi
  log "watchdog: all $((N_SHARDS + 2)) hosts OK"
  return 0
}

cmd_down() {
  require_project
  log "tearing down bench cluster"

  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    if gcloud compute instances describe "$name" --zone="$ZONE" >/dev/null 2>&1; then
      log "deleting $name"
      gcloud compute instances delete "$name" --zone="$ZONE" --quiet >/dev/null
    fi
  done
  local cname; cname="$(coord_name)"
  if gcloud compute instances describe "$cname" --zone="$ZONE" >/dev/null 2>&1; then
    log "deleting $cname"
    gcloud compute instances delete "$cname" --zone="$ZONE" --quiet >/dev/null
  fi
  local dname; dname="$(db_name)"
  if gcloud compute instances describe "$dname" --zone="$ZONE" >/dev/null 2>&1; then
    log "deleting $dname"
    gcloud compute instances delete "$dname" --zone="$ZONE" --quiet >/dev/null
  fi

  for fw in "$FIREWALL_GRPC" "$FIREWALL_SRS" "$FIREWALL_REDIS" "$FIREWALL_SSH"; do
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
                   Walks PUBLISH_FILL_PERCENTS (default 0,30,60,90),
                   restarting shards per stage with the per-shard
                   prefill count for that fill level. For each stage,
                   sweeps PUBLISH_BATCH_SIZES (default
                   4096..131072) and emits one JSON per fill level
                   under LOCAL_PUBLISH_BENCH_DIR (default
                   /tmp/aegon-publish-bench). Bootstrap between
                   stages should hit cache and be near-instant.
  restart-shards   Restart all shards with a fresh --prefill-count using
                   already-uploaded binaries + on-disk SRS cache + FLUSHALL.
                   Cheap per-call (~30 s) — use in (prefill,batch) sweeps.
                   Cache hit makes the bootstrap step a no-op too, so
                   follow with `bootstrap` then `bench`.
  bench            Run aegon_coordinator_bench on the coordinator, fetch JSON
  lookup-bench     Large-regime combined lookup + publish bench.
                   Walks LOOKUP_FILL_PERCENTS (default mirrors
                   PUBLISH_FILL_PERCENTS = 0,30,60,90), restarting
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
    restart-shards) cmd_restart_shards ;;
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
