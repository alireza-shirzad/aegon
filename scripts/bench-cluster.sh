#!/usr/bin/env bash
# bench-cluster.sh — spin up a production-scale Aegon shard cluster and
# run aegon_coordinator_bench against it.
#
# Subcommands:
#   up        Provision VPC + firewall + N high-memory shard VMs + coordinator VM
#   deploy    Build binaries locally (in a Docker linux/amd64 container on
#             macOS), scp aegon_shard_server to every shard VM, scp
#             aegon_coordinator_bench to the coordinator. Each shard
#             server generates its SRS in-process from --setup-seed and
#             then prefills its polynomials with --prefill-count entries.
#   bench     Run aegon_coordinator_bench on the coordinator, retrieve
#             /tmp/aegon-bench.json from it
#   logs N    Tail the shard log on aegon-shard-N
#   down      Delete every instance + the VPC this script created
#
# Required env (or defaults):
#   PROJECT             GCP project ID                (no default; required)
#   ZONE                GCE zone                      (us-central1-f)
#   N_SHARDS            number of shards (power of 2) (32)
#   SHARD_LOG_CAPACITY  log_2 slots per shard         (29)
#   KZH_K               KZH-k block parameter         (10 — optimal_kzh_k(29))
#   TOTAL_PRELOAD_LOG2  log_2 of total prefilled users (8 → 2^8=256 split across shards)
#   BATCH_SIZES         comma-separated sweep sizes   (2,4,8,...,16384)
#   SAMPLES_PER_BATCH   timed publishes per batch     (1)
#   SETUP_SEED          deterministic SRS gen seed    (42)
#   PREFILL_SEED        deterministic prefill seed    (1)
#   SHARD_MACHINE_TYPE  GCE machine type for shards   (n2-standard-16 — 64 GB RAM)
#   COORD_MACHINE_TYPE  GCE machine type coordinator  (n2-standard-4)
#
# At the defaults above (N_SHARDS=32, SHARD_LOG_CAPACITY=29, KZH_K=10):
#   * Each shard's KZH-k SRS at log_cap=29 / kzh_k=10 is dominated by
#     H_1: 2^29 entries × 64 B = 32 GiB. Plus H_2..H_10 (smaller),
#     working set, and KZH aux state during the in-process SRS gen
#     pass, peak memory lands around ~40-45 GiB.
#   * n2-standard-16 (64 GB RAM) has empirically been enough for this;
#     n2-highmem-16 (128 GB) is overkill but the safe choice if
#     you've changed the per-shard log_cap upward or are stacking
#     multiple shards per machine.
#   * 32 × n2-standard-16 ≈ $25/hr. Run `down` aggressively.
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
N_SHARDS="${N_SHARDS:-32}"
SHARD_LOG_CAPACITY="${SHARD_LOG_CAPACITY:-29}"
KZH_K="${KZH_K:-10}"
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
FIREWALL_SSH="aegon-bench-ssh"
FIREWALL_REDIS="aegon-bench-redis"
SHARD_TAG="aegon-bench-shard"
COORD_TAG="aegon-bench-coord"
DB_TAG="aegon-bench-db"
SHARD_PORT=50051
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
    gcloud compute ssh "$instance" --zone="$ZONE" --tunnel-through-iap \
      --quiet --command="$cmd" &
    local _ssh_pid=$!
    ( sleep 30 && kill "$_ssh_pid" 2>/dev/null ) &
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
  remote "$name" "if [ -f /tmp/aegon-shard.pid ]; then \
      kill \$(cat /tmp/aegon-shard.pid) 2>/dev/null || true; \
    fi; \
    pkill -x aegon_shard_ser 2>/dev/null || true; \
    sleep 2; \
    mkdir -p \$HOME/aegon-run \$HOME/artifacts/srs && \
    cd \$HOME/aegon-run && \
    nohup $REMOTE_BIN_DIR/aegon_shard_server \
      --bind 0.0.0.0:$SHARD_PORT \
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
    exit 0" \
    fire-and-forget
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
          --bin aegon_shard_server --bin aegon_coordinator_bench"
    remote_bin_dir="$REPO_ROOT/target/x86_64-unknown-linux-gnu/release"
  else
    log "building release binaries (aegon_shard_server, aegon_coordinator_bench)"
    (cd "$REPO_ROOT" && cargo build --release -p akd $cargo_features \
      --bin aegon_shard_server --bin aegon_coordinator_bench) >/dev/null
  fi
  [[ -x "$remote_bin_dir/aegon_shard_server" ]]      || die "aegon_shard_server missing"
  [[ -x "$remote_bin_dir/aegon_coordinator_bench" ]] || die "aegon_coordinator_bench missing"

  local per_shard; per_shard="$(prefill_per_shard)"
  log "shards will prefill ${per_shard} entries each (total = 2^$TOTAL_PRELOAD_LOG2)"

  # ---- DB tier: install (idempotent) + FLUSHALL ----
  # Bring Redis up first because each shard PINGs it at startup (5s
  # timeout) and exits if it can't reach. FLUSHALL ensures every deploy
  # starts from a clean keyspace — otherwise a previous run's slot keys
  # would be visible to open-addressing and we'd see false "occupied"
  # for empty polynomial slots.
  local dname; dname="$(db_name)"
  log "[$dname] installing + (re)starting redis on :$REDIS_PORT"
  # Wait for unattended-upgrades (which runs at boot on fresh Ubuntu
  # images) to release the apt lock. Without this, the install line
  # races with the daemon and fails with
  #   E: Could not get lock /var/lib/dpkg/lock-frontend.
  # 90 s is generous: cloud-init's unattended-upgrades pass typically
  # takes 30-60 s on a fresh n2-standard-2.
  remote "$dname" "set -e; \
    for i in \$(seq 1 90); do \
      if ! sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
         && ! sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1; then \
        break; \
      fi; \
      sleep 1; \
    done; \
    if ! dpkg -s redis-server >/dev/null 2>&1; then \
      sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq redis-server; \
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
  log "[$cname] uploading aegon_coordinator_bench"
  scp_to "$cname" "$remote_bin_dir/aegon_coordinator_bench"
  remote "$cname" "sudo mkdir -p $REMOTE_BIN_DIR && \
    sudo mv /tmp/aegon_coordinator_bench $REMOTE_BIN_DIR/ && \
    sudo chmod +x $REMOTE_BIN_DIR/aegon_coordinator_bench"

  log "deploy started. shards are doing SRS gen + prefill in parallel."
  log "at log_cap=$SHARD_LOG_CAPACITY this takes a while — check progress with"
  log "  ./scripts/bench-cluster.sh logs 0"
  log "wait for 'aegon_shard_server listening on ...' on every shard before running bench."
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

# Run aegon_lookup_bench on the coordinator. The lookup bench owns its
# own publish cycle (publishes PRELOAD_COUNT labels then samples
# lookup_label/value/history for both server-direct and client paths),
# so it expects to be run AFTER deploy + (optionally) AFTER the publish
# bench but NOT mixed with one.
#
# Why scp the binary fresh each time: the deploy step only pushes
# aegon_coordinator_bench, not aegon_lookup_bench. If you're running
# the lookup bench after a deploy, the binary isn't on the coord yet.
LOOKUP_PRELOAD_COUNT="${LOOKUP_PRELOAD_COUNT:-256}"
LOOKUP_SAMPLES_PER_LEVEL="${LOOKUP_SAMPLES_PER_LEVEL:-20}"
REMOTE_LOOKUP_OUT="/tmp/aegon-lookup-bench.json"
LOCAL_LOOKUP_OUT="${LOCAL_LOOKUP_OUT:-/tmp/aegon-lookup-bench.json}"
cmd_lookup_bench() {
  require_project
  require_power_of_two "$N_SHARDS"

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
  local csv; csv="$(shard_endpoints_csv)"
  local db_ip; db_ip="$(db_internal_ip)"
  log "[$cname] uploading aegon_lookup_bench"
  scp_to "$cname" "$remote_bin_dir/aegon_lookup_bench"
  remote "$cname" "sudo mkdir -p $REMOTE_BIN_DIR && \
    sudo mv /tmp/aegon_lookup_bench $REMOTE_BIN_DIR/ && \
    sudo chmod +x $REMOTE_BIN_DIR/aegon_lookup_bench"

  log "[$cname] running aegon_lookup_bench (preload=$LOOKUP_PRELOAD_COUNT, samples=$LOOKUP_SAMPLES_PER_LEVEL)"
  remote "$cname" \
    "mkdir -p \$HOME/aegon-run && \
     cd \$HOME/aegon-run && \
     $REMOTE_BIN_DIR/aegon_lookup_bench \
       --shard-log-capacity $SHARD_LOG_CAPACITY \
       --kzh-k $KZH_K \
       --setup-seed $SETUP_SEED \
       --endpoints $csv \
       --db-url redis://$db_ip:$REDIS_PORT \
       --preload-counts $LOOKUP_PRELOAD_COUNT \
       --samples-per-level $LOOKUP_SAMPLES_PER_LEVEL \
       --output $REMOTE_LOOKUP_OUT" \
    stream
  log "retrieving $REMOTE_LOOKUP_OUT -> $LOCAL_LOOKUP_OUT"
  scp_from "$cname" "$REMOTE_LOOKUP_OUT" "$LOCAL_LOOKUP_OUT"
  log "lookup bench JSON saved to $LOCAL_LOOKUP_OUT"
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

  for fw in "$FIREWALL_GRPC" "$FIREWALL_REDIS" "$FIREWALL_SSH"; do
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
                   (each does in-process SRS gen + --prefill-count locally)
  restart-shards   Restart all shards with a fresh --prefill-count using
                   already-uploaded binaries + on-disk SRS cache + FLUSHALL.
                   Cheap per-call (~30 s) — use in (prefill,batch) sweeps.
  bench            Run aegon_coordinator_bench on the coordinator, fetch JSON
  lookup-bench     Run aegon_lookup_bench on the coordinator (publishes
                   LOOKUP_PRELOAD_COUNT labels then samples
                   server/client lookup_label, lookup_value,
                   lookup_history), fetch JSON
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

At defaults (32 shards × n2-highmem-16): ~\$38/hr. Tear down promptly.
EOF
}

main() {
  local sub="${1:-}"
  shift || true
  case "$sub" in
    up)             cmd_up ;;
    deploy)         cmd_deploy ;;
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
