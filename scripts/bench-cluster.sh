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
#   TOTAL_PRELOAD_LOG2  log_2 of total prefilled users (28 → 2^28 split across shards)
#   BATCH_SIZES         comma-separated sweep sizes   (10,100,1000,10000)
#   SAMPLES_PER_BATCH   timed publishes per batch     (5)
#   SETUP_SEED          deterministic SRS gen seed    (42)
#   PREFILL_SEED        deterministic prefill seed    (1)
#   SHARD_MACHINE_TYPE  GCE machine type for shards   (n2-highmem-16 — 128 GB RAM)
#   COORD_MACHINE_TYPE  GCE machine type coordinator  (n2-standard-4)
#
# At the defaults above (N_SHARDS=32, SHARD_LOG_CAPACITY=29, KZH_K=10):
#   * Each shard machine needs ~64 GB RAM. n2-highmem-16 (128 GB) leaves
#     headroom; n2-highmem-8 (64 GB) is right at the edge and may OOM
#     during the in-process SRS gen pass.
#   * 32 × n2-highmem-16 ≈ $38/hr. Run `down` aggressively.
#   * No Redis: this bench measures publish-only and the open-addressing
#     trail goes via gRPC `is_index_slot_occupied`, not Redis EXISTS.
#
# This script is a prototyping aid, not production infrastructure.

set -euo pipefail

PROJECT="${PROJECT:-}"
ZONE="${ZONE:-us-central1-f}"
N_SHARDS="${N_SHARDS:-32}"
SHARD_LOG_CAPACITY="${SHARD_LOG_CAPACITY:-29}"
KZH_K="${KZH_K:-10}"
TOTAL_PRELOAD_LOG2="${TOTAL_PRELOAD_LOG2:-28}"
BATCH_SIZES="${BATCH_SIZES:-10,100,1000,10000}"
SAMPLES_PER_BATCH="${SAMPLES_PER_BATCH:-5}"
SETUP_SEED="${SETUP_SEED:-42}"
PREFILL_SEED="${PREFILL_SEED:-1}"
SHARD_MACHINE_TYPE="${SHARD_MACHINE_TYPE:-n2-highmem-16}"
COORD_MACHINE_TYPE="${COORD_MACHINE_TYPE:-n2-standard-4}"

NETWORK="aegon-bench-vpc"
FIREWALL_GRPC="aegon-bench-grpc"
FIREWALL_SSH="aegon-bench-ssh"
SHARD_TAG="aegon-bench-shard"
COORD_TAG="aegon-bench-coord"
SHARD_PORT=50051
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
      --image-family="ubuntu-2204-lts" --image-project="ubuntu-os-cloud" \
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
      --image-family="ubuntu-2204-lts" --image-project="ubuntu-os-cloud" \
      --boot-disk-size=20GB >/dev/null
  fi

  log "instances up. waiting 30s for SSH to settle..."
  sleep 30
  log "ready. next: ./scripts/bench-cluster.sh deploy"
}

cmd_deploy() {
  require_project
  require_power_of_two "$N_SHARDS"

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
        cargo build --release -p akd --target x86_64-unknown-linux-gnu \
          --bin aegon_shard_server --bin aegon_coordinator_bench"
    remote_bin_dir="$REPO_ROOT/target/x86_64-unknown-linux-gnu/release"
  else
    log "building release binaries (aegon_shard_server, aegon_coordinator_bench)"
    (cd "$REPO_ROOT" && cargo build --release -p akd \
      --bin aegon_shard_server --bin aegon_coordinator_bench) >/dev/null
  fi
  [[ -x "$remote_bin_dir/aegon_shard_server" ]]      || die "aegon_shard_server missing"
  [[ -x "$remote_bin_dir/aegon_coordinator_bench" ]] || die "aegon_coordinator_bench missing"

  local per_shard; per_shard="$(prefill_per_shard)"
  log "shards will prefill ${per_shard} entries each (total = 2^$TOTAL_PRELOAD_LOG2)"

  # ---- push to every shard + start ----
  # Each shard generates its own SRS in-process from --setup-seed
  # (no file transfer of a 30 GB SRS), then prefills with
  # --prefill-count entries (deterministic, seeded by PREFILL_SEED + i
  # so different shards fill different slot patterns).
  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    log "[$name] uploading aegon_shard_server"
    scp_to "$name" "$remote_bin_dir/aegon_shard_server"
    remote "$name" "sudo mkdir -p $REMOTE_BIN_DIR && \
      sudo mv /tmp/aegon_shard_server $REMOTE_BIN_DIR/ && \
      sudo chmod +x $REMOTE_BIN_DIR/aegon_shard_server"
    local shard_prefill_seed=$((PREFILL_SEED + i))
    log "[$name] starting shard server (shard_id=$i, prefill_count=$per_shard, prefill_seed=$shard_prefill_seed)"
    # PID-file restart pattern (see cluster.sh for the rationale on
    # avoiding `pkill -f`).
    remote "$name" "if [ -f /tmp/aegon-shard.pid ]; then \
        kill \$(cat /tmp/aegon-shard.pid) 2>/dev/null || true; \
        sleep 1; \
      fi; \
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
        > /tmp/aegon-shard.log 2>&1 < /dev/null & \
      echo \$! > /tmp/aegon-shard.pid"
  done

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

cmd_bench() {
  require_project
  require_power_of_two "$N_SHARDS"

  local cname; cname="$(coord_name)"
  local csv; csv="$(shard_endpoints_csv)"
  log "endpoints (first 2 shown): $(echo "$csv" | cut -d, -f1-2),..."
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
       --output $REMOTE_BENCH_OUT" \
    stream
  log "retrieving $REMOTE_BENCH_OUT -> $LOCAL_BENCH_OUT"
  scp_from "$cname" "$REMOTE_BENCH_OUT" "$LOCAL_BENCH_OUT"
  log "bench JSON saved to $LOCAL_BENCH_OUT"
}

cmd_logs() {
  require_project
  local idx="${1:-0}"
  local name; name="$(shard_name "$idx")"
  log "[$name] tail /tmp/aegon-shard.log"
  remote "$name" "tail -f /tmp/aegon-shard.log" stream
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

  for fw in "$FIREWALL_GRPC" "$FIREWALL_SSH"; do
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

  up        Provision VPC, firewall, $N_SHARDS shards + coordinator
  deploy    Build binaries, push to every node, start shard servers
            (each does in-process SRS gen + --prefill-count locally)
  bench     Run aegon_coordinator_bench on the coordinator, fetch JSON
  logs N    Tail the shard log on $SHARD_TAG-N
  down      Delete every instance + the VPC

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
    up)     cmd_up ;;
    deploy) cmd_deploy ;;
    bench)  cmd_bench ;;
    logs)   cmd_logs "$@" ;;
    down)   cmd_down ;;
    ""|-h|--help|help) usage ;;
    *) usage; die "unknown subcommand: $sub" ;;
  esac
}

main "$@"
