#!/usr/bin/env bash
# cluster.sh — spin up / tear down / smoke-test an Aegon cluster on GCE.
#
# Subcommands:
#   up        Provision VPC + firewall + shard machines + coordinator
#   deploy    Build binaries locally, scp them + the SRS to every machine,
#             and start the shard servers
#   smoke     Run aegon_coordinator_smoke against the live cluster
#   logs N    Tail the shard log on aegon-shard-N
#   down      Delete every instance + the VPC this script created
#
# Required env (or defaults):
#   PROJECT             GCP project ID                (no default; required)
#   ZONE                GCE zone                      (us-central1-f)
#   N_SHARDS            number of shards (power of 2) (4)
#   SHARD_LOG_CAPACITY  log2 slots per shard          (20)
#   KZH_K               KZH-k block parameter         (6 — optimal_kzh_k(20))
#   SHARD_MACHINE_TYPE  GCE machine type for shards   (n2-standard-4)
#   COORD_MACHINE_TYPE  GCE machine type coordinator  (e2-small)
#
# Prereqs:
#   * gcloud installed + authenticated (`gcloud auth login`)
#   * gcloud configured for $PROJECT   (`gcloud config set project $PROJECT`)
#   * the repository cargo-builds cleanly with `cargo build --release -p akd`
#
# This script is a prototyping aid, not production infrastructure. It
# creates resources tagged `aegon-cluster` so the teardown can find them.

set -euo pipefail

PROJECT="${PROJECT:-}"
ZONE="${ZONE:-us-central1-f}"
N_SHARDS="${N_SHARDS:-4}"
SHARD_LOG_CAPACITY="${SHARD_LOG_CAPACITY:-20}"
KZH_K="${KZH_K:-6}"
SHARD_MACHINE_TYPE="${SHARD_MACHINE_TYPE:-n2-standard-4}"
COORD_MACHINE_TYPE="${COORD_MACHINE_TYPE:-e2-small}"

NETWORK="aegon-vpc"
FIREWALL_GRPC="aegon-grpc"
FIREWALL_SSH="aegon-ssh"
SHARD_TAG="aegon-shard"
COORD_TAG="aegon-coordinator"
SHARD_PORT=50051

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_SRS="/tmp/aegon-cluster.srs"
REMOTE_SRS_PATH="/etc/aegon/shard.srs"
REMOTE_BIN_DIR="/opt/aegon/bin"

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

# Run a remote command on an instance. Quiet by default; pass a third arg to
# preserve the SSH stream (useful for interactive logs).
#
# --tunnel-through-iap routes SSH via Identity-Aware Proxy because the
# instances have no external IP. This requires the operator to have
# roles/iap.tunnelResourceAccessor on the project (project owners/editors
# already do).
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

shard_name() { echo "${SHARD_TAG}-$1"; }
coord_name() { echo "${COORD_TAG}"; }

shard_internal_ip() {
  gcloud compute instances describe "$(shard_name "$1")" --zone="$ZONE" \
    --format='value(networkInterfaces[0].networkIP)'
}

# Build the endpoint list the coordinator passes via --endpoints.
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

  log "PROJECT=$PROJECT ZONE=$ZONE N_SHARDS=$N_SHARDS"
  log "shard machine type=$SHARD_MACHINE_TYPE, coordinator type=$COORD_MACHINE_TYPE"

  # ---- VPC ----
  if gcloud compute networks describe "$NETWORK" >/dev/null 2>&1; then
    log "VPC $NETWORK already exists, skipping create"
  else
    log "creating VPC $NETWORK"
    gcloud compute networks create "$NETWORK" --subnet-mode=auto >/dev/null
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
  # The instances have no external IP (some projects enforce
  # constraints/compute.vmExternalIpAccess). We rely on IAP tunneling
  # for SSH, which connects through 35.235.240.0/20 — that's Google's
  # published IAP source range, the only addresses that need port 22.
  if gcloud compute firewall-rules describe "$FIREWALL_SSH" >/dev/null 2>&1; then
    log "firewall $FIREWALL_SSH exists"
  else
    log "creating firewall $FIREWALL_SSH (IAP -> instances:22)"
    gcloud compute firewall-rules create "$FIREWALL_SSH" \
      --network="$NETWORK" \
      --allow="tcp:22" \
      --source-ranges="35.235.240.0/20" >/dev/null
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
      --boot-disk-size=50GB >/dev/null
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

  log "instances up. waiting 20s for SSH to settle..."
  sleep 20
  log "ready. run: ./scripts/cluster.sh deploy"
}

cmd_deploy() {
  require_project
  require_power_of_two "$N_SHARDS"

  # On macOS hosts the remote-bound binaries must be x86_64-unknown-linux-gnu.
  # We run cargo inside a Linux Docker container (no cross-compile needed —
  # the build host inside the container is itself x86_64 Linux), which avoids
  # the `cross` × `rustup` toolchain-install incompatibility on recent
  # versions. `aegon_srs_gen` is always built for the native host because it
  # runs on the operator's machine, not on the VMs.
  local local_bin_dir="$REPO_ROOT/target/release"
  local remote_bin_dir="$local_bin_dir"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    command -v docker >/dev/null || die "macOS host needs Docker (we build the Linux binaries inside a container)"
    docker info >/dev/null 2>&1  || die "Docker daemon unreachable — start Docker Desktop and retry"

    log "macOS host: building aegon_srs_gen natively for the operator's laptop"
    (cd "$REPO_ROOT" && cargo build --release -p akd --bin aegon_srs_gen) >/dev/null

    log "building shard + coordinator binaries inside Docker (first run pulls rust image — ~5 min on Apple Silicon)"
    # --platform linux/amd64 forces x86_64 even on Apple Silicon (target VMs
    # are x86_64). --target keeps the in-container build artefacts separate
    # from the host's target/release/ so the native aegon_srs_gen binary
    # isn't clobbered. apt-get installs protoc for the tonic build.
    docker run --rm --platform linux/amd64 \
      -v "$REPO_ROOT:/workspace" -w /workspace \
      rust:slim-bookworm \
      bash -c "set -e; \
        apt-get update >/dev/null && \
        apt-get install -y --no-install-recommends protobuf-compiler ca-certificates >/dev/null && \
        cargo build --release -p akd --target x86_64-unknown-linux-gnu \
          --bin aegon_shard_server --bin aegon_coordinator_smoke"
    remote_bin_dir="$REPO_ROOT/target/x86_64-unknown-linux-gnu/release"
  else
    log "building release binaries"
    (cd "$REPO_ROOT" && cargo build --release -p akd \
      --bin aegon_srs_gen --bin aegon_shard_server --bin aegon_coordinator_smoke) \
      >/dev/null
  fi

  [[ -x "$local_bin_dir/aegon_srs_gen" ]]               || die "aegon_srs_gen missing"
  [[ -x "$remote_bin_dir/aegon_shard_server" ]]         || die "aegon_shard_server missing"
  [[ -x "$remote_bin_dir/aegon_coordinator_smoke" ]]    || die "aegon_coordinator_smoke missing"

  # ---- generate SRS locally ----
  log "generating SRS (shard_log_capacity=$SHARD_LOG_CAPACITY, kzh_k=$KZH_K)"
  rm -f "$LOCAL_SRS"
  "$local_bin_dir/aegon_srs_gen" \
    --shard-log-capacity "$SHARD_LOG_CAPACITY" --kzh-k "$KZH_K" \
    --seed 42 --out "$LOCAL_SRS"

  # ---- push everything to every shard ----
  for ((i = 0; i < N_SHARDS; i++)); do
    local name; name="$(shard_name "$i")"
    log "[$name] uploading SRS + binary"
    scp_to "$name" "$LOCAL_SRS" "$remote_bin_dir/aegon_shard_server"
    remote "$name" "sudo mkdir -p /etc/aegon $REMOTE_BIN_DIR && \
      sudo mv /tmp/aegon-cluster.srs $REMOTE_SRS_PATH && \
      sudo mv /tmp/aegon_shard_server $REMOTE_BIN_DIR/ && \
      sudo chmod +x $REMOTE_BIN_DIR/aegon_shard_server"
    log "[$name] starting shard server"
    # nohup + & + < /dev/null + 2>&1 so SSH session closes cleanly.
    remote "$name" "pkill -f aegon_shard_server >/dev/null 2>&1; sleep 1; \
      nohup $REMOTE_BIN_DIR/aegon_shard_server \
        --bind 0.0.0.0:$SHARD_PORT \
        --shard-log-capacity $SHARD_LOG_CAPACITY \
        --kzh-k $KZH_K \
        --srs-path $REMOTE_SRS_PATH \
        > /tmp/aegon-shard.log 2>&1 < /dev/null &"
  done

  # ---- push to coordinator ----
  local cname; cname="$(coord_name)"
  log "[$cname] uploading SRS + smoke client"
  scp_to "$cname" "$LOCAL_SRS" "$remote_bin_dir/aegon_coordinator_smoke"
  remote "$cname" "sudo mkdir -p /etc/aegon $REMOTE_BIN_DIR && \
    sudo mv /tmp/aegon-cluster.srs $REMOTE_SRS_PATH && \
    sudo mv /tmp/aegon_coordinator_smoke $REMOTE_BIN_DIR/ && \
    sudo chmod +x $REMOTE_BIN_DIR/aegon_coordinator_smoke"

  log "deploy complete. waiting 5s for shards to bind..."
  sleep 5
  log "ready. run: ./scripts/cluster.sh smoke"
}

cmd_smoke() {
  require_project
  require_power_of_two "$N_SHARDS"

  local cname; cname="$(coord_name)"
  local csv; csv="$(shard_endpoints_csv)"
  log "endpoints: $csv"
  log "[$cname] running coordinator smoke client"
  remote "$cname" \
    "$REMOTE_BIN_DIR/aegon_coordinator_smoke \
       --shard-log-capacity $SHARD_LOG_CAPACITY \
       --kzh-k $KZH_K \
       --srs-path $REMOTE_SRS_PATH \
       --endpoints $csv \
       --n-users 256" \
    stream
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
  log "tearing down cluster"

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
  deploy    Build binaries, generate SRS, push to every node, start shards
  smoke     Run aegon_coordinator_smoke against the live cluster
  logs N    Tail the shard log on aegon-shard-N
  down      Delete every instance + the VPC

Required env: PROJECT=<gcp-project-id>
Optional env: ZONE, N_SHARDS, SHARD_LOG_CAPACITY, KZH_K,
              SHARD_MACHINE_TYPE, COORD_MACHINE_TYPE
EOF
}

main() {
  local sub="${1:-}"
  shift || true
  case "$sub" in
    up)     cmd_up ;;
    deploy) cmd_deploy ;;
    smoke)  cmd_smoke ;;
    logs)   cmd_logs "$@" ;;
    down)   cmd_down ;;
    ""|-h|--help|help) usage ;;
    *) usage; die "unknown subcommand: $sub" ;;
  esac
}

main "$@"
