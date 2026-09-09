#!/usr/bin/env bash
# run-small-on-remote.sh — execute the small-regime bench suite on
# *separate* GCE VMs (one bench, one masking server) so the masking
# work doesn't share CPU with the bench. Mirrors the medium-cluster
# topology: bench VM + masking VM + intra-subnet firewall rule.
#
# Outputs land under bench-results/_remote-small/{setup,publish,lookup}/
# so they don't clobber the local-machine small results.
#
# Env (all optional):
#   PROJECT, ZONE, MACHINE_TYPE, IMAGE_FAMILY, IMAGE_PROJECT
#   NETWORK, SUBNET                — project-specific networking
#   BENCH_VM, MASKING_VM           — VM names (defaults: aegon-small-bench, aegon-small-masking)
#   SKIP_TEARDOWN=1                — keep both VMs after the run (debugging)

set -euo pipefail

PROJECT="${PROJECT:-jbonneau-mwalfish-9a0c}"
ZONE="${ZONE:-us-central1-f}"
BENCH_VM="${BENCH_VM:-${VM_NAME:-aegon-small-bench}}"
MASKING_VM="${MASKING_VM:-aegon-small-masking}"
MACHINE_TYPE="${MACHINE_TYPE:-n2-standard-16}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2604-lts-amd64}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
NETWORK="${NETWORK:-jbonneau-mwalfish-net}"
SUBNET="${SUBNET:-jbonneau-mwalfish-subnet-01}"
FIREWALL_NAME="${FIREWALL_NAME:-aegon-small-masking-fw}"

# Small-regime parameters — keep in sync with
# scripts/{setup,publish,lookup}-bench.sh defaults.
SHARD_LOG_CAPACITY="${SHARD_LOG_CAPACITY:-22}"
TRUE_LOG_CAPACITY="${TRUE_LOG_CAPACITY:-20}"
# optimal_kzh_k(22) = 7 — hardcoded so the masking server (which needs
# an explicit --kzh-k) stays in sync with what the bench would pick.
KZH_K="${KZH_K:-7}"
PUBLISH_BATCH_SIZES="${PUBLISH_BATCH_SIZES:-2,4,8,16,32,64}"
FILL_PERCENTS="${FILL_PERCENTS:-1,30,60,90}"
LOOKUP_SAMPLES="${LOOKUP_SAMPLES:-20}"
PUBLISH_SAMPLES="${PUBLISH_SAMPLES:-3}"
AUDIT_SAMPLES="${AUDIT_SAMPLES:-5}"
SETUP_SEED="${SETUP_SEED:-42}"
PREFILL_SEED="${PREFILL_SEED:-1}"
MASKING_PORT="${MASKING_PORT:-50061}"
MASKING_QUEUE="${MASKING_QUEUE:-16}"
MASKING_PRODUCERS="${MASKING_PRODUCERS:-2}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_OUT="$REPO_ROOT/bench-results/_remote-small"
REMOTE_BIN="/tmp/aegon-bin"
REMOTE_OUT="/tmp/aegon-bench-out"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

ssh_on() {
  local vm="$1"; shift
  gcloud compute ssh "$vm" \
    --project="$PROJECT" --zone="$ZONE" \
    --tunnel-through-iap --quiet \
    --command="$1"
}

scp_to_vm() {
  # usage: scp_to_vm VM_NAME LOCAL_PATH [LOCAL_PATH ...] REMOTE_DEST
  local vm="$1"; shift
  local args=("$@")
  local last_idx=$((${#args[@]} - 1))
  local remote_dest="${args[$last_idx]}"
  unset 'args[last_idx]'
  gcloud compute scp --recurse \
    --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap --quiet \
    "${args[@]}" "$vm:$remote_dest"
}

delete_vm() {
  local vm="$1"
  log "deleting $vm"
  gcloud compute instances delete "$vm" \
    --project="$PROJECT" --zone="$ZONE" --quiet >/dev/null 2>&1 || true
}

cleanup() {
  if [[ "${SKIP_TEARDOWN:-0}" == "1" ]]; then
    log "SKIP_TEARDOWN=1: leaving $BENCH_VM and $MASKING_VM in place"
    return 0
  fi
  delete_vm "$BENCH_VM"
  delete_vm "$MASKING_VM"
  log "deleting firewall $FIREWALL_NAME"
  gcloud compute firewall-rules delete "$FIREWALL_NAME" \
    --project="$PROJECT" --quiet >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --- 0. build the bench binaries locally -------------------------------
log "building release binaries on dev box"
(cd "$REPO_ROOT" && cargo build --release -p akd \
   --bin aegon_setup_bench \
   --bin aegon_publish_bench \
   --bin aegon_lookup_bench \
   --bin aegon_masking_server) >/dev/null
for b in aegon_setup_bench aegon_publish_bench aegon_lookup_bench aegon_masking_server; do
  [[ -x "$REPO_ROOT/target/release/$b" ]] || { echo "binary missing: $b" >&2; exit 1; }
done

# --- 1. create both VMs (idempotent) -----------------------------------
for vm in "$BENCH_VM" "$MASKING_VM"; do
  if gcloud compute instances describe "$vm" \
     --project="$PROJECT" --zone="$ZONE" --format='value(name)' >/dev/null 2>&1; then
    log "stale $vm found — deleting before recreating"
    gcloud compute instances delete "$vm" \
      --project="$PROJECT" --zone="$ZONE" --quiet >/dev/null 2>&1 || true
  fi
done
for vm in "$BENCH_VM" "$MASKING_VM"; do
  log "creating $vm ($MACHINE_TYPE, $ZONE)"
  gcloud compute instances create "$vm" \
    --project="$PROJECT" --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=100GB \
    --network="$NETWORK" --subnet="$SUBNET" \
    --no-address \
    --quiet >/dev/null
done

# --- 1a. firewall: bench -> masking on $MASKING_PORT -------------------
# Use a permissive intra-subnet rule (source-ranges = the subnet CIDR)
# rather than a tag-based rule, matching how bench-cluster.sh exposes
# the masking endpoint. Idempotent: delete-if-exists, then create.
log "ensuring firewall $FIREWALL_NAME for tcp:$MASKING_PORT"
gcloud compute firewall-rules delete "$FIREWALL_NAME" \
  --project="$PROJECT" --quiet >/dev/null 2>&1 || true
SUBNET_CIDR="$(gcloud compute networks subnets describe "$SUBNET" \
  --project="$PROJECT" --region="${ZONE%-*}" \
  --format='value(ipCidrRange)')"
gcloud compute firewall-rules create "$FIREWALL_NAME" \
  --project="$PROJECT" --network="$NETWORK" \
  --direction=INGRESS \
  --source-ranges="$SUBNET_CIDR" \
  --allow="tcp:$MASKING_PORT" \
  --quiet >/dev/null

# --- 2. wait for SSH on both VMs ---------------------------------------
for vm in "$BENCH_VM" "$MASKING_VM"; do
  log "waiting for SSH on $vm"
  for i in $(seq 1 30); do
    if ssh_on "$vm" "echo ok" >/dev/null 2>&1; then
      log "SSH on $vm ready after $i attempt(s)"
      break
    fi
    sleep 5
  done
done

# --- 3. push binaries to both VMs --------------------------------------
# The bench binaries cache the SRS at `cwd/../artifacts/srs/...` (see
# akd_core/src/aegon_crypto/pcs/kzhk/mod.rs:139). They have to run from
# a subdirectory so the `..` resolves to a writable path. We use
# $HOME/work as the cwd; the SRS will land in $HOME/artifacts/srs.
REMOTE_RUN_DIR="\$HOME/work"

log "pushing binaries to $BENCH_VM:$REMOTE_BIN"
ssh_on "$BENCH_VM" "mkdir -p $REMOTE_BIN $REMOTE_OUT/setup $REMOTE_OUT/publish $REMOTE_OUT/lookup $REMOTE_RUN_DIR \$HOME/artifacts/srs"
scp_to_vm "$BENCH_VM" \
  "$REPO_ROOT/target/release/aegon_setup_bench" \
  "$REPO_ROOT/target/release/aegon_publish_bench" \
  "$REPO_ROOT/target/release/aegon_lookup_bench" \
  "$REMOTE_BIN/"

log "pushing masking server binary to $MASKING_VM:$REMOTE_BIN"
ssh_on "$MASKING_VM" "mkdir -p $REMOTE_BIN $REMOTE_RUN_DIR \$HOME/artifacts/srs"
scp_to_vm "$MASKING_VM" \
  "$REPO_ROOT/target/release/aegon_masking_server" \
  "$REMOTE_BIN/"

# --- 4. start masking server on $MASKING_VM (bind 0.0.0.0) -------------
# Use the fire-and-forget pattern from bench-cluster.sh: IAP SSH won't
# release its channel while a backgrounded child holds inherited fds,
# so launch async and watchdog-kill after 120s. The bind check below
# confirms the server actually came up.
log "[$MASKING_VM] launching aegon_masking_server :$MASKING_PORT (queue=$MASKING_QUEUE producers=$MASKING_PRODUCERS)"
gcloud compute ssh "$MASKING_VM" \
  --project="$PROJECT" --zone="$ZONE" \
  --tunnel-through-iap --quiet \
  --command="
    cd $REMOTE_RUN_DIR && \
    setsid nohup $REMOTE_BIN/aegon_masking_server \
      --bind 0.0.0.0:$MASKING_PORT \
      --num-vars $SHARD_LOG_CAPACITY \
      --kzh-k $KZH_K \
      --setup-seed $SETUP_SEED \
      --queue-size $MASKING_QUEUE \
      --producers $MASKING_PRODUCERS \
      > /tmp/aegon-masking.log 2>&1 < /dev/null &
    echo \$! > /tmp/aegon-masking.pid
    disown 2>/dev/null || true
    echo SPAWNED
  " &
_launch_pid=$!
( sleep 120 && kill "$_launch_pid" 2>/dev/null ) &
_watchdog_pid=$!
wait "$_launch_pid" 2>/dev/null || true
kill "$_watchdog_pid" 2>/dev/null || true

log "[$MASKING_VM] waiting for masking server to bind :$MASKING_PORT"
for _ in $(seq 1 60); do
  if ssh_on "$MASKING_VM" "ss -tln | grep -q ':$MASKING_PORT '" 2>/dev/null; then
    log "[$MASKING_VM] masking server bound"
    break
  fi
  sleep 2
done

# Resolve internal IP of the masking VM so the bench can dial it.
MASKING_IP="$(gcloud compute instances describe "$MASKING_VM" \
  --project="$PROJECT" --zone="$ZONE" \
  --format='value(networkInterfaces[0].networkIP)')"
MASKING_ENDPOINT="http://$MASKING_IP:$MASKING_PORT"
log "masking endpoint resolved: $MASKING_ENDPOINT"

# --- 5. run setup-bench on $BENCH_VM -----------------------------------
log "[$BENCH_VM] aegon_setup_bench (shard_log_capacity=$SHARD_LOG_CAPACITY, auto kzh_k)"
ssh_on "$BENCH_VM" "cd $REMOTE_RUN_DIR && $REMOTE_BIN/aegon_setup_bench \
  --shard-log-capacity $SHARD_LOG_CAPACITY \
  --setup-seed $SETUP_SEED \
  --label small \
  --out $REMOTE_OUT/setup/small.json"

# --- 6. run publish-bench on $BENCH_VM ---------------------------------
log "[$BENCH_VM] aegon_publish_bench (fills=$FILL_PERCENTS, batches=$PUBLISH_BATCH_SIZES)"
ssh_on "$BENCH_VM" "cd $REMOTE_RUN_DIR && $REMOTE_BIN/aegon_publish_bench \
  --shard-log-capacity $SHARD_LOG_CAPACITY \
  --true-log-capacity $TRUE_LOG_CAPACITY \
  --n-shards 1 \
  --fill-percents $FILL_PERCENTS \
  --batch-sizes $PUBLISH_BATCH_SIZES \
  --samples-per-batch $PUBLISH_SAMPLES \
  --setup-seed $SETUP_SEED \
  --prefill-seed $PREFILL_SEED \
  --private \
  --out $REMOTE_OUT/publish/small.json"

# --- 7. run lookup-bench on $BENCH_VM, once per fill percent -----------
IFS=',' read -ra _fill_pcts <<< "$FILL_PERCENTS"
true_cap=$(( 1 << TRUE_LOG_CAPACITY ))
for pct in "${_fill_pcts[@]}"; do
  # initial_prefill_count = (true_cap * pct/100) - LOOKUP_NS_SIZE; floor at 0.
  # LOOKUP_NS_SIZE is the small-regime default 1000 (matches lookup-bench.sh).
  prefill=$(( (true_cap * pct) / 100 - 1000 ))
  (( prefill < 0 )) && prefill=0
  log "[$BENCH_VM] aegon_lookup_bench fill=${pct}% (initial_prefill=$prefill, masking=$MASKING_ENDPOINT)"
  ssh_on "$BENCH_VM" "cd $REMOTE_RUN_DIR && $REMOTE_BIN/aegon_lookup_bench \
    --shard-log-capacity $SHARD_LOG_CAPACITY \
    --true-log-capacity $TRUE_LOG_CAPACITY \
    --kzh-k $KZH_K \
    --n-shards 1 \
    --fill-percents $pct \
    --initial-prefill-count $prefill \
    --prefill-seed $PREFILL_SEED \
    --setup-seed $SETUP_SEED \
    --samples-per-level $LOOKUP_SAMPLES \
    --publish-batch-sizes $PUBLISH_BATCH_SIZES \
    --publish-samples-per-batch $PUBLISH_SAMPLES \
    --audit-samples $AUDIT_SAMPLES \
    --private \
    --masking-addr $MASKING_ENDPOINT \
    --output $REMOTE_OUT/lookup/small_fill${pct}.json"
done

# --- 8. fetch JSONs from $BENCH_VM -------------------------------------
log "fetching JSONs to $LOCAL_OUT"
mkdir -p "$LOCAL_OUT/setup" "$LOCAL_OUT/publish" "$LOCAL_OUT/lookup"
scp_to_vm_back() {
  gcloud compute scp --recurse \
    --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap --quiet \
    "$BENCH_VM:$1" "$2"
}
scp_to_vm_back "$REMOTE_OUT/setup/small.json"   "$LOCAL_OUT/setup/"
scp_to_vm_back "$REMOTE_OUT/publish/small.json" "$LOCAL_OUT/publish/"
for pct in "${_fill_pcts[@]}"; do
  scp_to_vm_back "$REMOTE_OUT/lookup/small_fill${pct}.json" "$LOCAL_OUT/lookup/"
done

log "done. JSONs:"
find "$LOCAL_OUT" -name '*.json' -printf '  %p (%s bytes)\n' | sort
