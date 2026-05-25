#!/usr/bin/env bash
# Standalone watcher for medium #3. The wrapper run-medium-cluster.sh
# died locally at ~14:04 UTC for unclear reasons, but the bench unit
# itself is still running on the coord (linger working as intended).
# This script polls the coord every 5 minutes and when the unit goes
# inactive, scp's the JSON output back to local. No teardown — let the
# user decide.

set -uo pipefail

PROJECT=jbonneau-mwalfish-9a0c
ZONE=us-central1-f
CNAME=aegon-bench-coord
REMOTE_JSON=/tmp/aegon-lookup-bench.json
REMOTE_LOG=/tmp/aegon-lookup-bench.log
LOCAL_LOG=/home/alrshir/aegon/bench-results/lookup/medium-lookup-bench.log
LOCAL_JSON=/home/alrshir/aegon/bench-results/lookup/medium-combined.json
WATCHER_LOG=/home/alrshir/aegon/bench-results/_watcher.log

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

log() {
  echo "[$(ts)] $*" | tee -a "$WATCHER_LOG"
}

last_bytes=0

while true; do
  # 1. Check unit state
  state=$(timeout 30 gcloud compute ssh "$CNAME" --tunnel-through-iap \
    --zone="$ZONE" --project="$PROJECT" --quiet \
    --command="sudo systemctl is-active aegon-lookup-bench 2>&1" 2>/dev/null \
    | tr -d '[:space:]')

  # 2. Tail new log bytes since last poll
  total_bytes=$(timeout 30 gcloud compute ssh "$CNAME" --tunnel-through-iap \
    --zone="$ZONE" --project="$PROJECT" --quiet \
    --command="wc -c < $REMOTE_LOG 2>/dev/null || echo 0" 2>/dev/null \
    | tr -d '[:space:]')
  total_bytes="${total_bytes:-0}"
  if [[ "$total_bytes" =~ ^[0-9]+$ ]] && (( total_bytes > last_bytes )); then
    timeout 60 gcloud compute ssh "$CNAME" --tunnel-through-iap \
      --zone="$ZONE" --project="$PROJECT" --quiet \
      --command="tail -c +$((last_bytes + 1)) $REMOTE_LOG 2>/dev/null" 2>/dev/null \
      >> "$LOCAL_LOG"
    last_bytes="$total_bytes"
  fi

  # 3. Always fetch JSON snapshot — preserves partial data if anything kills the bench mid-run
  timeout 30 gcloud compute scp --tunnel-through-iap \
    --zone="$ZONE" --project="$PROJECT" --quiet \
    "$CNAME:$REMOTE_JSON" "$LOCAL_JSON" 2>/dev/null || true

  log "state=$state bytes=$total_bytes (local json=$(wc -c < "$LOCAL_JSON" 2>/dev/null || echo 0))"

  if [[ "$state" != "active" && "$state" != "activating" ]]; then
    log "bench unit no longer active (state=$state) — final fetch + exit"
    timeout 60 gcloud compute scp --tunnel-through-iap \
      --zone="$ZONE" --project="$PROJECT" --quiet \
      "$CNAME:$REMOTE_JSON" "$LOCAL_JSON" 2>&1 | tee -a "$WATCHER_LOG"
    log "watcher exiting; cluster NOT torn down — run 'PROJECT=$PROJECT scripts/bench-cluster.sh down' to clean up"
    exit 0
  fi

  sleep 300
done
