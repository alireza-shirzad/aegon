# scripts/

Cluster-bringup helpers. The main entry point is `cluster.sh`, which
spins up an N-shard Aegon cluster on GCE, deploys the binaries, runs
the smoke test, and tears it all down on request.

## Prerequisites

- `gcloud` CLI installed and authenticated:
  ```bash
  gcloud auth login
  ```
- A GCP project you have permission to create instances in.
- The repository builds locally: `cargo build --release -p akd`.
- The remote VMs are `x86_64` Ubuntu 22.04, so the binaries shipped
  to them must be `x86_64-unknown-linux-gnu`.
  - **Linux x86_64 hosts**: nothing extra — `cargo` already targets
    the right triple.
  - **macOS hosts (Intel or Apple Silicon)**: install
    [`cross`](https://github.com/cross-rs/cross) and have Docker
    running. The deploy step auto-detects macOS and switches to
    `cross` for the two remote-bound binaries.
    ```bash
    cargo install cross
    # plus: Docker Desktop running
    ```
    `aegon_srs_gen` is still built natively (it runs on your laptop,
    not on the VMs). `Cross.toml` at the repo root tells the cross
    container to install `protoc` for the tonic build.

## Quick start

```bash
export PROJECT=your-gcp-project-id

# Provision VPC + 4 shard machines + 1 coordinator.
./scripts/cluster.sh up

# Build binaries locally, generate SRS, ship everything, start servers.
./scripts/cluster.sh deploy

# Run the coordinator smoke client against the live cluster.
./scripts/cluster.sh smoke

# Watch shard 0's log.
./scripts/cluster.sh logs 0

# Delete every instance + the VPC.
./scripts/cluster.sh down
```

Total wall-clock time on a healthy connection: `up` ≈ 1–2 min,
`deploy` ≈ 1 min, `smoke` ≈ 10 s. So end-to-end from zero to a
verified cluster smoke test is well under 5 minutes.

## Tuning

All knobs are environment variables with defaults; only `PROJECT` is
required.

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `PROJECT` | _none_ | GCP project ID |
| `ZONE` | `us-central1-f` | GCE zone (all nodes co-located) |
| `N_SHARDS` | `4` | Number of shard machines (must be a power of 2) |
| `SHARD_LOG_CAPACITY` | `20` | `log_2` slots per shard |
| `KZH_K` | `6` | KZH-k block parameter (set to `optimal_kzh_k(SHARD_LOG_CAPACITY)`) |
| `SHARD_MACHINE_TYPE` | `n2-standard-4` | GCE machine type for shards |
| `COORD_MACHINE_TYPE` | `e2-small` | GCE machine type for coordinator |

### Picking shard size

The defaults (`SHARD_LOG_CAPACITY=20, KZH_K=6`) are tuned for cheap
validation — SRS gen is sub-second, RSS is well under 1 GB, every
machine class fits. Once the cluster wiring is proven, bump to your
target deployment:

```bash
# 32-shard production scale (4B users × 4× slack = 2^34 slots)
N_SHARDS=32 SHARD_LOG_CAPACITY=29 KZH_K=10 \
  SHARD_MACHINE_TYPE=n2-standard-16 \
  ./scripts/cluster.sh up
```

`KZH_K` should follow `optimal_kzh_k(SHARD_LOG_CAPACITY)` from
`akd::aegon::presets`:

```
shard_log_capacity:  20  21..23  24..26  27..28  29..31  32..34  35+
kzh_k:                6     7       8       9       10      11     12
```

## Cost estimate

For the default 4-shard cluster (`n2-standard-4` × 4 + `e2-small`):
roughly **$0.82/hr**. A typical validation session of 2–3 hours
costs under $3. A 32-shard production-scale cluster
(`n2-standard-16` × 32 + `e2-small`): roughly **$25/hr**, so don't
forget to `./scripts/cluster.sh down` when you're done.

## Caveats

- This is a **prototyping** script. Instances are created without
  external IPs; SSH/scp goes through IAP tunneling (firewall only
  accepts Google's `35.235.240.0/20` IAP range). The operator running
  this script needs `roles/iap.tunnelResourceAccessor` on the project
  (project owners/editors already have it).
- The SRS is generated locally via `aegon_srs_gen --seed 42`. The
  underlying call is `gen_srs_for_testing`, **not** a trusted setup.
  For real production, replace `aegon_srs_gen` with a binary that
  loads a ceremony output and re-run `deploy`.
- Shard servers are started with `nohup ... &`; they survive SSH
  disconnect but won't auto-restart on crash. Add a systemd unit
  (see the main README) before depending on the cluster.
- The script doesn't yet wire up TLS. Coordinator ↔ shard traffic is
  plaintext HTTP/2 inside the VPC. Adequate for inside a trusted
  network; not adequate for anything else.
