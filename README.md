# Aegon

Sharded, polynomial-commitment-backed auditable key directory. This crate
preserves the public AKD API surface (publish / lookup / consistency / audit)
but replaces the original SEEMless+Merkle backend with the **Aegon** engine
built on KZH-k polynomial commitments, sharded across N machines and
coordinated over gRPC.

This is a fork of [facebook/akd](https://github.com/facebook/akd); the
top-level wire types (`AkdLabel`, `AkdValue`, `LookupProof`, etc.) are
intentionally unchanged, so downstream callers can migrate without
rewrites.

---

## What it does

- **Publish.** Coordinator batches `(label, value)` pairs, routes them to
  shards via VRF-style open addressing across the full hypercube,
  computes one shared Fiat-Shamir scalar over all shards' new
  commitments, and asks every shard to finalize the epoch.
- **Lookup.** Coordinator walks the cross-shard probe trail for a
  label, asks each shard for its PCS opening at the relevant slot, and
  bundles those openings with Merkle paths anchoring them under the
  epoch's per-shard commitment Merkle root.
- **Consistency proof.** Same shape as lookup but on the per-shard
  `rand_index` / `rand_value` snapshots — proves a user's slot didn't
  change between two epochs, with legitimate rejection (`Ok(false)`)
  when the user did update their value.
- **Audit.** Auditor verifies the published Merkle root, re-derives the
  shared FS scalar, and checks each shard's chain witness at one
  FS-random evaluation point. Constant-cost regardless of batch size.

All three proofs verify against a single `ShardedEpochCommitment`
(epoch + Merkle root over per-shard commits).

---

## Quick start (single process, in-memory shards)

```rust
use akd::aegon::{ShardedAegon, ShardedAegonConfig, Sha256Hash, verify_sharded_lookup};
use ark_bn254::Bn254;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;

let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
    .shard_log_capacity(8)        // 256 slots per shard
    .log_n_shards(2)              // 4 shards
    .private(false)
    .kzh_k(2)
    .build()?;

let mut rng = ChaCha20Rng::seed_from_u64(0);
let mut server = ShardedAegon::<Bn254, Pcs, Sha256Hash>::setup(&mut rng, &cfg)?;

let (commit, _audit_proof) = server.publish(&[
    (b"alice".to_vec(), b"alice-v1".to_vec()),
    (b"bob".to_vec(), b"bob-v1".to_vec()),
])?;

let proof = server.lookup(&b"alice".to_vec())?;
let ctx = server.sharded_verifier_context();
let ok = verify_sharded_lookup::<Bn254, Pcs, Sha256Hash>(
    &ctx, &commit, &b"alice".to_vec(), &b"alice-v1".to_vec(), &proof,
)?;
assert!(ok);
```

For the integration through the legacy AKD `Directory` API (with
`AkdLabel`, `AkdValue`, etc.), see `akd::Directory` and
`akd::aegon_facade::verify_lookup`.

---

## Cluster deployment (N machines + 1 coordinator)

The system splits cleanly along four roles:

1. **Setup machine** — runs once, produces the SRS file.
2. **Shard machines** (×N) — each runs `aegon_shard_server` over gRPC.
3. **Coordinator machine** — runs the calling application, holds a
   `ShardedAegon` configured with `ShardTransport::Remote { endpoints }`.
4. **DB machine** — runs Redis. The coordinator writes the raw
   `(label, value)` bytes here on publish and reads them back on
   lookup so it can return the value alongside the proof. The
   polynomial commitments only bind hashes of `(label, value)`; the
   DB is a side-channel for retrieval, and the verifier re-hashes
   the bytes itself.

### Automated: one script

For prototyping on GCE, [`scripts/cluster.sh`](scripts/cluster.sh) does
the whole flow:

```bash
export PROJECT=your-gcp-project-id

./scripts/cluster.sh up      # VPC + 4 shards + 1 coordinator + 1 db
./scripts/cluster.sh deploy  # build, generate SRS, ship, start servers, install redis
./scripts/cluster.sh smoke   # publish + lookup + verify (queries redis)
./scripts/cluster.sh down    # delete everything
```

End-to-end runtime is under five minutes for the default 4-shard
config. See [`scripts/README.md`](scripts/README.md) for tuning knobs
(`N_SHARDS`, `SHARD_LOG_CAPACITY`, machine types).

The rest of this section walks through the same flow manually if you
want to understand or customize each step.

### Picking parameters

For a target user count `N_users`, pick:

```
log_capacity = ceil(log2(N_users * 4))   # 4× slack → ≤25% load factor
log_n_shards = pick_so_that_shard_log_capacity_fits_one_machine
shard_log_capacity = log_capacity - log_n_shards
kzh_k             = optimal_kzh_k(shard_log_capacity)
```

The `optimal_kzh_k` function (in `akd::aegon::presets`) minimizes the
aux-precomputation cost `f(k) = k(k − 1) · 2^(N/k)` and is tabulated
for `N ∈ [20, 35]`. For production at `shard_log_capacity = 29`, it
returns `k = 10`. Validated on a 16 vCPU / 64 GB box: setup ≈ 12 min,
peak RSS ≈ 37 GB (with `gen_srs_for_testing`; load-from-file is much
faster).

Example: 4B users × 4× slack ≈ 2³⁴ slots. Split as 2⁵ shards × 2²⁹
slots each → 32 machines, each at `shard_log_capacity = 29, kzh_k = 10`.

### Step 1: Generate the SRS, once

Run this on **one** machine (any machine with enough RAM to do an SRS
gen at `shard_log_capacity`). The output is one file you'll copy to
every shard + the coordinator.

```bash
cargo build --release -p akd --bin aegon_srs_gen
./target/release/aegon_srs_gen \
  --shard-log-capacity 29 \
  --kzh-k 10 \
  --seed 42 \
  --out /tmp/shard.srs
```

> ⚠️ `aegon_srs_gen` calls `gen_srs_for_testing` under the hood,
> which is **not** a trusted setup. For real deployment, replace
> this binary with one that loads a ceremony output. The on-wire
> shape (file containing
> `serialize_compressed(prover_param) || serialize_compressed(verifier_param)`)
> stays the same — only the source of the SRS changes.

Distribute the file to every shard machine + the coordinator:

```bash
gsutil cp /tmp/shard.srs gs://your-aegon-srs/shard.srs
# on each shard + coordinator:
sudo mkdir -p /etc/aegon && gsutil cp gs://your-aegon-srs/shard.srs /etc/aegon/
```

### Step 2: Run a shard server on every shard machine

Build the binary:

```bash
cargo build --release -p akd --bin aegon_shard_server
```

Run one per machine:

```bash
./target/release/aegon_shard_server \
  --bind 0.0.0.0:50051 \
  --shard-log-capacity 29 \
  --kzh-k 10 \
  --srs-path /etc/aegon/shard.srs
```

Optional flags:

| flag | purpose |
|---|---|
| `--private` | enable zk mode (`KZH-k zk=true`) |
| `--tls-cert <pem>` + `--tls-key <pem>` | enable TLS (the coordinator must point at the matching CA, see below) |
| `--setup-seed <u64>` | **test only:** generate the SRS in-process instead of loading from disk |

For systemd, a minimal unit looks like:

```ini
# /etc/systemd/system/aegon-shard.service
[Unit]
Description=Aegon Shard
After=network.target

[Service]
ExecStart=/opt/aegon/bin/aegon_shard_server \
  --bind 0.0.0.0:50051 \
  --shard-log-capacity 29 \
  --kzh-k 10 \
  --srs-path /etc/aegon/shard.srs
Restart=on-failure
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### Step 3: Wire the coordinator up

```rust
use akd::aegon::{
    DbSource, Sha256Hash, ShardTransport, ShardedAegon, ShardedAegonConfig, SrsSource,
};
use ark_bn254::Bn254;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

let cfg = ShardedAegonConfig::<Bn254, KZHK<Bn254>>::builder()
    .shard_log_capacity(29)
    .log_n_shards(5)
    .kzh_k(10)
    .shards(ShardTransport::Remote {
        endpoints: (0..32)
            .map(|i| format!("http://aegon-shard-{i}.internal:50051"))
            .collect(),
    })
    // Coordinator still needs the verifier_param to build its
    // VerifierContext; the prover_param is large and lives on the
    // shard machines.
    .srs(SrsSource::Path("/etc/aegon/shard.srs".into()))
    // Coordinator-side label→value KV store. Omit (or pass
    // DbSource::None) for single-process tests; for cluster
    // deployments, point at the Redis VM.
    .db(DbSource::Redis("redis://aegon-db.internal:6379".into()))
    .build()?;

let mut rng = ChaCha20Rng::seed_from_u64(0);
let mut server = ShardedAegon::<Bn254, KZHK<Bn254>, Sha256Hash>::setup(&mut rng, &cfg)?;
```

`setup` here is the only point where the coordinator talks to all 32
shards — it does the gRPC handshake and reads each shard's initial
commitment. After that, every `publish` / `lookup` /
`consistency_proof` call does whatever subset of gRPC calls the
protocol requires.

### Step 4: Smoke-test the cluster

After every shard is up, run the smoke client on the coordinator
machine:

```bash
cargo build --release -p akd --bin aegon_coordinator_smoke
./target/release/aegon_coordinator_smoke \
  --shard-log-capacity 29 --kzh-k 10 \
  --srs-path /etc/aegon/shard.srs \
  --endpoints http://aegon-shard-0:50051,http://aegon-shard-1:50051,...,http://aegon-shard-31:50051 \
  --db-url redis://aegon-db.internal:6379 \
  --n-users 1024
```

The binary brings up a `ShardedAegon` against the live cluster,
publishes `--n-users` deterministic users, looks each one up, and
verifies the proof. When `--db-url` is passed, every lookup
additionally cross-checks that the value Redis returns matches what
was published. The binary prints setup/publish/lookup wall-clock
timings plus a pass/fail summary. Exit code 0 = healthy cluster.

`--db-url` is optional: omitting it falls back to the single-process
mode where the smoke client trusts the values it just published and
the coordinator's `lookup` returns an empty value vector.

### TLS

Server side:

```bash
aegon_shard_server \
  --bind 0.0.0.0:50051 \
  --shard-log-capacity 29 --kzh-k 10 --srs-path /etc/aegon/shard.srs \
  --tls-cert /etc/aegon/server.crt \
  --tls-key  /etc/aegon/server.key
```

Coordinator side — use `GrpcShardClientConfig` directly instead of the
`.shards(...)` builder shortcut:

```rust
use akd::aegon::shard_grpc::{GrpcShardClientConfig, GrpcShardClient};

let ca_pem = std::fs::read("/etc/aegon/ca.crt")?;
let cfg = GrpcShardClientConfig::new(
    "https://aegon-shard-7.internal:50051".into(),
    verifier_context,
    29,
)
.with_tls_ca(ca_pem)
.with_tls_domain("aegon-shard-7.internal");
let client = GrpcShardClient::<Bn254, KZHK<Bn254>>::connect_with(cfg)?;
```

If you want the same TLS config applied to all 32 shards via
`ShardTransport::Remote`, today you build the clients manually and
swap them into `ShardedAegon` — the `.shards(ShardTransport::Remote {
endpoints })` shortcut does plaintext only. A future iteration will
add a TLS-aware variant.

### Retry policy

Reads (`open_*`, `is_index_slot_occupied`, `current_commitment`)
retry on transport-level failures (server `Unavailable`/`Unknown`)
using exponential backoff. Writes (`publish_phase_1` /
`publish_phase_2`) do **not** auto-retry — those aren't idempotent
and need protocol-level coordination (out of scope for v1).

Default policy is `max_attempts = 5, initial_backoff = 50ms,
max_backoff = 2s`. Override via `GrpcShardClientConfig::with_retry`.

---

## Architecture summary

```
                  ┌─────────────────┐         ┌──────────┐
                  │  Coordinator    │ ──TCP─▶ │  Redis   │
                  │ (your app +     │         │ (label → │
                  │  ShardedAegon)  │ ◀──TCP─ │  value)  │
                  └────────┬────────┘         └──────────┘
                           │ gRPC × N (publish / lookup / consistency / audit)
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐       ┌────▼─────┐       ┌────▼─────┐
   │ shard 0  │       │ shard 1  │  ...  │ shard N-1│
   │ Aegon    │       │ Aegon    │       │ Aegon    │
   │  - SRS   │       │  - SRS   │       │  - SRS   │
   │  - polys │       │  - polys │       │  - polys │
   └──────────┘       └──────────┘       └──────────┘
```

- Each shard owns four polynomials (`index`, `value`, `rand_index`,
  `rand_value`) and the matching commitments + KZH-k auxiliary state.
  The polynomials are over `H_F(label)` / `H_F(value)`; the raw bytes
  never touch a shard machine.
- The coordinator owns the routing table (label → cross-shard probe
  trail), the FS chain scalars, the running Merkle root, and the
  Redis client. On publish it writes `(label, value)` to Redis; on
  lookup it reads the value back so it can return it to the caller
  alongside the proof.
- The verifier never trusts what Redis returns — it re-hashes the
  bytes and checks the proof binds to that hash. Redis is a
  retrieval side-channel, not part of the soundness argument.
- Open addressing trails are derived from `H(ctr, label) → (shard_id,
  slot)`, deterministic from the public hash + config. Sub-millisecond
  lookups in the common case (`ctr0 = 0`, one shard, one PCS opening +
  one Merkle path).

For more depth see the per-module docs:

- [`akd::aegon::sharded`](akd/src/aegon/sharded.rs) — coordinator + sharded proof types
- [`akd::aegon::shard_grpc`](akd/src/aegon/shard_grpc.rs) — `ShardHandle` trait, server/client adapters
- [`akd::aegon::server`](akd/src/aegon/server.rs) — single-shard `Aegon`
- [`akd::aegon::verify`](akd/src/aegon/verify.rs), [`audit`](akd/src/aegon/audit.rs), [`consistency`](akd/src/aegon/consistency.rs) — verifier paths

---

## Build & test

```bash
cargo build --release -p akd
cargo test  --release -p akd --tests
```

There are two `#[ignore]`'d benchmarks worth running before sizing:

```bash
# 32-shard publish + lookup at log_capacity=15 per shard
cargo test --release -p akd --test sharded_aegon -- \
  --ignored bench_setup_and_publish --nocapture

# One shard at production size (log_capacity=29, k=10). ~12 min, ~37 GB RSS.
cargo test --release -p akd --test sharded_aegon -- \
  --ignored bench_production_shard_scale --nocapture
```

---

## Top-level directory organization

| Subfolder    | Description |
| :---         | :---        |
| `akd`        | Main library. Sharded coordinator, single-shard engine, gRPC layer, `aegon_shard_server` binary. |
| `akd_core`   | Cryptographic primitives — KZH-k PCS, transcripts, hash, MSM. |
| `examples`   | Worked usage examples plus utilities. |
| `xtask`      | Workspace-wide tooling (code coverage). |

---

## Status

This is research-grade software. Not yet audited.

The original (SEEMless-based) AKD was audited by NCC Group in 2023; that
audit covered a different codebase. The Aegon engine that powers this
fork has not been independently reviewed.

---

## License

Dual MIT / Apache-2.0, inherited from upstream AKD. See `LICENSE-MIT`
and `LICENSE-APACHE`.
