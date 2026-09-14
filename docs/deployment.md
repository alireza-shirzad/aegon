# Cluster deployment (N machines + 1 coordinator)

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

## Automated: one script

For prototyping on GCE, [`scripts/cluster.sh`](../scripts/cluster.sh) does
the whole flow:

```bash
export PROJECT=your-gcp-project-id

./scripts/cluster.sh up      # VPC + 4 shards + 1 coordinator + 1 db
./scripts/cluster.sh deploy  # build, generate SRS, ship, start servers, install redis
./scripts/cluster.sh smoke   # publish + lookup + verify (queries redis)
./scripts/cluster.sh down    # delete everything
```

End-to-end runtime is under five minutes for the default 4-shard
config. See [`scripts/README.md`](../scripts/README.md) for tuning knobs
(`N_SHARDS`, `SHARD_LOG_CAPACITY`, machine types).

The rest of this section walks through the same flow manually if you
want to understand or customize each step.

## Picking parameters

For a target user count `N_users`, pick:

```
log_capacity = ceil(log2(N_users * 4))   # 4× slack → ≤25% load factor
log_n_shards = pick_so_that_shard_log_capacity_fits_one_machine
shard_log_capacity = log_capacity - log_n_shards
kzh_k             = optimal_kzh_k(shard_log_capacity)
```

The `optimal_kzh_k` function (in `aegon::presets`) minimizes the
aux-precomputation cost `f(k) = k(k − 1) · 2^(N/k)` and is tabulated
for `N ∈ [20, 35]`. For production at `shard_log_capacity = 29`, it
returns `k = 10`. Validated on a 16 vCPU / 64 GB box: setup ≈ 12 min,
peak RSS ≈ 37 GB (with `gen_srs_for_testing`; load-from-file is much
faster).

Example: 4B users × 4× slack ≈ 2³⁴ slots. Split as 2⁵ shards × 2²⁹
slots each → 32 machines, each at `shard_log_capacity = 29, kzh_k = 10`.

## Step 1: Generate the SRS, once

Run this on **one** machine (any machine with enough RAM to do an SRS
gen at `shard_log_capacity`). The output is one file you'll copy to
every shard + the coordinator.

```bash
cargo build --release -p aegon --bin aegon_srs_gen
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

## Step 2: Run a shard server on every shard machine

Build the binary:

```bash
cargo build --release -p aegon --bin aegon_shard_server
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

## Step 3: Wire the coordinator up

```rust
use aegon::ivc::adapter::hooks_for;
use aegon::{DbSource, Sha256Hash, ShardTransport, ShardedAegon, ShardedAegonConfig, SrsSource};
use aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;

let cfg = ShardedAegonConfig::<Bn254, KZHK<Bn254>>::builder()
    .shard_log_capacity(29)
    .log_n_shards(5)
    .kzh_k(10)
    // Must match the transcript the shard servers run (`--audit-fs`,
    // Poseidon by default) or every audit is rejected.
    .audit_fs(hooks_for(Default::default()))
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

let mut server = ShardedAegon::<Bn254, KZHK<Bn254>, Sha256Hash>::setup(&cfg)?;
```

`setup` here is the only point where the coordinator talks to all 32
shards — it does the gRPC handshake and reads each shard's initial
commitment. After that, every `publish` / `lookup` /
`consistency_proof` call does whatever subset of gRPC calls the
protocol requires.

## Step 4: Smoke-test the cluster

After every shard is up, run the smoke client on the coordinator
machine:

```bash
cargo build --release -p aegon --bin aegon_coordinator_smoke
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

## TLS

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
use aegon::shard_grpc::{GrpcShardClientConfig, GrpcShardClient};

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

## Retry policy

Reads (`open_*`, `is_index_slot_occupied`, `current_commitment`)
retry on transport-level failures (server `Unavailable`/`Unknown`)
using exponential backoff. Writes (`publish_phase_1` /
`publish_phase_2`) do **not** auto-retry — those aren't idempotent
and need protocol-level coordination (out of scope for v1).

Default policy is `max_attempts = 5, initial_backoff = 50ms,
max_backoff = 2s`. Override via `GrpcShardClientConfig::with_retry`.


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

- [`aegon::sharded`](../aegon/src/sharded.rs) — coordinator + sharded proof types
- [`aegon::shard_grpc`](../aegon/src/shard_grpc.rs) — `ShardHandle` trait, server/client adapters
- [`aegon::server`](../aegon/src/server.rs) — single-shard `Aegon`
- [`aegon::verify`](../aegon/src/verify.rs), [`audit`](../aegon/src/audit.rs), [`consistency`](../aegon/src/consistency.rs) — verifier paths
