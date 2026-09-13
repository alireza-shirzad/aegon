<p align="center">
  <img src="icon.png" alt="Aegon" width="160">
</p>

<h1 align="center">Aegon</h1>

<p align="center">
  <em>Self-auditable key transparency</em>
</p>

Sharded, polynomial-commitment-backed key transparency. **Aegon** is built
on KZH-k polynomial commitments, sharded across N machines, and coordinated
over gRPC. Every epoch is auditable in constant time, and an auditor that
falls behind can catch up with a single recursive proof.

The repository has two crates: `aegon`, the engine, servers, and clients;
and `aegon_crypto`, the polynomial commitment scheme and supporting
primitives. It started as a fork of [facebook/akd](https://github.com/facebook/akd),
but none of AKD's directory, Merkle-tree backend, or API remains — only its
ECVRF implementation (see `NOTICE`).

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

## Prerequisites

Two system libraries are needed before `cargo build` will work; neither is
pulled in by Cargo.

| Dependency | Why | Install |
| :--- | :--- | :--- |
| `protoc` | `aegon`'s build script compiles the gRPC `.proto` specs | `apt install protobuf-compiler` / `brew install protobuf` |
| `libclang` | `librocksdb-sys` runs `bindgen` | `apt install libclang-dev clang` / `brew install llvm` |

On Linux, installing `libclang-dev` is enough. On macOS, Homebrew's LLVM is
not on the default search path and the build fails with a `dyld` error naming
`libclang.dylib`; export both of these:

```bash
export LIBCLANG_PATH=/opt/homebrew/opt/llvm/lib
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/llvm/lib
```

The Rust toolchain is pinned in `rust-toolchain.toml` and installs itself on
first `cargo` invocation. `Cargo.lock` is committed deliberately — this
repository's published results are timing measurements, and a floating
dependency graph moves them.

---

## Run a dictionary on your laptop

The quickest way to see Aegon work is a single-shard dictionary in one
process: no cluster, no network, nothing to configure. One command runs a
short story across three epochs:

```bash
# Build and run the example. `ivc_audit` enables the recursive (IVC) audit
# used in epoch 3. The first build takes a few minutes; the run takes seconds.
cargo run --release -p aegon --features ivc_audit --example laptop_dictionary
```

```text
Computing SRS
epoch 0: empty dictionary

epoch 1: alice and bob join
  client looks up alice -> "alice-key-1", proof verifies: true

epoch 2: bob changes his key
  auditor checks epoch 0 -> 1: true
  auditor checks epoch 1 -> 2: true

epoch 3: alice changes her key
  prover folds all 3 transitions into one proof
  offline auditor verifies one proof covering 3 epochs: true
```

The whole program is [`aegon/examples/laptop_dictionary.rs`](aegon/examples/laptop_dictionary.rs).
Here it is step by step.

### Setup

```rust
// One shard with 1,024 slots, backed by a RocksDB store.
let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
    .shard_log_capacity(10)                  // 2^10 = 1,024 slots
    .log_n_shards(0)                         // 2^0 = one shard
    .private(true)                           // hide values from auditors; needed for IVC (set before kzh_k)
    .kzh_k(optimal_kzh_k(10))                // commitment-scheme parameter for this size
    .audit_fs(hooks_for(AuditFs::Poseidon))  // the hash auditors recompute; IVC needs Poseidon
    .db(DbSource::Rocks(db_path))            // where the server stores users' values
    .build()?;

// Start the server. The seed generates a throwaway trusted setup,
// fine for a demo; a real deployment loads one from a setup ceremony.
let mut rng = ChaCha20Rng::seed_from_u64(42);
let mut server = Server::setup(&mut rng, &cfg)?;

// All a client or auditor ever holds: the public verification context,
// plus the commitment the server publishes at each epoch.
let ctx = server.sharded_verifier_context();
let mut published = vec![server.epoch_commitment(0).ok_or("no epoch 0")?];
```

### Epoch 1: Alice and Bob join, and a client looks Alice up

```rust
// The server publishes a batch of (user, key) pairs. That starts epoch 1
// and returns the commitment it publishes for it.
let epoch_1 = server.publish_two_layer(&[
    (b"alice".to_vec(), b"alice-key-1".to_vec()),
    (b"bob".to_vec(), b"bob-key-1".to_vec()),
])?;
published.push(epoch_1.clone());

// The client asks for Alice. The server returns her key and a proof.
let (value, proof) = server.lookup_two_layer(&b"alice".to_vec())?;

// The client checks the proof against epoch 1's published commitment.
// If the server returned the wrong key, this would fail.
let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
    &ctx, &epoch_1, &b"alice".to_vec(), &value, &proof,
)?;
```

### Epoch 2: Bob changes his key, and an auditor checks each epoch

```rust
// Publishing Bob's new key starts epoch 2.
published.push(server.publish_two_layer(&[(b"bob".to_vec(), b"bob-key-2".to_vec())])?);

// The classic audit checks one epoch transition at a time, using only the
// published commitments. It carries a small state from each check to the
// next, so the auditor starts at epoch 0 and must see every transition.
let mut audit_state = ShardedAuditState::<Fr>::default();
for pair in published.windows(2) {
    let ok = verify_sharded_invariance::<Bn254, Pcs>(&ctx, &mut audit_state, &pair[0], &pair[1])?;
}
```

### Epoch 3: Alice changes her key, and an offline auditor catches up at once

```rust
// Publishing Alice's new key starts epoch 3.
published.push(server.publish_two_layer(&[(b"alice".to_vec(), b"alice-key-2".to_vec())])?);

// An auditor who missed all three epochs would normally replay them one by
// one. Instead, a prover folds every transition into a single recursive
// proof. It needs only the published commitments, so anyone can run it.
let h = ctx.inner.verifier_param.get_h();                      // a public parameter
let ivc = Arc::new(IvcAuditParams::setup(fs_params_from_epoch(&published[1])?, h)?);
let mut prover = IvcAuditProver::new(ivc.clone(), &epoch_commitments(&published[0]))?;
for epoch in &published[1..] {
    // Each fold absorbs one epoch's commitments and its blinding proof.
    prover.fold_epoch(&epoch_commitments(epoch), &epoch_sigma_witnesses(epoch)?)?;
}

// The offline auditor makes one call: verify the proof, and confirm it ends
// at the commitment the server published for epoch 3.
let latest = published.last().unwrap();
let verified = verify_against_merkle_root(
    &ivc,
    prover.proof().ok_or("nothing folded")?,
    prover.num_steps(),                  // how many transitions the proof covers
    prover.z0(),                         // the proof's starting state (epoch 0)
    &epoch_commitments(latest),          // epoch 3's published commitments
    latest.merkle_root,                  // and their published root
    || merkle_root(&latest.per_shard),
)?;
```

### Make it your own

Edit the users and keys in the example, add more epochs, or look someone up
at a different epoch, then rerun the same command. A few things worth knowing:

- **Size.** `SHARD_LOG_CAPACITY = 10` gives 1,024 slots. Each step up
  doubles the slots, and setup time and memory grow with them.
- **Setup is cached.** The first run generates the trusted setup (the SRS)
  and saves it to `../artifacts/srs/`, relative to the directory you run
  from; later runs print `Loading SRS`. This setup comes from a fixed seed
  and is **for testing only**.
- **Each run starts fresh.** The example wipes its store at the start and
  end. Remove those lines to keep state between runs.

---

## Run it as servers

The same kind of single-shard dictionary, split into the processes a real
deployment uses: a shard server, a coordinator, and a client, talking gRPC
on localhost.

```bash
# Build the three binaries once.
cargo build --release -p aegon --bin aegon_shard_server --bin aegon_coordinator_server --bin aegon_client
```

Then open three terminals. All three processes must use the same
`--shard-log-capacity`, `--kzh-k`, and `--setup-seed`; the shared seed is
what gives them the same test setup.

**Terminal 1: the shard.**

```bash
# The shard holds the dictionary's data and computes proofs.
#   --bind                where it listens
#   --shard-log-capacity  log2 of its slot count (10 = 1,024 slots)
#   --kzh-k               commitment-scheme parameter for that size
#   --setup-seed          generates the test setup (every process must match)
./target/release/aegon_shard_server \
  --bind 127.0.0.1:50051 \
  --shard-log-capacity 10 --kzh-k 3 --setup-seed 42
```

Wait for `aegon_shard_server listening on 127.0.0.1:50051`.

**Terminal 2: the coordinator**, preloaded with 100 users.

```bash
# The coordinator is what clients talk to. It routes each request to the shard.
#   --listen           where clients connect
#   --endpoints        the shard(s) to use: one shard here
#   --seed-batch-size  publish 100 test users at startup
./target/release/aegon_coordinator_server \
  --listen 127.0.0.1:50100 \
  --endpoints http://127.0.0.1:50051 \
  --shard-log-capacity 10 --kzh-k 3 --setup-seed 42 \
  --seed-batch-size 100
```

Wait for `coordinator: serving on 127.0.0.1:50100`. The preloaded users are
`b100-s0-u0` through `b100-s0-u99`, with values `v-0` through `v-99`.

**Terminal 3: look someone up.**

```bash
# The client fetches the current commitment, looks the user up, and verifies
# both proofs itself.
#   --log-n-shards    log2 of the shard count (0 = one shard)
#   --label           the user to look up
#   --expected-value  the value the proof must match
./target/release/aegon_client \
  --coordinator http://127.0.0.1:50100 \
  --shard-log-capacity 10 --log-n-shards 0 --kzh-k 3 --setup-seed 42 \
  --label b100-s0-u0 --expected-value v-0
```

```text
client: connecting to http://127.0.0.1:50100 ...
client: current commitment OK
client: lookup_label OK in 4.7 ms → shard 0, slot len 10
client: lookup_value OK in 4.1 ms → eval matches H_F(value), proof verified
client: done.
```

The client doesn't take the server's word for anything. Ask it to verify a
value the server never published, and it refuses:

```bash
# Same user, wrong value: verification must fail.
./target/release/aegon_client \
  --coordinator http://127.0.0.1:50100 \
  --shard-log-capacity 10 --log-n-shards 0 --kzh-k 3 --setup-seed 42 \
  --label b100-s0-u0 --expected-value not-the-value
```

```text
error: lookup_value_with_bytes: proof verification failed: value opening does not match H_F(value)
```

Stop the servers with Ctrl-C. They keep state in memory, so each start is a
fresh dictionary. `aegon_client` only looks users up; to publish your own
entries to a running coordinator, use `CoordinatorClient::publish_two_layer`
from the `aegon` crate.

---

## Going further

- [`docs/deployment.md`](docs/deployment.md): running across many machines
  (SRS generation, one shard server per machine, the coordinator, TLS, and
  retries), plus an architecture overview.
- [`docs/development.md`](docs/development.md): building, testing, known
  test limitations, and fast-forward (IVC) auditing.
- [`SECURITY.md`](SECURITY.md): what this software does not protect against.
  Read it before deploying anything.

## Repository layout

| Path            | Contents |
| :---            | :---     |
| `aegon`         | The engine, gRPC layer, IVC auditing, and every server, client, and benchmark binary. |
| `aegon_crypto`  | KZH-k polynomial commitments, multilinear arithmetic, transcripts, MSM, ECVRF. |
| `docs`          | Deployment and development guides. |
| `scripts`       | Cluster bring-up and benchmark drivers (Google Cloud). |
| `bench-results` | Benchmark outputs and the plotting scripts behind the paper's figures. |
| `xtask`         | Code-coverage tooling. |

---

## Status

This is research-grade software. It has not been independently audited.

---

## Citing

This is the reference implementation for:

> Hossein Hafezi, Alireza Shirzad, Benedikt Bünz, Kevin Lewi, Dillon George,
> and Joseph Bonneau. *Aegon: Self-Auditable Key Transparency.* Cryptology
> ePrint Archive, Paper 2026/1681, 2026. <https://eprint.iacr.org/2026/1681>

```bibtex
@misc{cryptoeprint:2026/1681,
      author = {Hossein Hafezi and Alireza Shirzad and Benedikt Bünz and Kevin Lewi and Dillon George and Joseph Bonneau},
      title = {Aegon: Self-Auditable Key Transparency},
      howpublished = {Cryptology {ePrint} Archive, Paper 2026/1681},
      year = {2026},
      url = {https://eprint.iacr.org/2026/1681}
}
```

---

## License

MIT. See `LICENSE`.

`NOTICE` records what this repository derives from. The ECVRF implementation
in `aegon_crypto::ecvrf` comes from [facebook/akd](https://github.com/facebook/akd)
(Meta Platforms, MIT OR Apache-2.0; this repository exercises the MIT option and
preserves Meta's copyright notice). Parts of the multilinear arithmetic, PCS
traits, and transcript in `aegon_crypto` derive from
[EspressoSystems/hyperplonk](https://github.com/EspressoSystems/hyperplonk),
also MIT. Source files carry the header of whichever copyright applies.

Security caveats — unaudited cryptography, a non-ceremonial trusted setup,
and unauthenticated transports — are in `SECURITY.md`. Read it before
deploying anything.
