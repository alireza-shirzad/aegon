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
process. No cluster, no network, and nothing to configure.

```bash
cargo run --release -p aegon --example laptop_dictionary
```

The first build takes a few minutes. After that the program finishes in
seconds:

```text
Computing SRS
dictionary ready at epoch 0
published 3 users -> epoch 1
lookup alice = "alice-key-v1", proof verifies: true
alice rotated her key -> epoch 2
lookup alice = "alice-key-v2", proof verifies: true
alice's history verifies:
  epoch 2: "alice-key-v2"
  epoch 1: "alice-key-v1"
bob unchanged since epoch 1: true
alice unchanged since epoch 1: false
audit epoch 0 -> 1: true
audit epoch 1 -> 2: true
```

### What it just did

The whole program is [`aegon/examples/laptop_dictionary.rs`](aegon/examples/laptop_dictionary.rs),
about 140 lines. Open it and follow along:

1. **Set up** a dictionary with one shard and 1,024 slots, backed by a
   RocksDB store in your temp directory.
2. **Publish** keys for alice, bob, and carol. Each publish creates a new
   epoch with a commitment that clients and auditors check against.
3. **Look up** alice's key and verify the proof against epoch 1.
4. **Update** alice's key, then look it up again under epoch 2.
5. **Read alice's history**: every epoch in which her key changed, verified.
6. **Check consistency.** Bob can prove his key hasn't changed since epoch 1;
   the same check for alice returns `false`, because hers did.
7. **Audit** every epoch transition using nothing but the published
   commitments.

### Make it your own

Edit the labels and values, add more publishes, or look up a label from an
older epoch, then rerun the same command. The core calls are:

```rust
let commit = dict.publish_two_layer(&[(label.clone(), value.clone())])?;

let (value, proof) = dict.lookup_two_layer(&label)?;
let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
    &ctx, &commit, &label, &value, &proof,
)?;

let history = dict.lookup_history(&label)?;
verify_lookup_history::<Bn254, Pcs, Sha256Hash>(&ctx, &history)?;
```

A few things worth knowing:

- **Size.** `SHARD_LOG_CAPACITY = 10` gives 1,024 slots. Each step up
  doubles the slots, and setup time and memory grow with them.
- **Setup is cached.** The first run generates the structured reference
  string (SRS) and saves it to `../artifacts/srs/`, relative to the
  directory you run from. Later runs print `Loading SRS` instead. This SRS
  comes from a fixed seed and is **for testing only**.
- **Each run starts fresh.** The store is wiped at the start and end.
  Remove those lines to keep state between runs.

---

## Run it as servers

The same single-shard dictionary, split into the processes a real
deployment uses: a shard server, a coordinator, and a client, talking gRPC
on localhost. Build the binaries once:

```bash
cargo build --release -p aegon --bin aegon_shard_server --bin aegon_coordinator_server --bin aegon_client
```

Then use three terminals. All three must agree on `--shard-log-capacity`,
`--kzh-k`, and `--setup-seed`; the shared seed is what gives them the same
test SRS.

**Terminal 1: the shard.**

```bash
./target/release/aegon_shard_server \
  --bind 127.0.0.1:50051 \
  --shard-log-capacity 10 --kzh-k 3 --setup-seed 42
```

Wait for `aegon_shard_server listening on 127.0.0.1:50051`.

**Terminal 2: the coordinator**, preloaded with 100 users.

```bash
./target/release/aegon_coordinator_server \
  --listen 127.0.0.1:50100 \
  --endpoints http://127.0.0.1:50051 \
  --shard-log-capacity 10 --kzh-k 3 --setup-seed 42 \
  --seed-batch-size 100
```

Wait for `coordinator: serving on 127.0.0.1:50100`. The preloaded users are
labeled `b100-s0-u0` through `b100-s0-u99`, with values `v-0` through `v-99`.

**Terminal 3: look someone up.**

```bash
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

The client verifies both proofs itself; it doesn't take the server's word.
Pass a value the server never published and verification fails:

```bash
./target/release/aegon_client \
  --coordinator http://127.0.0.1:50100 \
  --shard-log-capacity 10 --log-n-shards 0 --kzh-k 3 --setup-seed 42 \
  --label b100-s0-u0 --expected-value not-the-value
```

```text
error: lookup_value_with_bytes: proof verification failed: value opening does not match H_F(value)
```

Stop the servers with Ctrl-C. They keep state in memory, so each start is a
fresh dictionary. `aegon_client` only looks up; to publish your own entries
against a running coordinator, use `CoordinatorClient::publish_two_layer`
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
