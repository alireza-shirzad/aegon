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

## Using Aegon

Add the crates to your project:

```toml
[dependencies]
aegon        = { git = "https://github.com/alireza-shirzad/aegon" }
aegon_crypto = { git = "https://github.com/alireza-shirzad/aegon" }
ark-bn254    = "0.5"
ark-std      = "0.5"
rand_chacha  = "0.3"

# Also copy the [patch.crates-io] block from this repository's Cargo.toml
# into your workspace root. Cargo only reads it there, and without it the
# arkworks crates resolve to incompatible versions.
```

Then start from this template. Every setting is at the top, with its
options in the comments; the rest of the program runs one of each
operation.

```rust
use aegon::audit_fs::AuditFs;
use aegon::ivc::adapter::hooks_for;
use aegon::{
    optimal_kzh_k, shard_log_capacity_for_two_layer, verify_lookup_history,
    verify_sharded_consistency_two_layer, verify_sharded_invariance,
    verify_sharded_lookup_two_layer, DbSource, ShardTransport, ShardedAegon, ShardedAegonConfig,
    ShardedAuditState, SrsSource,
};
use aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::{Bn254, Fr};
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

// ============================== Settings ==============================

/// log2 of how many users the dictionary holds: 10 means 1,024 users.
const TRUE_LOG_CAPACITY: usize = 10;

/// log2 of the shard count: 2 means 4 shards. Capacity is split evenly
/// across them, so more shards means smaller, faster shards.
const LOG_N_SHARDS: usize = 2;

/// Private mode hides users' values from auditors. Required for IVC
/// (fast-forward) auditing.
const PRIVATE: bool = false;

/// The hash auditors recompute: `Poseidon` (supports IVC auditing) or
/// `Sha256`. Servers and auditors must agree, and it can't change later.
const AUDIT_FS: AuditFs = AuditFs::Poseidon;

/// How many independent audit chains to split the shards into. Must divide
/// the shard count. More groups let an IVC audit fold in parallel.
const CHAIN_GROUPS: usize = 1;

/// How users are assigned to slots. `aegon::Sha256Hash` can be computed by
/// anyone; `aegon::EcVrfHash` hides assignments behind a VRF key (then also
/// call `set_vrf_prover`, below).
type Hash = aegon::Sha256Hash;

/// The polynomial commitment scheme: KZH-k over BN254.
type Pcs = KZHK<Bn254>;

// ======================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Each shard's size, including Aegon's 4x headroom for placing users.
    let shard_log_capacity = shard_log_capacity_for_two_layer(TRUE_LOG_CAPACITY, LOG_N_SHARDS);

    // Local storage for this demo, wiped so every run starts fresh.
    let db = std::env::temp_dir().join("aegon-template");
    let _ = std::fs::remove_dir_all(&db);
    let _ = std::fs::remove_dir_all(db.with_extension("shards"));

    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(shard_log_capacity)
        .log_n_shards(LOG_N_SHARDS)
        .private(PRIVATE) // must come before `kzh_k`
        .kzh_k(optimal_kzh_k(shard_log_capacity))
        .audit_fs(hooks_for(AUDIT_FS))
        .chain_groups(CHAIN_GROUPS)
        // Where the shards run:
        //   ShardTransport::InProcess                   all in this process
        //   ShardTransport::Remote { endpoints: urls }  one aegon_shard_server per
        //                                               shard, 2^LOG_N_SHARDS URLs
        .shards(ShardTransport::InProcess)
        // The trusted setup (SRS):
        //   SrsSource::DangerouslyGenerate  throwaway, made from the rng below; testing only
        //   SrsSource::Path(file)           load one produced by a setup ceremony
        .srs(SrsSource::DangerouslyGenerate)
        // Where users' values and history are stored:
        //   DbSource::None        memory only: lookups return no value bytes, no history
        //   DbSource::Rocks(dir)  a local RocksDB
        //   DbSource::Redis(url)  a shared Redis (with remote shards only)
        .db(DbSource::Rocks(db))
        // In private mode, in-process shards mask their proofs themselves. To
        // use separate aegon_masking_server processes instead:
        //   .masking_addrs(vec!["http://127.0.0.1:50061".into()])
        .build()?;

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut server = ShardedAegon::<Bn254, Pcs, Hash>::setup(&mut rng, &cfg)?;

    // With `type Hash = aegon::EcVrfHash`, give the server its VRF key, from
    // AEGON_VRF_SEED or AEGON_VRF_KEY_PATH:
    //   server.set_vrf_prover(aegon::VrfProver::from_env());

    // What clients and auditors hold: the public verification context (take it
    // after setting a VRF key) and the commitment published at each epoch.
    let ctx = server.sharded_verifier_context();
    let epoch_0 = server.current_commitment();

    // ---- Server: publish a batch. Each publish starts a new epoch. ----
    let alice = b"alice".to_vec();
    let epoch_1 = server.publish_two_layer(&[(alice.clone(), b"alice-key-1".to_vec())])?;

    // ---- Client: look Alice up and check the proof against epoch 1. ----
    let (value, proof) = server.lookup_two_layer(&alice)?;
    assert!(verify_sharded_lookup_two_layer::<Bn254, Pcs, Hash>(
        &ctx, &epoch_1, &alice, &value, &proof
    )?);

    // ---- Client: every change to Alice's value (needs a DbSource). ----
    let history = server.lookup_history(&alice)?;
    verify_lookup_history::<Bn254, Pcs, Hash>(&ctx, &history)?;

    // ---- Client: prove Alice's entry hasn't changed since epoch 1. ----
    let epoch_2 = server.publish_two_layer(&[(b"bob".to_vec(), b"bob-key-1".to_vec())])?;
    let proof = server.consistency_proof_two_layer(&alice, epoch_1.epoch)?;
    assert!(verify_sharded_consistency_two_layer::<Bn254, Pcs, Hash>(
        &ctx, &epoch_1, &epoch_2, &alice, &proof
    )?);

    // ---- Auditor: check every epoch transition, in order, from epoch 0. ----
    // The audit state tracks one running value per chain group.
    let mut audit = ShardedAuditState::<Fr>::with_groups(CHAIN_GROUPS);
    for (prev, next) in [(&epoch_0, &epoch_1), (&epoch_1, &epoch_2)] {
        assert!(verify_sharded_invariance::<Bn254, Pcs>(&ctx, &mut audit, prev, next)?);
    }

    println!("every check passed");
    Ok(())
}
```

For a complete, runnable walkthrough — including fast-forward (IVC)
auditing and running the shard, coordinator, and client as separate
processes — see [`aegon/examples`](aegon/examples/README.md).

---

## Going further

- [`docs/deployment.md`](docs/deployment.md): running across many machines
  (SRS generation, one shard server per machine, the coordinator, TLS, and
  retries), plus an architecture overview.
- [`docs/development.md`](docs/development.md): building, testing, known
  test limitations, and fast-forward (IVC) auditing.
- [`SECURITY.md`](SECURITY.md): what this software does not protect against.
  Read it before deploying anything.

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
