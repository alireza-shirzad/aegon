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

Then follow a walkthrough. Each one is a program you can run as it stands:

- [One machine](aegon/examples/README.md): a dictionary in a single
  process, across three epochs, ending in a fast-forward (IVC) audit —
  then the same dictionary split into a shard server, a coordinator, and
  a client.
- [A four-shard cluster](aegon/examples/README-cluster.md): four shards,
  two masking servers, and a coordinator, each its own process, running in
  private mode.

- [Every setting in one file](aegon/examples/configured_dictionary.rs): a
  short program with every builder call — capacity, shard count, private
  mode, audit transcript, chain groups, transport, storage, VRF key — each
  with its default and options in the comments. Copy it and edit.

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
