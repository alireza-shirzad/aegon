# Build & test

```bash
cargo build --release -p aegon
cargo test  --release -p aegon --tests
```

There are two `#[ignore]`'d benchmarks worth running before sizing:

```bash
# 32-shard publish + lookup at log_capacity=15 per shard
cargo test --release -p aegon --test sharded_aegon -- \
  --ignored bench_setup_and_publish --nocapture

# One shard at production size (log_capacity=29, k=10). ~12 min, ~37 GB RSS.
cargo test --release -p aegon --test sharded_aegon -- \
  --ignored bench_production_shard_scale --nocapture
```

## Test status

`cargo test -p aegon` and `cargo test -p aegon_crypto` are both green, and CI
keeps them that way. Test the two crates in **separate** invocations: a combined
`-p aegon -p aegon_crypto` unifies features, which switches `aegon_crypto` to
`parallel` and runs its SRS-heavy unit tests under nested rayon pools,
exhausting the thread limit inside arkworks' MSM.

A few tests are `#[ignore]`d, all of them slow benchmarks rather than known
failures; each carries its reason in the attribute, and `--ignored` runs them.

### Known limitation: `aegon_crypto` alone, with `parallel`

`cargo test -p aegon_crypto --features parallel` fails, and the cause is upstream.
The pinned arkworks revision builds a **fresh rayon `ThreadPool` per chunk, on
every MSM call**, and unwraps the result:

```rust
// algebra @ 598a5fb, ec/src/scalar_mul/variable_base/mod.rs:546
let result = rayon::ThreadPoolBuilder::new()
    .num_threads(THREADS_PER_CHUNK.min(rayon::current_num_threads()))
    .build()
    .unwrap()
    .install(|| msm_bigint_wnaf_parallel::<V>(bases, scalars));
```

`msm_unchecked` → `msm_bigint` → `msm_bigint_wnaf` is the path BN254 G1 takes
(`NEGATION_IS_CHEAP`), and `num_chunks` is `current_num_threads() / 2`, so each
MSM spawns and tears down roughly one thread per core.

Two ways that surfaces, both in debug builds:

* At the default stack, `test_dense_boolean_k4` overflows a worker's stack --
  alone, at `--test-threads=1`, so it is depth and not contention.
* Raise `RUST_MIN_STACK` enough to clear that and the four `k5` tests fail
  instead, because spawning those per-chunk pools with large stacks returns
  `EAGAIN`: `ThreadPoolBuildError { IOError(Os { code: 35, WouldBlock }) }`.

`aegon` depends on `aegon_crypto` with `parallel` enabled and exercises the same code
through its own suite, which passes -- its MSMs are smaller, so it does not
reach either edge. That is why the two crates are tested separately.

Worth knowing beyond the test failure: this per-call pool construction is on
the hot path for every commit and opening, release builds included. Whether a
newer arkworks revision avoids it is untested here; the revisions are pinned
deliberately, and changing them moves the published measurements.

### In-process shards now carry their own store

Until recently an in-process shard had no storage: its publish write-sink
discarded every chunk and its reads returned empty, so `ShardTransport::InProcess`
served empty values and empty history at *any* `DbSource` -- silently, as `Ok`.
`DbSource` configured only the coordinator. Two things changed:

* `Aegon` now holds an optional `Box<dyn Db>`. `ShardedAegon::setup` opens one
  RocksDB per in-process shard under `<db-path>.shards/<i>`, mirroring the
  per-shard store a gRPC deployment gets. Each shard needs its own, because
  `key_history_openings_local(epoch)` carries no shard discriminator --
  shards sharing a keyspace would overwrite each other every epoch. For that
  reason `DbSource::Redis` combined with `ShardTransport::InProcess` is now a
  setup-time error rather than silent corruption.
* A history entry's `prev_shard_commit` is read from the epoch snapshot rather
  than the live commitment fields. By the time it was captured, phase 1 had
  already overwritten those with the *new* epoch's values while `self.epoch`
  was still the old number, so entries anchored their `rand_value_pre_proof`
  against the wrong commitment. The genesis publish was also skipped entirely,
  so a label's placement never appeared in its own history.

## Fast-forward (IVC) auditing

Behind the off-by-default `ivc_audit` feature. An auditor that has been
offline verifies one recursive proof instead of replaying every missed
epoch, at a cost independent of how many it missed.

```bash
cargo test -p aegon --features ivc_audit
cargo run --release -p aegon --features ivc_audit --bin aegon_ivc_bench -- --help
cargo run --release -p aegon --features ivc_audit --example ivc_audit_grouped_e2e
```

The feature gates the folding circuit and prover only. The Poseidon audit
transcript they depend on is the **default** and is always compiled in, so a
directory started today can be fast-forward audited later without having been
built with `ivc_audit` — which matters, because the transcript is fixed from a
chain's first epoch and cannot be changed later.

Every server, client, and benchmark binary takes `--audit-fs poseidon`
(default) or `--audit-fs sha256`. In library code, install the hooks on the
config builder:

```rust
use aegon::{audit_fs::AuditFs, ivc::adapter::hooks_for};
let cfg = ShardedAegonConfig::<Bn254, KZHK<Bn254>>::builder()
    .audit_fs(hooks_for(AuditFs::Poseidon))   // or AuditFs::Sha256
    // ...
    .build()?;
```

Note that the builder itself is generic over the curve and so defaults to
SHA-256: Poseidon hashes into BN254's base field and is undefined elsewhere.
Library code that wants the Poseidon default has to ask for it, as above.

Servers and auditors must agree: a mismatch rejects every epoch. The Merkle
commitment, lookup, consistency, and history paths are unaffected and stay on
SHA-256 either way. See `SECURITY.md`.
