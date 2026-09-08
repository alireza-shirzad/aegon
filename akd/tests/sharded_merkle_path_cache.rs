// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Verifies that the precomputed Merkle path cache inside
//! `ShardedEpochCommitment` is byte-identical to the bare
//! `build_merkle_path` walk it replaces, and that the cache survives
//! a `CanonicalSerialize` round trip (rebuilt deterministically from
//! `per_shard`, not carried on the wire).

use akd::aegon::{
    build_merkle_path, merkle_root, Sha256Hash, ShardedAegon, ShardedAegonConfig,
    ShardedEpochCommitment,
};
use ark_bn254::Bn254;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Pcs = akd_core::aegon_crypto::pcs::kzhk::KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, Sha256Hash>;

fn config(log_capacity: usize, log_n_shards: usize) -> ShardedAegonConfig<Bn254, Pcs> {
    ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(log_capacity - log_n_shards)
        .log_n_shards(log_n_shards)
        .private(false)
        .kzh_k(2)
        .build()
        .expect("config")
}

fn fresh(log_capacity: usize, log_n_shards: usize) -> Sharded {
    let mut rng = ChaCha20Rng::seed_from_u64(0x9001);
    Sharded::setup(&mut rng, &config(log_capacity, log_n_shards)).expect("setup")
}

/// At genesis (epoch 0, empty per-shard polynomials) the cached paths
/// must match `build_merkle_path` on every shard. Also covers the
/// `n_shards = 1` degenerate case (path is empty for the lone shard;
/// root equals the leaf hash).
#[test]
fn cached_paths_match_bare_walk_at_every_shard() {
    for &log_n_shards in &[0usize, 1, 2, 3] {
        let n_shards = 1usize << log_n_shards;
        let server = fresh(6 + log_n_shards, log_n_shards);
        let commit = server.current_commitment();
        assert_eq!(commit.per_shard.len(), n_shards);
        // The struct's `merkle_root` field must equal the bare
        // walk's root — confirms the constructor's bookkeeping.
        let bare_root = merkle_root::<Bn254, Pcs>(&commit.per_shard);
        assert_eq!(
            commit.merkle_root, bare_root,
            "merkle_root mismatch at n_shards={n_shards}"
        );
        for i in 0..n_shards {
            let cached = commit.merkle_path(i);
            let bare = build_merkle_path::<Bn254, Pcs>(&commit.per_shard, i);
            assert_eq!(
                cached,
                bare.as_slice(),
                "path mismatch at n_shards={n_shards} shard_id={i}"
            );
        }
    }
}

/// Path cache must rebuild from `per_shard` after a CanonicalSerialize
/// round trip — the wire format deliberately omits the cache to save
/// bytes, so the deserialiser reconstructs it. Equality is then both
/// `merkle_root` (which is also serialised, as a tamper check) and
/// per-shard `merkle_path(i)`.
#[test]
fn cache_round_trips_through_canonical_serialize() {
    let server = fresh(8, 2); // 4 shards, log_capacity=8 total
    let commit = server.current_commitment();

    // Round-trip uncompressed (production wire encoding used
    // everywhere in Aegon).
    let mut bytes = Vec::new();
    commit
        .serialize_uncompressed(&mut bytes)
        .expect("serialize");
    let restored: ShardedEpochCommitment<Bn254, Pcs> =
        ShardedEpochCommitment::<Bn254, Pcs>::deserialize_uncompressed_unchecked(&bytes[..])
            .expect("deserialize");

    assert_eq!(restored.epoch, commit.epoch);
    assert_eq!(restored.merkle_root, commit.merkle_root);
    assert_eq!(restored.per_shard.len(), commit.per_shard.len());
    // Field-by-field commit equality is awkward (no PartialEq), so
    // re-serialise the inner per-shard slice and compare bytes.
    let mut a = Vec::new();
    let mut b = Vec::new();
    commit.per_shard.serialize_uncompressed(&mut a).unwrap();
    restored.per_shard.serialize_uncompressed(&mut b).unwrap();
    assert_eq!(a, b, "per_shard bytes drift across round trip");
    // Path cache must be reconstructed identically.
    for i in 0..commit.per_shard.len() {
        assert_eq!(
            commit.merkle_path(i),
            restored.merkle_path(i),
            "rebuilt path differs at shard_id={i}"
        );
    }
}

/// Tamper check: if the per-shard slice on the wire is consistent
/// with itself but the announced `merkle_root` is wrong (someone
/// modified the bytes), deserialise must reject. This is the
/// CanonicalDeserialize impl's only added invariant — exercising it
/// confirms the rebuilt cache is being cross-checked.
#[test]
fn deserialize_rejects_root_per_shard_mismatch() {
    let server = fresh(8, 2);
    let commit = server.current_commitment();
    let mut bytes = Vec::new();
    commit
        .serialize_uncompressed(&mut bytes)
        .expect("serialize");
    // Wire layout: epoch (8 B u64) || merkle_root (32 B) || per_shard.
    // Corrupt the first byte of merkle_root.
    bytes[8] ^= 0xFF;
    let result =
        ShardedEpochCommitment::<Bn254, Pcs>::deserialize_uncompressed_unchecked(&bytes[..]);
    assert!(
        result.is_err(),
        "deserialize must reject a tampered merkle_root that disagrees with per_shard"
    );
}
