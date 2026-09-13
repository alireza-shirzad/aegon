// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! A small Aegon dictionary on one machine: one shard, 1,024 slots, a
//! local RocksDB store. Walks through every operation once.
//!
//! ```text
//! cargo run --release -p aegon --example laptop_dictionary
//! ```
//!
//! Edit the labels and values below to experiment.

use aegon::ivc::adapter::hooks_for;
use aegon::{
    optimal_kzh_k, verify_lookup_history, verify_sharded_consistency_two_layer,
    verify_sharded_invariance, verify_sharded_lookup_two_layer, DbSource, Sha256Hash, ShardedAegon,
    ShardedAegonConfig, ShardedAuditState,
};
use aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::{Bn254, Fr};
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;
type Dictionary = ShardedAegon<Bn254, Pcs, Sha256Hash>;

/// log2 of the slot count. 10 means 1,024 slots, enough for ~256 users
/// at Aegon's 4x over-provisioning.
const SHARD_LOG_CAPACITY: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 1. Set up -------------------------------------------------------
    // A fresh store for each run. In-process shards keep their own
    // RocksDB next to it, under `<path>.shards/`.
    let db_path = std::env::temp_dir().join("aegon-laptop-dictionary");
    let shard_db_path = db_path.with_extension("shards");
    let _ = std::fs::remove_dir_all(&db_path);
    let _ = std::fs::remove_dir_all(&shard_db_path);

    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(SHARD_LOG_CAPACITY)
        .log_n_shards(0) // 2^0 = 1 shard
        .private(false)
        .kzh_k(optimal_kzh_k(SHARD_LOG_CAPACITY))
        .audit_fs(hooks_for(Default::default())) // Poseidon transcript
        .db(DbSource::Rocks(db_path.clone())) // needed for history
        .build()?;

    // Generates a throwaway SRS from this seed. Fine on a laptop; a real
    // deployment loads one from a trusted-setup ceremony instead.
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut dict = Dictionary::setup(&mut rng, &cfg)?;
    let ctx = dict.sharded_verifier_context();
    println!(
        "dictionary ready at epoch {}",
        dict.current_commitment().epoch
    );

    // Everything a client or auditor checks is anchored to a published
    // epoch commitment. Keep each one, as a bulletin board would.
    let mut commitments = vec![dict.current_commitment()];

    // ---- 2. Publish ------------------------------------------------------
    let epoch_1 = dict.publish_two_layer(&[
        (b"alice".to_vec(), b"alice-key-v1".to_vec()),
        (b"bob".to_vec(), b"bob-key-v1".to_vec()),
        (b"carol".to_vec(), b"carol-key-v1".to_vec()),
    ])?;
    println!("published 3 users -> epoch {}", epoch_1.epoch);
    commitments.push(epoch_1.clone());

    // ---- 3. Look up and verify -------------------------------------------
    let alice = b"alice".to_vec();
    let (value, proof) = dict.lookup_two_layer(&alice)?;
    let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
        &ctx, &epoch_1, &alice, &value, &proof,
    )?;
    println!(
        "lookup alice = {:?}, proof verifies: {ok}",
        String::from_utf8_lossy(&value)
    );

    // ---- 4. Update a value -----------------------------------------------
    let epoch_2 = dict.publish_two_layer(&[(alice.clone(), b"alice-key-v2".to_vec())])?;
    println!("alice rotated her key -> epoch {}", epoch_2.epoch);
    commitments.push(epoch_2.clone());

    let (value, proof) = dict.lookup_two_layer(&alice)?;
    let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
        &ctx, &epoch_2, &alice, &value, &proof,
    )?;
    println!(
        "lookup alice = {:?}, proof verifies: {ok}",
        String::from_utf8_lossy(&value)
    );

    // ---- 5. Value history ------------------------------------------------
    // Every epoch in which alice's value changed, newest first.
    let history = dict.lookup_history(&alice)?;
    verify_lookup_history::<Bn254, Pcs, Sha256Hash>(&ctx, &history)?;
    println!("alice's history verifies:");
    for entry in &history.entries {
        println!(
            "  epoch {}: {:?}",
            entry.epoch,
            String::from_utf8_lossy(&entry.value_bytes)
        );
    }

    // ---- 6. Consistency: "has my entry changed since epoch N?" ----------
    // Bob has not touched his key since epoch 1, so this accepts.
    let bob = b"bob".to_vec();
    let proof = dict.consistency_proof_two_layer(&bob, epoch_1.epoch)?;
    let unchanged = verify_sharded_consistency_two_layer::<Bn254, Pcs, Sha256Hash>(
        &ctx, &epoch_1, &epoch_2, &bob, &proof,
    )?;
    println!("bob unchanged since epoch 1: {unchanged}");

    // Alice changed hers, so the same check comes back false.
    let proof = dict.consistency_proof_two_layer(&alice, epoch_1.epoch)?;
    let unchanged = verify_sharded_consistency_two_layer::<Bn254, Pcs, Sha256Hash>(
        &ctx, &epoch_1, &epoch_2, &alice, &proof,
    )?;
    println!("alice unchanged since epoch 1: {unchanged}");

    // ---- 7. Audit --------------------------------------------------------
    // An auditor checks each transition from the published commitments
    // alone, carrying a small running state from one epoch to the next.
    let mut audit = ShardedAuditState::<Fr>::default();
    for pair in commitments.windows(2) {
        let ok = verify_sharded_invariance::<Bn254, Pcs>(&ctx, &mut audit, &pair[0], &pair[1])?;
        println!("audit epoch {} -> {}: {ok}", pair[0].epoch, pair[1].epoch);
    }

    drop(dict);
    let _ = std::fs::remove_dir_all(&db_path);
    let _ = std::fs::remove_dir_all(&shard_db_path);
    Ok(())
}
