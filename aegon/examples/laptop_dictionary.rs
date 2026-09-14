// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! A small Aegon dictionary on one machine, followed across three epochs:
//!
//! * **Epoch 1.** Alice and Bob join. A client looks Alice up and checks
//!   the proof.
//! * **Epoch 2.** Bob changes his key. An auditor who has been watching
//!   checks the chain one epoch at a time (the classic audit).
//! * **Epoch 3.** Alice changes her key. An auditor who was offline the
//!   whole time checks every epoch at once, with a single recursive proof
//!   (the IVC audit).
//!
//! ```text
//! cargo run --release -p aegon --features ivc_audit --example laptop_dictionary
//! ```

use std::sync::Arc;

use aegon::audit_fs::AuditFs;
use aegon::ivc::adapter::{
    epoch_commitments, epoch_sigma_witnesses, fs_params_from_epoch, hooks_for,
};
use aegon::ivc::prover::{IvcAuditParams, IvcAuditProver};
use aegon::ivc::verifier::verify_against_merkle_root;
use aegon::{
    merkle_root, verify_sharded_invariance, verify_sharded_lookup_two_layer, DbSource, Sha256Hash,
    ShardedAegon, ShardedAegonConfig, ShardedAuditState, SrsSource,
};
use aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::{Bn254, Fr};

/// The polynomial commitment scheme: KZH-k over the BN254 curve.
type Pcs = KZHK<Bn254>;
/// The server. `Sha256Hash` decides which slot each user lands in.
type Server = ShardedAegon<Bn254, Pcs, Sha256Hash>;

/// log2 of how many users the dictionary holds: 8 means 256 users. The
/// shard gets 4x as many slots (1,024), which keeps placement cheap.
const LOG_CAPACITY: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // =====================================================================
    // Setup
    // =====================================================================

    // Start from an empty store on every run. The server keeps users'
    // values here; its shard keeps its own store next to it.
    let db_path = std::env::temp_dir().join("aegon-laptop-dictionary");
    let shard_db_path = db_path.with_extension("shards");
    let _ = std::fs::remove_dir_all(&db_path);
    let _ = std::fs::remove_dir_all(&shard_db_path);

    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .log_capacity(LOG_CAPACITY)
        // 2^0 = one shard.
        .log_n_shards(0)
        // Private mode hides users' values from auditors, and gives every
        // epoch the proof the IVC audit folds.
        .private(true)
        // The hash auditors recompute. IVC auditing requires Poseidon, and
        // the choice is fixed for the life of the dictionary.
        .audit_fs(hooks_for(AuditFs::Poseidon))
        .db(DbSource::Rocks(db_path.clone()))
        // A throwaway trusted setup generated from this seed. Fine for a
        // demo; a real deployment loads one produced by a setup ceremony.
        .srs(SrsSource::DangerouslyGenerate { seed: 42 })
        .build()?;

    let mut server = Server::setup(&cfg)?;

    // Everything a client or auditor needs: the public verification
    // context, and the commitment the server publishes at each epoch.
    // Nobody but the server ever touches `server`.
    let ctx = server.sharded_verifier_context();
    let mut published = vec![server.epoch_commitment(0).ok_or("no epoch 0")?];
    println!("epoch 0: empty dictionary");

    // =====================================================================
    // Epoch 1: Alice and Bob join; a client looks Alice up
    // =====================================================================

    let epoch_1 = server.publish_two_layer(&[
        (b"alice".to_vec(), b"alice-key-1".to_vec()),
        (b"bob".to_vec(), b"bob-key-1".to_vec()),
    ])?;
    published.push(epoch_1.clone());
    println!("\nepoch 1: alice and bob join");

    // The server answers with Alice's value and a proof...
    let alice = b"alice".to_vec();
    let (value, proof) = server.lookup_two_layer(&alice)?;

    // ...and the client checks that proof against epoch 1's published
    // commitment. If the server lied about the value, this fails.
    let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
        &ctx, &epoch_1, &alice, &value, &proof,
    )?;
    println!(
        "  client looks up alice -> {:?}, proof verifies: {ok}",
        String::from_utf8_lossy(&value)
    );

    // =====================================================================
    // Epoch 2: Bob changes his key; the classic audit
    // =====================================================================

    let epoch_2 = server.publish_two_layer(&[(b"bob".to_vec(), b"bob-key-2".to_vec())])?;
    published.push(epoch_2);
    println!("\nepoch 2: bob changes his key");

    // The classic auditor checks one transition at a time, reading only
    // the published commitments. It carries a small running state from
    // each check to the next, so it has to start at epoch 0 and see
    // every transition in order.
    let mut audit_state = ShardedAuditState::<Fr>::default();
    for pair in published.windows(2) {
        let ok =
            verify_sharded_invariance::<Bn254, Pcs>(&ctx, &mut audit_state, &pair[0], &pair[1])?;
        println!(
            "  auditor checks epoch {} -> {}: {ok}",
            pair[0].epoch, pair[1].epoch
        );
    }

    // =====================================================================
    // Epoch 3: Alice changes her key; the IVC audit
    // =====================================================================

    let epoch_3 = server.publish_two_layer(&[(alice.clone(), b"alice-key-2".to_vec())])?;
    published.push(epoch_3);
    println!("\nepoch 3: alice changes her key");

    // An auditor who missed all three epochs would normally have to
    // replay them one by one. Instead, a prover folds every transition
    // into one recursive proof. The prover needs no secrets, only the
    // published commitments, so anyone can run it.
    let genesis = &published[0];
    let latest = published.last().ok_or("nothing published")?;

    // One-time setup of the recursive proof system for this dictionary's
    // shape. `h` comes from the public verification context.
    let h = ctx.inner.verifier_param.get_h();
    let ivc = Arc::new(IvcAuditParams::setup(
        fs_params_from_epoch(&published[1])?,
        h,
    )?);

    let mut prover = IvcAuditProver::new(ivc.clone(), &epoch_commitments(genesis))?;
    for epoch in &published[1..] {
        // Each fold absorbs one epoch's commitments and its blinding proof.
        prover.fold_epoch(&epoch_commitments(epoch), &epoch_sigma_witnesses(epoch)?)?;
    }
    println!(
        "  prover folds all {} transitions into one proof",
        prover.num_steps()
    );

    // The offline auditor makes one call. It checks the proof, then that
    // the proof ends at the commitment the server published for epoch 3.
    let verified = verify_against_merkle_root(
        &ivc,
        prover.proof().ok_or("nothing folded")?,
        prover.num_steps(),
        prover.z0(),
        &epoch_commitments(latest),
        latest.merkle_root,
        || merkle_root(&latest.per_shard),
    )?;
    println!(
        "  offline auditor verifies one proof covering {} epochs: true",
        verified.epochs
    );

    // Clean up the store.
    drop(server);
    let _ = std::fs::remove_dir_all(&db_path);
    let _ = std::fs::remove_dir_all(&shard_db_path);
    Ok(())
}
