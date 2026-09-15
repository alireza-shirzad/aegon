// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Every configuration setting in one place, with its default and options
//! in the comments, followed by one of each operation: publish, lookup,
//! history, consistency proof, and audit.
//!
//! Copy this file as the starting point for your own deployment.
//!
//! ```text
//! cargo run --release -p aegon --example configured_dictionary
//! ```

use aegon::ivc::adapter::hooks_for;
use aegon::{
    verify_lookup_history, verify_sharded_consistency_two_layer, verify_sharded_invariance,
    verify_sharded_lookup_two_layer, AuditFs, DbSource, Sha256Hash, ShardTransport, ShardedAegon,
    ShardedAegonConfig, ShardedAuditState, SrsSource,
};
use aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::{Bn254, Fr};

/// The polynomial commitment scheme: KZH-k over BN254.
type Pcs = KZHK<Bn254>;

/// How users are assigned to slots when no VRF key is set (see
/// `.vrf_prover` below): `Sha256Hash`, which anyone can compute, or
/// `aegon::EcVrfHash`, keyed from `AEGON_VRF_SEED` / `AEGON_VRF_KEY_PATH`.
type Hash = Sha256Hash;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Local storage for this demo, wiped so every run starts fresh.
    let db = std::env::temp_dir().join("aegon-template");
    let _ = std::fs::remove_dir_all(&db);
    let _ = std::fs::remove_dir_all(db.with_extension("shards"));

    // Every setting is a builder call taking a plain value, so it can come
    // from code, command-line flags, or a config file. Only `log_capacity`
    // and `log_n_shards` are required; every other call shows its default.
    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        // log2 of how many users the dictionary holds: 10 means 1,024.
        .log_capacity(10)
        // log2 of the shard count: 2 means 4 shards. Capacity is split
        // evenly across them, so more shards means smaller, faster shards.
        .log_n_shards(2)
        // log2 of the slots per unit of capacity. Placement needs the
        // headroom: at 2 (four slots per user) the dictionary is at most a
        // quarter full. Default: 2.
        .log_over_provisioning_factor(2)
        // Private mode: commit with the hiding (zk) variant of KZH-k, mask
        // the openings, and publish the blinding proof each epoch's audit
        // then requires. It hides users' values from auditors, costs
        // proving time, and IVC auditing needs it. Default: false.
        .private(false)
        // The hash auditors recompute: Poseidon (supports IVC auditing) or
        // Sha256. Servers and auditors must agree, and it can't change
        // later. Default: Sha256.
        .audit_fs(hooks_for(AuditFs::Poseidon))
        // Independent audit chains to split the shards into. Must divide the
        // shard count; more groups let an IVC audit fold in parallel.
        // Default: 1.
        .chain_groups(1)
        // Where the shards run. Default: InProcess.
        //   ShardTransport::InProcess                   all in this process
        //   ShardTransport::Remote { endpoints: urls }  one aegon_shard_server per shard
        .shards(ShardTransport::InProcess)
        // The trusted setup (SRS). Default: DangerouslyGenerate { seed: 0 }.
        //   SrsSource::DangerouslyGenerate { seed }  throwaway, from the seed; testing only
        //   SrsSource::Path(file)                    load one produced by a setup ceremony
        .srs(SrsSource::DangerouslyGenerate { seed: 42 })
        // Where users' values and history are stored. Default: None.
        //   DbSource::None        nothing stored: lookups return no value bytes
        //   DbSource::Rocks(dir)  a local RocksDB
        //   DbSource::Redis(url)  a shared Redis (with remote shards only)
        .db(DbSource::Rocks(db))
        // Optional, with no default call shown:
        //   .kzh_k(k)                      commitment-scheme parameter; defaults to the best for the shard size
        //   .vrf_prover(VrfProver::from_seed(&seed))
        //                                  hide which slot each user lands in behind a VRF key;
        //                                  when set, it is used instead of `Hash`
        //   .masking_addrs(vec![url])      private mode: mask proofs on separate aegon_masking_server processes
        .build()?;

    let mut server = ShardedAegon::<Bn254, Pcs, Hash>::setup(&cfg)?;

    // What clients and auditors hold: the public verification context and
    // the commitment published at each epoch.
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
    let mut audit = ShardedAuditState::<Fr>::with_groups(cfg.chain_groups);
    for (prev, next) in [(&epoch_0, &epoch_1), (&epoch_1, &epoch_2)] {
        assert!(verify_sharded_invariance::<Bn254, Pcs>(
            &ctx, &mut audit, prev, next
        )?);
    }

    println!("every check passed");
    Ok(())
}
