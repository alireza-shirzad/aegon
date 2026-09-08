// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! End-to-end Aegon flow with ECVRF wired in.
//!
//! Spins up an in-process ShardedAegon with `EcVrfHash`, attaches a
//! VRF prover (so `lookup_label` populates `vrf_proof`), publishes a
//! handful of labels, looks each one up, and verifies the proofs
//! using a verifier-side context whose `vrf_verifier` is the
//! corresponding public key.
//!
//! This exercises the exact code path a production gRPC deployment
//! will follow once the proto fields are added:
//!   * server side: `lookup_label` calls `VrfProver::prove_h_bits`
//!     per probe and stuffs the 80-byte proof into `ShardedProbe.vrf_proof`;
//!   * client side: `verify_lookup_label` consumes that proof via
//!     `VrfVerifier::verify_h_bits` to recover slot bits, instead of
//!     re-hashing the label locally.
//!
//! Comparison case: also runs the same flow with `Sha256Hash` and
//! confirms it still verifies (the legacy path), so the two suites
//! coexist cleanly.

use akd::aegon::{
    verify_lookup_label_two_layer, verify_lookup_value, DbSource, EcVrfHash, Sha256Hash,
    ShardTransport, ShardedAegon, ShardedAegonConfig, VrfProver, BENCH_VRF_SEED,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;

fn build_cfg(log_capacity: usize, log_n_shards: usize) -> ShardedAegonConfig<Bn254, Pcs> {
    ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(log_capacity - log_n_shards)
        .log_n_shards(log_n_shards)
        .private(false)
        .kzh_k(2)
        .shards(ShardTransport::InProcess)
        .db(DbSource::None)
        .build()
        .expect("config builds")
}

fn run_ecvrf_path() {
    println!("\n── ECVRF path ──");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA5A5);
    let cfg = build_cfg(12, 1);
    let mut server: ShardedAegon<Bn254, Pcs, EcVrfHash> =
        ShardedAegon::<Bn254, Pcs, EcVrfHash>::setup(&mut rng, &cfg)
            .expect("ShardedAegon::setup with EcVrfHash");

    // Wire the prover. Use the deterministic bench seed so the
    // verifier-side can construct a matching public key from the
    // server's `sharded_verifier_context()`.
    server.set_vrf_prover(VrfProver::from_seed(&BENCH_VRF_SEED));
    println!("  set VrfProver — server-side prove path active.");

    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..6)
        .map(|i| {
            (
                format!("alice-{i}").into_bytes(),
                format!("pk-{i}").into_bytes(),
            )
        })
        .collect();
    let commit = server
        .publish_two_layer(&updates)
        .expect("publish_two_layer");
    println!(
        "  published {} labels @ epoch {}.",
        updates.len(),
        commit.epoch
    );

    let ctx = server.sharded_verifier_context();
    assert!(
        ctx.vrf_verifier.is_some(),
        "verifier context carries vrf_verifier when prover is attached"
    );

    let mut verified = 0usize;
    for (label, value) in &updates {
        let (_db_value, _full_proof) = server.lookup_two_layer(label).expect("lookup_two_layer");
        let (slot, label_proof) = server
            .lookup_label_two_layer(label)
            .expect("lookup_label_two_layer");
        // Sanity: every probe carries an 80-byte VRF proof — both
        // the inter-shard route trail and the intra-shard slot trail.
        for (i, p) in label_proof.route.iter().enumerate() {
            assert_eq!(
                p.vrf_proof.len(),
                80,
                "route[{i}] for {label:?}: vrf_proof length should be 80 (RFC 9381)"
            );
        }
        for (i, p) in label_proof.slots.iter().enumerate() {
            assert_eq!(
                p.vrf_proof.len(),
                80,
                "slot[{i}] for {label:?}: vrf_proof length should be 80 (RFC 9381)"
            );
        }
        let recovered = verify_lookup_label_two_layer::<Bn254, Pcs, EcVrfHash>(
            &ctx,
            &commit,
            label,
            &label_proof,
        )
        .expect("verify_lookup_label_two_layer accepts honest VRF proofs");
        assert_eq!(recovered, slot, "recovered slot must match server's slot");
        let value_proof = server.lookup_value(&slot).expect("lookup_value");
        let ok = verify_lookup_value::<Bn254, Pcs, EcVrfHash>(
            &ctx,
            &commit,
            &recovered,
            value,
            &value_proof,
        )
        .expect("verify_lookup_value");
        assert!(ok, "value verify must accept the published value");
        verified += 1;
    }
    println!(
        "  verified {verified}/{} label+value openings (ECVRF).",
        updates.len()
    );

    // ---- Negative test: a verifier with the wrong public key
    // rejects honest proofs from the legitimate server. The
    // two-layer routing model couples the COORD's per-instance
    // prover to the SHARD's deployment-wide `H::h_slot` key (they
    // must match for publish/lookup to round-trip), so the negative
    // case lives on the VERIFIER side: an unrelated public key
    // can't validate any honest VRF proof.
    use akd::aegon::VrfVerifier;
    let (label, _value) = &updates[0];
    let (_, honest_proof) = server
        .lookup_label_two_layer(label)
        .expect("lookup_label_two_layer honest");
    let wrong_prover = VrfProver::from_seed(b"a-totally-different-32-byte-seed");
    let mut wrong_ctx = ctx.clone();
    wrong_ctx.vrf_verifier = Some(VrfVerifier::new(wrong_prover.public_key().clone()));
    let res = verify_lookup_label_two_layer::<Bn254, Pcs, EcVrfHash>(
        &wrong_ctx,
        &commit,
        label,
        &honest_proof,
    );
    assert!(
        res.is_err(),
        "honest proofs MUST be rejected under a different verifier key, got {:?}",
        res
    );
    println!("  honest proof under a wrong-key verifier: REJECTED ✓");
}

fn run_sha256_path() {
    println!("\n── SHA-256 path ──");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA5A5);
    let cfg = build_cfg(12, 1);
    let mut server: ShardedAegon<Bn254, Pcs, Sha256Hash> =
        ShardedAegon::<Bn254, Pcs, Sha256Hash>::setup(&mut rng, &cfg).expect("setup");

    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..6)
        .map(|i| {
            (
                format!("bob-{i}").into_bytes(),
                format!("pk-{i}").into_bytes(),
            )
        })
        .collect();
    let commit = server
        .publish_two_layer(&updates)
        .expect("publish_two_layer");

    let ctx = server.sharded_verifier_context();
    assert!(
        ctx.vrf_verifier.is_none(),
        "Sha256Hash path: verifier context must NOT carry a vrf_verifier"
    );

    for (label, value) in &updates {
        let (slot, label_proof) = server
            .lookup_label_two_layer(label)
            .expect("lookup_label_two_layer");
        // SHA-256 deployment: bits are publicly computable, so no
        // VRF proofs ride either trail.
        for p in &label_proof.route {
            assert!(
                p.vrf_proof.is_empty(),
                "Sha256Hash path: route vrf_proof must be empty"
            );
        }
        for p in &label_proof.slots {
            assert!(
                p.vrf_proof.is_empty(),
                "Sha256Hash path: slot vrf_proof must be empty"
            );
        }
        let recovered = verify_lookup_label_two_layer::<Bn254, Pcs, Sha256Hash>(
            &ctx,
            &commit,
            label,
            &label_proof,
        )
        .expect("verify_lookup_label_two_layer");
        let value_proof = server.lookup_value(&slot).expect("lookup_value");
        let ok = verify_lookup_value::<Bn254, Pcs, Sha256Hash>(
            &ctx,
            &commit,
            &recovered,
            value,
            &value_proof,
        )
        .expect("verify_lookup_value");
        assert!(ok);
    }
    println!(
        "  verified {}/{} label+value openings (SHA-256).",
        updates.len(),
        updates.len()
    );
}

fn main() {
    println!("=== Aegon ECVRF end-to-end with ShardedAegon ===");
    run_ecvrf_path();
    run_sha256_path();
    println!("\nBoth suites verified successfully. ECVRF wiring is production-shaped.");
}
