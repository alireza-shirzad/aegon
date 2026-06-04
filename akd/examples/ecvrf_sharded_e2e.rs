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
    verify_lookup_label, verify_lookup_value, DbSource, EcVrfHash, Sha256Hash, ShardTransport,
    ShardedAegon, ShardedAegonConfig, VrfProver, BENCH_VRF_SEED,
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
        .map(|i| (format!("alice-{i}").into_bytes(), format!("pk-{i}").into_bytes()))
        .collect();
    let commit = server.publish(&updates).expect("publish");
    println!("  published {} labels @ epoch {}.", updates.len(), commit.epoch);

    let ctx = server.sharded_verifier_context();
    assert!(
        ctx.vrf_verifier.is_some(),
        "verifier context carries vrf_verifier when prover is attached"
    );

    let mut verified = 0usize;
    for (label, value) in &updates {
        let (_db_value, _full_proof) = server.lookup(label).expect("lookup");
        let (slot, label_proof) = server.lookup_label(label).expect("lookup_label");
        // Sanity: every probe carries an 80-byte VRF proof.
        for (i, probe) in label_proof.probes.iter().enumerate() {
            assert_eq!(
                probe.vrf_proof.len(),
                80,
                "probe {i} for {label:?}: vrf_proof length should be 80 (RFC 9381)"
            );
        }
        let recovered =
            verify_lookup_label::<Bn254, Pcs, EcVrfHash>(&ctx, &commit, label, &label_proof)
                .expect("verify_lookup_label accepts honest VRF proofs");
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
    println!("  verified {verified}/{} label+value openings (ECVRF).", updates.len());

    // ---- Negative test: an unrelated key must NOT verify. ----
    let evil_prover = VrfProver::from_seed(b"a-totally-different-32-byte-seed");
    let mut evil_server: ShardedAegon<Bn254, Pcs, EcVrfHash> =
        ShardedAegon::<Bn254, Pcs, EcVrfHash>::setup(
            &mut ChaCha20Rng::seed_from_u64(0xA5A5),
            &cfg,
        )
        .expect("setup evil_server");
    evil_server.set_vrf_prover(evil_prover);
    let _ = evil_server.publish(&updates).expect("publish evil");
    let (_, evil_proof) = evil_server.lookup_label(&updates[0].0).expect("lookup_label evil");
    let res = verify_lookup_label::<Bn254, Pcs, EcVrfHash>(&ctx, &commit, &updates[0].0, &evil_proof);
    assert!(
        matches!(res, Err(_)),
        "proofs from a different VRF key MUST be rejected, got {:?}",
        res
    );
    println!("  proof from a wrong-key prover: REJECTED ✓");
}

fn run_sha256_path() {
    println!("\n── SHA-256 legacy path ──");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA5A5);
    let cfg = build_cfg(12, 1);
    let mut server: ShardedAegon<Bn254, Pcs, Sha256Hash> =
        ShardedAegon::<Bn254, Pcs, Sha256Hash>::setup(&mut rng, &cfg).expect("setup");

    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..6)
        .map(|i| (format!("bob-{i}").into_bytes(), format!("pk-{i}").into_bytes()))
        .collect();
    let commit = server.publish(&updates).expect("publish");

    let ctx = server.sharded_verifier_context();
    assert!(
        ctx.vrf_verifier.is_none(),
        "Sha256Hash path: verifier context must NOT carry a vrf_verifier"
    );

    for (label, value) in &updates {
        let (slot, label_proof) = server.lookup_label(label).expect("lookup_label");
        for probe in &label_proof.probes {
            assert!(
                probe.vrf_proof.is_empty(),
                "Sha256Hash path: probe.vrf_proof must be empty"
            );
        }
        let recovered =
            verify_lookup_label::<Bn254, Pcs, Sha256Hash>(&ctx, &commit, label, &label_proof)
                .expect("verify_lookup_label legacy path");
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
    println!("  verified {}/{} label+value openings (SHA-256).", updates.len(), updates.len());
}

fn main() {
    println!("=== Aegon ECVRF end-to-end with ShardedAegon ===");
    run_ecvrf_path();
    run_sha256_path();
    println!("\nBoth suites verified successfully. ECVRF wiring is production-shaped.");
}
