//! End-to-end exercise of the audit-path sigma protocol (§7).
//!
//! Confirms three things on the zk publish/audit path:
//!
//! 1. A zk-mode publish produces a `audit_value_blinding_proof` on
//!    every per-shard `EpochCommitment`.
//! 2. An honest audit transition (`verify_sharded_invariance` on the
//!    prev/next pair, with the same `vk`) accepts.
//! 3. Negative paths: tampering with `r_value` or stripping the
//!    proof in zk mode both cause the audit to reject.
//!
//! Mirror case: same flow against a NON-hiding SRS (`private =
//! false`) where the proof slot stays `None` and the bare group
//! equation is what the auditor checks. Confirms the non-zk path
//! still works after wiring the optional proof through every
//! construction site.
//!
//! Why this lives as an example rather than `cargo test`: the
//! workspace's `akd --lib --tests` path is currently blocked on a
//! pre-existing `PublishCorruption` / `publish_malicious_update`
//! breakage in `tests/test_errors.rs` (reproduces on `main` without
//! any of this branch's changes). Running the integration here keeps
//! the audit-path sigma covered by a real publish+verify loop without
//! having to fix the unrelated test_errors gating.

use akd::aegon::{
    verify_sharded_invariance, ShardedAuditState, DbSource, EcVrfHash, ShardTransport, ShardedAegon,
    ShardedAegonConfig, VrfProver, BENCH_VRF_SEED,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::{Bn254, Fr};
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;

fn build_cfg(log_capacity: usize, log_n_shards: usize, zk: bool) -> ShardedAegonConfig<Bn254, Pcs> {
    ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(log_capacity - log_n_shards)
        .log_n_shards(log_n_shards)
        .private(zk)
        .kzh_k(2)
        .shards(ShardTransport::InProcess)
        .db(DbSource::None)
        .build()
        .expect("config builds")
}

fn run_zk_path() {
    println!("\n── zk SRS path ──");
    let mut rng = ChaCha20Rng::seed_from_u64(0xAEC0);
    let cfg = build_cfg(12, 1, true);
    let mut server: ShardedAegon<Bn254, Pcs, EcVrfHash> =
        ShardedAegon::<Bn254, Pcs, EcVrfHash>::setup(&mut rng, &cfg).expect("setup zk");
    server.set_vrf_prover(VrfProver::from_seed(&BENCH_VRF_SEED));

    let ctx = server.sharded_verifier_context();
    let prev = server
        .epoch_commitment(0)
        .expect("epoch-0 commit retained");

    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..6)
        .map(|i| (format!("alice-{i}").into_bytes(), format!("pk-{i}").into_bytes()))
        .collect();
    let next = server.publish_two_layer(&updates).expect("publish_two_layer zk");
    println!("  published epoch {}.", next.epoch);

    // (1) Each per-shard leaf must carry a sigma proof under hiding SRS.
    for (i, shard_commit) in next.per_shard.iter().enumerate() {
        assert!(
            shard_commit.audit_value_blinding_proof.is_some(),
            "zk shard {i} must carry a sigma proof on the value chain",
        );
    }
    println!("  ✓ every per-shard EpochCommitment has Some(proof).");

    // (2) Honest audit transition accepts.
    {
        let mut audit_state = ShardedAuditState::<Fr>::default();
        let ok = verify_sharded_invariance(&ctx, &mut audit_state, &prev, &next).expect("audit");
        assert!(ok, "honest zk transition must pass the audit");
    }
    println!("  ✓ honest verify_sharded_invariance accepts.");

    // (3a) Tampering: swap one shard's value_commitment with another
    // shard's. The Schnorr transcript binds the new (wrong)
    // value_commitment, so the challenge `e` no longer matches what
    // the proof was generated against, and `s·h ≠ R + e·residue`.
    {
        let mut tampered = next.clone();
        let swapped = tampered.per_shard[1].value_commitment.clone();
        tampered.per_shard[0].value_commitment = swapped;
        // Rebuild merkle root so the structural pre-check passes and
        // the audit actually reaches verify_chain.
        tampered.merkle_root = akd::aegon::merkle_root(&tampered.per_shard);
        let mut audit_state = ShardedAuditState::<Fr>::default();
        let ok = verify_sharded_invariance(&ctx, &mut audit_state, &prev, &tampered)
            .expect("audit");
        assert!(!ok, "tampered value_commitment must trip the sigma check");
    }
    println!("  ✓ tampered next.value_commitment is rejected.");

    // (3b) Strip the proof and confirm the policy gate rejects in zk
    // mode (paper §7: zk SRS REQUIRES the sigma proof — a missing
    // proof signals the server skipped re-randomisation).
    {
        let mut stripped = next.clone();
        for shard_commit in stripped.per_shard.iter_mut() {
            shard_commit.audit_value_blinding_proof = None;
        }
        stripped.merkle_root = akd::aegon::merkle_root(&stripped.per_shard);
        let mut audit_state = ShardedAuditState::<Fr>::default();
        let ok = verify_sharded_invariance(&ctx, &mut audit_state, &prev, &stripped)
            .expect("audit");
        assert!(!ok, "missing sigma proof in zk mode must be rejected");
    }
    println!("  ✓ missing proof in zk mode is rejected (policy gate).");
}

fn run_non_zk_path() {
    println!("\n── non-zk SRS path ──");
    let mut rng = ChaCha20Rng::seed_from_u64(0xAEC1);
    let cfg = build_cfg(12, 1, false);
    let mut server: ShardedAegon<Bn254, Pcs, EcVrfHash> =
        ShardedAegon::<Bn254, Pcs, EcVrfHash>::setup(&mut rng, &cfg).expect("setup non-zk");
    server.set_vrf_prover(VrfProver::from_seed(&BENCH_VRF_SEED));

    let ctx = server.sharded_verifier_context();
    let prev = server.epoch_commitment(0).expect("epoch 0");
    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..6)
        .map(|i| (format!("bob-{i}").into_bytes(), format!("pk-{i}").into_bytes()))
        .collect();
    let next = server.publish_two_layer(&updates).expect("publish_two_layer non-zk");

    for (i, shard_commit) in next.per_shard.iter().enumerate() {
        assert!(
            shard_commit.audit_value_blinding_proof.is_none(),
            "non-zk shard {i} must NOT carry a sigma proof",
        );
    }
    println!("  ✓ every per-shard EpochCommitment has None proof slot.");

    let mut audit_state = ShardedAuditState::<Fr>::default();
    let ok =
        verify_sharded_invariance(&ctx, &mut audit_state, &prev, &next).expect("audit non-zk");
    assert!(ok, "honest non-zk transition must pass the bare chain check");
    println!("  ✓ honest non-zk transition accepts via bare equality.");
}

fn main() {
    run_zk_path();
    run_non_zk_path();
    println!("\nall sigma-audit checks passed.\n");
}
