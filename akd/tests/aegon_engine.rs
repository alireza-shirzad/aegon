//! End-to-end smoke tests for Aegon over KZH-k.
//!
//! Three layers exercised:
//!   - Lookup: server.publish + server.lookup + verify_lookup
//!   - Auditor: verify_invariance over the chain emitted by publish
//!   - User consistency: server.consistency_proof + verify_consistency
//!
//! Configuration is centralised through `AegonConfig` (built via the
//! KZH-k preset). Everything verifier-side is bundled into the
//! `VerifierContext` the server hands out via `verifier_context()`.

use akd::aegon::presets;
use akd::aegon::{
    verify_consistency, verify_invariance, verify_lookup, Aegon, AegonConfig, AuditState,
    InvarianceProof, Sha256Hash, VerifierContext,
};
use ark_bn254::Bn254;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;

type Pcs = KZHK<Bn254>;
type AegonKzh = Aegon<Bn254, Pcs, Sha256Hash>;

const LOG_CAPACITY: usize = 6;
const KZH_K: usize = 2;
const PRIVATE: bool = false;

fn config() -> AegonConfig<Bn254, Pcs> {
    presets::kzh::<Bn254>(LOG_CAPACITY, KZH_K, PRIVATE)
}

fn fresh_aegon() -> AegonKzh {
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56504);
    AegonKzh::setup(&mut rng, &config()).expect("setup")
}

fn fresh_private_aegon() -> AegonKzh {
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56505);
    AegonKzh::setup(&mut rng, &presets::kzh::<Bn254>(LOG_CAPACITY, KZH_K, true)).expect("setup")
}

// --------------------------------------------------------------------
// Config sanity
// --------------------------------------------------------------------

#[test]
fn config_exposes_user_choices_only() {
    // User-visible config: log_capacity + private + pcs_config.
    // PCS-internal block layout is queried at init time, not
    // configured here.
    let c = config();
    assert_eq!(c.log_capacity, LOG_CAPACITY);
    assert_eq!(c.private, PRIVATE);
    assert_eq!(c.dictionary_capacity(), 1u64 << LOG_CAPACITY);
}

#[test]
fn private_flag_is_threaded_into_kzh_zk() {
    // The kzh preset must propagate `private` into the backend's
    // own zk flag — otherwise Aegon::init's runtime cross-check
    // would refuse to start.
    let c_public = presets::kzh::<Bn254>(LOG_CAPACITY, KZH_K, false);
    assert!(!c_public.private);

    let c_private = presets::kzh::<Bn254>(LOG_CAPACITY, KZH_K, true);
    assert!(c_private.private);
}

#[test]
fn init_rejects_inconsistent_private_flag() {
    // Manually building an AegonConfig with `private = true` and a
    // non-zk pcs_config must be rejected at init time. This is the
    // common foot-gun the cross-check defends against.
    use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
    let bad = AegonConfig::<Bn254, Pcs>::new(
        LOG_CAPACITY,
        true,                            // user says: private
        KZHKConfig::new(KZH_K, false),   // PCS says: non-zk
    );
    let mut rng = ChaCha20Rng::seed_from_u64(0xDEADBEEF);
    let result = AegonKzh::setup(&mut rng, &bad);
    assert!(matches!(result, Err(akd::aegon::AegonError::Config(_))));
}

// --------------------------------------------------------------------
// Lookup path
// --------------------------------------------------------------------

#[test]
fn publish_lookup_verify_happy_path() {
    let mut server = fresh_aegon();
    let updates = vec![
        (b"alice".to_vec(), b"alice-key-v1".to_vec()),
        (b"bob".to_vec(), b"bob-key-v1".to_vec()),
        (b"carol".to_vec(), b"carol-key-v1".to_vec()),
    ];

    let (commitment, _audit) = server.publish(&updates).expect("publish");
    assert_eq!(commitment.epoch, 1);

    let ctx: VerifierContext<Bn254, Pcs> = server.verifier_context();
    for (label, value) in &updates {
        let proof = server.lookup(label).expect("lookup");
        let ok = verify_lookup::<Bn254, Pcs, Sha256Hash>(&ctx, &commitment, label, value, &proof)
            .expect("verify_lookup runs");
        assert!(ok, "honest proof must verify for label {label:?}");
    }
}

#[test]
fn verify_rejects_wrong_value() {
    let mut server = fresh_aegon();
    let label = b"alice".to_vec();
    let (commitment, _audit) = server
        .publish(&[(label.clone(), b"alice-key-v1".to_vec())])
        .expect("publish");

    let proof = server.lookup(&label).expect("lookup");
    let result = verify_lookup::<Bn254, Pcs, Sha256Hash>(
        &server.verifier_context(),
        &commitment,
        &label,
        &b"different-value".to_vec(),
        &proof,
    );
    assert!(matches!(result, Err(akd::aegon::AegonError::Verification(_))));
}

#[test]
fn lookup_unknown_label_errors() {
    let mut server = fresh_aegon();
    let _ = server.publish(&[]).expect("empty publish");
    let result = server.lookup(&b"never-registered".to_vec());
    assert!(matches!(result, Err(akd::aegon::AegonError::UnknownLabel(_))));
}

#[test]
fn republish_updates_value_in_place() {
    let mut server = fresh_aegon();
    let label = b"alice".to_vec();
    let _ = server
        .publish(&[(label.clone(), b"v1".to_vec())])
        .expect("publish v1");
    let (commitment_v2, _audit) = server
        .publish(&[(label.clone(), b"v2".to_vec())])
        .expect("publish v2");

    let ctx = server.verifier_context();
    let proof = server.lookup(&label).expect("lookup v2");
    let ok = verify_lookup::<Bn254, Pcs, Sha256Hash>(
        &ctx,
        &commitment_v2,
        &label,
        &b"v2".to_vec(),
        &proof,
    )
    .expect("verify");
    assert!(ok);

    let result = verify_lookup::<Bn254, Pcs, Sha256Hash>(
        &ctx,
        &commitment_v2,
        &label,
        &b"v1".to_vec(),
        &proof,
    );
    assert!(matches!(result, Err(akd::aegon::AegonError::Verification(_))));
}

// --------------------------------------------------------------------
// Auditor path
// --------------------------------------------------------------------

#[test]
fn verify_invariance_accepts_honest_chain() {
    let mut server = fresh_aegon();
    let prev0 = server.current_commitment();
    let ctx = server.verifier_context();
    let mut audit_state = AuditState::<<Bn254 as ark_ec::pairing::Pairing>::ScalarField>::default();

    let (com1, audit1) = server.publish(&[]).expect("publish empty");
    let ok = verify_invariance::<Bn254, Pcs>(&ctx, &mut audit_state, &prev0, &com1, &audit1)
        .expect("audit 0->1");
    assert!(ok);

    let (com2, audit2) = server
        .publish(&[
            (b"alice".to_vec(), b"a1".to_vec()),
            (b"bob".to_vec(), b"b1".to_vec()),
        ])
        .expect("publish two labels");
    let ok = verify_invariance::<Bn254, Pcs>(&ctx, &mut audit_state, &com1, &com2, &audit2)
        .expect("audit 1->2");
    assert!(ok);

    let (com3, audit3) = server
        .publish(&[(b"alice".to_vec(), b"a2".to_vec())])
        .expect("publish update");
    let ok = verify_invariance::<Bn254, Pcs>(&ctx, &mut audit_state, &com2, &com3, &audit3)
        .expect("audit 2->3");
    assert!(ok);
}

#[test]
fn verify_invariance_rejects_tampered_proof() {
    let mut server = fresh_aegon();
    let prev0 = server.current_commitment();
    let (com1, mut audit1) = server
        .publish(&[(b"alice".to_vec(), b"a1".to_vec())])
        .expect("publish");

    use ark_ff::UniformRand;
    let mut rng = ChaCha20Rng::seed_from_u64(0xBADBADBA);
    audit1.index_chain.next_rand_eval +=
        <Bn254 as ark_ec::pairing::Pairing>::ScalarField::rand(&mut rng);

    let mut audit_state = AuditState::<<Bn254 as ark_ec::pairing::Pairing>::ScalarField>::default();
    let result = verify_invariance::<Bn254, Pcs>(
        &server.verifier_context(),
        &mut audit_state,
        &prev0,
        &com1,
        &audit1,
    );
    assert!(matches!(result, Ok(false)));
}

#[test]
fn verify_invariance_rejects_skipped_epoch() {
    let mut server = fresh_aegon();
    let prev0 = server.current_commitment();
    let _ = server.publish(&[]).expect("e1");
    let (com2, _) = server.publish(&[]).expect("e2");
    let mut audit_state = AuditState::<<Bn254 as ark_ec::pairing::Pairing>::ScalarField>::default();
    let result = verify_invariance::<Bn254, Pcs>(
        &server.verifier_context(),
        &mut audit_state,
        &prev0,
        &com2,
        &dummy_invariance_proof(),
    );
    assert!(matches!(result, Err(akd::aegon::AegonError::Verification(_))));
}

fn dummy_invariance_proof() -> InvarianceProof<Bn254, Pcs> {
    let mut server = fresh_aegon();
    let (_, p) = server.publish(&[]).expect("dummy publish");
    p
}

// --------------------------------------------------------------------
// User consistency path
// --------------------------------------------------------------------

#[test]
fn verify_consistency_accepts_static_value_and_index() {
    let mut server = fresh_aegon();
    let label = b"alice".to_vec();
    let (s0_commit, _) = server
        .publish(&[(label.clone(), b"a1".to_vec())])
        .expect("e1");
    let _ = server
        .publish(&[(b"bob".to_vec(), b"b1".to_vec())])
        .expect("e2");
    let (s1_commit, _) = server
        .publish(&[(b"carol".to_vec(), b"c1".to_vec())])
        .expect("e3");

    let lookup_at_s0_proof = server.lookup(&label).expect("lookup");
    let expected_ctr0 = lookup_at_s0_proof.ctr0;

    let proof = server
        .consistency_proof(&label, s0_commit.epoch)
        .expect("consistency proof");

    let ok = verify_consistency::<Bn254, Pcs, Sha256Hash>(
        &server.verifier_context(),
        &s0_commit,
        &s1_commit,
        &label,
        expected_ctr0,
        &proof,
    )
    .expect("verify_consistency");
    assert!(ok);
}

#[test]
fn verify_consistency_rejects_changed_value() {
    let mut server = fresh_aegon();
    let label = b"alice".to_vec();
    let (s0_commit, _) = server
        .publish(&[(label.clone(), b"a1".to_vec())])
        .expect("e1");
    let lookup_at_s0 = server.lookup(&label).expect("lookup s0");
    let expected_ctr0 = lookup_at_s0.ctr0;

    let (s1_commit, _) = server
        .publish(&[(label.clone(), b"a2".to_vec())])
        .expect("e2");

    let proof = server
        .consistency_proof(&label, s0_commit.epoch)
        .expect("consistency proof");
    let ok = verify_consistency::<Bn254, Pcs, Sha256Hash>(
        &server.verifier_context(),
        &s0_commit,
        &s1_commit,
        &label,
        expected_ctr0,
        &proof,
    )
    .expect("verify runs");
    assert!(!ok, "consistency must reject when the value changed");
}

#[test]
fn verify_consistency_rejects_wrong_expected_ctr0() {
    let mut server = fresh_aegon();
    let label = b"alice".to_vec();
    let (s0_commit, _) = server
        .publish(&[(label.clone(), b"a1".to_vec())])
        .expect("e1");
    let (s1_commit, _) = server.publish(&[]).expect("e2");
    let lookup = server.lookup(&label).expect("lookup");
    let proof = server
        .consistency_proof(&label, s0_commit.epoch)
        .expect("consistency");

    let bogus = lookup.ctr0.wrapping_add(1);
    let result = verify_consistency::<Bn254, Pcs, Sha256Hash>(
        &server.verifier_context(),
        &s0_commit,
        &s1_commit,
        &label,
        bogus,
        &proof,
    );
    assert!(matches!(result, Err(akd::aegon::AegonError::Verification(_))));
}

// --------------------------------------------------------------------
// Private (zk) end-to-end
// --------------------------------------------------------------------

#[test]
fn private_mode_lookup_and_audit_roundtrip() {
    // Same flow as the public happy path, but with `private = true`.
    // Confirms the zk-KZH path is wired through every layer.
    let mut server = fresh_private_aegon();
    let label = b"alice".to_vec();
    let value = b"alice-key-v1".to_vec();
    let prev = server.current_commitment();

    let (commitment, audit) = server
        .publish(&[(label.clone(), value.clone())])
        .expect("private publish");
    let ctx = server.verifier_context();

    // Lookup
    let proof = server.lookup(&label).expect("private lookup");
    let ok = verify_lookup::<Bn254, Pcs, Sha256Hash>(&ctx, &commitment, &label, &value, &proof)
        .expect("verify_lookup");
    assert!(ok);

    // Auditor invariance
    let mut audit_state = AuditState::<<Bn254 as ark_ec::pairing::Pairing>::ScalarField>::default();
    let ok =
        verify_invariance::<Bn254, Pcs>(&ctx, &mut audit_state, &prev, &commitment, &audit).expect("audit");
    assert!(ok);
}

// --------------------------------------------------------------------
// End-to-end: two signup batches separated by idle epochs, followed by
// lookups + per-user audits. Every value the verifier receives is
// checked against the value the test put in.
// --------------------------------------------------------------------

#[test]
fn end_to_end_two_batches_with_idle_epochs() {
    let mut server = fresh_aegon();
    let ctx = server.verifier_context();
    let prev0 = server.current_commitment();
    let mut audit_state =
        AuditState::<<Bn254 as ark_ec::pairing::Pairing>::ScalarField>::default();

    let batch1: Vec<(Vec<u8>, Vec<u8>)> = (0..5u32)
        .map(|i| (format!("alice-{i}").into_bytes(), format!("alice-key-{i}").into_bytes()))
        .collect();
    let batch2: Vec<(Vec<u8>, Vec<u8>)> = (0..5u32)
        .map(|i| (format!("bob-{i}").into_bytes(), format!("bob-key-{i}").into_bytes()))
        .collect();

    // Epoch 1: batch1 signs up.
    let (com1, audit1) = server.publish(&batch1).expect("publish batch1");
    assert!(
        verify_invariance::<Bn254, Pcs>(&ctx, &mut audit_state, &prev0, &com1, &audit1)
            .expect("audit 0->1"),
        "auditor must accept honest 0->1 transition",
    );

    // Epoch 2: an idle epoch passes (no signups, no value updates).
    let (com2, audit2) = server.publish(&[]).expect("idle epoch 1->2");
    assert!(
        verify_invariance::<Bn254, Pcs>(&ctx, &mut audit_state, &com1, &com2, &audit2)
            .expect("audit 1->2"),
        "auditor must accept honest idle 1->2 transition",
    );

    // Epoch 3: batch2 signs up.
    let (com3, audit3) = server.publish(&batch2).expect("publish batch2");
    assert!(
        verify_invariance::<Bn254, Pcs>(&ctx, &mut audit_state, &com2, &com3, &audit3)
            .expect("audit 2->3"),
        "auditor must accept honest 2->3 transition",
    );

    // Epoch 4: another idle epoch passes.
    let (com4, audit4) = server.publish(&[]).expect("idle epoch 3->4");
    assert!(
        verify_invariance::<Bn254, Pcs>(&ctx, &mut audit_state, &com3, &com4, &audit4)
            .expect("audit 3->4"),
        "auditor must accept honest idle 3->4 transition",
    );
    assert_eq!(com4.epoch, 4);

    // Lookup phase: every user from both batches looks up their value
    // against the current epoch. verify_lookup re-checks h_f(value)
    // against the opening, so a successful verify *is* the assertion
    // that the returned data matches what was published.
    for (label, expected_value) in batch1.iter().chain(batch2.iter()) {
        let proof = server.lookup(label).expect("lookup at epoch 4");
        let ok = verify_lookup::<Bn254, Pcs, Sha256Hash>(
            &ctx,
            &com4,
            label,
            expected_value,
            &proof,
        )
        .expect("verify_lookup");
        assert!(ok, "lookup must verify for {label:?} at epoch 4");

        // Cross-check: a wrong value must be rejected. Guards against a
        // verifier that accepts unconditionally.
        let mut tampered = expected_value.clone();
        tampered.push(0xFF);
        let bad = verify_lookup::<Bn254, Pcs, Sha256Hash>(
            &ctx,
            &com4,
            label,
            &tampered,
            &proof,
        );
        assert!(
            matches!(bad, Err(akd::aegon::AegonError::Verification(_))),
            "wrong-value lookup must be rejected for {label:?}",
        );
    }

    // Per-user audits: each user proves their (slot, value) was unchanged
    // between the epoch they signed up in and the current epoch.
    // batch1 spans epoch1 -> epoch4 (three transitions, including one
    // signup and two idle); batch2 spans epoch3 -> epoch4 (one idle).
    for (label, _) in batch1.iter() {
        let lookup = server.lookup(label).expect("lookup for ctr0");
        let expected_ctr0 = lookup.ctr0;
        let s0_commit = server.epoch_commitment(1).expect("epoch 1 retained");
        let proof = server
            .consistency_proof(label, 1)
            .expect("consistency batch1");
        let ok = verify_consistency::<Bn254, Pcs, Sha256Hash>(
            &ctx,
            &s0_commit,
            &com4,
            label,
            expected_ctr0,
            &proof,
        )
        .expect("verify_consistency batch1");
        assert!(ok, "batch1 consistency must hold for {label:?}");
    }

    for (label, _) in batch2.iter() {
        let lookup = server.lookup(label).expect("lookup for ctr0");
        let expected_ctr0 = lookup.ctr0;
        let s0_commit = server.epoch_commitment(3).expect("epoch 3 retained");
        let proof = server
            .consistency_proof(label, 3)
            .expect("consistency batch2");
        let ok = verify_consistency::<Bn254, Pcs, Sha256Hash>(
            &ctx,
            &s0_commit,
            &com4,
            label,
            expected_ctr0,
            &proof,
        )
        .expect("verify_consistency batch2");
        assert!(ok, "batch2 consistency must hold for {label:?}");
    }
}

#[test]
fn consistency_proof_unknown_epoch_errors() {
    let mut server = fresh_aegon();
    let _ = server
        .publish(&[(b"alice".to_vec(), b"a1".to_vec())])
        .expect("e1");
    let result = server.consistency_proof(&b"alice".to_vec(), 999);
    assert!(matches!(result, Err(akd::aegon::AegonError::InvalidEpoch(_))));
}
