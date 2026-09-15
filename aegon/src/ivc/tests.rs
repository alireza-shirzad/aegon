// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! End-to-end IVC audit tests: fold a real chain of epochs, then
//! verify it as an auditor would.

use ark_bn254::Fr as ArkFr;
use std::sync::Arc;

use super::circuit::SigmaWitness;
use super::fs_poseidon::{poseidon_state_digest, ShardCommitments};
use super::prover::{IvcAuditParams, IvcAuditProver};
use super::synthetic::{genesis, honest_chain, params, rand_point, rng, test_h};
use super::verifier::{initial_state, verify_against_merkle_root, verify_ivc_audit};

/// Build parameters for a small directory. Kept at 2 shards so the
/// Nova setup and folding stay fast enough for a unit test; the shard
/// count does not change the statement, only its width.
fn small_setup(n_shards: usize) -> Arc<IvcAuditParams> {
    Arc::new(IvcAuditParams::setup(params(n_shards), test_h()).expect("IVC parameter setup"))
}

/// Fold three honest epochs and verify the single resulting proof.
///
/// This is the property the whole change exists for: the auditor does
/// `verify` **once**, not once per epoch, and its cost does not grow
/// with the length of the chain.
#[test]
fn folds_and_verifies_a_three_epoch_chain() {
    let mut r = rng(0xE2E_1);
    let p = params(2);
    let ivc = small_setup(p.n_shards);
    let (epochs, sigmas) = honest_chain(p, test_h(), 3, &mut r);

    let mut prover = IvcAuditProver::new(ivc.clone(), &epochs[0]).expect("prover init");
    for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
        prover.fold_epoch(next, sigma).expect("fold epoch");
    }

    assert_eq!(prover.num_steps(), 3);
    let proof = prover.proof().expect("a proof exists after folding");
    let verified = verify_ivc_audit(
        &ivc,
        proof,
        prover.num_steps(),
        prover.z0(),
        epochs.last().expect("non-empty"),
    )
    .expect("audit verifies");
    assert_eq!(verified.epochs, 3);
}

/// A verifier that reconstructs `z0` independently — from nothing but
/// the public genesis commitments — must accept the same proof. The
/// auditor must not have to trust the prover's `z0`.
#[test]
fn verifier_reconstructs_initial_state_independently() {
    let mut r = rng(0xE2E_2);
    let p = params(2);
    let ivc = small_setup(p.n_shards);
    let (epochs, sigmas) = honest_chain(p, test_h(), 2, &mut r);

    let mut prover = IvcAuditProver::new(ivc.clone(), &epochs[0]).expect("prover init");
    for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
        prover.fold_epoch(next, sigma).expect("fold epoch");
    }

    let z0 = initial_state(&ivc, &genesis(p.n_shards)).expect("independent z0");
    assert_eq!(z0, prover.z0());
    verify_ivc_audit(
        &ivc,
        prover.proof().expect("proof"),
        2,
        &z0,
        epochs.last().expect("non-empty"),
    )
    .expect("audit verifies against independently derived z0");
}

/// The proof must be bound to the epoch it describes. Verifying
/// against a *different* epoch's commitments has to fail, otherwise
/// a server could serve a stale proof alongside fresh commitments.
#[test]
fn rejects_mismatched_epoch_commitments() {
    let mut r = rng(0xE2E_3);
    let p = params(2);
    let ivc = small_setup(p.n_shards);
    let (epochs, sigmas) = honest_chain(p, test_h(), 3, &mut r);

    let mut prover = IvcAuditProver::new(ivc.clone(), &epochs[0]).expect("prover init");
    for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
        prover.fold_epoch(next, sigma).expect("fold epoch");
    }
    let proof = prover.proof().expect("proof");

    // An earlier epoch in the same chain.
    assert!(
        verify_ivc_audit(&ivc, proof, 3, prover.z0(), &epochs[2]).is_err(),
        "proof must not verify against the wrong epoch's commitments"
    );

    // A single perturbed commitment in the correct epoch.
    let mut tweaked = epochs.last().expect("non-empty").clone();
    tweaked[0].value = rand_point(&mut r);
    assert!(
        verify_ivc_audit(&ivc, proof, 3, prover.z0(), &tweaked).is_err(),
        "proof must not verify against tampered commitments"
    );
}

/// Claiming a different number of steps than were folded must fail.
#[test]
fn rejects_wrong_step_count() {
    let mut r = rng(0xE2E_4);
    let p = params(2);
    let ivc = small_setup(p.n_shards);
    let (epochs, sigmas) = honest_chain(p, test_h(), 2, &mut r);

    let mut prover = IvcAuditProver::new(ivc.clone(), &epochs[0]).expect("prover init");
    for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
        prover.fold_epoch(next, sigma).expect("fold epoch");
    }
    let proof = prover.proof().expect("proof");
    let last = epochs.last().expect("non-empty");

    for claimed in [1usize, 3, 4] {
        assert!(
            verify_ivc_audit(&ivc, proof, claimed, prover.z0(), last).is_err(),
            "verification accepted a claim of {claimed} steps for a 2-step proof"
        );
    }
    assert!(verify_ivc_audit(&ivc, proof, 2, prover.z0(), last).is_ok());
}

/// A dishonest transition must not yield a verifying proof.
///
/// Note *where* the check lands. Nova's `prove_step` is a folding
/// operation, not a checker: it will happily absorb an unsatisfying
/// witness into the running relaxed instance. Unsatisfiability
/// surfaces at `verify`, which runs `is_sat_relaxed` over the folded
/// instance. So a malicious server can *produce* bytes; what it
/// cannot do is produce bytes that an auditor accepts. The chain is
/// also poisoned permanently — every later epoch folded on top
/// inherits the bad instance — so one dishonest epoch invalidates the
/// whole recursive proof rather than just its own step.
#[test]
fn dishonest_transition_does_not_yield_a_verifying_proof() {
    let mut r = rng(0xE2E_5);
    let p = params(2);
    let ivc = small_setup(p.n_shards);
    let (epochs, sigmas) = honest_chain(p, test_h(), 2, &mut r);

    let mut prover = IvcAuditProver::new(ivc.clone(), &epochs[0]).expect("prover init");
    prover.fold_epoch(&epochs[1], &sigmas[0]).expect("epoch 1");

    // Corrupt epoch 2: rand_value no longer satisfies the chain
    // relation, so its Schnorr proof no longer certifies the residue.
    let mut bad = epochs[2].clone();
    bad[1].rand_value = rand_point(&mut r);
    let _ = prover.fold_epoch(&bad, &sigmas[1]);

    if let Some(proof) = prover.proof() {
        assert!(
            verify_ivc_audit(&ivc, proof, prover.num_steps(), prover.z0(), &bad).is_err(),
            "a chain containing a dishonest transition must not verify"
        );
    }
}

/// Once a bad epoch is folded in, even rewinding to claim only the
/// honest prefix must not verify: the running instance is already
/// contaminated.
#[test]
fn poisoned_chain_cannot_be_rewound_to_a_valid_prefix() {
    let mut r = rng(0xE2E_8);
    let p = params(2);
    let ivc = small_setup(p.n_shards);
    let (epochs, sigmas) = honest_chain(p, test_h(), 2, &mut r);

    let mut prover = IvcAuditProver::new(ivc.clone(), &epochs[0]).expect("prover init");
    prover.fold_epoch(&epochs[1], &sigmas[0]).expect("epoch 1");
    let mut bad = epochs[2].clone();
    bad[0].rand_index = rand_point(&mut r);
    let _ = prover.fold_epoch(&bad, &sigmas[1]);

    if let Some(proof) = prover.proof() {
        // Claiming the honest 1-epoch prefix must also fail.
        assert!(verify_ivc_audit(&ivc, proof, 1, prover.z0(), &epochs[1]).is_err());
    }
}

/// The bulletin-board binding: the same tuple that reproduces the
/// folded digest must also reproduce the published SHA256 Merkle
/// root. A wrong root fails even when the recursive proof is valid.
#[test]
fn merkle_root_binding_is_enforced() {
    let mut r = rng(0xE2E_6);
    let p = params(2);
    let ivc = small_setup(p.n_shards);
    let (epochs, sigmas) = honest_chain(p, test_h(), 2, &mut r);

    let mut prover = IvcAuditProver::new(ivc.clone(), &epochs[0]).expect("prover init");
    for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
        prover.fold_epoch(next, sigma).expect("fold epoch");
    }
    let proof = prover.proof().expect("proof");
    let last = epochs.last().expect("non-empty");
    let published = [7u8; 32];

    assert!(
        verify_against_merkle_root(&ivc, proof, 2, prover.z0(), last, published, || published,)
            .is_ok()
    );

    assert!(
        verify_against_merkle_root(&ivc, proof, 2, prover.z0(), last, published, || [9u8; 32])
            .is_err(),
        "a mismatched Merkle root must fail the audit"
    );
}

/// Shard-count mismatches between the parameters and the supplied
/// commitments must be caught rather than producing a nonsense
/// digest.
#[test]
fn rejects_shard_count_mismatch() {
    let ivc = small_setup(2);
    let wrong: Vec<ShardCommitments> = genesis(3);
    assert!(initial_state(&ivc, &wrong).is_err());

    let mut prover = IvcAuditProver::new(ivc.clone(), &genesis(2)).expect("prover init");
    let bad_sigma = vec![SigmaWitness::default(); 3];
    assert!(prover.fold_epoch(&genesis(3), &bad_sigma).is_err());
}

/// Sanity: the folded state digest is exactly the native digest of
/// the epoch, which is what makes the verifier's binding check
/// meaningful.
#[test]
fn state_digest_matches_native_after_folding() {
    let mut r = rng(0xE2E_7);
    let p = params(1);
    let ivc = small_setup(p.n_shards);
    let (epochs, sigmas) = honest_chain(p, test_h(), 1, &mut r);

    let mut prover = IvcAuditProver::new(ivc.clone(), &epochs[0]).expect("prover init");
    prover.fold_epoch(&epochs[1], &sigmas[0]).expect("fold");

    let z = prover
        .proof()
        .expect("proof")
        .verify(ivc.public_params(), 1, prover.z0())
        .expect("verify");
    assert_eq!(z[0], super::bridge::CircuitField::from(1u64));
    assert_eq!(z[3], poseidon_state_digest(ivc.ro_consts(), p, &epochs[1]));
    let _ = ArkFr::from(0u64);
}

/// The compressed proof — what a deployment actually publishes —
/// verifies the same statement, and stays bound to the same epoch.
#[test]
fn compressed_proof_verifies_and_stays_bound() {
    use super::prover::{compress, compressed_proof_size_bytes};
    use super::verifier::verify_compressed_ivc_audit;

    let mut r = rng(0xC0FFEE);
    let p = params(1);
    let ivc = small_setup(p.n_shards);
    let (epochs, sigmas) = honest_chain(p, test_h(), 2, &mut r);

    let mut prover = IvcAuditProver::new(ivc.clone(), &epochs[0]).expect("prover init");
    for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
        prover.fold_epoch(next, sigma).expect("fold epoch");
    }

    let (pk, vk) = ivc.compression_keys().expect("compression keys");
    let compressed = compress(&ivc, &pk, prover.proof().expect("proof")).expect("compress");
    let last = epochs.last().expect("non-empty");

    let verified = verify_compressed_ivc_audit(&ivc, &vk, &compressed, 2, prover.z0(), last)
        .expect("compressed audit verifies");
    assert_eq!(verified.epochs, 2);

    // Still bound to its epoch.
    assert!(
        verify_compressed_ivc_audit(&ivc, &vk, &compressed, 2, prover.z0(), &epochs[1]).is_err(),
        "compressed proof must be bound to the epoch it describes"
    );
    // And to its step count.
    assert!(
        verify_compressed_ivc_audit(&ivc, &vk, &compressed, 1, prover.z0(), last).is_err(),
        "compressed proof must be bound to its step count"
    );

    // Compression must actually be a compression.
    let recursive = super::prover::proof_size_bytes(prover.proof().expect("proof"));
    let published = compressed_proof_size_bytes(&compressed);
    assert!(
        published * 20 < recursive,
        "compressed proof ({published} B) is not meaningfully smaller than the \
         folding state ({recursive} B)"
    );
}

// ---------- group-sharded auditing -------------------------------------

use super::grouped::{
    verify_grouped_folding_proofs, verify_grouped_ivc_audit, GroupPlan, GroupedIvcAuditParams,
    GroupedIvcAuditProver,
};
use super::synthetic::honest_grouped_chain;

/// Four shards split into two independent chains fold and verify.
///
/// The load-bearing claim of group-sharding: the groups never
/// interact, so two chains over two shards each certify exactly what
/// one chain over four shards would.
#[test]
fn grouped_chain_folds_and_verifies() {
    let plan = GroupPlan::new(4, 2).expect("plan");
    let h = test_h();
    let mut r = rng(0x6C0DE_1);
    let (epochs, sigmas) = honest_grouped_chain(plan, 12, h, 2, &mut r);

    let gp = GroupedIvcAuditParams::setup(plan, 12, h).expect("grouped setup");
    let mut prover = GroupedIvcAuditProver::new(&gp, &epochs[0]).expect("prover");
    for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
        prover.fold_epoch(next, sigma).expect("fold");
    }
    assert_eq!(prover.num_steps(), 2);

    let proofs = prover.proofs().expect("proofs");
    assert_eq!(proofs.len(), 2, "one folding proof per group");
    let verified = verify_grouped_folding_proofs(
        &gp,
        &proofs,
        prover.num_steps(),
        &epochs[0],
        epochs.last().expect("non-empty"),
    )
    .expect("grouped audit verifies");
    assert_eq!(verified.epochs, 2);
}

/// One group's chain is proved and verified against a *different*
/// group's shards. Must fail: each group's proof is bound to its own
/// slice by the Poseidon state digest, so proofs are not
/// interchangeable even though every group shares one circuit shape.
#[test]
fn group_proofs_are_bound_to_their_own_shards() {
    let plan = GroupPlan::new(4, 2).expect("plan");
    let h = test_h();
    let mut r = rng(0x6C0DE_2);
    let (epochs, sigmas) = honest_grouped_chain(plan, 12, h, 1, &mut r);

    let gp = GroupedIvcAuditParams::setup(plan, 12, h).expect("grouped setup");
    let mut prover = GroupedIvcAuditProver::new(&gp, &epochs[0]).expect("prover");
    prover.fold_epoch(&epochs[1], &sigmas[0]).expect("fold");

    // Swap the two groups' proofs.
    let proofs = prover.proofs().expect("proofs");
    let swapped = vec![proofs[1], proofs[0]];
    let res = verify_grouped_folding_proofs(
        &gp,
        &swapped,
        prover.num_steps(),
        &epochs[0],
        epochs.last().expect("non-empty"),
    );
    assert!(
        res.is_err(),
        "a group's proof must not verify against another group's shards"
    );
}

/// Tampering with a shard in the *second* group must be caught. A
/// naive implementation that only checked group 0, or that derived
/// every group's state from the whole tuple, would pass this.
#[test]
fn tampering_any_group_is_caught() {
    let plan = GroupPlan::new(4, 2).expect("plan");
    let h = test_h();
    let mut r = rng(0x6C0DE_3);
    let (epochs, sigmas) = honest_grouped_chain(plan, 12, h, 1, &mut r);

    let gp = GroupedIvcAuditParams::setup(plan, 12, h).expect("grouped setup");
    let mut prover = GroupedIvcAuditProver::new(&gp, &epochs[0]).expect("prover");
    prover.fold_epoch(&epochs[1], &sigmas[0]).expect("fold");
    let proofs = prover.proofs().expect("proofs");

    // Shard 3 lives in group 1 under a 4-shard / 2-group plan.
    assert_eq!(plan.group_of(3), 1);
    let mut tampered = epochs[1].clone();
    tampered[3].rand_value = rand_point(&mut r);

    let res =
        verify_grouped_folding_proofs(&gp, &proofs, prover.num_steps(), &epochs[0], &tampered);
    assert!(
        res.is_err(),
        "a tampered shard in group 1 must be rejected, not just one in group 0"
    );
}

/// A single-group plan must behave exactly like the ungrouped prover.
/// This is the regression guard for the default configuration.
#[test]
fn single_group_matches_the_ungrouped_prover() {
    let h = test_h();
    let p = params(2);
    let mut r = rng(0x6C0DE_4);
    let (epochs, sigmas) = honest_chain(p, h, 2, &mut r);

    let plan = GroupPlan::single(2).expect("plan");
    let gp = GroupedIvcAuditParams::setup(plan, p.num_vars, h).expect("grouped setup");
    let mut grouped = GroupedIvcAuditProver::new(&gp, &epochs[0]).expect("grouped prover");

    let flat_params = small_setup(2);
    let mut flat = IvcAuditProver::new(flat_params.clone(), &epochs[0]).expect("flat prover");

    for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
        grouped.fold_epoch(next, sigma).expect("grouped fold");
        flat.fold_epoch(next, sigma).expect("flat fold");
    }

    assert_eq!(grouped.num_steps(), flat.num_steps());
    let latest = epochs.last().expect("non-empty");
    verify_grouped_folding_proofs(
        &gp,
        &grouped.proofs().expect("proofs"),
        grouped.num_steps(),
        &epochs[0],
        latest,
    )
    .expect("grouped verifies");
    verify_ivc_audit(
        &flat_params,
        flat.proof().expect("proof"),
        flat.num_steps(),
        flat.z0(),
        latest,
    )
    .expect("flat verifies");
}

/// The compressed (published) form of a grouped audit verifies, and
/// is still bound to the epoch it was taken at.
#[test]
fn grouped_compressed_audit_verifies_and_stays_bound() {
    let plan = GroupPlan::new(4, 2).expect("plan");
    let h = test_h();
    let mut r = rng(0x6C0DE_5);
    let (epochs, sigmas) = honest_grouped_chain(plan, 12, h, 1, &mut r);

    let gp = GroupedIvcAuditParams::setup(plan, 12, h).expect("grouped setup");
    let (pk, vk) = gp.compression_keys().expect("compression keys");
    let mut prover = GroupedIvcAuditProver::new(&gp, &epochs[0]).expect("prover");
    prover.fold_epoch(&epochs[1], &sigmas[0]).expect("fold");

    let published = prover.compress_all(&pk).expect("compress");
    assert_eq!(published.groups(), 2);
    assert!(published.size_bytes() > 0);

    verify_grouped_ivc_audit(&gp, &vk, &published, 1, &epochs[0], &epochs[1])
        .expect("compressed grouped audit verifies");

    // Same proof, wrong epoch tuple.
    let mut tampered = epochs[1].clone();
    tampered[0].rand_index = rand_point(&mut r);
    assert!(
        verify_grouped_ivc_audit(&gp, &vk, &published, 1, &epochs[0], &tampered).is_err(),
        "compressed grouped audit must stay bound to its epoch"
    );
}

/// Partition validation. Uneven groups would need per-group public
/// parameters, so they are rejected rather than silently supported.
#[test]
fn group_plan_requires_an_exact_division() {
    assert!(GroupPlan::new(128, 8).is_ok());
    assert!(GroupPlan::new(128, 6).is_err(), "6 does not divide 128");
    assert!(GroupPlan::new(4, 0).is_err());
    assert!(GroupPlan::new(0, 1).is_err());

    let plan = GroupPlan::new(128, 8).expect("plan");
    assert_eq!(plan.shards_per_group(), 16);
    assert_eq!(plan.range(0), 0..16);
    assert_eq!(plan.range(7), 112..128);
    assert_eq!(plan.group_of(0), 0);
    assert_eq!(plan.group_of(127), 7);

    // Splitting rejects a tuple that does not cover every shard.
    let short: Vec<u8> = vec![0; 127];
    assert!(plan.split(&short).is_err());
}

/// The runtime knob: `for_shards` takes the group count as a plain
/// value and must agree with the explicit-`GroupPlan` form.
#[test]
fn group_count_is_a_runtime_parameter() {
    let h = test_h();
    for groups in [1usize, 2, 4] {
        let a = GroupedIvcAuditParams::for_shards(4, groups, 12, h).expect("for_shards");
        let b = GroupedIvcAuditParams::setup(GroupPlan::new(4, groups).expect("plan"), 12, h)
            .expect("setup");
        assert_eq!(a.plan(), b.plan());
        assert_eq!(
            a.constraints_per_step(),
            b.constraints_per_step(),
            "the two constructors must build the same circuit at groups={groups}"
        );
    }

    // A count that does not divide the shard set is rejected at
    // construction, not at the first fold.
    assert!(GroupedIvcAuditParams::for_shards(4, 3, 12, h).is_err());
    assert!(GroupedIvcAuditParams::for_shards(4, 0, 12, h).is_err());
}

/// Changing the knob must actually change the circuit, and shrink it
/// in proportion. This is the property the whole knob exists for; a
/// version that silently ignored `groups` would pass every other test
/// in this file.
#[test]
fn more_groups_means_a_proportionally_smaller_circuit() {
    let h = test_h();
    let one = GroupedIvcAuditParams::for_shards(8, 1, 12, h).expect("g=1");
    let two = GroupedIvcAuditParams::for_shards(8, 2, 12, h).expect("g=2");
    let four = GroupedIvcAuditParams::for_shards(8, 4, 12, h).expect("g=4");

    assert_eq!(one.plan().shards_per_group(), 8);
    assert_eq!(two.plan().shards_per_group(), 4);
    assert_eq!(four.plan().shards_per_group(), 2);

    // Constraints are roughly affine in shards-per-group: a fixed
    // Nova augmentation plus a per-shard cost. So halving the group
    // size must strictly shrink the circuit, and the *marginal* cost
    // per shard must be about the same however the shards are split.
    assert!(four.constraints_per_step() < two.constraints_per_step());
    assert!(two.constraints_per_step() < one.constraints_per_step());

    // "About": the split is not exactly affine at these very small
    // group sizes, because the Poseidon state digest absorbs a fixed
    // number of elements per shard into a sponge that processes them
    // in fixed-size blocks -- so the digest's cost steps rather than
    // scales, and the step is proportionally large when a group holds
    // 2 shards. At deployment sizes it washes out: measured across
    // 16/32/64/128 shards the marginal cost is 8605.25 per shard to
    // the constraint. A 10% band catches a knob that is ignored
    // without pinning small-size rounding.
    let marginal_hi = (one.constraints_per_step() - two.constraints_per_step()) as f64 / 4.0;
    let marginal_lo = (two.constraints_per_step() - four.constraints_per_step()) as f64 / 2.0;
    assert!(
        (marginal_hi - marginal_lo).abs() / marginal_hi < 0.10,
        "per-shard constraint cost should not depend much on the group count: \
         {marginal_hi} vs {marginal_lo}"
    );
}
