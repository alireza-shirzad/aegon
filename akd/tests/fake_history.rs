//! Fake-history end-to-end test.
//!
//! Models a realistic AKD lifecycle: users sign up, look up their own
//! slot, update their values, and run their own consistency checks at
//! different times, while the auditor verifies every epoch transition
//! as it happens. Sign-up, lookup, value-update, consistency-check, and
//! audit calls are interleaved across an eight-epoch run.
//!
//! For each fetch from the directory, the test asserts that what
//! comes back matches what was put in — via [`aegon_facade::verify_lookup`]
//! for value match, [`aegon_facade::verify_consistency`] for slot
//! stability, and [`aegon_facade::verify_invariance`] (threading a
//! single `ShardedAuditState`) for the Fiat-Shamir-bound homomorphic chain.
//!
//! Value updates are exercised explicitly: a user who legitimately
//! changes their value WILL see their own consistency proof spanning
//! that change fail. The test asserts the verifier returns `Ok(false)`
//! in exactly those cases — confirming the user can *detect* their
//! own historical value change (which is the security property), not
//! a bug.

use akd::aegon_facade::{
    self, ConsistencyProof, EpochCommitment, ShardedAuditState, VerifierContext,
};
use akd::append_only_zks::AzksParallelismConfig;
use akd::directory::Directory;
use akd::ecvrf::HardCodedAkdVRF;
use akd::storage::memory::AsyncInMemoryDatabase;
use akd::storage::StorageManager;
use akd::{AkdLabel, AkdValue, EpochHash, ExperimentalConfiguration};

#[derive(Clone)]
struct TestDomainLabel;
impl akd::DomainLabel for TestDomainLabel {
    fn domain_label() -> &'static [u8] {
        b"AkdAegonFakeHistory"
    }
}

type Config = ExperimentalConfiguration<TestDomainLabel>;
type AkdDirectory = Directory<Config, AsyncInMemoryDatabase, HardCodedAkdVRF>;

#[derive(Clone)]
struct UserRecord {
    label: AkdLabel,
    /// Current value (mutated on update).
    value: AkdValue,
    /// Epoch at which the user first signed up — the slot was assigned
    /// here and never moves afterwards.
    signup_epoch: u64,
    /// Epoch of the most recent `value` write. Equal to `signup_epoch`
    /// if the user has never updated. Used to decide whether a
    /// consistency proof spanning some `s0` should succeed or fail.
    last_change_epoch: u64,
}

async fn fresh_directory() -> AkdDirectory {
    let db = AsyncInMemoryDatabase::new();
    let storage_manager = StorageManager::new_no_cache(db);
    let vrf = HardCodedAkdVRF {};
    Directory::<Config, _, _>::new(storage_manager, vrf, AzksParallelismConfig::default())
        .await
        .expect("Directory::new")
}

/// Apply one epoch's worth of changes: zero or more new sign-ups plus
/// zero or more value updates to existing users. Returns the new epoch
/// number. Mutates `truth` to reflect the post-publish ground state.
async fn apply_epoch(
    directory: &AkdDirectory,
    sign_ups: &[(&str, &str)],
    updates: &[(usize, &str)],
    truth: &mut Vec<UserRecord>,
) -> u64 {
    let mut entries: Vec<(AkdLabel, AkdValue)> = sign_ups
        .iter()
        .map(|(n, v)| (AkdLabel::from(*n), AkdValue::from(*v)))
        .collect();
    for (idx, new_value) in updates {
        entries.push((truth[*idx].label.clone(), AkdValue::from(*new_value)));
    }
    let EpochHash(epoch, _) = directory.publish(entries).await.expect("publish");
    for (name, value) in sign_ups {
        truth.push(UserRecord {
            label: AkdLabel::from(*name),
            value: AkdValue::from(*value),
            signup_epoch: epoch,
            last_change_epoch: epoch,
        });
    }
    for (idx, new_value) in updates {
        truth[*idx].value = AkdValue::from(*new_value);
        truth[*idx].last_change_epoch = epoch;
    }
    epoch
}

/// Auditor: verify the single new transition from `prev` to the
/// commitment at `next_epoch`, threading `audit_state` through. Returns
/// the new commitment to chain forward as the next `prev`.
async fn audit_one_transition(
    directory: &AkdDirectory,
    ctx: &VerifierContext,
    audit_state: &mut ShardedAuditState,
    prev: &EpochCommitment,
    next_epoch: u64,
) -> EpochCommitment {
    let next = directory
        .epoch_commitment(next_epoch)
        .await
        .expect("next commitment retained");
    let ok = aegon_facade::verify_invariance(ctx, audit_state, prev, &next)
        .expect("verify_invariance");
    assert!(
        ok,
        "auditor must accept the honest transition into epoch {next_epoch}"
    );
    next
}

/// Single user pulls their value, asserts the returned `AkdValue`
/// matches the ground-truth value we currently expect, and verifies
/// the proof.
async fn user_lookup(
    directory: &AkdDirectory,
    ctx: &VerifierContext,
    user: &UserRecord,
    current_epoch: u64,
) {
    let (proof, eh) = directory
        .lookup(user.label.clone())
        .await
        .expect("lookup");
    assert_eq!(
        eh.epoch(),
        current_epoch,
        "lookup must reflect the current epoch for {:?}",
        user.label
    );
    assert_eq!(
        &proof.value, &user.value,
        "lookup returned different bytes than we expect for {:?}",
        user.label
    );
    let ok = aegon_facade::verify_lookup(ctx, &user.label, &proof).expect("verify_lookup");
    assert!(ok, "honest lookup must verify for {:?}", user.label);
}

/// Run a consistency check from `s0_epoch` to `current.epoch` for a
/// user, asserting the verifier returns whatever the ground truth says
/// it should: `Ok(true)` iff the user's value has not changed in the
/// span `(s0_epoch, current.epoch]`; otherwise `Ok(false)` (the
/// "legitimate change" case the user explicitly knows about).
async fn user_consistency(
    directory: &AkdDirectory,
    ctx: &VerifierContext,
    user: &UserRecord,
    current: &EpochCommitment,
    s0_epoch: u64,
) {
    assert!(
        s0_epoch < current.epoch,
        "consistency requires s0 < s1: {} < {}",
        s0_epoch,
        current.epoch
    );
    let s0 = directory
        .epoch_commitment(s0_epoch)
        .await
        .unwrap_or_else(|| panic!("commitment retained at epoch {s0_epoch}"));
    let proof: ConsistencyProof = directory
        .consistency_proof(&user.label, s0_epoch)
        .await
        .expect("consistency_proof");
    let result = aegon_facade::verify_consistency(
        ctx,
        &s0,
        current,
        &user.label,
        &proof,
    )
    .expect("verify_consistency");

    // Did the user change their value strictly after s0 and at-or-before
    // current.epoch? If so, their slot's rand_value diverged and the
    // consistency check legitimately rejects.
    let value_changed = user.last_change_epoch > s0_epoch;
    if value_changed {
        assert!(
            !result,
            "consistency must reject when value changed in ({}, {}] for {:?}",
            s0_epoch, current.epoch, user.label
        );
    } else {
        assert!(
            result,
            "consistency must hold from epoch {} to epoch {} for {:?}",
            s0_epoch, current.epoch, user.label
        );
    }
}

#[tokio::test]
async fn interleaved_eight_epoch_lifecycle_with_updates() {
    let directory = fresh_directory().await;
    let ctx: VerifierContext = directory.verifier_context().await;
    let mut audit_state = ShardedAuditState::default();

    // Auditor's running view of the chain head.
    let mut prev: EpochCommitment = directory
        .epoch_commitment(0)
        .await
        .expect("epoch 0 retained");

    let mut truth: Vec<UserRecord> = Vec::new();

    // ===================== EPOCH 1 =====================
    // Three sign-ups. The auditor immediately verifies 0 -> 1; the
    // three new users each look themselves up.
    let e1 = apply_epoch(
        &directory,
        &[
            ("alice", "alice-v1"),
            ("bob", "bob-v1"),
            ("carol", "carol-v1"),
        ],
        &[],
        &mut truth,
    )
    .await;
    assert_eq!(e1, 1);
    prev = audit_one_transition(&directory, &ctx, &mut audit_state, &prev, e1).await;
    for idx in [0, 1, 2] {
        user_lookup(&directory, &ctx, &truth[idx], e1).await;
    }

    // ===================== EPOCH 2 =====================
    // Two more sign-ups. Auditor verifies, then carol (from e1) does a
    // lookup and alice runs a stable-value consistency proof e1 -> e2.
    let e2 = apply_epoch(
        &directory,
        &[("dave", "dave-v1"), ("eve", "eve-v1")],
        &[],
        &mut truth,
    )
    .await;
    assert_eq!(e2, 2);
    prev = audit_one_transition(&directory, &ctx, &mut audit_state, &prev, e2).await;
    user_lookup(&directory, &ctx, &truth[2], e2).await; // carol
    user_consistency(&directory, &ctx, &truth[0], &prev, /*s0=*/ 1).await; // alice e1->e2 (stable)

    // ===================== EPOCH 3 =====================
    // Idle: no sign-ups, no updates. Earlier users use the time to do
    // their own checks.
    let e3 = apply_epoch(&directory, &[], &[], &mut truth).await;
    assert_eq!(e3, 3);
    prev = audit_one_transition(&directory, &ctx, &mut audit_state, &prev, e3).await;
    user_lookup(&directory, &ctx, &truth[1], e3).await; // bob
    user_lookup(&directory, &ctx, &truth[3], e3).await; // dave
    user_consistency(&directory, &ctx, &truth[0], &prev, /*s0=*/ 1).await; // alice e1->e3
    user_consistency(&directory, &ctx, &truth[4], &prev, /*s0=*/ 2).await; // eve e2->e3

    // ===================== EPOCH 4 (UPDATES!) =====================
    // Two new sign-ups land AND alice updates her value
    // (alice-v1 -> alice-v2) in the same epoch.
    //
    // After this publish:
    //   * alice's lookup must return the *new* value
    //   * a consistency proof from BEFORE the change (e1, e2, e3) to
    //     here MUST fail — this is the legitimate-rejection case.
    //   * a consistency proof for any other user across the same span
    //     keeps succeeding.
    let e4 = apply_epoch(
        &directory,
        &[("frank", "frank-v1"), ("grace", "grace-v1")],
        &[(0, "alice-v2")], // alice updates her value
        &mut truth,
    )
    .await;
    assert_eq!(e4, 4);
    assert_eq!(truth[0].last_change_epoch, e4);
    prev = audit_one_transition(&directory, &ctx, &mut audit_state, &prev, e4).await;

    // Fresh lookup must return the new value.
    user_lookup(&directory, &ctx, &truth[0], e4).await; // alice's new value verified
    user_lookup(&directory, &ctx, &truth[5], e4).await; // frank
    // Alice's own consistency proof across the change is expected to
    // FAIL (Ok(false)) — and `user_consistency` enforces exactly that
    // by looking at `last_change_epoch`.
    user_consistency(&directory, &ctx, &truth[0], &prev, /*s0=*/ 1).await;
    // Bob did not change his value; his consistency from e1 still holds.
    user_consistency(&directory, &ctx, &truth[1], &prev, /*s0=*/ 1).await;

    // ===================== EPOCH 5 =====================
    // Idle. Various users check.
    let e5 = apply_epoch(&directory, &[], &[], &mut truth).await;
    assert_eq!(e5, 5);
    prev = audit_one_transition(&directory, &ctx, &mut audit_state, &prev, e5).await;
    user_lookup(&directory, &ctx, &truth[6], e5).await; // grace
    // Alice's consistency from POST-update (e4) onward holds.
    user_consistency(&directory, &ctx, &truth[0], &prev, /*s0=*/ 4).await;
    // From pre-update epochs it still rejects.
    user_consistency(&directory, &ctx, &truth[0], &prev, /*s0=*/ 1).await;
    user_consistency(&directory, &ctx, &truth[3], &prev, /*s0=*/ 2).await; // dave e2->e5 (stable)
    user_consistency(&directory, &ctx, &truth[5], &prev, /*s0=*/ 4).await; // frank e4->e5 (stable)

    // ===================== EPOCH 6 =====================
    // Three new sign-ups; carol *also* updates her value
    // (carol-v1 -> carol-v2) in this epoch.
    let e6 = apply_epoch(
        &directory,
        &[
            ("heidi", "heidi-v1"),
            ("ivan", "ivan-v1"),
            ("judy", "judy-v1"),
        ],
        &[(2, "carol-v2")], // carol updates her value
        &mut truth,
    )
    .await;
    assert_eq!(e6, 6);
    assert_eq!(truth[2].last_change_epoch, e6);
    prev = audit_one_transition(&directory, &ctx, &mut audit_state, &prev, e6).await;

    user_lookup(&directory, &ctx, &truth[2], e6).await; // carol's new value
    user_lookup(&directory, &ctx, &truth[7], e6).await; // heidi
    user_lookup(&directory, &ctx, &truth[9], e6).await; // judy
    // Carol's consistency from before her change rejects.
    user_consistency(&directory, &ctx, &truth[2], &prev, /*s0=*/ 1).await;
    user_consistency(&directory, &ctx, &truth[2], &prev, /*s0=*/ 5).await;
    // Alice's long-range consistency from e1 still rejects (her value
    // changed at e4); from e4 onward, it holds.
    user_consistency(&directory, &ctx, &truth[0], &prev, /*s0=*/ 1).await;
    user_consistency(&directory, &ctx, &truth[0], &prev, /*s0=*/ 4).await;

    // ===================== EPOCH 7 =====================
    // Idle except for bob's value update (bob-v1 -> bob-v2).
    let e7 = apply_epoch(&directory, &[], &[(1, "bob-v2")], &mut truth).await;
    assert_eq!(e7, 7);
    assert_eq!(truth[1].last_change_epoch, e7);
    prev = audit_one_transition(&directory, &ctx, &mut audit_state, &prev, e7).await;

    user_lookup(&directory, &ctx, &truth[1], e7).await; // bob's new value
    user_lookup(&directory, &ctx, &truth[4], e7).await; // eve (unchanged)
    user_lookup(&directory, &ctx, &truth[8], e7).await; // ivan (unchanged)
    // Bob's consistency from e1 (pre-change) rejects.
    user_consistency(&directory, &ctx, &truth[1], &prev, /*s0=*/ 1).await;
    // Dave's stable consistency from e2 still holds.
    user_consistency(&directory, &ctx, &truth[3], &prev, /*s0=*/ 2).await;
    // Heidi signed up at e6 with no changes — e6->e7 still holds.
    user_consistency(&directory, &ctx, &truth[7], &prev, /*s0=*/ 6).await;

    // ===================== EPOCH 8 =====================
    // Two final sign-ups. Every existing user does a final round of
    // checks. For each pre-existing user we exercise BOTH:
    //  (a) consistency from their sign-up epoch — expected to fail iff
    //      they ever changed their value;
    //  (b) consistency from their last_change_epoch — expected to
    //      always succeed (no further change since).
    let e8 = apply_epoch(
        &directory,
        &[("ken", "ken-v1"), ("lara", "lara-v1")],
        &[],
        &mut truth,
    )
    .await;
    assert_eq!(e8, 8);
    prev = audit_one_transition(&directory, &ctx, &mut audit_state, &prev, e8).await;

    user_lookup(&directory, &ctx, &truth[10], e8).await; // ken
    user_lookup(&directory, &ctx, &truth[11], e8).await; // lara

    let mut updated_count = 0usize;
    let mut unchanged_count = 0usize;
    for u in truth.iter().filter(|u| u.signup_epoch < e8) {
        // (a) From sign-up. Helper auto-decides expected outcome from
        //     last_change_epoch.
        user_consistency(&directory, &ctx, u, &prev, u.signup_epoch).await;
        // (b) From their last value-write — always succeeds.
        if u.last_change_epoch > u.signup_epoch {
            user_consistency(&directory, &ctx, u, &prev, u.last_change_epoch).await;
            updated_count += 1;
        } else {
            unchanged_count += 1;
        }
    }
    // We updated alice (e4), carol (e6), and bob (e7) — three users.
    assert_eq!(updated_count, 3, "alice/carol/bob were updated");
    // 10 pre-e8 users total, 3 updated, 7 unchanged.
    assert_eq!(unchanged_count, 7);

    // ===================== Auditor sanity =====================
    // Confirm the full chain (0 -> 8) re-verifies independently, with
    // a fresh `ShardedAuditState`. Value updates don't change the auditor's
    // story — the chain randomness depends only on commitments, not on
    // what data flowed through them.
    let mut fresh_state = ShardedAuditState::default();
    let mut fresh_prev = directory.epoch_commitment(0).await.unwrap();
    for i in 0..8u64 {
        let next = directory.epoch_commitment(i + 1).await.unwrap();
        let ok = aegon_facade::verify_invariance(&ctx, &mut fresh_state, &fresh_prev, &next)
            .expect("verify_invariance");
        assert!(ok, "full chain replay must accept transition {i}");
        fresh_prev = next;
    }
}
