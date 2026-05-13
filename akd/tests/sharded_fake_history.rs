//! Eight-epoch sharded-Aegon end-to-end smoke test.
//!
//! Mirrors `tests/fake_history.rs` but drives `ShardedAegon` directly
//! (without the Directory facade, which hasn't been plumbed for
//! sharding yet). On every epoch we exercise:
//!
//!   * `verify_sharded_invariance` — auditor threads a single
//!     `AuditState` across all eight transitions, verifying the
//!     shared FS chain + Merkle anchoring + per-shard chain witness.
//!   * `verify_sharded_lookup` — users pull their own slot, the
//!     verifier re-derives the cross-shard probe trail from the label.
//!   * `verify_sharded_consistency` — users prove their slot has not
//!     changed between two epochs (or, after a legitimate value
//!     update, get `Ok(false)`, which is the detectable property).
//!
//! Value updates happen at epochs 4 (alice), 6 (carol), and 7 (bob);
//! the helper auto-decides from `last_change_epoch` whether a
//! consistency proof should accept or reject.

use akd::aegon::{
    verify_sharded_consistency, verify_sharded_invariance, verify_sharded_lookup, AuditState,
    Sha256Hash, ShardedAegon, ShardedAegonConfig, ShardedEpochCommitment, ShardedVerifierContext,
};
use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Pcs = akd_core::aegon_crypto::pcs::kzhk::KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, Sha256Hash>;
type Commit = ShardedEpochCommitment<Bn254, Pcs>;
type ShardedCtx = ShardedVerifierContext<Bn254, Pcs>;

/// Per-user ground truth: current value and the epochs of sign-up /
/// most recent value write. Used to decide what each verifier *should*
/// say.
#[derive(Clone)]
struct UserRecord {
    label: Vec<u8>,
    value: Vec<u8>,
    signup_epoch: u64,
    last_change_epoch: u64,
}

fn fresh(log_capacity: usize, log_n_shards: usize) -> Sharded {
    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(log_capacity - log_n_shards)
        .log_n_shards(log_n_shards)
        .private(false)
        .kzh_k(2)
        .build()
        .expect("config builds");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    Sharded::setup(&mut rng, &cfg).expect("setup")
}

/// Auditor: verify the single new transition `prev -> next_commit`,
/// threading `audit_state` forward. Asserts acceptance.
fn audit_one_transition(
    server: &mut Sharded,
    ctx: &ShardedCtx,
    audit_state: &mut AuditState<<Bn254 as Pairing>::ScalarField>,
    prev: &Commit,
    sign_ups: &[(&str, &str)],
    updates: &[(usize, &str)],
    truth: &mut Vec<UserRecord>,
) -> Commit {
    let mut entries: Vec<(Vec<u8>, Vec<u8>)> = sign_ups
        .iter()
        .map(|(n, v)| (n.as_bytes().to_vec(), v.as_bytes().to_vec()))
        .collect();
    for (idx, new_value) in updates {
        entries.push((truth[*idx].label.clone(), new_value.as_bytes().to_vec()));
    }
    let (commit, invariance) = server.publish(&entries).expect("publish");
    for (name, value) in sign_ups {
        truth.push(UserRecord {
            label: name.as_bytes().to_vec(),
            value: value.as_bytes().to_vec(),
            signup_epoch: commit.epoch,
            last_change_epoch: commit.epoch,
        });
    }
    for (idx, new_value) in updates {
        truth[*idx].value = new_value.as_bytes().to_vec();
        truth[*idx].last_change_epoch = commit.epoch;
    }
    let ok = verify_sharded_invariance::<Bn254, Pcs>(ctx, audit_state, prev, &commit, &invariance)
        .expect("verify_sharded_invariance");
    assert!(
        ok,
        "auditor must accept the honest transition into epoch {}",
        commit.epoch
    );
    commit
}

fn user_lookup(server: &Sharded, ctx: &ShardedCtx, user: &UserRecord, commit: &Commit) {
    let proof = server.lookup(&user.label).expect("lookup");
    let ok = verify_sharded_lookup::<Bn254, Pcs, Sha256Hash>(
        ctx,
        commit,
        &user.label,
        &user.value,
        &proof,
    )
    .expect("verify_sharded_lookup");
    assert!(ok, "honest lookup must verify for {:?}", user.label);
}

fn user_consistency(
    server: &Sharded,
    ctx: &ShardedCtx,
    user: &UserRecord,
    s0: &Commit,
    s1: &Commit,
) {
    assert!(
        s0.epoch < s1.epoch,
        "consistency requires s0 < s1: {} < {}",
        s0.epoch,
        s1.epoch
    );
    // Pin ctr0 from a fresh lookup — this is the security-critical
    // step that prevents a server from substituting a different trail
    // length on the consistency proof.
    let lookup = server.lookup(&user.label).expect("lookup-for-ctr0");
    let expected_ctr0 = lookup.ctr0;
    let proof = server.consistency_proof(&user.label, s0.epoch).expect("consistency_proof");
    let result = verify_sharded_consistency::<Bn254, Pcs, Sha256Hash>(
        ctx,
        s0,
        s1,
        &user.label,
        expected_ctr0,
        &proof,
    )
    .expect("verify_sharded_consistency");

    let value_changed = user.last_change_epoch > s0.epoch;
    if value_changed {
        assert!(
            !result,
            "consistency must reject when value changed in ({}, {}] for {:?}",
            s0.epoch, s1.epoch, user.label
        );
    } else {
        assert!(
            result,
            "consistency must hold from epoch {} to epoch {} for {:?}",
            s0.epoch, s1.epoch, user.label
        );
    }
}

#[test]
fn sharded_interleaved_eight_epoch_lifecycle_with_updates() {
    // 4 shards × 64 slots = 256 total. Plenty of headroom for ~12 users
    // and exercises non-trivial cross-shard routing.
    let mut server = fresh(8, 2);
    let ctx: ShardedCtx = server.sharded_verifier_context();
    let mut audit_state = AuditState::default();

    let mut commits: Vec<Commit> = vec![server.epoch_commitment(0).expect("epoch 0")];
    let mut truth: Vec<UserRecord> = Vec::new();

    // ===================== EPOCH 1 =====================
    let c1 = audit_one_transition(
        &mut server,
        &ctx,
        &mut audit_state,
        &commits[0],
        &[
            ("alice", "alice-v1"),
            ("bob", "bob-v1"),
            ("carol", "carol-v1"),
        ],
        &[],
        &mut truth,
    );
    assert_eq!(c1.epoch, 1);
    commits.push(c1.clone());
    for idx in [0, 1, 2] {
        user_lookup(&server, &ctx, &truth[idx], &c1);
    }

    // ===================== EPOCH 2 =====================
    let c2 = audit_one_transition(
        &mut server,
        &ctx,
        &mut audit_state,
        &c1,
        &[("dave", "dave-v1"), ("eve", "eve-v1")],
        &[],
        &mut truth,
    );
    assert_eq!(c2.epoch, 2);
    commits.push(c2.clone());
    user_lookup(&server, &ctx, &truth[2], &c2); // carol
    user_consistency(&server, &ctx, &truth[0], &commits[1], &c2); // alice e1->e2

    // ===================== EPOCH 3 =====================
    let c3 = audit_one_transition(&mut server, &ctx, &mut audit_state, &c2, &[], &[], &mut truth);
    assert_eq!(c3.epoch, 3);
    commits.push(c3.clone());
    user_lookup(&server, &ctx, &truth[1], &c3); // bob
    user_lookup(&server, &ctx, &truth[3], &c3); // dave
    user_consistency(&server, &ctx, &truth[0], &commits[1], &c3); // alice e1->e3
    user_consistency(&server, &ctx, &truth[4], &commits[2], &c3); // eve e2->e3

    // ===================== EPOCH 4 (alice updates) =====================
    let c4 = audit_one_transition(
        &mut server,
        &ctx,
        &mut audit_state,
        &c3,
        &[("frank", "frank-v1"), ("grace", "grace-v1")],
        &[(0, "alice-v2")],
        &mut truth,
    );
    assert_eq!(c4.epoch, 4);
    assert_eq!(truth[0].last_change_epoch, 4);
    commits.push(c4.clone());
    user_lookup(&server, &ctx, &truth[0], &c4); // alice — new value
    user_lookup(&server, &ctx, &truth[5], &c4); // frank
    user_consistency(&server, &ctx, &truth[0], &commits[1], &c4); // alice e1->e4 must REJECT
    user_consistency(&server, &ctx, &truth[1], &commits[1], &c4); // bob still stable

    // ===================== EPOCH 5 =====================
    let c5 = audit_one_transition(&mut server, &ctx, &mut audit_state, &c4, &[], &[], &mut truth);
    assert_eq!(c5.epoch, 5);
    commits.push(c5.clone());
    user_lookup(&server, &ctx, &truth[6], &c5); // grace
    user_consistency(&server, &ctx, &truth[0], &commits[4], &c5); // alice e4->e5 ok
    user_consistency(&server, &ctx, &truth[0], &commits[1], &c5); // alice e1->e5 reject
    user_consistency(&server, &ctx, &truth[3], &commits[2], &c5); // dave e2->e5 ok
    user_consistency(&server, &ctx, &truth[5], &commits[4], &c5); // frank e4->e5 ok

    // ===================== EPOCH 6 (carol updates) =====================
    let c6 = audit_one_transition(
        &mut server,
        &ctx,
        &mut audit_state,
        &c5,
        &[
            ("heidi", "heidi-v1"),
            ("ivan", "ivan-v1"),
            ("judy", "judy-v1"),
        ],
        &[(2, "carol-v2")],
        &mut truth,
    );
    assert_eq!(c6.epoch, 6);
    assert_eq!(truth[2].last_change_epoch, 6);
    commits.push(c6.clone());
    user_lookup(&server, &ctx, &truth[2], &c6); // carol — new value
    user_lookup(&server, &ctx, &truth[7], &c6); // heidi
    user_lookup(&server, &ctx, &truth[9], &c6); // judy
    user_consistency(&server, &ctx, &truth[2], &commits[1], &c6); // carol e1->e6 reject
    user_consistency(&server, &ctx, &truth[2], &commits[5], &c6); // carol e5->e6 reject
    user_consistency(&server, &ctx, &truth[0], &commits[1], &c6); // alice still rejects from e1
    user_consistency(&server, &ctx, &truth[0], &commits[4], &c6); // alice from e4 ok

    // ===================== EPOCH 7 (bob updates) =====================
    let c7 = audit_one_transition(
        &mut server,
        &ctx,
        &mut audit_state,
        &c6,
        &[],
        &[(1, "bob-v2")],
        &mut truth,
    );
    assert_eq!(c7.epoch, 7);
    assert_eq!(truth[1].last_change_epoch, 7);
    commits.push(c7.clone());
    user_lookup(&server, &ctx, &truth[1], &c7); // bob — new value
    user_lookup(&server, &ctx, &truth[4], &c7); // eve
    user_lookup(&server, &ctx, &truth[8], &c7); // ivan
    user_consistency(&server, &ctx, &truth[1], &commits[1], &c7); // bob e1->e7 reject
    user_consistency(&server, &ctx, &truth[3], &commits[2], &c7); // dave e2->e7 ok
    user_consistency(&server, &ctx, &truth[7], &commits[6], &c7); // heidi e6->e7 ok

    // ===================== EPOCH 8 =====================
    let c8 = audit_one_transition(
        &mut server,
        &ctx,
        &mut audit_state,
        &c7,
        &[("ken", "ken-v1"), ("lara", "lara-v1")],
        &[],
        &mut truth,
    );
    assert_eq!(c8.epoch, 8);
    commits.push(c8.clone());
    user_lookup(&server, &ctx, &truth[10], &c8);
    user_lookup(&server, &ctx, &truth[11], &c8);

    let mut updated_count = 0usize;
    let mut unchanged_count = 0usize;
    for u in truth.iter().filter(|u| u.signup_epoch < 8) {
        let signup = &commits[u.signup_epoch as usize];
        user_consistency(&server, &ctx, u, signup, &c8);
        if u.last_change_epoch > u.signup_epoch {
            let last = &commits[u.last_change_epoch as usize];
            user_consistency(&server, &ctx, u, last, &c8);
            updated_count += 1;
        } else {
            unchanged_count += 1;
        }
    }
    assert_eq!(updated_count, 3, "alice/carol/bob were updated");
    assert_eq!(unchanged_count, 7);
}
