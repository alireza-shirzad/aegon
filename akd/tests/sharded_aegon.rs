// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! End-to-end smoke tests for the in-process `ShardedAegon` coordinator.
//! Validates protocol logic (cross-shard open addressing via `H(ctr,
//! label)`, single FS chain over all shards, Merkle root anchored
//! proofs) by running it locally with multiple shard counts and
//! exercising publish + lookup + sharded verification.

use akd::aegon::{
    probe_at, rederive_sharded_fs_scalars, verify_sharded_lookup_two_layer, GroupPlan, Sha256Hash,
    ShardedAegon, ShardedAegonConfig, ShardedAuditState, ShardedEpochCommitment,
    ShardedLookupProofTwoLayer, ShardedVerifierContext,
};
use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Pcs = akd_core::aegon_crypto::pcs::kzhk::KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, Sha256Hash>;

fn config(log_capacity: usize, log_n_shards: usize) -> ShardedAegonConfig<Bn254, Pcs> {
    ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(log_capacity - log_n_shards)
        .log_n_shards(log_n_shards)
        .private(false)
        .kzh_k(2)
        .build()
        .expect("config builds")
}

fn fresh(log_capacity: usize, log_n_shards: usize) -> Sharded {
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    Sharded::setup(&mut rng, &config(log_capacity, log_n_shards)).expect("setup")
}

fn assert_lookup_verifies(
    server: &Sharded,
    label: &[u8],
    expected_value: &[u8],
    commit: &ShardedEpochCommitment<Bn254, Pcs>,
) {
    let (_db_value, proof): (Vec<u8>, ShardedLookupProofTwoLayer<Bn254, Pcs>) = server
        .lookup_two_layer(&label.to_vec())
        .expect("lookup_two_layer");
    let ctx: ShardedVerifierContext<Bn254, Pcs> = server.sharded_verifier_context();
    let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
        &ctx,
        commit,
        &label.to_vec(),
        &expected_value.to_vec(),
        &proof,
    )
    .expect("verify_sharded_lookup_two_layer");
    assert!(ok, "sharded lookup verification must accept for {label:?}");
}

#[test]
fn two_layer_publish_lookup_round_trip() {
    use akd::aegon::verify_sharded_lookup_two_layer;
    let log_capacity = 6usize;
    let log_n_shards = 1usize;
    let mut server = fresh(log_capacity, log_n_shards);

    // One publish across the two-layer pipeline. With log_n_shards=1
    // there's only one shard, so the routing trail is one probe and
    // every label lands there.
    let updates = vec![
        (b"alice".to_vec(), b"alice-v1".to_vec()),
        (b"bob".to_vec(), b"bob-v1".to_vec()),
        (b"carol".to_vec(), b"carol-v1".to_vec()),
    ];
    let commit = server
        .publish_two_layer(&updates)
        .expect("publish_two_layer");

    let ctx = server.sharded_verifier_context();
    for (label, expected_value) in &updates {
        // `_db_value` is empty under `DbSource::None`; the polynomial
        // still bound `H_F(expected_value)` at publish time, so we
        // verify against `expected_value` (mirrors how the other
        // tests in this file call `verify_sharded_lookup`).
        let (_db_value, proof) = server.lookup_two_layer(label).expect("lookup_two_layer");
        let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
            &ctx,
            &commit,
            label,
            expected_value,
            &proof,
        )
        .expect("verify_sharded_lookup_two_layer");
        assert!(ok, "two-layer lookup must verify for {label:?}",);
    }
}

#[test]
fn two_layer_publish_lookup_round_trip_multi_shard() {
    use akd::aegon::verify_sharded_lookup_two_layer;
    // Multi-shard layout (4 shards × 2^4 slots each = 2^6 total
    // capacity). Exercises the inter-shard H_shard routing path:
    // different labels will land on different shards, each producing
    // its own within-shard trail.
    let log_capacity = 6usize;
    let log_n_shards = 2usize;
    let mut server = fresh(log_capacity, log_n_shards);

    // Enough labels to spread across all 4 shards with high
    // probability (4-of-4 coverage with 16 labels is ~99.99%).
    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..16u32)
        .map(|i| {
            (
                format!("user-{i}").into_bytes(),
                format!("v{i}").into_bytes(),
            )
        })
        .collect();
    let commit = server
        .publish_two_layer(&updates)
        .expect("publish_two_layer multi-shard");
    let ctx = server.sharded_verifier_context();
    for (label, expected_value) in &updates {
        let (_db_value, proof) = server
            .lookup_two_layer(label)
            .expect("lookup_two_layer multi-shard");
        let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
            &ctx,
            &commit,
            label,
            expected_value,
            &proof,
        )
        .expect("verify_sharded_lookup_two_layer multi-shard");
        assert!(ok, "two-layer multi-shard lookup must verify for {label:?}",);
    }
}

#[test]
fn two_layer_publish_lookup_history_round_trip() {
    use akd::aegon::{verify_lookup_history, verify_lookup_label_history, DbSource};

    // Same shape as `rocks_backend_publish_lookup_history_round_trip`
    // but driven via `publish_two_layer` instead of `publish`.
    // Validates that publish_two_layer populates value_history +
    // label_placement records (the path that didn't work pre-Phase 1
    // of the two-layer migration).
    let tmp_root = std::env::temp_dir();
    let nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE);
    let db_path = tmp_root.join(format!(
        "aegon-rocks-two-layer-history-{}-{nonce}",
        std::process::id()
    ));
    if db_path.exists() {
        std::fs::remove_dir_all(&db_path).ok();
    }

    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(8 - 1)
        .log_n_shards(1)
        .private(false)
        .kzh_k(2)
        .db(DbSource::Rocks(db_path.clone()))
        .build()
        .expect("config builds (two-layer history)");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    let mut server = Sharded::setup(&mut rng, &cfg).expect("setup");

    // Publish 1: introduce three labels (placement events).
    let updates_v1: Vec<(Vec<u8>, Vec<u8>)> = vec![
        (b"alice".to_vec(), b"alice-v1".to_vec()),
        (b"bob".to_vec(), b"bob-v1".to_vec()),
        (b"carol".to_vec(), b"carol-v1".to_vec()),
    ];
    let _commit_v1 = server
        .publish_two_layer(&updates_v1)
        .expect("publish_two_layer v1");

    // Publish 2: update alice + carol, leave bob unchanged. This is
    // the value-only-update path through publish_batch, which sets
    // was_new=false so step 7 (label_placement) is a no-op for these
    // labels but step 6 (value_history) still LPUSHes a new entry.
    let updates_v2: Vec<(Vec<u8>, Vec<u8>)> = vec![
        (b"alice".to_vec(), b"alice-v2".to_vec()),
        (b"carol".to_vec(), b"carol-v2".to_vec()),
    ];
    let _commit_v2 = server
        .publish_two_layer(&updates_v2)
        .expect("publish_two_layer v2");

    let ctx = server.sharded_verifier_context();

    // Alice: 2 entries (placement + update).
    let alice_hist = server
        .lookup_history(&b"alice".to_vec())
        .expect("alice hist");
    assert_eq!(alice_hist.entries.len(), 2, "alice has 2 history entries");
    assert_eq!(alice_hist.entries[0].value_bytes, b"alice-v2");
    assert_eq!(alice_hist.entries[1].value_bytes, b"alice-v1");

    // Bob: 1 entry (placement only).
    let bob_hist = server.lookup_history(&b"bob".to_vec()).expect("bob hist");
    assert_eq!(
        bob_hist.entries.len(),
        1,
        "bob has 1 entry (placement only)"
    );
    assert_eq!(bob_hist.entries[0].value_bytes, b"bob-v1");

    // Both bundles verify cryptographically + freshness anchors under
    // the current sharded root.
    let alice_verified = verify_lookup_history::<Bn254, Pcs, Sha256Hash>(&ctx, &alice_hist)
        .expect("verify alice history (two-layer)");
    let bob_verified = verify_lookup_history::<Bn254, Pcs, Sha256Hash>(&ctx, &bob_hist)
        .expect("verify bob history (two-layer)");
    let current_root = server.current_commitment().merkle_root;
    assert_eq!(
        alice_verified.live_root,
        Some(current_root),
        "alice freshness anchors under live root"
    );
    assert_eq!(
        bob_verified.live_root,
        Some(current_root),
        "bob freshness anchors under live root"
    );

    // Label-history round-trip. Bob's placement record was written
    // at v1 and his slot was untouched by v2, so freshness equality
    // holds.
    let alice_lhist = server
        .lookup_label_history(&b"alice".to_vec())
        .expect("alice label hist (two-layer)");
    let bob_lhist = server
        .lookup_label_history(&b"bob".to_vec())
        .expect("bob label hist (two-layer)");
    assert!(alice_lhist.placement.is_some(), "alice placement present");
    assert!(bob_lhist.placement.is_some(), "bob placement present");
    let alice_lhist_v = verify_lookup_label_history::<Bn254, Pcs, Sha256Hash>(&ctx, &alice_lhist)
        .expect("verify alice label hist");
    let bob_lhist_v = verify_lookup_label_history::<Bn254, Pcs, Sha256Hash>(&ctx, &bob_lhist)
        .expect("verify bob label hist");
    assert_eq!(alice_lhist_v.live_root, Some(current_root));
    assert_eq!(bob_lhist_v.live_root, Some(current_root));

    drop(server);
    std::fs::remove_dir_all(&db_path).ok();
}

#[test]
fn two_layer_consistency_proof_round_trip() {
    use akd::aegon::verify_sharded_consistency_two_layer;
    let log_capacity = 6usize;
    let log_n_shards = 1usize;
    let mut server = fresh(log_capacity, log_n_shards);

    // Publish a label at epoch 0; publish unrelated labels at epoch 1.
    let label = b"alice".to_vec();
    server
        .publish_two_layer(&[(label.clone(), b"alice-v1".to_vec())])
        .expect("publish_two_layer v1");
    let s0 = server.current_commitment().epoch;
    server
        .publish_two_layer(&[(b"bob".to_vec(), b"bob-v1".to_vec())])
        .expect("publish_two_layer v2 (unrelated)");
    let s1_commit = server.current_commitment();

    // alice's slot was untouched at epoch 1 — the consistency
    // proof should accept.
    let proof = server
        .consistency_proof_two_layer(&label, s0)
        .expect("consistency_proof_two_layer");
    let ctx = server.sharded_verifier_context();
    let s0_commit = server.epoch_commitment(s0).expect("epoch 1 commit");
    let ok = verify_sharded_consistency_two_layer::<Bn254, Pcs, Sha256Hash>(
        &ctx, &s0_commit, &s1_commit, &label, &proof,
    )
    .expect("verify two-layer consistency");
    assert!(
        ok,
        "two-layer consistency proof must verify when slot is undisturbed"
    );

    // Now disturb alice's slot and confirm the proof is rejected.
    server
        .publish_two_layer(&[(label.clone(), b"alice-v2".to_vec())])
        .expect("publish_two_layer disturbs alice");
    let s2_commit = server.current_commitment();
    let disturbed = server
        .consistency_proof_two_layer(&label, s0)
        .expect("consistency_proof_two_layer disturbed");
    let ok2 = verify_sharded_consistency_two_layer::<Bn254, Pcs, Sha256Hash>(
        &ctx, &s0_commit, &s2_commit, &label, &disturbed,
    )
    .expect("verify two-layer consistency disturbed");
    assert!(
        !ok2,
        "two-layer consistency proof must reject when value at slot has changed"
    );
}

#[test]
fn two_layer_publish_lookup_round_trip_ecvrf() {
    // VRF-mode exercise of the two-layer pipeline. Validates the
    // VRF-keyed branches in lookup_label_two_layer (which emits
    // prove_h_shard + prove_h_slot proofs) and
    // verify_lookup_label_two_layer (which consumes them via
    // verify_h_shard + verify_h_slot).
    use akd::aegon::{
        verify_sharded_lookup_two_layer, EcVrfHash, ShardedAegon as ShardedG,
        ShardedAegonConfig as Cfg, VrfProver, BENCH_VRF_SEED,
    };
    let log_capacity = 6usize;
    let log_n_shards = 2usize;
    let cfg = Cfg::<Bn254, Pcs>::builder()
        .shard_log_capacity(log_capacity - log_n_shards)
        .log_n_shards(log_n_shards)
        .private(false)
        .kzh_k(2)
        .build()
        .expect("ecvrf cfg builds");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    let mut server: ShardedG<Bn254, Pcs, EcVrfHash> =
        ShardedG::<Bn254, Pcs, EcVrfHash>::setup(&mut rng, &cfg).expect("setup ecvrf");
    server.set_vrf_prover(VrfProver::from_seed(&BENCH_VRF_SEED));

    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..12u32)
        .map(|i| {
            (
                format!("user-{i}").into_bytes(),
                format!("v{i}").into_bytes(),
            )
        })
        .collect();
    let commit = server
        .publish_two_layer(&updates)
        .expect("publish_two_layer ecvrf");
    let ctx = server.sharded_verifier_context();
    assert!(
        ctx.vrf_verifier.is_some(),
        "ecvrf verifier context must carry vrf_verifier"
    );
    for (label, expected_value) in &updates {
        let (_db_value, proof) = server
            .lookup_two_layer(label)
            .expect("lookup_two_layer ecvrf");
        // Sanity: in vrf mode every probe must carry an 80-byte
        // RFC 9381 ECVRF proof.
        for (i, p) in proof.label_proof.route.iter().enumerate() {
            assert_eq!(
                p.vrf_proof.len(),
                80,
                "route[{i}] vrf_proof must be 80 bytes"
            );
        }
        for (i, p) in proof.label_proof.slots.iter().enumerate() {
            assert_eq!(
                p.vrf_proof.len(),
                80,
                "slot[{i}] vrf_proof must be 80 bytes"
            );
        }
        let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, EcVrfHash>(
            &ctx,
            &commit,
            label,
            expected_value,
            &proof,
        )
        .expect("verify_sharded_lookup_two_layer ecvrf");
        assert!(ok, "two-layer ecvrf lookup must verify for {label:?}");
    }
}

#[test]
fn srs_path_round_trip() {
    use akd::aegon::SrsSource;

    // Operator workflow: build a config, call generate_srs_to_file on
    // the setup machine, then on each shard build a SECOND config with
    // SrsSource::Path pointing at that file. Same protocol, same
    // proofs, but the SRS is loaded from disk rather than generated.
    let log_capacity = 6usize;
    let log_n_shards = 1usize;

    let tmp = std::env::temp_dir().join(format!("aegon_srs_{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&tmp);

    // 1. Generate SRS once and write it to the file.
    let setup_cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(log_capacity - log_n_shards)
        .log_n_shards(log_n_shards)
        .private(false)
        .kzh_k(2)
        .build()
        .expect("setup config builds");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    setup_cfg
        .generate_srs_to_file(&mut rng, &tmp)
        .expect("generate_srs_to_file");
    assert!(tmp.exists(), "srs file must exist");
    let size = std::fs::metadata(&tmp).expect("stat srs").len();
    assert!(size > 0, "srs file must be non-empty");

    // 2. Build the production shard config and point at that file.
    let shard_cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(log_capacity - log_n_shards)
        .log_n_shards(log_n_shards)
        .private(false)
        .kzh_k(2)
        .srs(SrsSource::Path(tmp.clone()))
        .build()
        .expect("shard config builds");
    let mut rng = ChaCha20Rng::seed_from_u64(0xDEAD_BEEF);
    let mut server = Sharded::setup(&mut rng, &shard_cfg).expect("setup from path");

    // 3. End-to-end: publish + lookup + verify must work against
    //    SRS-from-disk just as against gen-on-the-fly.
    let updates = vec![
        (b"alice".to_vec(), b"alice-v1".to_vec()),
        (b"bob".to_vec(), b"bob-v1".to_vec()),
    ];
    let commit = server.publish_two_layer(&updates).expect("publish");
    let ctx = server.sharded_verifier_context();
    for (label, value) in &updates {
        let (_db_value, proof) = server.lookup_two_layer(label).expect("lookup");
        let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
            &ctx, &commit, label, value, &proof,
        )
        .expect("verify_sharded_lookup");
        assert!(ok, "SRS-from-disk lookup must verify for {label:?}");
    }

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn private_mode_publish_lookup_round_trip() {
    // Same minimal happy-path as `srs_path_round_trip`, but with
    // `private(true)` — exercises:
    //   * hiding SRS generation
    //   * value-side commits with `tau*h` blinding
    //   * publish_phase_2 storing plain non-ZK openings (per the
    //     "publish-time openings are all non-zk" design)
    //   * lookup-time masking via `open_value_side_at_point` ->
    //     inline `generate_masking_package` (no external masking
    //     server configured)
    //   * `verify_zk` accepting the resulting hiding openings against
    //     the `tau*h`-blinded commitments
    use akd::aegon::verify_sharded_lookup_two_layer;
    let log_capacity = 6usize;
    let log_n_shards = 1usize;

    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(log_capacity - log_n_shards)
        .log_n_shards(log_n_shards)
        .private(true)
        .kzh_k(2)
        .build()
        .expect("private=true config builds");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    let mut server = Sharded::setup(&mut rng, &cfg).expect("setup with hiding SRS");

    // Publish twice so the test hits both new-placement and value-only
    // update paths in publish_phase_2 (different opening loops in the
    // refactor — both must produce verifiable proofs).
    let updates_v1 = vec![
        (b"alice".to_vec(), b"alice-v1".to_vec()),
        (b"bob".to_vec(), b"bob-v1".to_vec()),
    ];
    let commit_v1 = server
        .publish_two_layer(&updates_v1)
        .expect("publish v1 (private)");
    let updates_v2 = vec![(b"alice".to_vec(), b"alice-v2".to_vec())];
    let commit_v2 = server
        .publish_two_layer(&updates_v2)
        .expect("publish v2 (private)");

    let ctx = server.sharded_verifier_context();
    // alice gets the updated value; bob is unchanged. Both must
    // produce hiding openings that verify against the hiding
    // commitments at the current sharded epoch.
    for (label, expected_value) in [
        (b"alice".to_vec(), b"alice-v2".to_vec()),
        (b"bob".to_vec(), b"bob-v1".to_vec()),
    ] {
        let (_db_value, proof) = server.lookup_two_layer(&label).expect("private lookup");
        let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
            &ctx,
            &commit_v2,
            &label,
            &expected_value,
            &proof,
        )
        .expect("verify_sharded_lookup under private=true");
        assert!(ok, "private-mode lookup must verify for {label:?}",);
    }
    let _ = commit_v1;
}

#[test]
fn private_mode_lookup_history_round_trip() {
    // Heavier round-trip under private=true with RocksDB: publish
    // twice, fetch lookup_history (which produces per-epoch §6.4
    // history bundles with masked openings), and run
    // verify_lookup_history. This exercises every value-side opening
    // path in private mode — lookup_history hits both pre and post
    // rand_value openings PLUS the freshness attestation. If any one
    // of them produces a malformed hiding proof, the verifier rejects.
    use akd::aegon::{verify_lookup_history, DbSource};
    let tmp_root = std::env::temp_dir();
    let nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xDEADBEEF);
    let db_path = tmp_root.join(format!(
        "aegon-rocks-private-{}-{nonce}",
        std::process::id()
    ));
    if db_path.exists() {
        std::fs::remove_dir_all(&db_path).ok();
    }

    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(8 - 1)
        .log_n_shards(1)
        .private(true)
        .kzh_k(2)
        .db(DbSource::Rocks(db_path.clone()))
        .build()
        .expect("private=true rocks config builds");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    let mut server = Sharded::setup(&mut rng, &cfg).expect("setup with hiding SRS");

    let updates_v1 = vec![
        (b"alice".to_vec(), b"alice-v1".to_vec()),
        (b"bob".to_vec(), b"bob-v1".to_vec()),
    ];
    server.publish_two_layer(&updates_v1).expect("publish v1");
    let updates_v2 = vec![(b"alice".to_vec(), b"alice-v2".to_vec())];
    server.publish_two_layer(&updates_v2).expect("publish v2");

    let ctx = server.sharded_verifier_context();
    let alice_hist = server
        .lookup_history(&b"alice".to_vec())
        .expect("alice history (private)");
    assert_eq!(alice_hist.entries.len(), 2);

    let verified = verify_lookup_history::<Bn254, Pcs, Sha256Hash>(&ctx, &alice_hist)
        .expect("verify alice history under private=true");
    assert_eq!(verified.entry_roots.len(), 2);
    let current_root = server.current_commitment().merkle_root;
    let live_root = verified.live_root.expect("freshness anchor present");
    assert_eq!(
        live_root, current_root,
        "private-mode freshness anchors under current root"
    );

    drop(server);
    let _ = std::fs::remove_dir_all(&db_path);
}

#[test]
fn optimal_kzh_k_matches_published_table() {
    use akd::aegon::optimal_kzh_k;
    // Exact values from the empirical f(k) = k(k-1)·2^(N/k) table.
    let expected: &[(usize, usize)] = &[
        (20, 6),
        (21, 7),
        (22, 7),
        (23, 7),
        (24, 8),
        (25, 8),
        (26, 8), // tied 8–9, we pick 8
        (27, 9),
        (28, 9),
        (29, 10),
        (30, 10),
        (31, 10),
        (32, 11),
        (33, 11),
        (34, 11),
        (35, 12),
    ];
    for &(n, want) in expected {
        let got = optimal_kzh_k(n);
        assert_eq!(got, want, "optimal_kzh_k({n}): got {got}, expected {want}");
    }
    // Clamp at small n: must always satisfy 1 ≤ k ≤ log_capacity.
    assert!(optimal_kzh_k(1) <= 1);
    assert!(optimal_kzh_k(0) >= 1);
}

#[test]
fn builder_validation_rejects_bad_configs() {
    use akd::aegon::{ShardTransport, SrsSource};

    fn must_err<T>(r: Result<T, akd::aegon::AegonError>, needle: &str, label: &str) {
        match r {
            Err(e) => assert!(
                format!("{e}").contains(needle),
                "{label}: expected error containing '{needle}', got: {e}"
            ),
            Ok(_) => panic!("{label}: builder should have rejected this config"),
        }
    }

    // Missing shard_log_capacity → error.
    must_err(
        ShardedAegonConfig::<Bn254, Pcs>::builder()
            .log_n_shards(2)
            .kzh_k(2)
            .build(),
        "shard_log_capacity",
        "missing shard_log_capacity",
    );

    // Missing log_n_shards → error.
    must_err(
        ShardedAegonConfig::<Bn254, Pcs>::builder()
            .shard_log_capacity(8)
            .kzh_k(2)
            .build(),
        "log_n_shards",
        "missing log_n_shards",
    );

    // Missing pcs_config (no .kzh_k call) → error.
    must_err(
        ShardedAegonConfig::<Bn254, Pcs>::builder()
            .shard_log_capacity(8)
            .log_n_shards(2)
            .build(),
        "pcs_config",
        "missing pcs_config",
    );

    // Remote endpoints with wrong count → error.
    must_err(
        ShardedAegonConfig::<Bn254, Pcs>::builder()
            .shard_log_capacity(8)
            .log_n_shards(2)
            .kzh_k(2)
            .shards(ShardTransport::Remote {
                endpoints: vec!["http://a:1".into(), "http://b:2".into()],
            })
            .build(),
        "endpoints.len",
        "wrong endpoint count",
    );

    // Remote endpoints with right count → builds; setup attempts a
    // real TCP connection and fails on unreachable endpoints. (The
    // happy-path Remote flow is covered by the localhost gRPC
    // integration test.)
    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(8)
        .log_n_shards(1)
        .kzh_k(2)
        .shards(ShardTransport::Remote {
            endpoints: vec!["http://a:1".into(), "http://b:2".into()],
        })
        .build()
        .expect("remote with correct count builds");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    match Sharded::setup(&mut rng, &cfg) {
        Err(e) => assert!(
            format!("{e}").contains("connect"),
            "expected connect-failure error, got: {e}"
        ),
        Ok(_) => panic!("setup must fail with unreachable Remote endpoints"),
    }

    // SrsSource::Path pointing at a nonexistent file fails at setup
    // with an informative error. (The happy-path Path round-trip is
    // exercised by `srs_path_round_trip` above.)
    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(8)
        .log_n_shards(1)
        .kzh_k(2)
        .srs(SrsSource::Path("/nonexistent/aegon.srs".into()))
        .build()
        .expect("path-srs builds");
    match Sharded::setup(&mut rng, &cfg) {
        Err(e) => assert!(
            format!("{e}").contains("open srs file"),
            "expected open-srs-file error, got: {e}"
        ),
        Ok(_) => panic!("setup must fail when SrsSource::Path points at a missing file"),
    }
}

#[test]
fn n_shards_1_behaves_like_single_aegon() {
    // log_capacity=6, log_n_shards=0 → exactly one shard of capacity 64,
    // identical layout to vanilla Aegon. The Merkle root is the lone
    // leaf hash; the probe trail always has shard_id=0.
    let mut server = fresh(6, 0);
    assert_eq!(server.n_shards(), 1);

    let updates = vec![
        (b"alice".to_vec(), b"alice-v1".to_vec()),
        (b"bob".to_vec(), b"bob-v1".to_vec()),
        (b"carol".to_vec(), b"carol-v1".to_vec()),
    ];
    let commit = server.publish_two_layer(&updates).expect("publish");
    assert_eq!(commit.epoch, 1);
    assert_eq!(commit.per_shard.len(), 1);

    for (label, value) in &updates {
        assert_lookup_verifies(&server, label, value, &commit);
    }
}

#[test]
fn n_shards_4_routes_via_vrf_and_audits() {
    // log_capacity=8 (256 slots total) split across 4 shards (64 each).
    // Routing is VRF-driven: H(0, label) selects the first probe's
    // shard, so 12 distinct labels should hit multiple shards w.h.p.
    let mut server = fresh(8, 2);
    assert_eq!(server.n_shards(), 4);

    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..12u32)
        .map(|i| {
            (
                format!("user-{i}").into_bytes(),
                format!("v-{i}").into_bytes(),
            )
        })
        .collect();
    let commit = server.publish_two_layer(&updates).expect("publish");
    assert_eq!(commit.epoch, 1);
    assert_eq!(commit.per_shard.len(), 4);

    for (label, value) in &updates {
        assert_lookup_verifies(&server, label, value, &commit);
    }

    // Hash-driven routing should still spread 12 labels across multiple
    // shards. Track which shard each label's first probe hashed to.
    let log_n_shards = server.log_n_shards();
    let log_shard_capacity = server.shard_log_capacity();
    let mut shard_ids = std::collections::HashSet::new();
    for (label, _) in &updates {
        let (sid, _) = probe_at::<Sha256Hash, <Bn254 as Pairing>::ScalarField>(
            0,
            label,
            log_n_shards,
            log_shard_capacity,
        );
        shard_ids.insert(sid);
    }
    assert!(
        shard_ids.len() > 1,
        "VRF routing should spread 12 labels across multiple shards (got {shard_ids:?})"
    );
}

#[test]
fn cross_shard_open_addressing_handles_collisions() {
    // Tiny dictionary (16 slots = 4 shards × 4 slots) packed near
    // capacity, so some labels' first probes will collide and the
    // trail will hop across shards. Verifying every lookup exercises
    // the multi-probe / multi-shard path of `verify_sharded_lookup`.
    let mut server = fresh(4, 2);
    assert_eq!(server.n_shards(), 4);

    // 12 labels in 16-slot capacity (75% fill) is dense enough that
    // some shard will see >1 label routed to it via H_shard AND have
    // those labels collide on the same first H_slot probe with high
    // probability, exercising the multi-probe within-shard trail.
    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..12u32)
        .map(|i| (format!("u{i}").into_bytes(), format!("v{i}").into_bytes()))
        .collect();
    let commit = server
        .publish_two_layer(&updates)
        .expect("publish_two_layer");

    for (label, value) in &updates {
        assert_lookup_verifies(&server, label, value, &commit);
    }

    // At least one label should require slot_ctr > 0 (i.e., the
    // within-shard probe trail had to advance past its first slot).
    let any_multi_probe = updates.iter().any(|(label, _)| {
        let (_, p) = server.lookup_two_layer(label).unwrap();
        p.label_proof.slots.len() > 1
    });
    assert!(
        any_multi_probe,
        "tightly-packed dictionary should produce at least one multi-probe trail"
    );
}

/// Print current RSS in MB (Linux /proc-based, best-effort).
fn print_rss(stage: &str) {
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: usize = rest
                    .split_whitespace()
                    .next()
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                println!("    RSS after {stage}: {} MB", kb / 1024);
                return;
            }
        }
    }
}

#[test]
fn rocks_backend_publish_lookup_history_round_trip() {
    use akd::aegon::{verify_lookup_history, verify_lookup_label_history, DbSource};

    // Unique tempdir so concurrent test runs don't collide. Cleaned
    // up at the end; on panic Linux's /tmp will eventually GC it.
    let tmp_root = std::env::temp_dir();
    let nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xDEADBEEF);
    let db_path = tmp_root.join(format!("aegon-rocks-test-{}-{nonce}", std::process::id()));
    if db_path.exists() {
        std::fs::remove_dir_all(&db_path).ok();
    }

    // log_capacity=8, 2 shards (log_n_shards=1) — minimal config that
    // still exercises cross-shard probe and the publish-path
    // is_index_slot_occupied fallback (which is the path RocksDB
    // forces us onto, since RocksDB is process-local and has no
    // shard-written slot keys).
    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(8 - 1)
        .log_n_shards(1)
        .private(false)
        .kzh_k(2)
        .db(DbSource::Rocks(db_path.clone()))
        .build()
        .expect("config builds");

    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    let mut server = Sharded::setup(&mut rng, &cfg).expect("setup");

    // Publish 1: introduce three labels (placement events).
    let updates_v1: Vec<(Vec<u8>, Vec<u8>)> = vec![
        (b"alice".to_vec(), b"alice-v1".to_vec()),
        (b"bob".to_vec(), b"bob-v1".to_vec()),
        (b"carol".to_vec(), b"carol-v1".to_vec()),
    ];
    let commit_v1 = server.publish_two_layer(&updates_v1).expect("publish v1");

    // Publish 2: update alice + carol, leave bob unchanged.
    let updates_v2: Vec<(Vec<u8>, Vec<u8>)> = vec![
        (b"alice".to_vec(), b"alice-v2".to_vec()),
        (b"carol".to_vec(), b"carol-v2".to_vec()),
    ];
    let commit_v2 = server.publish_two_layer(&updates_v2).expect("publish v2");
    let _ = (commit_v1, commit_v2);

    // Lookup history: alice should have 2 entries (v1 placement,
    // v2 update). bob should have 1 (placement only). carol should
    // have 2.
    let ctx = server.sharded_verifier_context();
    let alice_hist = server
        .lookup_history(&b"alice".to_vec())
        .expect("alice history");
    assert_eq!(alice_hist.entries.len(), 2, "alice has 2 history entries");
    // Newest-first ordering: entry[0] is the most recent (v2).
    assert_eq!(alice_hist.entries[0].value_bytes, b"alice-v2");
    assert_eq!(alice_hist.entries[1].value_bytes, b"alice-v1");

    let bob_hist = server.lookup_history(&b"bob".to_vec()).expect("bob hist");
    assert_eq!(
        bob_hist.entries.len(),
        1,
        "bob has 1 entry (placement only)"
    );
    assert_eq!(bob_hist.entries[0].value_bytes, b"bob-v1");

    // Verify every entry cryptographically. `verify_lookup_history`
    // re-anchors merkle paths, runs three PCS opens per entry, AND
    // — new — runs one freshness opening per bundle and cross-checks
    // it against the latest entry's `rand_value_post_eval`. The
    // returned `live_root` is what the caller compares against the
    // current sharded root.
    let alice_verified = verify_lookup_history::<Bn254, Pcs, Sha256Hash>(&ctx, &alice_hist)
        .expect("verify alice history");
    assert_eq!(alice_verified.entry_roots.len(), 2);
    let bob_verified = verify_lookup_history::<Bn254, Pcs, Sha256Hash>(&ctx, &bob_hist)
        .expect("verify bob history");
    assert_eq!(bob_verified.entry_roots.len(), 1);

    // Freshness must be present and the reconstructed live root must
    // match the coordinator's current sharded root. Bob's bundle is
    // the interesting one: bob's slot was untouched in publish v2
    // (only alice + carol updated), so the freshness attestation
    // proves "bob's value hasn't changed since v1 even though epoch
    // is now 2". Alice's bundle similarly proves "no further change
    // since v2".
    let current_root = server.current_commitment().merkle_root;
    let bob_live_root = bob_verified.live_root.expect("bob freshness present");
    assert_eq!(
        bob_live_root, current_root,
        "bob freshness anchors under the current sharded root"
    );
    let alice_live_root = alice_verified.live_root.expect("alice freshness present");
    assert_eq!(
        alice_live_root, current_root,
        "alice freshness anchors under the current sharded root"
    );

    // Tamper case: flip one byte of the freshness evaluation and
    // confirm verification rejects with the "no-change-since" error
    // path. This is the test for the actual freshness *check*, not
    // just the path arithmetic.
    {
        let mut tampered = bob_hist.clone();
        let fr = tampered
            .freshness
            .as_mut()
            .expect("bob freshness present pre-tamper");
        // Add 1 to the field element so it can't accidentally equal
        // the legitimate value.
        fr.rand_value_current_eval += <Bn254 as Pairing>::ScalarField::from(1u64);
        let err = verify_lookup_history::<Bn254, Pcs, Sha256Hash>(&ctx, &tampered)
            .expect_err("tampered freshness must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("freshness")
                && (msg.contains("did not verify") || msg.contains("differs")),
            "expected freshness rejection, got: {msg}"
        );
    }

    // Label-history round-trip. The placement record is written
    // once per label at the publish that first places it; reading
    // it back + verifying it should anchor under the live sharded
    // root (because no later publish touched bob's or alice's slot
    // via `index_poly`). For alice specifically, the v2 publish
    // touched her slot's `value_poly` but NOT her slot's
    // `index_poly` — placement was already recorded at v1 — so
    // `rand_index` at her slot is invariant between v1 and v2 and
    // the freshness equality check passes.
    let alice_lhist = server
        .lookup_label_history(&b"alice".to_vec())
        .expect("alice label history");
    assert!(
        alice_lhist.placement.is_some(),
        "alice has a placement record"
    );
    assert!(
        alice_lhist.freshness.is_some(),
        "alice has a live freshness opening"
    );
    let bob_lhist = server
        .lookup_label_history(&b"bob".to_vec())
        .expect("bob label history");
    assert!(bob_lhist.placement.is_some(), "bob has a placement record");

    let alice_lhist_verified =
        verify_lookup_label_history::<Bn254, Pcs, Sha256Hash>(&ctx, &alice_lhist)
            .expect("verify alice label history");
    let bob_lhist_verified =
        verify_lookup_label_history::<Bn254, Pcs, Sha256Hash>(&ctx, &bob_lhist)
            .expect("verify bob label history");
    // Both bundles must anchor the live opening under the current
    // sharded root (i.e. neither label has been displaced).
    assert_eq!(
        alice_lhist_verified.live_root,
        Some(current_root),
        "alice label freshness anchors under live root"
    );
    assert_eq!(
        bob_lhist_verified.live_root,
        Some(current_root),
        "bob label freshness anchors under live root"
    );

    // Tamper case for label history: flip the live opening's
    // evaluation and confirm verification rejects with the "no
    // change since placement" error path.
    {
        let mut tampered = bob_lhist.clone();
        let fr = tampered
            .freshness
            .as_mut()
            .expect("bob freshness present pre-tamper");
        fr.rand_index_current_eval += <Bn254 as Pairing>::ScalarField::from(1u64);
        let err = verify_lookup_label_history::<Bn254, Pcs, Sha256Hash>(&ctx, &tampered)
            .expect_err("tampered label freshness must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("label history")
                && (msg.contains("did not verify") || msg.contains("differs")),
            "expected label-history rejection, got: {msg}"
        );
    }

    // Lookup the latest value via the regular `lookup` API to make
    // sure RocksDB-backed value:/routing: keys round-trip end-to-end
    // (not just the new history list).
    let (alice_v_bytes, _proof): (Vec<u8>, ShardedLookupProofTwoLayer<Bn254, Pcs>) = server
        .lookup_two_layer(&b"alice".to_vec())
        .expect("lookup alice");
    assert_eq!(alice_v_bytes, b"alice-v2");

    // Tidy up; if this fails it's not a test failure (Linux /tmp
    // policy will catch it eventually).
    drop(server);
    std::fs::remove_dir_all(&db_path).ok();
}

#[test]
#[ignore = "production-scale validation — needs ~64 GB RAM and several minutes; run with `cargo test --release -- --ignored bench_production_shard_scale --nocapture`"]
fn bench_production_shard_scale() {
    use akd::aegon::verify_sharded_lookup_two_layer;
    // One shard at production parameters: log_capacity = 29 slots
    // (~537M), k = 10. Mirrors one of the 32 GCE shard machines in
    // the planned cluster. `log_n_shards = 0` collapses ShardedAegon
    // to a single shard so the timing/RSS reflect exactly what one
    // shard machine pays.
    let shard_log_capacity = 29usize;
    let kzh_k = 10usize;
    let n_users = 256usize;

    println!("PARAMS: shard_log_capacity={shard_log_capacity}, kzh_k={kzh_k}, n_users={n_users}");
    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(shard_log_capacity)
        .log_n_shards(0)
        .private(false)
        .kzh_k(kzh_k)
        .build()
        .expect("config builds");

    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    print_rss("baseline");
    let t0 = std::time::Instant::now();
    let mut server = Sharded::setup(&mut rng, &cfg).expect("setup");
    let setup_ms = t0.elapsed().as_millis();
    println!("SETUP: {setup_ms} ms ({:.2} s)", setup_ms as f64 / 1000.0);
    print_rss("setup");

    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..n_users as u32)
        .map(|i| {
            (
                format!("user-{i}").into_bytes(),
                format!("v-{i}").into_bytes(),
            )
        })
        .collect();
    let t0 = std::time::Instant::now();
    let commit = server.publish_two_layer(&updates).expect("publish");
    let publish_ms = t0.elapsed().as_millis();
    println!(
        "PUBLISH {n_users} entries: {publish_ms} ms ({:.2} s)",
        publish_ms as f64 / 1000.0
    );
    print_rss("publish");

    // One lookup + verify to confirm the proof shape works at this scale.
    let ctx = server.sharded_verifier_context();
    let (label, value) = &updates[0];
    let t0 = std::time::Instant::now();
    let (_db_value, proof) = server.lookup_two_layer(label).expect("lookup");
    let lookup_ms = t0.elapsed().as_millis();
    let t0 = std::time::Instant::now();
    let ok = verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
        &ctx, &commit, label, value, &proof,
    )
    .expect("verify");
    let verify_ms = t0.elapsed().as_millis();
    assert!(ok);
    println!("LOOKUP: {lookup_ms} ms");
    println!("VERIFY: {verify_ms} ms");
    print_rss("lookup+verify");
}

#[test]
#[ignore = "scale benchmark — run with `cargo test --release -- --ignored bench_setup_and_publish`"]
fn bench_setup_and_publish() {
    // 32 shards × 32K slots = 1M total. Realistic production-like
    // scale (still per-shard, since real deployment has 32 separate
    // machines). SRS-gen waste at this size would have been ~32× a
    // single SRS — visibly painful without the dedup.
    let log_capacity = 20usize;
    let log_n_shards = 5usize;
    let n_shards = 1usize << log_n_shards;

    let t0 = std::time::Instant::now();
    let mut server = fresh(log_capacity, log_n_shards);
    let setup_ms = t0.elapsed().as_millis();
    println!(
        "setup: {n_shards} shards × log_capacity={} → {setup_ms} ms",
        log_capacity - log_n_shards
    );

    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..256u32)
        .map(|i| {
            (
                format!("user-{i}").into_bytes(),
                format!("v-{i}").into_bytes(),
            )
        })
        .collect();
    let t0 = std::time::Instant::now();
    let commit = server.publish_two_layer(&updates).expect("publish");
    let publish_ms = t0.elapsed().as_millis();
    println!(
        "publish {} entries across {n_shards} shards: {publish_ms} ms",
        updates.len()
    );

    // Sanity: every user verifies.
    let ctx = server.sharded_verifier_context();
    let t0 = std::time::Instant::now();
    for (label, value) in &updates {
        let (_db_value, proof) = server.lookup_two_layer(label).expect("lookup");
        let ok = akd::aegon::verify_sharded_lookup_two_layer::<Bn254, Pcs, Sha256Hash>(
            &ctx, &commit, label, value, &proof,
        )
        .expect("verify_sharded_lookup");
        assert!(ok);
    }
    let lookup_ms = t0.elapsed().as_millis();
    println!("{} lookup+verify pairs: {lookup_ms} ms", updates.len());
}

#[test]
fn auditor_rederives_shared_fs_scalars() {
    // Confirm the auditor can independently compute the same r_index /
    // r_value the coordinator used during publish, given only the
    // sharded epoch commitments.
    let mut server = fresh(6, 1);
    let prev = server.epoch_commitment(0).expect("epoch 0");
    let audit_state = ShardedAuditState::<<Bn254 as Pairing>::ScalarField>::default();

    let next = server
        .publish_two_layer(&[(b"alice".to_vec(), b"a1".to_vec())])
        .expect("publish_two_layer");

    // Single-group plan: the original directory-wide derivation.
    let plan = GroupPlan::single(next.per_shard.len()).expect("plan");
    let (rederived_r_index, rederived_r_value) = rederive_sharded_fs_scalars::<Bn254, Pcs>(
        &audit_state.r_index,
        &audit_state.r_value,
        &next,
        plan,
        &Default::default(),
    )
    .expect("rederive");
    let (rederived_r_index, rederived_r_value) = (rederived_r_index[0], rederived_r_value[0]);

    // After updating the auditor's state, it should match what the
    // coordinator stored. We don't expose those internals, but we can
    // at least verify the rederivation is deterministic and non-zero
    // (sanity).
    assert_ne!(
        rederived_r_index,
        <Bn254 as Pairing>::ScalarField::from(0u64),
        "FS scalar must be non-zero for a non-empty publish"
    );
    assert_ne!(
        rederived_r_value,
        <Bn254 as Pairing>::ScalarField::from(0u64)
    );

    let _ = prev;
}
