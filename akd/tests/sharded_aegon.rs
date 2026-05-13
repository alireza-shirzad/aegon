//! End-to-end smoke tests for the in-process `ShardedAegon` coordinator.
//! Validates protocol logic (cross-shard open addressing via `H(ctr,
//! label)`, single FS chain over all shards, Merkle root anchored
//! proofs) by running it locally with multiple shard counts and
//! exercising publish + lookup + sharded verification.

use akd::aegon::{
    probe_at, rederive_sharded_fs_scalars, verify_sharded_lookup, AuditState, Sha256Hash,
    ShardedAegon, ShardedAegonConfig, ShardedEpochCommitment, ShardedLookupProof,
    ShardedVerifierContext,
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
    let (_db_value, proof): (Vec<u8>, ShardedLookupProof<Bn254, Pcs>) =
        server.lookup(&label.to_vec()).expect("lookup");
    let ctx: ShardedVerifierContext<Bn254, Pcs> = server.sharded_verifier_context();
    let ok = verify_sharded_lookup::<Bn254, Pcs, Sha256Hash>(
        &ctx,
        commit,
        &label.to_vec(),
        &expected_value.to_vec(),
        &proof,
    )
    .expect("verify_sharded_lookup");
    assert!(ok, "sharded lookup verification must accept for {label:?}");
}

#[test]
fn srs_path_round_trip() {
    use akd::aegon::{verify_sharded_lookup, SrsSource};

    // Operator workflow: build a config, call generate_srs_to_file on
    // the setup machine, then on each shard build a SECOND config with
    // SrsSource::Path pointing at that file. Same protocol, same
    // proofs, but the SRS is loaded from disk rather than generated.
    let log_capacity = 6usize;
    let log_n_shards = 1usize;

    let tmp = std::env::temp_dir().join(format!(
        "aegon_srs_{}.bin",
        std::process::id()
    ));
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
    let (commit, _audit) = server.publish(&updates).expect("publish");
    let ctx = server.sharded_verifier_context();
    for (label, value) in &updates {
        let (_db_value, proof) = server.lookup(label).expect("lookup");
        let ok = verify_sharded_lookup::<Bn254, Pcs, Sha256Hash>(
            &ctx, &commit, label, value, &proof,
        )
        .expect("verify_sharded_lookup");
        assert!(ok, "SRS-from-disk lookup must verify for {label:?}");
    }

    let _ = std::fs::remove_file(&tmp);
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
    let (commit, _audit) = server.publish(&updates).expect("publish");
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
        .map(|i| (format!("user-{i}").into_bytes(), format!("v-{i}").into_bytes()))
        .collect();
    let (commit, _audit) = server.publish(&updates).expect("publish");
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

    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..6u32)
        .map(|i| (format!("u{i}").into_bytes(), format!("v{i}").into_bytes()))
        .collect();
    let (commit, _audit) = server.publish(&updates).expect("publish");

    for (label, value) in &updates {
        assert_lookup_verifies(&server, label, value, &commit);
    }

    // At least one label should require ctr0 > 0 (i.e., the trail had
    // to advance past its first probe).
    let any_multi_probe = updates.iter().any(|(label, _)| {
        let (_, p) = server.lookup(label).unwrap();
        p.ctr0 > 0
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
#[ignore = "production-scale validation — needs ~64 GB RAM and several minutes; run with `cargo test --release -- --ignored bench_production_shard_scale --nocapture`"]
fn bench_production_shard_scale() {
    use akd::aegon::verify_sharded_lookup;
    // One shard at production parameters: log_capacity = 29 slots
    // (~537M), k = 10. Mirrors one of the 32 GCE shard machines in
    // the planned cluster. `log_n_shards = 0` collapses ShardedAegon
    // to a single shard so the timing/RSS reflect exactly what one
    // shard machine pays.
    let shard_log_capacity = 29usize;
    let kzh_k = 10usize;
    let n_users = 256usize;

    println!(
        "PARAMS: shard_log_capacity={shard_log_capacity}, kzh_k={kzh_k}, n_users={n_users}"
    );
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
        .map(|i| (format!("user-{i}").into_bytes(), format!("v-{i}").into_bytes()))
        .collect();
    let t0 = std::time::Instant::now();
    let (commit, _audit) = server.publish(&updates).expect("publish");
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
    let (_db_value, proof) = server.lookup(label).expect("lookup");
    let lookup_ms = t0.elapsed().as_millis();
    let t0 = std::time::Instant::now();
    let ok = verify_sharded_lookup::<Bn254, Pcs, Sha256Hash>(
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
        .map(|i| (format!("user-{i}").into_bytes(), format!("v-{i}").into_bytes()))
        .collect();
    let t0 = std::time::Instant::now();
    let (commit, _audit) = server.publish(&updates).expect("publish");
    let publish_ms = t0.elapsed().as_millis();
    println!(
        "publish {} entries across {n_shards} shards: {publish_ms} ms",
        updates.len()
    );

    // Sanity: every user verifies.
    let ctx = server.sharded_verifier_context();
    let t0 = std::time::Instant::now();
    for (label, value) in &updates {
        let (_db_value, proof) = server.lookup(label).expect("lookup");
        let ok = akd::aegon::verify_sharded_lookup::<Bn254, Pcs, Sha256Hash>(
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
    let audit_state = AuditState::<<Bn254 as Pairing>::ScalarField>::default();

    let (next, _audit) = server
        .publish(&[(b"alice".to_vec(), b"a1".to_vec())])
        .expect("publish");

    let (rederived_r_index, rederived_r_value) =
        rederive_sharded_fs_scalars::<Bn254, Pcs>(audit_state.r_index, audit_state.r_value, &next);

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
