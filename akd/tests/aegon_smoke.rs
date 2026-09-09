// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! End-to-end smoke test for the AKD-on-Aegon backend.
//!
//! Exercises the Category-1 path (publish, lookup, batch_lookup,
//! get_epoch_hash) plus the Category-3 facade (consistency_proof,
//! verify_lookup, verify_invariance, verify_consistency).

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
        b"AkdAegonSmoke"
    }
}

type Config = ExperimentalConfiguration<TestDomainLabel>;
type AkdDirectory = Directory<Config, AsyncInMemoryDatabase, HardCodedAkdVRF>;

async fn fresh_directory() -> AkdDirectory {
    let db = AsyncInMemoryDatabase::new();
    let storage_manager = StorageManager::new_no_cache(db);
    let vrf = HardCodedAkdVRF {};
    Directory::<Config, _, _>::new(storage_manager, vrf, AzksParallelismConfig::default())
        .await
        .expect("Directory::new")
}

#[tokio::test]
async fn publish_and_lookup_round_trip() {
    let directory = fresh_directory().await;
    let entries = vec![
        (AkdLabel::from("alice"), AkdValue::from("alice-key-v1")),
        (AkdLabel::from("bob"), AkdValue::from("bob-key-v1")),
        (AkdLabel::from("carol"), AkdValue::from("carol-key-v1")),
    ];

    let EpochHash(epoch, _root) = directory.publish(entries.clone()).await.expect("publish");
    assert_eq!(epoch, 1);

    let ctx: VerifierContext = directory.verifier_context().await;
    for (label, expected_value) in &entries {
        let (proof, eh) = directory.lookup(label.clone()).await.expect("lookup");
        assert_eq!(eh.epoch(), 1);
        assert_eq!(&proof.value, expected_value);
        let ok = aegon_facade::verify_lookup(&ctx, label, &proof).expect("verify_lookup");
        assert!(ok, "lookup must verify for {label:?}");
    }
}

#[tokio::test]
async fn batch_lookup_works() {
    let directory = fresh_directory().await;
    let entries: Vec<(AkdLabel, AkdValue)> = (0..4u32)
        .map(|i| {
            (
                AkdLabel::from(format!("u-{i}").as_str()),
                AkdValue::from(format!("v-{i}").as_str()),
            )
        })
        .collect();
    directory.publish(entries.clone()).await.expect("publish");

    let labels: Vec<AkdLabel> = entries.iter().map(|(l, _)| l.clone()).collect();
    let (proofs, eh) = directory.batch_lookup(&labels).await.expect("batch lookup");
    assert_eq!(proofs.len(), labels.len());
    assert_eq!(eh.epoch(), 1);

    let ctx = directory.verifier_context().await;
    for (proof, (label, expected)) in proofs.iter().zip(entries.iter()) {
        assert_eq!(&proof.value, expected);
        let ok = aegon_facade::verify_lookup(&ctx, label, proof).expect("verify_lookup");
        assert!(ok);
    }
}

#[tokio::test]
async fn auditor_walks_invariance_chain() {
    let directory = fresh_directory().await;
    let ctx = directory.verifier_context().await;

    let _ = directory
        .publish(vec![(AkdLabel::from("alice"), AkdValue::from("a1"))])
        .await
        .expect("e1");
    let _ = directory.publish(vec![]).await.expect("idle e2");
    let _ = directory
        .publish(vec![(AkdLabel::from("bob"), AkdValue::from("b1"))])
        .await
        .expect("e3");

    let commits: Vec<EpochCommitment> = directory
        .aegon_epoch_commits(0, 3)
        .await
        .expect("commit chain");
    assert_eq!(commits.len(), 4);

    let mut audit_state = ShardedAuditState::default();
    for i in 0..3 {
        let ok =
            aegon_facade::verify_invariance(&ctx, &mut audit_state, &commits[i], &commits[i + 1])
                .expect("verify_invariance");
        assert!(ok, "invariance must hold for transition {i}");
    }
}

#[tokio::test]
async fn consistency_proof_holds_across_idle_epochs() {
    let directory = fresh_directory().await;
    let ctx = directory.verifier_context().await;
    let label = AkdLabel::from("alice");

    let _ = directory
        .publish(vec![(label.clone(), AkdValue::from("a1"))])
        .await
        .expect("e1");
    let s0: EpochCommitment = directory.epoch_commitment(1).await.expect("e1 retained");

    let _ = directory.publish(vec![]).await.expect("e2 idle");
    let _ = directory
        .publish(vec![(AkdLabel::from("bob"), AkdValue::from("b1"))])
        .await
        .expect("e3 unrelated update");

    let s1 = directory.epoch_commitment(3).await.expect("e3 retained");

    let (_lookup_proof, _) = directory.lookup(label.clone()).await.expect("lookup");
    let proof: ConsistencyProof = directory
        .consistency_proof(&label, 1)
        .await
        .expect("consistency");
    let ok = aegon_facade::verify_consistency(&ctx, &s0, &s1, &label, &proof)
        .expect("verify_consistency");
    assert!(ok);
}

#[tokio::test]
#[should_panic(expected = "AKD-on-Aegon")]
async fn legacy_lookup_verify_panics() {
    let directory = fresh_directory().await;
    let _ = directory
        .publish(vec![(AkdLabel::from("alice"), AkdValue::from("a1"))])
        .await
        .expect("publish");
    let (proof, eh) = directory
        .lookup(AkdLabel::from("alice"))
        .await
        .expect("lookup");
    let _ = akd::client::lookup_verify::<Config>(
        &[],
        eh.hash(),
        eh.epoch(),
        AkdLabel::from("alice"),
        proof,
    );
}

// ---------- history: the AKD `key_history` replacement --------------

fn history_db_path(tag: &str) -> std::path::PathBuf {
    let nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE);
    std::env::temp_dir().join(format!(
        "akd-aegon-history-{tag}-{}-{nonce}",
        std::process::id()
    ))
}

async fn directory_with_store(tag: &str) -> AkdDirectory {
    let db = AsyncInMemoryDatabase::new();
    let storage_manager = StorageManager::new_no_cache(db);
    let vrf = HardCodedAkdVRF {};
    Directory::<Config, _, _>::new_with_db(
        storage_manager,
        vrf,
        AzksParallelismConfig::default(),
        akd::aegon::DbSource::Rocks(history_db_path(tag)),
    )
    .await
    .expect("Directory::new_with_db")
}

/// `value_history` returns one entry per epoch the label changed in,
/// and every entry verifies against the published commitments.
#[tokio::test]
async fn value_history_round_trip() {
    let directory = directory_with_store("value").await;
    let label = AkdLabel::from("alice");

    for v in ["alice-key-v1", "alice-key-v2", "alice-key-v3"] {
        directory
            .publish(vec![(label.clone(), AkdValue::from(v))])
            .await
            .expect("publish");
    }

    let history = directory
        .value_history(&label)
        .await
        .expect("value_history");
    assert_eq!(history.label, label.0);
    assert!(
        !history.entries.is_empty(),
        "three publishes must leave history behind"
    );
    assert!(
        history.freshness.is_some(),
        "a non-empty history carries a freshness attestation"
    );

    let ctx: VerifierContext = directory.verifier_context().await;
    let verified = aegon_facade::verify_value_history(&ctx, &history).expect("history verifies");
    assert_eq!(verified.entry_roots.len(), history.entries.len());
    assert!(verified.live_root.is_some());
}

/// `label_history` returns the placement record, and it verifies.
#[tokio::test]
async fn label_history_round_trip() {
    let directory = directory_with_store("label").await;
    let label = AkdLabel::from("bob");
    directory
        .publish(vec![(label.clone(), AkdValue::from("bob-key-v1"))])
        .await
        .expect("publish");

    let history = directory
        .label_history(&label)
        .await
        .expect("label_history");
    assert_eq!(history.label, label.0);
    assert!(
        history.placement.is_some(),
        "a published label must have a placement record"
    );

    let ctx: VerifierContext = directory.verifier_context().await;
    let verified = aegon_facade::verify_label_history(&ctx, &history).expect("placement verifies");
    assert!(verified.placement_root.is_some());
    assert!(verified.live_root.is_some());
}

/// Without a store, history reports *why* it is empty instead of
/// returning an empty bundle that reads like "this label never
/// changed". This is the trap `DbSource::None` used to set.
#[tokio::test]
async fn history_without_a_store_is_an_error_not_an_empty_answer() {
    let directory = fresh_directory().await;
    let label = AkdLabel::from("alice");
    directory
        .publish(vec![(label.clone(), AkdValue::from("alice-key-v1"))])
        .await
        .expect("publish");

    let err = directory
        .value_history(&label)
        .await
        .expect_err("DbSource::None cannot serve history");
    assert!(
        format!("{err}").contains("requires a key-value store"),
        "error should name the cause, got: {err}"
    );
    assert!(directory.label_history(&label).await.is_err());
}
