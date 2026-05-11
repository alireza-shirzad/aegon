//! End-to-end smoke test for the AKD-on-Aegon backend.
//!
//! Exercises the Category-1 path (publish, lookup, batch_lookup,
//! get_epoch_hash) plus the Category-3 facade (consistency_proof,
//! verify_lookup, verify_invariance, verify_consistency).

use akd::aegon_facade::{
    self, AuditState, ConsistencyProof, EpochCommitment, InvarianceProof, VerifierContext,
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
type AkdDirectory =
    Directory<Config, AsyncInMemoryDatabase, HardCodedAkdVRF>;

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

    let EpochHash(epoch, _root) = directory
        .publish(entries.clone())
        .await
        .expect("publish");
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
    let prev0: EpochCommitment = directory
        .epoch_commitment(0)
        .await
        .expect("epoch 0 retained");

    let _ = directory
        .publish(vec![(AkdLabel::from("alice"), AkdValue::from("a1"))])
        .await
        .expect("e1");
    let _ = directory.publish(vec![]).await.expect("idle e2");
    let _ = directory
        .publish(vec![(AkdLabel::from("bob"), AkdValue::from("b1"))])
        .await
        .expect("e3");

    let chain: Vec<InvarianceProof> = directory
        .aegon_invariance_proofs(0, 3)
        .await
        .expect("audit chain");
    assert_eq!(chain.len(), 3);

    let mut audit_state = AuditState::default();
    let mut prev = prev0;
    for (i, proof) in chain.iter().enumerate() {
        let next = directory
            .epoch_commitment((i + 1) as u64)
            .await
            .expect("commitment retained");
        let ok = aegon_facade::verify_invariance(&ctx, &mut audit_state, &prev, &next, proof)
            .expect("verify_invariance");
        assert!(ok, "invariance must hold for transition {i}");
        prev = next;
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

    let (lookup_proof, _) = directory.lookup(label.clone()).await.expect("lookup");
    let expected_ctr0 = aegon_facade::decode_ctr0(&lookup_proof).expect("decode ctr0");
    let proof: ConsistencyProof = directory
        .consistency_proof(&label, 1)
        .await
        .expect("consistency");
    let ok = aegon_facade::verify_consistency(&ctx, &s0, &s1, &label, expected_ctr0, &proof)
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
    let (proof, eh) = directory.lookup(AkdLabel::from("alice")).await.expect("lookup");
    let _ = akd::client::lookup_verify::<Config>(
        &[],
        eh.hash(),
        eh.epoch(),
        AkdLabel::from("alice"),
        proof,
    );
}
