// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! Contains the tests for error conditions and invariants that should be upheld
//! by the API.

use akd_core::configuration::Configuration;
use std::default::Default;

use crate::append_only_zks::AzksParallelismConfig;
use crate::storage::types::KeyData;
use crate::tree_node::TreeNodeWithPreviousValue;
use crate::{
    auditor::audit_verify,
    client::{key_history_verify, lookup_verify},
    directory::{Directory, ReadOnlyDirectory},
    ecvrf::{HardCodedAkdVRF, VRFKeyStorage},
    errors::{AkdError, DirectoryError, StorageError},
    storage::{
        manager::StorageManager, memory::AsyncInMemoryDatabase, types::DbRecord, types::ValueState,
        Database,
    },
    test_config,
    tests::{setup_mocked_db, MockLocalDatabase},
    AkdLabel, AkdValue, Azks, EpochHash, HistoryParams, HistoryVerificationParams, NodeLabel,
};

// This test is meant to test the function poll_for_azks_change
// which is meant to detect changes in the azks, to prevent inconsistencies
// between the local cache and storage.
test_config!(test_directory_polling_azks_change);
async fn test_directory_polling_azks_change<TC: Configuration>() -> Result<(), AkdError> {
    let db = AsyncInMemoryDatabase::new();
    let storage = StorageManager::new(db, None, None, None);
    let vrf = HardCodedAkdVRF {};
    // writer will write the AZKS record
    let writer = Directory::<TC, _, _>::new(
        storage.clone(),
        vrf.clone(),
        AzksParallelismConfig::default(),
    )
    .await?;

    writer
        .publish(vec![
            (AkdLabel::from("hello"), AkdValue::from("world")),
            (AkdLabel::from("hello2"), AkdValue::from("world2")),
        ])
        .await?;

    // reader will not write the AZKS but will be "polling" for AZKS changes
    let reader =
        ReadOnlyDirectory::<TC, _, _>::new(storage, vrf, AzksParallelismConfig::default()).await?;

    // start the poller
    let (tx, mut rx) = tokio::sync::mpsc::channel(10);
    let reader_clone = reader.clone();
    let _join_handle = tokio::task::spawn(async move {
        reader_clone
            .poll_for_azks_changes(tokio::time::Duration::from_millis(100), Some(tx))
            .await
    });

    // wait for a second to make sure the poller has started
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // verify a lookup proof, which will populate the cache
    async_poll_helper_proof(&reader, AkdValue::from("world")).await?;

    // publish epoch 2
    writer
        .publish(vec![
            (AkdLabel::from("hello"), AkdValue::from("world_2")),
            (AkdLabel::from("hello2"), AkdValue::from("world2_2")),
        ])
        .await?;

    // assert that the change is picked up in a reasonable time-frame and the cache is flushed
    let notification = tokio::time::timeout(tokio::time::Duration::from_secs(10), rx.recv()).await;
    assert!(matches!(notification, Ok(Some(()))));

    async_poll_helper_proof(&reader, AkdValue::from("world_2")).await?;

    Ok(())
}

// A test to ensure that any database error at the time a Directory is created
// does not automatically attempt to create a new aZKS. Only aZKS not found errors
// should assume that a successful read happened and no aZKS exists.
test_config!(test_directory_azks_bootstrapping);
async fn test_directory_azks_bootstrapping<TC: Configuration>() -> Result<(), AkdError> {
    let vrf = HardCodedAkdVRF {};

    // Verify that a Storage error results in an error when attempting to create the Directory
    let mut mock_db = MockLocalDatabase {
        ..Default::default()
    };
    mock_db
        .expect_get::<Azks>()
        .returning(|_| Err(StorageError::Connection("Fire!".to_string())));
    mock_db.expect_set().times(0);
    let storage = StorageManager::new_no_cache(mock_db);

    let maybe_akd =
        Directory::<TC, _, _>::new(storage, vrf.clone(), AzksParallelismConfig::default()).await;
    assert!(maybe_akd.is_err());

    // Verify that an aZKS not found error results in one being created with the Directory
    // We're creating an empty directory here, so this is the expected behavior for NotFound
    let mut mock_db = MockLocalDatabase {
        ..Default::default()
    };
    let test_db = AsyncInMemoryDatabase::new();
    setup_mocked_db(&mut mock_db, &test_db);
    let storage = StorageManager::new_no_cache(mock_db);

    let maybe_akd =
        Directory::<TC, _, _>::new(storage, vrf, AzksParallelismConfig::default()).await;
    assert!(maybe_akd.is_ok());

    let akd = maybe_akd.expect("Failed to get create a Directory!");
    let azks = akd.retrieve_azks().await.expect("Failed to get aZKS!");
    assert_eq!(0, azks.get_latest_epoch());

    Ok(())
}

// It is possible to perform a "dirty read" when reading states during a key history operation
// that will result in an epoch from the dirty read being higher than the aZKS epoch. In such an
// event, we ignore value states that are part of the dirty read. This test ensures that we do not
// inadvertently panic when inspecting marker versions due to "start version" and "end version"
// invariants being violated.
test_config!(test_key_history_dirty_reads);
async fn test_key_history_dirty_reads<TC: Configuration>() -> Result<(), AkdError> {
    let committed_epoch = 10;
    let dirty_epoch = 11;

    let mut mock_db = MockLocalDatabase::default();
    mock_db.expect_get::<Azks>().returning(move |_| {
        Ok(DbRecord::Azks(Azks {
            latest_epoch: committed_epoch,
            num_nodes: 1,
        }))
    });
    mock_db.expect_get_user_data().returning(move |_| {
        Ok(KeyData {
            states: vec![ValueState {
                value: AkdValue(Vec::new()),
                version: 2,
                label: NodeLabel {
                    label_val: [0u8; 32],
                    label_len: 32,
                },
                epoch: dirty_epoch,
                username: AkdLabel::from("ferris"),
            }],
        })
    });
    // We can just return some fake error at this point, as we're not validating
    // actual history proof functionality.
    mock_db
        .expect_get::<TreeNodeWithPreviousValue>()
        .returning(|_| Err(StorageError::Other("Fake!".to_string())));

    let storage = StorageManager::new_no_cache(mock_db);
    let vrf = HardCodedAkdVRF {};
    let akd = Directory::<TC, _, _>::new(storage, vrf, AzksParallelismConfig::default()).await?;

    // Ensure that we do not panic in this scenario, so we can just ignore the result.
    let _res = akd
        .key_history(&AkdLabel::from("ferris"), HistoryParams::MostRecent(1))
        .await;

    Ok(())
}

test_config!(test_read_during_publish);
async fn test_read_during_publish<TC: Configuration>() -> Result<(), AkdError> {
    let db = AsyncInMemoryDatabase::new();
    let storage = StorageManager::new_no_cache(db.clone());
    let vrf = HardCodedAkdVRF {};
    let akd = Directory::<TC, _, _>::new(storage, vrf, AzksParallelismConfig::default()).await?;

    // Publish once
    akd.publish(vec![
        (AkdLabel::from("hello"), AkdValue::from("world")),
        (AkdLabel::from("hello2"), AkdValue::from("world2")),
    ])
    .await
    .unwrap();
    // Get the root hash after the first publish
    let root_hash_1 = akd.get_epoch_hash().await?.1;
    // Publish updates for the same labels.
    akd.publish(vec![
        (AkdLabel::from("hello"), AkdValue::from("world_2")),
        (AkdLabel::from("hello2"), AkdValue::from("world2_2")),
    ])
    .await
    .unwrap();

    // Get the root hash after the second publish
    let root_hash_2 = akd.get_epoch_hash().await?.1;

    // Make the current azks a "checkpoint" to reset to later
    let checkpoint_azks = akd.retrieve_azks().await.unwrap();

    // Publish for the third time with a new label
    akd.publish(vec![
        (AkdLabel::from("hello"), AkdValue::from("world_3")),
        (AkdLabel::from("hello2"), AkdValue::from("world2_3")),
        (AkdLabel::from("hello3"), AkdValue::from("world3")),
    ])
    .await
    .unwrap();

    // Reset the azks record back to previous epoch, to emulate an akd reader
    // communicating with storage that is in the middle of a publish operation
    db.set(DbRecord::Azks(checkpoint_azks))
        .await
        .expect("Error resetting directory to previous epoch");

    // re-create the directory instance so it refreshes from storage
    let storage = StorageManager::new_no_cache(db.clone());
    let vrf = HardCodedAkdVRF {};
    let akd = ReadOnlyDirectory::<TC, _, _>::new(storage, vrf, AzksParallelismConfig::default())
        .await
        .unwrap();

    // Get the VRF public key
    let vrf_pk = akd.get_public_key().await.unwrap();

    // Lookup proof should contain the checkpoint epoch's value and still verify
    let (lookup_proof, root_hash) = akd.lookup(AkdLabel::from("hello")).await.unwrap();
    assert_eq!(AkdValue::from("world_2"), lookup_proof.value);
    lookup_verify::<TC>(
        vrf_pk.as_bytes(),
        root_hash.hash(),
        root_hash.epoch(),
        AkdLabel::from("hello"),
        lookup_proof,
    )
    .unwrap();

    // History proof should not contain the third epoch's update but still verify
    let (history_proof, root_hash) = akd
        .key_history(&AkdLabel::from("hello"), HistoryParams::default())
        .await
        .unwrap();
    key_history_verify::<TC>(
        vrf_pk.as_bytes(),
        root_hash.hash(),
        root_hash.epoch(),
        AkdLabel::from("hello"),
        history_proof,
        HistoryVerificationParams::default(),
    )
    .unwrap();

    // Lookup proof for the most recently added key (ahead of directory epoch) should
    // result in the entry not being found.
    let recently_added_lookup_result = akd.lookup(AkdLabel::from("hello3")).await;
    assert!(matches!(
        recently_added_lookup_result,
        Err(AkdError::Storage(StorageError::NotFound(_)))
    ));

    // History proof for the most recently added key (ahead of directory epoch) should
    // result in the entry not being found.
    let recently_added_history_result = akd
        .key_history(&AkdLabel::from("hello3"), HistoryParams::default())
        .await;
    assert!(matches!(
        recently_added_history_result,
        Err(AkdError::Storage(StorageError::NotFound(_)))
    ));

    // Audit proof should only work up until checkpoint's epoch
    let audit_proof = akd.audit(1, 2).await.unwrap();
    audit_verify::<TC>(vec![root_hash_1, root_hash_2], audit_proof)
        .await
        .unwrap();

    let invalid_audit = akd.audit(2, 3).await;
    assert!(invalid_audit.is_err());

    Ok(())
}

// The read-only mode of a directory is meant to simply read from memory.
// This test makes sure it throws errors appropriately, i.e. when trying to
// write to a read-only directory and when trying to read a directory when none
// exists in storage.
test_config!(test_directory_read_only_mode);
async fn test_directory_read_only_mode<TC: Configuration>() -> Result<(), AkdError> {
    let db = AsyncInMemoryDatabase::new();
    let storage = StorageManager::new_no_cache(db);
    let vrf = HardCodedAkdVRF {};
    // There is no AZKS object in the storage layer, directory construction should fail
    let akd =
        ReadOnlyDirectory::<TC, _, _>::new(storage, vrf, AzksParallelismConfig::default()).await;
    assert!(akd.is_err());

    Ok(())
}

// Test for attempting to publish duplicate entries as updates to the directory
test_config!(test_publish_duplicate_entries);
async fn test_publish_duplicate_entries<TC: Configuration>() -> Result<(), AkdError> {
    let db = AsyncInMemoryDatabase::new();
    let storage = StorageManager::new_no_cache(db);
    let vrf = HardCodedAkdVRF {};
    let akd =
        Directory::<TC, _, _>::new(storage, vrf.clone(), AzksParallelismConfig::default()).await?;

    // Create a set of updates
    let mut updates = vec![];
    for i in 0..10 {
        updates.push((
            AkdLabel(format!("hello1{i}").as_bytes().to_vec()),
            AkdValue(format!("hello1{i}").as_bytes().to_vec()),
        ));
    }

    // Now add a duplicate entry
    updates.push(updates[0].clone());

    // Attempt to publish -- this should throw an error because of the duplicate entry
    let Err(AkdError::Directory(DirectoryError::Publish(_))) = akd.publish(updates).await else {
        panic!("Expected a directory publish error");
    };

    Ok(())
}

// Verifies that the Aegon verifier rejects two flavors of a malicious
// server. Preserves the intent of the original Merkle/SEEMless
// `test_malicious_key_history` (server lies about history → client
// catches it) on the path that the current backend actually serves:
//
//   1. A lookup served with forged value bytes against an otherwise
//      honest commitment must fail value-side verification (analogous
//      to "UnmarkedStaleVersion" — server publishes a new value but
//      tries to convince the client an old value still holds).
//   2. An auditor walking the epoch chain must reject when one
//      epoch's `value_commitment` is tampered (analogous to
//      "MarkVersionStale one epoch late" — server lies about what the
//      transition contained, the chain homomorphism breaks).
test_config!(test_malicious_key_history);
async fn test_malicious_key_history<TC: Configuration>() -> Result<(), AkdError> {
    let db = AsyncInMemoryDatabase::new();
    let storage = StorageManager::new_no_cache(db);
    let vrf = HardCodedAkdVRF {};
    let akd = Directory::<TC, _, _>::new(storage, vrf, AzksParallelismConfig::default()).await?;

    // Honest baseline. We need every shard to be exercised in both
    // publishes so that scenario (3) below is a real tamper: with 4
    // default shards, publishing a single label leaves 3 shards' poly
    // commitments at the SRS identity, and swapping idle shards is a
    // no-op the auditor can't detect. Publishing a batch with enough
    // distinct labels in both epochs makes every per-shard
    // value_commitment distinct.
    let epoch1_batch: Vec<(AkdLabel, AkdValue)> = (0..16)
        .map(|i| (AkdLabel::from(format!("u{i}").as_str()), AkdValue::from(format!("v{i}-1").as_str())))
        .collect();
    let epoch2_batch: Vec<(AkdLabel, AkdValue)> = (0..16)
        .map(|i| (AkdLabel::from(format!("u{i}").as_str()), AkdValue::from(format!("v{i}-2").as_str())))
        .collect();
    akd.publish(epoch1_batch).await?;
    akd.publish(epoch2_batch).await?;

    let ctx = akd.verifier_context().await;

    // (1) Honest lookup verifies (positive sanity check).
    let target_label = AkdLabel::from("u0");
    let (honest_proof, _eh) = akd.lookup(target_label.clone()).await?;
    assert_eq!(AkdValue::from("v0-2"), honest_proof.value);
    assert!(
        crate::aegon_facade::verify_lookup(&ctx, &target_label, &honest_proof)?,
        "honest lookup must verify",
    );

    // (2) Malicious server scenario A — forged value: same commitment
    // and aegon proof bytes, but the wire claims a different value
    // than what was published. The verifier must reject — either by
    // returning Ok(false) or by surfacing an Err from the value-side
    // opening check (`H_F(value) != evaluation`). Both shapes mean
    // "client caught the lie."
    {
        let mut tampered = honest_proof.clone();
        tampered.value = AkdValue::from("FORGED");
        let verdict = crate::aegon_facade::verify_lookup(&ctx, &target_label, &tampered);
        assert!(
            matches!(verdict, Ok(false) | Err(_)),
            "verifier must reject a lookup whose claimed value disagrees with the proof, got {verdict:?}",
        );
    }

    // (3) Malicious server scenario B — tampered epoch-2 commitment:
    // overwrite the published value_commitment on shard 0 with one
    // that didn't come from an honest publish. The audit-invariance
    // check walks `value_commitment` across the chain via the
    // commitment-homomorphism path; the tampered commitment must
    // trip that check.
    //
    // The auditor's chain Fiat-Shamir scalar `r_value` is threaded
    // across transitions, so the audit_state for the 1->2 step must
    // be obtained by first auditing 0->1. Starting from
    // `AuditState::default()` directly at epoch 1 would use the wrong
    // r_value and even the honest transition would reject.
    {
        let epoch0 = akd.epoch_commitment(0).await.expect("epoch 0 commitment retained");
        let epoch1 = akd.epoch_commitment(1).await.expect("epoch 1 commitment retained");
        let honest_epoch2 = akd.epoch_commitment(2).await.expect("epoch 2 commitment retained");

        // Walk 0 -> 1 honestly to advance the audit state. Sanity-check
        // that the honest 1 -> 2 transition is accepted from that state.
        let mut audit_state = crate::aegon_facade::ShardedAuditState::default();
        assert!(
            crate::aegon_facade::verify_invariance(&ctx, &mut audit_state, &epoch0, &epoch1)?,
            "honest epoch 0 -> 1 transition must pass",
        );
        let baseline_state = audit_state.clone();
        let mut honest_check_state = baseline_state.clone();
        assert!(
            crate::aegon_facade::verify_invariance(
                &ctx,
                &mut honest_check_state,
                &epoch1,
                &honest_epoch2,
            )?,
            "honest epoch 1 -> 2 transition must pass",
        );

        // Construct a tampered epoch-2 commitment by overwriting shard 0's
        // `value_commitment` with one that didn't come from honest publish.
        let mut tampered_shards = honest_epoch2.per_shard.clone();
        if tampered_shards.len() >= 2 {
            tampered_shards[0].value_commitment = tampered_shards[1].value_commitment.clone();
        } else {
            // Single-shard case: replay the previous epoch's value_commitment,
            // pretending epoch 2's publish was a no-op on the value chain.
            tampered_shards[0].value_commitment = epoch1.per_shard[0].value_commitment.clone();
        }
        let tampered_epoch2 = crate::aegon::ShardedEpochCommitment::with_per_shard(
            honest_epoch2.epoch,
            tampered_shards,
        );

        let mut tampered_check_state = baseline_state.clone();
        let verdict = crate::aegon_facade::verify_invariance(
            &ctx,
            &mut tampered_check_state,
            &epoch1,
            &tampered_epoch2,
        );
        assert!(
            matches!(verdict, Ok(false) | Err(_)),
            "auditor must reject a transition whose value_commitment was tampered, got {verdict:?}",
        );
    }

    Ok(())
}

// Test key history verification for error handling of malformed key history proofs
test_config!(test_key_history_verify_malformed);
async fn test_key_history_verify_malformed<TC: Configuration>() -> Result<(), AkdError> {
    let db = AsyncInMemoryDatabase::new();
    let storage = StorageManager::new_no_cache(db);
    let vrf = HardCodedAkdVRF {};
    let akd =
        Directory::<TC, _, _>::new(storage, vrf.clone(), AzksParallelismConfig::default()).await?;

    let mut rng = rand::rngs::OsRng;
    for _ in 0..100 {
        let mut updates = vec![];
        updates.push((
            AkdLabel("label".to_string().as_bytes().to_vec()),
            AkdValue::random(&mut rng),
        ));
        akd.publish(updates.clone()).await?;
    }

    for _ in 0..100 {
        let mut updates = vec![];
        updates.push((
            AkdLabel("another label".to_string().as_bytes().to_vec()),
            AkdValue::random(&mut rng),
        ));
        akd.publish(updates.clone()).await?;
    }

    // Get the latest root hash
    let EpochHash(current_epoch, root_hash) = akd.get_epoch_hash().await?;
    // Get the VRF public key
    let vrf_pk = akd.get_public_key().await?;
    let target_label = AkdLabel("label".to_string().as_bytes().to_vec());

    let history_params_5 = HistoryParams::MostRecent(5);

    let (key_history_proof, _) = akd.key_history(&target_label, history_params_5).await?;

    let correct_verification_params = HistoryVerificationParams::Default {
        history_params: history_params_5,
    };

    // Normal verification should succeed
    key_history_verify::<TC>(
        vrf_pk.as_bytes(),
        root_hash,
        current_epoch,
        target_label.clone(),
        key_history_proof.clone(),
        correct_verification_params,
    )?;

    // Using an inconsistent set of history parameters should fail
    for bad_params in [
        HistoryParams::MostRecent(1),
        HistoryParams::MostRecent(4),
        HistoryParams::MostRecent(6),
        HistoryParams::default(),
    ] {
        assert!(key_history_verify::<TC>(
            vrf_pk.as_bytes(),
            root_hash,
            current_epoch,
            target_label.clone(),
            key_history_proof.clone(),
            HistoryVerificationParams::Default {
                history_params: bad_params
            },
        )
        .is_err());
    }

    let mut malformed_proof_1 = key_history_proof.clone();
    malformed_proof_1.past_marker_vrf_proofs = key_history_proof.past_marker_vrf_proofs
        [..key_history_proof.past_marker_vrf_proofs.len() - 1]
        .to_vec();
    let mut malformed_proof_2 = key_history_proof.clone();
    malformed_proof_2.existence_of_past_marker_proofs = key_history_proof
        .existence_of_past_marker_proofs
        [..key_history_proof.existence_of_past_marker_proofs.len() - 1]
        .to_vec();
    let mut malformed_proof_3 = key_history_proof.clone();
    malformed_proof_3.future_marker_vrf_proofs = key_history_proof.future_marker_vrf_proofs
        [..key_history_proof.future_marker_vrf_proofs.len() - 1]
        .to_vec();
    let mut malformed_proof_4 = key_history_proof.clone();
    malformed_proof_4.non_existence_of_future_marker_proofs = key_history_proof
        .non_existence_of_future_marker_proofs[..key_history_proof
        .non_existence_of_future_marker_proofs
        .len()
        - 1]
        .to_vec();

    // Malformed proof verification should fail
    for malformed_proof in [
        malformed_proof_1,
        malformed_proof_2,
        malformed_proof_3,
        malformed_proof_4,
    ] {
        assert!(key_history_verify::<TC>(
            vrf_pk.as_bytes(),
            root_hash,
            current_epoch,
            target_label.clone(),
            malformed_proof,
            correct_verification_params
        )
        .is_err());
    }

    let mut malformed_proof_start_version_is_zero = key_history_proof.clone();
    malformed_proof_start_version_is_zero.update_proofs[0].epoch = 0;
    let mut malformed_proof_end_version_exceeds_epoch = key_history_proof.clone();
    malformed_proof_end_version_exceeds_epoch.update_proofs[0].epoch = current_epoch + 1;

    // Malformed proof verification should fail
    for malformed_proof in [
        malformed_proof_start_version_is_zero,
        malformed_proof_end_version_exceeds_epoch,
    ] {
        assert!(key_history_verify::<TC>(
            vrf_pk.as_bytes(),
            root_hash,
            current_epoch,
            target_label.clone(),
            malformed_proof,
            correct_verification_params,
        )
        .is_err());
    }

    Ok(())
}

// Test lookup_verify where version number exceeds epoch (and it should throw an error)
test_config!(test_lookup_verify_invalid_version_number);
async fn test_lookup_verify_invalid_version_number<TC: Configuration>() -> Result<(), AkdError> {
    let db = AsyncInMemoryDatabase::new();
    let storage = StorageManager::new_no_cache(db);
    let vrf = HardCodedAkdVRF {};
    // epoch 0
    let akd =
        Directory::<TC, _, _>::new(storage, vrf.clone(), AzksParallelismConfig::default()).await?;

    // Create a set with 2 updates, (label, value) pairs
    // ("hello10", "hello10")
    // ("hello11", "hello11")
    let mut updates = vec![];
    for i in 0..2 {
        updates.push((
            AkdLabel(format!("hello1{i}").as_bytes().to_vec()),
            AkdValue(format!("hello1{i}").as_bytes().to_vec()),
        ));
    }
    // Repeatedly publish the updates. Afterwards, the akd's epoch will be 10.
    for _ in 0..10 {
        akd.publish(updates.clone()).await?;
    }

    // The label we will lookup is "hello10"
    let target_label = AkdLabel(format!("hello1{}", 0).as_bytes().to_vec());

    // retrieve the lookup proof
    let (lookup_proof, root_hash) = akd.lookup(target_label.clone()).await?;

    // Get the VRF public key
    let vrf_pk = vrf.get_vrf_public_key().await?;

    let akd_result = crate::client::lookup_verify::<TC>(
        vrf_pk.as_bytes(),
        root_hash.hash(),
        root_hash.epoch() - 1, // To fake a lower epoch and trigger the error condition
        target_label.clone(),
        lookup_proof,
    );

    // Check that the result is a verification error
    match akd_result {
        Err(akd_core::verify::VerificationError::LookupProof(_)) => (),
        _ => panic!("Expected an invalid epoch error"),
    }

    Ok(())
}

/*
=========== Test Helpers ===========
*/

async fn async_poll_helper_proof<TC: Configuration, T: Database + 'static, V: VRFKeyStorage>(
    reader: &ReadOnlyDirectory<TC, T, V>,
    value: AkdValue,
) -> Result<(), AkdError> {
    // reader should read "hello" and this will populate the "cache" a log
    let (lookup_proof, root_hash) = reader.lookup(AkdLabel::from("hello")).await?;
    assert_eq!(value, lookup_proof.value);
    let pk = reader.get_public_key().await?;
    lookup_verify::<TC>(
        pk.as_bytes(),
        root_hash.hash(),
        root_hash.epoch(),
        AkdLabel::from("hello"),
        lookup_proof,
    )?;
    Ok(())
}
