// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! Implementation of an auditable key directory, backed by the Aegon
//! polynomial-commitment engine. The public surface mirrors the original
//! AKD/SEEMless directory; internally everything routes through `aegon`.
//!
//! Methods that have a clean Aegon analogue (publish, lookup, batch
//! lookup, epoch-hash retrieval) are real implementations. Methods tied
//! to the Merkle / VRF version model (`key_history`, `audit`,
//! `poll_for_azks_changes`) are kept in the API as `unimplemented!()`
//! so source-level compatibility with downstream AKD consumers is
//! preserved.
//!
//! See the `aegon_facade` module at the bottom of this file for the
//! Aegon-native methods exposed alongside the legacy surface
//! (`consistency_proof`, `epoch_commitment`, etc.).

use crate::append_only_zks::{Azks, AzksParallelismConfig};
use crate::ecvrf::{VRFKeyStorage, VRFPublicKey};
use crate::errors::{AkdError, DirectoryError};
use crate::log::info;
use crate::storage::manager::StorageManager;
use crate::storage::Database;
use crate::{
    AkdLabel, AkdValue, AppendOnlyProof, Digest, EpochHash, HistoryProof, LookupProof, NodeLabel,
};

use crate::aegon::{
    optimal_kzh_k, Sha256Hash, ShardedAegon, ShardedAegonConfig, ShardedConsistencyProof,
    ShardedEpochCommitment, ShardedInvarianceProof, ShardedLookupProof, ShardedVerifierContext,
};
use akd_core::configuration::Configuration;
use akd_core::verify::history::HistoryParams;
use akd_core::types::{AzksValue, MembershipProof, NonMembershipProof};

use ark_bn254::Bn254;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use tokio::sync::{Mutex, RwLock};

// ---------- Aegon backend type aliases --------------------------------

/// Pairing curve used by the Aegon backend. BN254 lines up with what the
/// rest of the workspace tests/benches rely on.
pub type DirectoryE = Bn254;
/// PCS backend used by the Aegon engine. KZH-k with `k=2` (classical KZH).
pub type DirectoryPcs = KZHK<DirectoryE>;
/// Concrete sharded Aegon engine instantiated for this Directory.
pub type DirectoryAegon = ShardedAegon<DirectoryE, DirectoryPcs, Sha256Hash>;

// Default Aegon parameters. Hardcoded for v1; later we may thread them
// through `Directory::new` once we decide on the configuration story.
//
// Per-shard log_capacity = 8 (256 slots/shard) × 4 shards = 1024 total
// slots, plenty for tests without blowing setup time. `k` is chosen
// via `optimal_kzh_k` so the aux precomputation cost is minimized for
// whatever `shard_log_capacity` is in use; at the production setting
// of `shard_log_capacity = 29` this picks `k = 10`.
const DEFAULT_SHARD_LOG_CAPACITY: usize = 8;
const DEFAULT_LOG_N_SHARDS: usize = 2;
const DEFAULT_SETUP_SEED: u64 = 0xA56_0_AE60_0;

fn default_aegon_config() -> ShardedAegonConfig<DirectoryE, DirectoryPcs> {
    ShardedAegonConfig::<DirectoryE, DirectoryPcs>::builder()
        .shard_log_capacity(DEFAULT_SHARD_LOG_CAPACITY)
        .log_n_shards(DEFAULT_LOG_N_SHARDS)
        .private(false)
        .kzh_k(optimal_kzh_k(DEFAULT_SHARD_LOG_CAPACITY))
        .build()
        .expect("default ShardedAegonConfig must build")
}

// ---------- Directory struct ------------------------------------------

/// The representation of an auditable key directory.
pub struct Directory<TC, S: Database, V> {
    storage: StorageManager<S>,
    vrf: V,
    parallelism_config: AzksParallelismConfig,
    /// The cache lock guarantees that the cache is not flushed
    /// mid-proof generation. Retained for source compatibility with
    /// the original AKD; not load-bearing in the Aegon implementation.
    cache_lock: Arc<RwLock<()>>,
    /// Aegon engine instance. All directory state lives here.
    aegon: Arc<Mutex<DirectoryAegon>>,
    /// Plaintext-value cache. Aegon stores only field-element images of
    /// values, so to return `AkdValue` from a lookup we keep the raw
    /// bytes here keyed by label.
    label_values: Arc<RwLock<HashMap<AkdLabel, AkdValue>>>,
    /// Per-epoch sharded invariance proofs returned by
    /// `ShardedAegon::publish`. Indexed by `epoch - 1` (epoch 0 has no
    /// transition).
    invariance_proofs: Arc<RwLock<Vec<ShardedInvarianceProof<DirectoryE, DirectoryPcs>>>>,
    /// Snapshot of the sharded epoch commitments at every published
    /// epoch, including the empty epoch 0. Used to serve consistency
    /// proofs and epoch-by-epoch audits.
    epoch_commits: Arc<RwLock<Vec<ShardedEpochCommitment<DirectoryE, DirectoryPcs>>>>,
    tc: PhantomData<TC>,
}

// Manual Clone — see https://github.com/rust-lang/rust/issues/41481
impl<TC, S: Database, V: VRFKeyStorage> Clone for Directory<TC, S, V> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            vrf: self.vrf.clone(),
            parallelism_config: self.parallelism_config,
            cache_lock: self.cache_lock.clone(),
            aegon: self.aegon.clone(),
            label_values: self.label_values.clone(),
            invariance_proofs: self.invariance_proofs.clone(),
            epoch_commits: self.epoch_commits.clone(),
            tc: PhantomData,
        }
    }
}

impl<TC, S, V> Directory<TC, S, V>
where
    TC: Configuration,
    S: Database + 'static,
    V: VRFKeyStorage,
{
    /// Creates a new instance of an auditable key directory.
    ///
    /// `storage`, `vrf`, and `parallelism_config` are accepted for
    /// source-level compatibility with the original AKD API but are not
    /// load-bearing: state lives inside the Aegon engine, not in
    /// `storage`. The deterministic seed used for SRS generation is
    /// fixed; later versions may take it as a parameter.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn new(
        storage: StorageManager<S>,
        vrf: V,
        parallelism_config: AzksParallelismConfig,
    ) -> Result<Self, AkdError> {
        info!("Initialising AKD directory backed by ShardedAegon");
        let mut rng = ChaCha20Rng::seed_from_u64(DEFAULT_SETUP_SEED);
        let aegon = ShardedAegon::<DirectoryE, DirectoryPcs, Sha256Hash>::setup(
            &mut rng,
            &default_aegon_config(),
        )
        .map_err(|e| AkdError::Directory(DirectoryError::Publish(format!("aegon setup: {e}"))))?;

        let initial_commitment = aegon.current_commitment();

        Ok(Directory {
            storage,
            vrf,
            parallelism_config,
            cache_lock: Arc::new(RwLock::new(())),
            aegon: Arc::new(Mutex::new(aegon)),
            label_values: Arc::new(RwLock::new(HashMap::new())),
            invariance_proofs: Arc::new(RwLock::new(Vec::new())),
            epoch_commits: Arc::new(RwLock::new(vec![initial_commitment])),
            tc: PhantomData,
        })
    }

    /// Updates the directory to include the input label-value pairs.
    /// Returns the new epoch and a digest derived from the Aegon
    /// commitment. Errors if the batch contains duplicate labels.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all, fields(num_updates = updates.len())))]
    pub async fn publish(&self, updates: Vec<(AkdLabel, AkdValue)>) -> Result<EpochHash, AkdError> {
        let _guard = self.cache_lock.read().await;

        let distinct_set: HashSet<AkdLabel> =
            updates.iter().map(|(label, _)| label.clone()).collect();
        if distinct_set.len() != updates.len() {
            return Err(AkdError::Directory(DirectoryError::Publish(
                "Cannot publish with a set of entries that contain duplicate labels".to_string(),
            )));
        }

        // Convert AkdLabel/AkdValue -> aegon's Label/Value (both Vec<u8>).
        let aegon_updates: Vec<(crate::aegon::Label, crate::aegon::Value)> = updates
            .iter()
            .map(|(l, v)| (l.0.clone(), v.0.clone()))
            .collect();

        let (commitment, invariance) = {
            let mut aegon = self.aegon.lock().await;
            aegon
                .publish(&aegon_updates)
                .map_err(|e| AkdError::Directory(DirectoryError::Publish(format!("aegon publish: {e}"))))?
        };

        // Cache plaintext values so `lookup` can return them.
        {
            let mut cache = self.label_values.write().await;
            for (label, value) in updates {
                cache.insert(label, value);
            }
        }

        self.invariance_proofs.write().await.push(invariance);
        self.epoch_commits.write().await.push(commitment.clone());

        Ok(EpochHash(commitment.epoch, digest_of_commitment(&commitment)))
    }

    /// Provides proof of correctness for the latest version of the
    /// label's slot. Returns an [`Err`] if the label is not registered.
    ///
    /// The returned [`LookupProof`] preserves the AKD wire shape but
    /// carries the Aegon proof bytes inside `commitment_nonce` (see
    /// [`encode_lookup_payload`]). The Merkle-shaped fields are filled
    /// with zero values; legacy `lookup_verify` calls against this
    /// proof will fail. Use [`crate::aegon_facade::verify_lookup`] (or
    /// the re-export at the crate root) for verification.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn lookup(&self, akd_label: AkdLabel) -> Result<(LookupProof, EpochHash), AkdError> {
        let _guard = self.cache_lock.read().await;

        let aegon = self.aegon.lock().await;
        // The Sharded layer returns (value_from_db, proof). The
        // Directory keeps its own `label_values` map (below) for
        // single-process tests, so the DB-side value is discarded
        // here. With `DbSource::None`, the DB-side value is the
        // empty vector anyway.
        let (_db_value, proof) = aegon
            .lookup(&akd_label.0)
            .map_err(|e| AkdError::Directory(DirectoryError::Publish(format!("aegon lookup: {e}"))))?;
        let commitment = aegon.current_commitment();
        drop(aegon);

        let value = self
            .label_values
            .read()
            .await
            .get(&akd_label)
            .cloned()
            .unwrap_or_else(|| AkdValue(Vec::new()));

        let payload = encode_lookup_payload(&commitment, &proof);

        let lookup_proof = LookupProof {
            epoch: commitment.epoch,
            value,
            version: commitment.epoch, // Aegon has no per-label version chain
            existence_vrf_proof: Vec::new(),
            existence_proof: empty_membership_proof(),
            marker_vrf_proof: Vec::new(),
            marker_proof: empty_membership_proof(),
            freshness_vrf_proof: Vec::new(),
            freshness_proof: empty_nonmembership_proof(),
            commitment_nonce: payload,
        };

        let epoch_hash = EpochHash(commitment.epoch, digest_of_commitment(&commitment));
        Ok((lookup_proof, epoch_hash))
    }

    /// Allows efficient batch lookups. The current implementation just
    /// loops over [`Self::lookup`]; KZH-k does not yet expose a batched
    /// open primitive at this layer.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn batch_lookup(
        &self,
        akd_labels: &[AkdLabel],
    ) -> Result<(Vec<LookupProof>, EpochHash), AkdError> {
        let _guard = self.cache_lock.read().await;

        let mut proofs = Vec::with_capacity(akd_labels.len());
        let mut last_epoch_hash: Option<EpochHash> = None;
        for label in akd_labels {
            let (proof, eh) = self.lookup(label.clone()).await?;
            proofs.push(proof);
            last_epoch_hash = Some(eh);
        }

        let epoch_hash = match last_epoch_hash {
            Some(eh) => eh,
            None => self.get_epoch_hash().await?,
        };
        Ok((proofs, epoch_hash))
    }

    /// **Not implemented in the Aegon backend.**
    ///
    /// `key_history` returns the entire ordered version history of a
    /// label, which is fundamentally a SEEMless concept. Aegon stores
    /// only the latest value per label and proves consistency between
    /// two specific epochs (see
    /// [`Self::consistency_proof`]). Calling this method panics.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn key_history(
        &self,
        _akd_label: &AkdLabel,
        _params: HistoryParams,
    ) -> Result<(HistoryProof, EpochHash), AkdError> {
        unimplemented!(
            "key_history is a SEEMless-only concept; use Directory::consistency_proof on the Aegon backend"
        )
    }

    /// **Not implemented in the Aegon backend.** AKD's database polling
    /// loop is meaningless when state lives inside the in-memory Aegon
    /// engine.
    pub async fn poll_for_azks_changes(
        &self,
        _period: tokio::time::Duration,
        _change_detected: Option<tokio::sync::mpsc::Sender<()>>,
    ) -> Result<(), AkdError> {
        unimplemented!("poll_for_azks_changes is database-bound; the Aegon backend has no DB to poll")
    }

    /// **Not implemented via the legacy [`AppendOnlyProof`] shape.** The
    /// Aegon auditor proof carries different data (Fiat-Shamir-bound
    /// invariance witnesses) which does not fit inside
    /// `AppendOnlyProof`. Use
    /// [`Self::aegon_invariance_proofs`] to fetch the Aegon-shaped audit
    /// chain.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all, fields(start_epoch = audit_start_ep, end_epoch = audit_end_ep)))]
    pub async fn audit(
        &self,
        audit_start_ep: u64,
        audit_end_ep: u64,
    ) -> Result<AppendOnlyProof, AkdError> {
        let _ = (audit_start_ep, audit_end_ep);
        unimplemented!(
            "Directory::audit returns the legacy AppendOnlyProof shape; use Directory::aegon_invariance_proofs for the Aegon audit chain"
        )
    }

    /// **Internal helper retained for source compatibility.** Always
    /// errors — there is no AZKS in the Aegon backend.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub(crate) async fn retrieve_azks(&self) -> Result<Azks, AkdError> {
        Err(AkdError::Directory(DirectoryError::Publish(
            "AZKS does not exist in the Aegon backend".to_string(),
        )))
    }

    /// Returns a placeholder VRF public key. Aegon does not use the
    /// AKD ECVRF; the key is generated from `vrf` if available so
    /// downstream code that round-trips it does not break, otherwise
    /// a fixed dummy is returned.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn get_public_key(&self) -> Result<VRFPublicKey, AkdError> {
        Ok(self.vrf.get_vrf_public_key().await?)
    }

    /// Gets the root hash at the current epoch. Derived deterministically
    /// from the Aegon `EpochCommitment` (paper §6.2 commitment digest).
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn get_epoch_hash(&self) -> Result<EpochHash, AkdError> {
        let aegon = self.aegon.lock().await;
        let commitment = aegon.current_commitment();
        Ok(EpochHash(commitment.epoch, digest_of_commitment(&commitment)))
    }

    // ===================================================================
    // Aegon-native methods (Category 3 additions to the public API)
    // ===================================================================

    /// Build a sharded Aegon consistency proof showing the user's slot
    /// was unchanged between epoch `s0` and the current epoch.
    pub async fn consistency_proof(
        &self,
        akd_label: &AkdLabel,
        s0: u64,
    ) -> Result<ShardedConsistencyProof<DirectoryE, DirectoryPcs>, AkdError> {
        let aegon = self.aegon.lock().await;
        aegon
            .consistency_proof(&akd_label.0, s0)
            .map_err(|e| AkdError::Directory(DirectoryError::Publish(format!("aegon consistency: {e}"))))
    }

    /// Returns the sharded `EpochCommitment` for a past (or current) epoch.
    pub async fn epoch_commitment(
        &self,
        epoch: u64,
    ) -> Option<ShardedEpochCommitment<DirectoryE, DirectoryPcs>> {
        self.aegon.lock().await.epoch_commitment(epoch)
    }

    /// Returns the sharded `VerifierContext` used by the
    /// [`crate::aegon_facade`] verify functions.
    pub async fn verifier_context(&self) -> ShardedVerifierContext<DirectoryE, DirectoryPcs> {
        self.aegon.lock().await.sharded_verifier_context()
    }

    /// Returns the slice of per-transition sharded invariance proofs
    /// for the epoch range `[start_ep, end_ep)`. The auditor walks
    /// these via [`crate::aegon_facade::verify_invariance`].
    pub async fn aegon_invariance_proofs(
        &self,
        start_ep: u64,
        end_ep: u64,
    ) -> Result<Vec<ShardedInvarianceProof<DirectoryE, DirectoryPcs>>, AkdError> {
        let proofs = self.invariance_proofs.read().await;
        if end_ep as usize > proofs.len() + 1 || start_ep > end_ep {
            return Err(AkdError::Directory(DirectoryError::Publish(format!(
                "epoch range [{start_ep}, {end_ep}) outside available history of {} transitions",
                proofs.len()
            ))));
        }
        Ok(proofs[(start_ep as usize)..(end_ep as usize)].to_vec())
    }
}

/// A thin newtype which offers read-only interactivity with a [`Directory`].
#[derive(Clone)]
pub struct ReadOnlyDirectory<TC, S, V>(Directory<TC, S, V>)
where
    TC: Configuration,
    S: Database + Sync + Send,
    V: VRFKeyStorage;

impl<TC, S, V> ReadOnlyDirectory<TC, S, V>
where
    TC: Configuration,
    S: Database + 'static,
    V: VRFKeyStorage,
{
    /// Constructs a new instance of [`ReadOnlyDirectory`]. Wraps the
    /// inner [`Directory::new`] without imposing read-only semantics
    /// — the Aegon backend has no separate read-only mode.
    pub async fn new(
        storage: StorageManager<S>,
        vrf: V,
        parallelism_config: AzksParallelismConfig,
    ) -> Result<Self, AkdError> {
        Ok(Self(
            Directory::<TC, S, V>::new(storage, vrf, parallelism_config).await?,
        ))
    }

    /// Read-only access to [`Directory::lookup`].
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn lookup(&self, uname: AkdLabel) -> Result<(LookupProof, EpochHash), AkdError> {
        self.0.lookup(uname).await
    }

    /// Read-only access to [`Directory::batch_lookup`].
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn batch_lookup(
        &self,
        unames: &[AkdLabel],
    ) -> Result<(Vec<LookupProof>, EpochHash), AkdError> {
        self.0.batch_lookup(unames).await
    }

    /// Read-only access to [`Directory::key_history`]. Inherits the
    /// `unimplemented!()` body of the underlying call.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn key_history(
        &self,
        uname: &AkdLabel,
        params: HistoryParams,
    ) -> Result<(HistoryProof, EpochHash), AkdError> {
        self.0.key_history(uname, params).await
    }

    /// Read-only access to [`Directory::poll_for_azks_changes`].
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn poll_for_azks_changes(
        &self,
        period: tokio::time::Duration,
        change_detected: Option<tokio::sync::mpsc::Sender<()>>,
    ) -> Result<(), AkdError> {
        self.0.poll_for_azks_changes(period, change_detected).await
    }

    /// Read-only access to [`Directory::audit`].
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn audit(
        &self,
        audit_start_ep: u64,
        audit_end_ep: u64,
    ) -> Result<AppendOnlyProof, AkdError> {
        self.0.audit(audit_start_ep, audit_end_ep).await
    }

    /// Read-only access to [`Directory::get_epoch_hash`].
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn get_epoch_hash(&self) -> Result<EpochHash, AkdError> {
        self.0.get_epoch_hash().await
    }

    /// Read-only access to [`Directory::get_public_key`].
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn get_public_key(&self) -> Result<VRFPublicKey, AkdError> {
        self.0.get_public_key().await
    }
}

// Helper retained to avoid breaking pub(crate) references elsewhere.
pub(crate) fn get_marker_version(version: u64) -> u64 {
    (64 - version.leading_zeros() - 1).into()
}

// ---------- payload encoding ------------------------------------------

/// Wire-format envelope embedded in `LookupProof.commitment_nonce`.
/// Carries the sharded epoch commitment and the sharded Aegon lookup
/// proof so an Aegon-aware verifier has everything it needs without
/// leaning on AKD's legacy Merkle fields.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
struct LookupPayload {
    commitment: ShardedEpochCommitment<DirectoryE, DirectoryPcs>,
    proof: ShardedLookupProof<DirectoryE, DirectoryPcs>,
}

fn encode_lookup_payload(
    commitment: &ShardedEpochCommitment<DirectoryE, DirectoryPcs>,
    proof: &ShardedLookupProof<DirectoryE, DirectoryPcs>,
) -> Vec<u8> {
    let payload = LookupPayload {
        commitment: commitment.clone(),
        proof: proof.clone(),
    };
    let mut bytes = Vec::new();
    payload
        .serialize_compressed(&mut bytes)
        .expect("LookupPayload serialization is infallible for valid inputs");
    bytes
}

pub(crate) fn decode_lookup_payload(
    bytes: &[u8],
) -> Result<
    (
        ShardedEpochCommitment<DirectoryE, DirectoryPcs>,
        ShardedLookupProof<DirectoryE, DirectoryPcs>,
    ),
    AkdError,
> {
    let payload = LookupPayload::deserialize_compressed(bytes).map_err(|e| {
        AkdError::Directory(DirectoryError::Publish(format!(
            "decode lookup payload: {e}"
        )))
    })?;
    Ok((payload.commitment, payload.proof))
}

/// Build a `Digest` (32-byte hash) from a sharded `EpochCommitment`.
/// The Merkle root over the per-shard commitments already commits to
/// every shard's full state, so it doubles as the epoch digest.
fn digest_of_commitment(
    commit: &ShardedEpochCommitment<DirectoryE, DirectoryPcs>,
) -> Digest {
    commit.merkle_root
}

fn empty_node_label() -> NodeLabel {
    NodeLabel {
        label_val: [0u8; 32],
        label_len: 0,
    }
}

fn empty_membership_proof() -> MembershipProof {
    MembershipProof {
        label: empty_node_label(),
        hash_val: AzksValue([0u8; 32]),
        sibling_proofs: Vec::new(),
    }
}

fn empty_nonmembership_proof() -> NonMembershipProof {
    use akd_core::AzksElement;
    let placeholder = AzksElement {
        label: empty_node_label(),
        value: AzksValue([0u8; 32]),
    };
    NonMembershipProof {
        label: empty_node_label(),
        longest_prefix: empty_node_label(),
        longest_prefix_children: [placeholder; akd_core::ARITY],
        longest_prefix_membership_proof: empty_membership_proof(),
    }
}
