// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! Implementation of an auditable key directory

use crate::append_only_zks::{Azks, AzksParallelismConfig, InsertMode};
use crate::ecvrf::{VRFKeyStorage, VRFPublicKey};
use crate::errors::{AkdError, DirectoryError, StorageError};
use crate::helper_structs::LookupInfo;
use crate::log::{error, info};
use crate::storage::manager::StorageManager;
use crate::storage::types::{DbRecord, ValueState, ValueStateRetrievalFlag};
use crate::storage::Database;
use crate::{
    AkdLabel, AkdValue, AppendOnlyProof, AzksElement, Digest, EpochHash, HistoryProof, LookupProof,
    UpdateProof,
};

use crate::VersionFreshness;
use akd_core::configuration::Configuration;
use akd_core::utils::get_marker_versions;
use akd_core::verify::history::HistoryParams;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::RwLock;
#[cfg(feature = "tracing_instrument")]
use tracing::Instrument;

/// The representation of a auditable key directory
pub struct Directory<TC, S: Database, V> {
    storage: StorageManager<S>,
    vrf: V,
    parallelism_config: AzksParallelismConfig,
    /// The cache lock guarantees that the cache is not
    /// flushed mid-proof generation. We allow multiple proof generations
    /// to occur (RwLock.read() operations can have multiple) but we want
    /// to make sure no generations are underway when a cache flush occurs
    /// (in this case we do utilize the write() lock which can only occur 1
    /// at a time and gates further read() locks being acquired during write()).
    cache_lock: Arc<RwLock<()>>,
    tc: PhantomData<TC>,
}

// Manual implementation of Clone, see: https://github.com/rust-lang/rust/issues/41481
impl<TC, S: Database, V: VRFKeyStorage> Clone for Directory<TC, S, V> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            vrf: self.vrf.clone(),
            parallelism_config: self.parallelism_config,
            cache_lock: self.cache_lock.clone(),
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
    /// Creates a new (stateless) instance of a auditable key directory.
    /// Takes as input a pointer to the storage being used for this instance.
    /// The state is stored in the storage.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn new(
        storage: StorageManager<S>,
        vrf: V,
        parallelism_config: AzksParallelismConfig,
    ) -> Result<Self, AkdError> {
        let azks = Directory::<TC, S, V>::get_azks_from_storage(&storage, false).await;

        if let Err(AkdError::Storage(StorageError::NotFound(e))) = azks {
            info!("No aZKS was found in storage: {e}. Creating a new aZKS!");
            // generate + store a new azks only if one is not found
            let new_azks = Azks::new::<TC, _>(&storage).await?;
            storage.set(DbRecord::Azks(new_azks)).await?;
        } else {
            // If the value is `Ok`, we drop it since we're not using it below
            // In all other `Err` cases, we propagate the error to the caller
            let _res = azks?;
        }

        Ok(Directory {
            storage,
            vrf,
            parallelism_config,
            cache_lock: Arc::new(RwLock::new(())),
            tc: PhantomData,
        })
    }

    /// Updates the directory to include the input label-value pairs.
    ///
    /// Note that the vector of label-value pairs should not contain any entries with duplicate labels. This
    /// condition is explicitly checked, and an error will be returned if this is the case.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all, fields(num_updates = updates.len())))]
    pub async fn publish(&self, updates: Vec<(AkdLabel, AkdValue)>) -> Result<EpochHash, AkdError> {
        // The guard will be dropped at the end of the publish operation
        let _guard = self.cache_lock.read().await;

        // Check for duplicate labels and return an error if any are encountered
        let distinct_set: HashSet<AkdLabel> =
            updates.iter().map(|(label, _)| label.clone()).collect();
        if distinct_set.len() != updates.len() {
            return Err(AkdError::Directory(DirectoryError::Publish(
                "Cannot publish with a set of entries that contain duplicate labels".to_string(),
            )));
        }
        //TODO: Implement Aegon logic here
        todo!("Implement Aegon logic here");
    }

    /// Provides proof for correctness of latest version
    ///
    /// * `akd_label`: The target label to generate a lookup proof for
    ///
    /// Returns [Ok((LookupProof, EpochHash))] upon successful generation for the latest version
    /// of the target label's state. [Err(_)] otherwise
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn lookup(&self, akd_label: AkdLabel) -> Result<(LookupProof, EpochHash), AkdError> {
        // The guard will be dropped at the end of the proof generation
        let _guard = self.cache_lock.read().await;

        //TODO: Implement Aegon logic here
        todo!("Implement Aegon logic here");
    }

    // TODO(eoz): Call proof generations async
    /// Allows efficient batch lookups by preloading necessary nodes for the lookups.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn batch_lookup(
        &self,
        akd_labels: &[AkdLabel],
    ) -> Result<(Vec<LookupProof>, EpochHash), AkdError> {
        // The guard will be dropped at the end of the proof generation
        let _guard = self.cache_lock.read().await;
        //QUESTION: Do we need batch lookup?
        //TODO: Implement Aegon logic here
        todo!("Implement Aegon logic here");
    }

    /// Takes in the current state of the server and a label.
    /// If the label is present in the current state,
    /// this function returns all the values ever associated with it,
    /// and the epoch at which each value was first committed to the server state.
    /// It also returns the proof of the latest version being served at all times.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn key_history(
        &self,
        akd_label: &AkdLabel,
        params: HistoryParams,
    ) -> Result<(HistoryProof, EpochHash), AkdError> {
        // The guard will be dropped at the end of the proof generation
        #[cfg(not(feature = "tracing_instrument"))]
        let _guard = self.cache_lock.read().await;
        #[cfg(feature = "tracing_instrument")]
        let _guard = self
            .cache_lock
            .read()
            .instrument(tracing::info_span!("cache_lock.read"))
            .await;

        //TODO: Implement Aegon logic here
        todo!("Implement Aegon logic here");
    }

    /// Poll for changes in the epoch number of the AZKS struct
    /// stored in the storage layer. If an epoch change is detected,
    /// the object cache (if present) is flushed immediately so
    /// that new objects are retrieved from the storage layer against
    /// the "latest" epoch. There is a "special" flow in the storage layer
    /// to do a storage-layer retrieval which ignores the cache
    pub async fn poll_for_azks_changes(
        &self,
        period: tokio::time::Duration,
        change_detected: Option<tokio::sync::mpsc::Sender<()>>,
    ) -> Result<(), AkdError> {

        //QUESTION: Do we need this?


        //TODO: Implement Aegon logic here
        todo!("Implement Aegon logic here");
    }

    /// Returns an [AppendOnlyProof] for the leaves inserted into the underlying tree between
    /// the epochs `audit_start_ep` and `audit_end_ep`.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all, fields(start_epoch = audit_start_ep, end_epoch = audit_end_ep)))]
    pub async fn audit(
        &self,
        audit_start_ep: u64,
        audit_end_ep: u64,
    ) -> Result<AppendOnlyProof, AkdError> {
        // The guard will be dropped at the end of the proof generation
        #[cfg(not(feature = "tracing_instrument"))]
        let _guard = self.cache_lock.read().await;
        #[cfg(feature = "tracing_instrument")]
        let _guard = self
            .cache_lock
            .read()
            .instrument(tracing::info_span!("cache_lock.read"))
            .await;

        //TODO: Implement Aegon logic here
        todo!("Implement Aegon logic here");
    }

    /// Retrieves the [Azks]
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub(crate) async fn retrieve_azks(&self) -> Result<Azks, crate::errors::AkdError> {
        Directory::<TC, S, V>::get_azks_from_storage(&self.storage, false).await
    }

    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all, fields(ignore_cache = ignore_cache)))]
    async fn get_azks_from_storage(
        storage: &StorageManager<S>,
        ignore_cache: bool,
    ) -> Result<Azks, crate::errors::AkdError> {
        let got = if ignore_cache {
            storage
                .get_direct::<Azks>(&crate::append_only_zks::DEFAULT_AZKS_KEY)
                .await?
        } else {
            storage
                .get::<Azks>(&crate::append_only_zks::DEFAULT_AZKS_KEY)
                .await?
        };
        match got {
            DbRecord::Azks(azks) => Ok(azks),
            _ => {
                error!("No AZKS can be found. You should re-initialize the directory to create a new one");
                Err(AkdError::Storage(StorageError::NotFound(
                    "AZKS not found".to_string(),
                )))
            }
        }
    }

    // HELPERS //

    /// Use this function to retrieve the [VRFPublicKey] for this AKD.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn get_public_key(&self) -> Result<VRFPublicKey, AkdError> {
        Ok(self.vrf.get_vrf_public_key().await?)
    }


    /// Gets the root hash at the current epoch.
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn get_epoch_hash(&self) -> Result<EpochHash, AkdError> {
        let current_azks = self.retrieve_azks().await?;
        let latest_epoch = current_azks.get_latest_epoch();
        let root_hash = current_azks.get_root_hash::<TC, _>(&self.storage).await?;
        Ok(EpochHash(latest_epoch, root_hash))
    }

    // We simply hash the VRF private key to derive the commitment key
    async fn derive_commitment_key(&self) -> Result<Digest, AkdError> {
        let raw_key = self.vrf.retrieve().await?;
        let commitment_key = TC::hash(&raw_key);
        Ok(commitment_key)
    }
}

/// A thin newtype which offers read-only interactivity with a [Directory].
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
    /// Constructs a new instance of [ReadOnlyDirectory]. In the event that an [Azks]
    /// does not exist in the storage, or we're unable to retrieve it from storage, then
    /// a [DirectoryError] will be returned.
    pub async fn new(
        storage: StorageManager<S>,
        vrf: V,
        parallelism_config: AzksParallelismConfig,
    ) -> Result<Self, AkdError> {
        let azks = Directory::<TC, S, V>::get_azks_from_storage(&storage, false).await;

        if azks.is_err() {
            return Err(AkdError::Directory(DirectoryError::ReadOnlyDirectory(
                format!(
                    "Cannot start directory in read-only mode when AZKS is missing, error: {:?}",
                    azks.err()
                ),
            )));
        }

        Ok(Self(Directory {
            storage,
            vrf,
            parallelism_config,
            cache_lock: Arc::new(RwLock::new(())),
            tc: PhantomData,
        }))
    }

    /// Read-only access to [Directory::lookup](Directory::lookup).
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn lookup(&self, uname: AkdLabel) -> Result<(LookupProof, EpochHash), AkdError> {
        self.0.lookup(uname).await
    }

    /// Read-only access to [Directory::batch_lookup](Directory::batch_lookup).
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn batch_lookup(
        &self,
        unames: &[AkdLabel],
    ) -> Result<(Vec<LookupProof>, EpochHash), AkdError> {
        self.0.batch_lookup(unames).await
    }

    /// Read-only access to [Directory::key_history](Directory::key_history).
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn key_history(
        &self,
        uname: &AkdLabel,
        params: HistoryParams,
    ) -> Result<(HistoryProof, EpochHash), AkdError> {
        self.0.key_history(uname, params).await
    }

    /// Read-only access to [Directory::poll_for_azks_changes](Directory::poll_for_azks_changes).
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn poll_for_azks_changes(
        &self,
        period: tokio::time::Duration,
        change_detected: Option<tokio::sync::mpsc::Sender<()>>,
    ) -> Result<(), AkdError> {
        self.0.poll_for_azks_changes(period, change_detected).await
    }

    /// Read-only access to [Directory::audit](Directory::audit).
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn audit(
        &self,
        audit_start_ep: u64,
        audit_end_ep: u64,
    ) -> Result<AppendOnlyProof, AkdError> {
        self.0.audit(audit_start_ep, audit_end_ep).await
    }

    /// Read-only access to [Directory::get_epoch_hash].
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn get_epoch_hash(&self) -> Result<EpochHash, AkdError> {
        self.0.get_epoch_hash().await
    }

    /// Read-only access to [Directory::get_public_key](Directory::get_public_key).
    #[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
    pub async fn get_public_key(&self) -> Result<VRFPublicKey, AkdError> {
        self.0.get_public_key().await
    }
}

// Helpers
pub(crate) fn get_marker_version(version: u64) -> u64 {
    (64 - version.leading_zeros() - 1).into()
}
