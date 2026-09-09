// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use thiserror::Error;

/// Everything that can go wrong inside the Aegon engine.
///
/// Server, shard, and verifier paths all funnel into this one type;
/// the `Display` text on each variant is what surfaces over gRPC.
#[derive(Debug, Error)]
pub enum AegonError {
    /// Lookup or history was asked for a label that was never placed.
    #[error("label {0:?} is not present in the dictionary")]
    UnknownLabel(Vec<u8>),

    /// A publish tried to place a label that already owns a slot.
    #[error("label {0:?} is already assigned")]
    DuplicateLabel(Vec<u8>),

    /// The open-addressing probe sequence walked every slot without
    /// finding a free one. Raise the capacity or the over-provisioning
    /// factor.
    #[error("dictionary is full: open-addressing exhausted {capacity} slots without finding an empty index")]
    DictionaryFull {
        /// Number of slots probed before giving up.
        capacity: usize,
    },

    /// The requested epoch is outside the retained window -- either in
    /// the future, or old enough to have been pruned.
    #[error("epoch {0} is not retained in the server's history")]
    InvalidEpoch(u64),

    /// A configuration that cannot be satisfied, caught at setup time.
    #[error("configuration error: {0}")]
    Config(String),

    /// Propagated from the polynomial commitment scheme (commit, open,
    /// or trim).
    #[error("PCS error: {0:?}")]
    Pcs(akd_core::aegon_crypto::pcs::prelude::PCSError),

    /// Propagated from the Fiat-Shamir transcript.
    #[error("transcript error: {0:?}")]
    Transcript(akd_core::aegon_crypto::transcript::TranscriptError),

    /// A proof was well-formed but did not verify. The payload names
    /// the check that failed, so it is a `&'static str` rather than a
    /// formatted message.
    #[error("proof verification failed: {0}")]
    Verification(&'static str),

    /// Propagated from the key-value store (Redis or RocksDB).
    #[error("database error: {0}")]
    Database(String),

    /// A failure inside the Nova IVC audit path — parameter setup,
    /// folding a step, or verifying a recursive proof.
    #[cfg(feature = "ivc_audit")]
    #[error("IVC audit error: {0}")]
    Ivc(String),
}

impl From<akd_core::aegon_crypto::transcript::TranscriptError> for AegonError {
    fn from(e: akd_core::aegon_crypto::transcript::TranscriptError) -> Self {
        AegonError::Transcript(e)
    }
}

impl From<akd_core::aegon_crypto::pcs::prelude::PCSError> for AegonError {
    fn from(e: akd_core::aegon_crypto::pcs::prelude::PCSError) -> Self {
        AegonError::Pcs(e)
    }
}
