// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! Auditor entry points for the AKD public API.
//!
//! In the AKD-on-Aegon backend the legacy [`audit_verify`] /
//! [`verify_consecutive_append_only`] entry points are kept on the
//! public surface for source-level compatibility but cannot be
//! invoked: their inputs (a `Vec<Digest>` and an [`AppendOnlyProof`]
//! built from `AzksElement`s) do not carry the Fiat-Shamir-bound
//! invariance witnesses that the Aegon auditor checks. Auditors
//! should call [`crate::aegon_facade::verify_invariance`] over the
//! per-transition proofs returned by
//! [`crate::Directory::aegon_invariance_proofs`].

use akd_core::configuration::Configuration;

use crate::errors::AkdError;
use crate::{AppendOnlyProof, Digest, SingleAppendOnlyProof};

/// Verifies an audit proof, given start and end hashes.
///
/// **Not implemented in the AKD-on-Aegon backend.** See module-level
/// docs.
#[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
pub async fn audit_verify<TC: Configuration>(
    _hashes: Vec<Digest>,
    _proof: AppendOnlyProof,
) -> Result<(), AkdError> {
    unimplemented!(
        "AKD-on-Aegon: legacy audit_verify cannot transport the Aegon invariance proof \
         shape; use akd::aegon_facade::verify_invariance over Directory::aegon_invariance_proofs"
    )
}

/// Helper for audit, verifies an append-only proof over a single
/// transition.
///
/// **Not implemented in the AKD-on-Aegon backend.** See
/// [`audit_verify`].
#[cfg_attr(feature = "tracing_instrument", tracing::instrument(skip_all))]
pub async fn verify_consecutive_append_only<TC: Configuration>(
    _proof: &SingleAppendOnlyProof,
    _start_hash: Digest,
    _end_hash: Digest,
    _end_epoch: u64,
) -> Result<(), AkdError> {
    unimplemented!(
        "AKD-on-Aegon: legacy verify_consecutive_append_only cannot transport the Aegon \
         invariance proof shape; use akd::aegon_facade::verify_invariance"
    )
}
