// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Verification of lookup proofs.
//!
//! In the AKD-on-Aegon backend this function is retained as part of
//! the public surface for source-level compatibility but cannot be
//! invoked: the AKD `LookupProof` does not carry the Aegon
//! `VerifierContext` needed to actually check a polynomial-commitment
//! proof. Use `akd::aegon_facade::verify_lookup` instead, which
//! takes an explicit `VerifierContext` plus a `Directory` /
//! `EpochCommitment` and runs the real check.

use super::VerificationError;

use crate::configuration::Configuration;
use crate::hash::Digest;
use crate::{AkdLabel, LookupProof, VerifyResult};

/// Verifies a lookup proof.
///
/// **Not implemented in the Aegon backend.** This entry point is kept
/// to preserve the legacy AKD verifier API surface; it always panics.
/// Use `akd::aegon_facade::verify_lookup` for the real verification
/// path.
pub fn lookup_verify<TC: Configuration>(
    _vrf_public_key: &[u8],
    _root_hash: Digest,
    _current_epoch: u64,
    _akd_label: AkdLabel,
    _proof: LookupProof,
) -> Result<VerifyResult, VerificationError> {
    unimplemented!(
        "AKD-on-Aegon: use akd::aegon_facade::verify_lookup; the legacy lookup_verify cannot \
         convey the Aegon VerifierContext through its signature"
    )
}
