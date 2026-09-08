// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Aegon-native verifier surface for AKD.
//!
//! These are the Category-3 additions: functionality that exists in
//! the Aegon engine but has no analogue in the original SEEMless AKD
//! API. They live alongside the legacy `client::lookup_verify`,
//! `auditor::audit_verify`, etc. surface — those legacy functions
//! `unimplemented!()` in the AKD-on-Aegon backend, and downstream
//! callers should migrate to the entry points in this module.
//!
//! Concrete typing: everything is fixed to BN254 + KZH-k via the
//! [`crate::directory::DirectoryE`] / [`crate::directory::DirectoryPcs`]
//! aliases. We re-export type aliases so downstream callers do not
//! need to spell out the generic parameters.

use crate::directory::{decode_lookup_payload, DirectoryE, DirectoryPcs};
use crate::errors::AkdError;
use crate::{AkdLabel, LookupProof};

use crate::aegon::verify_sharded_consistency_two_layer as aegon_verify_consistency;
use crate::aegon::verify_sharded_invariance as aegon_verify_invariance;
use crate::aegon::verify_sharded_lookup_two_layer as aegon_verify_lookup;
use crate::aegon::AegonError;
use crate::aegon::EcVrfHash;

/// Sharded epoch commitment specialised to AKD's BN254+KZHK backend.
pub type EpochCommitment = crate::aegon::ShardedEpochCommitment<DirectoryE, DirectoryPcs>;
/// Sharded consistency proof specialised to AKD's BN254+KZHK backend.
pub type ConsistencyProof = crate::aegon::ShardedConsistencyProofTwoLayer<DirectoryE, DirectoryPcs>;
/// Sharded verifier context specialised to AKD's BN254+KZHK backend.
pub type VerifierContext = crate::aegon::ShardedVerifierContext<DirectoryE, DirectoryPcs>;
/// Aegon `AuditState` over BN254's scalar field.
pub type AuditState =
    crate::aegon::AuditState<<DirectoryE as ark_ec::pairing::Pairing>::ScalarField>;
/// Sharded audit state over BN254's scalar field. Carries one
/// Fiat-Shamir accumulator per chain group; `Default` is the
/// single-group deployment.
pub type ShardedAuditState =
    crate::aegon::ShardedAuditState<<DirectoryE as ark_ec::pairing::Pairing>::ScalarField>;

/// Verify a lookup proof produced by [`crate::Directory::lookup`]
/// against an explicit Aegon `VerifierContext`.
///
/// The Merkle-shaped fields of the legacy [`LookupProof`] wire format
/// are ignored; the real proof bytes ride inside `proof.commitment_nonce`
/// (see [`crate::directory`] for the wire contract). The `epoch` and
/// `value` fields are checked for consistency with the embedded
/// payload.
pub fn verify_lookup(
    ctx: &VerifierContext,
    label: &AkdLabel,
    proof: &LookupProof,
) -> Result<bool, AkdError> {
    let (commitment, aegon_proof) = decode_lookup_payload(&proof.commitment_nonce)?;
    if commitment.epoch != proof.epoch {
        return Err(AkdError::Directory(crate::errors::DirectoryError::Publish(
            format!(
                "lookup proof epoch mismatch: outer {} vs payload {}",
                proof.epoch, commitment.epoch
            ),
        )));
    }
    let value: crate::aegon::Value = proof.value.0.clone();
    let label_bytes: crate::aegon::Label = label.0.clone();
    verify_lookup_aegon(ctx, &commitment, &label_bytes, &value, &aegon_proof)
}

/// Direct passthrough for callers that already hold the typed Aegon
/// commitment + proof (e.g. inside test code that does not bother
/// going through the AKD wire format).
pub fn verify_lookup_aegon(
    ctx: &VerifierContext,
    commitment: &EpochCommitment,
    label: &crate::aegon::Label,
    value: &crate::aegon::Value,
    proof: &crate::aegon::ShardedLookupProofTwoLayer<DirectoryE, DirectoryPcs>,
) -> Result<bool, AkdError> {
    aegon_verify_lookup::<DirectoryE, DirectoryPcs, EcVrfHash>(ctx, commitment, label, value, proof)
        .map_err(map_aegon_err)
}

/// Verify a single-transition sharded invariance relation. Callers walk
/// consecutive pairs of the commitment chain returned by
/// [`crate::Directory::aegon_epoch_commits`] and call this for each
/// step, threading a single `ShardedAuditState`. No per-epoch proof bytes
/// flow — verification reads only from the published
/// `EpochCommitment`s (commitment-homomorphism path).
pub fn verify_invariance(
    ctx: &VerifierContext,
    audit_state: &mut ShardedAuditState,
    prev: &EpochCommitment,
    next: &EpochCommitment,
) -> Result<bool, AkdError> {
    aegon_verify_invariance::<DirectoryE, DirectoryPcs>(ctx, audit_state, prev, next)
        .map_err(map_aegon_err)
}

/// Verify a per-user consistency proof showing the user's slot did
/// not change between two epochs `s0 < s1`. In the two-layer
/// routing model the routing trail length is bundled in the proof
/// itself, so no client-side pinning is required.
pub fn verify_consistency(
    ctx: &VerifierContext,
    s0: &EpochCommitment,
    s1: &EpochCommitment,
    label: &AkdLabel,
    proof: &ConsistencyProof,
) -> Result<bool, AkdError> {
    let label_bytes: crate::aegon::Label = label.0.clone();
    aegon_verify_consistency::<DirectoryE, DirectoryPcs, EcVrfHash>(
        ctx,
        s0,
        s1,
        &label_bytes,
        proof,
    )
    .map_err(map_aegon_err)
}

fn map_aegon_err(e: AegonError) -> AkdError {
    AkdError::Directory(crate::errors::DirectoryError::Publish(format!(
        "aegon verify: {e}"
    )))
}
