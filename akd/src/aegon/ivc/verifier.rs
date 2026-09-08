// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Auditor side: verify one recursive proof instead of one proof per
//! epoch.
//!
//! ## What a successful verification means
//!
//! That *every* epoch transition from genesis to the epoch described
//! by `current` satisfied Aegon's invariance relation — the index
//! chain exactly, and the value chain up to a blinding shift
//! certified by each shard's Schnorr proof — with Fiat–Shamir
//! challenges the server could not have chosen after the fact.
//!
//! The cost is independent of how many epochs elapsed. That is the
//! point: an auditor can be offline for a week, fetch one proof, and
//! be current. It is the "fast-forwarding via recursive proofs"
//! scenario of the paper's §3, and it removes the standing
//! requirement to be online every epoch that
//! [`verify_sharded_invariance`](crate::aegon::verify_sharded_invariance)
//! imposes.
//!
//! ## Why the SHA256 Merkle root never enters the circuit
//!
//! The recursive proof on its own is about a *digest*, not about any
//! particular epoch. Step 3 below is what anchors it: the auditor
//! already downloads the epoch's `per_shard` tuple (the ~30 KB object
//! it downloads today), recomputes the Poseidon state digest over it,
//! and checks that against the proof's output. Because it holds the
//! same tuple, it can equally recompute the published SHA256 Merkle
//! root natively — exactly as it does today, with the bulletin-board
//! format unchanged. So SHA256, which would have dominated the
//! circuit, stays entirely outside it.

use nova_snark::traits::Engine;

use super::bridge::CircuitField;
use super::circuit::ARITY;
use super::fs_poseidon::{poseidon_state_digest, ShardCommitments};
use super::prover::{AuditProof, CompressedAuditProof, CompressedVerifierKey, IvcAuditParams, E1};
use crate::aegon::error::AegonError;

/// The initial folded state for a chain anchored at `genesis`:
/// `[epoch = 0, r_index = 0, r_value = 0, Poseidon(genesis)]`.
///
/// Both prover and verifier derive this from the same public genesis
/// commitments, so neither has to be trusted about it.
pub fn initial_state(
    params: &IvcAuditParams,
    genesis: &[ShardCommitments],
) -> Result<Vec<CircuitField>, AegonError> {
    let n = params.fs_params().n_shards;
    if genesis.len() != n {
        return Err(AegonError::Ivc(format!(
            "genesis commitments cover {} shards, parameters expect {n}",
            genesis.len()
        )));
    }
    Ok(vec![
        CircuitField::from(0u64),
        CircuitField::from(0u64),
        CircuitField::from(0u64),
        poseidon_state_digest(params.ro_consts(), params.fs_params(), genesis),
    ])
}

/// Outcome of a successful audit verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedAudit {
    /// Number of epoch transitions covered, i.e. the epoch number the
    /// chain has reached.
    pub epochs: u64,
}

/// Verify a recursive audit proof and bind it to a concrete epoch.
///
/// Steps:
///
/// 1. Verify the Nova proof for `num_steps` steps from `z0`.
/// 2. Check the folded epoch counter equals `num_steps` — so the
///    proof cannot claim more history than it folded.
/// 3. Check the folded state digest equals the Poseidon digest of the
///    `current` per-shard commitments the auditor holds. Without this
///    the proof would be about an unspecified state.
///
/// The caller is responsible for step 4: confirming `current` is
/// genuinely what the bulletin board published for that epoch, by
/// recomputing the SHA256 Merkle root over the same tuple. See
/// [`verify_against_merkle_root`] for the combined check.
pub fn verify_ivc_audit(
    params: &IvcAuditParams,
    proof: &AuditProof,
    num_steps: usize,
    z0: &[CircuitField],
    current: &[ShardCommitments],
) -> Result<VerifiedAudit, AegonError> {
    if z0.len() != ARITY {
        return Err(AegonError::Ivc(format!(
            "initial state must have arity {ARITY}, got {}",
            z0.len()
        )));
    }
    if num_steps == 0 {
        return Err(AegonError::Ivc(
            "a recursive audit proof must cover at least one epoch transition".into(),
        ));
    }
    let n = params.fs_params().n_shards;
    if current.len() != n {
        return Err(AegonError::Ivc(format!(
            "current commitments cover {} shards, parameters expect {n}",
            current.len()
        )));
    }

    let z_final = proof
        .verify(params.public_params(), num_steps, z0)
        .map_err(|e| AegonError::Ivc(format!("recursive proof rejected: {e}")))?;

    let expected_epoch = <E1 as Engine>::Scalar::from(num_steps as u64);
    if z_final[0] != expected_epoch {
        return Err(AegonError::Verification(
            "IVC audit: folded epoch counter does not match the claimed number of steps",
        ));
    }

    let expected_digest = poseidon_state_digest(params.ro_consts(), params.fs_params(), current);
    if z_final[3] != expected_digest {
        return Err(AegonError::Verification(
            "IVC audit: proof does not describe the supplied epoch commitments",
        ));
    }

    Ok(VerifiedAudit {
        epochs: num_steps as u64,
    })
}

/// [`verify_ivc_audit`] for a **compressed** proof.
///
/// Same statement, same three checks; only the proof encoding
/// differs. This is the form a deployment publishes, since the
/// uncompressed `RecursiveSNARK` carries the prover's folding state.
pub fn verify_compressed_ivc_audit(
    params: &IvcAuditParams,
    vk: &CompressedVerifierKey,
    proof: &CompressedAuditProof,
    num_steps: usize,
    z0: &[CircuitField],
    current: &[ShardCommitments],
) -> Result<VerifiedAudit, AegonError> {
    if num_steps == 0 {
        return Err(AegonError::Ivc(
            "a recursive audit proof must cover at least one epoch transition".into(),
        ));
    }
    let z_final = proof
        .verify(vk, num_steps, z0)
        .map_err(|e| AegonError::Ivc(format!("compressed proof rejected: {e}")))?;

    if z_final[0] != <E1 as Engine>::Scalar::from(num_steps as u64) {
        return Err(AegonError::Verification(
            "IVC audit: folded epoch counter does not match the claimed number of steps",
        ));
    }
    if z_final[3] != poseidon_state_digest(params.ro_consts(), params.fs_params(), current) {
        return Err(AegonError::Verification(
            "IVC audit: proof does not describe the supplied epoch commitments",
        ));
    }
    Ok(VerifiedAudit {
        epochs: num_steps as u64,
    })
}

/// [`verify_ivc_audit`] plus the bulletin-board binding.
///
/// `merkle_root_of` recomputes the SHA256 root over the same
/// `current` tuple — in practice
/// [`merkle_root`](crate::aegon::merkle_root) — and `published` is
/// the root the bulletin board carries for that epoch. Passing the
/// recomputation as a closure keeps this module free of the PCS
/// generics that `merkle_root` carries.
pub fn verify_against_merkle_root<F>(
    params: &IvcAuditParams,
    proof: &AuditProof,
    num_steps: usize,
    z0: &[CircuitField],
    current: &[ShardCommitments],
    published: [u8; 32],
    merkle_root_of: F,
) -> Result<VerifiedAudit, AegonError>
where
    F: FnOnce() -> [u8; 32],
{
    let verified = verify_ivc_audit(params, proof, num_steps, z0, current)?;
    if merkle_root_of() != published {
        return Err(AegonError::Verification(
            "IVC audit: supplied epoch commitments do not reconstruct the published Merkle root",
        ));
    }
    Ok(verified)
}
