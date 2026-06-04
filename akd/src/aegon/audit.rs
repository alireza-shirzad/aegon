//! Auditor's per-epoch invariance check (paper Fig. 4
//! `Auditor.VerifyInvariance`), commitment-homomorphism path.
//!
//! Stateless across calls except for an `AuditState` carrying the chain
//! Fiat-Shamir scalars `(r_index, r_value)` from one epoch transition to
//! the next. The check is constant-cost regardless of how many label
//! updates the epoch contained.
//!
//! For each chain (index, value):
//!   1. Recompute `r_n = O(prev_r, new_data_commitment)` to confirm the
//!      server used the canonical Fiat-Shamir scalar.
//!   2. Check the homomorphic relation **directly on commitments**:
//!         `C(rand_{n+1}) ?= C(rand_n) + r_n · ( C(poly_{n+1}) − C(poly_n) )`.
//!
//! Soundness: by binding of the PCS, if the commitments satisfy the
//! relation then the underlying polynomials do too — which is exactly
//! `rand_{n+1} = rand_n + r_n · (poly_{n+1} − poly_n)`. No PCS openings,
//! no Schwartz-Zippel evaluation point: just one group equation per
//! chain. Requires the PCS commitment to be linearly homomorphic
//! (`Add`, `Sub`, scalar `Mul`), which KZH-k satisfies — its commitment
//! is a Pedersen MSM on the evaluation vector.
//!
//! The auditor's old "openings at a random point" path is gone; in
//! exchange the auditor's per-epoch cost dropped from `O(N_shards × 8
//! PCS verifies)` to `O(N_shards × 2 group equations)`. No per-epoch
//! proof bytes flow alongside the commitments — every group element the
//! auditor needs is in [`EpochCommitment`] already.

use std::ops::{Add, Mul, Sub};

use ark_ec::pairing::Pairing;

use super::config::VerifierContext;
use super::error::AegonError;
use super::fs::derive_chain_scalar;
use super::sigma::{verify as verify_blinding_eq, BlindingEqProof};
use super::types::{AegonPcs, AuditState, EpochCommitment};
use akd_core::aegon_crypto::pcs::PCSGlobalParam;

/// Verify the invariance relation for a single epoch transition.
/// Updates `audit_state` with the new chain scalars on success.
pub fn verify_invariance<E, P>(
    ctx: &VerifierContext<E, P>,
    audit_state: &mut AuditState<E::ScalarField>,
    prev: &EpochCommitment<E, P>,
    next: &EpochCommitment<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    P::VerifierParam: PCSGlobalParam,
    P::Commitment: Clone
        + PartialEq
        + Add<Output = P::Commitment>
        + Sub<Output = P::Commitment>
        + Mul<E::ScalarField, Output = P::Commitment>,
{
    if next.epoch != prev.epoch + 1 {
        return Err(AegonError::Verification(
            "audit proof must cover a one-epoch transition",
        ));
    }

    let new_r_index = derive_chain_scalar::<E::ScalarField, P::Commitment>(
        b"aegon.fs.r_index",
        audit_state.r_index,
        &next.index_commitment,
    );
    // Index polynomials are always committed non-zk in Aegon (no
    // tau·h term, see `commit_with_aux_non_zk` in server.rs), so the
    // chain equation holds exactly in the group with no sigma proof.
    let index_ok = verify_chain::<E, P>(
        new_r_index,
        &prev.index_commitment,
        &next.index_commitment,
        &prev.rand_index_commitment,
        &next.rand_index_commitment,
        None,
        &ctx.verifier_param,
    );
    if !index_ok {
        return Ok(false);
    }

    let new_r_value = derive_chain_scalar::<E::ScalarField, P::Commitment>(
        b"aegon.fs.r_value",
        audit_state.r_value,
        &next.value_commitment,
    );
    // Value polynomials carry a `tau·h` hiding term under a zk SRS;
    // the published rand_value commitment was re-randomised with a
    // fresh tau, so the bare chain equation has a non-zero `c·h`
    // residue. `next.audit_value_blinding_proof` certifies that the
    // residue is in fact a known multiple of `h` (paper §7).
    //
    // Policy: under a zk SRS the proof MUST be present. A malicious
    // server skipping re-randomisation could otherwise ship a
    // deterministic chained tau, the bare equation would pass, and
    // observers of value-side openings across epochs could mount the
    // joint-leakage attack the privacy proof (App. D) rules out.
    // Enforce it here so the index chain (always non-hiding) stays
    // decoupled from this check.
    if PCSGlobalParam::is_zk(&ctx.verifier_param)
        && next.audit_value_blinding_proof.is_none()
    {
        return Ok(false);
    }
    let value_ok = verify_chain::<E, P>(
        new_r_value,
        &prev.value_commitment,
        &next.value_commitment,
        &prev.rand_value_commitment,
        &next.rand_value_commitment,
        next.audit_value_blinding_proof.as_ref(),
        &ctx.verifier_param,
    );
    if !value_ok {
        return Ok(false);
    }

    audit_state.r_index = new_r_index;
    audit_state.r_value = new_r_value;
    Ok(true)
}

/// One chain's commitment-homomorphism check:
///
/// ```text
///   C(rand_{n+1}) ?= C(rand_n) + r_n · ( C(poly_{n+1}) − C(poly_n) )
/// ```
///
/// In non-hiding mode (`blinding_proof = None`, non-zk SRS) this is a
/// bare group equality — three operations and one comparison.
///
/// In hiding mode the published rand commitment carries a fresh
/// blinding shift `c · h`, so the equation holds only modulo `h`.
/// The `blinding_proof` is a Schnorr proof that the residue
/// `next_rand − (prev_rand + r · (next_poly − prev_poly))` equals
/// `c · h` for some `c` the prover knows. The audit accepts iff that
/// Schnorr proof verifies. See [`super::sigma`] for the protocol.
///
/// Policy: under a zk SRS the proof MUST be `Some`. A `None` here in
/// zk mode would let a malicious server defeat the check by counting
/// on the bare group equation to fail (which it will) and the
/// caller to interpret that as a soft rejection — instead we reject
/// hard with `false`.
pub(super) fn verify_chain<E, P>(
    r_n: E::ScalarField,
    prev_poly_com: &P::Commitment,
    next_poly_com: &P::Commitment,
    prev_rand_com: &P::Commitment,
    next_rand_com: &P::Commitment,
    blinding_proof: Option<&BlindingEqProof<E, P>>,
    verifier_param: &P::VerifierParam,
) -> bool
where
    E: Pairing,
    P: AegonPcs<E>,
    P::VerifierParam: PCSGlobalParam,
    P::Commitment: Clone
        + PartialEq
        + Add<Output = P::Commitment>
        + Sub<Output = P::Commitment>
        + Mul<E::ScalarField, Output = P::Commitment>,
{
    match blinding_proof {
        Some(proof) => {
            // Verifying the Schnorr equation requires `h` from the
            // SRS — non-hiding SRSs don't expose one, so reject a
            // proof that arrived against the wrong SRS rather than
            // silently succeeding via `scaled_mask_generator_vk`
            // returning `None`.
            if !PCSGlobalParam::is_zk(verifier_param) {
                return false;
            }
            verify_blinding_eq::<E, P>(
                verifier_param,
                prev_poly_com,
                next_poly_com,
                prev_rand_com,
                next_rand_com,
                r_n,
                proof,
            )
        },
        None => {
            // Bare equality is correct whenever neither side carries
            // a `tau·h` term. Index chain hits this branch always;
            // value chain hits it under a non-hiding SRS. (Value
            // chain under hiding SRS is gated at the
            // `verify_invariance` policy check above.)
            let delta_poly: P::Commitment = next_poly_com.clone() - prev_poly_com.clone();
            let expected: P::Commitment = prev_rand_com.clone() + delta_poly * r_n;
            &expected == next_rand_com
        },
    }
}
