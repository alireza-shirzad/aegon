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
//! PCS verifies)` to `O(N_shards × 2 group equations)` and the wire
//! format for `InvarianceProof` is now empty.

use std::ops::{Add, Mul, Sub};

use ark_ec::pairing::Pairing;

use super::config::VerifierContext;
use super::error::AegonError;
use super::fs::derive_chain_scalar;
use super::types::{AegonPcs, AuditState, EpochCommitment, InvarianceProof};

/// Verify the invariance proof for a single epoch transition. Updates
/// `audit_state` with the new chain scalars on success.
pub fn verify_invariance<E, P>(
    _ctx: &VerifierContext<E, P>,
    audit_state: &mut AuditState<E::ScalarField>,
    prev: &EpochCommitment<E, P>,
    next: &EpochCommitment<E, P>,
    _proof: &InvarianceProof<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
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
    let index_ok = verify_chain::<E, P>(
        new_r_index,
        &prev.index_commitment,
        &next.index_commitment,
        &prev.rand_index_commitment,
        &next.rand_index_commitment,
    );
    if !index_ok {
        return Ok(false);
    }

    let new_r_value = derive_chain_scalar::<E::ScalarField, P::Commitment>(
        b"aegon.fs.r_value",
        audit_state.r_value,
        &next.value_commitment,
    );
    let value_ok = verify_chain::<E, P>(
        new_r_value,
        &prev.value_commitment,
        &next.value_commitment,
        &prev.rand_value_commitment,
        &next.rand_value_commitment,
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
/// Three group operations and one equality, no openings.
pub(super) fn verify_chain<E, P>(
    r_n: E::ScalarField,
    prev_poly_com: &P::Commitment,
    next_poly_com: &P::Commitment,
    prev_rand_com: &P::Commitment,
    next_rand_com: &P::Commitment,
) -> bool
where
    E: Pairing,
    P: AegonPcs<E>,
    P::Commitment: Clone
        + PartialEq
        + Add<Output = P::Commitment>
        + Sub<Output = P::Commitment>
        + Mul<E::ScalarField, Output = P::Commitment>,
{
    let delta_poly: P::Commitment = next_poly_com.clone() - prev_poly_com.clone();
    let expected: P::Commitment = prev_rand_com.clone() + delta_poly * r_n;
    &expected == next_rand_com
}
