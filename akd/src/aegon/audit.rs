//! Auditor's per-epoch invariance check (paper Fig. 4
//! `Auditor.VerifyInvariance`).
//!
//! Stateless across calls except for an `AuditState` carrying the chain
//! Fiat-Shamir scalars `(r_index, r_value)` from one epoch transition to
//! the next. The check is constant-cost regardless of how many label
//! updates the epoch contained.
//!
//! For each chain (index, value):
//!   1. Recompute `r_n = O(prev_r, new_data_commitment)` to confirm the
//!      server used the canonical Fiat-Shamir scalar.
//!   2. Derive a random evaluation point `⃗r` from all four commitments
//!      involved in the transition.
//!   3. Verify the four PCS openings (prev/next of data and rand
//!      polynomials, all at `⃗r`).
//!   4. Check the homomorphic relation on the evaluations:
//!         `next_rand_eval = prev_rand_eval + r_n · (next_poly_eval - prev_poly_eval)`.
//!
//! Step 4 enforces `rand_{n+1} = rand_n + r_n · (poly_{n+1} - poly_n)`
//! everywhere on the hypercube, with overwhelming probability via
//! Schwartz-Zippel.

use ark_ec::pairing::Pairing;
use akd_core::aegon_crypto::transcript::IOPTranscript;

use super::config::VerifierContext;
use super::error::AegonError;
use super::fs::{derive_chain_scalar, derive_eval_point};
use super::types::{AegonPcs, AuditState, ChainWitness, EpochCommitment, InvarianceProof};

/// Verify the invariance proof for a single epoch transition. Updates
/// `audit_state` with the new chain scalars on success.
pub fn verify_invariance<E, P>(
    ctx: &VerifierContext<E, P>,
    audit_state: &mut AuditState<E::ScalarField>,
    prev: &EpochCommitment<E, P>,
    next: &EpochCommitment<E, P>,
    proof: &InvarianceProof<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
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
        &ctx.verifier_param,
        ctx.log_capacity,
        b"aegon.invariance.index",
        new_r_index,
        &prev.index_commitment,
        &next.index_commitment,
        &prev.rand_index_commitment,
        &next.rand_index_commitment,
        &proof.index_chain,
    )?;
    if !index_ok {
        return Ok(false);
    }

    let new_r_value = derive_chain_scalar::<E::ScalarField, P::Commitment>(
        b"aegon.fs.r_value",
        audit_state.r_value,
        &next.value_commitment,
    );
    let value_ok = verify_chain::<E, P>(
        &ctx.verifier_param,
        ctx.log_capacity,
        b"aegon.invariance.value",
        new_r_value,
        &prev.value_commitment,
        &next.value_commitment,
        &prev.rand_value_commitment,
        &next.rand_value_commitment,
        &proof.value_chain,
    )?;
    if !value_ok {
        return Ok(false);
    }

    audit_state.r_index = new_r_index;
    audit_state.r_value = new_r_value;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_chain<E, P>(
    vk: &P::VerifierParam,
    log_capacity: usize,
    fs_label: &'static [u8],
    r_n: E::ScalarField,
    prev_poly_com: &P::Commitment,
    next_poly_com: &P::Commitment,
    prev_rand_com: &P::Commitment,
    next_rand_com: &P::Commitment,
    witness: &ChainWitness<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    let point = derive_eval_point::<E, P>(
        fs_label,
        log_capacity,
        prev_poly_com,
        next_poly_com,
        prev_rand_com,
        next_rand_com,
    );

    let verify_one = |com: &P::Commitment,
                      eval: &E::ScalarField,
                      proof: &P::Proof|
     -> Result<bool, AegonError> {
        let mut tr = IOPTranscript::<E::ScalarField>::new(fs_label);
        Ok(P::verify(vk, com, &point, eval, proof, &mut tr)?)
    };

    if !verify_one(prev_poly_com, &witness.prev_poly_eval, &witness.prev_poly_proof)? {
        return Ok(false);
    }
    if !verify_one(next_poly_com, &witness.next_poly_eval, &witness.next_poly_proof)? {
        return Ok(false);
    }
    if !verify_one(prev_rand_com, &witness.prev_rand_eval, &witness.prev_rand_proof)? {
        return Ok(false);
    }
    if !verify_one(next_rand_com, &witness.next_rand_eval, &witness.next_rand_proof)? {
        return Ok(false);
    }

    // Homomorphic relation on evaluations:
    //   next_rand_eval ?= prev_rand_eval + r_n · (next_poly_eval - prev_poly_eval)
    let delta = witness.next_poly_eval - witness.prev_poly_eval;
    let expected = witness.prev_rand_eval + r_n * delta;
    if expected != witness.next_rand_eval {
        return Ok(false);
    }
    Ok(true)
}
