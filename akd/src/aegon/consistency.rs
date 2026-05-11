//! User-side consistency check (paper §5.2 / §6.1).
//!
//! Given two epoch commitments `s0 < s1` (both fetched from the
//! bulletin board) and a `ConsistencyProof` from the server, the user
//! checks that:
//!
//!   * For every probe `ctr ∈ [0, ctr0]`, the rand-index polynomial
//!     opens to the *same* value at `H_bits(ctr, label)` in both `s0`
//!     and `s1`. Equality at the random-coefficient `rand` polynomial
//!     implies, with overwhelming probability, that the underlying
//!     index polynomial was unchanged at every intermediate epoch
//!     (paper Lemma 1).
//!
//!   * The rand-value polynomial opens to the same value at the user's
//!     slot `x_ctr0` in both `s0` and `s1`.
//!
//! Soundness depends on the auditor having verified the invariance
//! chain across the same epoch range — without invariance, the rand
//! polynomials are unanchored and equality at `s0`/`s1` proves
//! nothing.

use ark_ec::pairing::Pairing;
use akd_core::aegon_crypto::transcript::IOPTranscript;

use super::config::VerifierContext;
use super::error::AegonError;
use super::hash::{bool_index_to_point, HashSuite};
use super::types::{AegonPcs, ConsistencyProof, EpochCommitment, Label, RandPair};

/// Verify a consistency proof between two epoch commitments. The
/// expected `ctr0` is the probe count the user originally trusted at
/// `s0` (typically read off a `LookupProof` they kept around) — passed
/// in explicitly so a server can't unilaterally claim a different
/// canonical index.
///
/// Returns `Ok(true)` iff every opening verifies, every paired
/// evaluation agrees, and the proof's claimed `ctr0` matches the
/// expected value.
pub fn verify_consistency<E, P, H>(
    ctx: &VerifierContext<E, P>,
    s0: &EpochCommitment<E, P>,
    s1: &EpochCommitment<E, P>,
    label: &Label,
    expected_ctr0: u64,
    proof: &ConsistencyProof<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    if s0.epoch >= s1.epoch {
        return Err(AegonError::Verification(
            "consistency proof must cover s0 < s1",
        ));
    }
    if proof.ctr0 != expected_ctr0 {
        return Err(AegonError::Verification(
            "proof's claimed ctr0 does not match the user's trusted value",
        ));
    }
    if proof.index_witnesses.len() as u64 != proof.ctr0 + 1 {
        return Err(AegonError::Verification(
            "index_witnesses length does not match ctr0",
        ));
    }

    for (ctr, pair) in proof.index_witnesses.iter().enumerate() {
        let probe_bits = H::h_bits(ctr as u64, label, ctx.log_capacity);
        let point = bool_index_to_point::<E::ScalarField>(&probe_bits);
        if !verify_pair::<E, P>(
            &ctx.verifier_param,
            &point,
            &s0.rand_index_commitment,
            &s1.rand_index_commitment,
            pair,
            b"aegon.rand_index.open",
        )? {
            return Ok(false);
        }
    }

    let final_bits = H::h_bits(proof.ctr0, label, ctx.log_capacity);
    let value_point = bool_index_to_point::<E::ScalarField>(&final_bits);
    if !verify_pair::<E, P>(
        &ctx.verifier_param,
        &value_point,
        &s0.rand_value_commitment,
        &s1.rand_value_commitment,
        &proof.value_witness,
        b"aegon.rand_value.open",
    )? {
        return Ok(false);
    }

    Ok(true)
}

fn verify_pair<E, P>(
    vk: &P::VerifierParam,
    point: &Vec<E::ScalarField>,
    com_s0: &P::Commitment,
    com_s1: &P::Commitment,
    pair: &RandPair<E, P>,
    transcript_label: &'static [u8],
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    let mut tr0 = IOPTranscript::<E::ScalarField>::new(transcript_label);
    if !P::verify(vk, com_s0, point, &pair.eval_s0, &pair.proof_s0, &mut tr0)? {
        return Ok(false);
    }
    let mut tr1 = IOPTranscript::<E::ScalarField>::new(transcript_label);
    if !P::verify(vk, com_s1, point, &pair.eval_s1, &pair.proof_s1, &mut tr1)? {
        return Ok(false);
    }
    if pair.eval_s0 != pair.eval_s1 {
        return Ok(false);
    }
    Ok(true)
}
