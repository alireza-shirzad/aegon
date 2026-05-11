//! Fiat-Shamir helpers shared by server and verifiers.
//!
//! Two derivations are needed:
//!
//! * Per-epoch chain randomness `r_n` (paper Fig. 3, step 2). Drawn from the
//!   transcript `(prev_r, new_poly_commitment)`. Server samples it during
//!   `publish` and applies it to the `rand` polynomial; auditor recomputes
//!   the same value to verify the homomorphic relation.
//!
//! * Random evaluation point for the invariance check (paper Remark 2 /
//!   §6.2). Generic-PCS verification of the homomorphic relation
//!   `rand_{n+1} = rand_n + r_n · (poly_{n+1} - poly_n)` opens both sides
//!   at a uniformly random `⃗r ∈ F^μ` and checks the relation on
//!   evaluations. The point must depend on every commitment in the
//!   relation so the prover cannot adaptively choose it.

use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use akd_core::aegon_crypto::transcript::IOPTranscript;

/// Derive `r_n = O(prev_r, new_commitment)` per Fig. 3. The label
/// distinguishes the index chain from the value chain so they can never
/// collide.
pub(crate) fn derive_chain_scalar<F, C>(
    label: &'static [u8],
    prev_r: F,
    new_commitment: &C,
) -> F
where
    F: PrimeField,
    C: CanonicalSerialize,
{
    let mut t = IOPTranscript::<F>::new(label);
    t.append_field_element(b"prev_r", &prev_r)
        .expect("transcript append");
    t.append_serializable_element(b"new_com", new_commitment)
        .expect("transcript append");
    t.get_and_append_challenge(b"r")
        .expect("transcript challenge")
}

/// Derive a random evaluation point of length `num_vars` for the
/// invariance check on a given chain. Bound to all four commitments
/// involved (prev/next of the data poly, prev/next of the rand poly) so a
/// malicious server cannot pick it.
pub(crate) fn derive_eval_point<E, P>(
    label: &'static [u8],
    num_vars: usize,
    prev_poly_com: &P::Commitment,
    next_poly_com: &P::Commitment,
    prev_rand_com: &P::Commitment,
    next_rand_com: &P::Commitment,
) -> Vec<E::ScalarField>
where
    E: Pairing,
    P: akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme<E>,
{
    let mut t = IOPTranscript::<E::ScalarField>::new(label);
    t.append_serializable_element(b"prev_poly", prev_poly_com)
        .expect("transcript append");
    t.append_serializable_element(b"next_poly", next_poly_com)
        .expect("transcript append");
    t.append_serializable_element(b"prev_rand", prev_rand_com)
        .expect("transcript append");
    t.append_serializable_element(b"next_rand", next_rand_com)
        .expect("transcript append");
    (0..num_vars)
        .map(|i| {
            // Domain-separate per coordinate so reordering can't collide.
            let bytes = (i as u32).to_le_bytes();
            t.append_message(b"i", &bytes).expect("transcript append");
            t.get_and_append_challenge(b"r_i")
                .expect("transcript challenge")
        })
        .collect()
}
