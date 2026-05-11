//! Client-side verification of an Aegon lookup proof.
//!
//! Re-derives the open-addressing trace from `(label, ctr0)`, verifies the
//! PCS opening at each probe against the published `index_commitment`, and
//! checks the canonical-slot constraints from Fig. 4 of the paper:
//!
//!   - For each `ctr ∈ [0, ctr0)` : `index_n(x_ctr) ∉ {0, H_F(label)}`.
//!     If a probed slot were empty, open-addressing would have stopped
//!     there, contradicting the claim that the canonical index is `x_ctr0`.
//!     If a probed slot held `H_F(label)`, the label would already be
//!     registered at a smaller `ctr`, again contradicting `ctr0`.
//!   - At `ctr = ctr0` : `index_n(x_ctr0) == H_F(label)`.
//!   - The value opening verifies and equals `H_F(value)`.

use ark_ec::pairing::Pairing;
use ark_ff::Zero;
use akd_core::aegon_crypto::transcript::IOPTranscript;

use super::config::VerifierContext;
use super::error::AegonError;
use super::hash::{bool_index_to_point, HashSuite};
use super::types::{AegonPcs, EpochCommitment, Label, LookupProof, Value};

/// Verify a lookup proof. Returns `Ok(true)` iff every probe opening
/// verifies and the open-addressing constraints hold.
///
/// The verifier needs the same `log_capacity` and the same `HashSuite`
/// instantiation that the server used. Mismatch on either is detected at
/// the open-addressing check (the probed evaluations won't agree with
/// `H_F(label)`) but is treated here as a hard error rather than a silent
/// rejection.
pub fn verify_lookup<E, P, H>(
    ctx: &VerifierContext<E, P>,
    commitment: &EpochCommitment<E, P>,
    label: &Label,
    value: &Value,
    proof: &LookupProof<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    let expected_probe_count = proof.ctr0 as usize + 1;
    if proof.probes.len() != expected_probe_count {
        return Err(AegonError::Verification(
            "probe vector length does not match ctr0",
        ));
    }

    let h_label = H::h_f(label);

    // 1. Probe openings + open-addressing constraints.
    for (ctr, (evaluation, opening)) in proof.probes.iter().enumerate() {
        let probe_bits = H::h_bits(ctr as u64, label, ctx.log_capacity);
        let probe_point = bool_index_to_point::<E::ScalarField>(&probe_bits);

        let mut tr = IOPTranscript::<E::ScalarField>::new(b"aegon.index.open");
        let ok = P::verify(
            &ctx.verifier_param,
            &commitment.index_commitment,
            &probe_point,
            evaluation,
            opening,
            &mut tr,
        )?;
        if !ok {
            return Ok(false);
        }

        if (ctr as u64) < proof.ctr0 {
            if evaluation.is_zero() {
                return Err(AegonError::Verification(
                    "earlier probe slot is empty: server picked a non-canonical index",
                ));
            }
            if *evaluation == h_label {
                return Err(AegonError::Verification(
                    "earlier probe slot holds H_F(label): label was already assigned at a smaller counter",
                ));
            }
        } else if *evaluation != h_label {
            return Err(AegonError::Verification(
                "final probe slot does not hold H_F(label)",
            ));
        }
    }

    // 2. Value opening at the same point as the final probe.
    let final_bits = H::h_bits(proof.ctr0, label, ctx.log_capacity);
    let value_point = bool_index_to_point::<E::ScalarField>(&final_bits);

    let mut tr = IOPTranscript::<E::ScalarField>::new(b"aegon.value.open");
    let ok = P::verify(
        &ctx.verifier_param,
        &commitment.value_commitment,
        &value_point,
        &proof.value_evaluation,
        &proof.value_proof,
        &mut tr,
    )?;
    if !ok {
        return Ok(false);
    }

    let expected_value = H::h_f(value);
    if proof.value_evaluation != expected_value {
        return Err(AegonError::Verification(
            "value opening does not match H_F(value)",
        ));
    }

    Ok(true)
}
