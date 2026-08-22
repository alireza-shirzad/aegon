//! Schnorr-style sigma protocol for "commitment difference is a known
//! multiple of `h`" — the audit-path bridge between zk-blinded
//! commitments and the bare homomorphism equation.
//!
//! ## Why this exists
//!
//! In zk mode the value-side commitments carry a Pedersen-style
//! blinding term: `C_hide(f) = ⟨f, H₁⟩ + τ_f · h`. The auditor's
//! invariance check
//!
//! ```text
//!   C(rand_{n+1}) ?= C(rand_n) + r_n · (C(val_{n+1}) − C(val_n))
//! ```
//!
//! is a group equation, so equality requires the blinding terms to
//! satisfy the same chain rule:
//!
//! ```text
//!   τ_rand_{n+1} ?= τ_rand_n + r_n · (τ_val_{n+1} − τ_val_n).
//! ```
//!
//! If the server picked `τ_rand_{n+1}` deterministically to satisfy
//! this — i.e. carried the chain-combined tau out of the homomorphic
//! commitment update — the audit passes but the joint distribution of
//! taus across epochs lives in a strict subspace and `rand_value`
//! openings across epochs become linearly correlated. That breaks
//! the zk simulator argument (paper Appendix D), which assumes
//! independent fresh blindings per commitment.
//!
//! The fix (paper §7 "Checking Homomorphic Relations Over zk-KZH"):
//! the server **re-randomises** every published `rand_value`
//! commitment with a freshly-sampled `τ`, then proves that the
//! residue
//!
//! ```text
//!   D = C(rand_{n+1}) − ( C(rand_n) + r_n · (C(val_{n+1}) − C(val_n)) )
//! ```
//!
//! is a scalar multiple of `h` (rather than zero). `D = c · h` where
//! `c = τ_rand_{n+1} − τ_chained` is known only to the server. A
//! Schnorr proof of knowledge of `c` w.r.t. base `h` finishes it: the
//! auditor accepts iff the proof verifies, certifying the underlying
//! polynomials still satisfy the chain rule even though the
//! commitments' blindings don't.
//!
//! ## Protocol shape
//!
//! Standard Fiat-Shamir Schnorr over the hiding generator `h`:
//!
//! * **Prover**:
//!   1. Sample `k ← F` uniformly.
//!   2. `R := k · h`.
//!   3. `e := H( prev_val ∥ next_val ∥ prev_rand ∥ next_rand ∥ r ∥ R )`.
//!   4. `s := k + e · c`.
//!   5. Send `(R, s)`.
//! * **Verifier**:
//!   1. Recompute `e` from the same transcript.
//!   2. Compute `D := next_rand − (prev_rand + r · (next_val − prev_val))`.
//!   3. Accept iff `s · h == R + e · D`.
//!
//! Soundness reduces to the discrete log of `h` (in turn the binding
//! property of the underlying KZH-k SRS). Zero-knowledge of `c` and
//! of `τ` follows from the standard Schnorr simulator.
//!
//! ## Scope
//!
//! The protocol runs only on the **value chain**. Index polynomials
//! are committed non-zk in Aegon (the masking-server protocol only
//! needs `τ` on the value side; see `commit_with_aux_non_zk` vs
//! `commit_with_aux_value_side` in [`super::server`]), so their
//! audit equation already holds exactly with no blinding residue and
//! no sigma proof is needed there.

use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::{CryptoRng, RngCore};
use std::ops::{Add, Mul, Sub};

use super::types::AegonPcs;

/// Fiat-Shamir Schnorr proof that a public commitment residue equals
/// `c · h` for some `c ∈ F` known to the prover.
///
/// Wire format: one group element (`r_commit = k · h`, commitment-
/// shaped so the auditor can run `Add`/`Sub` against the residue) plus
/// one scalar (`response = k + e · c`). Round-trips through arkworks
/// `CanonicalSerialize` alongside the rest of `EpochCommitment`.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindingEqProof<E: Pairing, P: AegonPcs<E>> {
    pub r_commit: P::Commitment,
    pub response: E::ScalarField,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for BlindingEqProof<E, P> {
    fn clone(&self) -> Self {
        Self {
            r_commit: self.r_commit.clone(),
            response: self.response,
        }
    }
}

impl<E: Pairing, P: AegonPcs<E>> PartialEq for BlindingEqProof<E, P>
where
    P::Commitment: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.r_commit == other.r_commit && self.response == other.response
    }
}

// The Fiat-Shamir challenge is no longer computed here: it is
// supplied by the deployment's
// [`AuditFsHooks`](super::audit_fs::AuditFsHooks), so that the same
// derivation can be recomputed inside the IVC folding circuit. The
// SHA256 implementation that used to live in this file is now
// `audit_fs::sha256_sigma_challenge` and remains the default, so
// `AuditFs::Sha256` deployments are byte-identical to before.

/// Build a proof that `residue = c · h` where
/// `residue = next_rand − (prev_rand + r_chain · (next_val − prev_val))`
/// and `c` is the witness (the blinding shift the server applied when
/// re-randomising the published `next_rand` commitment).
///
/// `pp` is the PCS prover parameter; the function uses it to lift
/// `k · h` into the commitment group via [`AegonPcs::scaled_mask_generator_pp`].
/// Returns `None` when the SRS is non-hiding (no `h` available);
/// callers in that mode should leave the proof slot empty and let the
/// audit path enforce `residue == 0` directly.
#[allow(clippy::too_many_arguments)]
pub fn prove<E, P, R>(
    pp: &P::ProverParam,
    model: &P::Commitment,
    prev_poly_com: &P::Commitment,
    next_poly_com: &P::Commitment,
    prev_rand_com: &P::Commitment,
    next_rand_com: &P::Commitment,
    r_chain: E::ScalarField,
    c: E::ScalarField,
    audit_fs: &super::audit_fs::AuditFsHooks<E, P>,
    rng: &mut R,
) -> Option<BlindingEqProof<E, P>>
where
    E: Pairing,
    P: AegonPcs<E>,
    R: RngCore + CryptoRng,
{
    let k = E::ScalarField::rand(rng);
    let r_commit = P::scaled_mask_generator_pp(pp, model, k)?;
    let e = audit_fs.sigma_challenge(
        prev_poly_com,
        next_poly_com,
        prev_rand_com,
        next_rand_com,
        r_chain,
        &r_commit,
    );
    let response = k + e * c;
    Some(BlindingEqProof { r_commit, response })
}

/// Verify a [`BlindingEqProof`] against the residue derived from the
/// four public chain commitments and the chain scalar. Returns `true`
/// iff the Schnorr equation `s · h == R + e · residue` holds.
///
/// `vk` reads `h` out of the verifier parameters; the auditor never
/// needs prover state. Returns `false` if the SRS is non-hiding (so
/// the proof should not exist in the first place) — the audit caller
/// in that branch must compare `residue == 0` instead.
#[allow(clippy::too_many_arguments)]
pub fn verify<E, P>(
    vk: &P::VerifierParam,
    prev_poly_com: &P::Commitment,
    next_poly_com: &P::Commitment,
    prev_rand_com: &P::Commitment,
    next_rand_com: &P::Commitment,
    r_chain: E::ScalarField,
    proof: &BlindingEqProof<E, P>,
    audit_fs: &super::audit_fs::AuditFsHooks<E, P>,
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
    let e = audit_fs.sigma_challenge(
        prev_poly_com,
        next_poly_com,
        prev_rand_com,
        next_rand_com,
        r_chain,
        &proof.r_commit,
    );
    let delta_poly = next_poly_com.clone() - prev_poly_com.clone();
    let chained = prev_rand_com.clone() + delta_poly * r_chain;
    let residue = next_rand_com.clone() - chained;
    let s_h = match P::scaled_mask_generator_vk(vk, &proof.r_commit, proof.response) {
        Some(c) => c,
        None => return false,
    };
    let rhs = proof.r_commit.clone() + residue * e;
    s_h == rhs
}

#[cfg(test)]
mod tests {
    use super::*;
    use akd_core::aegon_crypto::pcs::kzhk::KZHK;
    use akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme;
    use akd_core::aegon_crypto::pcs::StructuredReferenceString;
    use ark_bn254::{Bn254, Fr};
    use ark_std::rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    type Pcs = KZHK<Bn254>;

    /// Build a `(pp, vk, model)` triple from a fresh zk SRS. The
    /// `model` commitment is a `KZHKCommitment` with the right `nv`
    /// for `scaled_mask_generator_*` to lift `c · h` into a
    /// commitment that Add/Subs against the residue.
    fn fixture(
        seed: u64,
        num_vars: usize,
        k: usize,
    ) -> (
        <Pcs as PolynomialCommitmentScheme<Bn254>>::ProverParam,
        <Pcs as PolynomialCommitmentScheme<Bn254>>::VerifierParam,
        <Pcs as PolynomialCommitmentScheme<Bn254>>::Commitment,
    ) {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let srs = <akd_core::aegon_crypto::pcs::kzhk::srs::KZHKUniversalParams<Bn254> as StructuredReferenceString<Bn254>>::gen_srs_for_testing(
            &mut rng, k, true, num_vars,
        )
        .expect("srs gen");
        let (pp, vk) =
            <akd_core::aegon_crypto::pcs::kzhk::srs::KZHKUniversalParams<Bn254> as StructuredReferenceString<Bn254>>::trim(&srs, num_vars).expect("trim");
        let model =
            <Pcs as PolynomialCommitmentScheme<Bn254>>::scaled_mask_generator_pp(&pp, &Default::default(), Fr::from(1u64))
                .expect("zk model");
        (pp, vk, model)
    }

    /// Round-trip check: a correctly produced proof verifies under
    /// the matching residue and verifier param.
    #[test]
    fn round_trip_verifies() {
        let (pp, vk, model) = fixture(0xA56_6, 10, 5);
        let mut rng = ChaCha20Rng::seed_from_u64(1);

        let prev_val = model.clone();
        let next_val = model.clone();
        let prev_rand = model.clone();
        let r_chain = Fr::from(7u64);
        let c_witness = Fr::from(123_456u64);
        // next_rand = chain + c·h is exactly what the publish path
        // produces after re-randomising the chained commitment.
        let chain = prev_rand.clone() + (next_val.clone() - prev_val.clone()) * r_chain;
        let bump =
            <Pcs as PolynomialCommitmentScheme<Bn254>>::scaled_mask_generator_pp(&pp, &model, c_witness)
                .expect("zk pp has h");
        let next_rand = chain + bump;

        let hooks = super::super::audit_fs::AuditFsHooks::<Bn254, Pcs>::sha256();
        let proof = prove::<Bn254, Pcs, _>(
            &pp, &model, &prev_val, &next_val, &prev_rand, &next_rand, r_chain, c_witness, &hooks,
            &mut rng,
        )
        .expect("zk pp has h");
        assert!(verify::<Bn254, Pcs>(
            &vk, &prev_val, &next_val, &prev_rand, &next_rand, r_chain, &proof, &hooks
        ));
    }

    /// Tamper checks: changing the chain scalar or dropping the
    /// `c·h` bump trips the Schnorr verification.
    #[test]
    fn tamper_breaks_verification() {
        let (pp, vk, model) = fixture(0xA56_7, 10, 5);
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let prev_val = model.clone();
        let next_val = model.clone();
        let prev_rand = model.clone();
        let r_chain = Fr::from(11u64);
        let c_witness = Fr::from(99u64);
        let chain = prev_rand.clone() + (next_val.clone() - prev_val.clone()) * r_chain;
        let bump =
            <Pcs as PolynomialCommitmentScheme<Bn254>>::scaled_mask_generator_pp(&pp, &model, c_witness)
                .unwrap();
        let next_rand = chain.clone() + bump;
        let hooks = super::super::audit_fs::AuditFsHooks::<Bn254, Pcs>::sha256();
        let proof = prove::<Bn254, Pcs, _>(
            &pp, &model, &prev_val, &next_val, &prev_rand, &next_rand, r_chain, c_witness, &hooks,
            &mut rng,
        )
        .unwrap();
        assert!(!verify::<Bn254, Pcs>(
            &vk,
            &prev_val,
            &next_val,
            &prev_rand,
            &next_rand,
            r_chain + Fr::from(1u64),
            &proof,
            &hooks
        ));
        let chain_only = prev_rand.clone() + (next_val.clone() - prev_val.clone()) * r_chain;
        assert!(!verify::<Bn254, Pcs>(
            &vk,
            &prev_val,
            &next_val,
            &prev_rand,
            &chain_only,
            r_chain,
            &proof,
            &hooks
        ));
    }
}
