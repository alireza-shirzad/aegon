//! Which hash the *audit path's* Fiat–Shamir derivations use.
//!
//! ## Why this is selectable at all
//!
//! Two derivations decide whether an epoch transition is accepted:
//! the chain scalars `(r_index, r_value)`, and the per-shard Schnorr
//! challenge `e` behind the value-chain blinding-equality proof
//! (paper §7). Both are SHA256 today, which is the right choice for a
//! native auditor — it is fast and needs no extra machinery.
//!
//! It is the wrong choice inside a folding circuit. The IVC auditor
//! ([`crate::aegon::ivc`]) has to *recompute* these challenges in
//! R1CS, because a server free to pick `r` after seeing the
//! commitments could satisfy the chain equation with forged
//! polynomials. At 128 shards, SHA256 there costs on the order of 4M
//! constraints per epoch — several times the cost of all the elliptic
//! curve work the audit actually consists of. Poseidon over the
//! circuit field costs roughly 200k.
//!
//! So the derivations are pluggable. [`AuditFs::Sha256`] is the
//! default and is byte-for-byte what the system did before; a
//! deployment that wants IVC auditing selects [`AuditFs::Poseidon`],
//! and server and auditor must agree.
//!
//! ## Scope
//!
//! Only these two derivations are affected. In particular the SHA256
//! Merkle leaf/root/path hashing is **unchanged** — it never has to
//! enter the circuit, because the auditor already holds the epoch's
//! `per_shard` tuple and recomputes the root natively. Lookup,
//! consistency and history proofs are untouched.
//!
//! ## Shape
//!
//! Plain function pointers, deliberately: an `Arc<dyn Fn ..>` would
//! impose a `'static` bound on the PCS type parameter and that bound
//! then cascades through every generic impl that touches a config or
//! a verifier context. Function pointers carry the same
//! swappability with none of that. The derivations recover whatever
//! shape they need (`num_vars`, and `n_shards` where it is
//! meaningful) from the commitments they are handed.

use std::fmt;

use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use sha2::{Digest, Sha256};

use super::types::AegonPcs;

/// Which hash family the audit-path Fiat–Shamir derivations use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AuditFs {
    /// The original SHA256 derivations. Default; unchanged wire
    /// behaviour.
    #[default]
    Sha256,
    /// Poseidon over BN254's base field, so the derivations can be
    /// recomputed cheaply inside the Nova folding circuit. Requires
    /// the `ivc_audit` feature to construct — see
    /// [`crate::aegon::ivc::adapter::poseidon_audit_fs`].
    Poseidon,
}

/// Derive a chain scalar from the previous one and every shard's new
/// data commitment (in shard-id order).
pub type ChainScalarFn<E, P> = fn(
    &'static [u8],
    <E as Pairing>::ScalarField,
    &[<P as akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme<E>>::Commitment],
) -> <E as Pairing>::ScalarField;

/// Derive the Schnorr challenge for one shard's value-chain
/// blinding-equality proof, from the four chain commitments, the
/// chain scalar, and the prover's Schnorr commitment `R`.
pub type SigmaChallengeFn<E, P> = fn(
    &<P as akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme<E>>::Commitment,
    &<P as akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme<E>>::Commitment,
    &<P as akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme<E>>::Commitment,
    &<P as akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme<E>>::Commitment,
    <E as Pairing>::ScalarField,
    &<P as akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme<E>>::Commitment,
) -> <E as Pairing>::ScalarField;

/// The audit path's Fiat–Shamir derivations, as a swappable bundle.
///
/// Server and auditor must be configured identically: the server
/// applies `chain_scalar` when building an epoch and `sigma_challenge`
/// when proving the blinding equality, and the auditor recomputes
/// both. A mismatch shows up as a failed audit, not as silent
/// acceptance.
pub struct AuditFsHooks<E: Pairing, P: AegonPcs<E>> {
    kind: AuditFs,
    chain_scalar: ChainScalarFn<E, P>,
    sigma_challenge: SigmaChallengeFn<E, P>,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for AuditFsHooks<E, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Pairing, P: AegonPcs<E>> Copy for AuditFsHooks<E, P> {}

impl<E: Pairing, P: AegonPcs<E>> fmt::Debug for AuditFsHooks<E, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuditFsHooks")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl<E: Pairing, P: AegonPcs<E>> AuditFsHooks<E, P> {
    /// Assemble a custom bundle. Used by
    /// [`crate::aegon::ivc::adapter`] to install the Poseidon
    /// derivations; deployments normally take [`Self::sha256`].
    pub fn new(
        kind: AuditFs,
        chain_scalar: ChainScalarFn<E, P>,
        sigma_challenge: SigmaChallengeFn<E, P>,
    ) -> Self {
        Self {
            kind,
            chain_scalar,
            sigma_challenge,
        }
    }

    /// The original SHA256 derivations.
    pub const fn sha256() -> Self {
        Self {
            kind: AuditFs::Sha256,
            chain_scalar: sha256_chain_scalar::<E, P>,
            sigma_challenge: sha256_sigma_challenge::<E, P>,
        }
    }

    /// Which hash family this bundle uses.
    pub fn kind(&self) -> AuditFs {
        self.kind
    }

    /// `r' = H(domain, prev_r, commits…)`.
    pub fn chain_scalar(
        &self,
        domain: &'static [u8],
        prev: E::ScalarField,
        commits: &[P::Commitment],
    ) -> E::ScalarField {
        (self.chain_scalar)(domain, prev, commits)
    }

    /// The Schnorr challenge `e` for one shard's blinding-equality proof.
    pub fn sigma_challenge(
        &self,
        prev_poly: &P::Commitment,
        next_poly: &P::Commitment,
        prev_rand: &P::Commitment,
        next_rand: &P::Commitment,
        r_chain: E::ScalarField,
        r_commit: &P::Commitment,
    ) -> E::ScalarField {
        (self.sigma_challenge)(prev_poly, next_poly, prev_rand, next_rand, r_chain, r_commit)
    }
}

impl<E: Pairing, P: AegonPcs<E>> Default for AuditFsHooks<E, P> {
    fn default() -> Self {
        Self::sha256()
    }
}

// ---------- the SHA256 derivations -------------------------------------
//
// Bodies lifted verbatim from `sharded::fs_chain_scalar` and
// `sigma::fs_challenge` so that `AuditFs::Sha256` is bit-identical to
// the pre-existing behaviour.

/// FS-derive a field-element challenge from `(domain, prev, &[commits…])`.
pub fn sha256_chain_scalar<E: Pairing, P: AegonPcs<E>>(
    domain: &'static [u8],
    prev: E::ScalarField,
    commits: &[P::Commitment],
) -> E::ScalarField {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    let mut buf = Vec::new();
    prev.serialize_compressed(&mut buf)
        .expect("F serialize is infallible");
    hasher.update(&buf);
    for c in commits {
        buf.clear();
        c.serialize_compressed(&mut buf)
            .expect("commit serialize is infallible");
        hasher.update(&buf);
    }
    let digest = hasher.finalize();
    E::ScalarField::from_le_bytes_mod_order(&digest)
}

const SIGMA_DOMAIN: &[u8] = b"aegon.audit.blinding_eq.v1";

/// The blinding-equality Schnorr challenge. Binds all four chain
/// commitments, the chain scalar and `R`, so the prover cannot pick
/// any of them after seeing the others.
pub fn sha256_sigma_challenge<E: Pairing, P: AegonPcs<E>>(
    prev_poly: &P::Commitment,
    next_poly: &P::Commitment,
    prev_rand: &P::Commitment,
    next_rand: &P::Commitment,
    r_chain: E::ScalarField,
    r_commit: &P::Commitment,
) -> E::ScalarField {
    use akd_core::aegon_crypto::transcript::IOPTranscript;
    let mut t = IOPTranscript::<E::ScalarField>::new(SIGMA_DOMAIN);
    t.append_serializable_element(b"prev_val", prev_poly)
        .expect("transcript append");
    t.append_serializable_element(b"next_val", next_poly)
        .expect("transcript append");
    t.append_serializable_element(b"prev_rand", prev_rand)
        .expect("transcript append");
    t.append_serializable_element(b"next_rand", next_rand)
        .expect("transcript append");
    t.append_field_element(b"r", &r_chain)
        .expect("transcript append");
    t.append_serializable_element(b"R", r_commit)
        .expect("transcript append");
    t.get_and_append_challenge(b"e")
        .expect("transcript challenge")
}
