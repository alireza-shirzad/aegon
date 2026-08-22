//! IVC auditing: fold Aegon's per-epoch invariance check into a Nova
//! recursive proof (paper §3, "fast-forwarding via recursive proofs").
//!
//! ## The problem
//!
//! [`verify_sharded_invariance`](crate::aegon::verify_sharded_invariance)
//! checks exactly one epoch transition and threads an
//! [`AuditState`](crate::aegon::AuditState) forward. An auditor that
//! misses epochs cannot resume: the chain scalars
//! `(r_index, r_value)` are defined recursively, so catching up means
//! replaying every intermediate transition. In a deployment with
//! minute-long epochs that is a standing requirement to be online
//! forever, which is precisely what makes third-party auditors
//! necessary today.
//!
//! ## What this module does
//!
//! One Nova folding step per epoch transition. The auditor then
//! verifies a *single* recursive proof, at cost independent of how
//! many epochs elapsed, and binds it to the current epoch with two
//! cheap native checks.
//!
//! ## Why the circuit is small
//!
//! A KZH-k commitment is one BN254 G1 point
//! ([`KZHKCommitment`](akd_core::aegon_crypto::pcs::kzhk::structs::KZHKCommitment)),
//! so the whole per-shard audit is:
//!
//! ```text
//!   index chain:  rand_idx' == rand_idx + r_i·(idx' − idx)
//!   value chain:  D = rand_val' − (rand_val + r_v·(val' − val))
//!                 s·h == R + e·D                     (Schnorr, §7)
//! ```
//!
//! Four scalar multiplications and a handful of additions per shard.
//! By running the Nova curve cycle so the circuit's native field is
//! BN254's *base* field, all of that is native arithmetic — see
//! [`bridge`] for the details of the cycle orientation.
//!
//! ## The one protocol change
//!
//! The Fiat–Shamir derivation of `(r_index, r_value)` and of the
//! Schnorr challenge `e` **must** be recomputed inside the circuit —
//! it is exactly what stops a malicious server choosing `r` after
//! seeing the commitments. SHA256 there would cost roughly 4M
//! constraints per epoch at 128 shards, dwarfing all the group
//! arithmetic. So those two derivations move to Poseidon over the
//! circuit field, selected at runtime via
//! [`AuditFs`](crate::aegon::config::AuditFs).
//!
//! Everything else is untouched: the SHA256 Merkle leaf/root/paths
//! stay exactly as they are, because they never need to enter the
//! circuit. The verifier already downloads the epoch's `per_shard`
//! tuple, so it checks the SHA256 root natively just as it does
//! today.

pub mod adapter;
pub mod bridge;
pub mod circuit;
pub mod fs_poseidon;
/// Group-sharded auditing: split the shard set into independent
/// folding chains that prove in parallel. See the module docs.
pub mod grouped;
pub mod prover;
#[cfg(test)]
mod tests;

/// Synthetic epoch-transition fixtures, shared by the unit tests and
/// the `aegon_ivc_bench` binary. Not part of the audit protocol.
pub mod synthetic;
pub mod verifier;
