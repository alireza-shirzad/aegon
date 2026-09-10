// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Bridge between Aegon's published types and the IVC audit path.
//!
//! Two jobs:
//!
//! * extract the group elements the folding circuit consumes out of a
//!   published [`ShardedEpochCommitment`], and
//! * supply the Poseidon [`AuditFsHooks`] that a deployment must
//!   install on both server and verifier for IVC auditing to work.
//!
//! Everything here is concrete at `(Bn254, KZH-k)`, deliberately. The
//! rest of Aegon is generic over the pairing and the PCS, but the
//! folding circuit is not and cannot be: its whole efficiency
//! argument rests on the circuit's native field being BN254's base
//! field (see [`super::bridge`]). Pinning the instantiation here
//! keeps that assumption in one visible place instead of smearing a
//! BN254 bound across the generic core.

use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::{Bn254, Fr as ArkFr, G1Affine as ArkG1Affine};

use super::fs_poseidon::SigmaWitness;
use super::fs_poseidon::{
    domain, poseidon_chain_scalar, poseidon_sigma_challenge, ro_constants_cached, FsParams,
    ShardCommitments,
};
use crate::aegon::audit_fs::{AuditFs, AuditFsHooks};
use crate::aegon::error::AegonError;
use crate::aegon::sharded::ShardedEpochCommitment;
use crate::aegon::types::EpochCommitment;

/// The PCS the IVC audit path is defined against.
pub type Pcs = KZHK<Bn254>;
/// A single shard's published epoch commitment.
pub type ShardEpoch = EpochCommitment<Bn254, Pcs>;
/// A whole epoch's published commitment, across all shards.
pub type ShardedEpoch = ShardedEpochCommitment<Bn254, Pcs>;

type Commitment =
    <Pcs as akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme<Bn254>>::Commitment;

fn point(c: &Commitment) -> ArkG1Affine {
    c.get_commitment()
}

/// The four group elements of one shard's epoch commitment, in the
/// order the state digest absorbs them.
pub fn shard_commitments(c: &ShardEpoch) -> ShardCommitments {
    ShardCommitments {
        index: point(&c.index_commitment),
        value: point(&c.value_commitment),
        rand_index: point(&c.rand_index_commitment),
        rand_value: point(&c.rand_value_commitment),
    }
}

/// Every shard's commitments for one epoch, in shard-id order.
pub fn epoch_commitments(e: &ShardedEpoch) -> Vec<ShardCommitments> {
    e.per_shard.iter().map(shard_commitments).collect()
}

/// Every shard's value-chain Schnorr proof for one epoch.
///
/// Errors when a shard is missing its proof. Under a hiding SRS that
/// is a protocol violation, not a recoverable state: the audit path
/// requires the re-randomisation proof from *every* shard (paper §7),
/// and one shard skipping it would break the privacy argument even if
/// all the others comply.
pub fn epoch_sigma_witnesses(e: &ShardedEpoch) -> Result<Vec<SigmaWitness>, AegonError> {
    e.per_shard
        .iter()
        .enumerate()
        .map(|(j, c)| {
            let proof = c.audit_value_blinding_proof.as_ref().ok_or_else(|| {
                AegonError::Ivc(format!(
                    "shard {j} of epoch {} has no value-chain blinding proof; IVC auditing \
                     requires a hiding SRS with re-randomisation enabled",
                    e.epoch
                ))
            })?;
            Ok(SigmaWitness {
                r_commit: point(&proof.r_commit),
                response: proof.response,
            })
        })
        .collect()
}

/// Derive the [`FsParams`] for a deployment from a published epoch.
///
/// `num_vars` comes from the commitments themselves (every shard's
/// polynomials are identically shaped, since all shards share one
/// SRS), and `n_shards` from the tuple length.
pub fn fs_params_from_epoch(e: &ShardedEpoch) -> Result<FsParams, AegonError> {
    let first = e
        .per_shard
        .first()
        .ok_or_else(|| AegonError::Ivc("epoch commitment has no shards".into()))?;
    Ok(FsParams {
        num_vars: first.index_commitment.get_num_vars(),
        n_shards: e.per_shard.len(),
    })
}

// ---------- the Poseidon audit-FS bundle -------------------------------
//
// These are plain `fn` items so they can be stored as function
// pointers in `AuditFsHooks`. Everything they need beyond their
// arguments — the polynomial width and the shard count — is
// recoverable from the commitments they are handed, which is why the
// bundle needs no captured state.

fn params_from(commits: &[Commitment]) -> FsParams {
    FsParams {
        num_vars: commits.first().map(|c| c.get_num_vars()).unwrap_or(0),
        n_shards: commits.len(),
    }
}

fn poseidon_chain(label: &'static [u8], prev: ArkFr, commits: &[Commitment]) -> ArkFr {
    // Map the byte tag the rest of the system uses onto the numeric
    // domain the circuit allocates as a constant.
    let dom = match label {
        b"aegon.sharded.fs.r_value" => domain::CHAIN_VALUE,
        _ => domain::CHAIN_INDEX,
    };
    let points: Vec<ArkG1Affine> = commits.iter().map(point).collect();
    poseidon_chain_scalar(
        ro_constants_cached(),
        dom,
        params_from(commits),
        prev,
        &points,
    )
}

fn poseidon_sigma(
    prev_poly: &Commitment,
    next_poly: &Commitment,
    prev_rand: &Commitment,
    next_rand: &Commitment,
    r_chain: ArkFr,
    r_commit: &Commitment,
) -> ArkFr {
    // A sigma transcript is per-shard, so it binds the polynomial
    // width only: the chain scalar `r_chain`, which *is* in the
    // transcript, already binds every shard's commitments.
    poseidon_sigma_challenge(
        ro_constants_cached(),
        prev_poly.get_num_vars(),
        &point(prev_poly),
        &point(next_poly),
        &point(prev_rand),
        &point(next_rand),
        r_chain,
        &point(r_commit),
    )
}

/// The Poseidon audit-path Fiat–Shamir bundle.
///
/// Install this on the server (via
/// [`ShardedAegonConfigBuilder::audit_fs`](crate::aegon::ShardedAegonConfigBuilder))
/// **and** on every verifier context
/// ([`VerifierContext::with_audit_fs`](crate::aegon::VerifierContext)).
/// A deployment that installs it on only one side will simply fail
/// every audit — the mismatch cannot be mistaken for acceptance.
pub fn poseidon_audit_fs() -> AuditFsHooks<Bn254, Pcs> {
    AuditFsHooks::new(AuditFs::Poseidon, poseidon_chain, poseidon_sigma)
}

/// The hook bundle for a named transcript, at this crate's concrete
/// `(BN254, KZH-k)` instantiation.
///
/// This is what a deployment should call: [`AuditFs::default`] is
/// [`AuditFs::Poseidon`], so `hooks_for(Default::default())` installs
/// the default transcript, and a `--audit-fs sha256` flag threads
/// straight through.
///
/// Note the generic [`AuditFsHooks::default`] still yields SHA-256, and
/// has to: the Poseidon derivations hash into BN254's base field and are
/// simply undefined for another curve, so a generic
/// `AuditFsHooks<E, P>` has no Poseidon to fall back to. Every concrete
/// entry point in this crate — the `Directory`, the shard and
/// coordinator servers, the client — routes through here instead.
pub fn hooks_for(kind: AuditFs) -> AuditFsHooks<Bn254, Pcs> {
    match kind {
        AuditFs::Poseidon => poseidon_audit_fs(),
        AuditFs::Sha256 => AuditFsHooks::sha256(),
    }
}

// ---------- deriving the partition from the deployment ------------------

use crate::aegon::chain_groups::GroupPlan;
use crate::aegon::sharded::ShardedVerifierContext;

/// The chain-group partition a deployment audits under, taken from
/// its verifier context.
///
/// **Use this rather than building a [`GroupPlan`] by hand.** The
/// IVC's group count is not a free parameter: the folding circuit
/// re-derives the Fiat–Shamir chain scalars *in circuit*, absorbing
/// the commitments of the shards in its group. If the server derived
/// `r` over all 128 shards and the circuit absorbs 16, the scalars
/// differ and every chain check fails — with no diagnostic beyond a
/// proof that will not verify. The two counts are one parameter, and
/// [`ShardedVerifierContext::chain_groups`] is where it lives.
pub fn group_plan_from_context<E, P>(
    ctx: &ShardedVerifierContext<E, P>,
) -> Result<GroupPlan, AegonError>
where
    E: ark_ec::pairing::Pairing,
    P: crate::aegon::types::AegonPcs<E>,
{
    ctx.group_plan()
}

/// The partition for a deployment described by a published epoch, at
/// a stated group count.
///
/// For callers holding a [`ShardedEpoch`] but no verifier context —
/// a detached folding prover tailing the bulletin board, say. The
/// shard count comes from the epoch tuple; `groups` must still match
/// what the server was configured with, since nothing in the
/// published data reveals it.
pub fn group_plan_from_epoch(e: &ShardedEpoch, groups: usize) -> Result<GroupPlan, AegonError> {
    GroupPlan::new(e.per_shard.len(), groups)
}
