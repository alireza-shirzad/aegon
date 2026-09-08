// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Synthetic epoch transitions for testing and benchmarking.
//!
//! Builds honest epoch transitions from **pure group arithmetic**
//! rather than by driving a real `ShardedAegon` publish. That is not
//! a shortcut: the circuit's statement is entirely about group
//! relations between published commitments, so synthesising those
//! relations directly exercises every constraint while keeping the
//! unit tests free of SRS generation.
//!
//! It is also what makes the benchmark honest at scale: the audit's
//! cost depends on the shard count, not on how many keys the
//! directory holds, so a 128-shard measurement does not require
//! standing up 128 real shards with a planetary-scale SRS.
//!
//! The `ivc_audit_e2e` example covers the real publish path, and
//! confirms these fixtures match what an honest server actually
//! produces.

use ark_bn254::{Fr as ArkFr, G1Affine as ArkG1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::PrimeField;
use ark_std::rand::SeedableRng;
use ark_std::UniformRand;
use rand_chacha::ChaCha20Rng;

use super::circuit::SigmaWitness;
use super::fs_poseidon::{
    domain, poseidon_chain_scalar, poseidon_sigma_challenge, ro_constants, FsParams,
    ShardCommitments,
};

/// Stand-in for the SRS hiding generator. The circuit only ever uses
/// `h` through group operations, so any generator multiple exercises
/// exactly the constraints a real SRS `h` would.
pub fn test_h() -> ArkG1Affine {
    (G1Projective::generator() * ArkFr::from(0xABCD_EF01_2345_6789u64)).into_affine()
}

/// A deterministic RNG for reproducible fixtures.
pub fn rng(seed: u64) -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(seed)
}

/// A uniformly random non-identity BN254 G1 point.
pub fn rand_point(rng: &mut ChaCha20Rng) -> ArkG1Affine {
    (G1Projective::generator() * ArkFr::rand(rng)).into_affine()
}

/// Deployment shape for a directory with `n_shards` shards.
pub fn params(n_shards: usize) -> FsParams {
    FsParams {
        num_vars: 12,
        n_shards,
    }
}

/// Epoch 0 under `Server.Init`: every polynomial is zero, so every
/// commitment is the identity.
pub fn genesis(n: usize) -> Vec<ShardCommitments> {
    vec![
        ShardCommitments {
            index: ArkG1Affine::identity(),
            value: ArkG1Affine::identity(),
            rand_index: ArkG1Affine::identity(),
            rand_value: ArkG1Affine::identity(),
        };
        n
    ]
}

/// A random non-genesis epoch, for tests that do not care how the
/// state was reached.
pub fn random_epoch(n: usize, rng: &mut ChaCha20Rng) -> Vec<ShardCommitments> {
    (0..n)
        .map(|_| ShardCommitments {
            index: rand_point(rng),
            value: rand_point(rng),
            rand_index: rand_point(rng),
            rand_value: rand_point(rng),
        })
        .collect()
}

/// One honest epoch transition and everything derived from it.
pub struct Transition {
    /// Epoch `n` commitments.
    pub prev: Vec<ShardCommitments>,
    /// Epoch `n+1` commitments.
    pub next: Vec<ShardCommitments>,
    /// One value-chain Schnorr proof per shard.
    pub sigma: Vec<SigmaWitness>,
    /// The index chain scalar this transition derived.
    pub r_index: ArkFr,
    /// The value chain scalar this transition derived.
    pub r_value: ArkFr,
}

/// Build an honest transition out of `prev`.
///
/// Reproduces exactly what an honest server's publish path does:
///
/// * the index chain is *exact* — index polynomials are never
///   blinded, so `rand_index' = rand_index + r_index·Δ`;
/// * the value chain carries a **fresh** blinding `c·h` on the
///   published `rand_value`, and a Schnorr proof of knowledge of `c`.
///   Re-randomising per epoch is what the paper's §7 requires; a
///   server that instead carried the chained blinding through would
///   make value openings linearly correlated across epochs and break
///   the privacy argument.
///
/// With `with_updates = false` the data commitments are left
/// untouched, which is the common production case of a shard that saw
/// no activity this epoch — every delta is then the identity.
pub fn honest_transition(
    p: FsParams,
    h: ArkG1Affine,
    prev: &[ShardCommitments],
    prev_r_index: ArkFr,
    prev_r_value: ArkFr,
    rng: &mut ChaCha20Rng,
    with_updates: bool,
) -> Transition {
    let c = ro_constants();

    let bump = |base: ArkG1Affine, rng: &mut ChaCha20Rng| {
        if with_updates {
            (base + rand_point(rng)).into_affine()
        } else {
            base
        }
    };
    let new_index: Vec<ArkG1Affine> = prev.iter().map(|s| bump(s.index, rng)).collect();
    let new_value: Vec<ArkG1Affine> = prev.iter().map(|s| bump(s.value, rng)).collect();

    let r_index = poseidon_chain_scalar(&c, domain::CHAIN_INDEX, p, prev_r_index, &new_index);
    let r_value = poseidon_chain_scalar(&c, domain::CHAIN_VALUE, p, prev_r_value, &new_value);

    let mut next = Vec::with_capacity(prev.len());
    let mut sigma = Vec::with_capacity(prev.len());
    for (j, s) in prev.iter().enumerate() {
        let d_index = (new_index[j].into_group() - s.index.into_group()) * r_index;
        let rand_index = (s.rand_index.into_group() + d_index).into_affine();

        let d_value = (new_value[j].into_group() - s.value.into_group()) * r_value;
        let chained = s.rand_value.into_group() + d_value;
        let blind = ArkFr::rand(rng);
        let rand_value = (chained + h.into_group() * blind).into_affine();

        let nc = ShardCommitments {
            index: new_index[j],
            value: new_value[j],
            rand_index,
            rand_value,
        };

        // Schnorr proof of knowledge of `blind` w.r.t. base h.
        let k = ArkFr::rand(rng);
        let r_commit = (h.into_group() * k).into_affine();
        let e = poseidon_sigma_challenge(
            &c,
            p.num_vars,
            &s.value,
            &nc.value,
            &s.rand_value,
            &nc.rand_value,
            r_value,
            &r_commit,
        );
        sigma.push(SigmaWitness {
            r_commit,
            response: k + e * blind,
        });
        next.push(nc);
    }

    Transition {
        prev: prev.to_vec(),
        next,
        sigma,
        r_index,
        r_value,
    }
}

/// Fold `n_epochs` honest transitions starting from genesis,
/// returning every epoch's commitments and the sigma proofs for each
/// transition.
pub fn honest_chain(
    p: FsParams,
    h: ArkG1Affine,
    n_epochs: usize,
    rng: &mut ChaCha20Rng,
) -> (Vec<Vec<ShardCommitments>>, Vec<Vec<SigmaWitness>>) {
    let mut epochs = vec![genesis(p.n_shards)];
    let mut sigmas = Vec::new();
    let (mut ri, mut rv) = (ArkFr::from(0u64), ArkFr::from(0u64));
    for _ in 0..n_epochs {
        let t = honest_transition(p, h, epochs.last().expect("non-empty"), ri, rv, rng, true);
        ri = t.r_index;
        rv = t.r_value;
        sigmas.push(t.sigma.clone());
        epochs.push(t.next.clone());
    }
    (epochs, sigmas)
}

/// Fold `n_epochs` honest transitions for a **group-sharded**
/// deployment: `plan.groups()` independent chains, each over its own
/// contiguous slice of shards, stitched back into full-width epoch
/// tuples.
///
/// This is what an honest server publishes once its chain scalars are
/// derived per group rather than globally — each group runs its own
/// rolling Fiat–Shamir accumulator, so shard `j`'s equations are
/// weighted by its group's `r`, not by a directory-wide one.
///
/// Returns `(epochs, sigmas)` in the same full-width shape
/// [`honest_chain`] returns, so the classic auditor and the grouped
/// IVC prover can be run against identical data.
pub fn honest_grouped_chain(
    plan: super::grouped::GroupPlan,
    num_vars: usize,
    h: ArkG1Affine,
    n_epochs: usize,
    rng: &mut ChaCha20Rng,
) -> (Vec<Vec<ShardCommitments>>, Vec<Vec<SigmaWitness>>) {
    let gp = FsParams {
        num_vars,
        n_shards: plan.shards_per_group(),
    };
    // Each group is an independent chain. Distinct RNG streams so the
    // groups are not accidentally identical, which would hide an
    // indexing bug behind coincidence.
    let per_group: Vec<(Vec<Vec<ShardCommitments>>, Vec<Vec<SigmaWitness>>)> = (0..plan.groups())
        .map(|g| {
            let mut r = rng_from(rng, g as u64);
            honest_chain(gp, h, n_epochs, &mut r)
        })
        .collect();

    let epochs = (0..=n_epochs)
        .map(|e| {
            per_group
                .iter()
                .flat_map(|(ep, _)| ep[e].iter().cloned())
                .collect()
        })
        .collect();
    let sigmas = (0..n_epochs)
        .map(|e| {
            per_group
                .iter()
                .flat_map(|(_, sg)| sg[e].iter().cloned())
                .collect()
        })
        .collect();
    (epochs, sigmas)
}

/// A fresh deterministic stream derived from `parent` and `tag`, so
/// sibling groups get independent but reproducible fixtures.
fn rng_from(parent: &mut ChaCha20Rng, tag: u64) -> ChaCha20Rng {
    let seed = ArkFr::rand(parent).into_bigint().0[0] ^ tag.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    ChaCha20Rng::seed_from_u64(seed)
}

// ---------- lifting into the published types ---------------------------
//
// The fixtures above deal in bare group elements, which is all the
// folding circuit consumes. The *classic* per-epoch auditor consumes
// full `ShardedEpochCommitment`s, so to compare the two on identical
// data the bench needs to lift one into the other.

use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKCommitment;
use ark_bn254::Bn254;

use crate::aegon::sharded::ShardedEpochCommitment;
use crate::aegon::sigma::BlindingEqProof;
use crate::aegon::types::EpochCommitment;

/// The PCS the IVC audit path is defined against.
pub type Pcs = akd_core::aegon_crypto::pcs::kzhk::KZHK<Bn254>;

fn commit(p: ArkG1Affine, num_vars: usize) -> KZHKCommitment<Bn254> {
    KZHKCommitment::new(p, num_vars)
}

/// Lift synthetic group elements into a published
/// [`ShardedEpochCommitment`], so the classic
/// [`verify_sharded_invariance`](crate::aegon::verify_sharded_invariance)
/// can be benchmarked against exactly the data the IVC prover folds.
///
/// The Merkle root is computed by the real
/// [`ShardedEpochCommitment::with_per_shard`], so the structural
/// pre-check in the classic auditor does real work rather than being
/// short-circuited.
pub fn as_sharded_epoch_commitment(
    epoch: u64,
    shards: &[ShardCommitments],
    sigmas: Option<&[SigmaWitness]>,
    num_vars: usize,
) -> ShardedEpochCommitment<Bn254, Pcs> {
    let per_shard: Vec<EpochCommitment<Bn254, Pcs>> = shards
        .iter()
        .enumerate()
        .map(|(j, s)| EpochCommitment {
            epoch,
            index_commitment: commit(s.index, num_vars),
            value_commitment: commit(s.value, num_vars),
            rand_index_commitment: commit(s.rand_index, num_vars),
            rand_value_commitment: commit(s.rand_value, num_vars),
            audit_value_blinding_proof: sigmas.map(|w| BlindingEqProof {
                r_commit: commit(w[j].r_commit, num_vars),
                response: w[j].response,
            }),
            _e: std::marker::PhantomData,
        })
        .collect();
    ShardedEpochCommitment::with_per_shard(epoch, per_shard)
}
