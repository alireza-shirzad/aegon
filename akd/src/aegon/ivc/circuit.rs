// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! The Nova step circuit: one Aegon epoch transition per folding step.
//!
//! ## The statement
//!
//! The folded state has arity 4:
//!
//! ```text
//!   z = [ epoch, r_index, r_value, state_digest ]
//! ```
//!
//! `state_digest` is [`poseidon_state_digest`] over the epoch's
//! per-shard commitment tuple. It is the hinge of the whole design:
//! it lets a verifier holding the *concrete* commitments for some
//! epoch decide whether the recursive proof is talking about that
//! epoch, without the circuit ever seeing a Merkle root.
//!
//! One step proves the transition `n -> n+1`:
//!
//! 1. `Poseidon(prev.per_shard) == z_in.state_digest`. Without this
//!    the prover could feed in a `prev` unrelated to the history it
//!    has been folding, and the chain would prove nothing.
//! 2. Re-derive `r_index'` and `r_value'` by Fiat–Shamir over *all*
//!    shards' new data commitments, chained from `z_in.r_*`. This is
//!    the step that makes the proof meaningful: a server that could
//!    choose `r` after seeing the commitments could satisfy the chain
//!    equation with forged polynomials.
//! 3. Per shard: the index-chain group equality, and the value-chain
//!    residue plus its Schnorr blinding-equality proof (paper §7).
//! 4. `z_out = [ epoch+1, r_index', r_value', Poseidon(next.per_shard) ]`.
//!
//! ## Cost
//!
//! Per shard: three 128-bit variable-base scalar multiplications
//! (`r_index'·Δ`, `r_value'·Δ`, `e·D`) and one full-width one
//! (`s·h`), plus a handful of complete additions and one Poseidon
//! permutation for `e`. Everything is native field arithmetic — see
//! [`super::bridge`] for why the circuit field is BN254's base field.
//!
//! ## Edge cases that are *not* edge cases here
//!
//! * **A shard with no updates.** Then `Δ = O`, the identity. This is
//!   the common case, not a corner case, and Nova's `scalar_mul`
//!   handles an identity base correctly: the incomplete-addition
//!   witness path guards its divisions, and the final result is
//!   conditionally selected back to the identity.
//! * **The genesis epoch.** Epoch 0 commits zero polynomials, so all
//!   four commitments *are* the identity. Points are therefore
//!   allocated as [`AllocatedPoint`] (which carries an explicit
//!   `is_infinity` flag) rather than the cheaper
//!   `AllocatedPointNonInfinity`.
//! * **Point equality.** Expressed as "the difference is the
//!   identity" rather than coordinate-wise equality, because a point
//!   at infinity does not have canonical `(x, y)`. Nova's complete
//!   `AllocatedPoint::add` returns the identity for `P + (−P)` and
//!   handles identity operands, so this is correct even at genesis.

use ark_bn254::{Fr as ArkFr, G1Affine as ArkG1Affine};
use ark_ff::{BigInteger, PrimeField};
use nova_snark::{
    frontend::{num::AllocatedNum, AllocatedBit, Assignment, ConstraintSystem, SynthesisError},
    gadgets::{
        ecc::AllocatedPoint,
        utils::{alloc_constant, alloc_zero, le_bits_to_num},
    },
    provider::poseidon::{PoseidonConstantsCircuit, PoseidonROCircuit},
    traits::{circuit::StepCircuit, ROCircuitTrait},
};

use super::bridge::{ark_g1_to_coords, CircuitField, PointCoords, PointEngine};
use super::fs_poseidon::{domain, FsParams, ShardCommitments, CHALLENGE_BITS, DIGEST_BITS};

/// Bit width of a Schnorr response `s`. Full `Fr` width — unlike the
/// Fiat–Shamir challenges, `s` is a prover-supplied field element and
/// is not truncated.
pub const SCALAR_BITS: usize = 254;

/// Arity of the folded state.
pub const ARITY: usize = 4;

const Z_EPOCH: usize = 0;
const Z_R_INDEX: usize = 1;
const Z_R_VALUE: usize = 2;
const Z_DIGEST: usize = 3;

/// One shard's Schnorr blinding-equality proof, as consumed by the
/// circuit. Mirrors [`BlindingEqProof`](crate::aegon::BlindingEqProof)
/// with the commitment unwrapped to a bare group element.
#[derive(Clone, Copy, Debug)]
pub struct SigmaWitness {
    /// The prover's Schnorr commitment `R = k·h`.
    pub r_commit: ArkG1Affine,
    /// The response `s = k + e·c`.
    pub response: ArkFr,
}

impl Default for SigmaWitness {
    fn default() -> Self {
        Self {
            r_commit: ArkG1Affine::identity(),
            response: ArkFr::from(0u64),
        }
    }
}

/// Everything one folding step needs beyond the public state.
///
/// All of it is already public in
/// [`ShardedEpochCommitment`](crate::aegon::ShardedEpochCommitment) —
/// including the Schnorr proofs — so producing this witness requires
/// no server secrets. It is a *circuit* witness, not a secret.
#[derive(Clone, Debug)]
pub struct AuditStepWitness {
    /// Epoch `n` per-shard commitments, in shard-id order.
    pub prev: Vec<ShardCommitments>,
    /// Epoch `n+1` per-shard commitments, in shard-id order.
    pub next: Vec<ShardCommitments>,
    /// One value-chain Schnorr proof per shard.
    pub sigma: Vec<SigmaWitness>,
}

impl AuditStepWitness {
    /// A shape-only placeholder. Used when synthesizing the circuit
    /// for `PublicParams::setup`, where no real witness exists yet.
    /// The R1CS shape does not depend on the values, only on
    /// `n_shards`, so this produces exactly the circuit that the real
    /// witness will later satisfy.
    pub fn placeholder(n_shards: usize) -> Self {
        let ident = ShardCommitments {
            index: ArkG1Affine::identity(),
            value: ArkG1Affine::identity(),
            rand_index: ArkG1Affine::identity(),
            rand_value: ArkG1Affine::identity(),
        };
        Self {
            prev: vec![ident; n_shards],
            next: vec![ident; n_shards],
            sigma: vec![SigmaWitness::default(); n_shards],
        }
    }
}

/// The Nova step circuit for one epoch transition.
#[derive(Clone)]
pub struct AuditStepCircuit {
    params: FsParams,
    ro_consts: PoseidonConstantsCircuit<CircuitField>,
    /// The SRS hiding generator `h`, baked in as a circuit constant.
    ///
    /// Baking it in rather than passing it as witness is deliberate:
    /// `h` is fixed for a deployment and part of the verifier key, so
    /// making it a constant folds it into the `PublicParams` digest.
    /// The verifier's parameters therefore pin which SRS the proof is
    /// about.
    h: PointCoords,
    witness: Option<AuditStepWitness>,
}

impl AuditStepCircuit {
    /// Build a circuit instance.
    ///
    /// Pass `witness = None` for shape synthesis (`PublicParams::setup`)
    /// and `Some(..)` when proving a step.
    pub fn new(
        params: FsParams,
        ro_consts: PoseidonConstantsCircuit<CircuitField>,
        h: ArkG1Affine,
        witness: Option<AuditStepWitness>,
    ) -> Self {
        Self {
            params,
            ro_consts,
            h: ark_g1_to_coords(&h),
            witness,
        }
    }

    /// Same circuit, different step witness. The shape is unchanged,
    /// so the `PublicParams` built from any instance stay valid.
    pub fn with_witness(&self, witness: AuditStepWitness) -> Self {
        Self {
            witness: Some(witness),
            ..self.clone()
        }
    }

    /// The deployment shape this circuit was built for.
    pub fn params(&self) -> FsParams {
        self.params
    }
}

/// Allocate a BN254 G1 point as `(x, y, is_infinity)`.
fn alloc_point<CS: ConstraintSystem<CircuitField>>(
    cs: CS,
    coords: PointCoords,
) -> Result<AllocatedPoint<PointEngine>, SynthesisError> {
    AllocatedPoint::<PointEngine>::alloc(cs, Some(coords.as_alloc_coords()))
}

/// Little-endian bit decomposition of an `Fr` element, truncated or
/// zero-extended to `width` bits.
fn scalar_bits_le(f: ArkFr, width: usize) -> Vec<bool> {
    let mut bits = f.into_bigint().to_bits_le();
    bits.resize(width, false);
    bits.truncate(width);
    bits
}

/// Allocate `width` witness bits for a scalar.
///
/// No range check against the field modulus is performed, and none is
/// needed: these bits are only ever consumed by `scalar_mul`, which
/// interprets them by double-and-add. A `width`-bit string therefore
/// acts as the scalar `value mod r`, so an out-of-range assignment is
/// simply a different in-range scalar. For the Schnorr response that
/// is exactly the statement we want — `∃ s, R : s·h = R + e·D`.
fn alloc_scalar_bits<CS: ConstraintSystem<CircuitField>>(
    mut cs: CS,
    value: ArkFr,
    width: usize,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
    let bits = scalar_bits_le(value, width);
    bits.iter()
        .enumerate()
        .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("bit {i}")), Some(*b)))
        .collect()
}

/// Enforce `a == b` for two curve points.
///
/// Checks that `a − b` is the identity rather than comparing
/// coordinates, because the identity has no canonical `(x, y)`: two
/// points both at infinity may carry different coordinate
/// assignments. Nova's `add` is the complete addition law, so
/// `a + (−b)` correctly yields the identity when `a == b`, including
/// when both are already the identity.
fn enforce_points_equal<CS: ConstraintSystem<CircuitField>>(
    mut cs: CS,
    a: &AllocatedPoint<PointEngine>,
    b: &AllocatedPoint<PointEngine>,
) -> Result<(), SynthesisError> {
    let neg_b = b.negate(cs.namespace(|| "negate rhs"))?;
    let diff = a.add(cs.namespace(|| "difference"), &neg_b)?;
    cs.enforce(
        || "difference is the identity",
        |lc| lc + diff.is_infinity.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
    );
    Ok(())
}

/// Absorb a point into the RO as `(x, y, is_infinity)` — the same
/// three elements, in the same order, as the native
/// [`super::fs_poseidon`] side.
fn absorb_point(ro: &mut PoseidonROCircuit<CircuitField>, p: &AllocatedPoint<PointEngine>) {
    ro.absorb(&p.x);
    ro.absorb(&p.y);
    ro.absorb(&p.is_infinity);
}

impl StepCircuit<CircuitField> for AuditStepCircuit {
    fn arity(&self) -> usize {
        ARITY
    }

    fn synthesize<CS: ConstraintSystem<CircuitField>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<CircuitField>],
    ) -> Result<Vec<AllocatedNum<CircuitField>>, SynthesisError> {
        let n = self.params.n_shards;
        let w = self
            .witness
            .clone()
            .unwrap_or_else(|| AuditStepWitness::placeholder(n));
        if w.prev.len() != n || w.next.len() != n || w.sigma.len() != n {
            // A mismatch would silently change the R1CS shape, so
            // refuse rather than synthesize a circuit that cannot be
            // folded against the published parameters.
            return Err(SynthesisError::AssignmentMissing);
        }

        // Transcript prefix constants, shared by every derivation.
        let nv_c = alloc_constant(
            cs.namespace(|| "const num_vars"),
            &CircuitField::from(self.params.num_vars as u64),
        )?;
        let ns_c = alloc_constant(
            cs.namespace(|| "const n_shards"),
            &CircuitField::from(n as u64),
        )?;

        // ---- allocate the epoch's commitments ----------------------
        let alloc_shards =
            |cs: &mut CS,
             tag: &str,
             shards: &[ShardCommitments]|
             -> Result<Vec<[AllocatedPoint<PointEngine>; 4]>, SynthesisError> {
                shards
                    .iter()
                    .enumerate()
                    .map(|(j, s)| {
                        let pts = s.as_array();
                        let names = ["index", "value", "rand_index", "rand_value"];
                        let mut out = Vec::with_capacity(4);
                        for (p, name) in pts.iter().zip(names) {
                            out.push(alloc_point(
                                cs.namespace(|| format!("{tag} shard {j} {name}")),
                                ark_g1_to_coords(p),
                            )?);
                        }
                        Ok([
                            out[0].clone(),
                            out[1].clone(),
                            out[2].clone(),
                            out[3].clone(),
                        ])
                    })
                    .collect()
            };
        let prev = alloc_shards(cs, "prev", &w.prev)?;
        let next = alloc_shards(cs, "next", &w.next)?;

        // ---- (1) prev must match the folded state digest -----------
        let prev_digest = {
            let mut ro = PoseidonROCircuit::<CircuitField>::new(self.ro_consts.clone());
            let d = alloc_constant(
                cs.namespace(|| "const dom state digest (prev)"),
                &CircuitField::from(domain::STATE_DIGEST),
            )?;
            ro.absorb(&d);
            ro.absorb(&nv_c);
            ro.absorb(&ns_c);
            for shard in &prev {
                for p in shard.iter() {
                    absorb_point(&mut ro, p);
                }
            }
            let bits = ro.squeeze(cs.namespace(|| "squeeze prev digest"), DIGEST_BITS, false)?;
            le_bits_to_num(cs.namespace(|| "prev digest num"), &bits)?
        };
        cs.enforce(
            || "prev commitments match folded state digest",
            |lc| lc + prev_digest.get_variable() - z[Z_DIGEST].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );

        // ---- (2) re-derive the chain scalars -----------------------
        // Absorbs every shard's NEW data commitment, in shard-id
        // order, chained from the previous epoch's scalar. This is
        // what pins `r` to the commitments instead of letting the
        // server choose it.
        let derive_chain = |cs: &mut CS,
                            tag: &str,
                            dom: u64,
                            prev_r: &AllocatedNum<CircuitField>,
                            slot: usize|
         -> Result<
            (Vec<AllocatedBit>, AllocatedNum<CircuitField>),
            SynthesisError,
        > {
            let mut ro = PoseidonROCircuit::<CircuitField>::new(self.ro_consts.clone());
            let d = alloc_constant(
                cs.namespace(|| format!("const dom {tag}")),
                &CircuitField::from(dom),
            )?;
            ro.absorb(&d);
            ro.absorb(&nv_c);
            ro.absorb(&ns_c);
            ro.absorb(prev_r);
            for shard in &next {
                absorb_point(&mut ro, &shard[slot]);
            }
            let bits = ro.squeeze(
                cs.namespace(|| format!("squeeze {tag}")),
                CHALLENGE_BITS,
                false,
            )?;
            let num = le_bits_to_num(cs.namespace(|| format!("{tag} num")), &bits)?;
            Ok((bits, num))
        };
        let (r_index_bits, r_index_num) =
            derive_chain(cs, "r_index", domain::CHAIN_INDEX, &z[Z_R_INDEX], 0)?;
        let (r_value_bits, r_value_num) =
            derive_chain(cs, "r_value", domain::CHAIN_VALUE, &z[Z_R_VALUE], 1)?;

        // The hiding generator, as a constant point.
        let h_point = {
            let zero = alloc_zero(cs.namespace(|| "h is_infinity"));
            AllocatedPoint::<PointEngine>::alloc_constant(
                cs.namespace(|| "srs generator h"),
                (self.h.x, self.h.y),
                zero,
            )?
        };

        // ---- (3) per-shard chain checks ----------------------------
        for j in 0..n {
            let (p_index, p_value, p_rand_index, p_rand_value) =
                (&prev[j][0], &prev[j][1], &prev[j][2], &prev[j][3]);
            let (n_index, n_value, n_rand_index, n_rand_value) =
                (&next[j][0], &next[j][1], &next[j][2], &next[j][3]);

            // --- index chain (exact; index polys are never blinded) ---
            //   rand_index' == rand_index + r_index·(index' − index)
            let neg_p_index = p_index.negate(cs.namespace(|| format!("s{j} neg prev index")))?;
            let delta_index =
                n_index.add(cs.namespace(|| format!("s{j} delta index")), &neg_p_index)?;
            let scaled_index = delta_index.scalar_mul(
                cs.namespace(|| format!("s{j} r_index * delta index")),
                &r_index_bits,
            )?;
            let expected_rand_index = p_rand_index.add(
                cs.namespace(|| format!("s{j} expected rand_index")),
                &scaled_index,
            )?;
            enforce_points_equal(
                cs.namespace(|| format!("s{j} index chain")),
                &expected_rand_index,
                n_rand_index,
            )?;

            // --- value chain (blinded; residue certified by Schnorr) ---
            //   D = rand_value' − (rand_value + r_value·(value' − value))
            let neg_p_value = p_value.negate(cs.namespace(|| format!("s{j} neg prev value")))?;
            let delta_value =
                n_value.add(cs.namespace(|| format!("s{j} delta value")), &neg_p_value)?;
            let scaled_value = delta_value.scalar_mul(
                cs.namespace(|| format!("s{j} r_value * delta value")),
                &r_value_bits,
            )?;
            let chained = p_rand_value.add(
                cs.namespace(|| format!("s{j} chained rand_value")),
                &scaled_value,
            )?;
            let neg_chained = chained.negate(cs.namespace(|| format!("s{j} neg chained")))?;
            let residue =
                n_rand_value.add(cs.namespace(|| format!("s{j} residue")), &neg_chained)?;

            // Schnorr commitment R. This is the one point in the
            // circuit that a malicious prover picks freely — the
            // others are all pinned by the state digests — so it is
            // the one that needs an explicit on-curve check. BN254 G1
            // has cofactor 1, so on-curve implies in-subgroup and no
            // separate subgroup check is required.
            let r_commit = alloc_point(
                cs.namespace(|| format!("s{j} schnorr R")),
                ark_g1_to_coords(&w.sigma[j].r_commit),
            )?;
            r_commit.check_on_curve(cs.namespace(|| format!("s{j} R on curve")))?;

            // e = Poseidon(dom, shape, prev_val, next_val, prev_rand, next_rand, r_value, R)
            let e_bits = {
                let mut ro = PoseidonROCircuit::<CircuitField>::new(self.ro_consts.clone());
                let d = alloc_constant(
                    cs.namespace(|| format!("s{j} const dom sigma")),
                    &CircuitField::from(domain::SIGMA_CHALLENGE),
                )?;
                // Per-shard transcript: binds the polynomial width
                // but not the shard count — `r_value` already binds
                // every shard. Matches `poseidon_sigma_challenge`.
                ro.absorb(&d);
                ro.absorb(&nv_c);
                for p in [p_value, n_value, p_rand_value, n_rand_value] {
                    absorb_point(&mut ro, p);
                }
                ro.absorb(&r_value_num);
                absorb_point(&mut ro, &r_commit);
                ro.squeeze(
                    cs.namespace(|| format!("s{j} squeeze e")),
                    CHALLENGE_BITS,
                    false,
                )?
            };

            // s·h == R + e·D
            let s_bits = alloc_scalar_bits(
                cs.namespace(|| format!("s{j} schnorr s bits")),
                w.sigma[j].response,
                SCALAR_BITS,
            )?;
            let lhs = h_point.scalar_mul(cs.namespace(|| format!("s{j} s * h")), &s_bits)?;
            let e_d = residue.scalar_mul(cs.namespace(|| format!("s{j} e * residue")), &e_bits)?;
            let rhs = r_commit.add(cs.namespace(|| format!("s{j} R + e*D")), &e_d)?;
            enforce_points_equal(cs.namespace(|| format!("s{j} schnorr eq")), &lhs, &rhs)?;
        }

        // ---- (4) output state --------------------------------------
        let next_digest = {
            let mut ro = PoseidonROCircuit::<CircuitField>::new(self.ro_consts.clone());
            let d = alloc_constant(
                cs.namespace(|| "const dom state digest (next)"),
                &CircuitField::from(domain::STATE_DIGEST),
            )?;
            ro.absorb(&d);
            ro.absorb(&nv_c);
            ro.absorb(&ns_c);
            for shard in &next {
                for p in shard.iter() {
                    absorb_point(&mut ro, p);
                }
            }
            let bits = ro.squeeze(cs.namespace(|| "squeeze next digest"), DIGEST_BITS, false)?;
            le_bits_to_num(cs.namespace(|| "next digest num"), &bits)?
        };

        let next_epoch = AllocatedNum::alloc(cs.namespace(|| "next epoch"), || {
            Ok(*z[Z_EPOCH].get_value().get()? + CircuitField::from(1u64))
        })?;
        cs.enforce(
            || "epoch increments by one",
            |lc| lc + next_epoch.get_variable() - z[Z_EPOCH].get_variable() - CS::one(),
            |lc| lc + CS::one(),
            |lc| lc,
        );

        Ok(vec![next_epoch, r_index_num, r_value_num, next_digest])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aegon::ivc::bridge::ark_fr_to_circuit;
    use crate::aegon::ivc::fs_poseidon::{poseidon_state_digest, ro_constants};
    use crate::aegon::ivc::synthetic::{
        genesis, honest_transition, params, rand_point, random_epoch, rng, test_h,
    };
    use ark_ec::AffineRepr;
    use ark_std::UniformRand;
    use nova_snark::frontend::test_cs::TestConstraintSystem;
    use rand_chacha::ChaCha20Rng;

    type TestCS = TestConstraintSystem<CircuitField>;

    /// Synthesize one step, returning the constraint system and the
    /// circuit's output state.
    fn synth(
        p: FsParams,
        h: ArkG1Affine,
        w: AuditStepWitness,
        z_in: [CircuitField; ARITY],
    ) -> (TestCS, Vec<CircuitField>) {
        let mut cs = TestCS::new();
        let z: Vec<AllocatedNum<CircuitField>> = z_in
            .iter()
            .enumerate()
            .map(|(i, v)| AllocatedNum::alloc(cs.namespace(|| format!("z{i}")), || Ok(*v)).unwrap())
            .collect();
        let circuit = AuditStepCircuit::new(p, ro_constants(), h, Some(w));
        let out = circuit.synthesize(&mut cs, &z).expect("synthesis");
        let vals = out.iter().map(|n| n.get_value().unwrap()).collect();
        (cs, vals)
    }

    fn z_in_for(
        p: FsParams,
        prev: &[ShardCommitments],
        ri: ArkFr,
        rv: ArkFr,
        epoch: u64,
    ) -> [CircuitField; ARITY] {
        [
            CircuitField::from(epoch),
            ark_fr_to_circuit(ri),
            ark_fr_to_circuit(rv),
            poseidon_state_digest(&ro_constants(), p, prev),
        ]
    }

    /// The headline test: an honest transition satisfies every
    /// constraint, and the circuit's output state matches what the
    /// native Poseidon helpers compute independently. This is the
    /// native-vs-circuit equivalence gate — if the two Poseidon
    /// implementations ever diverged, this is what would catch it.
    #[test]
    fn honest_transition_is_satisfied_and_matches_native() {
        let mut r = rng(0xA1);
        let p = params(2);
        let h = test_h();
        let prev = genesis(p.n_shards);
        let zero = ArkFr::from(0u64);
        let t = honest_transition(p, h, &prev, zero, zero, &mut r, true);

        let w = AuditStepWitness {
            prev: t.prev.clone(),
            next: t.next.clone(),
            sigma: t.sigma.clone(),
        };
        let (cs, out) = synth(p, h, w, z_in_for(p, &t.prev, zero, zero, 0));

        assert!(
            cs.is_satisfied(),
            "unsatisfied: {:?}",
            cs.which_is_unsatisfied()
        );
        assert_eq!(out[Z_EPOCH], CircuitField::from(1u64));
        assert_eq!(out[Z_R_INDEX], ark_fr_to_circuit(t.r_index));
        assert_eq!(out[Z_R_VALUE], ark_fr_to_circuit(t.r_value));
        assert_eq!(
            out[Z_DIGEST],
            poseidon_state_digest(&ro_constants(), p, &t.next)
        );
        println!(
            "n_shards={} constraints={}",
            p.n_shards,
            cs.num_constraints()
        );
    }

    /// A shard with no updates makes every delta the identity. This
    /// is the *common* case in production (most shards see no
    /// activity in a given epoch), so it must be satisfiable, not
    /// merely non-crashing.
    #[test]
    fn epoch_with_no_updates_is_satisfied() {
        let mut r = rng(0xA2);
        let p = params(2);
        let h = test_h();
        let prev = random_epoch(p.n_shards, &mut r);
        let (ri, rv) = (ArkFr::from(5u64), ArkFr::from(6u64));
        let t = honest_transition(p, h, &prev, ri, rv, &mut r, false);
        let w = AuditStepWitness {
            prev: t.prev.clone(),
            next: t.next.clone(),
            sigma: t.sigma.clone(),
        };
        let (cs, _) = synth(p, h, w, z_in_for(p, &t.prev, ri, rv, 7));
        assert!(
            cs.is_satisfied(),
            "unsatisfied: {:?}",
            cs.which_is_unsatisfied()
        );
    }

    /// Genesis: epoch 0 commits zero polynomials, so every point is
    /// the identity. Exercises the infinity paths end to end.
    #[test]
    fn genesis_transition_is_satisfied() {
        let mut r = rng(0xA3);
        let p = params(1);
        let h = test_h();
        let prev = genesis(p.n_shards);
        let zero = ArkFr::from(0u64);
        let t = honest_transition(p, h, &prev, zero, zero, &mut r, false);
        let w = AuditStepWitness {
            prev: t.prev.clone(),
            next: t.next.clone(),
            sigma: t.sigma.clone(),
        };
        let (cs, _) = synth(p, h, w, z_in_for(p, &t.prev, zero, zero, 0));
        assert!(
            cs.is_satisfied(),
            "unsatisfied: {:?}",
            cs.which_is_unsatisfied()
        );
    }

    // ---- negative tests -------------------------------------------
    //
    // Each of these is an attack the audit is supposed to catch. The
    // circuit must become UNSATISFIABLE, which is what stops a
    // malicious server producing a folding step at all.

    fn tampered(mutate: impl FnOnce(&mut AuditStepWitness, &mut ChaCha20Rng)) -> bool {
        let mut r = rng(0xB0);
        let p = params(2);
        let h = test_h();
        let prev = random_epoch(p.n_shards, &mut r);
        let (ri, rv) = (ArkFr::from(11u64), ArkFr::from(12u64));
        let t = honest_transition(p, h, &prev, ri, rv, &mut r, true);
        let mut w = AuditStepWitness {
            prev: t.prev.clone(),
            next: t.next.clone(),
            sigma: t.sigma.clone(),
        };
        mutate(&mut w, &mut r);
        let (cs, _) = synth(p, h, w, z_in_for(p, &t.prev, ri, rv, 3));
        cs.is_satisfied()
    }

    /// Forging a new rand_index commitment breaks the index chain.
    #[test]
    fn rejects_tampered_rand_index() {
        assert!(!tampered(|w, r| w.next[0].rand_index = rand_point(r)));
    }

    /// Forging a new rand_value breaks the Schnorr equation: the
    /// residue is no longer the `c*h` the proof was made for.
    #[test]
    fn rejects_tampered_rand_value() {
        assert!(!tampered(|w, r| w.next[1].rand_value = rand_point(r)));
    }

    /// Changing a data commitment shifts the Fiat-Shamir scalar, so
    /// both chains fail. This is the attack the in-circuit FS
    /// re-derivation exists to prevent.
    #[test]
    fn rejects_tampered_data_commitment() {
        assert!(!tampered(|w, r| w.next[0].index = rand_point(r)));
    }

    /// A `prev` that does not match the folded state digest must be
    /// rejected -- otherwise the prover could splice in an unrelated
    /// history.
    #[test]
    fn rejects_prev_not_matching_state_digest() {
        assert!(!tampered(|w, r| w.prev[0].value = rand_point(r)));
    }

    /// A forged Schnorr response cannot satisfy `s*h = R + e*D`.
    #[test]
    fn rejects_forged_schnorr_response() {
        assert!(!tampered(|w, r| w.sigma[0].response = ArkFr::rand(r)));
    }

    /// Nor can a substituted Schnorr commitment, since `e` binds `R`.
    #[test]
    fn rejects_forged_schnorr_commitment() {
        assert!(!tampered(|w, r| w.sigma[1].r_commit = rand_point(r)));
    }

    /// An off-curve Schnorr commitment must be rejected by the
    /// explicit on-curve check. `R` is the only freely-chosen point
    /// in the circuit, so this is the one place the check matters.
    #[test]
    fn rejects_off_curve_schnorr_commitment() {
        assert!(!tampered(|w, _r| {
            let (x, y) = w.sigma[0].r_commit.xy().expect("non-identity");
            w.sigma[0].r_commit = ArkG1Affine::new_unchecked(x, y + ark_bn254::Fq::from(1u64));
        }));
    }

    /// Shard order is part of the statement: swapping two shards'
    /// commitments must not verify.
    #[test]
    fn rejects_swapped_shards() {
        assert!(!tampered(|w, _r| w.next.swap(0, 1)));
    }

    /// Constraint count should scale linearly in the shard count and
    /// stay in the range the design targets. Guards against a
    /// refactor accidentally introducing non-native arithmetic, which
    /// would blow this up by orders of magnitude.
    #[test]
    fn constraint_count_scales_linearly() {
        let mut r = rng(0xC0);
        let h = test_h();
        let zero = ArkFr::from(0u64);
        let mut counts = Vec::new();
        for n in [1usize, 2, 4] {
            let p = params(n);
            let prev = genesis(n);
            let t = honest_transition(p, h, &prev, zero, zero, &mut r, true);
            let w = AuditStepWitness {
                prev: t.prev.clone(),
                next: t.next.clone(),
                sigma: t.sigma.clone(),
            };
            let (cs, _) = synth(p, h, w, z_in_for(p, &t.prev, zero, zero, 0));
            assert!(cs.is_satisfied());
            counts.push(cs.num_constraints());
            println!("n_shards={n} constraints={}", cs.num_constraints());
        }
        let per_shard = (counts[2] - counts[1]) / 2;
        println!("marginal constraints/shard = {per_shard}");
        assert!(
            (5_000..40_000).contains(&per_shard),
            "per-shard cost {per_shard} outside the expected native-arithmetic range; \
             a non-native fallback may have crept in"
        );
    }
}
