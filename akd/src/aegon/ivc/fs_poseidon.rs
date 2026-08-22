//! Poseidon Fiat–Shamir for the audit path — the *native* side.
//!
//! ## Why the audit FS has to change at all
//!
//! The chain scalars `(r_index, r_value)` and the per-shard Schnorr
//! challenge `e` are what stop a malicious server from choosing its
//! randomness *after* seeing the commitments. A folding circuit that
//! merely accepted them as witness would prove nothing: the server
//! could pick any `r` that makes the chain equation hold. So the
//! circuit has to recompute them, which means whatever hash produces
//! them must be cheap in R1CS.
//!
//! SHA256 is not. At 128 shards the two chain derivations alone
//! absorb ~8 KB, and with the 128 Schnorr challenges that is on the
//! order of 4M constraints per epoch — several times the cost of all
//! the elliptic-curve work the audit actually consists of. (This is
//! the same wall the paper cites for Hekaton in §2.1.)
//!
//! So on the [`AuditFs::Poseidon`](crate::aegon::config::AuditFs)
//! path these three derivations use Poseidon over the circuit field
//! instead, at roughly 200k constraints total.
//!
//! ## What did *not* change
//!
//! The Merkle leaf/root/path hashing
//! ([`merkle_leaf`](crate::aegon::sharded::merkle_leaf) and friends)
//! stays on SHA256, because it never has to enter the circuit: the
//! verifier already downloads the epoch's `per_shard` tuple, so it
//! recomputes the SHA256 root natively exactly as it does today. Nor
//! do lookup, consistency or history proofs change in any way.
//!
//! ## Why native and in-circuit agree
//!
//! Both sides use Nova's own Poseidon — [`PoseidonRO`] here and
//! `PoseidonROCircuit` in [`super::circuit`]. Nova declares them a
//! matched pair (`ROTrait::CircuitRO`), they share one
//! `PoseidonConstantsCircuit` instance, and both `new()` into
//! `ROMode::Wide`. There is no second set of round constants to keep
//! in sync, and the circuit module pins the agreement with an
//! explicit native-vs-circuit comparison test.
//!
//! ## Absorb discipline
//!
//! Every derivation absorbs, in order:
//!
//! 1. a domain tag (a distinct small constant per derivation),
//! 2. the binding parameters `nv` and `n_shards`,
//! 3. the derivation's own inputs.
//!
//! Absorbing `nv`/`n_shards` matters because a group element is fed
//! in as bare coordinates: unlike the SHA256 path, which hashed
//! `KZHKCommitment`'s `CanonicalSerialize` output (and so covered the
//! `nv` tag incidentally), Poseidon here sees only `(x, y,
//! is_infinity)`. Binding the shape explicitly keeps transcripts
//! from two differently-configured directories from ever colliding.

use std::sync::OnceLock;

use ark_bn254::{Fr as ArkFr, G1Affine as ArkG1Affine};
use nova_snark::{
    constants::{NUM_CHALLENGE_BITS, NUM_HASH_BITS},
    provider::poseidon::{PoseidonConstantsCircuit, PoseidonRO},
    traits::ROTrait,
};

use super::bridge::{
    ark_fr_to_circuit, ark_g1_to_coords, circuit_to_ark_fr, CircuitField, PointCoords,
};

/// Domain tags. Distinct small field constants rather than hashed
/// byte strings, so the circuit can allocate them as constants for
/// free. They mirror the byte tags the SHA256 path uses:
/// `aegon.sharded.fs.r_index`, `aegon.sharded.fs.r_value`,
/// `aegon.audit.blinding_eq.v1`.
pub mod domain {
    /// `aegon.sharded.fs.r_index`
    pub const CHAIN_INDEX: u64 = 1;
    /// `aegon.sharded.fs.r_value`
    pub const CHAIN_VALUE: u64 = 2;
    /// `aegon.audit.blinding_eq.v1`
    pub const SIGMA_CHALLENGE: u64 = 3;
    /// Running digest of an epoch's per-shard commitment tuple.
    /// Has no SHA256 counterpart — it exists only to let the IVC
    /// state commit to the epoch it is describing.
    pub const STATE_DIGEST: u64 = 4;
}

/// Bit width of a Fiat–Shamir challenge. Matches Nova's own
/// `NUM_CHALLENGE_BITS`; 128 bits gives 128-bit soundness for the
/// Schwartz–Zippel and Schnorr arguments while halving the cost of
/// every in-circuit scalar multiplication relative to a full-width
/// scalar.
pub const CHALLENGE_BITS: usize = NUM_CHALLENGE_BITS;

/// Bit width of the IVC state digest. Matches Nova's `NUM_HASH_BITS`
/// (the same width Nova uses for its own folded state hashes), for
/// 125-bit collision resistance.
pub const DIGEST_BITS: usize = NUM_HASH_BITS;

/// Shape parameters bound into every transcript. Two directories with
/// different polynomial sizes or shard counts must never produce
/// colliding challenges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FsParams {
    /// Number of variables of the committed multilinear polynomials,
    /// i.e. `KZHKCommitment::get_num_vars`.
    pub num_vars: usize,
    /// Number of shards in the directory.
    pub n_shards: usize,
}

/// The Poseidon constants shared by the native and circuit sides.
///
/// Building these generates full round-constant and MDS tables for
/// both sponge widths, which costs on the order of a hundred
/// milliseconds — far more than the derivation it parameterises. They
/// are deterministic, so [`ro_constants_cached`] builds them once per
/// process and every hot path should use that. This owned-value
/// constructor exists for the circuit, whose RO takes the constants
/// by value.
pub fn ro_constants() -> PoseidonConstantsCircuit<CircuitField> {
    ro_constants_cached().clone()
}

/// Process-wide cached Poseidon constants.
///
/// Every native derivation goes through here. Without the cache the
/// server would regenerate the tables on every chain-scalar and every
/// per-shard Schnorr challenge, which dominates the actual audit
/// arithmetic by two orders of magnitude.
pub fn ro_constants_cached() -> &'static PoseidonConstantsCircuit<CircuitField> {
    static CONSTANTS: OnceLock<PoseidonConstantsCircuit<CircuitField>> = OnceLock::new();
    CONSTANTS.get_or_init(PoseidonConstantsCircuit::<CircuitField>::default)
}

fn new_ro(constants: &PoseidonConstantsCircuit<CircuitField>) -> PoseidonRO<CircuitField> {
    PoseidonRO::new(constants.clone())
}

/// Absorb the fixed transcript prefix: domain tag then shape.
fn absorb_prefix(ro: &mut PoseidonRO<CircuitField>, domain: u64, params: FsParams) {
    ro.absorb(CircuitField::from(domain));
    ro.absorb(CircuitField::from(params.num_vars as u64));
    ro.absorb(CircuitField::from(params.n_shards as u64));
}

/// Prefix for per-shard transcripts, which bind the polynomial width
/// but not the shard count.
fn absorb_prefix_one_shard(ro: &mut PoseidonRO<CircuitField>, domain: u64, num_vars: usize) {
    ro.absorb(CircuitField::from(domain));
    ro.absorb(CircuitField::from(num_vars as u64));
}

/// Absorb one BN254 G1 point as `(x, y, is_infinity)` — the same
/// three field elements, in the same order, that Nova's
/// `AllocatedPoint::absorb_in_ro` feeds the circuit RO.
fn absorb_point(ro: &mut PoseidonRO<CircuitField>, p: &PointCoords) {
    ro.absorb(p.x);
    ro.absorb(p.y);
    ro.absorb(p.infinity_field());
}

/// Derive a chain scalar `r' = Poseidon(domain, shape, prev_r, commits…)`.
///
/// The sharded analogue of `fs_chain_scalar` on the SHA256 path:
/// `commits` is every shard's new data commitment, in shard-id order,
/// so the challenge binds the whole epoch rather than one shard.
///
/// Returns a 128-bit value, which embeds into `Fr` without reduction.
pub fn poseidon_chain_scalar(
    constants: &PoseidonConstantsCircuit<CircuitField>,
    domain: u64,
    params: FsParams,
    prev_r: ArkFr,
    commits: &[ArkG1Affine],
) -> ArkFr {
    let mut ro = new_ro(constants);
    absorb_prefix(&mut ro, domain, params);
    ro.absorb(ark_fr_to_circuit(prev_r));
    for c in commits {
        absorb_point(&mut ro, &ark_g1_to_coords(c));
    }
    circuit_to_ark_fr(ro.squeeze(CHALLENGE_BITS, false))
}

/// Derive the Schnorr challenge `e` for one shard's value-chain
/// blinding-equality proof (paper §7).
///
/// Mirrors the transcript of `sigma::fs_challenge` on the SHA256
/// path: the four chain commitments, the chain scalar, and the
/// prover's Schnorr commitment `R`. Binding all six stops the prover
/// choosing any of them after seeing the others.
///
/// Takes `num_vars` rather than a full [`FsParams`] because this
/// transcript is *per shard*: the shard count is neither available at
/// the call site nor meaningful here, since `r_chain` — which is in
/// the transcript — already binds every shard's commitments.
#[allow(clippy::too_many_arguments)]
pub fn poseidon_sigma_challenge(
    constants: &PoseidonConstantsCircuit<CircuitField>,
    num_vars: usize,
    prev_poly: &ArkG1Affine,
    next_poly: &ArkG1Affine,
    prev_rand: &ArkG1Affine,
    next_rand: &ArkG1Affine,
    r_chain: ArkFr,
    r_commit: &ArkG1Affine,
) -> ArkFr {
    let mut ro = new_ro(constants);
    absorb_prefix_one_shard(&mut ro, domain::SIGMA_CHALLENGE, num_vars);
    for p in [prev_poly, next_poly, prev_rand, next_rand] {
        absorb_point(&mut ro, &ark_g1_to_coords(p));
    }
    ro.absorb(ark_fr_to_circuit(r_chain));
    absorb_point(&mut ro, &ark_g1_to_coords(r_commit));
    circuit_to_ark_fr(ro.squeeze(CHALLENGE_BITS, false))
}

/// One shard's four published commitments, in the canonical order
/// they are absorbed into the state digest.
///
/// Deliberately mirrors the field order of
/// [`EpochCommitment`](crate::aegon::EpochCommitment) and of
/// `merkle_leaf`, so the Poseidon digest and the SHA256 Merkle leaf
/// cover exactly the same data.
#[derive(Clone, Copy, Debug)]
pub struct ShardCommitments {
    /// Commitment to the shard's index polynomial.
    pub index: ArkG1Affine,
    /// Commitment to the shard's value polynomial.
    pub value: ArkG1Affine,
    /// Commitment to the index chain's randomised polynomial.
    pub rand_index: ArkG1Affine,
    /// Commitment to the value chain's randomised polynomial.
    pub rand_value: ArkG1Affine,
}

impl ShardCommitments {
    /// The four points in absorb order.
    pub fn as_array(&self) -> [ArkG1Affine; 4] {
        [self.index, self.value, self.rand_index, self.rand_value]
    }
}

/// Poseidon digest of a whole epoch's per-shard commitment tuple.
///
/// This is the IVC state's handle on "which epoch am I describing".
/// The verifier recomputes it over the `per_shard` tuple it already
/// downloaded and compares against the folded output, which is what
/// ties an otherwise free-floating recursive proof to the concrete
/// commitments on the bulletin board.
///
/// Returns a `DIGEST_BITS`-wide value in the circuit field.
pub fn poseidon_state_digest(
    constants: &PoseidonConstantsCircuit<CircuitField>,
    params: FsParams,
    per_shard: &[ShardCommitments],
) -> CircuitField {
    let mut ro = new_ro(constants);
    absorb_prefix(&mut ro, domain::STATE_DIGEST, params);
    for shard in per_shard {
        for p in shard.as_array() {
            absorb_point(&mut ro, &ark_g1_to_coords(&p));
        }
    }
    ro.squeeze(DIGEST_BITS, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::PrimeGroup;
    use ark_std::rand::SeedableRng;
    use ark_std::UniformRand;
    use rand_chacha::ChaCha20Rng;

    fn pt(i: u64) -> ArkG1Affine {
        (ark_bn254::G1Projective::generator() * ArkFr::from(i + 1)).into()
    }

    fn params() -> FsParams {
        FsParams {
            num_vars: 10,
            n_shards: 4,
        }
    }

    #[test]
    fn chain_scalar_is_deterministic() {
        let c = ro_constants();
        let commits: Vec<_> = (0..4).map(pt).collect();
        let a =
            poseidon_chain_scalar(&c, domain::CHAIN_INDEX, params(), ArkFr::from(7u64), &commits);
        let b =
            poseidon_chain_scalar(&c, domain::CHAIN_INDEX, params(), ArkFr::from(7u64), &commits);
        assert_eq!(a, b);
    }

    /// A challenge must move if *any* transcript input moves —
    /// otherwise a malicious server gets freedom it should not have.
    #[test]
    fn chain_scalar_binds_every_input() {
        let c = ro_constants();
        let commits: Vec<_> = (0..4).map(pt).collect();
        let base =
            poseidon_chain_scalar(&c, domain::CHAIN_INDEX, params(), ArkFr::from(7u64), &commits);

        // different domain (index vs value chain)
        assert_ne!(
            base,
            poseidon_chain_scalar(&c, domain::CHAIN_VALUE, params(), ArkFr::from(7u64), &commits)
        );
        // different previous chain scalar
        assert_ne!(
            base,
            poseidon_chain_scalar(&c, domain::CHAIN_INDEX, params(), ArkFr::from(8u64), &commits)
        );
        // different shape
        let other_params = FsParams {
            num_vars: 11,
            n_shards: 4,
        };
        assert_ne!(
            base,
            poseidon_chain_scalar(
                &c,
                domain::CHAIN_INDEX,
                other_params,
                ArkFr::from(7u64),
                &commits
            )
        );
        // a single perturbed commitment
        let mut tweaked = commits.clone();
        tweaked[2] = pt(99);
        assert_ne!(
            base,
            poseidon_chain_scalar(&c, domain::CHAIN_INDEX, params(), ArkFr::from(7u64), &tweaked)
        );
        // reordered commitments (shard-id order is part of the statement)
        let mut swapped = commits.clone();
        swapped.swap(0, 1);
        assert_ne!(
            base,
            poseidon_chain_scalar(&c, domain::CHAIN_INDEX, params(), ArkFr::from(7u64), &swapped)
        );
    }

    /// Challenges are truncated to `CHALLENGE_BITS`, so they must
    /// survive the `Fr` embedding with room to spare. If this ever
    /// failed, `circuit_to_ark_fr` would silently reduce.
    #[test]
    fn chain_scalar_fits_in_challenge_width() {
        use ark_ff::{BigInteger, PrimeField};
        let c = ro_constants();
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        for _ in 0..64 {
            let commits: Vec<ArkG1Affine> = (0..3)
                .map(|_| (ark_bn254::G1Projective::generator() * ArkFr::rand(&mut rng)).into())
                .collect();
            let r = poseidon_chain_scalar(
                &c,
                domain::CHAIN_INDEX,
                params(),
                ArkFr::rand(&mut rng),
                &commits,
            );
            let bits = r.into_bigint().to_bits_le();
            assert!(
                bits.iter().skip(CHALLENGE_BITS).all(|b| !b),
                "challenge exceeded {CHALLENGE_BITS} bits"
            );
        }
    }

    #[test]
    fn sigma_challenge_binds_every_input() {
        let c = ro_constants();
        let (a, b, d, e, f) = (pt(1), pt(2), pt(3), pt(4), pt(5));
        let base = poseidon_sigma_challenge(&c, params().num_vars, &a, &b, &d, &e, ArkFr::from(3u64), &f);
        assert_ne!(
            base,
            poseidon_sigma_challenge(&c, params().num_vars, &pt(9), &b, &d, &e, ArkFr::from(3u64), &f)
        );
        assert_ne!(
            base,
            poseidon_sigma_challenge(&c, params().num_vars, &a, &b, &d, &e, ArkFr::from(4u64), &f)
        );
        assert_ne!(
            base,
            poseidon_sigma_challenge(&c, params().num_vars, &a, &b, &d, &e, ArkFr::from(3u64), &pt(9))
        );
        // argument order must matter: swapping prev/next changes the claim
        assert_ne!(
            base,
            poseidon_sigma_challenge(&c, params().num_vars, &b, &a, &d, &e, ArkFr::from(3u64), &f)
        );
    }

    #[test]
    fn state_digest_binds_every_shard_and_slot() {
        let c = ro_constants();
        let shard = |i: u64| ShardCommitments {
            index: pt(i * 4),
            value: pt(i * 4 + 1),
            rand_index: pt(i * 4 + 2),
            rand_value: pt(i * 4 + 3),
        };
        let base_shards: Vec<_> = (0..4).map(shard).collect();
        let base = poseidon_state_digest(&c, params(), &base_shards);
        assert_eq!(base, poseidon_state_digest(&c, params(), &base_shards));

        // perturb each of the four slots on one shard
        for slot in 0..4 {
            let mut s = base_shards.clone();
            let arr = &mut s[2];
            match slot {
                0 => arr.index = pt(777),
                1 => arr.value = pt(777),
                2 => arr.rand_index = pt(777),
                _ => arr.rand_value = pt(777),
            }
            assert_ne!(base, poseidon_state_digest(&c, params(), &s), "slot {slot}");
        }

        // shard order is part of the statement
        let mut reordered = base_shards.clone();
        reordered.swap(0, 3);
        assert_ne!(base, poseidon_state_digest(&c, params(), &reordered));
    }

    /// The identity commitment (a zero polynomial, i.e. the genesis
    /// epoch) must be absorbed distinguishably rather than as a
    /// silent `(0, 0)`.
    #[test]
    fn identity_commitment_is_distinguishable() {
        let c = ro_constants();
        let ident = ArkG1Affine::identity();
        let d0 =
            poseidon_chain_scalar(&c, domain::CHAIN_INDEX, params(), ArkFr::from(0u64), &[ident]);
        let d1 =
            poseidon_chain_scalar(&c, domain::CHAIN_INDEX, params(), ArkFr::from(0u64), &[pt(0)]);
        assert_ne!(d0, d1);
    }
}
