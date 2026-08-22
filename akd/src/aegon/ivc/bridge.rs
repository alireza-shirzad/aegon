//! Coordinate bridge between the arkworks BN254 the rest of Aegon is
//! built on and the halo2curves BN254 that `nova-snark` speaks.
//!
//! ## Why two BN254s
//!
//! Aegon's PCS, commitments and field arithmetic are arkworks
//! (`ark_bn254`). `nova-snark` is built on `halo2curves`/`ff`. They
//! describe the *same* curve but with disjoint Rust types, so every
//! group element crossing into the folding circuit needs a
//! conversion. The conversion is byte-level and canonical: arkworks
//! and halo2curves both expose a little-endian 32-byte canonical
//! encoding of a field element, so the round trip is exact and
//! carries no modular reduction.
//!
//! ## The curve cycle, and why the circuit field is `Fq`
//!
//! The audit relation Aegon needs to prove is *entirely* BN254 G1
//! arithmetic (see [`super`]). BN254 G1 coordinates live in `Fq` —
//! BN254's **base** field. Nova's [`AllocatedPoint<E>`] gadget
//! operates inside a `ConstraintSystem<E::Base>`, so a circuit whose
//! native field is `Fq` gets BN254 point arithmetic *for free*
//! (native field muls, no emulation).
//!
//! `nova-snark` 0.75 has a single step circuit, over `E1::Scalar`.
//! So we run the cycle "backwards" relative to the usual BN254 Nova
//! setup:
//!
//! ```text
//!   E1 = GrumpkinEngine    E1::Scalar = Fq_bn254   <- circuit field
//!   E2 = Bn256EngineIPA    E2::Base   = Fq_bn254
//! ```
//!
//! Grumpkin is defined over `Fr_bn254` with group order `Fq_bn254`,
//! so `E1::Base == E2::Scalar` and `E2::Base == E1::Scalar`: a valid
//! Nova cycle in either orientation. Running it the conventional way
//! (`E1 = Bn256…`) would put the circuit over `Fr` and force every
//! coordinate operation through non-native field emulation, costing
//! roughly two orders of magnitude more constraints per scalar
//! multiplication.
//!
//! [`AllocatedPoint<E>`]: nova_snark::gadgets::ecc::AllocatedPoint

use ark_bn254::{Fq as ArkFq, Fr as ArkFr, G1Affine as ArkG1Affine};
use ark_ec::AffineRepr;
use ark_ff::{BigInteger, PrimeField as ArkPrimeField};
use ff::PrimeField as FfPrimeField;
use nova_snark::{
    provider::{bn256_grumpkin::bn256, Bn256EngineIPA, GrumpkinEngine},
    traits::Engine,
};

/// The step circuit's native field: BN254's base field `Fq`.
///
/// This is simultaneously `E1::Scalar` (Grumpkin's scalar field, so
/// Nova folds over it) and `E2::Base` (BN254's coordinate field, so
/// `AllocatedPoint<E2>` is native in it). That coincidence is the
/// whole reason this design is cheap.
pub type CircuitField = <GrumpkinEngine as Engine>::Scalar;

/// BN254's scalar field `Fr` in halo2curves form. Chain scalars and
/// Schnorr responses live here.
pub type Bn254Scalar = <Bn256EngineIPA as Engine>::Scalar;

/// The Nova engine whose G1 points the circuit manipulates natively.
pub type PointEngine = Bn256EngineIPA;

// Compile-time proof that the cycle lines up as documented above: the
// circuit field must be *both* Grumpkin's scalar field and BN254's
// base field. If halo2curves ever stopped aliasing these, this fails
// to compile rather than silently producing an emulated circuit.
const _: () = {
    fn _assert_circuit_field_is_bn254_base(x: CircuitField) -> bn256::Base {
        x
    }
    fn _assert_point_engine_base_matches(x: <PointEngine as Engine>::Base) -> CircuitField {
        x
    }
};

/// Convert a canonical little-endian 32-byte encoding into a
/// halo2curves field element. Returns `None` if the bytes are not a
/// canonical representative (i.e. >= the modulus), which cannot
/// happen for values that came out of arkworks.
fn from_le_bytes<F: FfPrimeField>(bytes: &[u8]) -> Option<F> {
    let mut repr = F::Repr::default();
    let width = repr.as_ref().len();
    if bytes.len() > width {
        return None;
    }
    repr.as_mut()[..bytes.len()].copy_from_slice(bytes);
    Option::from(F::from_repr(repr))
}

/// `ark_bn254::Fq` -> the circuit field.
///
/// Total: every arkworks field element is canonical by construction,
/// so `from_repr` always succeeds. A failure would mean the two
/// crates disagree about the modulus, which is a build-configuration
/// bug rather than a runtime condition — hence the panic.
pub fn ark_fq_to_circuit(x: ArkFq) -> CircuitField {
    let le = x.into_bigint().to_bytes_le();
    from_le_bytes::<CircuitField>(&le)
        .expect("ark_bn254::Fq and halo2curves bn256::Fq share a modulus")
}

/// The circuit field -> `ark_bn254::Fq`. Used by tests and by the
/// native-side FS helpers that need to hand a digest back to
/// arkworks code.
pub fn circuit_to_ark_fq(x: CircuitField) -> ArkFq {
    let repr = x.to_repr();
    ArkFq::from_le_bytes_mod_order(repr.as_ref())
}

/// `ark_bn254::Fr` -> halo2curves `Fr`.
pub fn ark_fr_to_h2(x: ArkFr) -> Bn254Scalar {
    let le = x.into_bigint().to_bytes_le();
    from_le_bytes::<Bn254Scalar>(&le)
        .expect("ark_bn254::Fr and halo2curves bn256::Fr share a modulus")
}

/// halo2curves `Fr` -> `ark_bn254::Fr`.
pub fn h2_fr_to_ark(x: Bn254Scalar) -> ArkFr {
    let repr = x.to_repr();
    ArkFr::from_le_bytes_mod_order(repr.as_ref())
}

/// `ark_bn254::Fr` -> the circuit field `Fq`.
///
/// **Injective.** BN254's scalar modulus `r` is strictly smaller than
/// its base modulus `q`, so every `Fr` element has a unique canonical
/// representative in `Fq` and no reduction occurs. This is what lets
/// the folded IVC state carry the chain scalars `(r_index, r_value)`
/// as ordinary circuit-field elements instead of an emulated
/// non-native encoding.
pub fn ark_fr_to_circuit(x: ArkFr) -> CircuitField {
    let le = x.into_bigint().to_bytes_le();
    from_le_bytes::<CircuitField>(&le).expect("r_bn254 < q_bn254, so every Fr embeds into Fq")
}

/// Partial inverse of [`ark_fr_to_circuit`].
///
/// Only a right inverse: `Fq` is the larger field, so an arbitrary
/// `Fq` element need not be in the image. Callers must know the value
/// is a genuine `Fr` representative — which holds for anything that
/// came out of [`ark_fr_to_circuit`] or out of a truncated Poseidon
/// squeeze (at most 250 bits, hence far below `r`).
pub fn circuit_to_ark_fr(x: CircuitField) -> ArkFr {
    let repr = x.to_repr();
    ArkFr::from_le_bytes_mod_order(repr.as_ref())
}

/// A BN254 G1 point in the shape Nova's ECC gadget consumes:
/// affine coordinates plus an explicit infinity flag.
///
/// Both arkworks and Nova represent the identity as `(0, 0)` with the
/// flag set, so the encoding is unambiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointCoords {
    /// Affine x-coordinate; zero at the identity.
    pub x: CircuitField,
    /// Affine y-coordinate; zero at the identity.
    pub y: CircuitField,
    /// Set exactly at the point at infinity.
    pub is_infinity: bool,
}

impl PointCoords {
    /// The identity, as Nova's `AllocatedPoint::default` encodes it.
    pub fn identity() -> Self {
        Self {
            x: CircuitField::from(0u64),
            y: CircuitField::from(0u64),
            is_infinity: true,
        }
    }

    /// Shape expected by `AllocatedPoint::alloc`.
    pub fn as_alloc_coords(&self) -> (CircuitField, CircuitField, bool) {
        (self.x, self.y, self.is_infinity)
    }

    /// `is_infinity` as a field element, matching how Nova's
    /// `AllocatedPoint` stores and absorbs the flag.
    pub fn infinity_field(&self) -> CircuitField {
        if self.is_infinity {
            CircuitField::from(1u64)
        } else {
            CircuitField::from(0u64)
        }
    }
}

/// Decompose an arkworks BN254 G1 point into circuit-field
/// coordinates. `AffineRepr::xy` returns `None` exactly at the
/// identity, which is where the infinity flag comes from.
pub fn ark_g1_to_coords(p: &ArkG1Affine) -> PointCoords {
    match p.xy() {
        Some((x, y)) => PointCoords {
            x: ark_fq_to_circuit(x),
            y: ark_fq_to_circuit(y),
            is_infinity: false,
        },
        None => PointCoords::identity(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{AdditiveGroup, PrimeGroup};
    use ark_std::rand::SeedableRng;
    use ark_std::UniformRand;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn fq_round_trips() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xF0);
        for _ in 0..256 {
            let a = ArkFq::rand(&mut rng);
            assert_eq!(circuit_to_ark_fq(ark_fq_to_circuit(a)), a);
        }
    }

    #[test]
    fn fr_round_trips() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xF1);
        for _ in 0..256 {
            let a = ArkFr::rand(&mut rng);
            assert_eq!(h2_fr_to_ark(ark_fr_to_h2(a)), a);
        }
    }

    /// Boundary values are where a byte-level bridge breaks: zero,
    /// one, and the largest canonical representative `p - 1`.
    #[test]
    fn field_edge_values_round_trip() {
        for a in [ArkFq::ZERO, ArkFq::from(1u64), -ArkFq::from(1u64)] {
            assert_eq!(circuit_to_ark_fq(ark_fq_to_circuit(a)), a);
        }
        for a in [ArkFr::ZERO, ArkFr::from(1u64), -ArkFr::from(1u64)] {
            assert_eq!(h2_fr_to_ark(ark_fr_to_h2(a)), a);
        }
    }

    /// Field arithmetic must survive the crossing, otherwise the
    /// circuit would be proving a statement about different numbers
    /// than the native auditor checked.
    #[test]
    fn addition_is_preserved() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xF2);
        for _ in 0..64 {
            let a = ArkFq::rand(&mut rng);
            let b = ArkFq::rand(&mut rng);
            assert_eq!(
                ark_fq_to_circuit(a + b),
                ark_fq_to_circuit(a) + ark_fq_to_circuit(b)
            );
        }
    }

    /// The embedding `Fr -> Fq` is only sound because r < q. Assert
    /// that ordering directly rather than trusting it.
    #[test]
    fn scalar_modulus_is_below_base_modulus() {
        let r = <ArkFr as ArkPrimeField>::MODULUS;
        let q = <ArkFq as ArkPrimeField>::MODULUS;
        let r_bytes = r.to_bytes_le();
        let q_bytes = q.to_bytes_le();
        assert_eq!(r_bytes.len(), q_bytes.len());
        // Compare as little-endian magnitudes, most significant first.
        let mut ord = std::cmp::Ordering::Equal;
        for i in (0..r_bytes.len()).rev() {
            ord = r_bytes[i].cmp(&q_bytes[i]);
            if ord != std::cmp::Ordering::Equal {
                break;
            }
        }
        assert_eq!(ord, std::cmp::Ordering::Less, "expected r < q for BN254");
    }

    #[test]
    fn fr_embeds_into_circuit_field_injectively() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xF4);
        for _ in 0..256 {
            let a = ArkFr::rand(&mut rng);
            assert_eq!(circuit_to_ark_fr(ark_fr_to_circuit(a)), a);
        }
        for a in [ArkFr::ZERO, ArkFr::from(1u64), -ArkFr::from(1u64)] {
            assert_eq!(circuit_to_ark_fr(ark_fr_to_circuit(a)), a);
        }
    }

    #[test]
    fn identity_maps_to_infinity() {
        let c = ark_g1_to_coords(&ArkG1Affine::identity());
        assert!(c.is_infinity);
        assert_eq!(c.x, CircuitField::from(0u64));
        assert_eq!(c.y, CircuitField::from(0u64));
        assert_eq!(c.infinity_field(), CircuitField::from(1u64));
    }

    /// A real generator multiple must land on the halo2curves curve —
    /// this is the check that catches a coordinate-encoding mistake
    /// that round-trip tests alone would miss.
    #[test]
    fn generator_multiples_land_on_curve() {
        use ff::Field;
        let mut rng = ChaCha20Rng::seed_from_u64(0xF3);
        for _ in 0..32 {
            let s = ArkFr::rand(&mut rng);
            let p: ArkG1Affine = (ark_bn254::G1Projective::generator() * s).into();
            let c = ark_g1_to_coords(&p);
            assert!(!c.is_infinity);
            // y^2 == x^3 + 3 for BN254 G1.
            let lhs = c.y * c.y;
            let rhs = c.x * c.x * c.x + CircuitField::from(3u64);
            assert_eq!(lhs, rhs, "bridged point is off the halo2curves curve");
            let _ = CircuitField::ONE;
        }
    }
}
