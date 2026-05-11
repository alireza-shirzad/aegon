//! Hash suite for Aegon.
//!
//! Aegon's index assignment relies on two hash functions (paper §5.1):
//!   - `H_bits(ctr, label) -> {0,1}^{log N}` : maps a (probe counter, label) to a
//!     boolean point on the hypercube. Need not be collision-resistant on its
//!     own — uniqueness is enforced by the open-addressing protocol.
//!   - `H_F(label) -> F \ {0}` : the field-element value stored at a label's
//!     assigned index. Must be collision-resistant and preimage-resistant on
//!     `F`.
//!
//! In production, both are instantiated with a VRF (paper §6.1 / §6.5) so that
//! clients cannot enumerate the index distribution offline. Here we provide a
//! deterministic SHA-256 instantiation suitable for the protocol skeleton; the
//! `HashSuite` trait lets callers swap in a VRF later without touching the
//! Aegon core.

use ark_ff::PrimeField;
use sha2::{Digest, Sha256};

/// The two hash functions needed by Aegon's index-assignment protocol.
pub trait HashSuite<F: PrimeField> {
    /// Maps `(ctr, label)` to a boolean vector of length `num_vars`.
    fn h_bits(ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool>;

    /// Maps `label` to a non-zero field element.
    fn h_f(label: &[u8]) -> F;
}

/// SHA-256-based deterministic hash suite. Domain-separated by tag bytes so
/// `H_bits` and `H_F` cannot collide.
pub struct Sha256Hash;

impl<F: PrimeField> HashSuite<F> for Sha256Hash {
    fn h_bits(ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool> {
        // Stretch SHA-256 output if num_vars exceeds 256 bits.
        let mut bits = Vec::with_capacity(num_vars);
        let mut counter: u32 = 0;
        while bits.len() < num_vars {
            let mut h = Sha256::new();
            h.update(b"aegon.h_bits");
            h.update(ctr.to_le_bytes());
            h.update(counter.to_le_bytes());
            h.update((label.len() as u64).to_le_bytes());
            h.update(label);
            let digest = h.finalize();
            for byte in digest.iter() {
                for bit in 0..8 {
                    if bits.len() == num_vars {
                        return bits;
                    }
                    bits.push((byte >> bit) & 1 == 1);
                }
            }
            counter += 1;
        }
        bits
    }

    fn h_f(label: &[u8]) -> F {
        // Reduce SHA-256 output mod p; rehash on the (negligibly probable) zero
        // case so the post-condition `H_F(label) != 0` holds without leaking
        // anything about `label`.
        let mut counter: u32 = 0;
        loop {
            let mut h = Sha256::new();
            h.update(b"aegon.h_f");
            h.update(counter.to_le_bytes());
            h.update((label.len() as u64).to_le_bytes());
            h.update(label);
            let digest = h.finalize();
            let value = F::from_le_bytes_mod_order(&digest);
            if !value.is_zero() {
                return value;
            }
            counter += 1;
        }
    }
}

/// Convert a boolean index `bits` to the `usize` encoding used by the
/// underlying multilinear PCS's evaluation table.
///
/// Block-major C-order: `bits` is split into `dims.len()` consecutive
/// blocks of widths `dims[0], dims[1], ...`; each block is encoded
/// little-endian within itself, and earlier blocks land in the higher
/// bits of the final index. This matches KZH-k's internal layout
/// (Figure 14, paper) — earlier-numbered variables (`X_1` etc.) occupy
/// the high bits because the H tensor is stored in C-order with the
/// first axis varying slowest.
///
/// For PCSs that use the ark_poly default convention (single block,
/// variable `i` in bit `i`), pass `dims = &[bits.len()]`.
///
/// # Panics
///
/// If `dims.iter().sum::<usize>() != bits.len()`.
pub fn bool_index_to_usize(bits: &[bool], dims: &[usize]) -> usize {
    debug_assert_eq!(
        dims.iter().copied().sum::<usize>(),
        bits.len(),
        "dims must partition bits"
    );
    let mut idx: usize = 0;
    let mut start = 0;
    for &d in dims {
        let end = start + d;
        let block = bits_le_to_usize(&bits[start..end]);
        idx = (idx << d) | block;
        start = end;
    }
    idx
}

fn bits_le_to_usize(bits: &[bool]) -> usize {
    let mut acc: usize = 0;
    for (i, &b) in bits.iter().enumerate() {
        if b {
            acc |= 1 << i;
        }
    }
    acc
}

/// Convert a boolean index to a `Vec<F>` evaluation point.
pub fn bool_index_to_point<F: PrimeField>(bits: &[bool]) -> Vec<F> {
    bits.iter()
        .map(|&b| if b { F::one() } else { F::zero() })
        .collect()
}
