// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Quick smoke for EcVrfHash. Verifies:
//!   1. h_bits is deterministic (same input -> same output).
//!   2. h_bits respects num_vars.
//!   3. The output is verifiable end-to-end against the corresponding
//!      public key (this is the property SHA-256 cannot provide).
fn main() {
    use akd::aegon::{EcVrfHash, HashSuite};
    use akd_core::ecvrf::{Output, VRFPrivateKey, VRFPublicKey};
    use ark_bn254::Fr;

    let label = b"alice@example.com";
    let ctr: u64 = 0;
    let num_vars = 27usize;

    // (1) determinism
    let bits_a = <EcVrfHash as HashSuite<Fr>>::h_bits(ctr, label, num_vars);
    let bits_b = <EcVrfHash as HashSuite<Fr>>::h_bits(ctr, label, num_vars);
    assert_eq!(bits_a, bits_b, "h_bits must be deterministic");
    assert_eq!(bits_a.len(), num_vars, "h_bits must respect num_vars");

    // (2) different counter -> different bits w.h.p.
    let bits_c = <EcVrfHash as HashSuite<Fr>>::h_bits(ctr + 1, label, num_vars);
    assert_ne!(
        bits_a, bits_c,
        "different counter should give different bits"
    );

    // (3) verifiability: reconstruct alpha, run prove on the same key,
    //     check the proof verifies under the corresponding pk.
    let seed: [u8; 32] = *b"aegon-bench-ecvrf-edwards25519!\0";
    let sk = VRFPrivateKey::try_from(seed.as_slice()).unwrap();
    let pk = VRFPublicKey::from(&sk);

    let mut alpha = Vec::new();
    alpha.extend_from_slice(b"aegon.h_bits");
    alpha.extend_from_slice(&ctr.to_le_bytes());
    alpha.extend_from_slice(&(label.len() as u64).to_le_bytes());
    alpha.extend_from_slice(label);

    let proof = sk.prove(&alpha);
    pk.verify(&proof, &alpha).expect("VRF proof must verify");

    // (4) the bits derived from the proof's Output match h_bits.
    let out = Output::from(&proof).to_bytes();
    let mut bits_from_proof = Vec::with_capacity(num_vars);
    'outer: for byte in out.iter() {
        for b in 0..8 {
            if bits_from_proof.len() == num_vars {
                break 'outer;
            }
            bits_from_proof.push((byte >> b) & 1 == 1);
        }
    }
    assert_eq!(bits_a, bits_from_proof, "h_bits must equal Output bits");

    println!(
        "EcVrfHash smoke OK: {} bits, deterministic, verifiable.",
        num_vars
    );
}
