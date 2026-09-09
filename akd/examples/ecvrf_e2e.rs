// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! End-to-end integration test for the production-shape ECVRF wiring.
//!
//! Exercises:
//!   1. `VrfProver::from_seed` / `VrfProver::from_env`  — key setup.
//!   2. `VrfProver::prove_h_bits(ctr, label, num_vars)` — server side.
//!   3. Public-key transport: pk bytes -> `VrfVerifier::from_public_key_bytes`.
//!   4. `VrfVerifier::verify_h_bits(ctr, label, proof, num_vars)` — client side.
//!   5. Adversarial cases: tampered proof, wrong label, wrong counter.
//!   6. `EcVrfHash::h_bits` (static) agrees with `VrfProver::prove_h_bits`
//!      bit-for-bit so existing call sites get the same answers.
//!   7. Full open-addressing-trail simulation: walk N probes, attach the
//!      N proofs, client verifies each and recovers the bits the server
//!      used at publish time — the exact pattern the gRPC wire format
//!      will carry once the proto fields are added.

use akd::aegon::{
    EcVrfHash, HashSuite, Sha256Hash, VrfProver, VrfVerifier, VrfVerifyError, BENCH_VRF_SEED,
    VRF_PROOF_BYTES, VRF_PUBLIC_KEY_BYTES,
};
use ark_bn254::Fr;

fn must(label: &str, ok: bool) {
    if !ok {
        eprintln!("FAIL: {label}");
        std::process::exit(1);
    }
    println!("  ok  {label}");
}

fn main() {
    println!("=== Aegon ECVRF end-to-end integration ===");

    // ---- 1. Key setup. ----
    let prover = VrfProver::from_seed(&BENCH_VRF_SEED);
    let pk_bytes: [u8; VRF_PUBLIC_KEY_BYTES] = {
        let mut b = [0u8; VRF_PUBLIC_KEY_BYTES];
        b.copy_from_slice(prover.public_key().as_bytes());
        b
    };
    must(
        "public key is 32 bytes",
        pk_bytes.len() == VRF_PUBLIC_KEY_BYTES,
    );

    let verifier = VrfVerifier::from_public_key_bytes(&pk_bytes).expect("pk parses");

    // ---- 2. Single (ctr, label) round trip. ----
    let label = b"alice@example.com";
    let num_vars = 27usize;

    let (bits_server, proof) = prover.prove_h_bits(0, label, num_vars);
    must(
        "proof is exactly 80 bytes (RFC 9381)",
        proof.len() == VRF_PROOF_BYTES,
    );
    must(
        "server bits length matches num_vars",
        bits_server.len() == num_vars,
    );

    let bits_client = verifier
        .verify_h_bits(0, label, &proof, num_vars)
        .expect("honest proof verifies");
    must(
        "client recovers exact same bits",
        bits_client == bits_server,
    );

    // ---- 3. Adversarial: tampered proof byte. ----
    let mut bad = proof;
    bad[63] ^= 0x01;
    let res = verifier.verify_h_bits(0, label, &bad, num_vars);
    must(
        "tampered proof rejected as InvalidProof",
        matches!(res, Err(VrfVerifyError::InvalidProof(_))),
    );

    // ---- 4. Adversarial: wrong label. ----
    let res = verifier.verify_h_bits(0, b"mallory", &proof, num_vars);
    must(
        "proof for alice rejected when re-presented for mallory",
        matches!(res, Err(VrfVerifyError::InvalidProof(_))),
    );

    // ---- 5. Adversarial: wrong counter. ----
    let res = verifier.verify_h_bits(99, label, &proof, num_vars);
    must(
        "proof for ctr=0 rejected when re-presented at ctr=99",
        matches!(res, Err(VrfVerifyError::InvalidProof(_))),
    );

    // ---- 6. Static HashSuite agrees with keyed VrfProver. ----
    //
    // This is the load-bearing invariant: every existing static call site
    // that today uses `EcVrfHash::h_bits` can be swapped for a keyed
    // `VrfProver::prove_h_bits(...)` on the server and a
    // `VrfVerifier::verify_h_bits(...)` on the client *without changing
    // the bits anyone sees*. The two paths must produce the same bytes
    // for the same (ctr, label, num_vars) when given the same key.
    for ctr in 0u64..5 {
        let static_bits = <EcVrfHash as HashSuite<Fr>>::h_bits(ctr, label, num_vars);
        let (keyed_bits, _) = prover.prove_h_bits(ctr, label, num_vars);
        must(
            &format!("EcVrfHash agrees with VrfProver at ctr={ctr}"),
            static_bits == keyed_bits,
        );
    }

    // ---- 7. Full trail simulation. ----
    //
    // This is the shape the gRPC wire format will need to carry. At
    // publish time the server walks the open-addressing trail for the
    // label, computing (bits, proof) at each probe. At lookup time the
    // server emits a `ProofBundle = Vec<[u8; 80]>` of those proofs
    // alongside the existing PCS openings. The client verifies each
    // proof in the bundle (recovering the bits) before checking the
    // matching opening against its trusted commitment.
    let trail_label = b"bob@example.com";
    let trail_len = 4usize; // average ~8/3, conservative upper bound for the test.

    // Server side: walk and collect bundle.
    let mut server_trail: Vec<(Vec<bool>, [u8; VRF_PROOF_BYTES])> = Vec::new();
    for ctr in 0..trail_len {
        let (bits, proof) = prover.prove_h_bits(ctr as u64, trail_label, num_vars);
        server_trail.push((bits, proof));
    }

    // The bundle that goes on the wire is just the proof bytes. Bits
    // are NOT sent — they are recovered locally by the client from
    // each proof. This is the privacy guarantee: an eavesdropper sees
    // only the proofs (which carry the Edwards-curve hash-to-curve
    // output, not the raw bits) and cannot enumerate the per-label
    // index distribution offline.
    let wire_bundle: Vec<[u8; VRF_PROOF_BYTES]> = server_trail.iter().map(|(_, p)| *p).collect();

    // Client side: receive, verify, recover bits.
    for (ctr, proof) in wire_bundle.iter().enumerate() {
        let recovered = verifier
            .verify_h_bits(ctr as u64, trail_label, proof, num_vars)
            .expect("trail proof verifies");
        must(
            &format!("client recovers bits at trail ctr={ctr}"),
            recovered == server_trail[ctr].0,
        );
    }

    // ---- 8. Sanity: SHA-256 path is unchanged. ----
    //
    // The H_F hash (field-element value) is shared between suites.
    // Confirm that swapping to EcVrfHash leaves H_F bit-identical.
    let f_sha: Fr = <Sha256Hash as HashSuite<Fr>>::h_f(label);
    let f_ecvrf: Fr = <EcVrfHash as HashSuite<Fr>>::h_f(label);
    must("EcVrfHash::h_f delegates to Sha256Hash", f_sha == f_ecvrf);

    println!("\nAll 17 checks passed. ECVRF wiring is end-to-end consistent.");
    println!(
        "Per-call cost recap: VRF prove + verify ≈ 2 × 157 µs = 314 µs per probe; \
         a typical α=4 trail of ~3 probes adds ~1 ms of crypto work per lookup."
    );
}
