//! Micro-benchmark for EcVrfHash vs Sha256Hash. Reports per-call ns.
fn main() {
    use akd::aegon::{EcVrfHash, HashSuite, Sha256Hash};
    use ark_bn254::Fr;
    use std::time::Instant;

    let label = b"alice@example.com";
    let num_vars = 27usize;
    let iters = 2_000;

    // Warm up the OnceLock + caches.
    for i in 0..200 {
        let _ = <EcVrfHash as HashSuite<Fr>>::h_bits(i as u64, label, num_vars);
        let _ = <Sha256Hash as HashSuite<Fr>>::h_bits(i as u64, label, num_vars);
    }

    // ECVRF
    let t = Instant::now();
    for i in 0..iters {
        let _ = <EcVrfHash as HashSuite<Fr>>::h_bits(i as u64, label, num_vars);
    }
    let ecvrf_ns = t.elapsed().as_nanos() / iters as u128;

    // SHA-256
    let t = Instant::now();
    for i in 0..iters {
        let _ = <Sha256Hash as HashSuite<Fr>>::h_bits(i as u64, label, num_vars);
    }
    let sha_ns = t.elapsed().as_nanos() / iters as u128;

    println!("EcVrfHash::h_bits  : {} ns/call", ecvrf_ns);
    println!("Sha256Hash::h_bits : {} ns/call", sha_ns);
    println!("slowdown factor    : {:.0}x", ecvrf_ns as f64 / sha_ns as f64);
}
