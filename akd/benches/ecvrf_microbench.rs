//! Microbench for the ECVRF prove/verify path used by `EcVrfHash`.
//!
//! What this measures: per-call cost of
//!
//!   * `VrfProver::prove_h_bits(ctr, label, num_vars)`
//!   * `VrfVerifier::verify_h_bits(ctr, label, &proof, num_vars)`
//!   * `Sha256Hash::h_bits(ctr, label, num_vars)`  — non-VRF baseline
//!
//! across a few representative `num_vars` (== `shard_log_capacity`):
//! 10 (toy), 16 (publish-sweep default), 22 (rss_probe), 27 (large
//! cluster). The bits output is `num_vars` long; everything else
//! about the call is independent of `num_vars` so the curve should
//! be flat for prove/verify and ~linear in `num_vars` only via the
//! tail bit-extract.
//!
//! Per-call wall-clock cost is what dominates the cluster benches'
//! publish/lookup latency once we flipped from `Sha256Hash` to
//! `EcVrfHash` — track this in CI so a future ed25519-dalek bump
//! that regresses prove time gets caught here, not in a one-off
//! re-run of the cluster harness.
//!
//! ## Running
//!
//! ```text
//! cargo bench --bench ecvrf_microbench
//! ```
//!
//! Override the `num_vars` sweep at runtime:
//!
//! ```text
//! AEGON_BENCH_VRF_NUM_VARS=16,22 cargo bench --bench ecvrf_microbench
//! ```

use std::sync::OnceLock;

use akd::aegon::{HashSuite, Sha256Hash, VrfProver, VrfVerifier, BENCH_VRF_SEED};
use ark_bn254::Fr;

const DEFAULT_NUM_VARS: &[usize] = &[10, 16, 22, 27];
const LABEL: &[u8] = b"aegon-microbench-label-fixed";
const CTR: u64 = 42;

fn num_vars() -> &'static [usize] {
    static V: OnceLock<Vec<usize>> = OnceLock::new();
    V.get_or_init(|| {
        std::env::var("AEGON_BENCH_VRF_NUM_VARS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_NUM_VARS.to_vec())
    })
    .as_slice()
}

fn prover() -> &'static VrfProver {
    static P: OnceLock<VrfProver> = OnceLock::new();
    P.get_or_init(|| VrfProver::from_seed(&BENCH_VRF_SEED))
}

fn verifier() -> &'static VrfVerifier {
    static V: OnceLock<VrfVerifier> = OnceLock::new();
    V.get_or_init(|| VrfVerifier::new(prover().public_key().clone()))
}

#[divan::bench(args = num_vars().iter().copied().collect::<Vec<_>>(), sample_count = 50)]
fn vrf_prove(bencher: divan::Bencher, nv: usize) {
    let p = prover();
    bencher.bench_local(|| {
        let (bits, proof) = p.prove_h_bits(CTR, LABEL, nv);
        divan::black_box((bits, proof))
    });
}

#[divan::bench(args = num_vars().iter().copied().collect::<Vec<_>>(), sample_count = 50)]
fn vrf_verify(bencher: divan::Bencher, nv: usize) {
    let p = prover();
    let v = verifier();
    let (_bits, proof) = p.prove_h_bits(CTR, LABEL, nv);
    bencher.bench_local(|| {
        let bits = v
            .verify_h_bits(CTR, LABEL, &proof, nv)
            .expect("proof verifies");
        divan::black_box(bits)
    });
}

#[divan::bench(args = num_vars().iter().copied().collect::<Vec<_>>(), sample_count = 50)]
fn sha256_h_bits(bencher: divan::Bencher, nv: usize) {
    bencher.bench_local(|| {
        let bits = <Sha256Hash as HashSuite<Fr>>::h_bits(CTR, LABEL, nv);
        divan::black_box(bits)
    });
}

fn main() {
    eprintln!(
        "ecvrf_microbench: num_vars={:?} label={:?} ctr={}",
        num_vars(),
        std::str::from_utf8(LABEL).unwrap_or("<binary>"),
        CTR,
    );
    divan::main();
}
