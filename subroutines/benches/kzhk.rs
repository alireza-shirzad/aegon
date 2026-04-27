//! Divan benchmarks for the KZH-k polynomial commitment scheme.
//!
//! Sweeps a single [`BenchCase`] (`nv`, `k`, `zk`, `boolean`) across each
//! stage of the pipeline: commit, update_state, open, verify.
//!
//! - `nv`:      number of variables (polynomial size is `2^nv`).
//! - `k`:       number of blocks the variables are split across (KZH-k).
//! - `zk`:      plain KZH-k (false) vs the hiding Sigma-protocol variant (true).
//! - `boolean`: opening point is drawn from `{0,1}^nv` (true) or `F^nv` (false).
//!              Only meaningful for `open` / `verify`; ignored by the other
//!              stages.
//!
//! Run:
//! ```text
//! cargo bench -p subroutines --bench kzhk
//! cargo bench -p subroutines --bench kzhk --features parallel
//! ```

use ark_bn254::{Bn254 as E, Fr};
use ark_std::{test_rng, UniformRand};
use arithmetic::multilinear_polynomial::rand_sparse_mle;
use divan::{black_box, Bencher};
use std::fmt;
use subroutines::{
    pcs::kzhk::{structs::KZHKConfig, KZHK},
    poly::DenseOrSparseMLE,
    PolynomialCommitmentScheme,
};
use transcript::IOPTranscript;

fn main() {
    // Install the workspace-wide tracing subscriber so `#[instrument]`
    // and `debug_span!` spans print their durations. Filter via
    // `RUST_LOG=subroutines=debug,arithmetic=debug` (see
    // `util::tracing::init` for defaults).
    util::tracing::init().expect("tracing init");
    divan::main();
}

/// One point in the benchmark sweep. A compact `Debug` impl is provided so
/// divan shows readable bench names like
/// `nv=16/k=2/zk=true/bool=false/sp=100`.
///
/// `sparsity` is a percentage in `[0, 100]`: the polynomial is built as a
/// sparse MLE with `floor(sparsity * 2^nv / 100)` random non-zero entries.
#[derive(Clone, Copy)]
struct BenchCase {
    nv: usize,
    k: usize,
    zk: bool,
    boolean: bool,
    sparsity: u8,
}

impl fmt::Debug for BenchCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "nv={}/k={}/zk={}/bool={}/sp={}",
            self.nv, self.k, self.zk, self.boolean, self.sparsity
        )
    }
}

const fn case(nv: usize, k: usize, zk: bool, boolean: bool, sparsity: u8) -> BenchCase {
    assert!(sparsity <= 100, "sparsity must be a percentage in [0, 100]");
    BenchCase {
        nv,
        k,
        zk,
        boolean,
        sparsity,
    }
}

/// Sweep points. Edit this list to add or trim bench configurations.
const CASES: &[BenchCase] = &[
    // case(16, 2, false, false, 100),
    // case(16, 2, false, true,  100),
    // case(16, 2, true,  false, 100),
    // case(16, 2, true,  true,  100),
    // case(17, 2, true,  true,  100),
    // case(18, 2, true,  true,  100),
    // case(19, 2, true,  true,  100),
    // case(27, 8, true,  true,  100),
    case(28, 2, false, true, 5),
    case(28, 2, false, true, 10),
    case(28, 2, false, true, 20),
    case(28, 2, false, true, 50),
];

type ProverParam = <KZHK<E> as PolynomialCommitmentScheme<E>>::ProverParam;
type VerifierParam = <KZHK<E> as PolynomialCommitmentScheme<E>>::VerifierParam;
type Prepared = (ProverParam, VerifierParam, DenseOrSparseMLE<Fr>, Vec<Fr>);

/// Samples an opening point of length `nv`, either Boolean or fully random.
fn sample_point(nv: usize, boolean: bool, rng: &mut impl ark_std::rand::Rng) -> Vec<Fr> {
    if boolean {
        (0..nv)
            .map(|_| Fr::from((usize::rand(rng) % 2) as u64))
            .collect()
    } else {
        (0..nv).map(|_| Fr::rand(rng)).collect()
    }
}

/// Prepares `(ck, vk, poly, point)` outside the timed region.
fn prepare(c: BenchCase) -> Prepared {
    let mut rng = test_rng();
    let srs = KZHK::<E>::gen_srs_for_testing(KZHKConfig::new(c.k, c.zk), &mut rng, c.nv).unwrap();
    let (ck, vk) = KZHK::<E>::trim(srs, None, Some(c.nv)).unwrap();
    let domain_size = 1u128 << c.nv;
    let nnz = ((domain_size * c.sparsity as u128) / 100) as usize;
    let poly = DenseOrSparseMLE::Sparse(rand_sparse_mle::<Fr, _>(c.nv, nnz, &mut rng));
    let point = sample_point(c.nv, c.boolean, &mut rng);
    (ck, vk, poly, point)
}

// ---------------------------------------------------------------------------
// commit (boolean is ignored)
// ---------------------------------------------------------------------------

// #[divan::bench(args = CASES, sample_count = 1, sample_size = 1)]
// fn commit(bencher: Bencher, c: &BenchCase) {
//     let (ck, _vk, poly, _point) = prepare(*c);
//     bencher.bench_local(|| {
//         let out = KZHK::<E>::commit(&ck, &poly).unwrap();
//         black_box(out);
//     });
// }

// ---------------------------------------------------------------------------
// update_state (boolean is ignored; fills the `d_bool` Boolean-opening table)
// ---------------------------------------------------------------------------

// #[divan::bench(args = CASES, sample_count = 1, sample_size = 1)]
// fn update_state(bencher: Bencher, c: &BenchCase) {
//     let (ck, _vk, poly, _point) = prepare(*c);
//     let (com, state0) = KZHK::<E>::commit(&ck, &poly).unwrap();
//     bencher
//         .with_inputs(|| state0.clone())
//         .bench_local_values(|mut state| {
//             KZHK::<E>::update_state(&ck, &poly, &com, &mut state).unwrap();
//             black_box(state);
//         });
// }

// ---------------------------------------------------------------------------
// open
// ---------------------------------------------------------------------------

#[divan::bench(args = CASES, sample_count = 10, sample_size = 1)]
fn open(bencher: Bencher, c: &BenchCase) {
    let (ck, _vk, poly, point) = prepare(*c);
    let (com, mut state) = KZHK::<E>::commit(&ck, &poly).unwrap();
    KZHK::<E>::update_state(&ck, &poly, &com, &mut state).unwrap();
    bencher
        .with_inputs(|| IOPTranscript::<Fr>::new(b"bench_kzhk"))
        .bench_local_values(|mut transcript| {
            let out = KZHK::<E>::open(&ck, &com, &poly, &point, &state, &mut transcript).unwrap();
            black_box(out);
        });
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

// #[divan::bench(args = CASES, sample_count = 1, sample_size = 1)]
// fn verify(bencher: Bencher, c: &BenchCase) {
//     let (ck, vk, poly, point) = prepare(*c);
//     let (com, mut state) = KZHK::<E>::commit(&ck, &poly).unwrap();
//     KZHK::<E>::update_state(&ck, &poly, &com, &mut state).unwrap();
//     let mut prover_transcript = IOPTranscript::<Fr>::new(b"bench_kzhk");
//     let (proof, value) =
//         KZHK::<E>::open(&ck, &com, &poly, &point, &state, &mut prover_transcript).unwrap();
//     bencher
//         .with_inputs(|| IOPTranscript::<Fr>::new(b"bench_kzhk"))
//         .bench_local_values(|mut transcript| {
//             let ok = KZHK::<E>::verify(&vk, &com, &point, &value, &proof, &mut transcript).unwrap();
//             black_box(ok);
//         });
// }
