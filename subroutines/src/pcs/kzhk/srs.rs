//! KZH-k structured reference string (SRS).
//!
//! Implements the setup described in Figure 14 of ePrint 2025/1580: for
//! each block `j in [k]` the setup samples a vector of trapdoors
//! `mu_{b,j}` indexed by `b in {0,1}^{d_j}`, then forms the `G1` tensors
//! `H_t[b_t, ..., b_k] = g^{prod_{j=t}^{k} mu_{b_j, j}}` for `t = 1..k`
//! (with the full-product tensor `H_1` used for commitments and the
//! partial-product tensors `H_2..H_k` used for opening), and the `G2`
//! side elements `V_{b,j} = v^{mu_{b,j}}` used for the per-level pairing
//! checks in verification. The optional `hiding_sparsity` records the
//! Appendix-D sparse-masking parameter `k * N^{1/k}` for zk openings.

use std::sync::Arc;

use crate::{
    pcs::{kzhk::structs::Tensor, PCSGlobalParam},
    PCSError, StructuredReferenceString,
};
use ark_ec::{pairing::Pairing, scalar_mul::BatchMulPreprocessing, CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{rand::Rng, One, UniformRand};
use ndarray::{ArrayD, IxDyn};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Universal parameters for KZH-k.
///
/// Fields correspond to the SRS described in Figure 14 / Appendix E:
///
/// - `dimensions` = `[d_1, ..., d_k]`, the block sizes. Their sum is
///   `log_2 N`, the total number of variables.
/// - `h_tensors` = `[H_1, H_2, ..., H_k]`: for each `t`, `H_t` is the
///   `(d_t + d_{t+1} + ... + d_k)`-dimensional `G1` tensor whose entry
///   at index `(b_t, ..., b_k)` is `g` raised to the product of the
///   corresponding trapdoors. `H_1` is used for commitments; `H_t` for
///   `t > 1` is used to commit to partial evaluations during opening.
/// - `v_mat` = `V_{b,j}` for `j in [k]`, `b in {0,1}^{d_j}`: the `G2`-side
///   powers `v^{mu_{b,j}}` preprocessed for pairings.
/// - `v`, `g`: `G2` and `G1` generators used during setup.
/// - `h`: independent `G1` generator with unknown discrete log with
///   respect to `g` used as the hiding base for Appendix-D blinding.
/// - `hiding_sparsity`: `Some(k * N^{1/k})` for zk SRS (drives the sparse
///   masking polynomial size from Lemmas 4 and 5); `None` for plain KZH-k.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct KZHKUniversalParams<E: Pairing> {
    dimensions: Vec<usize>,
    h_tensors: Arc<Vec<Tensor<E::G1Affine>>>,
    v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
    v: E::G2Affine,
    g: E::G1Affine,
    h: E::G1Affine,
    hiding_sparsity: Option<usize>,
}

impl<E: Pairing> PCSGlobalParam for KZHKUniversalParams<E> {
    fn is_zk(&self) -> bool {
        self.hiding_sparsity.is_some()
    }
}

impl<E: Pairing> KZHKUniversalParams<E> {
    /// Create a new universal parameter
    pub fn new(
        dimensions: Vec<usize>,
        h_tensors: Arc<Vec<Tensor<E::G1Affine>>>,
        v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
        v: E::G2Affine,
        g: E::G1Affine,
        h: E::G1Affine,
        hiding_sparsity: Option<usize>,
    ) -> Self {
        Self {
            dimensions,
            h_tensors,
            v_mat,
            v,
            g,
            h,
            hiding_sparsity,
        }
    }

    pub fn get_dimensions(&self) -> &Vec<usize> {
        &self.dimensions
    }

    pub fn get_h_tensors(&self) -> &Vec<Tensor<E::G1Affine>> {
        &self.h_tensors
    }

    pub fn get_v_mat(&self) -> &Vec<Vec<E::G2Prepared>> {
        &self.v_mat
    }

    pub fn get_v(&self) -> E::G2Affine {
        self.v
    }

    pub fn get_g(&self) -> E::G1Affine {
        self.g
    }
    pub fn get_h(&self) -> E::G1Affine {
        self.h
    }
    pub fn get_hiding_sparsity(&self) -> Option<usize> {
        self.hiding_sparsity
    }
}

/// Prover parameters: the subset of the universal SRS needed to commit,
/// update auxiliaries, and open. Mirrors the notation of Figure 14:
/// keeps all `H_t` tensors and `v_mat`, plus the hiding base `h` and
/// the optional sparsity for the zk masking polynomial.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct KZHKProverParam<E: Pairing> {
    dimensions: Vec<usize>,
    h_tensors: Arc<Vec<Tensor<E::G1Affine>>>,
    v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
    h: E::G1Affine,
    hiding_sparsity: Option<usize>,
}
impl<E: Pairing> KZHKProverParam<E> {
    /// Create a new prover parameter
    pub fn new(
        dimensions: Vec<usize>,
        h_tensors: Arc<Vec<Tensor<E::G1Affine>>>,
        v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
        h: E::G1Affine,
        hiding_sparsity: Option<usize>,
    ) -> Self {
        Self {
            dimensions,
            h_tensors,
            v_mat,
            h,
            hiding_sparsity,
        }
    }

    pub fn get_dimensions(&self) -> &Vec<usize> {
        &self.dimensions
    }

    pub fn get_h_tensors(&self) -> &Vec<Tensor<E::G1Affine>> {
        &self.h_tensors
    }

    pub fn get_v_mat(&self) -> &Vec<Vec<E::G2Prepared>> {
        &self.v_mat
    }

    pub fn get_h(&self) -> E::G1Affine {
        self.h
    }

    pub fn get_hiding_sparsity(&self) -> Option<usize> {
        self.hiding_sparsity
    }
}
impl<E: Pairing> PCSGlobalParam for KZHKVerifierParam<E> {
    fn is_zk(&self) -> bool {
        self.hiding_sparsity.is_some()
    }
}
/// Verifier parameters: the small projection of the universal SRS the
/// verifier needs.
///
/// Keeps only `H_k` (the innermost tensor, used to check the final
/// commitment against the tail polynomial), the prepared `v_mat` used
/// in the per-level pairing equations, the hiding base `h`, and
/// `minus_v = -v` which is cached to rewrite the per-level check
/// `e(C_{j-1}, V) = prod_b e(D_{j,b}, V_{b,j})` as a single multi-pairing
/// that must equal 1.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct KZHKVerifierParam<E: Pairing> {
    dimensions: Vec<usize>,
    h_tensor: Arc<Tensor<E::G1Affine>>,
    minus_v: E::G2Affine,
    v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
    h: E::G1Affine,
    hiding_sparsity: Option<usize>,
}

impl<E: Pairing> KZHKVerifierParam<E> {
    /// Create a new verifier parameter
    pub fn new(
        dimensions: Vec<usize>,
        h_tensor: Arc<Tensor<E::G1Affine>>,
        v: E::G2Affine,
        v_mat: Arc<Vec<Vec<E::G2Prepared>>>,
        h: E::G1Affine,
        hiding_sparsity: Option<usize>,
    ) -> Self {
        Self {
            dimensions,
            h_tensor,
            minus_v: -v,
            v_mat,
            h,
            hiding_sparsity,
        }
    }

    pub fn get_dimensions(&self) -> &Vec<usize> {
        &self.dimensions
    }

    pub fn get_h_tensor(&self) -> &Tensor<E::G1Affine> {
        &self.h_tensor
    }

    pub fn get_minus_v(&self) -> E::G2Affine {
        self.minus_v
    }

    pub fn get_v_mat(&self) -> &Vec<Vec<E::G2Prepared>> {
        &self.v_mat
    }

    pub fn get_h(&self) -> E::G1Affine {
        self.h
    }

    pub fn get_hiding_sparsity(&self) -> Option<usize> {
        self.hiding_sparsity
    }
}

impl<E: Pairing> PCSGlobalParam for KZHKProverParam<E> {
    fn is_zk(&self) -> bool {
        self.hiding_sparsity.is_some()
    }
}

impl<E: Pairing> StructuredReferenceString<E> for KZHKUniversalParams<E> {
    type ProverParam = KZHKProverParam<E>;
    type VerifierParam = KZHKVerifierParam<E>;

    /// Extract the prover parameters from the public parameters.
    fn extract_prover_param(&self, _supported_num_vars: usize) -> Self::ProverParam {
        KZHKProverParam::new(
            self.dimensions.clone(),
            self.h_tensors.clone(),
            self.v_mat.clone(),
            self.h,
            self.hiding_sparsity,
        )
    }

    /// Extract the verifier parameters from the public parameters.
    fn extract_verifier_param(&self, _supported_num_vars: usize) -> Self::VerifierParam {
        KZHKVerifierParam::new(
            self.dimensions.clone(),
            self.h_tensors[self.dimensions.len() - 1].clone().into(),
            self.v,
            self.v_mat.clone(),
            self.h,
            self.hiding_sparsity,
        )
    }

    fn trim(
        &self,
        supported_num_vars: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        Ok((
            self.extract_prover_param(supported_num_vars),
            self.extract_verifier_param(supported_num_vars),
        ))
    }

    /// Samples a fresh KZH-k SRS supporting polynomials in `num_vars`
    /// variables split into `k` blocks. Implements the setup procedure
    /// of Figure 14: samples fresh trapdoors `{mu_{b,j}}`, builds the
    /// tensor family `H_1..H_k` by expanding products of trapdoors and
    /// performing a batched scalar multiplication against `g`, and
    /// builds `v_mat` by batched scalar multiplication against `v`.
    /// When `zk` is set, additionally records
    /// `hiding_sparsity = ceil(k * N^{1/k})` as prescribed by
    /// Lemmas 4 and 5 of Appendix D for the sparse masking polynomial.
    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        k: usize,
        zk: bool,
        num_vars: usize,
    ) -> Result<KZHKUniversalParams<E>, PCSError> {
        // ----- Dimensions: split num_vars across k -----
        let d = num_vars / k;
        let remainder_d = num_vars % k;
        let mut dimensions = vec![d; k];
        for dim in dimensions.iter_mut().take(remainder_d) {
            *dim += 1;
        }

        // ----- Public generators -----
        let g = E::G1::rand(rng);
        let h = E::G1::rand(rng);
        let v = E::G2::rand(rng);

        // ----- Trapdoors mu_mat: mu_mat[j].len() = 2^{d_j} -----
        let mu_mat: Vec<Vec<E::ScalarField>> = (0..k)
            .map(|j| {
                (0..(1usize << dimensions[j]))
                    .map(|_| E::ScalarField::rand(rng))
                    .collect()
            })
            .collect();

        let mu_mat = Arc::new(mu_mat);
        let dimensions_arc = Arc::new(dimensions.clone());

        // ---------- Build H_t tensors (outer sequential to bound RAM) ----------
        let h_tenso_span = tracing::debug_span!("KZHK::gen_srs_for_testing::h_tensors");
        let h_tenso_guard = h_tenso_span.enter();
        let mut h_tensors: Vec<Tensor<E::G1Affine>> = Vec::with_capacity(k);

        for t in 0..k {
            let dims = &dimensions_arc[t..];
            let shape: Vec<usize> = dims.iter().map(|&dj| 1usize << dj).collect();
            let len: usize = shape.iter().product();
            let axes = shape.len();

            // 1) Build scalar buffer exps[r_t,...,r_{k-1}] = ∏_{j=t}^{k-1} mu_mat[j][r_j]
            let mut exps: Vec<E::ScalarField> = vec![E::ScalarField::one(); len];

            // axis_stride = product of sizes of trailing axes processed so far (C-order).
            let mut axis_stride = 1usize;
            for a in (0..axes).rev() {
                let size_a = shape[a]; // = 2^{d_{t+a}}
                let block = size_a * axis_stride; // elements per full cycle along this axis
                let j = t + a; // global mu axis
                let mu_j = &mu_mat[j];

                #[cfg(feature = "parallel")]
                {
                    use rayon::slice::ParallelSliceMut;

                    exps.par_chunks_mut(block).for_each(|chunk| {
                        // chunk layout: [ r=0 segment | r=1 segment | ... ] each of length
                        // axis_stride
                        for r in 0..size_a {
                            let mu = mu_j[r];
                            let seg = &mut chunk[r * axis_stride..(r + 1) * axis_stride];
                            for e in seg.iter_mut() {
                                *e *= mu;
                            }
                        }
                    });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for chunk in exps.chunks_mut(block) {
                        for r in 0..size_a {
                            let mu = mu_j[r];
                            let seg = &mut chunk[r * axis_stride..(r + 1) * axis_stride];
                            for e in seg.iter_mut() {
                                *e *= mu;
                            }
                        }
                    }
                }

                axis_stride *= size_a;
            }

            // 2) One batch mul on base g, returning affine points directly. NOTE: if your
            //    API expects "max_degree + 1" instead of count, adjust accordingly.
            let table_g = BatchMulPreprocessing::new(g, len);
            let flat_affine: Vec<E::G1Affine> = table_g.batch_mul(&exps);

            // 3) Pack into ndarray (C-order)
            let arr = ArrayD::from_shape_vec(IxDyn(&shape), flat_affine)
                .expect("shape consistent with buffer length");
            h_tensors.push(Tensor(arr));
        }

        let h_tensors = Arc::new(h_tensors);
        drop(h_tenso_guard);

        // ---------- Build v_mat (parallel per j), also via BatchMulPreprocessing
        // ----------
        let v_mat_span = tracing::debug_span!("KZHK::gen_srs_for_testing::v_mat");
        let v_mat_guard = v_mat_span.enter();

        let v_mat: Vec<Vec<<E as Pairing>::G2Prepared>> = {
            #[cfg(feature = "parallel")]
            {
                (0..k)
                    .into_par_iter()
                    .map(|j| {
                        let rows = 1usize << dimensions_arc[j];
                        let table_v = BatchMulPreprocessing::new(v, rows);
                        let aff: Vec<E::G2Affine> = table_v.batch_mul(&mu_mat[j]);
                        aff.into_iter()
                            .map(<E as Pairing>::G2Prepared::from)
                            .collect()
                    })
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..k)
                    .map(|j| {
                        let rows = 1usize << dimensions_arc[j];
                        let table_v = BatchMulPreprocessing::new(v, rows);
                        let aff: Vec<E::G2Affine> = table_v.batch_mul(&mu_mat[j]);
                        aff.into_iter()
                            .map(<E as Pairing>::G2Prepared::from)
                            .collect()
                    })
                    .collect()
            }
        };

        let v_mat = Arc::new(v_mat);
        drop(v_mat_guard);
        let hiding_sparsity = if zk {
            Some(ceil_k_root_scaled(1u128 << num_vars, k as u32) as usize)
        } else {
            None
        };

        Ok(KZHKUniversalParams::new(
            (*dimensions_arc).clone(),
            h_tensors,
            v_mat,
            v.into_affine(),
            g.into_affine(),
            h.into_affine(),
            hiding_sparsity,
        ))
    }
}

/// Computes `ceil(k * n^{1/k})` exactly in integer arithmetic.
///
/// Used to size the sparse masking polynomial in the zk variant
/// (Appendix D, Lemmas 4 and 5): the number of non-zero coefficients
/// needed for hiding is `k * N^{1/k}` and this helper rounds up without
/// resorting to floating-point roots.
pub fn ceil_k_root_scaled(n: u128, k: u32) -> u128 {
    debug_assert!(k > 0, "k must be >= 1");
    if n == 0 {
        return 0;
    }
    if k == 1 {
        return n;
    }

    // Floor k-th root of n (u128), by integer binary search.
    let r_floor = kth_root_floor_u128(n, k);

    // Search m in [k*r_floor, k*(r_floor+1)] s.t. m is the smallest with (m/k)^k >=
    // n. Equivalently: m^k >= n * k^k.
    let lo = (r_floor).saturating_mul(k as u128);
    let hi = ((r_floor + 1) as u128).saturating_mul(k as u128);

    let target = BigUint::from(n) * pow_big(&BigUint::from(k as u128), k);
    let mut l = BigUint::from(lo);
    let mut r = BigUint::from(hi);
    let one = BigUint::one();

    while l < r {
        let mid = (&l + &r) >> 1; // integer mid
        let lhs = pow_big(&mid, k); // mid^k
        if lhs >= target {
            r = mid; // feasible
        } else {
            l = &mid + &one; // infeasible
        }
    }
    l.to_u128().expect("result does not fit in u128")
}

/// `floor(n^{1/k})` for `u128` via binary search.
fn kth_root_floor_u128(n: u128, k: u32) -> u128 {
    if n <= 1 {
        return n;
    }
    let mut lo: u128 = 1;
    let mut hi: u128 = n; // 128 iterations worst-case

    let mut ans = 1;
    while lo <= hi {
        let mid = lo + ((hi - lo) >> 1);
        if pow_le_u128(mid, k, n) {
            ans = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    ans
}

/// Returns true iff `x^k <= n`, computed without overflow (early exit).
fn pow_le_u128(x: u128, k: u32, n: u128) -> bool {
    if k == 0 {
        return 1 <= n;
    }
    let mut acc: u128 = 1;
    for _ in 0..k {
        // Early stop if acc*x would exceed n
        if x != 0 && acc > n / x {
            return false;
        }
        acc *= x;
    }
    acc <= n
}

/// BigUint pow by repeated squaring.
fn pow_big(x: &BigUint, mut k: u32) -> BigUint {
    let mut base = x.clone();
    let mut acc = BigUint::one();
    while k > 0 {
        if (k & 1) == 1 {
            acc *= &base;
        }
        if k > 1 {
            // Avoid simultaneous mutable and immutable borrow of base
            base = &base * &base;
        }
        k >>= 1;
    }
    acc
}
