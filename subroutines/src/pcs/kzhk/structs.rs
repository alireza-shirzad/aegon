use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};

use crate::poly::DenseOrSparseMLE;
use ark_serialize::{
    self, CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid,
    Validate, Write,
};
use ark_ff::One;
use ark_std::{cfg_into_iter, cfg_iter, cfg_iter_mut, ops::Sub, Zero};
use derivative::Derivative;
use ndarray::{ArrayD, IxDyn};
#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use std::collections::BTreeMap;
use std::ops::{Add, Deref, DerefMut, Range};

/// Configuration for the KZH-k scheme.
///
/// - `k`: number of blocks the polynomial variables are split across
///   (see the module-level docs in `mod.rs` for the proof-size/cost
///   trade-off).
/// - `zk`: whether to instantiate the hiding/zero-knowledge variant
///   from Appendix D. When `false`, the plain KZH-k of Figure 14 is
///   used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KZHKConfig {
    pub k: usize,
    pub zk: bool,
}

impl KZHKConfig {
    /// Constructs a new configuration with the given `k` and zk flag.
    pub fn new(k: usize, zk: bool) -> Self {
        Self { k, zk }
    }
}

///////////////// Commitment //////////////////////

#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Copy(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
/// A KZH-k commitment `C = <f, H_1>` (Figure 14, Commit), or its
/// hiding variant `C + tau*h` (Appendix D) when the SRS is zk.
///
/// - `com`: the `G1` group element itself.
/// - `nv`: the number of variables of the committed polynomial,
///   retained so that addition of commitments can sanity-check
///   compatibility.
pub struct KZHKCommitment<E: Pairing> {
    com: E::G1Affine,
    nv: usize,
}
impl<E: Pairing> Add for KZHKCommitment<E> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        debug_assert_eq!(self.nv, other.nv, "commitments for different nv!");
        let com = (self.com + other.com).into_affine();
        KZHKCommitment::new(com, self.nv)
    }
}

impl<E: Pairing> Sub for KZHKCommitment<E> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        debug_assert_eq!(self.nv, other.nv, "commitments for different nv!");
        let com = (self.com - other.com).into_affine();
        KZHKCommitment::new(com, self.nv)
    }
}

impl<'b, E: Pairing> Add<&'b KZHKCommitment<E>> for &KZHKCommitment<E> {
    type Output = KZHKCommitment<E>;

    fn add(self, rhs: &'b KZHKCommitment<E>) -> Self::Output {
        debug_assert_eq!(self.nv, rhs.nv, "commitments for different nv!");
        let com = (self.com + rhs.com).into_affine();
        KZHKCommitment::new(com, self.nv)
    }
}

impl<'b, E: Pairing> Sub<&'b KZHKCommitment<E>> for &KZHKCommitment<E> {
    type Output = KZHKCommitment<E>;

    fn sub(self, rhs: &'b KZHKCommitment<E>) -> Self::Output {
        debug_assert_eq!(self.nv, rhs.nv, "commitments for different nv!");
        let com = (self.com - rhs.com).into_affine();
        KZHKCommitment::new(com, self.nv)
    }
}

impl<E: Pairing> KZHKCommitment<E> {
    /// Create a new commitment
    pub fn new(com: E::G1Affine, nv: usize) -> Self {
        Self { com, nv }
    }

    /// Get the commitment
    pub fn get_commitment(&self) -> E::G1Affine {
        self.com
    }

    /// Get the number of variables
    pub fn get_num_vars(&self) -> usize {
        self.nv
    }
}

////////////// Prover state /////////////////

/// One row of the Boolean auxiliary table.
///
/// Logically a `Vec<G1Affine>` of length `dj_size = 2^{d_1+...+d_j}`, but
/// stored either as a flat dense vector (the default for dense input
/// polynomials) or as a `BTreeMap` of just the non-zero positions (for
/// sparse input polynomials whose aux row would otherwise be
/// overwhelmingly empty). Positions absent from a `Sparse` row are
/// implicitly the affine zero (`G1Affine::zero()`).
///
/// The `Sparse` variant is the optimisation used by `update_state_sparse`
/// when `nnz << dj_size`: it avoids the `O(dj_size)` allocation,
/// per-empty-bucket `G1::zero()` write, and `normalize_batch` over the
/// full row. The opening side (`open_*_bool_inner`) then extracts the
/// proof's `2^{d_j}`-sized slice via `O(2^{d_j})` map lookups instead of
/// slice indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxRow<E: Pairing> {
    Dense(Vec<E::G1Affine>),
    Sparse {
        len: usize,
        entries: BTreeMap<usize, E::G1Affine>,
    },
}

// Manual `CanonicalSerialize`/`CanonicalDeserialize`/`Valid` impls.
// The arkworks derive macro panics on enums with `BTreeMap` and a
// pairing-bound generic; we encode by-hand instead. Format:
//   * Dense:  tag = 0u8, then the inner `Vec<G1Affine>`.
//   * Sparse: tag = 1u8, then `len` (u64), then the entries as
//             `Vec<(u64, G1Affine)>`.
// The encoding is non-canonical (a `Sparse` row that happens to be
// fully populated round-trips to the same `Sparse`, not to `Dense`),
// but it's only used as opaque transport for `KZHKState`, which we
// never compare across the boundary.
impl<E: Pairing> CanonicalSerialize for AuxRow<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            AuxRow::Dense(v) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                v.serialize_with_mode(&mut writer, compress)?;
            }
            AuxRow::Sparse { len, entries } => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                (*len as u64).serialize_with_mode(&mut writer, compress)?;
                let pairs: Vec<(u64, E::G1Affine)> =
                    entries.iter().map(|(&k, &v)| (k as u64, v)).collect();
                pairs.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            AuxRow::Dense(v) => 0u8.serialized_size(compress) + v.serialized_size(compress),
            AuxRow::Sparse { len, entries } => {
                let pairs: Vec<(u64, E::G1Affine)> =
                    entries.iter().map(|(&k, &v)| (k as u64, v)).collect();
                1u8.serialized_size(compress)
                    + (*len as u64).serialized_size(compress)
                    + pairs.serialized_size(compress)
            }
        }
    }
}

impl<E: Pairing> Valid for AuxRow<E> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            AuxRow::Dense(v) => v.check(),
            AuxRow::Sparse { entries, .. } => {
                for (_, p) in entries {
                    p.check()?;
                }
                Ok(())
            }
        }
    }
}

impl<E: Pairing> CanonicalDeserialize for AuxRow<E> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match tag {
            0 => {
                let v =
                    Vec::<E::G1Affine>::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(AuxRow::Dense(v))
            }
            1 => {
                let len = u64::deserialize_with_mode(&mut reader, compress, validate)? as usize;
                let pairs = Vec::<(u64, E::G1Affine)>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?;
                let entries: BTreeMap<usize, E::G1Affine> =
                    pairs.into_iter().map(|(k, v)| (k as usize, v)).collect();
                Ok(AuxRow::Sparse { len, entries })
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl<E: Pairing> AuxRow<E> {
    /// Logical length of the row (matches the dense `dj_size`).
    pub fn len(&self) -> usize {
        match self {
            AuxRow::Dense(v) => v.len(),
            AuxRow::Sparse { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read position `i`. Returns the affine zero for sparse rows when
    /// the position has no stored entry.
    #[inline]
    pub fn get(&self, i: usize) -> E::G1Affine {
        match self {
            AuxRow::Dense(v) => v[i],
            AuxRow::Sparse { entries, .. } => {
                entries.get(&i).copied().unwrap_or_else(E::G1Affine::zero)
            },
        }
    }

    /// Materialize the slice `[range.start, range.end)` as a `Vec<G1Affine>`.
    /// Used by the open paths to build the proof's `D_j` vector. Length of
    /// the returned vec is `range.end - range.start`.
    pub fn slice(&self, range: Range<usize>) -> Vec<E::G1Affine> {
        match self {
            AuxRow::Dense(v) => v[range].to_vec(),
            AuxRow::Sparse { entries, .. } => {
                let len = range.end - range.start;
                let mut out = vec![E::G1Affine::zero(); len];
                for (&k, &v) in entries.range(range.clone()) {
                    out[k - range.start] = v;
                }
                out
            },
        }
    }

    /// Force the row into the dense representation. Used by the
    /// `Add`/`Sub` impls on `KZHKState` when one operand is sparse and
    /// the other is dense — combining them homogeneously is simpler
    /// than maintaining a separate Sparse + Sparse code path that
    /// has to be careful about implicit zeros.
    pub fn into_dense(self) -> Vec<E::G1Affine> {
        match self {
            AuxRow::Dense(v) => v,
            AuxRow::Sparse { len, entries } => {
                let mut v = vec![E::G1Affine::zero(); len];
                for (k, val) in entries {
                    v[k] = val;
                }
                v
            },
        }
    }

    /// Pairwise combine two rows under `op` (used by the `Add`/`Sub`
    /// impls). `Dense + Dense` stays dense; everything else densifies
    /// first for simplicity.
    fn pairwise<F>(self, rhs: Self, op: F) -> Self
    where
        F: Fn(E::G1Affine, E::G1Affine) -> E::G1Affine + Sync,
    {
        let lhs = self.into_dense();
        let rhs = rhs.into_dense();
        assert_eq!(lhs.len(), rhs.len(), "AuxRow: length mismatch in combine");
        let out: Vec<E::G1Affine> =
            cfg_iter!(lhs).zip(cfg_iter!(rhs)).map(|(&a, &b)| op(a, b)).collect();
        AuxRow::Dense(out)
    }
}

/// Prover-side state attached to a commitment.
///
/// - `d_bool`: the table of precomputed row-commitments
///   `aux_{b_1,...,b_j} = <f(b_1,...,b_j, X_{j+1},...), H_{j+1}>`
///   for levels `j = 1..k-1` (Figure 14). Each [`AuxRow`] holds the
///   `2^{d_1+...+d_j}` auxiliary group elements for level `j` in
///   little-endian ordering. This table powers the free Boolean
///   opening: when the query point is Boolean, the level-`j` proof
///   vector `D_j` is a contiguous slice of `d_bool[j]`. The row may
///   be stored sparsely (see [`AuxRow::Sparse`]) when the input
///   polynomial is sparse enough that allocating a dense `dj_size`-
///   length vector would be wasteful.
/// - `tau`: the hiding scalar sampled by [`crate::pcs::kzhk::KZHK::commit`]
///   when the SRS is zk (Appendix D). Retained so the prover can
///   derandomize during opening.
/// - `sparsity`: upper bound on the number of non-zero coefficients of the
///   committed polynomial, recorded at commit time. For a dense polynomial
///   this is `2^num_vars`; for a sparse polynomial it's the size of its
///   non-zero coefficient map. `update_state` reads this to decide whether
///   the per-chunk / per-bucket MSMs are small enough (≤ the naive-MSM
///   threshold in `msm.rs`) that the outer loop can be parallelized
///   without triggering nested rayon pool builds inside arkworks'
///   Pippenger.
#[derive(Debug, Derivative, CanonicalSerialize, CanonicalDeserialize, Clone, PartialEq, Eq)]
pub struct KZHKState<E: Pairing> {
    tau: Option<E::ScalarField>,
    d_bool: Option<Vec<AuxRow<E>>>,
    sparsity: Option<usize>,
}

impl<E: Pairing> KZHKState<E> {
    /// Create a new prover state.
    pub fn new(
        tau: Option<E::ScalarField>,
        d_bool: Option<Vec<AuxRow<E>>>,
        sparsity: Option<usize>,
    ) -> Self {
        Self { tau, d_bool, sparsity }
    }

    /// Borrow the Boolean auxiliary table `d_bool`.
    pub fn get_d_bool(&self) -> &Vec<AuxRow<E>> {
        self.d_bool.as_ref().unwrap()
    }

    /// Borrow the hiding scalar `tau` (zk variant only).
    pub fn get_tau(&self) -> &E::ScalarField {
        self.tau.as_ref().unwrap()
    }

    /// Upper bound on the committed polynomial's non-zero count, or `None`
    /// if it wasn't recorded.
    pub fn get_sparsity(&self) -> Option<usize> {
        self.sparsity
    }

    pub fn set_d_bool(&mut self, d_bool: Vec<AuxRow<E>>) {
        self.d_bool = Some(d_bool);
    }
}

impl<E: Pairing> Default for KZHKState<E> {
    fn default() -> Self {
        KZHKState {
            d_bool: None,
            tau: None,
            sparsity: None,
        }
    }
}

impl<E: Pairing> Add for KZHKState<E> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self == KZHKState::default() {
            return rhs;
        }
        if rhs == KZHKState::default() {
            return self;
        }
        let lhs_rows = self.d_bool.unwrap();
        let rhs_rows = rhs.d_bool.unwrap();
        assert_eq!(
            lhs_rows.len(),
            rhs_rows.len(),
            "Auxiliary information must have the same length"
        );
        let out_d_bool: Vec<AuxRow<E>> = lhs_rows
            .into_iter()
            .zip(rhs_rows.into_iter())
            .map(|(ra, rb)| ra.pairwise(rb, |x, y| (x + y).into_affine()))
            .collect();
        KZHKState {
            d_bool: Some(out_d_bool),
            tau: None,
            sparsity: None,
        }
    }
}

impl<E: Pairing> Sub for KZHKState<E> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self == KZHKState::default() {
            return rhs;
        }
        if rhs == KZHKState::default() {
            return self;
        }
        let lhs_rows = self.d_bool.unwrap();
        let rhs_rows = rhs.d_bool.unwrap();
        assert_eq!(
            lhs_rows.len(),
            rhs_rows.len(),
            "Auxiliary information must have the same length"
        );
        let out_d_bool: Vec<AuxRow<E>> = lhs_rows
            .into_iter()
            .zip(rhs_rows.into_iter())
            .map(|(ra, rb)| ra.pairwise(rb, |x, y| (x - y).into_affine()))
            .collect();
        KZHKState {
            d_bool: Some(out_d_bool),
            tau: None,
            sparsity: None,
        }
    }
}

///////////// Opening Proof /////////////////

/// KZH-k opening proof `pi = ({D_j}_{j=1}^{k-1}, f_{x_1..x_{k-1}})`
/// (Figure 14) extended with the Sigma-protocol transcript used by the
/// zero-knowledge variant from Appendix D.
///
/// - `d`: per-level row-commitment vectors `D_j` that the verifier
///   checks pairwise against the previous level's commitment.
/// - `f`: the tail polynomial `f_{x_1..x_{k-1}}` — i.e. `f` partially
///   evaluated at the first `k-1` block sub-points. In the last step
///   the verifier opens it at `x_k` to obtain the claimed value.
/// - `r_hide`, `y_r`, `rho_prime`: Sigma-protocol components of the
///   zk opener. `r_hide` is the commitment to the sparse masking
///   polynomial `r(X)`, `y_r = r(x)`, and
///   `rho_prime = alpha*tau + rho` is the derandomizing scalar that
///   lets the verifier linearize away the hiding bases. All three are
///   `None` for plain openings.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct KZHKOpeningProof<E: Pairing> {
    d: Vec<Vec<E::G1Affine>>,
    f: DenseOrSparseMLE<E::ScalarField>,
    r_hide: Option<KZHKCommitment<E>>,
    y_r: Option<E::ScalarField>,
    rho_prime: Option<E::ScalarField>,
}

impl<E: Pairing> KZHKOpeningProof<E> {
    /// Create a new opening proof
    pub fn new(
        d: Vec<Vec<E::G1Affine>>,
        f: DenseOrSparseMLE<E::ScalarField>,
        r_hide: Option<KZHKCommitment<E>>,
        y_r: Option<E::ScalarField>,
        rho_prime: Option<E::ScalarField>,
    ) -> Self {
        Self {
            d,
            f,
            r_hide,
            y_r,
            rho_prime,
        }
    }

    /// Get the evaluation of quotients
    pub fn get_d(&self) -> &Vec<Vec<E::G1Affine>> {
        &self.d
    }

    /// Get the opening proof
    pub fn get_f(&self) -> &DenseOrSparseMLE<E::ScalarField> {
        &self.f
    }

    pub fn get_r_hide(&self) -> &Option<KZHKCommitment<E>> {
        &self.r_hide
    }

    /// Get the y_r value
    pub fn get_y_r(&self) -> &Option<E::ScalarField> {
        &self.y_r
    }

    /// Get the rho_prime value
    pub fn get_rho_prime(&self) -> &Option<E::ScalarField> {
        &self.rho_prime
    }

    pub fn set_rho_prime(&mut self, rho_prime: E::ScalarField) {
        self.rho_prime = Some(rho_prime);
    }

    pub fn set_y_r(&mut self, y_r: E::ScalarField) {
        self.y_r = Some(y_r);
    }

    pub fn set_r_hide(&mut self, r_hide: KZHKCommitment<E>) {
        self.r_hide = Some(r_hide);
    }
}

impl<E: Pairing> Default for KZHKOpeningProof<E> {
    fn default() -> Self {
        KZHKOpeningProof {
            d: vec![],
            f: DenseOrSparseMLE::zero(),
            r_hide: None,
            y_r: None,
            rho_prime: None,
        }
    }
}

/// Batch-normalize a `Vec<Vec<G1>>` into `Vec<Vec<G1Affine>>` with one
/// Montgomery batch inversion across every entry, instead of one
/// inversion per entry. Preserves row shape.
fn batch_normalize_rows<E: Pairing>(
    proj_rows: Vec<Vec<E::G1>>,
) -> Vec<Vec<E::G1Affine>> {
    let row_lens: Vec<usize> = proj_rows.iter().map(|r| r.len()).collect();
    let total: usize = row_lens.iter().sum();
    if total == 0 {
        return row_lens.into_iter().map(|_| Vec::new()).collect();
    }
    let flat: Vec<E::G1> = proj_rows.into_iter().flatten().collect();
    let flat_aff = <E::G1 as CurveGroup>::normalize_batch(&flat);
    let mut out: Vec<Vec<E::G1Affine>> = Vec::with_capacity(row_lens.len());
    let mut idx = 0;
    for len in row_lens {
        out.push(flat_aff[idx..idx + len].to_vec());
        idx += len;
    }
    out
}

impl<E: Pairing> core::ops::Mul<E::ScalarField> for KZHKOpeningProof<E> {
    type Output = Self;

    fn mul(self, rhs: E::ScalarField) -> Self::Output {
        if rhs.is_zero() {
            return Self::default();
        }
        if self == Self::default() {
            return self;
        }
        if rhs.is_one() {
            return self;
        }
        let proj_rows: Vec<Vec<E::G1>> = cfg_into_iter!(self.d)
            .map(|row| row.into_iter().map(|x| x * rhs).collect())
            .collect();
        let out_d = batch_normalize_rows::<E>(proj_rows);
        let mut f_out = self.f;
        mul_poly_by_cnst_in_place(&mut f_out, rhs);
        KZHKOpeningProof {
            d: out_d,
            f: f_out,
            r_hide: None,
            y_r: None,
            rho_prime: None,
        }
    }
}

impl<'a, E: Pairing> core::ops::Mul<E::ScalarField> for &'a KZHKOpeningProof<E> {
    type Output = KZHKOpeningProof<E>;

    fn mul(self, rhs: E::ScalarField) -> Self::Output {
        if rhs.is_zero() {
            return KZHKOpeningProof::default();
        }
        if rhs.is_one() {
            return self.clone();
        }
        let proj_rows: Vec<Vec<E::G1>> = cfg_iter!(self.d)
            .map(|row| row.iter().map(|x| *x * rhs).collect())
            .collect();
        let out_d = batch_normalize_rows::<E>(proj_rows);
        let mut f_out = self.f.clone();
        mul_poly_by_cnst_in_place(&mut f_out, rhs);
        KZHKOpeningProof {
            d: out_d,
            f: f_out,
            r_hide: None,
            y_r: None,
            rho_prime: None,
        }
    }
}

impl<E: Pairing> core::ops::MulAssign<E::ScalarField> for KZHKOpeningProof<E> {
    fn mul_assign(&mut self, rhs: E::ScalarField) {
        if rhs.is_zero() {
            self.d.clear();
            self.f = DenseOrSparseMLE::zero();
            return;
        }
        if rhs.is_one() {
            return;
        }
        let proj_rows: Vec<Vec<E::G1>> = cfg_iter!(self.d)
            .map(|row| row.iter().map(|x| *x * rhs).collect())
            .collect();
        self.d = batch_normalize_rows::<E>(proj_rows);
        mul_poly_by_cnst_in_place(&mut self.f, rhs);
    }
}

impl<E: Pairing> Add for KZHKOpeningProof<E> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self == KZHKOpeningProof::default() {
            return rhs;
        }
        if rhs == KZHKOpeningProof::default() {
            return self;
        }
        assert_eq!(
            self.d.len(),
            rhs.d.len(),
            "Auxiliary information must have the same length"
        );
        let zipped: Vec<(Vec<E::G1Affine>, Vec<E::G1Affine>)> =
            self.d.iter().cloned().zip(rhs.d.iter().cloned()).collect();
        let proj_rows: Vec<Vec<E::G1>> = cfg_into_iter!(zipped)
            .map(|(ra, rb)| {
                assert_eq!(ra.len(), rb.len(), "column count mismatch in a row");
                ra.into_iter()
                    .zip(rb.into_iter())
                    .map(|(x, y)| x + y)
                    .collect()
            })
            .collect();
        let out_d = batch_normalize_rows::<E>(proj_rows);
        if self.f == DenseOrSparseMLE::zero() {
            return KZHKOpeningProof {
                d: out_d,
                f: rhs.f,
                r_hide: None,
                y_r: None,
                rho_prime: None,
            };
        }

        if rhs.f == DenseOrSparseMLE::zero() {
            return KZHKOpeningProof {
                d: out_d,
                f: self.f,
                r_hide: None,
                y_r: None,
                rho_prime: None,
            };
        }

        let f_out = match (&self.f, &rhs.f) {
            (DenseOrSparseMLE::Dense(ref a), DenseOrSparseMLE::Dense(ref b)) => {
                DenseOrSparseMLE::Dense(a + b)
            },
            (DenseOrSparseMLE::Sparse(ref a), DenseOrSparseMLE::Sparse(ref b)) => {
                DenseOrSparseMLE::Sparse(a + b)
            },
            (DenseOrSparseMLE::Dense(ref a), DenseOrSparseMLE::Sparse(ref _b)) => {
                let densed_b = rhs.f.to_dense();
                DenseOrSparseMLE::Dense(a + &densed_b)
            },
            (DenseOrSparseMLE::Sparse(ref _a), DenseOrSparseMLE::Dense(ref b)) => {
                let densed_a = self.f.to_dense();
                DenseOrSparseMLE::Dense(&densed_a + b)
            },
        };

        KZHKOpeningProof {
            d: out_d,
            f: f_out,
            r_hide: None,
            y_r: None,
            rho_prime: None,
        }
    }
}

impl<E: Pairing> Sub for KZHKOpeningProof<E> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self == KZHKOpeningProof::default() {
            return rhs;
        }
        if rhs == KZHKOpeningProof::default() {
            return self;
        }
        assert_eq!(
            self.d.len(),
            rhs.d.len(),
            "Auxiliary information must have the same length"
        );
        let zipped: Vec<(Vec<E::G1Affine>, Vec<E::G1Affine>)> =
            self.d.iter().cloned().zip(rhs.d.iter().cloned()).collect();
        let proj_rows: Vec<Vec<E::G1>> = cfg_into_iter!(zipped)
            .map(|(ra, rb)| {
                assert_eq!(ra.len(), rb.len(), "column count mismatch in a row");
                ra.into_iter()
                    .zip(rb.into_iter())
                    .map(|(x, y)| {
                        let yp: E::G1 = y.into();
                        let xp: E::G1 = x.into();
                        xp - yp
                    })
                    .collect()
            })
            .collect();
        let out_d = batch_normalize_rows::<E>(proj_rows);
        let f_out = self.f - rhs.f;
        KZHKOpeningProof {
            d: out_d,
            f: f_out,
            r_hide: None,
            y_r: None,
            rho_prime: None,
        }
    }
}
///////////////// Tensor and implementation ///////////////////

/// Local newtype wrapper around `ndarray::ArrayD<T>` so we can implement
/// `CanonicalSerialize`/`CanonicalDeserialize` without violating the orphan
/// rules.
#[derive(Clone, Debug)]
pub struct Tensor<T>(pub ArrayD<T>);

impl<T> From<ArrayD<T>> for Tensor<T> {
    fn from(a: ArrayD<T>) -> Self {
        Tensor(a)
    }
}
impl<T> From<Tensor<T>> for ArrayD<T> {
    fn from(w: Tensor<T>) -> Self {
        w.0
    }
}
impl<T> Deref for Tensor<T> {
    type Target = ArrayD<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn product_u64(shape: &[usize]) -> Result<u64, SerializationError> {
    let mut acc: u128 = 1;
    for &d in shape {
        acc = acc
            .checked_mul(d as u128)
            .ok_or(SerializationError::InvalidData)?;
    }
    u64::try_from(acc).map_err(|_| SerializationError::InvalidData)
}

/// Iterator to walk all indices in row-major order for a given shape.
struct RowMajorIx {
    idx: Vec<usize>,
    shape: Vec<usize>,
    done: bool,
}
impl RowMajorIx {
    fn new(shape: &[usize]) -> Self {
        let k = shape.len();
        let done = shape.contains(&0);
        Self {
            idx: vec![0; k],
            shape: shape.to_vec(),
            done,
        }
    }
}
impl Iterator for RowMajorIx {
    type Item = IxDyn;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let out = IxDyn(&self.idx);
        for ax in (0..self.shape.len()).rev() {
            self.idx[ax] += 1;
            if self.idx[ax] < self.shape[ax] {
                break;
            } else {
                self.idx[ax] = 0;
                if ax == 0 {
                    self.done = true;
                }
            }
        }
        Some(out)
    }
}

impl<T: CanonicalSerialize> CanonicalSerialize for Tensor<T> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut w: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // rank
        let rank = u32::try_from(self.ndim()).map_err(|_| SerializationError::InvalidData)?;
        rank.serialize_with_mode(&mut w, compress)?;
        // shape
        for &d in self.shape() {
            let d64 = u64::try_from(d).map_err(|_| SerializationError::InvalidData)?;
            d64.serialize_with_mode(&mut w, compress)?;
        }
        // element count
        let n = product_u64(self.shape())?;
        n.serialize_with_mode(&mut w, compress)?;
        // elements in row-major order
        let shape = self.shape().to_vec();
        if self.is_standard_layout() {
            if let Some(slice) = self.as_slice_memory_order() {
                for t in slice {
                    t.serialize_with_mode(&mut w, compress)?;
                }
                return Ok(());
            }
        }
        for ix in RowMajorIx::new(&shape) {
            self[ix].serialize_with_mode(&mut w, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut sz = 0usize;
        sz += u32::default().serialized_size(compress);
        sz += self.shape().len() * u64::default().serialized_size(compress);
        sz += u64::default().serialized_size(compress);
        if self.is_standard_layout() {
            if let Some(slice) = self.as_slice_memory_order() {
                return sz
                    + slice
                        .iter()
                        .map(|t| t.serialized_size(compress))
                        .sum::<usize>();
            }
        }
        let shape = self.shape().to_vec();
        sz + RowMajorIx::new(&shape)
            .map(|ix| self[ix].serialized_size(compress))
            .sum::<usize>()
    }
}

impl<T: Valid> Valid for Tensor<T> {
    fn check(&self) -> Result<(), SerializationError> {
        // Check each element
        if self.is_standard_layout() {
            if let Some(slice) = self.as_slice_memory_order() {
                for t in slice {
                    t.check()?;
                }
                return Ok(());
            }
        }

        let shape = self.shape().to_vec();
        for ix in RowMajorIx::new(&shape) {
            self[ix].check()?;
        }
        Ok(())
    }
}

impl<T: Valid + CanonicalDeserialize> CanonicalDeserialize for Tensor<T> {
    fn deserialize_with_mode<R: Read>(
        mut r: R,
        compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let k = u32::deserialize_with_mode(&mut r, compress, Validate::No)?;
        let k = usize::try_from(k).map_err(|_| SerializationError::InvalidData)?;
        // shape
        let mut shape = Vec::with_capacity(k);
        for _ in 0..k {
            let d = u64::deserialize_with_mode(&mut r, compress, Validate::No)?;
            shape.push(usize::try_from(d).map_err(|_| SerializationError::InvalidData)?);
        }
        // element count check
        let n_hdr = u64::deserialize_with_mode(&mut r, compress, Validate::No)?;
        let n_calc = product_u64(&shape)?;
        if n_hdr != n_calc {
            return Err(SerializationError::InvalidData);
        }
        let n = usize::try_from(n_calc).map_err(|_| SerializationError::InvalidData)?;
        // elements in row-major order
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(T::deserialize_with_mode(&mut r, compress, Validate::No)?);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|_| SerializationError::InvalidData)?;
        Ok(Tensor(arr))
    }
}

fn mul_poly_by_cnst_in_place<F>(poly: &mut DenseOrSparseMLE<F>, c: F)
where
    F: ark_ff::Field,
{
    match poly {
        DenseOrSparseMLE::Dense(dense) => {
            cfg_iter_mut!(dense.evaluations).for_each(|x| {
                *x *= c;
            });
        },
        DenseOrSparseMLE::Sparse(sparse) => {
            cfg_iter_mut!(sparse.evaluations).for_each(|(_, x)| {
                *x *= c;
            });
        },
    }
}
