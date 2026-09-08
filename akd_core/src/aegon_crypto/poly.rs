// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use ark_ff::Field;
use ark_poly::{
    DenseMultilinearExtension, MultilinearExtension, Polynomial, SparseMultilinearExtension,
};
use ark_serialize::{SerializationError, Valid};
use ark_std::{
    fmt::Debug,
    ops::{Add, AddAssign, Index, Neg, Sub, SubAssign},
    rand::Rng,
    Zero,
};
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum DenseOrSparseMLE<F: Field> {
    Dense(DenseMultilinearExtension<F>),
    Sparse(SparseMultilinearExtension<F>),
}

/// Borrowing counterpart to [`DenseOrSparseMLE`] used on the open path.
///
/// `DenseOrSparseMLE::Sparse(SparseMLE)` takes the polynomial by value,
/// which forces every `P::open(&DenseOrSparseMLE::Sparse(poly.clone()),
/// ...)` call site to deep-clone the underlying `BTreeMap` just to wrap
/// it in the enum. At publish-time that clone dominated the opening
/// phase (5 opens × labels × full-poly clone). This variant lets the
/// trait method take a borrowed view instead.
///
/// Copy because it's two pointers' worth — no allocation, no clone.
#[derive(Copy, Clone)]
pub enum DenseOrSparseMLERef<'a, F: Field> {
    Dense(&'a DenseMultilinearExtension<F>),
    Sparse(&'a SparseMultilinearExtension<F>),
}

impl<'a, F: Field> From<&'a DenseOrSparseMLE<F>> for DenseOrSparseMLERef<'a, F> {
    fn from(owned: &'a DenseOrSparseMLE<F>) -> Self {
        match owned {
            DenseOrSparseMLE::Dense(d) => DenseOrSparseMLERef::Dense(d),
            DenseOrSparseMLE::Sparse(s) => DenseOrSparseMLERef::Sparse(s),
        }
    }
}

impl<'a, F: Field> DenseOrSparseMLERef<'a, F> {
    pub fn num_vars(&self) -> usize {
        match self {
            DenseOrSparseMLERef::Dense(d) => d.num_vars,
            DenseOrSparseMLERef::Sparse(s) => s.num_vars,
        }
    }
}

impl<F: Field> DenseOrSparseMLE<F> {
    pub fn rand(num_vars: usize, rng: &mut impl Rng) -> Self {
        DenseOrSparseMLE::Sparse(SparseMultilinearExtension::rand(num_vars, rng))
    }

    /// Cheap reborrow into a [`DenseOrSparseMLERef`] for passing to
    /// `P::open` without consuming `self`.
    pub fn as_ref(&self) -> DenseOrSparseMLERef<'_, F> {
        DenseOrSparseMLERef::from(self)
    }

    pub fn to_dense(&self) -> DenseMultilinearExtension<F> {
        match self {
            DenseOrSparseMLE::Dense(mle) => mle.clone(),
            DenseOrSparseMLE::Sparse(mle) => mle.to_dense_multilinear_extension(),
        }
    }
    pub fn to_sparse(&self) -> SparseMultilinearExtension<F> {
        match self {
            DenseOrSparseMLE::Dense(_) => {
                panic!("Cannot convert Dense Multilinear Extension to Sparse")
            }
            DenseOrSparseMLE::Sparse(mle) => mle.clone(),
        }
    }
}

impl<F: Field> ark_serialize::CanonicalSerialize for DenseOrSparseMLE<F> {
    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        1 + match self {
            DenseOrSparseMLE::Dense(mle) => mle.serialized_size(compress),
            DenseOrSparseMLE::Sparse(mle) => mle.serialized_size(compress),
        }
    }

    fn serialize_with_mode<W: ark_std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        match self {
            DenseOrSparseMLE::Dense(mle) => {
                writer.write_all(&[0])?;
                mle.serialize_with_mode(writer, compress)
            }
            DenseOrSparseMLE::Sparse(mle) => {
                writer.write_all(&[1])?;
                mle.serialize_with_mode(writer, compress)
            }
        }
    }
}

impl<F: Field> ark_serialize::CanonicalDeserialize for DenseOrSparseMLE<F> {
    fn deserialize_with_mode<R: ark_std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let mut tag = [0u8; 1];
        reader.read_exact(&mut tag)?;
        match tag[0] {
            0 => Ok(DenseOrSparseMLE::Dense(
                DenseMultilinearExtension::deserialize_with_mode(reader, compress, validate)?,
            )),
            1 => Ok(DenseOrSparseMLE::Sparse(
                SparseMultilinearExtension::deserialize_with_mode(reader, compress, validate)?,
            )),
            _ => Err(SerializationError::InvalidData),
        }
    }
}
impl<F: Field> Valid for DenseOrSparseMLE<F> {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        match self {
            DenseOrSparseMLE::Dense(mle) => mle.check(),
            DenseOrSparseMLE::Sparse(mle) => mle.check(),
        }
    }
}

impl<F: Field> MultilinearExtension<F> for DenseOrSparseMLE<F> {
    fn num_vars(&self) -> usize {
        match self {
            DenseOrSparseMLE::Dense(mle) => mle.num_vars(),
            DenseOrSparseMLE::Sparse(mle) => mle.num_vars(),
        }
    }

    /// Outputs an `l`-variate multilinear extension where value of evaluations
    /// are sampled uniformly at random. The number of nonzero entries is
    /// `sqrt(2^num_vars)` and indices of those nonzero entries are distributed
    /// uniformly at random.
    fn rand<R: Rng>(num_vars: usize, rng: &mut R) -> Self {
        Self::Sparse(SparseMultilinearExtension::rand(num_vars, rng))
    }

    fn relabel(&self, a: usize, b: usize, k: usize) -> Self {
        match self {
            DenseOrSparseMLE::Dense(mle) => DenseOrSparseMLE::Dense(mle.relabel(a, b, k)),
            DenseOrSparseMLE::Sparse(mle) => DenseOrSparseMLE::Sparse(mle.relabel(a, b, k)),
        }
    }

    fn fix_variables(&self, partial_point: &[F]) -> Self {
        match self {
            DenseOrSparseMLE::Dense(mle) => {
                DenseOrSparseMLE::Dense(mle.fix_variables(partial_point))
            }
            DenseOrSparseMLE::Sparse(mle) => {
                DenseOrSparseMLE::Sparse(mle.fix_variables(partial_point))
            }
        }
    }

    fn to_evaluations(&self) -> Vec<F> {
        match self {
            DenseOrSparseMLE::Dense(mle) => mle.to_evaluations(),
            DenseOrSparseMLE::Sparse(mle) => mle.to_evaluations(),
        }
    }
}

impl<F: Field> Index<usize> for DenseOrSparseMLE<F> {
    type Output = F;

    /// Returns the evaluation of the polynomial at a point represented by
    /// index.
    ///
    /// Index represents a vector in {0,1}^`num_vars` in little endian form. For
    /// example, `0b1011` represents `P(1,1,0,1)`
    ///
    /// For Sparse multilinear polynomial, Lookup_evaluation takes log time to
    /// the size of polynomial.
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            DenseOrSparseMLE::Dense(mle) => &mle[index],
            DenseOrSparseMLE::Sparse(mle) => &mle[index],
        }
    }
}

impl<F: Field> Polynomial<F> for DenseOrSparseMLE<F> {
    type Point = Vec<F>;

    fn degree(&self) -> usize {
        match self {
            DenseOrSparseMLE::Dense(mle) => mle.degree(),
            DenseOrSparseMLE::Sparse(mle) => mle.degree(),
        }
    }

    fn evaluate(&self, point: &Self::Point) -> F {
        match self {
            DenseOrSparseMLE::Dense(mle) => mle.evaluate(point),
            DenseOrSparseMLE::Sparse(mle) => mle.evaluate(point),
        }
    }
}

impl<F: Field> Add for DenseOrSparseMLE<F> {
    type Output = DenseOrSparseMLE<F>;

    fn add(self, other: DenseOrSparseMLE<F>) -> Self {
        match (self, other) {
            (DenseOrSparseMLE::Dense(lhs), DenseOrSparseMLE::Dense(rhs)) => {
                DenseOrSparseMLE::Dense(lhs + rhs)
            }
            (DenseOrSparseMLE::Sparse(lhs), DenseOrSparseMLE::Sparse(rhs)) => {
                DenseOrSparseMLE::Sparse(lhs + rhs)
            }
            _ => {
                panic!("Cannot add Dense and Sparse Multilinear Extensions together");
            }
        }
    }
}

impl<'a, F: Field> Add<&'a DenseOrSparseMLE<F>> for &DenseOrSparseMLE<F> {
    type Output = DenseOrSparseMLE<F>;

    fn add(self, rhs: &'a DenseOrSparseMLE<F>) -> Self::Output {
        match (self, rhs) {
            (DenseOrSparseMLE::Dense(lhs), DenseOrSparseMLE::Dense(rhs)) => {
                DenseOrSparseMLE::Dense(lhs + rhs)
            }
            (DenseOrSparseMLE::Sparse(lhs), DenseOrSparseMLE::Sparse(rhs)) => {
                DenseOrSparseMLE::Sparse(lhs + rhs)
            }
            _ => {
                panic!("Cannot add Dense and Sparse Multilinear Extensions together");
            }
        }
    }
}

impl<F: Field> AddAssign for DenseOrSparseMLE<F> {
    fn add_assign(&mut self, other: Self) {
        match (self, other) {
            (DenseOrSparseMLE::Dense(lhs), DenseOrSparseMLE::Dense(rhs)) => {
                lhs.add_assign(rhs);
            }
            (DenseOrSparseMLE::Sparse(lhs), DenseOrSparseMLE::Sparse(rhs)) => {
                lhs.add_assign(rhs);
            }
            _ => {
                panic!("Cannot add Dense and Sparse Multilinear Extensions together");
            }
        }
    }
}

impl<'a, F: Field> AddAssign<&'a DenseOrSparseMLE<F>> for DenseOrSparseMLE<F> {
    fn add_assign(&mut self, other: &'a DenseOrSparseMLE<F>) {
        match (self, other) {
            (DenseOrSparseMLE::Dense(lhs), DenseOrSparseMLE::Dense(rhs)) => {
                lhs.add_assign(rhs);
            }
            (DenseOrSparseMLE::Sparse(lhs), DenseOrSparseMLE::Sparse(rhs)) => {
                lhs.add_assign(rhs);
            }
            _ => {
                panic!("Cannot add Dense and Sparse Multilinear Extensions together");
            }
        }
    }
}

impl<'a, F: Field> AddAssign<(F, &'a DenseOrSparseMLE<F>)> for DenseOrSparseMLE<F> {
    fn add_assign(&mut self, (f, other): (F, &'a DenseOrSparseMLE<F>)) {
        match (self, other) {
            (DenseOrSparseMLE::Dense(lhs), DenseOrSparseMLE::Dense(rhs)) => {
                lhs.add_assign((f, rhs));
            }
            (DenseOrSparseMLE::Sparse(lhs), DenseOrSparseMLE::Sparse(rhs)) => {
                lhs.add_assign((f, rhs));
            }
            _ => {
                panic!("Cannot add Dense and Sparse Multilinear Extensions together");
            }
        }
    }
}

impl<F: Field> Neg for DenseOrSparseMLE<F> {
    type Output = DenseOrSparseMLE<F>;

    fn neg(self) -> Self::Output {
        match self {
            DenseOrSparseMLE::Dense(mle) => DenseOrSparseMLE::Dense(-mle),
            DenseOrSparseMLE::Sparse(mle) => DenseOrSparseMLE::Sparse(-mle),
        }
    }
}

impl<F: Field> Sub for DenseOrSparseMLE<F> {
    type Output = DenseOrSparseMLE<F>;

    fn sub(self, other: DenseOrSparseMLE<F>) -> Self {
        match (self, other) {
            (DenseOrSparseMLE::Dense(lhs), DenseOrSparseMLE::Dense(rhs)) => {
                DenseOrSparseMLE::Dense(lhs - rhs)
            }
            (DenseOrSparseMLE::Sparse(lhs), DenseOrSparseMLE::Sparse(rhs)) => {
                DenseOrSparseMLE::Sparse(lhs - rhs)
            }
            _ => {
                panic!("Cannot subtract Dense and Sparse Multilinear Extensions together");
            }
        }
    }
}

impl<'a, F: Field> Sub<&'a DenseOrSparseMLE<F>> for &DenseOrSparseMLE<F> {
    type Output = DenseOrSparseMLE<F>;

    fn sub(self, rhs: &'a DenseOrSparseMLE<F>) -> Self::Output {
        match (self, rhs) {
            (DenseOrSparseMLE::Dense(lhs), DenseOrSparseMLE::Dense(rhs)) => {
                DenseOrSparseMLE::Dense(lhs - rhs)
            }
            (DenseOrSparseMLE::Sparse(lhs), DenseOrSparseMLE::Sparse(rhs)) => {
                DenseOrSparseMLE::Sparse(lhs - rhs)
            }
            _ => {
                panic!("Cannot subtract Dense and Sparse Multilinear Extensions together");
            }
        }
    }
}

impl<F: Field> SubAssign for DenseOrSparseMLE<F> {
    fn sub_assign(&mut self, other: Self) {
        match (self, other) {
            (DenseOrSparseMLE::Dense(lhs), DenseOrSparseMLE::Dense(rhs)) => {
                lhs.sub_assign(rhs);
            }
            (DenseOrSparseMLE::Sparse(lhs), DenseOrSparseMLE::Sparse(rhs)) => {
                lhs.sub_assign(rhs);
            }
            _ => {
                panic!("Cannot subtract Dense and Sparse Multilinear Extensions together");
            }
        }
    }
}

impl<'a, F: Field> SubAssign<&'a DenseOrSparseMLE<F>> for DenseOrSparseMLE<F> {
    fn sub_assign(&mut self, other: &'a DenseOrSparseMLE<F>) {
        match (self, other) {
            (DenseOrSparseMLE::Dense(lhs), DenseOrSparseMLE::Dense(rhs)) => {
                lhs.sub_assign(rhs);
            }
            (DenseOrSparseMLE::Sparse(lhs), DenseOrSparseMLE::Sparse(rhs)) => {
                lhs.sub_assign(rhs);
            }
            _ => {
                panic!("Cannot subtract Dense and Sparse Multilinear Extensions together");
            }
        }
    }
}

impl<F: Field> Zero for DenseOrSparseMLE<F> {
    fn zero() -> Self {
        DenseOrSparseMLE::Sparse(SparseMultilinearExtension::zero())
    }

    fn is_zero(&self) -> bool {
        match self {
            DenseOrSparseMLE::Dense(mle) => mle.is_zero(),
            DenseOrSparseMLE::Sparse(mle) => mle.is_zero(),
        }
    }
}

impl<F: Field> Debug for DenseOrSparseMLE<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenseOrSparseMLE::Dense(mle) => write!(f, "Dense({:?})", mle),
            DenseOrSparseMLE::Sparse(mle) => write!(f, "Sparse({:?})", mle),
        }
    }
}
