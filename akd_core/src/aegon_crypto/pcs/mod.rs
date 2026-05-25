mod errors;
pub mod kzhk;
pub mod prelude;
mod structs;

use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use errors::PCSError;
use std::{borrow::Borrow, default, fmt::Debug, hash::Hash};
use crate::aegon_crypto::poly::DenseOrSparseMLERef;
use crate::aegon_crypto::transcript::IOPTranscript;

/// This trait defines APIs for polynomial commitment schemes.
pub trait PolynomialCommitmentScheme<E: Pairing> {
    type Config: Clone + Debug + Default;
    type ProverParam: Clone + Sync;
    type VerifierParam: Clone + CanonicalSerialize + CanonicalDeserialize;
    type SRS: Clone + Debug + CanonicalSerialize + CanonicalDeserialize;
    type Polynomial: Clone + Debug + Hash + PartialEq + Eq;
    type Point: Clone + Ord + Debug + Sync + Hash + PartialEq + Eq;
    type Evaluation: Field;
    type Commitment: Clone
        + CanonicalSerialize
        + CanonicalDeserialize
        + Debug
        + PartialEq
        + Eq
        + Default
        + Send
        + Sync;
    type Proof: Clone
        + CanonicalSerialize
        + CanonicalDeserialize
        + Debug
        + PartialEq
        + Eq
        + Send
        + Sync;
    type BatchProof: CanonicalSerialize + CanonicalDeserialize + Clone + Debug + Eq;
    type State: Clone
        + CanonicalSerialize
        + CanonicalDeserialize
        + Debug
        + PartialEq
        + Eq
        + Send
        + Default
        + Sync;
    /// Opening-point-agnostic auxiliary that a dedicated masking server
    /// can precompute in bulk and a prover consumes at open-time to
    /// produce a hiding opening. PCSs that don't support the
    /// masking-server protocol leave this as `()`.
    type MaskingPackage: Clone
        + CanonicalSerialize
        + CanonicalDeserialize
        + Debug
        + Send
        + Sync;
    /// Per-polynomial hiding-state snapshot needed to re-mask a stored
    /// non-ZK opening into a ZK one (see
    /// [`Self::remask_with_package`]). For Pedersen-MSM-based hiding
    /// PCSs this is the polynomial's blinding scalar `tau` at the time
    /// the non-ZK opening was produced.
    type HidingScalar: Clone
        + CanonicalSerialize
        + CanonicalDeserialize
        + Debug
        + Send
        + Sync;

    fn gen_srs_for_testing<R: Rng>(
        conf: Self::Config,
        rng: &mut R,
        supported_size: usize,
    ) -> Result<Self::SRS, PCSError>;

    fn trim(
        srs: impl Borrow<Self::SRS>,
        supported_degree: Option<usize>,
        supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError>;

    fn commit(
        prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<(Self::Commitment, Self::State), PCSError>;

    /// Explicit hiding commitment: produces `C = <f, H_1> + tau*h` and
    /// returns the polynomial's `tau` inside the state (the `Some(tau)`
    /// branch of [`Self::HidingScalar`]). Used by callers that want
    /// hiding for one polynomial regardless of how other polynomials
    /// against the same SRS are committed — Aegon commits its value
    /// polynomials this way so the masking-server protocol has
    /// `tau_f` to plug into `rho_prime = alpha*tau_f + rho`.
    fn commit_zk(
        _prover_param: impl Borrow<Self::ProverParam>,
        _poly: &Self::Polynomial,
    ) -> Result<(Self::Commitment, Self::State), PCSError> {
        unimplemented!("PCS::commit_zk has no default — implement on hiding-capable PCSs")
    }

    /// Explicit plain commitment: `C = <f, H_1>` with no hiding term.
    /// The state's `tau` is `None`. Aegon commits its label-side
    /// polynomials this way so they carry no hiding overhead even
    /// against an SRS that *would* support hiding.
    fn commit_non_zk(
        _prover_param: impl Borrow<Self::ProverParam>,
        _poly: &Self::Polynomial,
    ) -> Result<(Self::Commitment, Self::State), PCSError> {
        unimplemented!("PCS::commit_non_zk has no default — implement on PCSs that distinguish hiding from non-hiding commits")
    }

    /// Block decomposition the PCS uses to map a multilinear polynomial's
    /// `2^num_vars` evaluations into its commitment table. The returned
    /// dimensions must sum to `num_vars` and ordered so that earlier
    /// blocks map to higher-order bits of the storage index — matching
    /// the C-order layout of an `ndarray` of shape `[2^d_1, ..., 2^d_k]`.
    ///
    /// The default treats the polynomial as a single block with
    /// `vec![num_vars]`, which corresponds to ark_poly's standard
    /// "variable `i` in bit `i`" encoding. PCSs that decompose the
    /// hypercube into smaller blocks (e.g. KZH-k) override this to
    /// expose their actual layout, so callers that need to insert at a
    /// specific Boolean point can compute the matching storage index
    /// without duplicating the PCS's own splitting formula.
    fn block_dims(_prover_param: &Self::ProverParam, num_vars: usize) -> Vec<usize> {
        vec![num_vars]
    }

    /// Verifier-side analogue of [`Self::block_dims`]. The PCS
    /// dimensions are public and can be recovered from either the
    /// prover or verifier projection of the SRS, so this exists for
    /// code paths that hold only the verifier_param (e.g. coordinator
    /// setup in remote-shard mode, which fetches verifier_param from
    /// a shard and never materializes prover_param).
    fn block_dims_from_verifier_param(
        _verifier_param: &Self::VerifierParam,
        num_vars: usize,
    ) -> Vec<usize> {
        // Match the default `block_dims` shape — a single block.
        vec![num_vars]
    }

    fn update_state(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomial: &Self::Polynomial,
        com: &Self::Commitment,
        state: &mut Self::State,
    ) -> Result<(), PCSError> {
        unimplemented!()
    }

    /// Linearly combine two prover `State`s in-place:
    /// `target ← target + scalar · other`.
    ///
    /// Mathematically this is the homomorphism `state(f + c·g) =
    /// state(f) + c·state(g)` — the same one that makes the commitment
    /// itself linearly homomorphic. For Pedersen-MSM-based PCSs the
    /// state is a row of group commitments, so the same linearity
    /// applies cell-by-cell.
    ///
    /// Implementations should walk the **sparse support** of `other`
    /// so the cost is `O(|support(other)|)`, not `O(|support(target)|)`.
    /// A correct but degenerate implementation could densify both and
    /// walk every cell; the speedup only materialises when implementors
    /// hand off to a per-non-zero loop.
    ///
    /// Default impl `unimplemented!()` — PCSs that need the
    /// incremental-publish path on the Aegon side must override this.
    fn fma_state(
        _prover_param: impl Borrow<Self::ProverParam>,
        _target: &mut Self::State,
        _scalar: E::ScalarField,
        _other: &Self::State,
    ) -> Result<(), PCSError> {
        unimplemented!("PCS::fma_state has no default — implement the homomorphism for your State")
    }

    /// Open `polynomial` at `point`. Takes the polynomial by borrowed
    /// reference via [`DenseOrSparseMLERef`] so the trait doesn't force a
    /// `BTreeMap` deep-clone at every call site — the sparse-poly clone
    /// to satisfy `&DenseOrSparseMLE::Sparse(poly.clone())` previously
    /// dominated the publish-phase opening time.
    fn open(
        prover_param: impl Borrow<Self::ProverParam>,
        commitment: &Self::Commitment,
        polynomial: DenseOrSparseMLERef<'_, E::ScalarField>,
        point: &Self::Point,
        state: &Self::State,
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError>;

    fn multi_open(
        _prover_param: impl Borrow<Self::ProverParam>,
        commitment: &Self::Commitment,
        _polynomials: &[DenseOrSparseMLERef<'_, E::ScalarField>],
        _point: &Self::Point,
        _states: &[Self::State],
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(Self::BatchProof, Self::Evaluation), PCSError> {
        unimplemented!()
    }

    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError>;

    fn batch_verify(
        _verifier_param: &Self::VerifierParam,
        _commitments: &[Self::Commitment],
        _states: Option<&[Self::State]>,
        _point: &Self::Point,
        _values: &[E::ScalarField],
        _batch_proof: &Self::BatchProof,
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        unimplemented!()
    }

    /// Explicit non-ZK opener. The default [`Self::open`] auto-dispatches
    /// to the hiding variant whenever the SRS is hiding; this entry
    /// point lets callers ask for a plain opening regardless (publish-
    /// time stored openings that will be re-masked later go through
    /// here). PCSs without a ZK variant can just delegate to `open`.
    fn open_non_zk(
        _prover_param: impl Borrow<Self::ProverParam>,
        _commitment: &Self::Commitment,
        _polynomial: DenseOrSparseMLERef<'_, E::ScalarField>,
        _point: &Self::Point,
        _state: &Self::State,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        unimplemented!("PCS::open_non_zk has no default — implement on hiding PCSs that need the masking-server protocol")
    }

    /// Get the hiding scalar `tau` for a polynomial's state. For PCSs
    /// whose [`Self::HidingScalar`] is `()` (non-hiding PCSs), this
    /// returns `()`. Used by the publish path to snapshot the per-
    /// epoch `tau` so the history-lookup path can re-mask stored
    /// openings.
    fn get_hiding_scalar(_state: &Self::State) -> Self::HidingScalar {
        unimplemented!("PCS::get_hiding_scalar has no default — implement on hiding PCSs that need the masking-server protocol")
    }

    /// Producer-side of the masking-server protocol: build an
    /// opening-point-agnostic auxiliary that a consumer can later
    /// turn into a hiding opening without sampling its own masking
    /// polynomial.
    fn generate_masking_package(
        _prover_param: impl Borrow<Self::ProverParam>,
        _num_vars: usize,
    ) -> Result<Self::MaskingPackage, PCSError> {
        unimplemented!("PCS::generate_masking_package has no default — implement on hiding PCSs that need the masking-server protocol")
    }

    /// Consumer-side of the masking-server protocol: hiding-open
    /// `polynomial` at `point` using a precomputed package.
    fn open_zk_with_package(
        _prover_param: impl Borrow<Self::ProverParam>,
        _commitment: &Self::Commitment,
        _polynomial: DenseOrSparseMLERef<'_, E::ScalarField>,
        _point: &Self::Point,
        _state: &Self::State,
        _transcript: &mut IOPTranscript<E::ScalarField>,
        _package: &Self::MaskingPackage,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        unimplemented!("PCS::open_zk_with_package has no default — implement on hiding PCSs that need the masking-server protocol")
    }

    /// Re-mask an already-computed non-ZK opening into a hiding one
    /// using a precomputed package and the polynomial's per-epoch
    /// hiding scalar `tau_f`. Used by the history-lookup path to
    /// upgrade publish-time stored plain openings without re-opening
    /// the polynomial.
    fn remask_with_package(
        _prover_param: impl Borrow<Self::ProverParam>,
        _commitment: &Self::Commitment,
        _point: &Self::Point,
        _value: &E::ScalarField,
        _non_zk_proof: Self::Proof,
        _tau_f: &Self::HidingScalar,
        _transcript: &mut IOPTranscript<E::ScalarField>,
        _package: &Self::MaskingPackage,
    ) -> Result<Self::Proof, PCSError> {
        unimplemented!("PCS::remask_with_package has no default — implement on hiding PCSs that need the masking-server protocol")
    }
}

/// API definitions for structured reference string
pub trait StructuredReferenceString<E: Pairing>: Sized + PCSGlobalParam {
    /// Prover parameters
    type ProverParam: PCSGlobalParam;
    /// Verifier parameters
    type VerifierParam:PCSGlobalParam;

    /// Extract the prover parameters from the public parameters.
    fn extract_prover_param(&self, supported_size: usize) -> Self::ProverParam;
    /// Extract the verifier parameters from the public parameters.
    fn extract_verifier_param(&self, supported_size: usize) -> Self::VerifierParam;

    /// Trim the universal parameters to specialize the public parameters
    /// for polynomials to the given `supported_size`, and
    /// returns committer key and verifier key.
    ///
    /// - For univariate polynomials, `supported_size` is the maximum degree.
    /// - For multilinear polynomials, `supported_size` is 2 to the number of
    ///   variables.
    ///
    /// `supported_log_size` should be in range `1..=params.log_size`
    fn trim(
        &self,
        supported_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError>;

    /// Build SRS for testing.
    ///
    /// - For univariate polynomials, `supported_size` is the maximum degree.
    /// - For multilinear polynomials, `supported_size` is the number of
    ///   variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        k: usize,
        zk: bool,
        supported_size: usize,
    ) -> Result<Self, PCSError>;
}

pub trait PCSGlobalParam {
    fn is_zk(&self) -> bool;
}