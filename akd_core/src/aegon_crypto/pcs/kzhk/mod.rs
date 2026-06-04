//! # KZH-k multilinear polynomial commitment scheme
//!
//! Implementation of the KZH-k polynomial commitment scheme from the IronDict
//! paper ("IronDict: Transparent Dictionaries from Polynomial Commitments",
//! ePrint 2025/1580), Appendix E ("KZH-k description", Figure 14) and
//! Appendix D (zero-knowledge variant).
//!
//! KZH-k is a pairing-based multilinear PCS that commits to a multilinear
//! polynomial `f(X_1, ..., X_k)` whose variables are split into `k` blocks of
//! dimensions `(d_1, ..., d_k)` with `N = 2^{d_1 + ... + d_k}`. The parameter
//! `k` trades off prover and server work against proof size: `k = 2` gives
//! `O(sqrt(N))`-size proofs (the classical KZH scheme), while larger `k`
//! shifts concrete cost toward the server in exchange for smaller proofs and
//! verifier work (see Table 1 of the paper).
//!
//! A distinguishing feature exploited by IronDict is the *free Boolean
//! opening*: when every entry of the query point `x_j` is Boolean, producing
//! the vector `D_j` at each level reduces to selecting a single precomputed
//! auxiliary commitment — no cryptographic work is required. This is what
//! makes the scheme suitable for a dictionary server answering short
//! membership/lookup queries.
//!
//! ## Variants
//!
//! - **Non-ZK KZH-k** (Figure 14): plain commitment `C = <f, H_1>`, openings
//!   consist of level-wise row-commitments `D_j` plus a final partial
//!   evaluation polynomial.
//! - **zk-KZH** (Appendix D): blinds the commitment with `C_hide = C + tau*h`
//!   and uses a Sigma-protocol to open without revealing `f`. A key
//!   optimization (Lemmas 4, 5) is that the masking polynomial `r(X)` can be
//!   sparse with only `k * N^{1/k}` structured non-zero coefficients, making
//!   the zk variant concretely lightweight.
//!
//! The unified [`KZHKConfig`] struct (fields `k` and `zk`) selects the
//! variant when generating the SRS.
#[cfg(feature = "parallel")]
use rayon::iter::IntoParallelRefMutIterator;
use std::collections::BTreeMap;
use crate::aegon_crypto::{
    pcs::{
        kzhk::{
            msm::{msm, naive_msm, NAIVE_THRESHOLD},
            srs::{KZHKProverParam, KZHKUniversalParams, KZHKVerifierParam},
            structs::{AuxRow, KZHKState, KZHKCommitment, KZHKConfig, KZHKOpeningProof},
        },
        PCSGlobalParam,
    },
    poly::{DenseOrSparseMLE, DenseOrSparseMLERef},
    PCSError, PolynomialCommitmentScheme, StructuredReferenceString,
};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::One;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension, SparseMultilinearExtension};
use ark_serialize::CanonicalDeserialize;
use ark_std::{
    cfg_into_iter, cfg_iter, cfg_iter_mut,
    rand::Rng,
    test_rng, Zero,
};
use std::{
    borrow::Borrow,
    env::current_dir,
    fs::{create_dir_all, File},
    io::{BufReader, BufWriter, Read, Write},
    marker::PhantomData,
};
use crate::aegon_crypto::transcript::IOPTranscript;
pub mod msm;
pub mod srs;
pub mod structs;
use crate::aegon_crypto::arithmetic::{
    bits_le_to_usize,
    multilinear_polynomial::{
        fix_last_variables, fix_last_variables_boolean, fix_last_variables_sparse,
        partially_eval_dense_poly_on_bool_point, partially_eval_sparse_poly_on_bool_point,
        rand_sparse_mle,
    },
    virtual_polynomial::build_eq_x_r,
};
use ark_serialize::CanonicalSerialize;
use ark_std::UniformRand;
#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    ParallelIterator,
};
mod test;

/// Type-level handle for the KZH-k PCS. All methods are associated
/// functions parameterized by the pairing engine `E`; the `k` field is
/// unused and kept only for API symmetry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KZHK<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
    k: usize,
}

impl<E> PolynomialCommitmentScheme<E> for KZHK<E>
where
    E: Pairing,
{
    type Config = KZHKConfig;
    type ProverParam = KZHKProverParam<E>;
    type VerifierParam = KZHKVerifierParam<E>;
    type SRS = KZHKUniversalParams<E>;
    type Polynomial = DenseOrSparseMLE<E::ScalarField>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = KZHKCommitment<E>;
    type Proof = KZHKOpeningProof<E>;
    type BatchProof = KZHKOpeningProof<E>;
    type State = KZHKState<E>;
    type MaskingPackage = crate::aegon_crypto::pcs::kzhk::structs::KZHKMaskingPackage<E>;
    type HidingScalar = E::ScalarField;

    /// Generates (or loads) a KZH-k SRS for testing.
    ///
    /// The SRS samples trapdoors `{mu_{b,j}}` for each of the `k` blocks and
    /// builds the tensor families `H_1, ..., H_k` in `G1` and the pairing
    /// elements `V_{b,j}` in `G2` described in Figure 14. Because SRS
    /// generation is expensive (multiple MSMs of size `N`), the result is
    /// cached on disk under
    /// `../artifacts/srs/srs_{k}_{supported_size}_{zk}.bin` and reused on
    /// subsequent invocations of the same size.
    fn gen_srs_for_testing<R: Rng>(
        conf: Self::Config,
        _rng: &mut R,
        supported_size: usize,
    ) -> Result<Self::SRS, PCSError> {
        let k = conf.k;
        let zk = conf.zk;
        // SRS is cached on disk keyed by (k, num_vars, zk) because generating
        // it requires k full-size MSMs in G1 — reusing across test runs saves
        // significant time. The `zk` tag is part of the key because the zk
        // SRS carries a `hiding_sparsity` field that flips `is_zk()`, which
        // changes the open dispatch — sharing a file across both flavours
        // would silently route non-zk runs through the zk path.
        let srs_path = current_dir().unwrap().join(format!(
            "../artifacts/srs/srs_{:?}_{}_{}.bin",
            k,
            supported_size,
            if zk { "zk" } else { "nozk" }
        ));
        let srs = if srs_path.exists() {
            eprintln!("Loading SRS");
            // Stream from disk through the deserializer instead of
            // reading the entire file into a `Vec<u8>` first. Holding
            // both the on-disk bytes and the decoded SRS doubled peak
            // RAM at load — at nv ≥ 28 the file alone is tens of GB.
            let reader = BufReader::new(File::open(&srs_path).unwrap());
            Self::SRS::deserialize_uncompressed_unchecked(reader).unwrap_or_else(|_| {
                panic!("Failed to deserialize SRS from {:?}", srs_path);
            })
        } else {
            eprintln!("Computing SRS");
            let mut rng = test_rng();
            let srs =
                KZHKUniversalParams::gen_srs_for_testing(&mut rng, k, zk, supported_size).unwrap();
            if let Some(parent) = srs_path.parent() {
                create_dir_all(parent).unwrap_or_else(|_| {
                    panic!("could not create directory for SRS at {:?}", parent)
                });
            }
            // Stream the serialization directly into the file. The
            // previous code serialized into an intermediate `Vec<u8>`
            // first and then wrote that buffer out — at nv ≥ 28 the
            // intermediate copy is tens of GB on top of the live SRS,
            // which OOMs even before any disk I/O happens.
            let mut writer = BufWriter::new(
                File::create(srs_path.clone())
                    .unwrap_or_else(|_| panic!("could not create file for SRS at {:?}", srs_path)),
            );
            srs.serialize_uncompressed(&mut writer).unwrap();
            writer.flush().unwrap();
            srs
        };
        Ok(srs)
    }

    /// Extracts prover and verifier parameters from the universal SRS.
    /// The total number of variables must equal the sum of block dimensions
    /// `d_1 + ... + d_k` fixed at SRS generation time.
    fn trim(
        srs: impl Borrow<Self::SRS>,
        _supported_degree: Option<usize>,
        supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let srs = srs.borrow();
        let supp_nv = supported_num_vars.unwrap();
        assert_eq!(srs.get_dimensions().iter().sum::<usize>(), supp_nv);
        Ok((
            srs.extract_prover_param(supp_nv),
            srs.extract_verifier_param(supp_nv),
        ))
    }

    /// KZH-k stores its commitment table as an `ndarray` of shape
    /// `[2^{d_1}, ..., 2^{d_k}]` in C-order, so the first block of
    /// variables lands in the highest-order bits of the storage index.
    /// Callers that mutate the polynomial's BTreeMap directly need
    /// these dims to keep insertions and openings aligned.
    fn block_dims(prover_param: &Self::ProverParam, _num_vars: usize) -> Vec<usize> {
        prover_param.get_dimensions().clone()
    }

    /// Verifier-side dims. KZH-k stores the same `dimensions` vector
    /// in both `KZHKProverParam` and `KZHKVerifierParam`, so the
    /// verifier path gets the exact same layout the prover used —
    /// matching is essential because openings must address the same
    /// Boolean-point decomposition both sides expect.
    fn block_dims_from_verifier_param(
        verifier_param: &Self::VerifierParam,
        _num_vars: usize,
    ) -> Vec<usize> {
        verifier_param.get_dimensions().clone()
    }

    /// Commits to a multilinear polynomial `f`. Dispatches to the zk or
    /// non-zk variant based on the SRS configuration. In the non-zk case the
    /// commitment is `C = <f, H_1>` (Figure 14, Commit); in the zk case it
    /// is blinded as `C + tau*h` (Appendix D).
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Commit")]
    fn commit(
        prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<(Self::Commitment, Self::State), PCSError> {
        let result = if !prover_param.borrow().is_zk() {
            Ok(Self::commit_non_zk(prover_param, poly).unwrap())
        } else {
            Self::commit_zk(prover_param, poly)
        };
        result
    }

    /// Precomputes the row-commitment auxiliaries
    /// `aux_{b_1,...,b_j} = <f(b_1,...,b_j, X_{j+1},...), H_{j+1}>`
    /// for each level `j = 1..k-1` (Figure 14). These are the building
    /// blocks for the "free Boolean opening" shortcut: when the opening
    /// point is Boolean, the level-`j` proof vector `D_j` is obtained by a
    /// plain slice of these stored group elements — no MSM required.
    fn update_state(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomial: &Self::Polynomial,
        com: &Self::Commitment,
        state: &mut Self::State,
    ) -> Result<(), PCSError> {
        Self::update_state_inner(prover_param, polynomial, com, state)
    }

    /// Sparse-walk FMA on the prover state. Drives the §6.4
    /// incremental-publish path: the system commits + aux's the
    /// **delta polynomial** for the current batch (size `batch`) and
    /// then merges it into the prior epoch's state via this method.
    /// Cost is `O(k · batch)` regardless of how many users the
    /// dictionary already holds — see [`KZHKState::iadd_scaled`].
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::FMAState")]
    fn fma_state(
        _prover_param: impl Borrow<Self::ProverParam>,
        target: &mut Self::State,
        scalar: E::ScalarField,
        other: &Self::State,
    ) -> Result<(), PCSError> {
        target.iadd_scaled(scalar, other);
        Ok(())
    }

    /// Produces an opening proof `pi = ({D_j}_{j=1}^{k-1}, f_{x_1..x_{k-1}})`
    /// of `f` at the point `(x_1, ..., x_k)` and the evaluation `y = f(x)`.
    /// Dispatches to the zk Sigma-protocol variant (Appendix D) when the
    /// SRS is hiding.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Open")]
    fn open(
        prover_param: impl Borrow<Self::ProverParam>,
        commitment: &Self::Commitment,
        polynomial: DenseOrSparseMLERef<'_, E::ScalarField>,
        point: &Self::Point,
        state: &Self::State,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let result = if !prover_param.borrow().is_zk() {
            Self::open_non_zk(prover_param, commitment, polynomial, point, state)
        } else {
            Self::open_zk(prover_param, commitment, polynomial, point, state, transcript)
        };

        result
    }

    /// Opens a batch of polynomials at a common point by linearly
    /// aggregating them (currently without random challenges; see
    /// `multi_open_non_zk`).
    fn multi_open(
        prover_param: impl Borrow<Self::ProverParam>,
        commitment: &Self::Commitment,
        polynomials: &[DenseOrSparseMLERef<'_, E::ScalarField>],
        point: &Self::Point,
        states: &[Self::State],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(Self::BatchProof, Self::Evaluation), PCSError> {
        Self::multi_open_non_zk(
            prover_param,
            commitment,
            polynomials,
            point,
            states,
            transcript,
        )
    }

    /// Verifies an opening proof. The proof's contents (presence of the
    /// Sigma-protocol fields `r_hide`, `y_r`, `rho_prime`) determines
    /// whether the zk verifier (Appendix D) or the plain verifier
    /// (Figure 14) is invoked.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Verify")]
    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        let result = match (proof.get_r_hide(), proof.get_y_r(), proof.get_rho_prime()) {
            (Some(_), Some(_), Some(_)) => {
                Self::verify_zk(verifier_param, commitment, point, value, None, proof, transcript)
            },
            _ => Self::verify_non_zk(verifier_param, commitment, point, value, None, proof),
        };
        result
    }

    /// Verifies a batch opening proof against a set of commitments and
    /// claimed evaluations at a shared point.
    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        states: Option<&[Self::State]>,
        point: &Self::Point,
        values: &[E::ScalarField],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        Self::batch_verify_non_zk(
            verifier_param,
            commitments,
            states,
            point,
            values,
            batch_proof,
            transcript,
        )
    }

    /// Public-trait shim onto the (crate-private) `open_non_zk` so
    /// callers that want a plain opening even from a hiding SRS can
    /// reach it through the generic `P::` interface.
    fn open_non_zk(
        prover_param: impl Borrow<Self::ProverParam>,
        commitment: &Self::Commitment,
        polynomial: DenseOrSparseMLERef<'_, E::ScalarField>,
        point: &Self::Point,
        state: &Self::State,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        Self::open_non_zk(prover_param, commitment, polynomial, point, state)
    }

    /// Public-trait shim onto the (crate-private) `commit_zk`. Used
    /// by Aegon to commit value-side polynomials with hiding
    /// (`C = <f, H_1> + tau*h`) even when label-side polys against
    /// the same SRS use plain commits.
    fn commit_zk(
        prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<(Self::Commitment, Self::State), PCSError> {
        Self::commit_zk(prover_param, poly)
    }

    /// Public-trait shim onto the (crate-private) `commit_non_zk`.
    /// Used by Aegon to commit label-side polynomials plain — no
    /// hiding overhead — even when value-side polys against the
    /// same SRS are hiding.
    fn commit_non_zk(
        prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<(Self::Commitment, Self::State), PCSError> {
        Self::commit_non_zk(prover_param, poly)
    }

    fn get_hiding_scalar(state: &Self::State) -> Self::HidingScalar {
        // `state.tau` is `None` on non-hiding (`zk=false`) SRS — return
        // zero so callers can capture per-epoch hiding scalars
        // unconditionally without panicking on plain mode. The remask
        // path is a no-op in non-hiding mode anyway (it checks
        // `is_zk()` and short-circuits), so the zero is never used.
        state
            .maybe_tau()
            .copied()
            .unwrap_or_else(<E::ScalarField as Zero>::zero)
    }

    fn rerandomise_hiding_scalar<R>(
        state: &mut Self::State,
        rng: &mut R,
    ) -> Option<E::ScalarField>
    where
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
    {
        let tau_old = *state.maybe_tau()?;
        let tau_new = E::ScalarField::rand(rng);
        state.set_tau(tau_new);
        Some(tau_new - tau_old)
    }

    fn scaled_mask_generator_pp(
        pp: &Self::ProverParam,
        model: &Self::Commitment,
        scalar: E::ScalarField,
    ) -> Option<Self::Commitment> {
        if !PCSGlobalParam::is_zk(pp) {
            return None;
        }
        let h_scaled = (pp.get_h() * scalar).into_affine();
        Some(KZHKCommitment::new(h_scaled, model.get_num_vars()))
    }

    fn scaled_mask_generator_vk(
        vk: &Self::VerifierParam,
        model: &Self::Commitment,
        scalar: E::ScalarField,
    ) -> Option<Self::Commitment> {
        if !PCSGlobalParam::is_zk(vk) {
            return None;
        }
        let h_scaled = (vk.get_h() * scalar).into_affine();
        Some(KZHKCommitment::new(h_scaled, model.get_num_vars()))
    }

    fn generate_masking_package(
        prover_param: impl Borrow<Self::ProverParam>,
        num_vars: usize,
    ) -> Result<Self::MaskingPackage, PCSError> {
        let pp = prover_param.borrow();
        // A masking package only makes sense against a hiding SRS:
        // the polynomial's commitment carries its own commit-time
        // `tau_f * h` term, and `rho_prime = alpha * tau_f + rho`
        // requires `tau_f` to be defined. On a non-hiding SRS
        // (`state.tau = None`), the consumer can't construct a valid
        // hiding opening from a masking package — callers fall back
        // to a plain non-ZK opening instead.
        assert!(
            pp.is_zk(),
            "generate_masking_package: prover param must be hiding (zk SRS)"
        );
        let r_poly = rand_sparse_mle(
            num_vars,
            pp.get_hiding_sparsity().unwrap(),
            &mut test_rng(),
        );
        let r_poly_wrapped = DenseOrSparseMLE::Sparse(r_poly.clone());
        let (r_hide, mut r_state) = Self::commit(pp, &r_poly_wrapped)?;
        let rho = *r_state.get_tau();
        Self::update_state(pp, &r_poly_wrapped, &r_hide, &mut r_state)?;
        Ok(crate::aegon_crypto::pcs::kzhk::structs::KZHKMaskingPackage::new(
            num_vars, r_poly, r_hide, r_state, rho,
        ))
    }

    fn open_zk_with_package(
        prover_param: impl Borrow<Self::ProverParam>,
        commitment: &Self::Commitment,
        polynomial: DenseOrSparseMLERef<'_, E::ScalarField>,
        point: &Self::Point,
        state: &Self::State,
        transcript: &mut IOPTranscript<E::ScalarField>,
        package: &Self::MaskingPackage,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let pp = prover_param.borrow();
        assert_eq!(
            package.num_vars,
            polynomial.num_vars(),
            "open_zk_with_package: package num_vars {} != polynomial num_vars {}",
            package.num_vars,
            polynomial.num_vars(),
        );
        let (non_zk_opening, non_zk_value) =
            Self::open_non_zk(pp, commitment, polynomial, point, state)?;
        // `tau_f` is the polynomial's commit-time blinding scalar
        // (Appendix D notation), fixed at commit time and threaded
        // through every opening. The masking package's own `rho`
        // blinds the masking polynomial `r`; both terms appear in
        // `rho_prime = alpha * tau_f + rho` so the verifier can
        // de-randomise the hiding offset of `C_lin`.
        let proof = Self::apply_masking_package(
            pp,
            commitment,
            point,
            &non_zk_value,
            non_zk_opening,
            state.get_tau(),
            transcript,
            package,
        )?;
        Ok((proof, non_zk_value))
    }

    fn remask_with_package(
        prover_param: impl Borrow<Self::ProverParam>,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        non_zk_proof: Self::Proof,
        tau_f: &Self::HidingScalar,
        transcript: &mut IOPTranscript<E::ScalarField>,
        package: &Self::MaskingPackage,
    ) -> Result<Self::Proof, PCSError> {
        let pp = prover_param.borrow();
        Self::apply_masking_package(
            pp,
            commitment,
            point,
            value,
            non_zk_proof,
            tau_f,
            transcript,
            package,
        )
    }
}

impl<E: Pairing> KZHK<E> {
    /// zk commitment from Appendix D: returns `C_hide = C + tau*h` where
    /// `tau` is a fresh blinding factor returned in the auxiliary info so
    /// the prover can later derandomize during opening.
    ///
    /// The blinding `tau*h` is folded directly into the commitment MSM by
    /// appending `(tau, h)` to the scalar/base inputs — avoiding the extra
    /// scalar multiplication and group addition that a post-hoc blinding
    /// would incur.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Commit-ZK")]
    fn commit_zk(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        poly: &DenseOrSparseMLE<E::ScalarField>,
    ) -> Result<(KZHKCommitment<E>, KZHKState<E>), PCSError> {
        let pp: &KZHKProverParam<E> = prover_param.borrow();
        let tau = E::ScalarField::rand(&mut test_rng());
        let blinding = Some((tau, pp.get_h()));
        let com = match poly {
            DenseOrSparseMLE::Dense(poly) => Self::commit_dense_inner(pp, poly, blinding)?,
            DenseOrSparseMLE::Sparse(poly) => Self::commit_sparse_inner(pp, poly, blinding)?,
        };
        let sparsity = sparsity_of(poly);
        let result = Ok((com, KZHKState::new(Some(tau), None, Some(sparsity))));
        result
    }

    /// Plain KZH-k commitment `C = <f, H_1>` (Figure 14, Commit). The
    /// MSM is selected by polynomial representation (dense vs sparse).
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Commit-Non-ZK")]
    fn commit_non_zk(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        poly: &DenseOrSparseMLE<E::ScalarField>,
    ) -> Result<(KZHKCommitment<E>, KZHKState<E>), PCSError> {
        let non_zk_com = match poly {
            DenseOrSparseMLE::Dense(poly) => Self::commit_dense_inner(prover_param, poly, None),
            DenseOrSparseMLE::Sparse(poly) => Self::commit_sparse_inner(prover_param, poly, None),
        };
        let sparsity = sparsity_of(poly);
        let state = KZHKState::new(None, None, Some(sparsity));
        let result = Ok((non_zk_com.unwrap(), state));
        result
    }

    /// Computes the Boolean auxiliary table of row-commitments and stores
    /// it in `state`. Used as backing store for the free-Boolean-opening
    /// shortcut (see [`KZHK::update_state`]).
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::CompAux")]
    fn update_state_inner(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        polynomial: &DenseOrSparseMLE<E::ScalarField>,
        com: &KZHKCommitment<E>,
        state: &mut KZHKState<E>,
    ) -> Result<(), PCSError> {
        let result = match polynomial {
            DenseOrSparseMLE::Dense(poly) => Self::update_state_dense(prover_param, poly, com, state),
            DenseOrSparseMLE::Sparse(poly) => Self::update_state_sparse(prover_param, poly, com, state),
        };
        result
    }

    /// Non-ZK opening implementing Figure 14, step 2: for each level
    /// `j = 1..k-1`, produces the vector `D_j` of commitments to the
    /// partial evaluation of `f` on the `j`-th block. Dispatches to one
    /// of four specialized implementations depending on whether the
    /// polynomial is dense/sparse and whether the point is Boolean
    /// (Boolean points trigger the free-opening shortcut).
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Open-Non-ZK")]
    fn open_non_zk(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        _commitment: &KZHKCommitment<E>,
        polynomial: DenseOrSparseMLERef<'_, E::ScalarField>,
        point: &[E::ScalarField],
        state: &KZHKState<E>,
    ) -> Result<(KZHKOpeningProof<E>, E::ScalarField), PCSError> {
        //TODO: Make the iters here parallel
        let is_boolean_point = point.iter().all(|&x| x.is_zero() || x.is_one());
        let result = match (is_boolean_point, polynomial) {
            (true, DenseOrSparseMLERef::Dense(poly)) => {
                Self::open_dense_bool_inner(prover_param, poly, point, state)
            },
            (true, DenseOrSparseMLERef::Sparse(poly)) => {
                Self::open_sparse_bool_inner(prover_param, poly, point, state)
            },
            (false, DenseOrSparseMLERef::Dense(poly)) => {
                Self::open_dense_non_bool_inner(prover_param, poly, point, state)
            },
            (false, DenseOrSparseMLERef::Sparse(poly)) => {
                Self::open_sparse_non_bool_inner(prover_param, poly, point, state)
            },
        };
        result
    }

    /// zk opening from Appendix D. Samples a sparse masking polynomial
    /// `r(X)` of structured form (Lemmas 4, 5 — only `k * N^{1/k}`
    /// non-zero coefficients suffice), commits to it as `R_hide`, opens
    /// the non-hiding combination `alpha*f + r` at the challenge point,
    /// and sends `rho_prime = alpha*tau + rho` to derandomize the
    /// verifier's linearization.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Open-ZK")]
    fn open_zk(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        commitment: &KZHKCommitment<E>,
        polynomial: DenseOrSparseMLERef<'_, E::ScalarField>,
        point: &[E::ScalarField],
        state: &KZHKState<E>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(KZHKOpeningProof<E>, E::ScalarField), PCSError> {
        let prover_param: &KZHKProverParam<E> = prover_param.borrow();
        let (non_zk_opening, non_zk_value) =
            Self::open_non_zk(prover_param, commitment, polynomial, point, state)?;
        let r_poly: SparseMultilinearExtension<E::ScalarField> = rand_sparse_mle(
            polynomial.num_vars(),
            prover_param.get_hiding_sparsity().unwrap(),
            &mut test_rng(),
        );
        let r_poly_wrapped = DenseOrSparseMLE::Sparse(r_poly.clone());
        let (r_hide, mut r_state) = Self::commit(prover_param, &r_poly_wrapped)?;
        let rho = *r_state.get_tau();
        Self::update_state(prover_param, &r_poly_wrapped, &r_hide, &mut r_state)?;
        let (r_opening, y_r) =
            Self::open_non_zk(prover_param, &r_hide, r_poly_wrapped.as_ref(), point, &r_state)?;
        // Fiat-Shamir: derive alpha from the prover's first-round messages.
        // Verifier replays the same appends in the same order — see
        // `verify_zk`. Once both sides commit to (C, point, y, R_hide)
        // before the challenge, alpha is binding for the sigma protocol.
        let alpha = Self::derive_alpha(transcript, commitment, point, &non_zk_value, &r_hide)?;
        // rho_prime = alpha*tau + rho derandomizes the hiding offset so that
        // `alpha*C_hide + R_hide - rho_prime * h` equals the non-hiding
        // commitment to `alpha*f + r`.
        let rho_prime = alpha * state.get_tau() + rho;
        let mut output_opening = non_zk_opening * alpha + r_opening;
        output_opening.set_r_hide(r_hide);
        output_opening.set_y_r(y_r);
        output_opening.set_rho_prime(rho_prime);
        Ok((output_opening, non_zk_value))
    }

    /// Shared core of the masking-server consumer path: given a
    /// precomputed non-ZK opening and a masking package, produce the
    /// linearized ZK opening. Mirrors steps 3–7 of [`Self::open_zk`].
    fn apply_masking_package(
        prover_param: &KZHKProverParam<E>,
        commitment: &KZHKCommitment<E>,
        point: &[E::ScalarField],
        value: &E::ScalarField,
        non_zk_opening: KZHKOpeningProof<E>,
        tau_f: &E::ScalarField,
        transcript: &mut IOPTranscript<E::ScalarField>,
        package: &crate::aegon_crypto::pcs::kzhk::structs::KZHKMaskingPackage<E>,
    ) -> Result<KZHKOpeningProof<E>, PCSError> {
        let r_poly_wrapped = DenseOrSparseMLE::Sparse(package.r_poly.clone());
        let (r_opening, y_r) = Self::open_non_zk(
            prover_param,
            &package.r_hide,
            r_poly_wrapped.as_ref(),
            point,
            &package.r_state,
        )?;
        let alpha = Self::derive_alpha(transcript, commitment, point, value, &package.r_hide)?;
        let rho_prime = alpha * tau_f + package.rho;
        let mut output_opening = non_zk_opening * alpha + r_opening;
        output_opening.set_r_hide(package.r_hide.clone());
        output_opening.set_y_r(y_r);
        output_opening.set_rho_prime(rho_prime);
        Ok(output_opening)
    }

    /// Fiat-Shamir derivation of the sigma-protocol challenge `alpha`.
    /// Both prover and verifier call this with the same inputs in the
    /// same order, so they agree on `alpha` without communication.
    fn derive_alpha(
        transcript: &mut IOPTranscript<E::ScalarField>,
        commitment: &KZHKCommitment<E>,
        point: &[E::ScalarField],
        value: &E::ScalarField,
        r_hide: &KZHKCommitment<E>,
    ) -> Result<E::ScalarField, PCSError> {
        transcript.append_serializable_element(b"C", &commitment.get_commitment())?;
        for p in point {
            transcript.append_serializable_element(b"point", p)?;
        }
        transcript.append_serializable_element(b"y", value)?;
        transcript.append_serializable_element(b"R_hide", &r_hide.get_commitment())?;
        Ok(transcript.get_and_append_challenge(b"alpha")?)
    }

    /// Batched opening by plain sum (no random linear combination).
    ///
    /// NOTE: not sound as a proof-of-knowledge batch opener because it
    /// lacks random challenges — kept for benchmarking and as a baseline.
    fn multi_open_non_zk(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        commitment: &KZHKCommitment<E>,
        polynomials: &[DenseOrSparseMLERef<'_, E::ScalarField>],
        point: &Vec<E::ScalarField>,
        states: &[KZHKState<E>],
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(KZHKOpeningProof<E>, E::ScalarField), PCSError> {
        let num_vars = point.len();
        let mut aggr_state: KZHKState<E> = KZHKState::default();
        let (agg_poly, aggr_state) = match polynomials[0] {
            DenseOrSparseMLERef::Dense(_) => {
                let mut aggr_poly = DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    vec![E::ScalarField::zero(); 1usize << num_vars],
                );
                for (poly, state) in polynomials.iter().zip(states.iter()) {
                    if let DenseOrSparseMLERef::Dense(dense_poly) = *poly {
                        aggr_poly += dense_poly;
                        aggr_state = aggr_state + state.clone();
                    } else {
                        panic!("All polynomials must be dense here");
                    }
                }
                (DenseOrSparseMLE::Dense(aggr_poly), aggr_state)
            },
            DenseOrSparseMLERef::Sparse(_) => {
                let mut aggr_poly =
                    SparseMultilinearExtension::from_evaluations(num_vars, Vec::new());
                for (poly, state) in polynomials.iter().zip(states.iter()) {
                    if let DenseOrSparseMLERef::Sparse(sparse_poly) = *poly {
                        aggr_poly += sparse_poly;
                        aggr_state = aggr_state + state.clone();
                    } else {
                        panic!("All polynomials must be sparse here");
                    }
                }

                (DenseOrSparseMLE::Sparse(aggr_poly), aggr_state)
            },
        };
        Self::open_non_zk(prover_param, commitment, agg_poly.as_ref(), point, &aggr_state)
    }

    /// zk verifier (Appendix D): reconstructs the non-hiding commitment
    /// `alpha*C_hide + R_hide - rho_prime*h` and the non-hiding value
    /// `alpha*y + y_r`, then delegates to [`Self::verify_non_zk`] on the
    /// linearized instance.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Verify-ZK")]
    fn verify_zk(
        verifier_param: &KZHKVerifierParam<E>,
        commitment: &KZHKCommitment<E>,
        point: &[E::ScalarField],
        value: &E::ScalarField,
        _state: Option<&KZHKState<E>>,
        proof: &KZHKOpeningProof<E>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        let r_hide = proof.get_r_hide().as_ref().unwrap();
        let alpha = Self::derive_alpha(transcript, commitment, point, value, r_hide)?;
        let c_lin = (commitment.get_commitment().into_group() * alpha
            + r_hide.get_commitment().into_group()
            - verifier_param.get_h() * proof.get_rho_prime().unwrap())
        .into_affine();
        let lin_commitment = KZHKCommitment::new(c_lin, commitment.get_num_vars());
        let lin_value = *value * alpha + proof.get_y_r().unwrap();
        let result = Self::verify_non_zk(
            verifier_param,
            &lin_commitment,
            point,
            &lin_value,
            None,
            proof,
        );
        result
    }

    /// Non-ZK verifier implementing Figure 14, Verify:
    /// 1. For each level `j = 1..k-1`, check the pairing identity
    ///    `e(C_{j-1}, V) = prod_b e(D_{j,b}, V_{b,j})` asserting that
    ///    `D_j` commits to a valid partial evaluation of `f` on block `j`.
    ///    After passing, fold `C_j = <D_j, eq(point_j)>` for the next step.
    /// 2. Finally check `C_{k-1} = <f_{x_1..x_{k-1}}, H_k>` and that the
    ///    tail polynomial evaluates to `y` at `x_k`.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Verify-Non-ZK")]
    fn verify_non_zk(
        verifier_param: &KZHKVerifierParam<E>,
        commitment: &KZHKCommitment<E>,
        point: &[E::ScalarField],
        value: &E::ScalarField,
        _state: Option<&KZHKState<E>>,
        proof: &KZHKOpeningProof<E>,
    ) -> Result<bool, PCSError> {
        let k = verifier_param.get_dimensions().len();
        let mut cj = commitment.get_commitment();
        let decomposed_point = KZHK::<E>::decompose_point(verifier_param.get_dimensions(), point);
        let pairing_loop_span = tracing::debug_span!("KZH::Verify::PairingLoop");
        let pairing_loop_guard = pairing_loop_span.enter();

        // TODO: See if it's worth it to randomely combine all multi-pairings
        for (j, point_part) in decomposed_point.iter().take(k - 1).enumerate() {
            let cj_prepared = <E as Pairing>::G1Prepared::from(cj);
            let minus_v_prepared = <E as Pairing>::G2Prepared::from(verifier_param.get_minus_v());

            let mut g1_terms = Vec::with_capacity(1 + proof.get_d()[j].len());
            let mut g2_terms = Vec::with_capacity(1 + verifier_param.get_v_mat()[j].len());

            g1_terms.push(cj_prepared);
            g2_terms.push(minus_v_prepared.clone());

            g1_terms.extend(
                proof.get_d()[j]
                    .iter()
                    .copied()
                    .map(<E as Pairing>::G1Prepared::from),
            );
            g2_terms.extend(verifier_param.get_v_mat()[j].iter().cloned());

            let prod = E::multi_pairing(g1_terms, g2_terms);
            if !prod.is_zero() {
                // Pairing identity at level `j` rejected: either the
                // committed `D_j` row doesn't correspond to a valid
                // partial evaluation, or the upstream `C_{j-1}` we're
                // checking it against doesn't either. Could equally
                // come from a tampered proof or from a tampered
                // `(commitment, value, R_hide)` triple feeding
                // `verify_zk`'s `C_lin` reconstruction. Either way,
                // it's a rejection — not a programmer bug — so we
                // surface `Ok(false)` rather than `debug_assert!`.
                return Ok(false);
            }

            let eq_poly = build_eq_x_r(point_part).unwrap();
            cj = msm::<E::G1>(&proof.get_d()[j], &eq_poly.evaluations).into_affine();
        }
        drop(pairing_loop_guard);
        // Checking c_{k-1}
        let cj_check_span = tracing::debug_span!("KZH::Verify::CJCheck");
        let cj_check_guard = cj_check_span.enter();
        let alleged_last_cj = E::G1::msm(
            verifier_param
                .get_h_tensor()
                .as_slice_memory_order()
                .unwrap(),
            &proof.get_f().to_evaluations(),
        )
        .unwrap()
        .into_affine();
        if cj != alleged_last_cj {
            return Ok(false);
        }
        drop(cj_check_guard);
        // Evaluation Check
        let eval_check_span = tracing::debug_span!("KZH::Verify::EvalCheck");
        let eval_check_guard = eval_check_span.enter();
        let eval_ok = match proof.get_f() {
            DenseOrSparseMLE::Dense(f) => {
                fix_last_variables(f, &decomposed_point[k - 1])[0] == *value
            },
            DenseOrSparseMLE::Sparse(f) => {
                fix_last_variables_sparse(f, &decomposed_point[k - 1])[0] == *value
            },
        };
        drop(eval_check_guard);
        Ok(eval_ok)
    }

    /// Batch verifier: sums commitments and values and delegates to
    /// single-point `verify`. Mirrors the simple aggregation of
    /// `multi_open_non_zk` and inherits its lack of random batching.
    fn batch_verify_non_zk(
        verifier_param: &KZHKVerifierParam<E>,
        commitments: &[KZHKCommitment<E>],
        states: Option<&[KZHKState<E>]>,
        point: &Vec<E::ScalarField>,
        values: &[E::ScalarField],
        batch_proof: &KZHKOpeningProof<E>,
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        let mut aggr_comm = KZHKCommitment::default();
        let mut aggr_value = E::ScalarField::zero();
        for ((comm, _aux), value) in commitments.iter().zip(states.iter()).zip(values.iter()) {
            aggr_comm = aggr_comm + *comm;
            aggr_value += value;
        }

        Self::verify(
            verifier_param,
            &aggr_comm,
            point,
            &aggr_value,
            batch_proof,
            _transcript,
        )
    }

    /// Dense path of `commit_non_zk`: one MSM of the dense evaluation
    /// vector against the flattened `H_1` tensor.
    ///
    /// When `blinding = Some((tau, h))`, the pair is appended to the MSM
    /// inputs so the returned commitment is `C + tau*h` (Appendix D). This
    /// folds the hiding factor into the single MSM rather than performing
    /// a separate scalar multiplication.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Commit_Dense")]
    fn commit_dense_inner(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        poly: &DenseMultilinearExtension<E::ScalarField>,
        blinding: Option<(E::ScalarField, E::G1Affine)>,
    ) -> Result<KZHKCommitment<E>, PCSError> {
        let prover_param: &KZHKProverParam<E> = prover_param.borrow();
        let h_bases = prover_param.get_h_tensors()[0]
            .as_slice_memory_order()
            .unwrap();
        let com = if let Some((tau, h)) = blinding {
            let mut bases: Vec<E::G1Affine> = Vec::with_capacity(h_bases.len() + 1);
            bases.extend_from_slice(h_bases);
            bases.push(h);
            let mut scalars: Vec<E::ScalarField> = Vec::with_capacity(poly.evaluations.len() + 1);
            scalars.extend_from_slice(&poly.evaluations);
            scalars.push(tau);
            msm::<E::G1>(&bases, &scalars)
        } else {
            msm::<E::G1>(h_bases, &poly.evaluations)
        };
        Ok(KZHKCommitment::new(com.into(), poly.num_vars()))
    }

    /// Sparse path of `commit_non_zk`: gathers only the `H_1` bases
    /// corresponding to non-zero coefficients of the sparse polynomial
    /// and runs a much smaller MSM. This is the path used by the sparse
    /// masking polynomial in [`Self::open_zk`].
    ///
    /// When `blinding = Some((tau, h))`, `(tau, h)` is appended to the
    /// MSM so the returned commitment is `C + tau*h` (Appendix D).
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Commit_Sparse")]
    fn commit_sparse_inner(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        sparse_poly: &SparseMultilinearExtension<E::ScalarField>,
        blinding: Option<(E::ScalarField, E::G1Affine)>,
    ) -> Result<KZHKCommitment<E>, PCSError> {
        let prover_param: &KZHKProverParam<E> = prover_param.borrow();
        let h_mat = prover_param.get_h_tensors()[0]
            .as_slice_memory_order()
            .unwrap();
        let nnz = sparse_poly.evaluations.len();
        let extra = blinding.is_some() as usize;
        let mut scalars: Vec<E::ScalarField> = Vec::with_capacity(nnz + extra);
        let mut bases: Vec<E::G1Affine> = Vec::with_capacity(nnz + extra);
        for (&index, &value) in sparse_poly.evaluations.iter() {
            bases.push(h_mat[index]);
            scalars.push(value);
        }
        if let Some((tau, h)) = blinding {
            bases.push(h);
            scalars.push(tau);
        }
        let com = msm::<E::G1>(&bases, &scalars);
        Ok(KZHKCommitment::new(
            com.into_affine(),
            sparse_poly.num_vars(),
        ))
    }

    /// Dense implementation of [`Self::update_state`]: for each level
    /// `j = 1..k-1`, computes the full row of `2^{d_1+...+d_j}` commitments
    /// `aux_{b_1,...,b_j} = <f(b_1,...,b_j, X_{j+1},...), H_{j+1}>` by
    /// splitting the dense evaluation table into contiguous chunks and
    /// running one MSM per chunk.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::CompAux_Dense")]
    fn update_state_dense(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        polynomial: &DenseMultilinearExtension<E::ScalarField>,
        _com: &KZHKCommitment<E>,
        state: &mut KZHKState<E>,
    ) -> Result<(), PCSError> {
        let prover_param: &KZHKProverParam<E> = prover_param.borrow();
        let dimensions = prover_param.get_dimensions();
        let k = dimensions.len();
        debug_assert!(k >= 2, "need at least 2 blocks to build d_i's");

        let mut d_bool: Vec<AuxRow<E>> = Vec::with_capacity(k - 1);
        let mut prefix_vars: usize = 0;

        for (j, &dim) in dimensions.iter().take(k - 1).enumerate() {
            // Update prefix sum of variables up to and including block j
            prefix_vars += dim;

            // Number of i's (outer loop) and length of each partial evaluation
            let dj_size = 1usize << prefix_vars;
            let rem_vars = polynomial.num_vars() - prefix_vars;
            let eval_len = 1usize << rem_vars;

            // Choose H_t. Natural generalization uses [j]; if you intended to always use
            // [0], replace `j` with `0` below.
            let h_slice = prover_param.get_h_tensors()[j + 1]
                .as_slice_memory_order()
                .expect("H_t must be contiguous (standard layout)");

            // Build d_{j}.
            //
            // We parallelize the outer loop *only* when every per-chunk MSM
            // is small enough that `msm` takes its naive path
            // (no nested rayon pool inside arkworks' Pippenger). For dense
            // inputs, `eval_len` is a tight bound on the per-chunk MSM size.
            let per_msm_size = eval_len;
            let parallel_outer_safe = per_msm_size <= NAIVE_THRESHOLD;
            let mut d_j = vec![E::G1Affine::zero(); dj_size];
            if parallel_outer_safe {
                cfg_iter_mut!(d_j).enumerate().for_each(|(i, d_j_i)| {
                    let scalars =
                        partially_eval_dense_poly_on_bool_point(polynomial, i, eval_len);
                    *d_j_i = msm::<E::G1>(h_slice, scalars.as_slice()).into_affine();
                });
            } else {
                d_j.iter_mut().enumerate().for_each(|(i, d_j_i)| {
                    let scalars =
                        partially_eval_dense_poly_on_bool_point(polynomial, i, eval_len);
                    *d_j_i = msm::<E::G1>(h_slice, scalars.as_slice()).into_affine();
                });
            }

            d_bool.push(AuxRow::Dense(d_j));
        }
        state.set_d_bool(d_bool);
        Ok(())
    }

    /// Sparse counterpart of [`Self::update_state_dense`]: exploits the
    /// sparse coefficient map so each per-cell MSM only sees the
    /// non-zero entries falling in its Boolean window.
    ///
    /// Strategy (post-flattening). The aux table is a 2-D structure
    /// indexed by `(level j, prefix b_1...b_j)`. The two loops are
    /// independent — every cell's MSM is computed from `polynomial`'s
    /// non-zeros and `H_{j+1}`, with no data dependency between cells.
    /// We exploit that by collecting **all** non-empty cells across
    /// **all** levels into one flat `Vec<(j, prefix, bases, scalars)>`
    /// and running a single `cfg_into_iter!` over it.
    ///
    /// Why this beats nested parallelism: with k=20 we have ~`(k-1)·c`
    /// independent MSMs (mostly size-1 at sparse levels). Nested rayon
    /// (par over levels × par over cells) creates hundreds of small
    /// tasks competing for the global pool; profiling showed per-level
    /// wall time was ~10 ms even when actual work was µs because the
    /// scheduler was overwhelmed. One flat par_iter with N tasks of
    /// uniform shape gives rayon a clean work-stealing problem.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::CompAux_Sparse")]
    fn update_state_sparse(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        polynomial: &SparseMultilinearExtension<E::ScalarField>,
        _com: &KZHKCommitment<E>,
        state: &mut KZHKState<E>,
    ) -> Result<(), PCSError> {
        let prover_param: &KZHKProverParam<E> = prover_param.borrow();
        let dimensions = prover_param.get_dimensions();
        let k = dimensions.len();
        debug_assert!(k >= 2, "need at least 2 blocks to build d_i's");

        // Prefix sums: dj_size = 2^(d_1 + ... + d_j) at level j.
        let prefix_vars_vec: Vec<usize> = {
            let mut prefix_vars: usize = 0;
            dimensions
                .iter()
                .take(k - 1)
                .map(|&dim| {
                    prefix_vars += dim;
                    prefix_vars
                })
                .collect()
        };

        // Per-level metadata: (dj_size, go_sparse). When
        // `nnz · SPARSE_DENOM < dj_size`, the level's row is stored as
        // `AuxRow::Sparse` (BTreeMap keyed by prefix); otherwise as
        // `AuxRow::Dense` (Vec<G1Affine> of length dj_size, with empty
        // cells filled with the affine zero).
        const SPARSE_DENOM: usize = 4;
        let nnz = polynomial.evaluations.len();
        let level_meta: Vec<(usize, bool)> = (0..k - 1)
            .map(|j| {
                let dj_size = 1usize << prefix_vars_vec[j];
                let go_sparse = nnz.saturating_mul(SPARSE_DENOM) < dj_size;
                (dj_size, go_sparse)
            })
            .collect();

        // Step 1: build per-level *flat* arenas. Each level gets one
        // backing `Vec<G1Affine>` + one `Vec<ScalarField>` sized to nnz,
        // plus a `Vec<(prefix, Range<usize>)>` describing where each
        // cell's slice lives. The previous implementation allocated
        // *two* Vecs per cell (~21k allocations for nnz=2546 at k=10,
        // ~240k at nnz=63583), which dominated the per-cell wall time
        // — per-cell µs grew superlinearly with workload, classic
        // allocator-pressure signature. Three Vecs per level (≈27 for
        // k=10) is bounded irrespective of nnz.
        struct LevelArena<E: Pairing> {
            flat_bases: Vec<E::G1Affine>,
            flat_scalars: Vec<E::ScalarField>,
            /// `(prefix, range_into_flat_*)` per non-empty cell, ordered
            /// by prefix so downstream Sparse rows have canonical order.
            cells: Vec<(usize, std::ops::Range<usize>)>,
        }
        let level_arenas: Vec<LevelArena<E>> = {
            let _span = tracing::info_span!(
                "KZH::CompAux::BucketSort",
                k = k - 1,
                nnz = nnz
            )
            .entered();
            let mut arenas: Vec<LevelArena<E>> = Vec::with_capacity(k - 1);
            for j in 0..k - 1 {
                let prefix_var = prefix_vars_vec[j];
                let rem_vars = polynomial.num_vars() - prefix_var;
                let mask = if rem_vars == 0 { 0 } else { (1usize << rem_vars) - 1 };
                let h_slice = prover_param.get_h_tensors()[j + 1]
                    .as_slice_memory_order()
                    .expect("H_t must be contiguous (standard layout)");

                // Pass 1: compute (prefix, local_idx, value) per nz.
                let mut entries: Vec<(usize, usize, E::ScalarField)> = polynomial
                    .evaluations
                    .iter()
                    .map(|(&gidx, &v)| {
                        let prefix = if rem_vars == 0 { 0 } else { gidx >> rem_vars };
                        let local_idx = gidx & mask;
                        (prefix, local_idx, v)
                    })
                    .collect();

                // Pass 2: sort by prefix → cells become contiguous.
                entries.sort_unstable_by_key(|e| e.0);

                // Pass 3: pack into flat buffers, record cell ranges.
                let n = entries.len();
                let mut flat_bases: Vec<E::G1Affine> = Vec::with_capacity(n);
                let mut flat_scalars: Vec<E::ScalarField> = Vec::with_capacity(n);
                let mut cells: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
                let mut i = 0;
                while i < n {
                    let p = entries[i].0;
                    let start = flat_bases.len();
                    while i < n && entries[i].0 == p {
                        flat_bases.push(h_slice[entries[i].1]);
                        flat_scalars.push(entries[i].2);
                        i += 1;
                    }
                    cells.push((p, start..flat_bases.len()));
                }

                arenas.push(LevelArena {
                    flat_bases,
                    flat_scalars,
                    cells,
                });
            }
            arenas
        };

        // Step 2: flatten cells across levels into one work list.
        // Items: `(level_j, prefix, range_into_arena)`. The MSM at each
        // item produces aux[level_j][prefix].
        let flat: Vec<(usize, usize, std::ops::Range<usize>)> = {
            let _span = tracing::info_span!("KZH::CompAux::Flatten").entered();
            let mut flat = Vec::new();
            for (j, arena) in level_arenas.iter().enumerate() {
                for (prefix, range) in &arena.cells {
                    flat.push((j, *prefix, range.clone()));
                }
            }
            flat
        };

        // Cell-size profiling for the ParallelMSM workload. Emits one
        // info event per CompAux call with the histogram + percentiles
        // of MSM input sizes — exposes whether the workload is
        // dominated by tiny cells (where naive wins), medium cells
        // (where Pippenger might help if we could dodge nested-rayon),
        // or one giant cell that serialises the tail. Cheap: just a
        // sort + bucket count over `flat`, runs once per call.
        {
            let mut sizes: Vec<usize> = flat
                .iter()
                .map(|(_, _, r)| r.end - r.start)
                .collect();
            sizes.sort_unstable();
            let n = sizes.len();
            let sum: usize = sizes.iter().sum();
            let max = sizes.last().copied().unwrap_or(0);
            let p50 = if n > 0 { sizes[n / 2] } else { 0 };
            let p90 = if n > 0 { sizes[(n * 9) / 10] } else { 0 };
            let p99 = if n > 0 { sizes[(n * 99) / 100] } else { 0 };
            let mut h = [0usize; 8]; // 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65+
            for &s in &sizes {
                let b = match s {
                    0..=1 => 0,
                    2 => 1,
                    3..=4 => 2,
                    5..=8 => 3,
                    9..=16 => 4,
                    17..=32 => 5,
                    33..=64 => 6,
                    _ => 7,
                };
                h[b] += 1;
            }
            tracing::info!(
                target: "akd_core::aegon_crypto::pcs::kzhk",
                cells = n,
                sum_size = sum,
                max_size = max,
                p50 = p50,
                p90 = p90,
                p99 = p99,
                h1 = h[0],
                h2 = h[1],
                h3_4 = h[2],
                h5_8 = h[3],
                h9_16 = h[4],
                h17_32 = h[5],
                h33_64 = h[6],
                h65p = h[7],
                "ParallelMSM cell-size profile"
            );
        }

        // Step 3: hybrid-dispatch sweep across all (level, prefix)
        // pairs.
        //
        // Cell-size profiling on the publish workload (see profile
        // emitted just above) shows the distribution is heavily
        // long-tailed: ~83% of cells are size 1, p99 is ~14, but max
        // can run to several hundred — a few fat cells contain most of
        // the actual scalar-mul work. A uniform `naive_msm` over all
        // cells (the previous strategy) leaves performance on the
        // table at the fat tail; uniform Pippenger via `msm()` from
        // inside the outer `par_iter` blew the rayon worker stack on
        // the prefill workload because nested rayon pools accumulate
        // frames.
        //
        // Resolution: split the work by cell size, using the strategy
        // that wins per regime, and avoid nesting rayon pools.
        //
        //   Phase A — small cells (size < SEQ_PIPP_THRESHOLD).
        //     Run them in parallel via `cfg_iter!` + `naive_msm`.
        //     `naive_msm` has near-zero per-call overhead, the cells
        //     are tiny, and rayon's outer parallelism is the only
        //     parallelism source — exactly the regime where the
        //     previous design was already optimal.
        //
        //   Phase B — large cells (size ≥ SEQ_PIPP_THRESHOLD).
        //     Loop *sequentially* and call `msm()` per cell. Each
        //     `msm()` is free to spin up its own rayon pool (Pippenger
        //     against `THREAD_TABLE` width) because we're not inside
        //     a `par_iter` anymore — no nested-rayon stack accumulation.
        //     For these cells Pippenger is much faster than naive
        //     (per calibration: ~3-9× at sizes 32-256+), and the cell
        //     count is small (typically ≤ 100 on the publish path),
        //     so the sequential outer is fine.
        //
        // Threshold = 32 was picked from the calibration table:
        // sequential `msm()` only beats per-cell-of-the-parallel-pool
        // (naive_cost / num_cores) once n ≳ 32 on n2-standard-4. Below
        // 32, the parallel-naive path's amortised cost is smaller than
        // a single Pippenger call even though Pippenger is faster
        // per-call than naive.
        const SEQ_PIPP_THRESHOLD: usize = 32;

        // Phase A: parallel naive over the small cells. The closure
        // emits `E::G1::zero()` as a placeholder for large cells so
        // we keep one flat output buffer aligned with `flat`. The
        // sequential phase below overwrites those placeholders.
        let mut projectives: Vec<E::G1> = {
            let _span = tracing::info_span!(
                "KZH::CompAux::ParallelMSM",
                cells = flat.len()
            )
            .entered();
            cfg_iter!(flat)
                .map(|(j, _prefix, range)| {
                    let arena = &level_arenas[*j];
                    let bases = &arena.flat_bases[range.start..range.end];
                    let scalars = &arena.flat_scalars[range.start..range.end];
                    let n = bases.len();
                    if n == 0 {
                        E::G1::zero()
                    } else if n == 1 {
                        bases[0] * scalars[0]
                    } else if n < SEQ_PIPP_THRESHOLD {
                        naive_msm::<E::G1>(bases, scalars)
                    } else {
                        // Big cell — leave a placeholder. Calling
                        // `msm()` here would nest rayon pools inside
                        // the outer `par_iter` and overflow worker
                        // stacks (this is the historical failure
                        // mode that forced uniform-naive in the first
                        // place).
                        E::G1::zero()
                    }
                })
                .collect()
        };

        // Phase B: sequential Pippenger over the large cells. We're
        // outside any `par_iter` here, so each `msm()` call may safely
        // install its own thread pool. The sequential outer loop
        // means at most one Pippenger pool is active at a time —
        // bounded stack usage, no nesting.
        {
            let large_count = flat
                .iter()
                .filter(|(_, _, r)| r.end - r.start >= SEQ_PIPP_THRESHOLD)
                .count();
            let _span = tracing::info_span!(
                "KZH::CompAux::SequentialPippenger",
                cells = large_count
            )
            .entered();
            for (i, (j, _prefix, range)) in flat.iter().enumerate() {
                let len = range.end - range.start;
                if len < SEQ_PIPP_THRESHOLD {
                    continue;
                }
                let arena = &level_arenas[*j];
                let bases = &arena.flat_bases[range.start..range.end];
                let scalars = &arena.flat_scalars[range.start..range.end];
                projectives[i] = msm::<E::G1>(bases, scalars);
            }
        }

        // Step 4: one batch normalization over every non-empty cell
        // across every level.
        let affines = {
            let _span = tracing::info_span!(
                "KZH::CompAux::NormalizeBatch",
                cells = projectives.len()
            )
            .entered();
            <E::G1 as CurveGroup>::normalize_batch(&projectives)
        };

        // Step 5: rebuild per-level rows from the flat results.
        let d_bool: Vec<AuxRow<E>> = {
            let _span = tracing::info_span!("KZH::CompAux::RebuildRows").entered();
            let mut sparse_entries: Vec<BTreeMap<usize, E::G1Affine>> =
                (0..k - 1).map(|_| BTreeMap::new()).collect();
            for ((j, prefix, _), aff) in flat.iter().zip(affines.iter()) {
                sparse_entries[*j].insert(*prefix, *aff);
            }

            sparse_entries
                .into_iter()
                .enumerate()
                .map(|(j, entries)| {
                    let (dj_size, go_sparse) = level_meta[j];
                    if go_sparse {
                        AuxRow::Sparse {
                            len: dj_size,
                            entries,
                        }
                    } else {
                        let mut dense = vec![E::G1Affine::zero(); dj_size];
                        for (prefix, aff) in entries {
                            dense[prefix] = aff;
                        }
                        AuxRow::Dense(dense)
                    }
                })
                .collect()
        };

        state.set_d_bool(d_bool);
        Ok(())
    }

    /// Dense non-Boolean opening path of Figure 14: at each level `j`,
    /// commits `f`'s current dense partial-evaluation table chunk-wise
    /// against `H_{j+1}` to produce `D_j`, then reduces the polynomial
    /// by `fix_last_variables` on block `j`.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Open_Dense")]
    fn open_dense_non_bool_inner(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        polynomial: &DenseMultilinearExtension<E::ScalarField>,
        point: &[E::ScalarField],
        _state: &KZHKState<E>,
    ) -> Result<(KZHKOpeningProof<E>, E::ScalarField), PCSError> {
        let prover_param: &KZHKProverParam<E> = prover_param.borrow();
        let mut d = Vec::new();
        let k = prover_param.get_dimensions().len();
        let decomposed_point = KZHK::<E>::decompose_point(prover_param.get_dimensions(), point);
        let mut partial_polynomial = polynomial.clone();
        for (j, point_part) in decomposed_point.iter().take(k - 1).enumerate() {
            let partial_polynomial_evals = &partial_polynomial.evaluations;
            // Now start iterating over the boolean partial evaluations
            let num_chunks = 1 << prover_param.get_dimensions()[j];
            assert_eq!(partial_polynomial_evals.len() % num_chunks, 0);
            let chunk_len: usize = partial_polynomial_evals.len() / num_chunks; // = 2^(n-r)
            debug_assert!(chunk_len > 0);
            let h_slice = prover_param.get_h_tensors()[j + 1]
                .as_slice_memory_order()
                .expect("H_t must be contiguous");
            // immutable
            let dj: Vec<E::G1Affine> = cfg_into_iter!(0..num_chunks)
                .map(|i| {
                    let off = i * chunk_len;
                    let chunk = &partial_polynomial_evals[off..off + chunk_len];
                    msm::<E::G1>(h_slice, chunk).into_affine()
                })
                .collect();
            d.push(dj);

            partial_polynomial = fix_last_variables(&partial_polynomial, point_part);
        }
        let f = DenseOrSparseMLE::Dense(partial_polynomial.clone());
        let eval = fix_last_variables(&partial_polynomial, &decomposed_point[k - 1])[0];
        Ok((KZHKOpeningProof::new(d, f, None, None, None), eval))
    }

    /// Dense Boolean opening: the "free opening" shortcut. When the
    /// point is Boolean, each `D_j` is a contiguous slice of the
    /// precomputed auxiliary table `aux_d_bool[j]` — no group
    /// operations are needed, only indexing.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Open_Dense_Boolean")]
    fn open_dense_bool_inner(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        polynomial: &DenseMultilinearExtension<E::ScalarField>,
        point: &[E::ScalarField],
        state: &KZHKState<E>,
    ) -> Result<(KZHKOpeningProof<E>, E::ScalarField), PCSError> {
        let prover_param: &KZHKProverParam<E> = prover_param.borrow();

        let aux_d_bool = state.get_d_bool();
        let mut d: Vec<Vec<E::G1Affine>> = Vec::new();

        let dims = prover_param.get_dimensions();
        let k = dims.len();

        let decomposed_point = KZHK::<E>::decompose_point(dims, point);
        let mut partial_polynomial = polynomial.clone();

        // Track the integer encoding of the already-fixed (prefix) boolean blocks,
        // little-endian.
        let mut eb: usize = 0;

        for (j, partial_point) in decomposed_point.iter().take(k - 1).enumerate() {
            let block_dim = dims[j];
            let start = eb << block_dim; // == eb * 2^{block_dim}
            let end = start + (1 << block_dim);

            let aux_vec = &aux_d_bool[j];
            debug_assert!(end <= aux_vec.len(), "state slice OOB");

            // `AuxRow::slice` gives the proof's `D_j` directly — for
            // a Dense row this is a `to_vec` memcpy; for a Sparse row
            // it's `2^{d_j}` map lookups, defaulting to affine zero.
            d.push(aux_vec.slice(start..end));

            // Reduce the dense polynomial on this boolean block (sequential dependency)
            partial_polynomial = fix_last_variables_boolean(&partial_polynomial, partial_point);

            // Update eb to include this block for the next iteration:
            // new_bits = [partial_point || old_bits] (LE), so:
            // eb_next = bits_le(partial_point) + (eb << block_dim)
            let s = bits_le_to_usize(partial_point);
            eb = s + (eb << block_dim);
        }

        let f = DenseOrSparseMLE::Dense(partial_polynomial.clone());
        let eval = fix_last_variables_boolean(&partial_polynomial, &decomposed_point[k - 1])[0];

        Ok((KZHKOpeningProof::new(d, f, None, None, None), eval))
    }
    /// Sparse non-Boolean opening: same level structure as the dense
    /// variant but each per-chunk MSM only sees the non-zero entries
    /// of the current sparse partial polynomial within that window.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Open_Sparse")]
    fn open_sparse_non_bool_inner(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        polynomial: &SparseMultilinearExtension<E::ScalarField>,
        point: &[E::ScalarField],
        _state: &KZHKState<E>,
    ) -> Result<(KZHKOpeningProof<E>, E::ScalarField), PCSError> {
        let prover_param: &KZHKProverParam<E> = prover_param.borrow();
        let mut d = Vec::new();
        let k = prover_param.get_dimensions().len();
        let decomposed_point = KZHK::<E>::decompose_point(prover_param.get_dimensions(), point);
        let mut partial_polynomial = polynomial.clone();

        for (j, point_part) in decomposed_point.iter().take(k - 1).enumerate() {
            // Same partitioning as dense:
            let num_chunks = 1usize << prover_param.get_dimensions()[j]; // 2^{block_j}
            let chunk_len =
                1usize << (partial_polynomial.num_vars - prover_param.get_dimensions()[j]); // 2^{remaining - block_j}
            let domain_len = 1usize << partial_polynomial.num_vars;
            debug_assert_eq!((num_chunks * chunk_len), domain_len);

            let h_slice = prover_param.get_h_tensors()[j + 1]
                .as_slice_memory_order()
                .expect("H_t must be contiguous");
            debug_assert_eq!(h_slice.len(), chunk_len);

            // Iterate windows in increasing "x-index" order (matches dense & verifier eq
            // order)
            let mut dj = vec![E::G1Affine::zero(); num_chunks];
            cfg_iter_mut!(dj).enumerate().for_each(|(i, d_j_i)| {
                let base = i * chunk_len;
                // Gather non-zeros in [base, base+chunk_len) and rebase to local [0..chunk_len)
                let mut bases = Vec::new();
                let mut scalars = Vec::new();
                for (&gidx, &val) in partial_polynomial.evaluations.range(base..base + chunk_len) {
                    let local = gidx - base;
                    bases.push(h_slice[local]);
                    scalars.push(val);
                }
                let acc = if scalars.is_empty() {
                    E::G1Affine::zero()
                } else {
                    msm::<E::G1>(&bases, &scalars).into_affine()
                };
                *d_j_i = acc;
            });
            d.push(dj);

            // Reduce the last block by the point part (must match dense orientation)
            partial_polynomial = fix_last_variables_sparse(&partial_polynomial, point_part);
        }

        let f = DenseOrSparseMLE::Sparse(partial_polynomial.clone());
        let eval = fix_last_variables_sparse(&partial_polynomial, &decomposed_point[k - 1])[0];
        Ok((KZHKOpeningProof::new(d, f, None, None, None), eval))
    }

    /// Sparse Boolean opening: sparse analogue of
    /// [`Self::open_dense_bool_inner`] — `D_j` is read directly from the
    /// auxiliary table, no cryptographic work per level.
    ///
    /// Implementation note: fixing blocks `0..k-1` to a Boolean prefix
    /// `(b_1, ..., b_{k-1})` is exactly the contiguous slice of
    /// `polynomial.evaluations` at indices `[eb << d_k, (eb+1) << d_k)`
    /// where `eb` packs the prefix bits. We compute `eb` from the aux
    /// reads (no polynomial work in the loop), then materialize the
    /// residual `f` with a single `BTreeMap::range` and look up the
    /// final evaluation with a single `BTreeMap::get`. This avoids the
    /// `O(nnz)` polynomial clone + chained `fix_last_variables_*` calls
    /// that the previous implementation performed at every level.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Open_Sparse_Boolean")]
    fn open_sparse_bool_inner(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        polynomial: &SparseMultilinearExtension<E::ScalarField>,
        point: &[E::ScalarField],
        state: &KZHKState<E>,
    ) -> Result<(KZHKOpeningProof<E>, E::ScalarField), PCSError> {
        let prover_param: &KZHKProverParam<E> = prover_param.borrow();
        let dims = prover_param.get_dimensions();
        let k = dims.len();

        let aux_d_bool = state.get_d_bool();
        let decomposed_point = KZHK::<E>::decompose_point(dims, point);

        // Per-level aux reads only. After the loop, `eb` holds the full
        // packed prefix `[b_1][b_2]...[b_{k-1}]` (LE, MSB-first across
        // blocks).
        let mut d: Vec<Vec<E::G1Affine>> = Vec::with_capacity(k - 1);
        let mut eb: usize = 0;
        for (j, partial_point) in decomposed_point.iter().take(k - 1).enumerate() {
            let block_dim = dims[j];
            let start = eb << block_dim;
            let end = start + (1 << block_dim);

            let aux_vec = &aux_d_bool[j];
            debug_assert!(end <= aux_vec.len(), "state slice OOB");

            // `AuxRow::slice` gives the proof's `D_j` directly. For a
            // `Dense` row this is a `to_vec` memcpy; for a `Sparse` row
            // it's `2^{d_j}` `BTreeMap::get`s defaulting to affine zero
            // (cheap because `2^{d_j}` is small — typically 2..16 in
            // KZH-k regimes).
            d.push(aux_vec.slice(start..end));

            eb = bits_le_to_usize(partial_point) + (eb << block_dim);
        }

        // Materialize the residual `f` directly from the original
        // polynomial via one `BTreeMap::range` over the prefix window.
        // Equivalent to the chain of `fix_last_variables_boolean_sparse`
        // calls but touches only `O(nnz / 2^{n - d_k})` entries instead
        // of `O(nnz)`.
        let last_dim = dims[k - 1];
        let win_start = eb << last_dim;
        let win_end = win_start + (1 << last_dim);
        let pairs: Vec<(usize, E::ScalarField)> = polynomial
            .evaluations
            .range(win_start..win_end)
            .map(|(&g, &v)| (g - win_start, v))
            .collect();
        let f_residual = SparseMultilinearExtension::from_evaluations(last_dim, &pairs);

        // The final evaluation is `polynomial(b_1, ..., b_{k-1}, b_k)` —
        // a single Boolean point in the original hypercube. One
        // `BTreeMap::get` instead of another full reduction pass.
        let final_idx = win_start | bits_le_to_usize(&decomposed_point[k - 1]);
        let eval = polynomial
            .evaluations
            .get(&final_idx)
            .copied()
            .unwrap_or_else(E::ScalarField::zero);

        Ok((
            KZHKOpeningProof::new(d, DenseOrSparseMLE::Sparse(f_residual), None, None, None),
            eval,
        ))
    }

    /// Splits a flat evaluation point into `k` block-sized sub-points
    /// matching the SRS's block dimensions `(d_1, ..., d_k)`.
    fn decompose_point(dimensions: &[usize], point: &[E::ScalarField]) -> Vec<Vec<E::ScalarField>> {
        let mut decomposed = Vec::new();
        let mut start = 0;
        for &dim in dimensions {
            let end = start + dim;
            decomposed.push(point[start..end].to_vec());
            start = end;
        }
        decomposed
    }
}

/// Cross‑compat “for_each_with_scratch”: uses `for_each_init` in parallel
/// builds, and a single reusable scratch in sequential builds.
#[macro_export]
macro_rules! cfg_for_each_with_scratch {
    ($iter:expr, $make_scratch:expr, |$scratch:ident, $item:pat_param| $body:block) => {{
        #[cfg(feature = "parallel")]
        {
            ($iter).for_each_init($make_scratch, |$scratch, $item| $body);
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut $scratch = $make_scratch();
            for $item in $iter {
                $body
            }
        }
    }};
}
/// Default choice of the `k` parameter for a polynomial in `poly_size`
/// variables. Returns `poly_size / 2`, which keeps block dimensions near
/// `sqrt(N)` — a reasonable balance between proof size and prover cost
/// for general-purpose use. Callers should supply their own `k` via
/// [`KZHKConfig`] when optimizing for a specific workload.
/// Upper bound on the number of non-zero coefficients of `poly`.
///
/// For a dense polynomial this is `2^num_vars` (the full evaluation table);
/// for a sparse polynomial it's the size of the non-zero coefficient map.
/// `update_state` uses this to decide whether per-chunk / per-bucket MSMs
/// are small enough to run through the naive-MSM fast path in
/// `msm`, in which case the outer loop can safely parallelize.
fn sparsity_of<F: ark_ff::Field>(poly: &DenseOrSparseMLE<F>) -> usize {
    match poly {
        DenseOrSparseMLE::Dense(p) => p.evaluations.len(),
        DenseOrSparseMLE::Sparse(p) => p.evaluations.len(),
    }
}

pub fn compute_k(poly_size: usize, _is_zk: bool) -> usize {
    poly_size / 2
}
