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
use crate::{
    pcs::{
        kzhk::{
            msm::{msm_wrapper_g1, NAIVE_MSM_THRESHOLD},
            srs::{KZHKProverParam, KZHKUniversalParams, KZHKVerifierParam},
            structs::{KZHKState, KZHKCommitment, KZHKConfig, KZHKOpeningProof},
        },
        PCSGlobalParam,
    },
    poly::DenseOrSparseMLE,
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
use transcript::IOPTranscript;
pub mod msm;
pub mod srs;
pub mod structs;
use arithmetic::{
    bits_le_to_usize,
    multilinear_polynomial::{
        fix_last_variables, fix_last_variables_boolean,
        fix_last_variables_boolean_sparse, fix_last_variables_sparse,
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

    /// Generates (or loads) a KZH-k SRS for testing.
    ///
    /// The SRS samples trapdoors `{mu_{b,j}}` for each of the `k` blocks and
    /// builds the tensor families `H_1, ..., H_k` in `G1` and the pairing
    /// elements `V_{b,j}` in `G2` described in Figure 14. Because SRS
    /// generation is expensive (multiple MSMs of size `N`), the result is
    /// cached on disk under `../artifacts/srs/srs_{k}_{supported_size}.bin`
    /// and reused on subsequent invocations of the same size.
    fn gen_srs_for_testing<R: Rng>(
        conf: Self::Config,
        _rng: &mut R,
        supported_size: usize,
    ) -> Result<Self::SRS, PCSError> {
        let k = conf.k;
        let zk = conf.zk;
        // SRS is cached on disk keyed by (k, num_vars) because generating it
        // requires k full-size MSMs in G1 — reusing across test runs saves
        // significant time.
        let srs_path = current_dir()
            .unwrap()
            .join(format!("../artifacts/srs/srs_{:?}_{}.bin", k, supported_size));
        let srs = if srs_path.exists() {
            eprintln!("Loading SRS");
            let mut buffer = Vec::new();
            BufReader::new(File::open(&srs_path).unwrap())
                .read_to_end(&mut buffer)
                .unwrap();
            Self::SRS::deserialize_uncompressed_unchecked(&buffer[..]).unwrap_or_else(|_| {
                panic!("Failed to deserialize SRS from {:?}", srs_path);
            })
        } else {
            eprintln!("Computing SRS");
            let mut rng = test_rng();
            let srs =
                KZHKUniversalParams::gen_srs_for_testing(&mut rng, k, zk, supported_size).unwrap();
            let mut serialized = Vec::new();
            srs.serialize_uncompressed(&mut serialized).unwrap();
            if let Some(parent) = srs_path.parent() {
                create_dir_all(parent).unwrap_or_else(|_| {
                    panic!("could not create directory for SRS at {:?}", parent)
                });
            }
            BufWriter::new(
                File::create(srs_path.clone())
                    .unwrap_or_else(|_| panic!("could not create file for SRS at {:?}", srs_path)),
            )
            .write_all(&serialized)
            .unwrap();
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

    /// Produces an opening proof `pi = ({D_j}_{j=1}^{k-1}, f_{x_1..x_{k-1}})`
    /// of `f` at the point `(x_1, ..., x_k)` and the evaluation `y = f(x)`.
    /// Dispatches to the zk Sigma-protocol variant (Appendix D) when the
    /// SRS is hiding.
    #[tracing::instrument(level = "debug", skip_all, name = "KZH::Open")]
    fn open(
        prover_param: impl Borrow<Self::ProverParam>,
        commitment: &Self::Commitment,
        polynomial: &Self::Polynomial,
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
        polynomials: &[&Self::Polynomial],
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
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        let result = match (proof.get_r_hide(), proof.get_y_r(), proof.get_rho_prime()) {
            (Some(_), Some(_), Some(_)) => {
                Self::verify_zk(verifier_param, commitment, point, value, None, proof)
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
        polynomial: &DenseOrSparseMLE<E::ScalarField>,
        point: &[E::ScalarField],
        state: &KZHKState<E>,
    ) -> Result<(KZHKOpeningProof<E>, E::ScalarField), PCSError> {
        //TODO: Make the iters here parallel
        let is_boolean_point = point.iter().all(|&x| x.is_zero() || x.is_one());
        let result = match (is_boolean_point, polynomial) {
            (true, DenseOrSparseMLE::Dense(poly)) => {
                Self::open_dense_bool_inner(prover_param, poly, point, state)
            },
            (true, DenseOrSparseMLE::Sparse(poly)) => {
                Self::open_sparse_bool_inner(prover_param, poly, point, state)
            },
            (false, DenseOrSparseMLE::Dense(poly)) => {
                Self::open_dense_non_bool_inner(prover_param, poly, point, state)
            },
            (false, DenseOrSparseMLE::Sparse(poly)) => {
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
        polynomial: &DenseOrSparseMLE<E::ScalarField>,
        point: &[E::ScalarField],
        state: &KZHKState<E>,
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(KZHKOpeningProof<E>, E::ScalarField), PCSError> {
        let prover_param: &KZHKProverParam<E> = prover_param.borrow();
        // The zk path
        let (non_zk_opening, non_zk_value) =
            Self::open_non_zk(prover_param, commitment, polynomial, point, state)?;
        // Sampling the sparse polynomial r(X)
        let r_poly: SparseMultilinearExtension<E::ScalarField> = rand_sparse_mle(
            polynomial.num_vars(),
            prover_param.get_hiding_sparsity().unwrap(),
            &mut test_rng(),
        );
        let r_poly_wrapped = DenseOrSparseMLE::Sparse(r_poly.clone());
        // Commit to the sparse masking polynomial r(X), then build its
        // Boolean-opening state. The opening of r itself is plain (non-zk):
        // r is the mask, it doesn't need to be re-masked.
        let (r_hide, mut r_state) = Self::commit(prover_param, &r_poly_wrapped)?;
        let rho = *r_state.get_tau();
        Self::update_state(prover_param, &r_poly_wrapped, &r_hide, &mut r_state)?;
        let (r_opening, y_r) =
            Self::open_non_zk(prover_param, &r_hide, &r_poly_wrapped, point, &r_state)?;
        // Sigma-protocol challenge (currently fixed to 1; a transcript-derived
        // Fiat-Shamir challenge would replace this in a production setting).
        let alpha = E::ScalarField::one();
        // rho_prime = alpha*tau + rho derandomizes the hiding offset so that
        // `alpha*C_hide + R_hide - rho_prime * h` equals the non-hiding
        // commitment to `alpha*f + r`.
        let rho_prime = alpha * state.get_tau() + rho;
        let mut output_opening = non_zk_opening * alpha + r_opening;
        output_opening.set_r_hide(r_hide);
        output_opening.set_y_r(y_r);
        output_opening.set_rho_prime(rho_prime);
        let result = Ok((output_opening, non_zk_value));
        result
    }

    /// Batched opening by plain sum (no random linear combination).
    ///
    /// NOTE: not sound as a proof-of-knowledge batch opener because it
    /// lacks random challenges — kept for benchmarking and as a baseline.
    fn multi_open_non_zk(
        prover_param: impl Borrow<KZHKProverParam<E>>,
        commitment: &KZHKCommitment<E>,
        polynomials: &[&DenseOrSparseMLE<E::ScalarField>],
        point: &Vec<E::ScalarField>,
        states: &[KZHKState<E>],
        _transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(KZHKOpeningProof<E>, E::ScalarField), PCSError> {
        let num_vars = point.len();
        let mut aggr_state: KZHKState<E> = KZHKState::default();
        let (agg_poly, aggr_state) = match polynomials[0] {
            DenseOrSparseMLE::Dense(_) => {
                let mut aggr_poly = DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    vec![E::ScalarField::zero(); 1usize << num_vars],
                );
                for (poly, state) in polynomials.iter().zip(states.iter()) {
                    if let DenseOrSparseMLE::Dense(dense_poly) = poly {
                        aggr_poly += dense_poly;
                        aggr_state = aggr_state + state.clone();
                    } else {
                        panic!("All polynomials must be dense here");
                    }
                }
                (DenseOrSparseMLE::Dense(aggr_poly), aggr_state)
            },
            DenseOrSparseMLE::Sparse(_) => {
                let mut aggr_poly =
                    SparseMultilinearExtension::from_evaluations(num_vars, Vec::new());
                for (poly, state) in polynomials.iter().zip(states.iter()) {
                    if let DenseOrSparseMLE::Sparse(sparse_poly) = poly {
                        aggr_poly += sparse_poly;
                        aggr_state = aggr_state + state.clone();
                    } else {
                        panic!("All polynomials must be sparse here");
                    }
                }

                (DenseOrSparseMLE::Sparse(aggr_poly), aggr_state)
            },
        };
        Self::open_non_zk(prover_param, commitment, &agg_poly, point, &aggr_state)
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
    ) -> Result<bool, PCSError> {
        let alpha = E::ScalarField::one();
        let c_lin = (commitment.get_commitment().into_group() * alpha
            + proof.get_r_hide().unwrap().get_commitment().into_group()
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
            debug_assert!(prod.is_zero());

            let eq_poly = build_eq_x_r(point_part).unwrap();
            cj = msm_wrapper_g1::<E>(&proof.get_d()[j], &eq_poly.evaluations).into_affine();
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
        assert_eq!(cj, alleged_last_cj);
        drop(cj_check_guard);
        // Evaluation Check
        let eval_check_span = tracing::debug_span!("KZH::Verify::EvalCheck");
        let eval_check_guard = eval_check_span.enter();
        let _p = match proof.get_f() {
            DenseOrSparseMLE::Dense(f) => {
                fix_last_variables(f, &decomposed_point[k - 1])[0] == *value
            },
            DenseOrSparseMLE::Sparse(f) => {
                fix_last_variables_sparse(f, &decomposed_point[k - 1])[0] == *value
            },
        };
        drop(eval_check_guard);
        Ok(true)
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
            msm_wrapper_g1::<E>(&bases, &scalars)
        } else {
            msm_wrapper_g1::<E>(h_bases, &poly.evaluations)
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
        let com = msm_wrapper_g1::<E>(&bases, &scalars);
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

        let mut d_bool: Vec<Vec<E::G1Affine>> = Vec::with_capacity(k - 1);
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
            // is small enough that `msm_wrapper_g1` takes its naive path
            // (no nested rayon pool inside arkworks' Pippenger). For dense
            // inputs, `eval_len` is a tight bound on the per-chunk MSM size.
            let per_msm_size = eval_len;
            let parallel_outer_safe = per_msm_size <= NAIVE_MSM_THRESHOLD;
            let mut d_j = vec![E::G1Affine::zero(); dj_size];
            if parallel_outer_safe {
                cfg_iter_mut!(d_j).enumerate().for_each(|(i, d_j_i)| {
                    let scalars =
                        partially_eval_dense_poly_on_bool_point(polynomial, i, eval_len);
                    *d_j_i = msm_wrapper_g1::<E>(h_slice, scalars.as_slice()).into_affine();
                });
            } else {
                d_j.iter_mut().enumerate().for_each(|(i, d_j_i)| {
                    let scalars =
                        partially_eval_dense_poly_on_bool_point(polynomial, i, eval_len);
                    *d_j_i = msm_wrapper_g1::<E>(h_slice, scalars.as_slice()).into_affine();
                });
            }

            d_bool.push(d_j);
        }
        state.set_d_bool(d_bool);
        Ok(())
    }

    /// Sparse counterpart of [`Self::update_state_dense`]: exploits the
    /// sparse coefficient map so that each per-chunk MSM sees only the
    /// non-zero entries falling in its Boolean window.
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

        // Build prefix sums of dimensions up to each block (exclusive of the last)
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

        // Compute the Boolean aux table d_bool. For each level
        // `j = 0..k-1` we must produce `d_j[i] = <f(b_1..b_j=i, X_{j+1..k}), H_{j+1}>`
        // for every Boolean prefix `i ∈ [0, 2^{prefix_var})` (Figure 14).
        //
        // The previous implementation called `partially_eval_sparse_poly_on_bool_point`
        // once per prefix, doing `dj_size` BTreeMap range seeks. For highly sparse
        // polynomials that was wasteful (each seek is `O(log nnz)` and yields few
        // entries). Instead we:
        //   1. Single-pass bucket the sparse entries by prefix in `O(nnz)` — each
        //      non-zero at global index `g` lives in exactly one prefix `g >> rem_vars`
        //      with local index `g & mask`.
        //   2. Run the per-prefix MSMs in parallel (they're independent and use the
        //      same `H_{j+1}` bases).
        //
        // Both the outer level loop and inner bucket loop stay sequential;
        // the parallelism comes from the per-bucket MSM itself. See the
        // comment on step (2) below.
        let d_bool: Vec<Vec<E::G1Affine>> = {
            (0..k - 1)
                .map(|j| {
                    let prefix_var = prefix_vars_vec[j];
                    // `dj_size` = number of Boolean prefixes at this level.
                    // `rem_vars` = variables left after fixing the prefix; `eval_len`
                    // is the size of each per-prefix partial evaluation.
                    let dj_size = 1usize << prefix_var;
                    let rem_vars = polynomial.num_vars() - prefix_var;
                    let mask = (1usize << rem_vars) - 1;

                    // `H_{j+1}` is the tensor used for this level's inner commitments.
                    let h_slice = prover_param.get_h_tensors()[j + 1]
                        .as_slice_memory_order()
                        .expect("H_t must be contiguous (standard layout)");

                    // (1) Bucket non-zero entries by prefix in a single pass.
                    // BTreeMap iteration is sequential and O(nnz); avoids the
                    // per-prefix range seeks the old implementation was doing.
                    let mut buckets: Vec<(Vec<E::G1Affine>, Vec<E::ScalarField>)> =
                        (0..dj_size).map(|_| (Vec::new(), Vec::new())).collect();
                    for (&global_idx, &value) in polynomial.evaluations.iter() {
                        let prefix = global_idx >> rem_vars;
                        let local_idx = global_idx & mask;
                        let (bases, scalars) = &mut buckets[prefix];
                        bases.push(h_slice[local_idx]);
                        scalars.push(value);
                    }

                    // (2) Run the per-prefix MSMs. What matters for the
                    // outer parallelism decision is the *max per-bucket*
                    // size, not `c`: when the non-zeros spread across many
                    // buckets (typical for random sparse inputs, e.g. the
                    // Appendix-D masking polynomial with `c = k·2^(n/k)`
                    // spread over `2^(n/k)` prefixes → ~`k` per bucket),
                    // every individual MSM is tiny and takes the naive
                    // path in `msm_wrapper_g1` — no arkworks Pippenger, no
                    // nested rayon pool build — so it's safe to
                    // parallelize the outer bucket loop. One extra
                    // O(dj_size) scan to find the max is negligible next
                    // to the MSM work.
                    let max_bucket =
                        buckets.iter().map(|(_, s)| s.len()).max().unwrap_or(0);
                    let parallel_outer_safe = max_bucket <= NAIVE_MSM_THRESHOLD;

                    // Compute all per-bucket MSMs in PROJECTIVE form first,
                    // then fold them into affine in a single batched
                    // normalization. One batched `normalize_batch` amortizes
                    // Montgomery's trick over all `dj_size` points — a
                    // single field inversion total instead of one per bucket.
                    // Field inversion is ~100× more expensive than field
                    // multiplication on BN254, so for `dj_size = 1024` this
                    // shaves roughly `1023` inversions off the critical path.
                    let projective: Vec<E::G1> = if parallel_outer_safe {
                        cfg_into_iter!(buckets)
                            .map(|(bases, scalars)| {
                                if scalars.is_empty() {
                                    E::G1::zero()
                                } else {
                                    msm_wrapper_g1::<E>(&bases, &scalars)
                                }
                            })
                            .collect()
                    } else {
                        buckets
                            .into_iter()
                            .map(|(bases, scalars)| {
                                if scalars.is_empty() {
                                    E::G1::zero()
                                } else {
                                    msm_wrapper_g1::<E>(&bases, &scalars)
                                }
                            })
                            .collect()
                    };
                    <E::G1 as CurveGroup>::normalize_batch(&projective)
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
                    msm_wrapper_g1::<E>(h_slice, chunk).into_affine()
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

            // Parallel clone of the state slice -> d_j
            let d_j: Vec<E::G1Affine> = cfg_iter!(aux_vec[start..end]).cloned().collect();

            d.push(d_j);

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
                    msm_wrapper_g1::<E>(&bases, &scalars).into_affine()
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

        let mut d: Vec<Vec<E::G1Affine>> = Vec::with_capacity(k - 1);
        let mut partial_polynomial = polynomial.clone();

        // eb encodes the already-fixed boolean prefix in little-endian
        let mut eb: usize = 0;

        for (j, partial_point) in decomposed_point.iter().take(k - 1).enumerate() {
            let block_dim = dims[j];
            let start = eb << block_dim; // == eb * 2^{block_dim}
            let end = start + (1 << block_dim);

            let aux_vec = &aux_d_bool[j];
            debug_assert!(end <= aux_vec.len(), "state slice OOB");

            // Parallel clone of state slice -> d_j
            let d_j: Vec<E::G1Affine> = cfg_iter!(aux_vec[start..end]).cloned().collect();

            d.push(d_j);

            // Reduce the last block on the sparse polynomial (sequential dependency)
            partial_polynomial =
                fix_last_variables_boolean_sparse(&partial_polynomial, partial_point);

            // Update eb to include this block for next iteration:
            // new_bits = [partial_point || old_bits] (LE)
            let s = bits_le_to_usize(partial_point);
            eb = s + (eb << block_dim);
        }

        let f = DenseOrSparseMLE::Sparse(partial_polynomial.clone());
        let eval =
            fix_last_variables_boolean_sparse(&partial_polynomial, &decomposed_point[k - 1])[0];

        Ok((KZHKOpeningProof::new(d, f, None, None, None), eval))
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
/// `msm_wrapper_g1`, in which case the outer loop can safely parallelize.
fn sparsity_of<F: ark_ff::Field>(poly: &DenseOrSparseMLE<F>) -> usize {
    match poly {
        DenseOrSparseMLE::Dense(p) => p.evaluations.len(),
        DenseOrSparseMLE::Sparse(p) => p.evaluations.len(),
    }
}

pub fn compute_k(poly_size: usize, _is_zk: bool) -> usize {
    poly_size / 2
}
