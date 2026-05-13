use std::marker::PhantomData;

use ark_ec::pairing::Pairing;
use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme;
use akd_core::aegon_crypto::poly::DenseOrSparseMLE;

pub type Label = Vec<u8>;
pub type Value = Vec<u8>;

/// Trait alias for the PCS shape Aegon needs everywhere. Avoids
/// repeating the where clause on every type.
///
/// Note: where Aegon also needs the prover/verifier params to expose
/// `PCSGlobalParam::is_zk()` (e.g. the runtime `private` cross-check
/// in `Aegon::init`), call sites add that bound explicitly. Rust does
/// not yet auto-propagate supertrait where-clauses, so making it a
/// supertrait bound here would force every downstream impl to repeat
/// the bound anyway.
pub trait AegonPcs<E: Pairing>:
    PolynomialCommitmentScheme<
    E,
    Polynomial = DenseOrSparseMLE<E::ScalarField>,
    Point = Vec<E::ScalarField>,
    Evaluation = E::ScalarField,
>
{
}

impl<E, T> AegonPcs<E> for T
where
    E: Pairing,
    T: PolynomialCommitmentScheme<
        E,
        Polynomial = DenseOrSparseMLE<E::ScalarField>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
    >,
{
}

/// Per-epoch state published on the bulletin board (paper §6.2).
///
/// Four commitments per epoch: the `index` and `value` polynomials Aegon
/// uses for lookups, plus the two `rand` polynomials that anchor the
/// per-user consistency / per-epoch invariance machinery (paper §5.2,
/// §6.1). All four are needed by the auditor; users fetch the rand
/// commitments to verify their slot did not change between two epochs.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct EpochCommitment<E: Pairing, P: AegonPcs<E>> {
    pub epoch: u64,
    pub index_commitment: P::Commitment,
    pub value_commitment: P::Commitment,
    pub rand_index_commitment: P::Commitment,
    pub rand_value_commitment: P::Commitment,
    pub _e: PhantomData<E>,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for EpochCommitment<E, P> {
    fn clone(&self) -> Self {
        Self {
            epoch: self.epoch,
            index_commitment: self.index_commitment.clone(),
            value_commitment: self.value_commitment.clone(),
            rand_index_commitment: self.rand_index_commitment.clone(),
            rand_value_commitment: self.rand_value_commitment.clone(),
            _e: PhantomData,
        }
    }
}

/// Lookup proof returned by `Aegon::lookup` (paper §5.1, Fig. 2). See
/// `verify::verify_lookup` for the post-conditions enforced by the
/// verifier.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LookupProof<E: Pairing, P: AegonPcs<E>> {
    pub ctr0: u64,
    pub probes: Vec<(E::ScalarField, P::Proof)>,
    pub value_evaluation: E::ScalarField,
    pub value_proof: P::Proof,
}

impl<E: Pairing, P: AegonPcs<E>> LookupProof<E, P> {
    pub fn num_probes(&self) -> usize {
        self.probes.len()
    }
}

// ---------- auditor-facing types --------------------------------------

/// One half of the per-epoch invariance proof: attests that for a chain
/// (index or value)
///
/// ```text
///   rand_{n+1} = rand_n + r_n · (poly_{n+1} - poly_n)
/// ```
///
/// holds, by opening all four polynomials at a Fiat-Shamir random point
/// `⃗r` and letting the verifier check the relation on evaluations
/// (paper Remark 2). Generic over the PCS — does not require commitment
/// homomorphism. For homomorphic PCSs a faster check on commitments is
/// possible as a future optimization.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ChainWitness<E: Pairing, P: AegonPcs<E>> {
    pub prev_poly_eval: E::ScalarField,
    pub next_poly_eval: E::ScalarField,
    pub prev_rand_eval: E::ScalarField,
    pub next_rand_eval: E::ScalarField,
    pub prev_poly_proof: P::Proof,
    pub next_poly_proof: P::Proof,
    pub prev_rand_proof: P::Proof,
    pub next_rand_proof: P::Proof,
}

/// Constant-size per-epoch proof that the server's transition from epoch
/// `n` to epoch `n+1` correctly updated both rand polynomials with the
/// Fiat-Shamir scalars derived from the new commitments. Auditors verify
/// this in `audit::verify_invariance`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct InvarianceProof<E: Pairing, P: AegonPcs<E>> {
    pub index_chain: ChainWitness<E, P>,
    pub value_chain: ChainWitness<E, P>,
}

/// The auditor's locally-tracked Fiat-Shamir state. Threaded across calls
/// to `verify_invariance` because each transition's chain randomness is
/// `O(prev_r, new_commitment)` and the auditor must recompute it. The
/// initial state (before the first transition) is `(0, 0)`.
#[derive(Clone, Copy, Debug)]
pub struct AuditState<F: Zero> {
    pub r_index: F,
    pub r_value: F,
}

impl<F: Zero> Default for AuditState<F> {
    fn default() -> Self {
        Self {
            r_index: F::zero(),
            r_value: F::zero(),
        }
    }
}

// ---------- user-facing consistency types ------------------------------

/// A pair of openings of the same polynomial at the same point in two
/// different epochs. The verifier checks both openings and then the
/// equality `eval_s0 == eval_s1`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RandPair<E: Pairing, P: AegonPcs<E>> {
    pub eval_s0: E::ScalarField,
    pub proof_s0: P::Proof,
    pub eval_s1: E::ScalarField,
    pub proof_s1: P::Proof,
}

/// Per-user proof that nothing in the user's slot changed between two
/// epochs `s0 < s1`.
///
/// * Index half: openings of `rand_index` at every probe point
///   `H_bits(ctr, label)` for `ctr ∈ [0, ctr0]`. If two of these
///   evaluations agree across `s0` and `s1`, then with overwhelming
///   probability `index_n(x_ctr)` was identical at every intermediate
///   epoch (paper Lemma 1 / §5.2). `ctr0` openings establish that the
///   user's canonical slot did not shift; the final opening at `x_ctr0`
///   establishes that the slot still belongs to the same label.
/// * Value half: a single opening pair of `rand_value` at the user's
///   slot `x_ctr0`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ConsistencyProof<E: Pairing, P: AegonPcs<E>> {
    pub ctr0: u64,
    pub index_witnesses: Vec<RandPair<E, P>>,
    pub value_witness: RandPair<E, P>,
}
