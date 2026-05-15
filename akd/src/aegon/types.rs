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

/// Per-new-label history witness emitted by a shard during
/// `publish_phase_2` (paper §6.4). For every brand-new slot the publish
/// batch touched on this shard, the shard opens:
///
/// * `rand_index` and `rand_value` **before** the rand-poly update
///   (i.e. at the prior epoch). For a fresh slot these openings are
///   the polynomials' values at a previously-empty address.
/// * `rand_index` and `rand_value` **after** the rand-poly update, plus
///   `value` (which was committed at the end of phase 1). These three
///   together are the new-epoch view of the slot.
///
/// A future history verifier replays the update equation
/// `rand_X_new(s) − rand_X_old(s) == r_X · (data_X_new(s) − data_X_old(s))`
/// at every slot in this list, using `data_X_old(s) = 0` (the
/// brand-new slot was empty at the prior epoch). `value` is bound
/// directly; the index-side data value is the canonical
/// `H_F(label)` which the verifier reconstructs from the label bytes
/// stored in Redis under `aegon:value:` and `aegon:routing:`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HistoryOpeningEntry<E: Pairing, P: AegonPcs<E>> {
    /// The slot bits this entry's openings are at. Same encoding as
    /// `LabelRouting::final_assignment().1` — bit-vector of length
    /// `shard_log_capacity`, low bit first.
    pub slot_bits: Vec<bool>,
    /// `rand_index_n(slot)` opened against this shard's prior-epoch
    /// `rand_index_commitment`. Will be zero for a brand-new slot,
    /// but the proof binds it to the prior commitment regardless.
    pub rand_index_pre_eval: E::ScalarField,
    pub rand_index_pre_proof: P::Proof,
    /// `rand_value_n(slot)` opened against the prior-epoch
    /// `rand_value_commitment`. Same caveat — zero on fresh slots.
    pub rand_value_pre_eval: E::ScalarField,
    pub rand_value_pre_proof: P::Proof,
    /// `rand_index_{n+1}(slot)` opened against the new-epoch
    /// `rand_index_commitment`. Equals `r_index_n · H_F(label)` on a
    /// fresh slot.
    pub rand_index_post_eval: E::ScalarField,
    pub rand_index_post_proof: P::Proof,
    /// `rand_value_{n+1}(slot)` opened against the new-epoch
    /// `rand_value_commitment`. Equals `r_value_n · H_F(value)` on a
    /// fresh slot.
    pub rand_value_post_eval: E::ScalarField,
    pub rand_value_post_proof: P::Proof,
    /// `value_{n+1}(slot) = H_F(value)` opened against the new-epoch
    /// `value_commitment`.
    pub value_post_eval: E::ScalarField,
    pub value_post_proof: P::Proof,
}

/// All §6.4 history witnesses one shard produced during one publish.
/// Empty when the publish carried no new labels for this shard (only
/// value-updates to already-occupied slots, which the rand-poly chain
/// covers via standard consistency proofs).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HistoryOpenings<E: Pairing, P: AegonPcs<E>> {
    pub entries: Vec<HistoryOpeningEntry<E, P>>,
}

impl<E: Pairing, P: AegonPcs<E>> Default for HistoryOpenings<E, P> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}
