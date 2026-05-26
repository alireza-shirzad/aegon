//! Server-side Aegon dictionary.
//!
//! Maintains four polynomials per epoch:
//!
//! * `index_n`, `value_n` — populated by `publish` and queried by
//!   `lookup`.
//! * `rand_index_n`, `rand_value_n` — randomized chain polynomials
//!   (paper §5.2 / §6.1) updated as
//!   `rand_{n+1} = rand_n + r_n · (poly_{n+1} - poly_n)` with `r_n`
//!   sampled via Fiat-Shamir from the previous chain randomness and the
//!   new commitment. These power both the per-epoch invariance proof
//!   the auditor verifies and the per-user consistency proof a client
//!   verifies for their own slot.
//!
//! For consistency proofs to reach back to a past epoch `s0`, the
//! server retains `EpochSnapshot`s of the rand polynomials at each
//! published epoch. v1 retains them indefinitely; the paper's §6.4
//! caching strategy (cache opening proofs at the time each label
//! changes, drop poly state for older epochs) is the natural
//! optimization but does not change any external API.

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

use ark_ec::pairing::Pairing;
use ark_ff::{One, UniformRand, Zero};
use ark_poly::SparseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use rayon::prelude::*;
use akd_core::aegon_crypto::pcs::PCSGlobalParam;
use akd_core::aegon_crypto::poly::{DenseOrSparseMLE, DenseOrSparseMLERef};
use akd_core::aegon_crypto::transcript::IOPTranscript;

use super::config::{AegonConfig, VerifierContext};
use super::db::{key_shard_state, DbSource, RedisDb};
use super::error::AegonError;
use super::fs::derive_chain_scalar;
use super::hash::{bool_index_to_point, bool_index_to_usize, HashSuite, Sha256Hash};
use super::instrument::log_rss_ctx;
use super::sharded::ShardWrite;
use super::types::{
    AegonPcs, EpochCommitment, HistoryOpeningEntry, HistoryOpenings, Label, LookupProof, RandPair,
    ValueChangeEntry,
    Value,
};

/// Snapshot of the polynomials needed for serving consistency proofs at
/// epoch `n`. The two data commitments are kept so we can return them in
/// `EpochCommitment`s; the rand polynomials and their PCS state are kept
/// so the server can produce opening proofs at arbitrary points later.
///
/// The four rand polynomial / state fields are `Option` so a shard that
/// never services `consistency_proof(label, s0)` (e.g. benchmark
/// deployments) can drop them at publish time via `Aegon::
/// set_retain_epoch_polys(false)`. The commitments are always kept —
/// `epoch_commitment(epoch)` (used by `verify_sharded_invariance`) only
/// reads those four fields.
#[derive(Clone)]
struct EpochSnapshot<E: Pairing, P: AegonPcs<E>> {
    index_commitment: P::Commitment,
    value_commitment: P::Commitment,

    rand_index_poly: Option<SparseMultilinearExtension<E::ScalarField>>,
    rand_index_commitment: P::Commitment,
    rand_index_state: Option<P::State>,

    rand_value_poly: Option<SparseMultilinearExtension<E::ScalarField>>,
    rand_value_commitment: P::Commitment,
    rand_value_state: Option<P::State>,

    /// Per-epoch hiding scalars for the value-side polynomials,
    /// snapshotted at the END of this publish. Always retained even
    /// when `rand_*_state` are dropped (the "no-retain-epoch-polys"
    /// bench mode) because masking a stored history opening at
    /// lookup time needs `tau_f` for the polynomial at the epoch the
    /// opening was produced. Only 32 B/scalar (64 B/epoch) so the
    /// growth is trivial — ~256 KB at 4K epochs even at planetary
    /// scale.
    ///
    /// `None` when the SRS is non-hiding (`is_zk = false`); the
    /// masking-server protocol short-circuits in that case so we
    /// never read these.
    value_tau: Option<P::HidingScalar>,
    rand_value_tau: Option<P::HidingScalar>,
}

pub struct Aegon<E, P, H = Sha256Hash>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    log_capacity: usize,
    /// Block layout used by the underlying PCS to map Boolean points to
    /// BTreeMap keys. Sums to `log_capacity`. For PCSs that follow
    /// ark_poly's default ("variable i in bit i"), this is `vec![log_capacity]`.
    /// For KZH-k with `k` blocks and even split, this is
    /// `vec![log_capacity / k; k]`.
    dims: Vec<usize>,

    prover_param: std::sync::Arc<P::ProverParam>,
    verifier_param: P::VerifierParam,

    /// Source of pre-built [`P::MaskingPackage`]s. In hiding mode
    /// (private = true) this is always populated: cluster shards wire
    /// up a remote [`super::masking::MaskingClient`] via
    /// [`Self::set_masking_source`], and in-process / test instances
    /// get a default in-process [`super::masking::MaskingPool`] built
    /// at construction time. In non-hiding mode the field stays
    /// `None` and value-side openings degrade to plain non-ZK proofs.
    masking_client: Option<std::sync::Arc<dyn super::masking::MaskingSource<E, P>>>,

    epoch: u64,

    // Live polynomials.
    index_poly: SparseMultilinearExtension<E::ScalarField>,
    value_poly: SparseMultilinearExtension<E::ScalarField>,
    rand_index_poly: SparseMultilinearExtension<E::ScalarField>,
    rand_value_poly: SparseMultilinearExtension<E::ScalarField>,

    // Live PCS state.
    index_commitment: P::Commitment,
    index_state: P::State,
    value_commitment: P::Commitment,
    value_state: P::State,
    rand_index_commitment: P::Commitment,
    rand_index_state: P::State,
    rand_value_commitment: P::Commitment,
    rand_value_state: P::State,

    // Fiat-Shamir state for the chain randomness. Initialised to zero;
    // each `publish` derives the next scalars from these and the new
    // commitments, then overwrites them.
    r_index: E::ScalarField,
    r_value: E::ScalarField,

    // Server bookkeeping for fast lookups.
    label_table: HashMap<Label, (Vec<bool>, u64)>,

    // Past epochs we can serve consistency proofs against. Keyed by
    // epoch number; epoch 0 is the empty initial state.
    epoch_history: BTreeMap<u64, EpochSnapshot<E, P>>,

    // When false, each `publish` records only the four commitments per
    // epoch (not the rand polynomials or their PCS state), so
    // `open_rand_*_at_slot_in_epoch` will fail for non-current epochs
    // but `epoch_commitment` / `verify_sharded_invariance` keep working.
    // Default `true`. Set `false` on shards that don't service
    // `consistency_proof` (e.g. bench cluster) to free ~50 MB/epoch.
    retain_epoch_polys: bool,

    // Set by publish_phase_1, consumed by publish_phase_2. None when
    // no publish is in flight. The two-phase split exists so that a
    // sharded coordinator can gather all 32 shards' new data commits
    // before deriving the shared Fiat-Shamir scalar; the single-shard
    // publish() wrapper sets and consumes this in one call.
    pending: Option<PendingPublish<E, P>>,

    _phantom: PhantomData<H>,
}

/// Self-contained snapshot of an [`Aegon`]'s live state — everything
/// needed to reconstruct the shard after a restart, given the same
/// `(prover_param, verifier_param)` (which come from the SRS file).
/// Used by the gRPC `ShardServer` to persist its state into the DB
/// at the end of every publish, and on startup to restore.
///
/// What's intentionally **not** in the checkpoint:
/// - `prover_param` / `verifier_param` — deterministic from the SRS;
///   the shard server loads them from disk before applying any
///   checkpoint.
/// - `epoch_history` — needed only to serve consistency proofs for
///   *past* epochs. Dropping it means a shard restart can serve
///   lookups + audits at the current epoch but not pre-restart
///   consistency proofs. Acceptable for benchmark deployments.
/// - `pending` — never non-`None` at a checkpoint boundary; the
///   shard server takes checkpoints after `publish_phase_2` finishes,
///   which clears `pending`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct AegonCheckpoint<E: Pairing, P: AegonPcs<E>> {
    pub epoch: u64,
    pub index_poly_evals: Vec<(u64, E::ScalarField)>,
    pub value_poly_evals: Vec<(u64, E::ScalarField)>,
    pub rand_index_poly_evals: Vec<(u64, E::ScalarField)>,
    pub rand_value_poly_evals: Vec<(u64, E::ScalarField)>,
    pub index_commitment: P::Commitment,
    pub index_state: P::State,
    pub value_commitment: P::Commitment,
    pub value_state: P::State,
    pub rand_index_commitment: P::Commitment,
    pub rand_index_state: P::State,
    pub rand_value_commitment: P::Commitment,
    pub rand_value_state: P::State,
    pub r_index: E::ScalarField,
    pub r_value: E::ScalarField,
    pub label_table_entries: Vec<(Vec<u8>, Vec<bool>, u64)>,
}

/// State carried by an [`Aegon`] across the two halves of a sharded
/// publish. The first half (`publish_phase_1`) commits to the new data
/// polynomials and stashes everything needed to (a) update the rand
/// polynomials with the externally-derived chain scalars and (b) build
/// the invariance proof. The second half (`publish_phase_2`) consumes
/// the stash.
struct PendingPublish<E: Pairing, P: AegonPcs<E>> {
    /// Delta polynomial on the index side — sparse with support exactly
    /// equal to the new-placement slots this batch touched. Stashed for
    /// phase_2's `update_rand_with_delta` which adds `r_index · delta`
    /// into `self.rand_index_poly` in place.
    delta_index_poly: SparseMultilinearExtension<E::ScalarField>,
    /// Same for value side. Support equals every slot whose value
    /// actually moved (new placements + value-only updates).
    delta_value_poly: SparseMultilinearExtension<E::ScalarField>,
    /// Commitment of `delta_index_poly` from phase 1. Reused in phase 2
    /// to derive the new rand_index commitment via the homomorphism
    /// `new_rand_index_com = rand_index_com + r_index · delta_index_com`,
    /// so phase 2 never has to run another MSM over the rand
    /// polynomial.
    delta_index_com: P::Commitment,
    delta_value_com: P::Commitment,
    /// Prover `State` for the same delta polynomials. The KZH-k aux
    /// table over a batch-sized support is built in phase 1 (cost
    /// `O(k · batch)`); phase 2 then uses the State homomorphism
    /// `new_rand_state = rand_state + r_X · delta_state` via in-place
    /// `P::fma_state` to avoid recomputing aux from scratch and to
    /// avoid cloning the (full-cap) prev state.
    delta_index_state: P::State,
    delta_value_state: P::State,
    /// Slot bits for every brand-new placement in this batch (entries
    /// whose `h_label` is `Some`). Populated by `publish_phase_1`,
    /// consumed by `publish_phase_2` to drive the §6.4 history-opening
    /// computation. Empty when the batch was value-updates only.
    new_label_slots: Vec<Vec<bool>>,
    /// Slot bits for every **value change** in this batch — both new
    /// placements (which always carry a non-zero value-side delta) and
    /// value-only updates on already-occupied slots. Populated by
    /// `publish_phase_1`, consumed by `publish_phase_2` to drive the
    /// per-slot value-history opening computation. A slot only ever
    /// appears once per publish (one publish carries one new value per
    /// slot). Order matches the input batch order, which means each
    /// slot's entry can be paired by `slot_bits` with the matching
    /// shard-write to recover the post-update value bytes.
    value_change_slots: Vec<Vec<bool>>,
}

impl<E, P, H> Aegon<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E>,
    P::ProverParam: PCSGlobalParam,
    P::VerifierParam: PCSGlobalParam,
    // Incremental publish needs the commitment-side homomorphism:
    // `prev + delta` for phase 1 data polys and `prev + r · delta` for
    // phase 2 rand polys. KZH-k's KZHKCommitment satisfies all three
    // (impls live in akd_core's structs.rs).
    P::Commitment: Clone
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    H: HashSuite<E::ScalarField>,
{
    /// Generate an SRS, trim it, and initialise an empty Aegon. The
    /// PCS-specific configuration (e.g. `k` and `zk` for KZH-k) is
    /// read off the `AegonConfig`.
    pub fn setup<R: Rng>(rng: &mut R, config: &AegonConfig<E, P>) -> Result<Self, AegonError>
    where
        E: Send + Sync + 'static,
        P: Send + Sync + 'static,
        P::ProverParam: Send + Sync + 'static,
        P::MaskingPackage: Send + 'static,
        E::ScalarField: Send + Sync + 'static,
    {
        // For multilinear PCSs, `supported_size` is the number of
        // variables (paper / trait docs), not the hypercube size.
        let srs = P::gen_srs_for_testing(config.pcs_config.clone(), rng, config.log_capacity)?;
        let (prover_param, verifier_param) = P::trim(&srs, None, Some(config.log_capacity))?;
        Self::init(prover_param, verifier_param, config)
    }

    /// Initialise from already-trimmed PCS parameters (e.g. when the
    /// SRS came from a multi-party ceremony rather than `setup`).
    ///
    /// Cross-checks `config.private` against the PCS's actual zk-ness
    /// (`PCSGlobalParam::is_zk()`) and refuses to start if they
    /// disagree. This catches a common mis-configuration: building an
    /// AegonConfig with `private = true` but a non-zk SRS, which
    /// would silently leak data the user expected to stay hidden.
    pub fn init(
        prover_param: P::ProverParam,
        verifier_param: P::VerifierParam,
        config: &AegonConfig<E, P>,
    ) -> Result<Self, AegonError>
    where
        E: Send + Sync + 'static,
        P: Send + Sync + 'static,
        P::ProverParam: Send + Sync + 'static,
        P::MaskingPackage: Send + 'static,
        E::ScalarField: Send + Sync + 'static,
    {
        Self::init_with_arc(std::sync::Arc::new(prover_param), verifier_param, config)
    }

    /// Same as [`Self::init`] but takes an already-wrapped
    /// `Arc<P::ProverParam>` so the caller can share the param with
    /// other consumers (e.g. an external [`super::masking::MaskingPool`]
    /// or a sibling Aegon instance) without paying a deep clone.
    ///
    /// In hiding mode (`config.private = true`) this constructor
    /// builds a default in-process [`super::masking::MaskingPool`]
    /// (2 producer threads, queue=16) so that every value-side opening
    /// can fetch a pre-built masking package — same architecture as
    /// the cluster path. Cluster shards override this with a remote
    /// [`super::masking::MaskingClient`] via [`Self::set_masking_source`].
    pub fn init_with_arc(
        prover_param: std::sync::Arc<P::ProverParam>,
        verifier_param: P::VerifierParam,
        config: &AegonConfig<E, P>,
    ) -> Result<Self, AegonError>
    where
        E: Send + Sync + 'static,
        P: Send + Sync + 'static,
        P::ProverParam: Send + Sync + 'static,
        P::MaskingPackage: Send + 'static,
        E::ScalarField: Send + Sync + 'static,
    {
        if prover_param.is_zk() != config.private {
            return Err(AegonError::Config(format!(
                "config.private = {} but PCS prover param is_zk() = {} — check that pcs_config matches the privacy flag",
                config.private,
                prover_param.is_zk(),
            )));
        }
        if verifier_param.is_zk() != config.private {
            return Err(AegonError::Config(format!(
                "config.private = {} but PCS verifier param is_zk() = {}",
                config.private,
                verifier_param.is_zk(),
            )));
        }

        let log_capacity = config.log_capacity;
        // Pull the PCS's own block layout out of the prover param.
        // KZH-k returns its SRS-time dims; PCSs that follow the
        // ark_poly default convention return `vec![log_capacity]`.
        let dims = P::block_dims(&prover_param, log_capacity);
        assert_eq!(
            dims.iter().sum::<usize>(),
            log_capacity,
            "PCS-reported block dims must sum to log_capacity"
        );
        let zero_poly = || SparseMultilinearExtension::from_evaluations(log_capacity, &[]);
        let index_poly = zero_poly();
        let value_poly = zero_poly();
        let rand_index_poly = zero_poly();
        let rand_value_poly = zero_poly();

        // Per-polynomial commit dispatch:
        //   * label side (index, rand_index) — always plain
        //     `C = <f, H_1>`, no hiding overhead, regardless of SRS.
        //   * value side (value, rand_value) — hiding
        //     `C = <f, H_1> + tau*h` whenever the SRS supports it,
        //     so the masking-server protocol has the polynomial's
        //     `tau_f` to plug into `rho_prime = alpha*tau_f + rho`.
        let (index_commitment, index_state) =
            commit_with_aux_non_zk::<E, P>(&prover_param, &index_poly)?;
        let (value_commitment, value_state) =
            commit_with_aux_value_side::<E, P>(&prover_param, &value_poly)?;
        let (rand_index_commitment, rand_index_state) =
            commit_with_aux_non_zk::<E, P>(&prover_param, &rand_index_poly)?;
        let (rand_value_commitment, rand_value_state) =
            commit_with_aux_value_side::<E, P>(&prover_param, &rand_value_poly)?;

        let mut epoch_history = BTreeMap::new();
        let (init_value_tau, init_rand_value_tau) =
            extract_value_taus::<E, P>(&prover_param, &value_state, &rand_value_state);
        epoch_history.insert(
            0,
            EpochSnapshot {
                index_commitment: index_commitment.clone(),
                value_commitment: value_commitment.clone(),
                rand_index_poly: Some(rand_index_poly.clone()),
                rand_index_commitment: rand_index_commitment.clone(),
                rand_index_state: Some(rand_index_state.clone()),
                rand_value_poly: Some(rand_value_poly.clone()),
                rand_value_commitment: rand_value_commitment.clone(),
                rand_value_state: Some(rand_value_state.clone()),
                value_tau: init_value_tau,
                rand_value_tau: init_rand_value_tau,
            },
        );

        // Default-on in-process masking pool for hiding mode. Cluster
        // shards swap this for a remote `MaskingClient` after init via
        // `set_masking_source`; tests / single-shard in-process benches
        // use this as-is so they exercise the same architecture as the
        // cluster path (value-side openings dequeue a pre-built package
        // rather than generating it inline on the critical path).
        let masking_client: Option<
            std::sync::Arc<dyn super::masking::MaskingSource<E, P>>,
        > = if config.private {
            let pool = super::masking::MaskingPool::<E, P>::new(
                std::sync::Arc::clone(&prover_param),
                log_capacity,
                /* queue_size */ 16,
                /* producer_count */ 2,
            );
            Some(std::sync::Arc::new(pool))
        } else {
            None
        };

        Ok(Self {
            log_capacity,
            dims,
            prover_param,
            verifier_param,
            masking_client,
            epoch: 0,
            index_poly,
            value_poly,
            rand_index_poly,
            rand_value_poly,
            index_commitment,
            index_state,
            value_commitment,
            value_state,
            rand_index_commitment,
            rand_index_state,
            rand_value_commitment,
            rand_value_state,
            r_index: E::ScalarField::zero(),
            r_value: E::ScalarField::zero(),
            label_table: HashMap::new(),
            epoch_history,
            retain_epoch_polys: true,
            pending: None,
            _phantom: PhantomData,
        })
    }

    /// Replace the masking source used for value-side openings.
    ///
    /// In hiding mode Aegon constructs a default in-process
    /// [`super::masking::MaskingPool`] automatically, so callers only
    /// need this when they want a different source — typically a
    /// cluster shard swapping the local pool for a remote
    /// [`super::masking::MaskingClient`]. Call before binding the gRPC
    /// socket; the field is consulted on every value-side open.
    pub fn set_masking_source(
        &mut self,
        source: std::sync::Arc<dyn super::masking::MaskingSource<E, P>>,
    ) {
        self.masking_client = Some(source);
    }

    /// Toggle whether each `publish` retains the rand polynomials + PCS
    /// state in `epoch_history`. Default `true`. When `false`, only the
    /// four commitments are kept per epoch — `epoch_commitment(epoch)`
    /// (used by `verify_sharded_invariance`) keeps working but
    /// `open_rand_*_at_slot_in_epoch` returns `InvalidEpoch` for any
    /// non-current epoch. Call once at startup; flipping mid-run leaves
    /// already-recorded snapshots in whatever shape they had.
    pub fn set_retain_epoch_polys(&mut self, retain: bool) {
        self.retain_epoch_polys = retain;
    }

    pub fn verifier_param(&self) -> P::VerifierParam {
        self.verifier_param.clone()
    }

    /// Snapshot every field in `[AegonCheckpoint]` for durable storage.
    /// Safe to call between epochs (i.e. with `self.pending == None`);
    /// at checkpoint time we never serialize an in-flight publish.
    pub fn capture_checkpoint(&self) -> AegonCheckpoint<E, P>
    where
        P::Commitment: Clone,
        P::State: Clone,
    {
        let extract = |poly: &SparseMultilinearExtension<E::ScalarField>| -> Vec<(u64, E::ScalarField)> {
            poly.evaluations
                .iter()
                .map(|(idx, v)| (*idx as u64, *v))
                .collect()
        };
        AegonCheckpoint {
            epoch: self.epoch,
            index_poly_evals: extract(&self.index_poly),
            value_poly_evals: extract(&self.value_poly),
            rand_index_poly_evals: extract(&self.rand_index_poly),
            rand_value_poly_evals: extract(&self.rand_value_poly),
            index_commitment: self.index_commitment.clone(),
            index_state: self.index_state.clone(),
            value_commitment: self.value_commitment.clone(),
            value_state: self.value_state.clone(),
            rand_index_commitment: self.rand_index_commitment.clone(),
            rand_index_state: self.rand_index_state.clone(),
            rand_value_commitment: self.rand_value_commitment.clone(),
            rand_value_state: self.rand_value_state.clone(),
            r_index: self.r_index,
            r_value: self.r_value,
            label_table_entries: self
                .label_table
                .iter()
                .map(|(k, (bits, ctr))| (k.clone(), bits.clone(), *ctr))
                .collect(),
        }
    }

    /// Rebuild an `Aegon` from a previously-captured checkpoint plus
    /// the deterministic `(prover_param, verifier_param)` from the SRS.
    /// `epoch_history` is left containing only the current-epoch
    /// snapshot — pre-restart consistency proofs are not recoverable
    /// from this minimal blob.
    pub fn restore_from_checkpoint(
        prover_param: P::ProverParam,
        verifier_param: P::VerifierParam,
        config: &AegonConfig<E, P>,
        ckpt: AegonCheckpoint<E, P>,
    ) -> Result<Self, AegonError>
    where
        P::Commitment: Clone,
        P::State: Clone,
        E: Send + Sync + 'static,
        P: Send + Sync + 'static,
        P::ProverParam: Send + Sync + 'static,
        P::MaskingPackage: Send + 'static,
        E::ScalarField: Send + Sync + 'static,
    {
        let prover_param = std::sync::Arc::new(prover_param);
        if prover_param.is_zk() != config.private {
            return Err(AegonError::Config(format!(
                "config.private = {} but PCS prover param is_zk() = {}",
                config.private,
                prover_param.is_zk(),
            )));
        }
        let log_capacity = config.log_capacity;
        let dims = P::block_dims(&prover_param, log_capacity);
        let to_sparse =
            |evals: Vec<(u64, E::ScalarField)>| -> SparseMultilinearExtension<E::ScalarField> {
                let pairs: Vec<(usize, E::ScalarField)> =
                    evals.into_iter().map(|(idx, v)| (idx as usize, v)).collect();
                SparseMultilinearExtension::from_evaluations(log_capacity, &pairs)
            };
        let index_poly = to_sparse(ckpt.index_poly_evals);
        let value_poly = to_sparse(ckpt.value_poly_evals);
        let rand_index_poly = to_sparse(ckpt.rand_index_poly_evals);
        let rand_value_poly = to_sparse(ckpt.rand_value_poly_evals);

        // Single epoch-history entry for the current epoch. Past
        // epochs' snapshots are unrecoverable from the checkpoint
        // alone; the shard will return InvalidEpoch for those.
        let mut epoch_history = BTreeMap::new();
        let (rest_value_tau, rest_rand_value_tau) = extract_value_taus::<E, P>(
            &prover_param,
            &ckpt.value_state,
            &ckpt.rand_value_state,
        );
        epoch_history.insert(
            ckpt.epoch,
            EpochSnapshot {
                index_commitment: ckpt.index_commitment.clone(),
                value_commitment: ckpt.value_commitment.clone(),
                rand_index_poly: Some(rand_index_poly.clone()),
                rand_index_commitment: ckpt.rand_index_commitment.clone(),
                rand_index_state: Some(ckpt.rand_index_state.clone()),
                rand_value_poly: Some(rand_value_poly.clone()),
                rand_value_commitment: ckpt.rand_value_commitment.clone(),
                rand_value_state: Some(ckpt.rand_value_state.clone()),
                value_tau: rest_value_tau,
                rand_value_tau: rest_rand_value_tau,
            },
        );

        let mut label_table: HashMap<Label, (Vec<bool>, u64)> =
            HashMap::with_capacity(ckpt.label_table_entries.len());
        for (label, bits, ctr) in ckpt.label_table_entries {
            label_table.insert(label, (bits, ctr));
        }

        // Same default-pool wiring as `init_with_arc` — restored
        // Aegon instances exercise the masking-server architecture
        // identically to fresh ones.
        let masking_client: Option<
            std::sync::Arc<dyn super::masking::MaskingSource<E, P>>,
        > = if config.private {
            let pool = super::masking::MaskingPool::<E, P>::new(
                std::sync::Arc::clone(&prover_param),
                log_capacity,
                /* queue_size */ 16,
                /* producer_count */ 2,
            );
            Some(std::sync::Arc::new(pool))
        } else {
            None
        };

        Ok(Self {
            log_capacity,
            dims,
            prover_param,
            verifier_param,
            masking_client,
            epoch: ckpt.epoch,
            index_poly,
            value_poly,
            rand_index_poly,
            rand_value_poly,
            index_commitment: ckpt.index_commitment,
            index_state: ckpt.index_state,
            value_commitment: ckpt.value_commitment,
            value_state: ckpt.value_state,
            rand_index_commitment: ckpt.rand_index_commitment,
            rand_index_state: ckpt.rand_index_state,
            rand_value_commitment: ckpt.rand_value_commitment,
            rand_value_state: ckpt.rand_value_state,
            r_index: ckpt.r_index,
            r_value: ckpt.r_value,
            label_table,
            epoch_history,
            retain_epoch_polys: true,
            pending: None,
            _phantom: PhantomData,
        })
    }

    /// Wipe this shard back to a fresh epoch-0 state, keeping the
    /// SRS-derived `(prover_param, verifier_param)` and any wired
    /// masking client. Equivalent to dropping `self` and re-calling
    /// `Aegon::init` with the same params, but doesn't require
    /// cloning the (expensive) PCS parameters.
    ///
    /// Used by the cluster benches: between fill_percent stages we
    /// reset every shard via gRPC, then prefill to the new target —
    /// avoids the kill-and-restart-with-different-CLI-flag dance.
    pub fn reset_state(&mut self) -> Result<(), AegonError> {
        let zero_poly = || SparseMultilinearExtension::from_evaluations(self.log_capacity, &[]);
        self.epoch = 0;
        self.index_poly = zero_poly();
        self.value_poly = zero_poly();
        self.rand_index_poly = zero_poly();
        self.rand_value_poly = zero_poly();
        let (index_commitment, index_state) =
            commit_with_aux_non_zk::<E, P>(self.prover_param.as_ref(), &self.index_poly)?;
        let (value_commitment, value_state) =
            commit_with_aux_value_side::<E, P>(self.prover_param.as_ref(), &self.value_poly)?;
        let (rand_index_commitment, rand_index_state) =
            commit_with_aux_non_zk::<E, P>(self.prover_param.as_ref(), &self.rand_index_poly)?;
        let (rand_value_commitment, rand_value_state) =
            commit_with_aux_value_side::<E, P>(self.prover_param.as_ref(), &self.rand_value_poly)?;
        self.index_commitment = index_commitment.clone();
        self.index_state = index_state;
        self.value_commitment = value_commitment.clone();
        self.value_state = value_state;
        self.rand_index_commitment = rand_index_commitment.clone();
        self.rand_index_state = rand_index_state.clone();
        self.rand_value_commitment = rand_value_commitment.clone();
        self.rand_value_state = rand_value_state.clone();
        self.r_index = E::ScalarField::zero();
        self.r_value = E::ScalarField::zero();
        self.label_table.clear();
        self.epoch_history.clear();
        let (reset_value_tau, reset_rand_value_tau) = self.snapshot_value_taus();
        self.epoch_history.insert(
            0,
            EpochSnapshot {
                index_commitment,
                value_commitment,
                rand_index_poly: Some(self.rand_index_poly.clone()),
                rand_index_commitment,
                rand_index_state: Some(rand_index_state),
                rand_value_poly: Some(self.rand_value_poly.clone()),
                rand_value_commitment,
                rand_value_state: Some(rand_value_state),
                value_tau: reset_value_tau,
                rand_value_tau: reset_rand_value_tau,
            },
        );
        self.pending = None;
        Ok(())
    }

    /// **Benchmark-only bulk-load**: populate `count` random
    /// `(slot, h_label, h_value)` entries directly into `index_poly` /
    /// `value_poly`, then recommit and rebuild the prover state. Used
    /// to bring a shard's polynomials to a "looks like the dictionary
    /// already has N users" state without going through the full
    /// publish protocol (which would serialize on the coordinator's
    /// open-addressing trail). Leaves `epoch = 0`, `r_index = r_value
    /// = 0`, and the rand polynomials empty — i.e. there's no FS-chain
    /// history covering the prefill, so an auditor walking the chain
    /// would only see transitions from this state forward, not into
    /// it. **Do not use in production.**
    ///
    /// Slot collisions are handled by simple overwrite — at sub-50%
    /// load factor (`count < 2^(log_capacity - 1)`) this is rare and
    /// doesn't materially change the resulting polynomial support
    /// size; at higher load factors `count` overstates the actual
    /// populated cell count.
    pub fn prefill_random<R: Rng>(
        &mut self,
        rng: &mut R,
        count: usize,
        db_source: &DbSource,
        shard_id: u32,
    ) -> Result<(), AegonError> {
        if self.pending.is_some() {
            return Err(AegonError::Config(
                "prefill_random called with a pending publish".into(),
            ));
        }
        if self.epoch != 0 {
            return Err(AegonError::Config(
                "prefill_random can only be called at epoch 0 (before any publish)".into(),
            ));
        }
        let capacity = 1usize << self.log_capacity;
        // Track unique slots we actually filled — random draws collide
        // at large `count`, so this can be smaller than `count`.
        let mut filled_slots: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(count);
        for _ in 0..count {
            // `rng.next_u64() as usize % capacity` is biased for
            // non-power-of-two `capacity`, but `capacity` here is
            // exactly `2^log_capacity` so the modulo is a clean mask.
            let slot = (rng.next_u64() as usize) & (capacity - 1);
            let h_label = E::ScalarField::rand(rng);
            let h_value = E::ScalarField::rand(rng);
            self.index_poly.evaluations.insert(slot, h_label);
            self.value_poly.evaluations.insert(slot, h_value);
            filled_slots.insert(slot);
        }
        // Recommit the populated data polynomials. Rand polys stay at
        // zero — there's been no publish, so the chain randomness is
        // still zero. Per-polynomial commit dispatch matches
        // `Aegon::init`: label side plain, value side hiding (when
        // SRS supports it).
        let (com_i, state_i) =
            commit_with_aux_non_zk::<E, P>(self.prover_param.as_ref(), &self.index_poly)?;
        let (com_v, state_v) =
            commit_with_aux_value_side::<E, P>(self.prover_param.as_ref(), &self.value_poly)?;
        self.index_commitment = com_i;
        self.index_state = state_i;
        self.value_commitment = com_v;
        self.value_state = state_v;
        // Refresh the epoch-0 snapshot so consistency-proof queries
        // against `epoch = 0` see the prefilled state, not the empty
        // state that `setup` originally inserted.
        let (prefill_value_tau, prefill_rand_value_tau) = self.snapshot_value_taus();
        self.epoch_history.insert(
            0,
            EpochSnapshot {
                index_commitment: self.index_commitment.clone(),
                value_commitment: self.value_commitment.clone(),
                rand_index_poly: Some(self.rand_index_poly.clone()),
                rand_index_commitment: self.rand_index_commitment.clone(),
                rand_index_state: Some(self.rand_index_state.clone()),
                rand_value_poly: Some(self.rand_value_poly.clone()),
                rand_value_commitment: self.rand_value_commitment.clone(),
                rand_value_state: Some(self.rand_value_state.clone()),
                value_tau: prefill_value_tau,
                rand_value_tau: prefill_rand_value_tau,
            },
        );
        // DELIBERATELY no DB writes here. The architecture has
        // shifted: the coordinator now owns all cross-process state
        // in its own local DB, and shards don't talk to a database
        // at all. The publish-time occupancy probe handles prefilled
        // slots via a gRPC fallback when the coord's local DB has no
        // entry (see `plan_phase_1_batches`). `db_source` is kept on
        // this signature purely so callers don't have to change
        // shape; it's unused by `prefill_with_random`.
        let _ = (db_source, filled_slots, shard_id);
        Ok(())
    }

    pub fn log_capacity(&self) -> usize {
        self.log_capacity
    }
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Returns a fresh `VerifierContext` that clients and auditors
    /// can use to call the `verify_*` functions. The bundle is
    /// `Clone`, so the server can hand out copies cheaply.
    pub fn verifier_context(&self) -> VerifierContext<E, P> {
        VerifierContext::new(self.log_capacity, self.verifier_param.clone())
    }

    pub fn current_commitment(&self) -> EpochCommitment<E, P> {
        EpochCommitment {
            epoch: self.epoch,
            index_commitment: self.index_commitment.clone(),
            value_commitment: self.value_commitment.clone(),
            rand_index_commitment: self.rand_index_commitment.clone(),
            rand_value_commitment: self.rand_value_commitment.clone(),
            _e: PhantomData,
        }
    }

    /// Whether the SRS this server was built against is hiding
    /// (`commit_zk` adds `tau*h` to value-side commitments). Cheap
    /// wrapper around the private `prover_param.is_zk()` so trait
    /// impls outside `Aegon` (in particular the in-process
    /// `ShardHandle for Aegon` impl) can branch on hiding-ness
    /// without breaking encapsulation.
    pub fn is_zk_srs(&self) -> bool {
        self.prover_param.is_zk()
    }

    /// Snapshot the current value-side hiding scalars (`tau_f` for
    /// `value_poly` and `rand_value_poly`). Returns `(None, None)`
    /// on a non-hiding SRS so callers always get a uniform tuple.
    ///
    /// These tiny scalars (32 B each) are retained per-epoch even
    /// when `retain_epoch_polys=false` drops the full state — the
    /// masking-server protocol at lookup time needs `tau_f` to
    /// compute `rho_prime = alpha*tau_f + rho`.
    fn snapshot_value_taus(&self) -> (Option<P::HidingScalar>, Option<P::HidingScalar>) {
        if self.prover_param.is_zk() {
            (
                Some(P::get_hiding_scalar(&self.value_state)),
                Some(P::get_hiding_scalar(&self.rand_value_state)),
            )
        } else {
            (None, None)
        }
    }

    /// Look up `tau_f` for `value_poly` at a past (or current) epoch.
    /// Returns `None` if the epoch is not retained OR the snapshot
    /// was recorded under a non-hiding SRS. Used by lookup-time
    /// masking of stored §6.4 openings.
    pub fn value_tau_at_epoch(&self, epoch: u64) -> Option<P::HidingScalar> {
        self.epoch_history.get(&epoch)?.value_tau.clone()
    }

    /// Look up `tau_f` for `rand_value_poly` at a past (or current)
    /// epoch. See [`Self::value_tau_at_epoch`] for semantics.
    pub fn rand_value_tau_at_epoch(&self, epoch: u64) -> Option<P::HidingScalar> {
        self.epoch_history.get(&epoch)?.rand_value_tau.clone()
    }

    /// Returns the epoch commitment for a past (or current) epoch, if the
    /// server has retained it. Useful for plumbing prior commitments into
    /// `verify_consistency` without forcing the caller to have stashed them
    /// out-of-band.
    pub fn epoch_commitment(&self, epoch: u64) -> Option<EpochCommitment<E, P>> {
        let snap = self.epoch_history.get(&epoch)?;
        Some(EpochCommitment {
            epoch,
            index_commitment: snap.index_commitment.clone(),
            value_commitment: snap.value_commitment.clone(),
            rand_index_commitment: snap.rand_index_commitment.clone(),
            rand_value_commitment: snap.rand_value_commitment.clone(),
            _e: PhantomData,
        })
    }

    /// Apply a batch of `(label, value)` updates and produce a new epoch.
    /// Returns the new `EpochCommitment` — auditors verify the transition
    /// straight off the commitment fields (commitment-homomorphism path),
    /// no per-epoch proof bytes flow.
    ///
    /// Convenience wrapper around [`Self::publish_phase_1`] +
    /// [`Self::publish_phase_2`] for the single-shard case: derives the
    /// Fiat-Shamir chain scalars from `(prev_r, new_data_commit)`
    /// internally. In a sharded deployment the coordinator instead
    /// calls phase 1 on every shard, derives `r` from all sub-commits,
    /// then broadcasts `r` to each shard's phase 2.
    pub fn publish(
        &mut self,
        updates: &[(Label, Value)],
    ) -> Result<EpochCommitment<E, P>, AegonError> {
        let prev_r_index = self.r_index;
        let prev_r_value = self.r_value;
        let (new_index_com, new_value_com) = self.publish_phase_1(updates)?;
        let new_r_index = derive_chain_scalar::<E::ScalarField, P::Commitment>(
            b"aegon.fs.r_index",
            prev_r_index,
            &new_index_com,
        );
        let new_r_value = derive_chain_scalar::<E::ScalarField, P::Commitment>(
            b"aegon.fs.r_value",
            prev_r_value,
            &new_value_com,
        );
        // Discard the §6.4 history openings: this convenience wrapper
        // is the non-sharded path, where there's no coordinator-side
        // DB to persist them to. Sharded callers go through
        // `ShardedAegon::publish`, which threads the openings into
        // `persist_publish_to_db`.
        let (commit, _history) = self.publish_phase_2(new_r_index, new_r_value)?;
        Ok(commit)
    }

    /// First half of a sharded publish: apply data updates, commit the
    /// new `index` and `value` polynomials, and stash the prev-state
    /// snapshot needed by phase 2. Returns the two new data commitments
    /// so the coordinator can build the Fiat-Shamir chain scalars.
    ///
    /// Self-driven open addressing: assigns a slot to each new label
    /// using `H_bits` over the local `log_capacity`. The coordinator-
    /// driven counterpart is [`Self::publish_phase_1_at_slots`].
    ///
    /// Calling phase 1 twice without an intervening phase 2 is an error
    /// (an outstanding pending publish would be overwritten).
    pub fn publish_phase_1(
        &mut self,
        updates: &[(Label, Value)],
    ) -> Result<(P::Commitment, P::Commitment), AegonError> {
        let mut seen: HashMap<&[u8], ()> = HashMap::with_capacity(updates.len());
        for (label, _) in updates {
            if seen.insert(label.as_slice(), ()).is_some() {
                return Err(AegonError::DuplicateLabel(label.clone()));
            }
        }

        // Decide slots for every label first (without mutating polynomials).
        // For collision-free assignment within a single batch, track which
        // slots have already been claimed by earlier entries.
        let mut claimed: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(updates.len());
        let mut batch: Vec<ShardWrite<E::ScalarField>> = Vec::with_capacity(updates.len());
        for (label, value) in updates {
            let (slot_bits, h_label_write) = match self.label_table.get(label) {
                // Existing label — slot is fixed, no need to re-write h_label.
                Some((slot, _ctr0)) => (slot.clone(), None),
                None => {
                    let (slot_bits, ctr0) = self.find_free_slot(label, &claimed)?;
                    self.label_table
                        .insert(label.clone(), (slot_bits.clone(), ctr0));
                    (slot_bits, Some(H::h_f(label)))
                },
            };
            let usize_idx = bool_index_to_usize(&slot_bits, &self.dims);
            claimed.insert(usize_idx);
            let h_value = H::h_f(value);
            batch.push(ShardWrite {
                slot_bits,
                h_label: h_label_write,
                h_value,
            });
        }

        self.publish_phase_1_at_slots(&batch)
    }

    /// Coordinator-driven phase 1: caller supplies pre-decided
    /// `(slot_bits, h_label, h_value)` triples and the shard skips its
    /// own open addressing entirely. `h_label` should be `H::h_f(label)`
    /// the first time a slot is claimed (writes the label hash into
    /// `index_poly`), or `F::zero()` for an existing label being value-
    /// updated (the slot already holds the right `H_f(label)`). The
    /// `ShardedAegon` coordinator drives open addressing across all
    /// shards and feeds the decisions in via this method.
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(
            level = "debug",
            skip_all,
            name = "Aegon::PublishPhase1",
            fields(batch_size = batch.len())
        )
    )]
    pub fn publish_phase_1_at_slots(
        &mut self,
        batch: &[ShardWrite<E::ScalarField>],
    ) -> Result<(P::Commitment, P::Commitment), AegonError> {
        if self.pending.is_some() {
            return Err(AegonError::Config(
                "publish_phase_1 called while an earlier publish is still pending; call publish_phase_2 first".into(),
            ));
        }
        let _phase1_total_t = std::time::Instant::now();
        let _rss_epoch = self.epoch;
        let _rss_batch = batch.len();
        let _rss_nnz_idx = self.index_poly.evaluations.len();
        log_rss_ctx(
            "phase1.enter",
            &format!(
                "epoch={} batch={} nnz_idx={}",
                _rss_epoch, _rss_batch, _rss_nnz_idx
            ),
        );

        // No prev_* snapshot needed. The data polynomials (index, value)
        // get mutated in place by the apply-writes loop below, and the
        // data commitments / states are combined in place at the end of
        // phase 1. The rand polynomials and their commitments / states
        // are NOT touched in phase 1; they remain at their prior-epoch
        // values until phase 2's pre-update opening pass captures any
        // openings against them, then phase 2 mutates them in place too.
        // This eliminates the 4 × (poly + state + commitment) clones the
        // previous version paid per publish, which at log_capacity = 27
        // dominated shard RSS (~10 GB peak per call).

        // Apply writes AND build delta polys in one pass. The deltas
        // are sparse polynomials with support exactly equal to the
        // touched slots — used below to commit + aux only the changes
        // (cost `O(k · batch)`) instead of recomputing the commitment
        // and aux over the full dictionary support (cost
        // `O(k · current_nnz)`). Also collect new-placement slot bits
        // for §6.4 history witnesses.
        #[cfg(feature = "tracing_instrument")]
        let _apply_writes_span =
            tracing::debug_span!("Aegon::Phase1::ApplyWritesBuildDeltas").entered();
        let _apply_writes_t = std::time::Instant::now();
        let mut delta_index_poly: SparseMultilinearExtension<E::ScalarField> =
            SparseMultilinearExtension::from_evaluations(self.index_poly.num_vars, &[]);
        let mut delta_value_poly: SparseMultilinearExtension<E::ScalarField> =
            SparseMultilinearExtension::from_evaluations(self.value_poly.num_vars, &[]);
        let mut new_label_slots: Vec<Vec<bool>> = Vec::new();
        // Every slot whose value actually changes this publish (delta
        // != 0). Includes brand-new placements *and* updates on
        // already-occupied slots. Drives the per-slot value-history
        // openings produced in phase 2.
        let mut value_change_slots: Vec<Vec<bool>> = Vec::new();
        for ShardWrite {
            slot_bits,
            h_label,
            h_value,
        } in batch
        {
            let usize_idx = bool_index_to_usize(slot_bits, &self.dims);
            // Value side: every batch entry updates value_poly. Compute
            // the per-slot delta from the pre-update polynomial state,
            // then commit the new value.
            let old_value = self
                .value_poly
                .evaluations
                .get(&usize_idx)
                .copied()
                .unwrap_or_else(<E::ScalarField as Zero>::zero);
            let value_delta = *h_value - old_value;
            if !value_delta.is_zero() {
                delta_value_poly.evaluations.insert(usize_idx, value_delta);
                // Only record a value-change history slot when the
                // value actually moved — re-publishing an identical
                // value is a no-op for the polynomial and shouldn't
                // pollute the user's value-history bundle.
                value_change_slots.push(slot_bits.clone());
            }
            self.set_value(usize_idx, *h_value);
            // Index side: only brand-new placements write h_label. The
            // old value at a brand-new slot is zero by construction
            // (open-addressing picks empty slots), so the delta is just
            // `h_label`.
            if let Some(h_label) = h_label {
                let old_index = self
                    .index_poly
                    .evaluations
                    .get(&usize_idx)
                    .copied()
                    .unwrap_or_else(<E::ScalarField as Zero>::zero);
                let index_delta = *h_label - old_index;
                if !index_delta.is_zero() {
                    delta_index_poly.evaluations.insert(usize_idx, index_delta);
                }
                self.index_poly.evaluations.insert(usize_idx, *h_label);
                new_label_slots.push(slot_bits.clone());
            }
        }
        #[cfg(feature = "tracing_instrument")]
        drop(_apply_writes_span);
        log_rss_ctx("phase1.post_delta_build", &format!("epoch={}", _rss_epoch));
        if super::instrument::publish_profile_enabled() {
            eprintln!(
                "[pub-profile] phase1.apply_writes_and_build_deltas: {:.3} ms (batch={} new_label={} value_change={})",
                _apply_writes_t.elapsed().as_secs_f64() * 1000.0,
                _rss_batch,
                new_label_slots.len(),
                value_change_slots.len(),
            );
        }

        // Commit + aux the delta polynomials (size `batch`). Cost is
        // `O(batch)` for `P::commit` and `O(k · batch)` for the
        // per-row aux fill inside `commit_with_aux`.
        // Same per-polynomial dispatch as init: delta_index is
        // label-side (plain), delta_value is value-side (hiding when
        // SRS supports it). Both feed the homomorphic combine
        // `new_com = prev_com + delta_com` and `new_state = prev_state +
        // delta_state` — for the hiding case, this carries
        // `tau_new = tau_prev + tau_delta` through `iadd_scaled`.
        // Index- and value-side delta commits are independent — they
        // touch disjoint SRS bases and produce independent (com, state)
        // pairs. Run them on two rayon threads via `join` so the bigger
        // of the two sets the wall time instead of the sum.
        let _delta_commit_t = std::time::Instant::now();
        // Borrow the inner `&P::ProverParam` once so the closures capture
        // the bare reference (Send iff `P::ProverParam: Sync`) rather
        // than the surrounding `&Arc<>` (which would need `P::ProverParam: Send`).
        let pp = self.prover_param.as_ref();
        let (idx_res, val_res) = rayon::join(
            || commit_with_aux_non_zk::<E, P>(pp, &delta_index_poly),
            || commit_with_aux_value_side::<E, P>(pp, &delta_value_poly),
        );
        let (delta_index_com, delta_index_state) = idx_res?;
        let (delta_value_com, delta_value_state) = val_res?;
        log_rss_ctx("phase1.post_delta_commits", &format!("epoch={}", _rss_epoch));
        if super::instrument::publish_profile_enabled() {
            eprintln!(
                "[pub-profile] phase1.delta_commits_parallel: {:.3} ms (delta_index_nnz={} delta_value_nnz={})",
                _delta_commit_t.elapsed().as_secs_f64() * 1000.0,
                delta_index_poly.evaluations.len(),
                delta_value_poly.evaluations.len(),
            );
        }

        // Homomorphism on commitments and on the prover state — done IN
        // PLACE on `self.*`:
        //   self.X_commitment += delta_X_com
        //   self.X_state      += delta_X_state    (sparse-walk FMA)
        // Both ops cost `O(|support(delta)|)` group ops, independent of
        // how big the prior epoch's support is. See
        // `KZHKState::iadd_scaled` for the per-row primitive.
        //
        // After this block `self.index_commitment`, `self.value_commitment`,
        // `self.index_state`, `self.value_state` are all the NEW
        // (post-publish) data-side values. `self.index_poly` and
        // `self.value_poly` were already updated in the apply-writes
        // loop. The rand_* fields remain at their prior epoch — phase 2
        // updates them.
        #[cfg(feature = "tracing_instrument")]
        let _combine_span = tracing::debug_span!("Aegon::Phase1::CombineHomomorphic").entered();
        let _combine_t = std::time::Instant::now();
        let new_index_com = self.index_commitment.clone() + delta_index_com.clone();
        let new_value_com = self.value_commitment.clone() + delta_value_com.clone();
        self.index_commitment = new_index_com.clone();
        self.value_commitment = new_value_com.clone();
        let _com_combine_ms = _combine_t.elapsed().as_secs_f64() * 1000.0;
        // Split-borrow `index_state` and `value_state` so the two
        // `fma_state` calls (each O(|support(delta)|·k) group ops) run
        // on disjoint state vectors in parallel via `rayon::join`.
        // `prover_param` and the delta states are `&` shared by both
        // closures, which is fine because they're read-only.
        let one = E::ScalarField::one();
        let _fma_t = std::time::Instant::now();
        {
            let pp = self.prover_param.as_ref();
            let is: &mut P::State = &mut self.index_state;
            let vs: &mut P::State = &mut self.value_state;
            let (idx_fma, val_fma) = rayon::join(
                || P::fma_state(pp, is, one, &delta_index_state),
                || P::fma_state(pp, vs, one, &delta_value_state),
            );
            idx_fma?;
            val_fma?;
        }
        #[cfg(feature = "tracing_instrument")]
        drop(_combine_span);
        log_rss_ctx("phase1.post_combine", &format!("epoch={}", _rss_epoch));
        if super::instrument::publish_profile_enabled() {
            eprintln!(
                "[pub-profile] phase1.commit_combine: {:.3} ms",
                _com_combine_ms,
            );
            eprintln!(
                "[pub-profile] phase1.fma_state_parallel: {:.3} ms",
                _fma_t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        self.pending = Some(PendingPublish {
            delta_index_poly,
            delta_value_poly,
            delta_index_com,
            delta_value_com,
            delta_index_state,
            delta_value_state,
            new_label_slots,
            value_change_slots,
        });
        log_rss_ctx("phase1.exit", &format!("epoch={}", _rss_epoch));
        if super::instrument::publish_profile_enabled() {
            eprintln!(
                "[pub-profile] PHASE1_TOTAL: {:.3} ms",
                _phase1_total_t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        Ok((new_index_com, new_value_com))
    }

    /// Second half of a sharded publish: consumes the pending state
    /// stashed by [`Self::publish_phase_1`], applies the externally-
    /// derived chain scalars to update the rand polynomials, commits
    /// them, and finalizes the new epoch. Returns the new
    /// `EpochCommitment` together with the §6.4 history witnesses for
    /// every brand-new placement this batch touched.
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(level = "debug", skip_all, name = "Aegon::PublishPhase2")
    )]
    pub fn publish_phase_2(
        &mut self,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
    ) -> Result<(EpochCommitment<E, P>, HistoryOpenings<E, P>), AegonError> {
        let _phase2_total_t = std::time::Instant::now();
        let _rss_epoch = self.epoch;
        log_rss_ctx("phase2.enter", &format!("epoch={}", _rss_epoch));
        let pending = self.pending.take().ok_or_else(|| {
            AegonError::Config(
                "publish_phase_2 called without a pending publish; call publish_phase_1 first"
                    .into(),
            )
        })?;
        let PendingPublish {
            delta_index_poly,
            delta_value_poly,
            delta_index_com,
            delta_value_com,
            delta_index_state,
            delta_value_state,
            new_label_slots,
            value_change_slots,
        } = pending;

        // Build the placement_idx map up-front — we need it both for
        // pre-opening value-only-update slots (below) and for assembling
        // value_change_entries (further down).
        let mut placement_idx: std::collections::HashMap<Vec<bool>, usize> =
            std::collections::HashMap::with_capacity(new_label_slots.len());
        for (i, slot) in new_label_slots.iter().enumerate() {
            placement_idx.insert(slot.clone(), i);
        }

        // §6.4 step 1 (PRE-update openings): open `rand_index` and
        // `rand_value` at every new-label slot **before** the
        // rand-polynomials are mutated. The openings bind against the
        // prior-epoch rand commitments — which at this point are still
        // `self.rand_*_commitment` (phase 1 doesn't touch rand_*; the
        // homomorphic combine below is the first mutation, and we run
        // it AFTER capturing these openings). Evaluations are zero by
        // construction (a brand-new slot has had no chain delta
        // applied to it through any prior epoch), but the PCS proof is
        // still required for the future history-check verifier.
        #[cfg(feature = "tracing_instrument")]
        let _pre_openings_span = tracing::debug_span!(
            "Aegon::Phase2::PreUpdateOpenings",
            new_slots = new_label_slots.len()
        )
        .entered();
        // Publish-time openings:
        // * `rand_index` is label-side, plain commit ⇒ plain opening.
        // * `rand_value` is value-side, hiding commit ⇒ goes through
        //   auto-dispatching `P::open`, which produces an inline ZK
        //   opening under a hiding SRS. The masking-server protocol
        //   is intentionally NOT on the publish hot path ("not in
        //   the publish"); inline sampling is fine here.
        //
        // All openings touch `&self.*` read-only — the homomorphic
        // combine below is the first mutation, and it runs after this
        // pass completes — so the per-slot work parallelizes cleanly.
        // KZH-k opening at log_cap=27, k=9 is `O(k · BTreeMap::get)`
        // per call (~50–100 µs); at 8 K new placements per shard this
        // pass is the wall of phase 2. Rayon `par_iter` over the slot
        // list spreads it across the shard's CPUs.
        let pp = self.prover_param.as_ref();
        let dims = &self.dims;
        let _pre_open_t = std::time::Instant::now();
        let pre_pairs: Vec<((E::ScalarField, P::Proof), (E::ScalarField, P::Proof))> =
            new_label_slots
                .par_iter()
                .map(|slot_bits| -> Result<_, AegonError> {
                    let ri = open_at_point_non_zk::<E, P>(
                        pp,
                        &self.rand_index_poly,
                        &self.rand_index_commitment,
                        &self.rand_index_state,
                        slot_bits,
                        dims,
                        b"aegon.rand_index.open",
                    )?;
                    let rv = open_at_point_non_zk::<E, P>(
                        pp,
                        &self.rand_value_poly,
                        &self.rand_value_commitment,
                        &self.rand_value_state,
                        slot_bits,
                        dims,
                        b"aegon.rand_value.open",
                    )?;
                    Ok((ri, rv))
                })
                .collect::<Result<Vec<_>, _>>()?;
        let (pre_rand_index, pre_rand_value): (
            Vec<(E::ScalarField, P::Proof)>,
            Vec<(E::ScalarField, P::Proof)>,
        ) = pre_pairs.into_iter().unzip();

        // Pre-update `rand_value` openings for value-only-update slots
        // (subset of `value_change_slots` that aren't new placements).
        // Captured here so we can pair them with their post-update
        // counterparts after the in-place rand mutation below. Also
        // parallelized — each call is an independent read of
        // `self.rand_value_*`.
        let voup_pre_rand_value: std::collections::HashMap<
            Vec<bool>,
            (E::ScalarField, P::Proof),
        > = value_change_slots
            .par_iter()
            .filter(|slot_bits| !placement_idx.contains_key(*slot_bits))
            .map(|slot_bits| -> Result<_, AegonError> {
                let p = open_at_point_non_zk::<E, P>(
                    pp,
                    &self.rand_value_poly,
                    &self.rand_value_commitment,
                    &self.rand_value_state,
                    slot_bits,
                    dims,
                    b"aegon.rand_value.open",
                )?;
                Ok((slot_bits.clone(), p))
            })
            .collect::<Result<std::collections::HashMap<_, _>, _>>()?;
        #[cfg(feature = "tracing_instrument")]
        drop(_pre_openings_span);
        if super::instrument::publish_profile_enabled() {
            eprintln!(
                "[pub-profile] phase2.pre_openings_parallel: {:.3} ms (placements={} voup={})",
                _pre_open_t.elapsed().as_secs_f64() * 1000.0,
                new_label_slots.len(),
                voup_pre_rand_value.len(),
            );
        }

        // Update rand polynomials in place: `rand_{n+1} = rand_n + r · ∆`,
        // using the explicit delta polynomials stashed by phase 1 (we
        // don't need prev/new pairs because the delta IS poly_{n+1} −
        // poly_n by construction in phase 1). Cost is
        // `O(|support(∆)|)`. The index and value updates touch disjoint
        // fields of `self` and can run on two rayon threads.
        let _rand_update_t = std::time::Instant::now();
        {
            let rip: &mut SparseMultilinearExtension<E::ScalarField> = &mut self.rand_index_poly;
            let rvp: &mut SparseMultilinearExtension<E::ScalarField> = &mut self.rand_value_poly;
            rayon::join(
                || update_rand_with_delta(rip, &delta_index_poly, new_r_index),
                || update_rand_with_delta(rvp, &delta_value_poly, new_r_value),
            );
        }
        if super::instrument::publish_profile_enabled() {
            eprintln!(
                "[pub-profile] phase2.update_rand_polys_parallel: {:.3} ms",
                _rand_update_t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        // Homomorphic combine on rand commitments / states — done IN
        // PLACE on `self.rand_*`. After this block, all four rand_*
        // fields on self hold the NEW (post-publish) values.
        //   self.rand_X_commitment = rand_X_commitment + r_X · delta_X_com
        //   self.rand_X_state     += r_X · delta_X_state    (sparse FMA)
        // The `+ r_X ·` part is one scalar-mul on the commitment group
        // element and `O(k · batch)` group ops on the state (only
        // touched cells get updated — see `KZHKState::iadd_scaled`).
        // The two FMAs touch disjoint state fields ⇒ split-borrow +
        // rayon::join.
        #[cfg(feature = "tracing_instrument")]
        let _combine_rand_span =
            tracing::debug_span!("Aegon::Phase2::CombineRandHomomorphic").entered();
        let _rand_combine_t = std::time::Instant::now();
        let new_rand_index_com = self.rand_index_commitment.clone()
            + delta_index_com.clone() * new_r_index;
        let new_rand_value_com = self.rand_value_commitment.clone()
            + delta_value_com.clone() * new_r_value;
        self.rand_index_commitment = new_rand_index_com.clone();
        self.rand_value_commitment = new_rand_value_com.clone();
        let _rand_com_combine_ms = _rand_combine_t.elapsed().as_secs_f64() * 1000.0;
        let _rand_fma_t = std::time::Instant::now();
        {
            let pp = self.prover_param.as_ref();
            let ris: &mut P::State = &mut self.rand_index_state;
            let rvs: &mut P::State = &mut self.rand_value_state;
            let (rfi, rfv) = rayon::join(
                || P::fma_state(pp, ris, new_r_index, &delta_index_state),
                || P::fma_state(pp, rvs, new_r_value, &delta_value_state),
            );
            rfi?;
            rfv?;
        }
        #[cfg(feature = "tracing_instrument")]
        drop(_combine_rand_span);
        log_rss_ctx("phase2.post_rand_combine", &format!("epoch={}", _rss_epoch));
        if super::instrument::publish_profile_enabled() {
            eprintln!(
                "[pub-profile] phase2.rand_commit_combine: {:.3} ms",
                _rand_com_combine_ms,
            );
            eprintln!(
                "[pub-profile] phase2.rand_fma_state_parallel: {:.3} ms",
                _rand_fma_t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        // §6.4 step 2 (POST-update openings): open `rand_index` and
        // `rand_value` (now at the new epoch) and `value` (also at the
        // new epoch — `self.value_poly` was committed at the end of
        // phase 1) at every new-label slot. Together with the
        // pre-openings above, this pins both endpoints of the update
        // equation
        // `rand_X_new(s) − rand_X_old(s) = r_X · (data_X_new(s) − 0)`
        // at every slot the verifier needs to check.
        #[cfg(feature = "tracing_instrument")]
        let _post_openings_span = tracing::debug_span!(
            "Aegon::Phase2::PostUpdateOpenings",
            new_slots = new_label_slots.len()
        )
        .entered();
        // All three openings per slot are pure reads on now-mutated
        // `self.*` (rand and value sides). Parallelize over slots —
        // this is symmetric to the pre-pass and roughly the same wall
        // weight. At 8K new placements per shard this trio dominates
        // phase 2 alongside the pre-pass.
        let pp = self.prover_param.as_ref();
        let dims = &self.dims;
        let _post_open_t = std::time::Instant::now();
        let post_triples: Vec<(
            (E::ScalarField, P::Proof),
            (E::ScalarField, P::Proof),
            (E::ScalarField, P::Proof),
        )> = new_label_slots
            .par_iter()
            .map(|slot_bits| -> Result<_, AegonError> {
                let ri = open_at_point_non_zk::<E, P>(
                    pp,
                    &self.rand_index_poly,
                    &self.rand_index_commitment,
                    &self.rand_index_state,
                    slot_bits,
                    dims,
                    b"aegon.rand_index.open",
                )?;
                let rv = open_at_point_non_zk::<E, P>(
                    pp,
                    &self.rand_value_poly,
                    &self.rand_value_commitment,
                    &self.rand_value_state,
                    slot_bits,
                    dims,
                    b"aegon.rand_value.open",
                )?;
                let v = open_at_point_non_zk::<E, P>(
                    pp,
                    &self.value_poly,
                    &self.value_commitment,
                    &self.value_state,
                    slot_bits,
                    dims,
                    b"aegon.value.open",
                )?;
                Ok((ri, rv, v))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut post_rand_index: Vec<(E::ScalarField, P::Proof)> =
            Vec::with_capacity(new_label_slots.len());
        let mut post_rand_value: Vec<(E::ScalarField, P::Proof)> =
            Vec::with_capacity(new_label_slots.len());
        let mut post_value: Vec<(E::ScalarField, P::Proof)> =
            Vec::with_capacity(new_label_slots.len());
        for (ri, rv, v) in post_triples {
            post_rand_index.push(ri);
            post_rand_value.push(rv);
            post_value.push(v);
        }
        #[cfg(feature = "tracing_instrument")]
        drop(_post_openings_span);
        if super::instrument::publish_profile_enabled() {
            eprintln!(
                "[pub-profile] phase2.post_openings_parallel: {:.3} ms (placements={})",
                _post_open_t.elapsed().as_secs_f64() * 1000.0,
                new_label_slots.len(),
            );
        }

        // Assemble the §6.4 history witness bundle. Per-slot evals +
        // proofs come from the two passes above, addressed by the
        // same `new_label_slots` ordering.
        let new_label_entries: Vec<HistoryOpeningEntry<E, P>> = new_label_slots
            .iter()
            .enumerate()
            .map(|(i, slot_bits)| HistoryOpeningEntry {
                slot_bits: slot_bits.clone(),
                rand_index_pre_eval: pre_rand_index[i].0,
                rand_index_pre_proof: pre_rand_index[i].1.clone(),
                rand_value_pre_eval: pre_rand_value[i].0,
                rand_value_pre_proof: pre_rand_value[i].1.clone(),
                rand_index_post_eval: post_rand_index[i].0,
                rand_index_post_proof: post_rand_index[i].1.clone(),
                rand_value_post_eval: post_rand_value[i].0,
                rand_value_post_proof: post_rand_value[i].1.clone(),
                value_post_eval: post_value[i].0,
                value_post_proof: post_value[i].1.clone(),
            })
            .collect();

        // Per-slot value-change openings (3 each). One entry per slot
        // whose value moved this publish — covers brand-new placements
        // (we reuse the post-update openings already computed above
        // when the placement coincides with the slot) AND value-only
        // updates on already-occupied slots (which require fresh
        // openings since the §6.4 placement loop ignored them).
        //
        // Strategy: split `value_change_slots` into "is a new
        // placement" (lookup by slot_bits in `new_label_slots`) vs
        // "is value-only update". For placements, copy the openings
        // out of the §6.4 bundle for free. For value-only updates, do
        // three fresh `open_at_point` calls (rand_value pre/post +
        // value post). rand_index is NOT touched here — it doesn't
        // move on value-only updates.
        #[cfg(feature = "tracing_instrument")]
        let _value_change_span = tracing::debug_span!(
            "Aegon::Phase2::ValueChangeOpenings",
            slots = value_change_slots.len()
        )
        .entered();
        // Parallelize over value_change_slots. Placement entries do no
        // crypto (just copy from the §6.4 bundle, free). VOUP entries
        // do 2 openings each — same per-call cost as the §6.4 passes,
        // so worth distributing across cores. `voup_pre_rand_value` is
        // a `HashMap` populated above; we read (not remove) here to
        // keep the closure `Fn` instead of `FnMut`.
        let _vc_open_t = std::time::Instant::now();
        let value_change_entries: Vec<ValueChangeEntry<E, P>> = value_change_slots
            .par_iter()
            .map(|slot_bits| -> Result<ValueChangeEntry<E, P>, AegonError> {
                if let Some(&i) = placement_idx.get(slot_bits) {
                    // Brand-new placement: reuse the already-computed
                    // openings from the §6.4 pre/post passes.
                    Ok(ValueChangeEntry {
                        slot_bits: slot_bits.clone(),
                        rand_value_pre_eval: pre_rand_value[i].0,
                        rand_value_pre_proof: pre_rand_value[i].1.clone(),
                        rand_value_post_eval: post_rand_value[i].0,
                        rand_value_post_proof: post_rand_value[i].1.clone(),
                        value_post_eval: post_value[i].0,
                        value_post_proof: post_value[i].1.clone(),
                    })
                } else {
                    // Value-only update on a slot that was already
                    // occupied. The §6.4 placement loop skipped this
                    // slot, so we pre-captured the pre-update
                    // rand_value opening above (in
                    // `voup_pre_rand_value`); the post-update
                    // rand_value and value openings are computed fresh
                    // against the now-mutated `self.*` state.
                    let pre = voup_pre_rand_value
                        .get(slot_bits)
                        .expect("pre-opening captured for every value-only-update slot")
                        .clone();
                    let post = open_at_point_non_zk::<E, P>(
                        pp,
                        &self.rand_value_poly,
                        &self.rand_value_commitment,
                        &self.rand_value_state,
                        slot_bits,
                        dims,
                        b"aegon.rand_value.open",
                    )?;
                    let val = open_at_point_non_zk::<E, P>(
                        pp,
                        &self.value_poly,
                        &self.value_commitment,
                        &self.value_state,
                        slot_bits,
                        dims,
                        b"aegon.value.open",
                    )?;
                    Ok(ValueChangeEntry {
                        slot_bits: slot_bits.clone(),
                        rand_value_pre_eval: pre.0,
                        rand_value_pre_proof: pre.1,
                        rand_value_post_eval: post.0,
                        rand_value_post_proof: post.1,
                        value_post_eval: val.0,
                        value_post_proof: val.1,
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "tracing_instrument")]
        drop(_value_change_span);
        if super::instrument::publish_profile_enabled() {
            eprintln!(
                "[pub-profile] phase2.value_change_openings_parallel: {:.3} ms (entries={} voup_fresh_opens={})",
                _vc_open_t.elapsed().as_secs_f64() * 1000.0,
                value_change_slots.len(),
                value_change_slots
                    .iter()
                    .filter(|s| !placement_idx.contains_key(*s))
                    .count(),
            );
        }

        let history = HistoryOpenings {
            entries: new_label_entries,
            value_changes: value_change_entries,
        };

        // Finalize the new epoch. `self.*` already holds the new
        // commitments / states / polynomials (mutated in place during
        // phase 1 and earlier in this phase), so the only state we
        // still need to update here is the Fiat-Shamir chain randomness
        // and the epoch counter.
        #[cfg(feature = "tracing_instrument")]
        let _finalize_span = tracing::debug_span!("Aegon::Phase2::FinalizeEpoch").entered();
        self.r_index = new_r_index;
        self.r_value = new_r_value;
        self.epoch += 1;

        // When `retain_epoch_polys` is false (bench mode), keep only the
        // four commitments per epoch and drop the rand polys + states.
        // `epoch_commitment(epoch)` (used by `verify_sharded_invariance`)
        // reads only the commitments; the dropped fields are needed only
        // by `open_rand_*_at_slot_in_epoch`, which the bench doesn't
        // call. Frees ~50 MB/epoch — turns an OOM at ~280 epochs into a
        // run that scales linearly through 90% fill.
        let (snap_rand_index_poly, snap_rand_index_state, snap_rand_value_poly, snap_rand_value_state) =
            if self.retain_epoch_polys {
                (
                    Some(self.rand_index_poly.clone()),
                    Some(self.rand_index_state.clone()),
                    Some(self.rand_value_poly.clone()),
                    Some(self.rand_value_state.clone()),
                )
            } else {
                (None, None, None, None)
            };
        // Always snapshot the new epoch's value-side `tau_f` scalars.
        // Even with `retain_epoch_polys=false` (bench mode) we keep
        // these so lookup_history's masking layer can find the right
        // `tau_f` per epoch. Two 32-byte field elements per epoch —
        // ~256 KB at 4K epochs, independent of `retain_epoch_polys`.
        let (publish_value_tau, publish_rand_value_tau) = self.snapshot_value_taus();
        self.epoch_history.insert(
            self.epoch,
            EpochSnapshot {
                index_commitment: self.index_commitment.clone(),
                value_commitment: self.value_commitment.clone(),
                rand_index_poly: snap_rand_index_poly,
                rand_index_commitment: self.rand_index_commitment.clone(),
                rand_index_state: snap_rand_index_state,
                rand_value_poly: snap_rand_value_poly,
                rand_value_commitment: self.rand_value_commitment.clone(),
                rand_value_state: snap_rand_value_state,
                value_tau: publish_value_tau,
                rand_value_tau: publish_rand_value_tau,
            },
        );
        log_rss_ctx("phase2.exit", &format!("epoch={}", self.epoch));
        if super::instrument::publish_profile_enabled() {
            eprintln!(
                "[pub-profile] PHASE2_TOTAL: {:.3} ms",
                _phase2_total_t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        Ok((self.current_commitment(), history))
    }

    /// Find the first free slot for `label` via local-only open addressing.
    /// Pure read (no mutation). `extra_claimed` is a set of `usize`-indices
    /// already claimed within the same in-flight batch — the caller is
    /// responsible for tracking these because `index_poly` is only
    /// mutated after the whole batch has been planned.
    fn find_free_slot(
        &self,
        label: &[u8],
        extra_claimed: &std::collections::HashSet<usize>,
    ) -> Result<(Vec<bool>, u64), AegonError> {
        let capacity = 1usize << self.log_capacity;
        for ctr in 0..(capacity as u64) {
            let bool_index = H::h_bits(ctr, label, self.log_capacity);
            let usize_index = bool_index_to_usize(&bool_index, &self.dims);
            let occupied = self
                .index_poly
                .evaluations
                .get(&usize_index)
                .map(|v| !v.is_zero())
                .unwrap_or(false)
                || extra_claimed.contains(&usize_index);
            if !occupied {
                return Ok((bool_index, ctr));
            }
        }
        Err(AegonError::DictionaryFull { capacity })
    }

    /// Whether `slot_bits` in the *current* index polynomial holds a
    /// non-zero entry. The sharded coordinator queries this while
    /// walking a probe trail across shards.
    pub fn is_index_slot_occupied(&self, slot_bits: &[bool]) -> bool {
        let idx = bool_index_to_usize(slot_bits, &self.dims);
        self.index_poly
            .evaluations
            .get(&idx)
            .map(|v| !v.is_zero())
            .unwrap_or(false)
    }

    /// Re-mask a publish-time **non-ZK** opening into a hiding (ZK)
    /// one. Used by the history-lookup path: every
    /// [`crate::aegon::sharded::StoredValueHistoryEntry`] carries
    /// three plain proofs computed at publish time; before the
    /// coordinator hands the entry to a user it asks the owning
    /// shard to upgrade each plain proof via this helper.
    ///
    /// Inputs mirror the masking-server protocol: `commitment` is
    /// the polynomial's commitment at the same epoch as the stored
    /// proof; `non_zk_proof` and `tau_f` are the snapshot the
    /// publish stored; `slot_bits` is the original opening point;
    /// `transcript_label` matches the one the publish-time open
    /// used.
    pub fn remask_value_side_proof(
        &self,
        commitment: &P::Commitment,
        slot_bits: &[bool],
        value: &E::ScalarField,
        non_zk_proof: P::Proof,
        tau_f: &P::HidingScalar,
        transcript_label: &'static [u8],
    ) -> Result<P::Proof, AegonError> {
        // Same hiding-SRS requirement as `open_value_side_at_point`:
        // remask needs the polynomial's commit-time `tau_f` to be
        // defined. On a non-hiding SRS, just pass the plain proof
        // through.
        if !self.prover_param.is_zk() {
            return Ok(non_zk_proof);
        }
        let point = bool_index_to_point::<E::ScalarField>(slot_bits);
        // Hiding mode always has a masking source — either the
        // default in-process pool wired by `init_with_arc` or a
        // remote `MaskingClient` swapped in via `set_masking_source`.
        let source = self.masking_client.as_ref().ok_or_else(|| {
            AegonError::Config(
                "hiding-mode Aegon missing a MaskingSource — this should be impossible".into(),
            )
        })?;
        let package = source
            .fetch_package(self.log_capacity)
            .map_err(|e| AegonError::Config(format!("masking fetch: {e}")))?;
        let mut tr = IOPTranscript::<E::ScalarField>::new(transcript_label);
        let proof = P::remask_with_package(
            self.prover_param.as_ref(),
            commitment,
            &point,
            value,
            non_zk_proof,
            tau_f,
            &mut tr,
            &package,
        )?;
        Ok(proof)
    }

    /// Helper: hiding-open `poly` at `slot_bits` using the masking-
    /// server protocol. Used by every value-side opening that goes to
    /// users (value-poly lookups + rand_value-poly freshness +
    /// rand_value-poly consistency openings).
    ///
    /// If `self.masking_client` is set, fetches a one-shot package
    /// from the masking server. Otherwise (typical for tests) it
    /// generates a package inline against `self.prover_param` — only
    /// works when the SRS is hiding, but tests are the only path that
    /// hits the fallback.
    fn open_value_side_at_point(
        &self,
        poly: &SparseMultilinearExtension<E::ScalarField>,
        com: &P::Commitment,
        state: &P::State,
        slot_bits: &[bool],
        transcript_label: &'static [u8],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        // The masking-server protocol requires a hiding SRS: the
        // polynomial's commit-time `tau_f` is needed to compute
        // `rho_prime = alpha * tau_f + rho`. On a non-hiding SRS the
        // commitment has no `tau_f * h` term, so the consumer can't
        // construct a valid ZK opening from a masking package —
        // fall back to a plain non-ZK opening. Keeps tests that run
        // with `private = false` working without a masking server,
        // while production (hiding SRS + masking client) always goes
        // through the ZK path.
        if !self.prover_param.is_zk() {
            return open_at_point_non_zk::<E, P>(
                self.prover_param.as_ref(),
                poly,
                com,
                state,
                slot_bits,
                &self.dims,
                transcript_label,
            );
        }
        let point = bool_index_to_point::<E::ScalarField>(slot_bits);
        let usize_idx = bool_index_to_usize(slot_bits, &self.dims);
        let evaluation = poly
            .evaluations
            .get(&usize_idx)
            .copied()
            .unwrap_or_else(<E::ScalarField as Zero>::zero);
        // Hiding mode always has a masking source — see comment in
        // `remask_value_side_proof`.
        let source = self.masking_client.as_ref().ok_or_else(|| {
            AegonError::Config(
                "hiding-mode Aegon missing a MaskingSource — this should be impossible".into(),
            )
        })?;
        let package = source
            .fetch_package(self.log_capacity)
            .map_err(|e| AegonError::Config(format!("masking fetch: {e}")))?;
        let mut tr = IOPTranscript::<E::ScalarField>::new(transcript_label);
        let (proof, _) = P::open_zk_with_package(
            self.prover_param.as_ref(),
            com,
            DenseOrSparseMLERef::Sparse(poly),
            &point,
            state,
            &mut tr,
            &package,
        )?;
        Ok((evaluation, proof))
    }

    /// Open the current `index_poly` commitment at `slot_bits`.
    /// Label-side opening: auto-dispatch on SRS hiding-ness. Under a
    /// hiding SRS this is still ZK (via the PCS's inline sampling) —
    /// the masking-server protocol is intentionally value-side only,
    /// so labels skip the round-trip and pay the inline cost instead.
    pub fn open_index_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        // Label commitments are plain (no `tau*h`), so the opening
        // must be plain too — match `commit_with_aux_non_zk` in
        // `init`/`restore`.
        open_at_point_non_zk::<E, P>(
            self.prover_param.as_ref(),
            &self.index_poly,
            &self.index_commitment,
            &self.index_state,
            slot_bits,
            &self.dims,
            b"aegon.index.open",
        )
    }

    /// Open the current `value_poly` commitment at `slot_bits`.
    /// Value-side opening — hiding (ZK) via the masking-server
    /// protocol.
    pub fn open_value_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        self.open_value_side_at_point(
            &self.value_poly,
            &self.value_commitment,
            &self.value_state,
            slot_bits,
            b"aegon.value.open",
        )
    }

    /// Open `rand_index_poly` at `slot_bits` for some retained `epoch`.
    /// Used by sharded consistency proofs to open both `s0` and the
    /// current epoch at the same probe point. Label-side opening:
    /// auto-dispatches on SRS hiding-ness (inline-ZK under hiding).
    ///
    /// Returns `AegonError::InvalidEpoch` when `epoch` is unknown OR
    /// when its snapshot was recorded with `retain_epoch_polys = false`
    /// (the polys + state were dropped on purpose to save memory).
    pub fn open_rand_index_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let snap = self
            .epoch_history
            .get(&epoch)
            .ok_or(AegonError::InvalidEpoch(epoch))?;
        let poly = snap
            .rand_index_poly
            .as_ref()
            .ok_or(AegonError::InvalidEpoch(epoch))?;
        let state = snap
            .rand_index_state
            .as_ref()
            .ok_or(AegonError::InvalidEpoch(epoch))?;
        // Label-side: plain commit ⇒ plain opening.
        open_at_point_non_zk::<E, P>(
            self.prover_param.as_ref(),
            poly,
            &snap.rand_index_commitment,
            state,
            slot_bits,
            &self.dims,
            b"aegon.rand_index.open",
        )
    }

    /// Open `rand_value_poly` at `slot_bits` for some retained `epoch`.
    /// Value-side opening — hiding (ZK) via the masking-server
    /// protocol.
    ///
    /// Returns `AegonError::InvalidEpoch` when `epoch` is unknown OR
    /// when its snapshot was recorded with `retain_epoch_polys = false`.
    pub fn open_rand_value_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let snap = self
            .epoch_history
            .get(&epoch)
            .ok_or(AegonError::InvalidEpoch(epoch))?;
        let poly = snap
            .rand_value_poly
            .as_ref()
            .ok_or(AegonError::InvalidEpoch(epoch))?;
        let state = snap
            .rand_value_state
            .as_ref()
            .ok_or(AegonError::InvalidEpoch(epoch))?;
        self.open_value_side_at_point(
            poly,
            &snap.rand_value_commitment,
            state,
            slot_bits,
            b"aegon.rand_value.open",
        )
    }

    /// Open the live `rand_value_poly` at `slot_bits`. Unlike the
    /// `_in_epoch` variant this does not require the epoch to have
    /// been retained in `epoch_history` — it opens against the
    /// currently-running commitment + state, which is always
    /// available. Used by `lookup_history` to ship the verifier a
    /// "no change since the latest history entry" attestation.
    /// Value-side opening — hiding (ZK) via the masking-server
    /// protocol.
    pub fn open_rand_value_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        self.open_value_side_at_point(
            &self.rand_value_poly,
            &self.rand_value_commitment,
            &self.rand_value_state,
            slot_bits,
            b"aegon.rand_value.open",
        )
    }

    /// Open the live `rand_index_poly` at `slot_bits`. Label-side
    /// mirror of [`open_rand_value_at_slot_current`]. Used by
    /// `lookup_label_history` to ship the verifier a "no change
    /// since placement" attestation. Label-side opening: auto-
    /// dispatches on SRS hiding-ness (inline-ZK under hiding).
    pub fn open_rand_index_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        // Label-side: plain commit ⇒ plain opening.
        open_at_point_non_zk::<E, P>(
            self.prover_param.as_ref(),
            &self.rand_index_poly,
            &self.rand_index_commitment,
            &self.rand_index_state,
            slot_bits,
            &self.dims,
            b"aegon.rand_index.open",
        )
    }

    fn set_value(&mut self, usize_index: usize, value: E::ScalarField) {
        if value.is_zero() {
            self.value_poly.evaluations.remove(&usize_index);
        } else {
            self.value_poly.evaluations.insert(usize_index, value);
        }
    }

    pub fn lookup(&self, label: &Label) -> Result<LookupProof<E, P>, AegonError> {
        let (bool_index, ctr0) = self
            .label_table
            .get(label)
            .ok_or_else(|| AegonError::UnknownLabel(label.clone()))?;

        // Label-side: plain commit ⇒ plain index-probe openings.
        let mut probes = Vec::with_capacity(*ctr0 as usize + 1);
        for ctr in 0..=*ctr0 {
            let probe_bits = H::h_bits(ctr, label, self.log_capacity);
            let (evaluation, proof) = open_at_point_non_zk::<E, P>(
                self.prover_param.as_ref(),
                &self.index_poly,
                &self.index_commitment,
                &self.index_state,
                &probe_bits,
                &self.dims,
                b"aegon.index.open",
            )?;
            probes.push((evaluation, proof));
        }

        // Value-side: the value-poly opening goes to the user, so
        // it's hiding (ZK) via the masking-server protocol.
        let (value_evaluation, value_proof) = self.open_value_side_at_point(
            &self.value_poly,
            &self.value_commitment,
            &self.value_state,
            bool_index,
            b"aegon.value.open",
        )?;

        Ok(LookupProof {
            ctr0: *ctr0,
            probes,
            value_evaluation,
            value_proof,
        })
    }

    /// Build a consistency proof that the user's slot at `label` was
    /// unchanged between `s0` and the current epoch. The server retains
    /// snapshots of the rand polynomials at every epoch, so reaching
    /// back to `s0` is just a map lookup. Returns `Err(InvalidEpoch)` if
    /// `s0` was not retained, `Err(UnknownLabel)` if the label has never
    /// been registered.
    pub fn consistency_proof(
        &self,
        label: &Label,
        s0: u64,
    ) -> Result<super::types::ConsistencyProof<E, P>, AegonError> {
        let snap_s0 = self
            .epoch_history
            .get(&s0)
            .ok_or(AegonError::InvalidEpoch(s0))?;
        // Pair-openings reach back to `s0`'s polynomial state — only
        // available when `retain_epoch_polys` was true at that publish.
        let snap_s0_rand_index_poly = snap_s0
            .rand_index_poly
            .as_ref()
            .ok_or(AegonError::InvalidEpoch(s0))?;
        let snap_s0_rand_index_state = snap_s0
            .rand_index_state
            .as_ref()
            .ok_or(AegonError::InvalidEpoch(s0))?;
        let snap_s0_rand_value_poly = snap_s0
            .rand_value_poly
            .as_ref()
            .ok_or(AegonError::InvalidEpoch(s0))?;
        let snap_s0_rand_value_state = snap_s0
            .rand_value_state
            .as_ref()
            .ok_or(AegonError::InvalidEpoch(s0))?;
        let (bool_index, ctr0) = self
            .label_table
            .get(label)
            .ok_or_else(|| AegonError::UnknownLabel(label.clone()))?;
        let ctr0 = *ctr0;

        // Index half: open rand_index at every probe point at both s0
        // and current. Equality at *every* probe rules out any change to
        // the user's open-addressing path (paper §6.1).
        let mut index_witnesses = Vec::with_capacity(ctr0 as usize + 1);
        for ctr in 0..=ctr0 {
            let probe_bits = H::h_bits(ctr, label, self.log_capacity);
            let point = bool_index_to_point::<E::ScalarField>(&probe_bits);

            // Label-side: plain commit ⇒ plain pair-opening.
            let pair = open_pair_non_zk::<E, P>(
                self.prover_param.as_ref(),
                &point,
                snap_s0_rand_index_poly,
                &snap_s0.rand_index_commitment,
                snap_s0_rand_index_state,
                &self.rand_index_poly,
                &self.rand_index_commitment,
                &self.rand_index_state,
            )?;
            index_witnesses.push(pair);
        }

        // Value half: open rand_value at the user's slot.
        let value_point = bool_index_to_point::<E::ScalarField>(bool_index);
        let value_witness = open_pair::<E, P>(
            self.prover_param.as_ref(),
            &value_point,
            snap_s0_rand_value_poly,
            &snap_s0.rand_value_commitment,
            snap_s0_rand_value_state,
            &self.rand_value_poly,
            &self.rand_value_commitment,
            &self.rand_value_state,
            b"aegon.rand_value.open",
        )?;

        Ok(super::types::ConsistencyProof {
            ctr0,
            index_witnesses,
            value_witness,
        })
    }
}

// ---------- helpers --------------------------------------------------

/// One-shot "open this poly at `slot_bits`" used by the slot-driven
/// API. Returns the evaluation alongside the proof. Used by the
/// sharded coordinator (which has already computed `slot_bits`) and
/// shared with the legacy label-driven lookup path internally.
fn open_at_point<E, P>(
    pp: &P::ProverParam,
    poly: &SparseMultilinearExtension<E::ScalarField>,
    com: &P::Commitment,
    state: &P::State,
    slot_bits: &[bool],
    dims: &[usize],
    transcript_label: &'static [u8],
) -> Result<(E::ScalarField, P::Proof), AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    let point = bool_index_to_point::<E::ScalarField>(slot_bits);
    let usize_idx = bool_index_to_usize(slot_bits, dims);
    let evaluation = poly
        .evaluations
        .get(&usize_idx)
        .copied()
        .unwrap_or_else(<E::ScalarField as Zero>::zero);
    let mut tr = IOPTranscript::<E::ScalarField>::new(transcript_label);
    let (proof, _) = P::open(
        pp,
        com,
        DenseOrSparseMLERef::Sparse(poly),
        &point,
        state,
        &mut tr,
    )?;
    Ok((evaluation, proof))
}

/// Explicit non-ZK opener — always produces a plain
/// `({D_j}, f_tail)` proof regardless of whether the SRS is hiding.
/// Used for:
/// * label-side openings (the masking-server protocol is value-side
///   only, so the label-side stays cheap and non-hiding by design),
/// * publish-time stored openings (the user-facing
///   "the masking server is not called in the publish path" rule —
///   stored value-side openings get re-masked at history-lookup time
///   via [`P::remask_with_package`]).
fn open_at_point_non_zk<E, P>(
    pp: &P::ProverParam,
    poly: &SparseMultilinearExtension<E::ScalarField>,
    com: &P::Commitment,
    state: &P::State,
    slot_bits: &[bool],
    dims: &[usize],
    _transcript_label: &'static [u8],
) -> Result<(E::ScalarField, P::Proof), AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    let point = bool_index_to_point::<E::ScalarField>(slot_bits);
    let usize_idx = bool_index_to_usize(slot_bits, dims);
    let evaluation = poly
        .evaluations
        .get(&usize_idx)
        .copied()
        .unwrap_or_else(<E::ScalarField as Zero>::zero);
    let (proof, _) = P::open_non_zk(
        pp,
        com,
        DenseOrSparseMLERef::Sparse(poly),
        &point,
        state,
    )?;
    Ok((evaluation, proof))
}

/// `commit` followed by `update_state`. KZH-k splits commitment from
/// per-row aux precomputation; PCSs that don't need aux state can
/// override `update_state` to a no-op. We always call it so every
/// backend gets a consistent commit→open contract.
#[cfg_attr(
    feature = "tracing_instrument",
    tracing::instrument(
        level = "debug",
        skip_all,
        name = "Aegon::CommitWithAux",
        fields(nnz = poly.evaluations.len())
    )
)]
fn commit_with_aux<E, P>(
    pp: &P::ProverParam,
    poly: &SparseMultilinearExtension<E::ScalarField>,
) -> Result<(P::Commitment, P::State), AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    let wrapped = DenseOrSparseMLE::Sparse(poly.clone());
    let (com, mut state) = P::commit(pp, &wrapped)?;
    P::update_state(pp, &wrapped, &com, &mut state)?;
    Ok((com, state))
}

/// Plain (non-hiding) variant of [`commit_with_aux`] used for the
/// **label-side** polynomials (`index_poly`, `rand_index_poly`).
/// Label polys carry no hiding overhead — no `tau*h` blinding in the
/// commitment, no `tau` in the state — regardless of whether the SRS
/// would otherwise support hiding. The masking-server protocol is
/// value-side only.
fn commit_with_aux_non_zk<E, P>(
    pp: &P::ProverParam,
    poly: &SparseMultilinearExtension<E::ScalarField>,
) -> Result<(P::Commitment, P::State), AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    let wrapped = DenseOrSparseMLE::Sparse(poly.clone());
    let (com, mut state) = P::commit_non_zk(pp, &wrapped)?;
    P::update_state(pp, &wrapped, &com, &mut state)?;
    Ok((com, state))
}

/// Hiding variant of [`commit_with_aux`] used for the **value-side**
/// polynomials (`value_poly`, `rand_value_poly`) when the SRS
/// supports hiding. Produces `C = <f, H_1> + tau*h` and stores the
/// polynomial's `tau` in the state — the masking-server protocol
/// reads it back as `tau_f` to compute `rho_prime = alpha*tau_f + rho`
/// at open time. Falls back to plain commit when the SRS is
/// non-hiding so test code using a plain SRS still works (value
/// openings just become non-ZK in that case).
fn commit_with_aux_value_side<E, P>(
    pp: &P::ProverParam,
    poly: &SparseMultilinearExtension<E::ScalarField>,
) -> Result<(P::Commitment, P::State), AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    P::ProverParam: akd_core::aegon_crypto::pcs::PCSGlobalParam,
{
    let wrapped = DenseOrSparseMLE::Sparse(poly.clone());
    let (com, mut state) = if pp.is_zk() {
        P::commit_zk(pp, &wrapped)?
    } else {
        P::commit_non_zk(pp, &wrapped)?
    };
    P::update_state(pp, &wrapped, &com, &mut state)?;
    Ok((com, state))
}

/// Free-fn analogue of [`Aegon::snapshot_value_taus`] for code paths
/// (init / restore / prefill) where `self` isn't yet built. Same
/// rule: under a hiding SRS, capture each value-side poly's `tau_f`
/// via `P::get_hiding_scalar`; under a non-hiding SRS, return
/// `(None, None)`.
fn extract_value_taus<E, P>(
    pp: &P::ProverParam,
    value_state: &P::State,
    rand_value_state: &P::State,
) -> (Option<P::HidingScalar>, Option<P::HidingScalar>)
where
    E: Pairing,
    P: AegonPcs<E>,
    P::ProverParam: akd_core::aegon_crypto::pcs::PCSGlobalParam,
{
    if pp.is_zk() {
        (
            Some(P::get_hiding_scalar(value_state)),
            Some(P::get_hiding_scalar(rand_value_state)),
        )
    } else {
        (None, None)
    }
}

/// `rand += r · (next - prev)` on sparse evaluation tables. The support
/// of the result is the union of the supports of `rand`, `prev`, and
/// `next`; we walk each non-zero entry of `(next - prev)` exactly once.
#[cfg_attr(
    feature = "tracing_instrument",
    tracing::instrument(level = "debug", skip_all, name = "Aegon::UpdateRand")
)]
fn update_rand<F: ark_ff::Field>(
    rand: &mut SparseMultilinearExtension<F>,
    prev: &SparseMultilinearExtension<F>,
    next: &SparseMultilinearExtension<F>,
    r: F,
) {
    // Indices where prev or next is non-zero — the only places ∆ is
    // non-zero.
    let mut indices: std::collections::BTreeSet<usize> =
        prev.evaluations.keys().copied().collect();
    indices.extend(next.evaluations.keys().copied());

    for idx in indices {
        let p = prev.evaluations.get(&idx).copied().unwrap_or(F::ZERO);
        let n = next.evaluations.get(&idx).copied().unwrap_or(F::ZERO);
        let delta = n - p;
        if delta.is_zero() {
            continue;
        }
        let contribution = r * delta;
        let cur = rand.evaluations.get(&idx).copied().unwrap_or(F::ZERO);
        let updated = cur + contribution;
        if updated.is_zero() {
            rand.evaluations.remove(&idx);
        } else {
            rand.evaluations.insert(idx, updated);
        }
    }
}

/// Same as [`update_rand`] but takes the delta polynomial directly,
/// avoiding the need to keep `prev` and `next` polynomials side-by-side
/// in memory. The caller (publish_phase_2) already builds `delta` in
/// phase_1, so this saves cloning prev across the phase boundary.
///
/// Computes `rand += r · delta` slot by slot, dropping any slot whose
/// updated value becomes zero.
fn update_rand_with_delta<F: ark_ff::Field>(
    rand: &mut SparseMultilinearExtension<F>,
    delta: &SparseMultilinearExtension<F>,
    r: F,
) {
    for (idx, d) in &delta.evaluations {
        if d.is_zero() {
            continue;
        }
        let contribution = r * *d;
        let cur = rand.evaluations.get(idx).copied().unwrap_or(F::ZERO);
        let updated = cur + contribution;
        if updated.is_zero() {
            rand.evaluations.remove(idx);
        } else {
            rand.evaluations.insert(*idx, updated);
        }
    }
}

/// Open a polynomial at `point` against two different epoch states
/// (`s0` and `s1`). Returns the paired evaluations and proofs the user
/// later checks for equality.
fn open_pair<E, P>(
    pp: &P::ProverParam,
    point: &Vec<E::ScalarField>,
    poly_s0: &SparseMultilinearExtension<E::ScalarField>,
    com_s0: &P::Commitment,
    state_s0: &P::State,
    poly_s1: &SparseMultilinearExtension<E::ScalarField>,
    com_s1: &P::Commitment,
    state_s1: &P::State,
    transcript_label: &'static [u8],
) -> Result<RandPair<E, P>, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    let mut tr0 = IOPTranscript::<E::ScalarField>::new(transcript_label);
    let (proof_s0, eval_s0) = P::open(
        pp,
        com_s0,
        DenseOrSparseMLERef::Sparse(poly_s0),
        point,
        state_s0,
        &mut tr0,
    )?;
    let mut tr1 = IOPTranscript::<E::ScalarField>::new(transcript_label);
    let (proof_s1, eval_s1) = P::open(
        pp,
        com_s1,
        DenseOrSparseMLERef::Sparse(poly_s1),
        point,
        state_s1,
        &mut tr1,
    )?;
    Ok(RandPair {
        eval_s0,
        proof_s0,
        eval_s1,
        proof_s1,
    })
}

/// Label-side variant of [`open_pair`]: always plain (non-ZK)
/// openings, regardless of SRS hiding-ness. Used for `rand_index`
/// pair openings in `build_consistency_proof` — label-side commits
/// are plain, so their openings must be plain too.
fn open_pair_non_zk<E, P>(
    pp: &P::ProverParam,
    point: &Vec<E::ScalarField>,
    poly_s0: &SparseMultilinearExtension<E::ScalarField>,
    com_s0: &P::Commitment,
    state_s0: &P::State,
    poly_s1: &SparseMultilinearExtension<E::ScalarField>,
    com_s1: &P::Commitment,
    state_s1: &P::State,
) -> Result<RandPair<E, P>, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    let (proof_s0, eval_s0) = P::open_non_zk(
        pp,
        com_s0,
        DenseOrSparseMLERef::Sparse(poly_s0),
        point,
        state_s0,
    )?;
    let (proof_s1, eval_s1) = P::open_non_zk(
        pp,
        com_s1,
        DenseOrSparseMLERef::Sparse(poly_s1),
        point,
        state_s1,
    )?;
    Ok(RandPair {
        eval_s0,
        proof_s0,
        eval_s1,
        proof_s1,
    })
}

// `build_invariance_proof` + `build_chain_witness` used to live here.
// Both became dead code when the auditor switched to a commitment-
// homomorphism check: the prover no longer opens the four polynomials
// at a Fiat-Shamir random point, because every group element the
// auditor needs is already in the published `EpochCommitment`. See
// `audit::verify_invariance` for the verifier side.

/// Try to load a previously-persisted [`AegonCheckpoint`] for the
/// given shard out of its private DB. Returns `Ok(None)` for a fresh
/// DB (no `aegon:shard:{shard_id}:state` key) or `DbSource::None`.
/// Returns `Err` if the key is present but malformed.
///
/// TODO(shard-checkpoint-fault-tolerance): paired with the disabled
/// per-publish writer in `shard_grpc.rs`. The current shipping
/// configuration never writes checkpoints, so this function will
/// always return `Ok(None)` from a fresh start. Kept compiled (not
/// `#[cfg]`-gated) so the read-path code stays exercised and ready
/// to flip on when fault tolerance is wired up.
pub fn load_aegon_checkpoint_from_db<E, P>(
    db_source: &DbSource,
    shard_id: u32,
) -> Result<Option<AegonCheckpoint<E, P>>, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    AegonCheckpoint<E, P>: CanonicalDeserialize,
{
    use super::db::Db;
    let db: Option<Box<dyn Db>> = match db_source {
        DbSource::None => None,
        DbSource::Redis(url) => Some(Box::new(RedisDb::connect(url)?)),
        DbSource::Rocks(path) => Some(Box::new(super::db::RocksDb::open(path)?)),
    };
    let Some(db) = db else { return Ok(None) };
    match db.get(&key_shard_state(shard_id))? {
        Some(bytes) => {
            let ckpt = AegonCheckpoint::<E, P>::deserialize_compressed(&bytes[..])
                .map_err(|e| {
                    AegonError::Database(format!("deserialize shard {shard_id} checkpoint: {e}"))
                })?;
            Ok(Some(ckpt))
        },
        None => Ok(None),
    }
}
