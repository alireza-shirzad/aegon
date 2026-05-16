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
use akd_core::aegon_crypto::pcs::PCSGlobalParam;
use akd_core::aegon_crypto::poly::DenseOrSparseMLE;
use akd_core::aegon_crypto::transcript::IOPTranscript;

use super::config::{AegonConfig, VerifierContext};
use super::db::{key_shard_state, DbSource, RedisDb};
use super::error::AegonError;
use super::fs::derive_chain_scalar;
use super::hash::{bool_index_to_point, bool_index_to_usize, HashSuite, Sha256Hash};
use super::sharded::ShardWrite;
use super::types::{
    AegonPcs, EpochCommitment, HistoryOpeningEntry, HistoryOpenings, Label, LookupProof, RandPair,
    Value,
};

/// Snapshot of the polynomials needed for serving consistency proofs at
/// epoch `n`. The two data commitments are kept so we can return them in
/// `EpochCommitment`s; the rand polynomials and their PCS state are kept
/// so the server can produce opening proofs at arbitrary points later.
#[derive(Clone)]
struct EpochSnapshot<E: Pairing, P: AegonPcs<E>> {
    index_commitment: P::Commitment,
    value_commitment: P::Commitment,

    rand_index_poly: SparseMultilinearExtension<E::ScalarField>,
    rand_index_commitment: P::Commitment,
    rand_index_state: P::State,

    rand_value_poly: SparseMultilinearExtension<E::ScalarField>,
    rand_value_commitment: P::Commitment,
    rand_value_state: P::State,
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

    prover_param: P::ProverParam,
    verifier_param: P::VerifierParam,

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
/// Used by the gRPC `ShardServer` to persist its state into Redis at
/// the end of every publish, and on startup to restore.
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
    prev_index_poly: SparseMultilinearExtension<E::ScalarField>,
    prev_value_poly: SparseMultilinearExtension<E::ScalarField>,
    prev_index_com: P::Commitment,
    prev_index_state: P::State,
    prev_value_com: P::Commitment,
    prev_value_state: P::State,
    prev_rand_index_poly: SparseMultilinearExtension<E::ScalarField>,
    prev_rand_index_com: P::Commitment,
    prev_rand_index_state: P::State,
    prev_rand_value_poly: SparseMultilinearExtension<E::ScalarField>,
    prev_rand_value_com: P::Commitment,
    prev_rand_value_state: P::State,
    new_index_com: P::Commitment,
    new_index_state: P::State,
    new_value_com: P::Commitment,
    new_value_state: P::State,
    /// Commitment of the **delta** polynomial committed in phase 1 —
    /// i.e. only the slots this batch touched, evaluations equal to
    /// `new − prev`. Reused in phase 2 to derive the rand-poly
    /// commitments via the homomorphism
    /// `delta_rand_X_com = r_X · delta_X_com`, so phase 2 never has to
    /// run another MSM over the rand polynomials.
    delta_index_com: P::Commitment,
    delta_value_com: P::Commitment,
    /// Prover `State` for the same delta polynomials. The KZH-k aux
    /// table over a batch-sized support is built in phase 1 (cost
    /// `O(k · batch)`); phase 2 then uses the State homomorphism
    /// `new_rand_state = prev_rand_state + r_X · delta_state` via
    /// `P::fma_state` to avoid recomputing aux from scratch.
    delta_index_state: P::State,
    delta_value_state: P::State,
    /// Slot bits for every brand-new placement in this batch (entries
    /// whose `h_label` is `Some`). Populated by `publish_phase_1`,
    /// consumed by `publish_phase_2` to drive the §6.4 history-opening
    /// computation. Empty when the batch was value-updates only.
    new_label_slots: Vec<Vec<bool>>,
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
    pub fn setup<R: Rng>(rng: &mut R, config: &AegonConfig<E, P>) -> Result<Self, AegonError> {
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
    ) -> Result<Self, AegonError> {
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

        let (index_commitment, index_state) =
            commit_with_aux::<E, P>(&prover_param, &index_poly)?;
        let (value_commitment, value_state) =
            commit_with_aux::<E, P>(&prover_param, &value_poly)?;
        let (rand_index_commitment, rand_index_state) =
            commit_with_aux::<E, P>(&prover_param, &rand_index_poly)?;
        let (rand_value_commitment, rand_value_state) =
            commit_with_aux::<E, P>(&prover_param, &rand_value_poly)?;

        let mut epoch_history = BTreeMap::new();
        epoch_history.insert(
            0,
            EpochSnapshot {
                index_commitment: index_commitment.clone(),
                value_commitment: value_commitment.clone(),
                rand_index_poly: rand_index_poly.clone(),
                rand_index_commitment: rand_index_commitment.clone(),
                rand_index_state: rand_index_state.clone(),
                rand_value_poly: rand_value_poly.clone(),
                rand_value_commitment: rand_value_commitment.clone(),
                rand_value_state: rand_value_state.clone(),
            },
        );

        Ok(Self {
            log_capacity,
            dims,
            prover_param,
            verifier_param,
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
            pending: None,
            _phantom: PhantomData,
        })
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
    {
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
        epoch_history.insert(
            ckpt.epoch,
            EpochSnapshot {
                index_commitment: ckpt.index_commitment.clone(),
                value_commitment: ckpt.value_commitment.clone(),
                rand_index_poly: rand_index_poly.clone(),
                rand_index_commitment: ckpt.rand_index_commitment.clone(),
                rand_index_state: ckpt.rand_index_state.clone(),
                rand_value_poly: rand_value_poly.clone(),
                rand_value_commitment: ckpt.rand_value_commitment.clone(),
                rand_value_state: ckpt.rand_value_state.clone(),
            },
        );

        let mut label_table: HashMap<Label, (Vec<bool>, u64)> =
            HashMap::with_capacity(ckpt.label_table_entries.len());
        for (label, bits, ctr) in ckpt.label_table_entries {
            label_table.insert(label, (bits, ctr));
        }

        Ok(Self {
            log_capacity,
            dims,
            prover_param,
            verifier_param,
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
            pending: None,
            _phantom: PhantomData,
        })
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
        // still zero.
        let (com_i, state_i) = commit_with_aux::<E, P>(&self.prover_param, &self.index_poly)?;
        let (com_v, state_v) = commit_with_aux::<E, P>(&self.prover_param, &self.value_poly)?;
        self.index_commitment = com_i;
        self.index_state = state_i;
        self.value_commitment = com_v;
        self.value_state = state_v;
        // Refresh the epoch-0 snapshot so consistency-proof queries
        // against `epoch = 0` see the prefilled state, not the empty
        // state that `setup` originally inserted.
        self.epoch_history.insert(
            0,
            EpochSnapshot {
                index_commitment: self.index_commitment.clone(),
                value_commitment: self.value_commitment.clone(),
                rand_index_poly: self.rand_index_poly.clone(),
                rand_index_commitment: self.rand_index_commitment.clone(),
                rand_index_state: self.rand_index_state.clone(),
                rand_value_poly: self.rand_value_poly.clone(),
                rand_value_commitment: self.rand_value_commitment.clone(),
                rand_value_state: self.rand_value_state.clone(),
            },
        );
        // If the operator wired Redis in, also publish an occupancy
        // bit per prefilled slot. The coordinator's open-addressing
        // checks `aegon:slot:{shard_id}:{slot}` via EXISTS; without
        // these writes it would believe the prefilled slots are empty
        // and silently overwrite real entries on the next publish.
        // Value is a marker — the production schema would store the
        // owning label bytes, but prefilled rows have no real label
        // and the bench never exercises the recovery path that reads
        // them. Chunked so a 2^23-per-shard prefill doesn't try to
        // pipeline 8M ops through one MULTI/EXEC.
        if let DbSource::Redis(url) = db_source {
            use super::db::{key_slot, Db, DbOp};
            let db = RedisDb::connect(url)?;
            const CHUNK: usize = 50_000;
            let slots: Vec<usize> = filled_slots.into_iter().collect();
            for chunk in slots.chunks(CHUNK) {
                let ops: Vec<DbOp> = chunk
                    .iter()
                    .map(|&slot| DbOp::Set {
                        key: key_slot(shard_id, slot),
                        value: b"prefilled".to_vec(),
                    })
                    .collect();
                db.write_atomic(&ops)?;
            }
        }
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
        // Redis to persist them to. Sharded callers go through
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

        // Snapshot prev epoch state — the invariance proof is over
        // (prev → next), so we capture before mutating.
        let prev_index_poly = self.index_poly.clone();
        let prev_value_poly = self.value_poly.clone();
        let prev_index_com = self.index_commitment.clone();
        let prev_index_state = self.index_state.clone();
        let prev_value_com = self.value_commitment.clone();
        let prev_value_state = self.value_state.clone();
        let prev_rand_index_poly = self.rand_index_poly.clone();
        let prev_rand_index_com = self.rand_index_commitment.clone();
        let prev_rand_index_state = self.rand_index_state.clone();
        let prev_rand_value_poly = self.rand_value_poly.clone();
        let prev_rand_value_com = self.rand_value_commitment.clone();
        let prev_rand_value_state = self.rand_value_state.clone();

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
        let mut delta_index_poly: SparseMultilinearExtension<E::ScalarField> =
            SparseMultilinearExtension::from_evaluations(self.index_poly.num_vars, &[]);
        let mut delta_value_poly: SparseMultilinearExtension<E::ScalarField> =
            SparseMultilinearExtension::from_evaluations(self.value_poly.num_vars, &[]);
        let mut new_label_slots: Vec<Vec<bool>> = Vec::new();
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

        // Commit + aux the delta polynomials (size `batch`). Cost is
        // `O(batch)` for `P::commit` and `O(k · batch)` for the
        // per-row aux fill inside `commit_with_aux`.
        let (delta_index_com, delta_index_state) =
            commit_with_aux::<E, P>(&self.prover_param, &delta_index_poly)?;
        let (delta_value_com, delta_value_state) =
            commit_with_aux::<E, P>(&self.prover_param, &delta_value_poly)?;

        // Homomorphism on commitments and on the prover state:
        //   new_com   = prev_com   + delta_com
        //   new_state = prev_state + delta_state    (sparse-walk FMA)
        // Both ops cost `O(|support(delta)|)` group ops, independent
        // of how big the prior epoch's support is. See
        // `KZHKState::iadd_scaled` for the per-row primitive.
        #[cfg(feature = "tracing_instrument")]
        let _combine_span = tracing::debug_span!("Aegon::Phase1::CombineHomomorphic").entered();
        let new_index_com = prev_index_com.clone() + delta_index_com.clone();
        let new_value_com = prev_value_com.clone() + delta_value_com.clone();
        let mut new_index_state = prev_index_state.clone();
        P::fma_state(
            &self.prover_param,
            &mut new_index_state,
            E::ScalarField::one(),
            &delta_index_state,
        )?;
        let mut new_value_state = prev_value_state.clone();
        P::fma_state(
            &self.prover_param,
            &mut new_value_state,
            E::ScalarField::one(),
            &delta_value_state,
        )?;
        #[cfg(feature = "tracing_instrument")]
        drop(_combine_span);

        self.pending = Some(PendingPublish {
            prev_index_poly,
            prev_value_poly,
            prev_index_com,
            prev_index_state,
            prev_value_com,
            prev_value_state,
            prev_rand_index_poly,
            prev_rand_index_com,
            prev_rand_index_state,
            prev_rand_value_poly,
            prev_rand_value_com,
            prev_rand_value_state,
            new_index_com: new_index_com.clone(),
            new_index_state,
            new_value_com: new_value_com.clone(),
            new_value_state,
            delta_index_com,
            delta_value_com,
            delta_index_state,
            delta_value_state,
            new_label_slots,
        });

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
        let pending = self.pending.take().ok_or_else(|| {
            AegonError::Config(
                "publish_phase_2 called without a pending publish; call publish_phase_1 first"
                    .into(),
            )
        })?;
        let PendingPublish {
            prev_index_poly,
            prev_value_poly,
            prev_index_com,
            prev_index_state,
            prev_value_com,
            prev_value_state,
            prev_rand_index_poly,
            prev_rand_index_com,
            prev_rand_index_state,
            prev_rand_value_poly,
            prev_rand_value_com,
            prev_rand_value_state,
            new_index_com,
            new_index_state,
            new_value_com,
            new_value_state,
            delta_index_com,
            delta_value_com,
            delta_index_state,
            delta_value_state,
            new_label_slots,
        } = pending;

        // §6.4 step 1: open `rand_index` and `rand_value` at each
        // new-label slot **before** the rand-polys are mutated. The
        // openings bind against the prior-epoch rand commitments
        // (`prev_rand_*_com`); evaluations are zero by construction
        // (a brand-new slot has had no chain delta applied to it
        // through any prior epoch), but the PCS proof is still required
        // for the future history-check verifier.
        #[cfg(feature = "tracing_instrument")]
        let _pre_openings_span = tracing::debug_span!(
            "Aegon::Phase2::PreUpdateOpenings",
            new_slots = new_label_slots.len()
        )
        .entered();
        let mut pre_rand_index: Vec<(E::ScalarField, P::Proof)> =
            Vec::with_capacity(new_label_slots.len());
        let mut pre_rand_value: Vec<(E::ScalarField, P::Proof)> =
            Vec::with_capacity(new_label_slots.len());
        for slot_bits in &new_label_slots {
            pre_rand_index.push(open_at_point::<E, P>(
                &self.prover_param,
                &prev_rand_index_poly,
                &prev_rand_index_com,
                &prev_rand_index_state,
                slot_bits,
                &self.dims,
                b"aegon.rand_index.open",
            )?);
            pre_rand_value.push(open_at_point::<E, P>(
                &self.prover_param,
                &prev_rand_value_poly,
                &prev_rand_value_com,
                &prev_rand_value_state,
                slot_bits,
                &self.dims,
                b"aegon.rand_value.open",
            )?);
        }
        #[cfg(feature = "tracing_instrument")]
        drop(_pre_openings_span);

        // Update rand polynomials: `rand_{n+1} = rand_n + r_n · ∆`.
        // The polynomial update is still done pointwise on the sparse
        // evaluation tables (cheap, `O(|support(∆)|)`), because future
        // openings of `rand_*_poly` need it to be consistent with the
        // commitment we publish below.
        update_rand(
            &mut self.rand_index_poly,
            &prev_index_poly,
            &self.index_poly,
            new_r_index,
        );
        update_rand(
            &mut self.rand_value_poly,
            &prev_value_poly,
            &self.value_poly,
            new_r_value,
        );

        // No new MSM for the rand commitments. Phase 1 already
        // committed + aux'd the data-side delta (`delta_index_com`,
        // `delta_index_state` and friends). Since rand_{n+1} − rand_n =
        // r_X · ∆_X holds pointwise on the polynomial, the same
        // homomorphism holds on commitments and on the prover state:
        //   new_rand_X_com   = prev_rand_X_com   + r_X · delta_X_com
        //   new_rand_X_state = prev_rand_X_state + r_X · delta_X_state
        // The `+ r_X ·` part is one scalar-mul on the commitment group
        // element and `O(k · batch)` group ops on the state (only
        // touched cells get updated — see `KZHKState::iadd_scaled`).
        #[cfg(feature = "tracing_instrument")]
        let _combine_rand_span =
            tracing::debug_span!("Aegon::Phase2::CombineRandHomomorphic").entered();
        let new_rand_index_com = prev_rand_index_com.clone()
            + delta_index_com.clone() * new_r_index;
        let new_rand_value_com = prev_rand_value_com.clone()
            + delta_value_com.clone() * new_r_value;
        let mut new_rand_index_state = prev_rand_index_state.clone();
        P::fma_state(
            &self.prover_param,
            &mut new_rand_index_state,
            new_r_index,
            &delta_index_state,
        )?;
        let mut new_rand_value_state = prev_rand_value_state.clone();
        P::fma_state(
            &self.prover_param,
            &mut new_rand_value_state,
            new_r_value,
            &delta_value_state,
        )?;
        #[cfg(feature = "tracing_instrument")]
        drop(_combine_rand_span);

        // §6.4 step 2: open `rand_index` and `rand_value` (now at the
        // new epoch) and `value` (also at the new epoch — the
        // value-poly was committed at the end of phase 1) at every
        // new-label slot. Together with the pre-openings above, this
        // pins both endpoints of the update equation
        // `rand_X_new(s) − rand_X_old(s) = r_X · (data_X_new(s) − 0)`
        // at every slot the verifier needs to check.
        #[cfg(feature = "tracing_instrument")]
        let _post_openings_span = tracing::debug_span!(
            "Aegon::Phase2::PostUpdateOpenings",
            new_slots = new_label_slots.len()
        )
        .entered();
        let mut post_rand_index: Vec<(E::ScalarField, P::Proof)> =
            Vec::with_capacity(new_label_slots.len());
        let mut post_rand_value: Vec<(E::ScalarField, P::Proof)> =
            Vec::with_capacity(new_label_slots.len());
        let mut post_value: Vec<(E::ScalarField, P::Proof)> =
            Vec::with_capacity(new_label_slots.len());
        for slot_bits in &new_label_slots {
            post_rand_index.push(open_at_point::<E, P>(
                &self.prover_param,
                &self.rand_index_poly,
                &new_rand_index_com,
                &new_rand_index_state,
                slot_bits,
                &self.dims,
                b"aegon.rand_index.open",
            )?);
            post_rand_value.push(open_at_point::<E, P>(
                &self.prover_param,
                &self.rand_value_poly,
                &new_rand_value_com,
                &new_rand_value_state,
                slot_bits,
                &self.dims,
                b"aegon.rand_value.open",
            )?);
            post_value.push(open_at_point::<E, P>(
                &self.prover_param,
                &self.value_poly,
                &new_value_com,
                &new_value_state,
                slot_bits,
                &self.dims,
                b"aegon.value.open",
            )?);
        }
        #[cfg(feature = "tracing_instrument")]
        drop(_post_openings_span);

        // The prev_*_poly / prev_*_state / prev_*_com bindings the
        // destructure pulled out of `PendingPublish` were threaded into
        // the old `build_invariance_proof` helper. With both that
        // helper and the empty `InvarianceProof` marker gone, suppress
        // the unused-binding warnings explicitly so the destructure
        // pattern stays one place.
        let _ = (
            &prev_index_poly,
            &prev_index_state,
            &prev_value_poly,
            &prev_value_state,
            &prev_index_com,
            &prev_value_com,
        );

        // Assemble the §6.4 history witness bundle. Per-slot evals +
        // proofs come from the two passes above, addressed by the
        // same `new_label_slots` ordering.
        let history = HistoryOpenings {
            entries: new_label_slots
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
                .collect(),
        };

        // Commit the new epoch to live state and history.
        #[cfg(feature = "tracing_instrument")]
        let _finalize_span = tracing::debug_span!("Aegon::Phase2::FinalizeEpoch").entered();
        self.index_commitment = new_index_com.clone();
        self.index_state = new_index_state;
        self.value_commitment = new_value_com.clone();
        self.value_state = new_value_state;
        self.rand_index_commitment = new_rand_index_com.clone();
        self.rand_index_state = new_rand_index_state.clone();
        self.rand_value_commitment = new_rand_value_com.clone();
        self.rand_value_state = new_rand_value_state.clone();
        self.r_index = new_r_index;
        self.r_value = new_r_value;
        self.epoch += 1;

        self.epoch_history.insert(
            self.epoch,
            EpochSnapshot {
                index_commitment: new_index_com,
                value_commitment: new_value_com,
                rand_index_poly: self.rand_index_poly.clone(),
                rand_index_commitment: new_rand_index_com,
                rand_index_state: new_rand_index_state,
                rand_value_poly: self.rand_value_poly.clone(),
                rand_value_commitment: new_rand_value_com,
                rand_value_state: new_rand_value_state,
            },
        );

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

    /// Open the current `index_poly` commitment at `slot_bits`.
    pub fn open_index_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        open_at_point::<E, P>(
            &self.prover_param,
            &self.index_poly,
            &self.index_commitment,
            &self.index_state,
            slot_bits,
            &self.dims,
            b"aegon.index.open",
        )
    }

    /// Open the current `value_poly` commitment at `slot_bits`.
    pub fn open_value_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        open_at_point::<E, P>(
            &self.prover_param,
            &self.value_poly,
            &self.value_commitment,
            &self.value_state,
            slot_bits,
            &self.dims,
            b"aegon.value.open",
        )
    }

    /// Open `rand_index_poly` at `slot_bits` for some retained `epoch`.
    /// Used by sharded consistency proofs to open both `s0` and the
    /// current epoch at the same probe point.
    pub fn open_rand_index_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let snap = self
            .epoch_history
            .get(&epoch)
            .ok_or(AegonError::InvalidEpoch(epoch))?;
        open_at_point::<E, P>(
            &self.prover_param,
            &snap.rand_index_poly,
            &snap.rand_index_commitment,
            &snap.rand_index_state,
            slot_bits,
            &self.dims,
            b"aegon.rand_index.open",
        )
    }

    /// Open `rand_value_poly` at `slot_bits` for some retained `epoch`.
    pub fn open_rand_value_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let snap = self
            .epoch_history
            .get(&epoch)
            .ok_or(AegonError::InvalidEpoch(epoch))?;
        open_at_point::<E, P>(
            &self.prover_param,
            &snap.rand_value_poly,
            &snap.rand_value_commitment,
            &snap.rand_value_state,
            slot_bits,
            &self.dims,
            b"aegon.rand_value.open",
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

        let mut probes = Vec::with_capacity(*ctr0 as usize + 1);
        for ctr in 0..=*ctr0 {
            let probe_bits = H::h_bits(ctr, label, self.log_capacity);
            let probe_point = bool_index_to_point::<E::ScalarField>(&probe_bits);
            let usize_idx = bool_index_to_usize(&probe_bits, &self.dims);
            let evaluation = self
                .index_poly
                .evaluations
                .get(&usize_idx)
                .copied()
                .unwrap_or_else(<E::ScalarField as Zero>::zero);
            let mut tr = IOPTranscript::<E::ScalarField>::new(b"aegon.index.open");
            let (proof, _) = P::open(
                &self.prover_param,
                &self.index_commitment,
                &DenseOrSparseMLE::Sparse(self.index_poly.clone()),
                &probe_point,
                &self.index_state,
                &mut tr,
            )?;
            probes.push((evaluation, proof));
        }

        let value_point = bool_index_to_point::<E::ScalarField>(bool_index);
        let usize_idx = bool_index_to_usize(bool_index, &self.dims);
        let value_evaluation = self
            .value_poly
            .evaluations
            .get(&usize_idx)
            .copied()
            .unwrap_or_else(<E::ScalarField as Zero>::zero);
        let mut tr = IOPTranscript::<E::ScalarField>::new(b"aegon.value.open");
        let (value_proof, _) = P::open(
            &self.prover_param,
            &self.value_commitment,
            &DenseOrSparseMLE::Sparse(self.value_poly.clone()),
            &value_point,
            &self.value_state,
            &mut tr,
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

            let pair = open_pair::<E, P>(
                &self.prover_param,
                &point,
                &snap_s0.rand_index_poly,
                &snap_s0.rand_index_commitment,
                &snap_s0.rand_index_state,
                &self.rand_index_poly,
                &self.rand_index_commitment,
                &self.rand_index_state,
                b"aegon.rand_index.open",
            )?;
            index_witnesses.push(pair);
        }

        // Value half: open rand_value at the user's slot.
        let value_point = bool_index_to_point::<E::ScalarField>(bool_index);
        let value_witness = open_pair::<E, P>(
            &self.prover_param,
            &value_point,
            &snap_s0.rand_value_poly,
            &snap_s0.rand_value_commitment,
            &snap_s0.rand_value_state,
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
        &DenseOrSparseMLE::Sparse(poly.clone()),
        &point,
        state,
        &mut tr,
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
        &DenseOrSparseMLE::Sparse(poly_s0.clone()),
        point,
        state_s0,
        &mut tr0,
    )?;
    let mut tr1 = IOPTranscript::<E::ScalarField>::new(transcript_label);
    let (proof_s1, eval_s1) = P::open(
        pp,
        com_s1,
        &DenseOrSparseMLE::Sparse(poly_s1.clone()),
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

// `build_invariance_proof` + `build_chain_witness` used to live here.
// Both became dead code when the auditor switched to a commitment-
// homomorphism check: the prover no longer opens the four polynomials
// at a Fiat-Shamir random point, because every group element the
// auditor needs is already in the published `EpochCommitment`. See
// `audit::verify_invariance` for the verifier side.

/// Try to load a previously-persisted [`AegonCheckpoint`] for the
/// given shard out of Redis. Returns `Ok(None)` for a fresh DB (no
/// `aegon:shard:{shard_id}:state` key) or `DbSource::None`. Returns
/// `Err` if the key is present but malformed.
///
/// Used by the `aegon_shard_server` binary on startup: if a checkpoint
/// exists, the binary restores the shard's state from it instead of
/// initializing fresh.
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
    match db_source {
        DbSource::None => Ok(None),
        DbSource::Redis(url) => {
            let db = RedisDb::connect(url)?;
            match db.get(&key_shard_state(shard_id))? {
                Some(bytes) => {
                    let ckpt = AegonCheckpoint::<E, P>::deserialize_compressed(&bytes[..])
                        .map_err(|e| {
                            AegonError::Database(format!(
                                "deserialize shard {shard_id} checkpoint: {e}"
                            ))
                        })?;
                    Ok(Some(ckpt))
                },
                None => Ok(None),
            }
        },
    }
}
