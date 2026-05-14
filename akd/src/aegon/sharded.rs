//! Sharded Aegon: a coordinator that owns N independent Aegon
//! shard instances, each storing a polynomial of size `2^(log_capacity -
//! log_n_shards)`. The coordinator handles
//!
//!   * cross-shard open addressing — `H(ctr, label)` yields a
//!     `(shard_id, slot)` pair, and the probe trail walks across
//!     shards until it lands on an empty slot;
//!   * a single Fiat-Shamir chain derived from *all* shards' new data
//!     commitments per epoch; and
//!   * a Merkle root over the per-shard `EpochCommitment`s that becomes
//!     the externally-visible "epoch commitment".
//!
//! Setting `n_shards = 1` collapses to the single-poly behaviour with a
//! degenerate one-leaf Merkle tree and a trail that always lives in
//! shard 0 — useful as a no-op compatibility mode.
//!
//! The coordinator is in-process for now. Wrapping each shard's
//! method calls in gRPC doesn't change the protocol: this file is the
//! source of truth for the coordinator-side logic, regardless of where
//! the shards live.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use ark_ec::pairing::Pairing;
use ark_ff::{Field, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use akd_core::aegon_crypto::pcs::PCSGlobalParam;
use akd_core::aegon_crypto::transcript::IOPTranscript;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use super::audit::verify_chain;
use super::config::{AegonConfig, VerifierContext};
use super::db::{
    key_coord_state, key_epoch_commit, key_history_openings, key_labels_set, key_routing,
    key_slot, key_value, Db, DbOp, DbSource, RedisDb,
};
use super::error::AegonError;
use super::hash::{bool_index_to_point, HashSuite, Sha256Hash};
use super::server::Aegon;
use super::types::{
    AegonPcs, AuditState, EpochCommitment, HistoryOpenings, InvarianceProof, Label, RandPair, Value,
};

/// Where the shards live, and how the coordinator talks to them.
///
/// `InProcess` is today's behaviour: each shard is an in-memory
/// `Aegon` owned by the `ShardedAegon`. `Remote` is the gRPC-backed
/// deployment story — the coordinator holds client handles to
/// shards running on `endpoints[i]`. `Remote` is parsed and
/// validated by [`ShardedAegonConfigBuilder::build`] but is not yet
/// honoured at setup time; calling `ShardedAegon::setup` with a
/// `Remote` transport returns an `unimplemented!()`-style error
/// until the gRPC layer lands.
#[derive(Clone, Debug)]
pub enum ShardTransport {
    /// Single-process: shards co-located in the same address space.
    InProcess,
    /// Each shard is a remote `tonic`/gRPC service. `endpoints[i]`
    /// is the address of shard `i` (e.g. `"http://10.0.0.7:50051"`
    /// or `"https://aegon-shard-7.svc.cluster.local:50051"`). Length
    /// must equal `1 << log_n_shards`.
    Remote { endpoints: Vec<String> },
}

impl Default for ShardTransport {
    fn default() -> Self {
        Self::InProcess
    }
}

/// Where the SRS / (prover_param, verifier_param) come from.
///
/// `DangerouslyGenerate` calls `P::gen_srs_for_testing` and is **not**
/// suitable for production — every Directory instance gets a fresh
/// (un-ceremonial) SRS. `Path` reads a previously-serialized SRS
/// from disk (the natural output of a trusted-setup ceremony).
/// `Path` is not yet wired up; using it returns
/// `AegonError::Config(...)` at setup time.
#[derive(Clone, Debug)]
pub enum SrsSource {
    /// Generate a fresh SRS from `P::gen_srs_for_testing`. Test-only.
    DangerouslyGenerate,
    /// Load `(prover_param, verifier_param)` from a previously-
    /// serialized file (canonical SRS, typically a trusted-setup
    /// ceremony output). Not yet implemented.
    Path(std::path::PathBuf),
}

impl Default for SrsSource {
    fn default() -> Self {
        Self::DangerouslyGenerate
    }
}

/// Configuration for a [`ShardedAegon`] deployment.
///
/// Construct via [`ShardedAegonConfig::builder`]. The fields are
/// `pub` so downstream code can inspect them, but the only correct
/// way to build a config is the builder, which enforces the
/// invariants (`log_n_shards` ≤ `shard_log_capacity` + headroom,
/// `endpoints.len() == 1 << log_n_shards` for remote transport, etc.).
#[derive(Clone)]
pub struct ShardedAegonConfig<E: Pairing, P: AegonPcs<E>> {
    /// Log capacity of *each shard*'s polynomial. Total dictionary
    /// capacity is `2^(shard_log_capacity + log_n_shards)`.
    pub shard_log_capacity: usize,
    /// `n_shards = 2^log_n_shards`.
    pub log_n_shards: usize,
    /// Privacy flag — propagates to the PCS's `zk` parameter.
    pub private: bool,
    /// PCS-specific configuration handed to every shard's Aegon.
    pub pcs_config: P::Config,
    /// Where the shards live (in-process / remote endpoints).
    pub shards: ShardTransport,
    /// Where the SRS comes from (test-mode generation, on-disk file).
    pub srs: SrsSource,
    /// Coordinator-side label→value store. Defaults to
    /// [`DbSource::None`] (no DB; lookup returns an empty value).
    pub db: DbSource,
    pub _e: PhantomData<E>,
}

impl<E: Pairing, P: AegonPcs<E>> ShardedAegonConfig<E, P> {
    /// Start a fluent builder. Required fields are
    /// `shard_log_capacity`, `log_n_shards`, and either `pcs_config`
    /// (generic) or — for the KZH-k backend — `kzh_k`.
    pub fn builder() -> ShardedAegonConfigBuilder<E, P> {
        ShardedAegonConfigBuilder::new()
    }

    pub fn n_shards(&self) -> usize {
        1usize << self.log_n_shards
    }

    /// Total log capacity across all shards.
    pub fn log_capacity(&self) -> usize {
        self.shard_log_capacity + self.log_n_shards
    }

    /// One-shot SRS generation + serialization. Run this once on a
    /// setup machine; the resulting file is the input to
    /// [`SrsSource::Path`] on every shard.
    ///
    /// The file layout is `prover_param || verifier_param`, both
    /// canonically-serialized via arkworks `CanonicalSerialize`. The
    /// SRS is sized for *one shard* (`self.shard_log_capacity`),
    /// since every shard's polynomial is identically-shaped.
    ///
    /// **Security note.** `gen_srs_for_testing` is, as the name
    /// implies, not a trusted setup. For real production, replace
    /// the body of this helper with a loader that pulls a ceremony
    /// output rather than generating one. The on-wire shape is the
    /// same either way, which is the point.
    pub fn generate_srs_to_file<R: Rng>(
        &self,
        rng: &mut R,
        path: &std::path::Path,
    ) -> Result<(), AegonError>
    where
        P::ProverParam: CanonicalSerialize,
        P::VerifierParam: CanonicalSerialize,
    {
        let srs = P::gen_srs_for_testing(
            self.pcs_config.clone(),
            rng,
            self.shard_log_capacity,
        )?;
        let (pk, vk) = P::trim(&srs, None, Some(self.shard_log_capacity))?;

        let mut file = std::fs::File::create(path).map_err(|e| {
            AegonError::Config(format!(
                "create srs file '{}': {e}",
                path.display()
            ))
        })?;
        pk.serialize_compressed(&mut file).map_err(|e| {
            AegonError::Config(format!("serialize prover_param: {e}"))
        })?;
        vk.serialize_compressed(&mut file).map_err(|e| {
            AegonError::Config(format!("serialize verifier_param: {e}"))
        })?;
        Ok(())
    }
}

/// Inverse of [`ShardedAegonConfig::generate_srs_to_file`]. Reads
/// the prover_param + verifier_param tuple that was written by
/// `generate_srs_to_file`. Public so the `aegon_shard_server`
/// binary can load its SRS at boot.
pub fn read_srs_from_file<E, P>(
    path: &std::path::Path,
) -> Result<(P::ProverParam, P::VerifierParam), AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    P::ProverParam: CanonicalDeserialize,
    P::VerifierParam: CanonicalDeserialize,
{
    let mut file = std::fs::File::open(path).map_err(|e| {
        AegonError::Config(format!(
            "open srs file '{}': {e}",
            path.display()
        ))
    })?;
    let pk = P::ProverParam::deserialize_compressed(&mut file).map_err(|e| {
        AegonError::Config(format!("deserialize prover_param: {e}"))
    })?;
    let vk = P::VerifierParam::deserialize_compressed(&mut file).map_err(|e| {
        AegonError::Config(format!("deserialize verifier_param: {e}"))
    })?;
    Ok((pk, vk))
}

/// Fluent builder for [`ShardedAegonConfig`].
///
/// All required fields are checked at [`Self::build`] time and
/// reported with a clear error message rather than a panic. The
/// builder also validates:
///   * `shard_log_capacity ≥ 1`,
///   * for `ShardTransport::Remote`, `endpoints.len() ==
///     1 << log_n_shards`,
///   * `SrsSource::Path` is not yet implemented (returns an error).
pub struct ShardedAegonConfigBuilder<E: Pairing, P: AegonPcs<E>> {
    shard_log_capacity: Option<usize>,
    log_n_shards: Option<usize>,
    private: bool,
    pcs_config: Option<P::Config>,
    shards: ShardTransport,
    srs: SrsSource,
    db: DbSource,
    _e: PhantomData<E>,
}

impl<E: Pairing, P: AegonPcs<E>> Default for ShardedAegonConfigBuilder<E, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Pairing, P: AegonPcs<E>> ShardedAegonConfigBuilder<E, P> {
    pub fn new() -> Self {
        Self {
            shard_log_capacity: None,
            log_n_shards: None,
            private: false,
            pcs_config: None,
            shards: ShardTransport::default(),
            srs: SrsSource::default(),
            db: DbSource::default(),
            _e: PhantomData,
        }
    }

    /// log-capacity of each shard's polynomial. **Required.**
    pub fn shard_log_capacity(mut self, v: usize) -> Self {
        self.shard_log_capacity = Some(v);
        self
    }

    /// `n_shards = 2^log_n_shards`. **Required.**
    pub fn log_n_shards(mut self, v: usize) -> Self {
        self.log_n_shards = Some(v);
        self
    }

    /// Whether to run in privacy-preserving mode (`zk = true` at the
    /// PCS layer). Defaults to `false`.
    ///
    /// For the KZH-k backend, call [`Self::private`] *before*
    /// [`ShardedAegonConfigBuilder::kzh_k`] so the `KZHKConfig.zk`
    /// flag agrees with this choice; otherwise `setup` will reject
    /// the mismatch.
    pub fn private(mut self, v: bool) -> Self {
        self.private = v;
        self
    }

    /// Provide the PCS-specific config directly. Mutually exclusive
    /// with backend-specific convenience methods like
    /// [`ShardedAegonConfigBuilder::kzh_k`].
    pub fn pcs_config(mut self, v: P::Config) -> Self {
        self.pcs_config = Some(v);
        self
    }

    /// Transport choice (in-process vs remote gRPC). Defaults to
    /// `ShardTransport::InProcess`.
    pub fn shards(mut self, v: ShardTransport) -> Self {
        self.shards = v;
        self
    }

    /// SRS source. Defaults to `SrsSource::DangerouslyGenerate`
    /// (suitable for tests; **never** production).
    pub fn srs(mut self, v: SrsSource) -> Self {
        self.srs = v;
        self
    }

    /// Coordinator-side label→value KV store. Defaults to
    /// [`DbSource::None`] — set to [`DbSource::Redis`] in cluster
    /// deployments so `lookup` can return the raw value bytes.
    pub fn db(mut self, v: DbSource) -> Self {
        self.db = v;
        self
    }

    pub fn build(self) -> Result<ShardedAegonConfig<E, P>, AegonError> {
        let shard_log_capacity = self.shard_log_capacity.ok_or_else(|| {
            AegonError::Config("ShardedAegonConfig: shard_log_capacity is required".into())
        })?;
        let log_n_shards = self.log_n_shards.ok_or_else(|| {
            AegonError::Config("ShardedAegonConfig: log_n_shards is required".into())
        })?;
        let pcs_config = self.pcs_config.ok_or_else(|| {
            AegonError::Config(
                "ShardedAegonConfig: pcs_config is required (for the KZH-k backend, call .kzh_k(k) instead)".into(),
            )
        })?;
        if shard_log_capacity == 0 {
            return Err(AegonError::Config(
                "shard_log_capacity must be at least 1".into(),
            ));
        }
        if let ShardTransport::Remote { endpoints } = &self.shards {
            let expected = 1usize << log_n_shards;
            if endpoints.len() != expected {
                return Err(AegonError::Config(format!(
                    "ShardTransport::Remote endpoints.len() = {} does not match 2^log_n_shards = {}",
                    endpoints.len(),
                    expected,
                )));
            }
        }
        Ok(ShardedAegonConfig {
            shard_log_capacity,
            log_n_shards,
            private: self.private,
            pcs_config,
            shards: self.shards,
            srs: self.srs,
            db: self.db,
            _e: PhantomData,
        })
    }
}

// KZH-k-specific convenience on the builder.
impl<E: Pairing> ShardedAegonConfigBuilder<E, akd_core::aegon_crypto::pcs::kzhk::KZHK<E>> {
    /// Set the KZH-k `k` parameter; the `zk` flag of `KZHKConfig` is
    /// taken from `self.private` at the time of this call. Call
    /// [`Self::private`] *before* this method.
    pub fn kzh_k(mut self, k: usize) -> Self {
        use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
        self.pcs_config = Some(KZHKConfig::new(k, self.private));
        self
    }
}

/// 32-byte SHA-256 digest. Used for the Merkle tree over per-shard
/// commitments and for the externally-visible "epoch hash".
pub type EpochDigest = [u8; 32];

/// One write into a shard's per-publish batch. The
/// [`ShardedAegon::plan_phase_1_batches`] coordinator decides where
/// every `(label, value)` update goes and turns it into one of these
/// per shard; the shard then applies the batch in
/// [`super::server::Aegon::publish_phase_1_at_slots`].
///
/// Field order is the wire-format order. CanonicalSerialize emits the
/// fields in declaration order, so the on-wire bytes are identical to
/// the legacy `(Vec<bool>, F, F)` tuple — older audit logs round-trip
/// unchanged.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardWrite<F: Field> {
    /// Slot inside *this* shard's polynomial — length equals
    /// `shard_log_capacity`. The coordinator produces it from
    /// `H_bits(ctr, label)`'s low bits and the shard reinterprets it
    /// as the boolean coordinate of a point on the multilinear
    /// hypercube. Identifies which evaluation cell in `index_poly` /
    /// `value_poly` this write targets.
    pub slot_bits: Vec<bool>,
    /// What to put at this slot in the **index** polynomial.
    ///
    /// - `H_F(label)` when the label is brand-new in this epoch — the
    ///   shard writes this hash as the slot's identity, which is what
    ///   the lookup-side open-addressing trail checks against.
    /// - **Zero** when the label already exists (the slot's identity
    ///   was set in a prior epoch and must not change). Zero is the
    ///   sentinel `Aegon::publish_phase_1_at_slots` interprets as
    ///   "skip the identity write"; safe because `H_F(label)` is a
    ///   hash output and never zero for any real label.
    pub h_label_or_zero: F,
    /// What to put at this slot in the **value** polynomial. Always
    /// `H_F(value)` — the value polynomial is overwritten every
    /// epoch the label appears, so there's no analogous "skip" mode.
    pub h_value: F,
}

/// One shard's slice of a publish batch: the list of writes a single
/// `ShardHandle::publish_phase_1_at_slots` call consumes.
/// [`ShardedAegon::plan_phase_1_batches`] returns a `Vec<SubBatch<F>>`
/// indexed by shard id.
pub type SubBatch<F> = Vec<ShardWrite<F>>;

/// Record of one **newly-placed** label in a publish — i.e., a label
/// that was not in `self.routing` before this call. Returned by
/// `plan_phase_1_batches` alongside the per-shard sub-batches so the
/// post-publish Redis durability barrier can write fresh `aegon:slot:*`
/// and `aegon:routing:*` keys without scanning the routing table.
///
/// (Existing labels — being value-updated — don't need fresh routing /
/// slot writes since those keys were committed in a prior epoch.)
#[derive(Clone, Debug)]
pub(crate) struct NewPlacement {
    pub(crate) label: Label,
    pub(crate) shard_id: u32,
    pub(crate) slot_idx: usize,
}

/// Coordinator state recovered from Redis on restart. Built by
/// `ShardedAegon::try_recover_from_db` and consumed in `setup`.
struct RecoveredState<E: Pairing, P: AegonPcs<E>> {
    epoch: u64,
    r_index: E::ScalarField,
    r_value: E::ScalarField,
    epoch_commits: Vec<ShardedEpochCommitment<E, P>>,
    routing: HashMap<Label, LabelRouting>,
}

/// External epoch commitment exposed by [`ShardedAegon::current_commitment`].
/// `merkle_root` is the root over `per_shard`; `per_shard[i]` is the
/// `EpochCommitment` produced by shard `i` for this epoch.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedEpochCommitment<E: Pairing, P: AegonPcs<E>> {
    pub epoch: u64,
    pub merkle_root: EpochDigest,
    pub per_shard: Vec<EpochCommitment<E, P>>,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for ShardedEpochCommitment<E, P> {
    fn clone(&self) -> Self {
        Self {
            epoch: self.epoch,
            merkle_root: self.merkle_root,
            per_shard: self.per_shard.clone(),
        }
    }
}

/// One probe along a sharded open-addressing trail. The verifier
/// recomputes `(shard_id, slot_bits)` from `H(ctr, label)`, then checks
/// that `merkle_path` reconstructs the epoch root from `leaf`, and that
/// the PCS `proof` verifies `evaluation` against `leaf.index_commitment`
/// at the slot.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedProbe<E: Pairing, P: AegonPcs<E>> {
    pub shard_id: u32,
    pub leaf: EpochCommitment<E, P>,
    pub merkle_path: Vec<EpochDigest>,
    pub evaluation: E::ScalarField,
    pub proof: P::Proof,
}

/// Lookup proof emitted by [`ShardedAegon::lookup`]. Carries one
/// `ShardedProbe` per `ctr` in `0..=ctr0` (cross-shard trail) plus the
/// value opening at the final probe's slot.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedLookupProof<E: Pairing, P: AegonPcs<E>> {
    /// Number of *additional* probes beyond ctr=0; trail length is `ctr0 + 1`.
    pub ctr0: u64,
    pub probes: Vec<ShardedProbe<E, P>>,
    /// Value opening at `probes[ctr0]`'s `(shard_id, slot)`. The shard
    /// and merkle anchor are reused from the final probe (saves a copy).
    pub value_evaluation: E::ScalarField,
    pub value_proof: P::Proof,
}

/// One probe's worth of consistency evidence: openings of `rand_index`
/// at the same `(shard_id, slot)` at both `s0` and the current epoch.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedRandPair<E: Pairing, P: AegonPcs<E>> {
    pub shard_id: u32,
    pub leaf_s0: EpochCommitment<E, P>,
    pub leaf_s1: EpochCommitment<E, P>,
    pub merkle_path_s0: Vec<EpochDigest>,
    pub merkle_path_s1: Vec<EpochDigest>,
    pub inner: RandPair<E, P>,
}

/// Consistency proof emitted by [`ShardedAegon::consistency_proof`].
/// Mirrors `aegon::ConsistencyProof` but every opening is anchored under
/// the per-epoch Merkle root via the bundled paths.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedConsistencyProof<E: Pairing, P: AegonPcs<E>> {
    pub ctr0: u64,
    pub index_witnesses: Vec<ShardedRandPair<E, P>>,
    pub value_witness: ShardedRandPair<E, P>,
}

/// Per-epoch invariance proof. The auditor verifies the Merkle root
/// over the new shard commits, derives the shared FS scalars, and
/// checks each shard's inner invariance witness.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedInvarianceProof<E: Pairing, P: AegonPcs<E>> {
    pub per_shard: Vec<InvarianceProof<E, P>>,
}

/// Verifier-side bundle, including the deployment's `log_n_shards`
/// (clients need it to recompute the probe trail).
#[derive(Clone)]
pub struct ShardedVerifierContext<E: Pairing, P: AegonPcs<E>> {
    /// `inner.log_capacity` is the *per-shard* log_capacity.
    pub inner: VerifierContext<E, P>,
    pub log_n_shards: usize,
}

impl<E: Pairing, P: AegonPcs<E>> ShardedVerifierContext<E, P> {
    pub fn new(inner: VerifierContext<E, P>, log_n_shards: usize) -> Self {
        Self {
            inner,
            log_n_shards,
        }
    }

    pub fn shard_log_capacity(&self) -> usize {
        self.inner.log_capacity
    }

    pub fn total_log_capacity(&self) -> usize {
        self.inner.log_capacity + self.log_n_shards
    }
}

// ---------- ShardedAegon -----------------------------------------------

/// Routing entry retained by the coordinator. The trail records every
/// probe along the open-addressing chain (one entry per `ctr` from 0 up
/// to and including `ctr0`); the last entry is the label's permanent
/// home.
///
/// Derives `CanonicalSerialize` so the coordinator can persist its
/// routing table to Redis and rebuild on restart.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub(crate) struct LabelRouting {
    pub(crate) trail: Vec<(u32, Vec<bool>)>,
}

impl LabelRouting {
    fn ctr0(&self) -> u64 {
        (self.trail.len() - 1) as u64
    }
    fn final_assignment(&self) -> &(u32, Vec<bool>) {
        self.trail.last().expect("trail non-empty by construction")
    }
}

/// Coordinator that owns `n_shards` independent [`Aegon`] instances.
pub struct ShardedAegon<E, P, H = Sha256Hash>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    shards: Vec<Box<dyn super::shard_grpc::ShardHandle<E, P, H>>>,
    /// PCS block dims; cached at setup so the coordinator can encode
    /// boolean slot vectors to canonical PCS indices without going
    /// through a per-call method on the ShardHandle. All shards
    /// share the same dims (they share one SRS), so we keep it once
    /// at this level.
    shard_dims: Vec<usize>,
    /// Per-shard verifier context. Cached at setup time for the same
    /// reason as `shard_dims`.
    shard_verifier_context: VerifierContext<E, P>,
    /// log capacity per shard.
    shard_log_capacity_cached: usize,

    log_n_shards: usize,

    epoch: u64,

    // Coordinator-side FS chain. Each new epoch's r is derived by
    // hashing prev_r with ALL the shards' new data commits.
    r_index: E::ScalarField,
    r_value: E::ScalarField,

    // Coordinator's view of past epochs.
    epoch_commits: Vec<ShardedEpochCommitment<E, P>>,

    // Routing table: label → full cross-shard probe trail. The trail is
    // deterministic from the label given the public log_n_shards and
    // shard log_capacity; the coordinator caches it for fast lookup and
    // value-update batches.
    routing: HashMap<Label, LabelRouting>,

    // Coordinator-side label→value store. `None` when the config asked
    // for `DbSource::None`: `lookup` then returns an empty value and
    // the caller is expected to know the value out-of-band.
    db: Option<Box<dyn Db>>,
}

impl<E, P, H> ShardedAegon<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam: PCSGlobalParam + CanonicalDeserialize + Send + Sync + 'static,
    P::VerifierParam: PCSGlobalParam + CanonicalDeserialize + Send + Sync + 'static,
    P::Commitment: Clone + Send + Sync + 'static,
    P::Proof: Clone + Send + Sync + 'static,
    P::State: Send + Sync + 'static,
    P::Polynomial: Send + Sync,
    P::Point: Send + Sync,
    P::Evaluation: Send + Sync,
    H: HashSuite<E::ScalarField> + Send + Sync + 'static,
{
    /// Build a fresh ShardedAegon: generates one SRS, trims it once,
    /// and hands cloned `(prover_param, verifier_param)` to every
    /// shard's [`Aegon::init`]. All shards share the same param set,
    /// which is sound because each one commits to a distinct
    /// polynomial — and avoids 32× redundant SRS generation, which
    /// dominates setup cost at production scale.
    pub fn setup<R: Rng>(
        rng: &mut R,
        config: &ShardedAegonConfig<E, P>,
    ) -> Result<Self, AegonError>
    where
        P::VerifierParam: Clone,
        // Bounds for boxing Aegon as a ShardHandle (in-process).
        Aegon<E, P, H>: super::shard_grpc::ShardHandle<E, P, H>,
    {
        let n_shards = config.n_shards();
        let shard_config: AegonConfig<E, P> = AegonConfig {
            log_capacity: config.shard_log_capacity,
            private: config.private,
            pcs_config: config.pcs_config.clone(),
            _e: PhantomData,
        };

        let (shards, shard_dims, shard_verifier_context): (
            Vec<Box<dyn super::shard_grpc::ShardHandle<E, P, H>>>,
            Vec<usize>,
            VerifierContext<E, P>,
        ) = match &config.shards {
            ShardTransport::InProcess => {
                // SRS: DangerouslyGenerate calls gen_srs_for_testing
                // (test mode). Path reads the SRS file written by
                // ShardedAegonConfig::generate_srs_to_file — production
                // flow where one setup machine generates the ceremony
                // output once and distributes it.
                let (prover_param, verifier_param) = match &config.srs {
                    SrsSource::DangerouslyGenerate => {
                        let srs = P::gen_srs_for_testing(
                            shard_config.pcs_config.clone(),
                            rng,
                            shard_config.log_capacity,
                        )?;
                        P::trim(&srs, None, Some(shard_config.log_capacity))?
                    },
                    SrsSource::Path(path) => read_srs_from_file::<E, P>(path)?,
                };
                let dims = P::block_dims(&prover_param, shard_config.log_capacity);
                let vctx = VerifierContext::new(
                    shard_config.log_capacity,
                    verifier_param.clone(),
                );
                let mut shards: Vec<Box<dyn super::shard_grpc::ShardHandle<E, P, H>>> =
                    Vec::with_capacity(n_shards);
                for _ in 0..n_shards {
                    let aegon = Aegon::<E, P, H>::init(
                        prover_param.clone(),
                        verifier_param.clone(),
                        &shard_config,
                    )?;
                    shards.push(Box::new(aegon));
                }
                (shards, dims, vctx)
            },
            ShardTransport::Remote { endpoints } => {
                // Remote shards already loaded their own SRS at boot
                // (via the shard-server binary). The coordinator just
                // connects. We *also* need a local verifier_param so
                // we can build a VerifierContext — that's loaded
                // from the same SrsSource here, since verifier_param
                // is small and shared. The big prover_param stays on
                // the shard machines.
                let (_prover_param_unused, verifier_param) = match &config.srs {
                    SrsSource::DangerouslyGenerate => {
                        let srs = P::gen_srs_for_testing(
                            shard_config.pcs_config.clone(),
                            rng,
                            shard_config.log_capacity,
                        )?;
                        P::trim(&srs, None, Some(shard_config.log_capacity))?
                    },
                    SrsSource::Path(path) => read_srs_from_file::<E, P>(path)?,
                };
                let dims = P::block_dims(&_prover_param_unused, shard_config.log_capacity);
                let vctx = VerifierContext::new(
                    shard_config.log_capacity,
                    verifier_param.clone(),
                );
                let mut shards: Vec<Box<dyn super::shard_grpc::ShardHandle<E, P, H>>> =
                    Vec::with_capacity(n_shards);
                for ep in endpoints {
                    let client = super::shard_grpc::GrpcShardClient::<E, P>::connect(
                        ep.clone(),
                        vctx.clone(),
                        shard_config.log_capacity,
                    )?;
                    shards.push(Box::new(client));
                }
                (shards, dims, vctx)
            },
        };

        let initial_per_shard: Vec<EpochCommitment<E, P>> =
            shards.iter().map(|s| s.current_commitment()).collect();
        let initial_root = merkle_root(&initial_per_shard);
        let initial_commit = ShardedEpochCommitment {
            epoch: 0,
            merkle_root: initial_root,
            per_shard: initial_per_shard,
        };

        // Coordinator-side KV store. Connect eagerly so a misconfigured
        // URL fails at setup, not on the first publish.
        let db: Option<Box<dyn Db>> = match &config.db {
            DbSource::None => None,
            DbSource::Redis(url) => Some(Box::new(RedisDb::connect(url)?)),
        };

        // Recovery path: if the connected DB already has coordinator
        // state, this is a restart — replay the state into the live
        // struct instead of starting fresh at epoch 0. Fresh DB or
        // `DbSource::None` falls through to the empty-genesis path.
        let recovered = match &db {
            Some(db_inner) => Self::try_recover_from_db(&**db_inner)?,
            None => None,
        };

        if let Some(rec) = recovered {
            Ok(Self {
                shards,
                shard_dims,
                shard_verifier_context,
                shard_log_capacity_cached: shard_config.log_capacity,
                log_n_shards: config.log_n_shards,
                epoch: rec.epoch,
                r_index: rec.r_index,
                r_value: rec.r_value,
                epoch_commits: rec.epoch_commits,
                routing: rec.routing,
                db,
            })
        } else {
            Ok(Self {
                shards,
                shard_dims,
                shard_verifier_context,
                shard_log_capacity_cached: shard_config.log_capacity,
                log_n_shards: config.log_n_shards,
                epoch: 0,
                r_index: E::ScalarField::zero(),
                r_value: E::ScalarField::zero(),
                epoch_commits: vec![initial_commit],
                routing: HashMap::new(),
                db,
            })
        }
    }

    /// Read the coordinator's durable state from Redis if any was
    /// previously persisted. Returns `None` on a fresh DB (no
    /// `aegon:coord:state` key), `Some(RecoveredState)` on a restart,
    /// `Err` if a key exists but a downstream get/deserialize fails
    /// — that means Redis is half-written or corrupt and the caller
    /// should refuse to come up, not silently start over.
    fn try_recover_from_db(
        db: &dyn Db,
    ) -> Result<Option<RecoveredState<E, P>>, AegonError> {
        let Some(state_bytes) = db.get(key_coord_state())? else {
            return Ok(None);
        };

        // 1. Parse coord:state → (epoch, r_index, r_value)
        let mut cursor = &state_bytes[..];
        let epoch: u64 = u64::deserialize_compressed(&mut cursor)
            .map_err(|e| AegonError::Database(format!("deserialize epoch: {e}")))?;
        let r_index: E::ScalarField =
            E::ScalarField::deserialize_compressed(&mut cursor)
                .map_err(|e| AegonError::Database(format!("deserialize r_index: {e}")))?;
        let r_value: E::ScalarField =
            E::ScalarField::deserialize_compressed(&mut cursor)
                .map_err(|e| AegonError::Database(format!("deserialize r_value: {e}")))?;

        // 2. Load every published epoch commitment in order.
        let mut epoch_commits: Vec<ShardedEpochCommitment<E, P>> =
            Vec::with_capacity(epoch as usize + 1);
        for e in 0..=epoch {
            let key = key_epoch_commit(e);
            let bytes = db.get(&key)?.ok_or_else(|| {
                AegonError::Database(format!(
                    "epoch commitment for epoch {e} missing from Redis"
                ))
            })?;
            let commit = ShardedEpochCommitment::<E, P>::deserialize_compressed(&bytes[..])
                .map_err(|err| {
                    AegonError::Database(format!(
                        "deserialize ShardedEpochCommitment for epoch {e}: {err}"
                    ))
                })?;
            epoch_commits.push(commit);
        }

        // 3. Rebuild the routing table: SMEMBERS aegon:labels, then
        //    GET aegon:routing:{label} for each.
        let labels = db.smembers(key_labels_set())?;
        let mut routing: HashMap<Label, LabelRouting> = HashMap::with_capacity(labels.len());
        for label in labels {
            let bytes = db.get(&key_routing(&label))?.ok_or_else(|| {
                AegonError::Database(format!(
                    "routing entry missing for label {label:?} (present in aegon:labels)"
                ))
            })?;
            let lr = LabelRouting::deserialize_compressed(&bytes[..]).map_err(|e| {
                AegonError::Database(format!(
                    "deserialize routing for {label:?}: {e}"
                ))
            })?;
            routing.insert(label, lr);
        }

        Ok(Some(RecoveredState {
            epoch,
            r_index,
            r_value,
            epoch_commits,
            routing,
        }))
    }

    pub fn n_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn log_n_shards(&self) -> usize {
        self.log_n_shards
    }

    pub fn shard_log_capacity(&self) -> usize {
        self.shard_log_capacity_cached
    }

    pub fn log_capacity(&self) -> usize {
        self.shard_log_capacity_cached + self.log_n_shards
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Verifier context (per-shard) — the sharded verifier wraps this
    /// with `log_n_shards` via [`ShardedVerifierContext`].
    pub fn verifier_context(&self) -> VerifierContext<E, P>
    where
        P::VerifierParam: Clone,
    {
        self.shard_verifier_context.clone()
    }

    /// Convenience: full sharded verifier context.
    pub fn sharded_verifier_context(&self) -> ShardedVerifierContext<E, P> {
        ShardedVerifierContext::new(self.verifier_context(), self.log_n_shards)
    }

    pub fn current_commitment(&self) -> ShardedEpochCommitment<E, P> {
        self.epoch_commits
            .last()
            .expect("epoch 0 always retained")
            .clone()
    }

    pub fn epoch_commitment(&self, epoch: u64) -> Option<ShardedEpochCommitment<E, P>> {
        self.epoch_commits.get(epoch as usize).cloned()
    }

    /// Apply a batch of updates and produce a new epoch.
    ///
    /// Six steps, each delegated to a private helper:
    ///
    ///   1. [`reject_duplicate_labels`](Self::reject_duplicate_labels) —
    ///      no label may appear twice in the batch.
    ///   2. [`plan_phase_1_batches`](Self::plan_phase_1_batches) —
    ///      cross-shard open-addressing assigns a `(shard_id, slot)`
    ///      to every new label; existing labels reuse their cached
    ///      trail. Produces one write list per shard.
    ///   3. [`run_phase_1`](Self::run_phase_1) — every shard applies
    ///      its slice in parallel, returning new `(index, value)`
    ///      commitments.
    ///   4. [`derive_chain_scalars`](Self::derive_chain_scalars) — a
    ///      single `(r_index, r_value)` pair is derived from prev-r +
    ///      *all* shards' new commits, binding the FS challenge to
    ///      every shard at once.
    ///   5. [`run_phase_2`](Self::run_phase_2) — every shard updates
    ///      its rand polynomials with the shared scalars and returns
    ///      its `EpochCommitment`.
    ///   6. [`finalize_epoch`](Self::finalize_epoch) + [`mirror_to_db`](Self::mirror_to_db)
    ///      — coordinator advances `(r_index, r_value, epoch)`, builds
    ///      the Merkle root, and mirrors raw `(label, value)` bytes
    ///      into Redis for the lookup path.
    pub fn publish(
        &mut self,
        updates: &[(Label, Value)],
    ) -> Result<(ShardedEpochCommitment<E, P>, ShardedInvarianceProof<E, P>), AegonError> {
        // There must not be any duplicate labels in the batch
        Self::reject_duplicate_labels(updates)?;

        let (sub_batches, new_placements) = self.plan_phase_1_batches(updates)?;
        let (new_index_commits, new_value_commits) = self.run_phase_1(&sub_batches)?;
        let (new_r_index, new_r_value) =
            self.derive_chain_scalars(&new_index_commits, &new_value_commits);
        let (per_shard_commits, per_shard_invariance, per_shard_history) =
            self.run_phase_2(new_r_index, new_r_value)?;
        let sharded_commit = self.finalize_epoch(per_shard_commits, new_r_index, new_r_value);
        self.persist_publish_to_db(
            updates,
            &new_placements,
            &sharded_commit,
            &per_shard_history,
        )?;
        Ok((
            sharded_commit,
            ShardedInvarianceProof {
                per_shard: per_shard_invariance,
            },
        ))
    }

    /// O(n) scan for `label` appearing twice. Errors out the whole
    /// batch on the first collision so phase 1 never sees a malformed
    /// input.
    fn reject_duplicate_labels(updates: &[(Label, Value)]) -> Result<(), AegonError> {
        let mut seen: HashSet<&[u8]> = HashSet::with_capacity(updates.len());
        for (label, _) in updates {
            if !seen.insert(label.as_slice()) {
                return Err(AegonError::DuplicateLabel(label.clone()));
            }
        }
        Ok(())
    }

    /// Decide where every `(label, value)` write lands. Returns one
    /// sub-batch per shard, each entry shaped
    /// `(slot_bits, h_label_or_zero, h_value)`:
    ///
    /// - **New label**: open-addressing finds the first empty
    ///   `(shard_id, slot)`; the trail is cached in `self.routing`,
    ///   and the sub-batch entry carries `h_label = H_F(label)`.
    /// - **Existing label**: reuse the cached trail and emit a value-
    ///   only update — `h_label = 0` is the sentinel meaning "don't
    ///   change the slot's identity field".
    ///
    /// `in_batch_claimed` per-shard sets ensure two new labels in the
    /// same publish can't collide on the same empty slot.
    fn plan_phase_1_batches(
        &mut self,
        updates: &[(Label, Value)],
    ) -> Result<(Vec<SubBatch<E::ScalarField>>, Vec<NewPlacement>), AegonError> {
        let n = self.shards.len();
        let mut sub_batches: Vec<SubBatch<E::ScalarField>> =
            (0..n).map(|_| Vec::new()).collect();
        let mut in_batch_claimed: Vec<HashSet<usize>> = (0..n).map(|_| HashSet::new()).collect();
        let mut new_placements: Vec<NewPlacement> = Vec::new();

        for (label, value) in updates {
            let h_value = H::h_f(value);
            if let Some(routing) = self.routing.get(label) {
                let (sid, slot_bits) = routing.final_assignment().clone();
                sub_batches[sid as usize].push(ShardWrite {
                    slot_bits,
                    h_label_or_zero: E::ScalarField::zero(),
                    h_value,
                });
            } else {
                let h_label = H::h_f(label);
                let trail = self.assign_trail(label, &mut in_batch_claimed)?;
                let (sid, slot_bits) = trail.final_assignment().clone();
                let slot_idx = bool_index_to_usize_dims(&slot_bits, &self.shard_dims);
                sub_batches[sid as usize].push(ShardWrite {
                    slot_bits,
                    h_label_or_zero: h_label,
                    h_value,
                });
                new_placements.push(NewPlacement {
                    label: label.clone(),
                    shard_id: sid,
                    slot_idx,
                });
                self.routing.insert(label.clone(), trail);
            }
        }
        Ok((sub_batches, new_placements))
    }

    /// Drive every shard's `publish_phase_1_at_slots` in parallel and
    /// transpose the per-shard `(index_com, value_com)` pairs into
    /// two shard-id-ordered vectors. Shards are independent (separate
    /// polynomials + state), so rayon's data-parallel pattern is
    /// safe; only the read-only prover_param is shared.
    fn run_phase_1(
        &mut self,
        sub_batches: &[SubBatch<E::ScalarField>],
    ) -> Result<(Vec<P::Commitment>, Vec<P::Commitment>), AegonError> {
        let phase_1: Vec<(P::Commitment, P::Commitment)> = self
            .shards
            .par_iter_mut()
            .zip(sub_batches.par_iter())
            .map(|(shard, batch)| shard.publish_phase_1_at_slots(batch))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(phase_1.into_iter().unzip())
    }

    /// Derive the shared `(r_index, r_value)` Fiat-Shamir scalars for
    /// this epoch transition. Each scalar is `O(prev_r, every shard's
    /// new data commitment)` — binding to all N commits up front is
    /// what stops a malicious server from re-tuning any one shard's
    /// commit after observing the chain randomness.
    fn derive_chain_scalars(
        &self,
        new_index_commits: &[P::Commitment],
        new_value_commits: &[P::Commitment],
    ) -> (E::ScalarField, E::ScalarField) {
        let new_r_index = fs_chain_scalar::<E::ScalarField, P::Commitment>(
            b"aegon.sharded.fs.r_index",
            self.r_index,
            new_index_commits,
        );
        let new_r_value = fs_chain_scalar::<E::ScalarField, P::Commitment>(
            b"aegon.sharded.fs.r_value",
            self.r_value,
            new_value_commits,
        );
        (new_r_index, new_r_value)
    }

    /// Drive every shard's `publish_phase_2` with the shared scalars
    /// in parallel and split the per-shard `(EpochCommitment,
    /// InvarianceProof, HistoryOpenings)` triples into shard-id-ordered
    /// vectors. After the commitment-homomorphism audit refactor the
    /// `InvarianceProof` is empty, so the second vec is essentially
    /// `Vec<()>` — the auditor recovers what it needs from the
    /// per-shard commitments. The third vec carries the §6.4 history
    /// witnesses (one `HistoryOpenings` per shard, possibly empty).
    fn run_phase_2(
        &mut self,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
    ) -> Result<
        (
            Vec<EpochCommitment<E, P>>,
            Vec<InvarianceProof<E, P>>,
            Vec<HistoryOpenings<E, P>>,
        ),
        AegonError,
    > {
        let phase_2: Vec<(
            EpochCommitment<E, P>,
            InvarianceProof<E, P>,
            HistoryOpenings<E, P>,
        )> = self
            .shards
            .par_iter_mut()
            .map(|shard| shard.publish_phase_2(new_r_index, new_r_value))
            .collect::<Result<Vec<_>, _>>()?;
        let mut commits = Vec::with_capacity(phase_2.len());
        let mut invariances = Vec::with_capacity(phase_2.len());
        let mut histories = Vec::with_capacity(phase_2.len());
        for (c, i, h) in phase_2 {
            commits.push(c);
            invariances.push(i);
            histories.push(h);
        }
        Ok((commits, invariances, histories))
    }

    /// Coordinator-side bookkeeping: advance `(r_index, r_value,
    /// epoch)`, build the Merkle root over the per-shard commits, and
    /// append the new `ShardedEpochCommitment` to the history. Returns
    /// the commitment the caller will publish externally.
    fn finalize_epoch(
        &mut self,
        per_shard_commits: Vec<EpochCommitment<E, P>>,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
    ) -> ShardedEpochCommitment<E, P> {
        self.r_index = new_r_index;
        self.r_value = new_r_value;
        self.epoch += 1;
        let merkle_root = merkle_root(&per_shard_commits);
        let sharded_commit = ShardedEpochCommitment {
            epoch: self.epoch,
            merkle_root,
            per_shard: per_shard_commits,
        };
        self.epoch_commits.push(sharded_commit.clone());
        sharded_commit
    }

    /// Durability barrier for one publish: write every key the
    /// coordinator (and a future restarted coordinator) needs to
    /// reconstruct its view, atomically via Redis MULTI/EXEC.
    ///
    /// One `aegon:value:{label}` and one `SADD aegon:labels {label}`
    /// per update (value-update or new). For brand-new labels, also
    /// one `aegon:routing:{label}` and one `aegon:slot:{shard}:{slot}`.
    /// Finally, two coordinator-global keys: `aegon:coord:state` (epoch
    /// + FS scalars) and `aegon:coord:epoch_commit:{epoch}` (the new
    /// sharded epoch commitment).
    ///
    /// The polynomial commitment binds `H_F(value)` at the right slot
    /// already — Redis is only the side-channel that lets `lookup`
    /// return the value alongside the proof, and the durability layer
    /// for crash recovery; the verifier still re-hashes everything it
    /// receives. No-op when no DB was configured (`DbSource::None`).
    fn persist_publish_to_db(
        &self,
        updates: &[(Label, Value)],
        new_placements: &[NewPlacement],
        sharded_commit: &ShardedEpochCommitment<E, P>,
        per_shard_history: &[HistoryOpenings<E, P>],
    ) -> Result<(), AegonError> {
        let Some(db) = &self.db else { return Ok(()) };

        // Pre-size: 2 ops per update (value SET + labels SADD) + 2 ops
        // per new placement (routing SET + slot SET) + 2 global ops
        // (coord:state SET + coord:epoch_commit:{epoch} SET) + 1 op per
        // non-empty shard's §6.4 history bundle.
        let non_empty_histories = per_shard_history.iter().filter(|h| !h.entries.is_empty()).count();
        let mut ops: Vec<DbOp> = Vec::with_capacity(
            updates.len() * 2 + new_placements.len() * 2 + 2 + non_empty_histories,
        );

        // 1. value:{label} + labels SADD for every update.
        for (label, value) in updates {
            ops.push(DbOp::Set {
                key: key_value(label),
                value: value.clone(),
            });
            ops.push(DbOp::SAdd {
                key: key_labels_set().to_vec(),
                member: label.clone(),
            });
        }

        // 2. routing:{label} + slot:{shard}:{slot} for new placements.
        for placement in new_placements {
            let routing = self.routing.get(&placement.label).ok_or_else(|| {
                AegonError::Database(format!(
                    "internal: routing missing for newly-placed label {:?}",
                    placement.label
                ))
            })?;
            let mut routing_bytes = Vec::new();
            routing
                .serialize_compressed(&mut routing_bytes)
                .map_err(|e| AegonError::Database(format!("serialize routing: {e}")))?;
            ops.push(DbOp::Set {
                key: key_routing(&placement.label),
                value: routing_bytes,
            });
            ops.push(DbOp::Set {
                key: key_slot(placement.shard_id, placement.slot_idx),
                value: placement.label.clone(),
            });
        }

        // 3. coord:state — one key, contains (epoch, r_index, r_value).
        let mut state_bytes = Vec::new();
        self.epoch
            .serialize_compressed(&mut state_bytes)
            .map_err(|e| AegonError::Database(format!("serialize epoch: {e}")))?;
        self.r_index
            .serialize_compressed(&mut state_bytes)
            .map_err(|e| AegonError::Database(format!("serialize r_index: {e}")))?;
        self.r_value
            .serialize_compressed(&mut state_bytes)
            .map_err(|e| AegonError::Database(format!("serialize r_value: {e}")))?;
        ops.push(DbOp::Set {
            key: key_coord_state().to_vec(),
            value: state_bytes,
        });

        // 4. coord:epoch_commit:{epoch} — the externally-published commitment.
        let mut commit_bytes = Vec::new();
        sharded_commit
            .serialize_compressed(&mut commit_bytes)
            .map_err(|e| AegonError::Database(format!("serialize epoch commit: {e}")))?;
        ops.push(DbOp::Set {
            key: key_epoch_commit(sharded_commit.epoch),
            value: commit_bytes,
        });

        // 5. openings:{epoch}:{shard_id} — §6.4 history witnesses. One
        // key per shard whose batch carried at least one brand-new
        // label. Shards that did only value-updates produced an empty
        // `HistoryOpenings`; persisting an empty bundle would just
        // waste a Redis SET, so those are skipped here.
        for (shard_id, history) in per_shard_history.iter().enumerate() {
            if history.entries.is_empty() {
                continue;
            }
            let mut history_bytes = Vec::new();
            history
                .serialize_compressed(&mut history_bytes)
                .map_err(|e| AegonError::Database(format!("serialize history openings: {e}")))?;
            ops.push(DbOp::Set {
                key: key_history_openings(sharded_commit.epoch, shard_id as u32),
                value: history_bytes,
            });
        }

        db.write_atomic(&ops)
    }

    /// Walk the cross-shard probe trail for a brand-new `label`, marking
    /// the first empty `(shard_id, slot)` as claimed in `in_batch_claimed`
    /// and returning the full trail (including the chosen final probe).
    /// `in_batch_claimed[s]` is the set of `usize`-encoded slot indices
    /// already claimed by earlier entries in *this* publish batch.
    fn assign_trail(
        &self,
        label: &[u8],
        in_batch_claimed: &mut [HashSet<usize>],
    ) -> Result<LabelRouting, AegonError> {
        let total_capacity = 1u64 << self.log_capacity();
        let mut trail: Vec<(u32, Vec<bool>)> = Vec::new();
        for ctr in 0..total_capacity {
            let (shard_id, slot_bits) =
                probe_at::<H, E::ScalarField>(ctr, label, self.log_n_shards, self.shard_log_capacity());
            trail.push((shard_id, slot_bits.clone()));

            let slot_idx = bool_index_to_usize_dims(&slot_bits, &self.shard_dims);
            // Occupancy check: when a DB is configured, ask Redis (one
            // EXISTS — no gRPC). Otherwise fall back to the shard
            // (in-process test path). Redis is authoritative once it's
            // configured because `persist_publish_to_db` writes
            // `aegon:slot:*` in the same atomic txn as the shard
            // commitments are finalized, so the two never disagree
            // unless we're mid-recovery.
            let occupied_prev = if let Some(db) = &self.db {
                db.exists(&key_slot(shard_id, slot_idx))?
            } else {
                self.shards[shard_id as usize].is_index_slot_occupied(&slot_bits)
            };
            let occupied_in_batch = in_batch_claimed[shard_id as usize].contains(&slot_idx);
            if !occupied_prev && !occupied_in_batch {
                in_batch_claimed[shard_id as usize].insert(slot_idx);
                return Ok(LabelRouting { trail });
            }
        }
        Err(AegonError::DictionaryFull {
            capacity: total_capacity as usize,
        })
    }

    /// Produce a sharded lookup proof for `label`, plus the raw value
    /// bytes from the coordinator's KV store (if one is configured).
    ///
    /// When the config is built with [`DbSource::None`] (the in-process
    /// tests' default), the returned value vector is empty and the
    /// caller must hand the verifier the value it already knows
    /// out-of-band. With [`DbSource::Redis`], the coordinator fetches
    /// the value from Redis here and returns it alongside the proof —
    /// the verifier still re-hashes it, the DB is just a retrieval
    /// side-channel.
    pub fn lookup(
        &self,
        label: &Label,
    ) -> Result<(Value, ShardedLookupProof<E, P>), AegonError> {
        let routing = self
            .routing
            .get(label)
            .ok_or_else(|| AegonError::UnknownLabel(label.clone()))?;
        let current = self.current_commitment();

        let mut probes: Vec<ShardedProbe<E, P>> = Vec::with_capacity(routing.trail.len());
        for (shard_id, slot_bits) in &routing.trail {
            let (evaluation, proof) = self.shards[*shard_id as usize].open_index_at_slot(slot_bits)?;
            let leaf = current.per_shard[*shard_id as usize].clone();
            let merkle_path = build_merkle_path(&current.per_shard, *shard_id as usize);
            probes.push(ShardedProbe {
                shard_id: *shard_id,
                leaf,
                merkle_path,
                evaluation,
                proof,
            });
        }

        let (final_shard, final_slot) = routing.final_assignment();
        let (value_evaluation, value_proof) =
            self.shards[*final_shard as usize].open_value_at_slot(final_slot)?;

        let value: Value = match &self.db {
            Some(db) => db.get(&key_value(label))?.ok_or_else(|| {
                AegonError::Database(format!(
                    "label {label:?} routed but missing from KV store"
                ))
            })?,
            None => Vec::new(),
        };

        Ok((
            value,
            ShardedLookupProof {
                ctr0: routing.ctr0(),
                probes,
                value_evaluation,
                value_proof,
            },
        ))
    }

    /// Produce a consistency proof for `label` from epoch `s0` to the
    /// current epoch. Opens `rand_index` at every probe in the label's
    /// trail (at both `s0` and now), plus `rand_value` at the final
    /// probe (at both epochs).
    pub fn consistency_proof(
        &self,
        label: &Label,
        s0: u64,
    ) -> Result<ShardedConsistencyProof<E, P>, AegonError> {
        let routing = self
            .routing
            .get(label)
            .ok_or_else(|| AegonError::UnknownLabel(label.clone()))?;
        let s0_commit = self
            .epoch_commitment(s0)
            .ok_or(AegonError::InvalidEpoch(s0))?;
        let s1_commit = self.current_commitment();
        let s1 = self.epoch;

        let mut index_witnesses: Vec<ShardedRandPair<E, P>> = Vec::with_capacity(routing.trail.len());
        for (shard_id, slot_bits) in &routing.trail {
            let (eval_s0, proof_s0) =
                self.shards[*shard_id as usize].open_rand_index_at_slot_in_epoch(slot_bits, s0)?;
            let (eval_s1, proof_s1) =
                self.shards[*shard_id as usize].open_rand_index_at_slot_in_epoch(slot_bits, s1)?;
            index_witnesses.push(ShardedRandPair {
                shard_id: *shard_id,
                leaf_s0: s0_commit.per_shard[*shard_id as usize].clone(),
                leaf_s1: s1_commit.per_shard[*shard_id as usize].clone(),
                merkle_path_s0: build_merkle_path(&s0_commit.per_shard, *shard_id as usize),
                merkle_path_s1: build_merkle_path(&s1_commit.per_shard, *shard_id as usize),
                inner: RandPair {
                    eval_s0,
                    proof_s0,
                    eval_s1,
                    proof_s1,
                },
            });
        }

        let (final_shard, final_slot) = routing.final_assignment();
        let (eval_s0, proof_s0) = self.shards[*final_shard as usize]
            .open_rand_value_at_slot_in_epoch(final_slot, s0)?;
        let (eval_s1, proof_s1) = self.shards[*final_shard as usize]
            .open_rand_value_at_slot_in_epoch(final_slot, s1)?;
        let value_witness = ShardedRandPair {
            shard_id: *final_shard,
            leaf_s0: s0_commit.per_shard[*final_shard as usize].clone(),
            leaf_s1: s1_commit.per_shard[*final_shard as usize].clone(),
            merkle_path_s0: build_merkle_path(&s0_commit.per_shard, *final_shard as usize),
            merkle_path_s1: build_merkle_path(&s1_commit.per_shard, *final_shard as usize),
            inner: RandPair {
                eval_s0,
                proof_s0,
                eval_s1,
                proof_s1,
            },
        };

        Ok(ShardedConsistencyProof {
            ctr0: routing.ctr0(),
            index_witnesses,
            value_witness,
        })
    }
}

// ---------- verifiers --------------------------------------------------

/// Verify a sharded lookup proof. Re-derives the probe trail from
/// `label` via `H_bits`, anchors every probe's `leaf` under the
/// announced `merkle_root`, verifies each PCS opening, and checks the
/// open-addressing invariants from the paper:
///
///   * `evaluation != 0` and `evaluation != H_f(label)` for every
///     intermediate probe (the slot is occupied by some other label);
///   * `evaluation == H_f(label)` at the final probe;
///   * the value opening verifies and equals `H_f(value)`.
pub fn verify_sharded_lookup<E, P, H>(
    ctx: &ShardedVerifierContext<E, P>,
    commit: &ShardedEpochCommitment<E, P>,
    label: &Label,
    value: &Value,
    proof: &ShardedLookupProof<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    let expected = proof.ctr0 as usize + 1;
    if proof.probes.len() != expected {
        return Err(AegonError::Verification(
            "sharded probe vector length does not match ctr0",
        ));
    }

    let h_label = H::h_f(label);

    for (ctr_us, probe) in proof.probes.iter().enumerate() {
        let ctr = ctr_us as u64;
        let (expected_shard, slot_bits) =
            probe_at::<H, E::ScalarField>(ctr, label, ctx.log_n_shards, ctx.shard_log_capacity());
        if expected_shard != probe.shard_id {
            return Err(AegonError::Verification(
                "probe shard_id does not match H(ctr, label)",
            ));
        }
        // Merkle anchor.
        let reconstructed = verify_merkle_path::<E, P>(
            &probe.leaf,
            probe.shard_id as usize,
            &probe.merkle_path,
        );
        if reconstructed != commit.merkle_root {
            return Err(AegonError::Verification(
                "probe merkle path does not reconstruct epoch root",
            ));
        }
        // PCS opening against the shard's index commitment.
        let point = bool_index_to_point::<E::ScalarField>(&slot_bits);
        let mut tr = IOPTranscript::<E::ScalarField>::new(b"aegon.index.open");
        let ok = P::verify(
            &ctx.inner.verifier_param,
            &probe.leaf.index_commitment,
            &point,
            &probe.evaluation,
            &probe.proof,
            &mut tr,
        )?;
        if !ok {
            return Ok(false);
        }
        // Open-addressing constraints.
        if ctr < proof.ctr0 {
            if probe.evaluation.is_zero() {
                return Err(AegonError::Verification(
                    "earlier probe slot is empty: server picked a non-canonical index",
                ));
            }
            if probe.evaluation == h_label {
                return Err(AegonError::Verification(
                    "earlier probe slot holds H_F(label): label was already assigned at a smaller counter",
                ));
            }
        } else if probe.evaluation != h_label {
            return Err(AegonError::Verification(
                "final probe slot does not hold H_F(label)",
            ));
        }
    }

    // Value opening anchored at the same shard/leaf as the final probe.
    let final_probe = proof.probes.last().expect("ctr0 + 1 >= 1 probes");
    let (_, final_slot) = probe_at::<H, E::ScalarField>(
        proof.ctr0,
        label,
        ctx.log_n_shards,
        ctx.shard_log_capacity(),
    );
    let value_point = bool_index_to_point::<E::ScalarField>(&final_slot);
    let mut tr = IOPTranscript::<E::ScalarField>::new(b"aegon.value.open");
    let ok = P::verify(
        &ctx.inner.verifier_param,
        &final_probe.leaf.value_commitment,
        &value_point,
        &proof.value_evaluation,
        &proof.value_proof,
        &mut tr,
    )?;
    if !ok {
        return Ok(false);
    }
    let expected_value = H::h_f(value);
    if proof.value_evaluation != expected_value {
        return Err(AegonError::Verification(
            "value opening does not match H_F(value)",
        ));
    }

    Ok(true)
}

/// Verify a sharded consistency proof. Recomputes the trail from
/// `label`, anchors every per-probe leaf under both epoch roots,
/// verifies every PCS opening, and checks that the `s0` and `s1`
/// evaluations agree pointwise (which is what guarantees the slot
/// didn't change between epochs — see paper §5.2 / §6.1).
///
/// `expected_ctr0` should be the `ctr0` the caller learned from a
/// fresh lookup against the *current* epoch. Pinning it on the user
/// side prevents a server from substituting a shorter or longer
/// trail (which would otherwise verify but for a different label's
/// effective slot).
pub fn verify_sharded_consistency<E, P, H>(
    ctx: &ShardedVerifierContext<E, P>,
    s0_commit: &ShardedEpochCommitment<E, P>,
    s1_commit: &ShardedEpochCommitment<E, P>,
    label: &Label,
    expected_ctr0: u64,
    proof: &ShardedConsistencyProof<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    if proof.ctr0 != expected_ctr0 {
        return Err(AegonError::Verification(
            "consistency proof ctr0 does not match the caller's expected ctr0",
        ));
    }
    let expected = proof.ctr0 as usize + 1;
    if proof.index_witnesses.len() != expected {
        return Err(AegonError::Verification(
            "consistency index_witnesses length does not match ctr0",
        ));
    }

    for (ctr_us, w) in proof.index_witnesses.iter().enumerate() {
        let ctr = ctr_us as u64;
        let (expected_shard, slot_bits) =
            probe_at::<H, E::ScalarField>(ctr, label, ctx.log_n_shards, ctx.shard_log_capacity());
        if expected_shard != w.shard_id {
            return Err(AegonError::Verification(
                "consistency probe shard_id does not match H(ctr, label)",
            ));
        }
        if verify_merkle_path::<E, P>(&w.leaf_s0, w.shard_id as usize, &w.merkle_path_s0)
            != s0_commit.merkle_root
        {
            return Err(AegonError::Verification(
                "consistency s0 merkle path does not reconstruct s0 root",
            ));
        }
        if verify_merkle_path::<E, P>(&w.leaf_s1, w.shard_id as usize, &w.merkle_path_s1)
            != s1_commit.merkle_root
        {
            return Err(AegonError::Verification(
                "consistency s1 merkle path does not reconstruct s1 root",
            ));
        }
        let point = bool_index_to_point::<E::ScalarField>(&slot_bits);
        let mut tr0 = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_index.open");
        let ok0 = P::verify(
            &ctx.inner.verifier_param,
            &w.leaf_s0.rand_index_commitment,
            &point,
            &w.inner.eval_s0,
            &w.inner.proof_s0,
            &mut tr0,
        )?;
        let mut tr1 = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_index.open");
        let ok1 = P::verify(
            &ctx.inner.verifier_param,
            &w.leaf_s1.rand_index_commitment,
            &point,
            &w.inner.eval_s1,
            &w.inner.proof_s1,
            &mut tr1,
        )?;
        if !(ok0 && ok1) {
            return Ok(false);
        }
        if w.inner.eval_s0 != w.inner.eval_s1 {
            // Slot's rand_index diverged → the user's open-addressing
            // path changed across (s0, s1]. Caller treats this as a
            // legitimate rejection, not a malformed proof.
            return Ok(false);
        }
    }

    // Value half.
    let w = &proof.value_witness;
    let (expected_shard, final_slot) = probe_at::<H, E::ScalarField>(
        proof.ctr0,
        label,
        ctx.log_n_shards,
        ctx.shard_log_capacity(),
    );
    if expected_shard != w.shard_id {
        return Err(AegonError::Verification(
            "consistency value witness shard_id does not match H(ctr0, label)",
        ));
    }
    if verify_merkle_path::<E, P>(&w.leaf_s0, w.shard_id as usize, &w.merkle_path_s0)
        != s0_commit.merkle_root
    {
        return Err(AegonError::Verification(
            "consistency value s0 merkle path does not reconstruct s0 root",
        ));
    }
    if verify_merkle_path::<E, P>(&w.leaf_s1, w.shard_id as usize, &w.merkle_path_s1)
        != s1_commit.merkle_root
    {
        return Err(AegonError::Verification(
            "consistency value s1 merkle path does not reconstruct s1 root",
        ));
    }
    let value_point = bool_index_to_point::<E::ScalarField>(&final_slot);
    let mut tr0 = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_value.open");
    let ok0 = P::verify(
        &ctx.inner.verifier_param,
        &w.leaf_s0.rand_value_commitment,
        &value_point,
        &w.inner.eval_s0,
        &w.inner.proof_s0,
        &mut tr0,
    )?;
    let mut tr1 = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_value.open");
    let ok1 = P::verify(
        &ctx.inner.verifier_param,
        &w.leaf_s1.rand_value_commitment,
        &value_point,
        &w.inner.eval_s1,
        &w.inner.proof_s1,
        &mut tr1,
    )?;
    if !(ok0 && ok1) {
        return Ok(false);
    }
    if w.inner.eval_s0 != w.inner.eval_s1 {
        // Value-slot's rand_value diverged → the user's value was
        // updated across (s0, s1]. Caller treats this as a legitimate
        // rejection, not a malformed proof.
        return Ok(false);
    }

    Ok(true)
}

/// Verify a sharded invariance proof for the transition `prev → next`.
///
/// Steps:
///   1. The announced `next.merkle_root` must reconstruct from
///      `next.per_shard` (auditor independently hashes the leaves).
///   2. Re-derive the *shared* `(r_index, r_value)` Fiat-Shamir scalars
///      from `audit_state.r_*` and *all* shards' new data commits, in
///      shard-id order — the same hash the coordinator used at publish
///      time.
///   3. For every shard `i`, run the standard single-shard invariance
///      chain check (paper Fig. 4) on `(prev.per_shard[i],
///      next.per_shard[i])` using the *shared* scalars — *not* the
///      single-shard derivation, since the coordinator committed every
///      shard to the same `r`.
///
/// On success, `audit_state` is advanced to the new chain scalars,
/// ready for the next transition.
pub fn verify_sharded_invariance<E, P>(
    _ctx: &ShardedVerifierContext<E, P>,
    audit_state: &mut AuditState<E::ScalarField>,
    prev: &ShardedEpochCommitment<E, P>,
    next: &ShardedEpochCommitment<E, P>,
    proof: &ShardedInvarianceProof<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    P::Commitment: Clone
        + PartialEq
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Sub<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
{
    if next.epoch != prev.epoch + 1 {
        return Err(AegonError::Verification(
            "sharded audit proof must cover a one-epoch transition",
        ));
    }
    if prev.per_shard.len() != next.per_shard.len() {
        return Err(AegonError::Verification(
            "sharded audit: prev and next per_shard lengths differ",
        ));
    }
    if proof.per_shard.len() != next.per_shard.len() {
        return Err(AegonError::Verification(
            "sharded audit: invariance per_shard length does not match commit per_shard length",
        ));
    }

    // (1) Merkle root reconstructs from the announced per-shard commits.
    if merkle_root(&next.per_shard) != next.merkle_root {
        return Err(AegonError::Verification(
            "sharded audit: announced merkle_root does not match per_shard leaves",
        ));
    }

    // (2) Re-derive shared FS scalars from prev_r and the full per-shard tuple.
    let (new_r_index, new_r_value) =
        rederive_sharded_fs_scalars::<E, P>(audit_state.r_index, audit_state.r_value, next);

    // (3) Per-shard chain checks with the *shared* scalars. The per-
    // shard `InvarianceProof` is an empty marker now — all the bytes
    // the auditor needs are in `prev.per_shard[i]` / `next.per_shard[i]`,
    // and `verify_chain` does the homomorphism check directly on
    // commitments.
    for i in 0..next.per_shard.len() {
        let prev_i = &prev.per_shard[i];
        let next_i = &next.per_shard[i];

        let index_ok = verify_chain::<E, P>(
            new_r_index,
            &prev_i.index_commitment,
            &next_i.index_commitment,
            &prev_i.rand_index_commitment,
            &next_i.rand_index_commitment,
        );
        if !index_ok {
            return Ok(false);
        }
        let value_ok = verify_chain::<E, P>(
            new_r_value,
            &prev_i.value_commitment,
            &next_i.value_commitment,
            &prev_i.rand_value_commitment,
            &next_i.rand_value_commitment,
        );
        if !value_ok {
            return Ok(false);
        }
    }

    audit_state.r_index = new_r_index;
    audit_state.r_value = new_r_value;
    Ok(true)
}

// ---------- probe / FS / Merkle helpers --------------------------------

/// Derive a `(shard_id, slot_bits)` probe from `(ctr, label)` using the
/// hash suite. The first `log_n_shards` bits of `H_bits` index the
/// shard; the remaining `log_shard_capacity` bits address a slot within
/// that shard's polynomial. The verifier recomputes this exactly to
/// confirm a probe's shard_id and slot.
pub fn probe_at<H, F>(
    ctr: u64,
    label: &[u8],
    log_n_shards: usize,
    log_shard_capacity: usize,
) -> (u32, Vec<bool>)
where
    H: HashSuite<F>,
    F: ark_ff::PrimeField,
{
    let total_bits = log_n_shards + log_shard_capacity;
    let all_bits = H::h_bits(ctr, label, total_bits);
    let mut shard_id: u32 = 0;
    for (i, b) in all_bits[..log_n_shards].iter().enumerate() {
        if *b {
            shard_id |= 1u32 << i;
        }
    }
    let slot_bits = all_bits[log_n_shards..].to_vec();
    (shard_id, slot_bits)
}

/// Re-export of `aegon::hash::bool_index_to_usize` under a name that
/// doesn't collide with the field above. Used internally to dedup
/// in-batch slot claims by their canonical PCS index.
fn bool_index_to_usize_dims(bits: &[bool], dims: &[usize]) -> usize {
    super::hash::bool_index_to_usize(bits, dims)
}

/// FS-derive a field-element challenge from `(domain, prev, &[commits...])`.
fn fs_chain_scalar<F, C>(domain: &'static [u8], prev: F, commits: &[C]) -> F
where
    F: ark_ff::PrimeField,
    C: CanonicalSerialize,
{
    let mut hasher = Sha256::new();
    hasher.update(domain);
    let mut buf = Vec::new();
    prev.serialize_compressed(&mut buf)
        .expect("F serialize is infallible");
    hasher.update(&buf);
    for c in commits {
        buf.clear();
        c.serialize_compressed(&mut buf)
            .expect("commit serialize is infallible");
        hasher.update(&buf);
    }
    let digest = hasher.finalize();
    F::from_le_bytes_mod_order(&digest)
}

/// Hash a single [`EpochCommitment`] into a 32-byte leaf digest.
pub fn merkle_leaf<E: Pairing, P: AegonPcs<E>>(c: &EpochCommitment<E, P>) -> EpochDigest {
    let mut hasher = Sha256::new();
    hasher.update(b"aegon.sharded.leaf");
    let mut buf = Vec::new();
    c.index_commitment.serialize_compressed(&mut buf).unwrap();
    c.value_commitment.serialize_compressed(&mut buf).unwrap();
    c.rand_index_commitment
        .serialize_compressed(&mut buf)
        .unwrap();
    c.rand_value_commitment
        .serialize_compressed(&mut buf)
        .unwrap();
    hasher.update(&buf);
    hasher.finalize().into()
}

fn merkle_parent(left: &EpochDigest, right: &EpochDigest) -> EpochDigest {
    let mut hasher = Sha256::new();
    hasher.update(b"aegon.sharded.node");
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// Compute the Merkle root over an exact power-of-two number of leaves.
/// `n_shards = 1` is supported: the root is just the single leaf.
pub fn merkle_root<E: Pairing, P: AegonPcs<E>>(
    per_shard: &[EpochCommitment<E, P>],
) -> EpochDigest {
    assert!(
        per_shard.len().is_power_of_two(),
        "merkle_root: leaf count must be a power of two (got {})",
        per_shard.len()
    );
    let mut layer: Vec<EpochDigest> = per_shard.iter().map(merkle_leaf).collect();
    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len() / 2);
        for pair in layer.chunks_exact(2) {
            next.push(merkle_parent(&pair[0], &pair[1]));
        }
        layer = next;
    }
    layer[0]
}

/// Build the sibling-only Merkle path for `leaf_index` against
/// `per_shard`. Path length is `log2(per_shard.len())`.
pub fn build_merkle_path<E: Pairing, P: AegonPcs<E>>(
    per_shard: &[EpochCommitment<E, P>],
    leaf_index: usize,
) -> Vec<EpochDigest> {
    assert!(per_shard.len().is_power_of_two());
    assert!(leaf_index < per_shard.len());
    let mut layer: Vec<EpochDigest> = per_shard.iter().map(merkle_leaf).collect();
    let mut idx = leaf_index;
    let mut path = Vec::with_capacity(layer.len().trailing_zeros() as usize);
    while layer.len() > 1 {
        let sibling = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
        path.push(layer[sibling]);
        let mut next = Vec::with_capacity(layer.len() / 2);
        for pair in layer.chunks_exact(2) {
            next.push(merkle_parent(&pair[0], &pair[1]));
        }
        layer = next;
        idx /= 2;
    }
    path
}

/// Verify a Merkle path. Returns the root the path reconstructs.
pub fn verify_merkle_path<E: Pairing, P: AegonPcs<E>>(
    leaf: &EpochCommitment<E, P>,
    leaf_index: usize,
    path: &[EpochDigest],
) -> EpochDigest {
    let mut hash = merkle_leaf(leaf);
    let mut idx = leaf_index;
    for sibling in path {
        hash = if idx % 2 == 0 {
            merkle_parent(&hash, sibling)
        } else {
            merkle_parent(sibling, &hash)
        };
        idx /= 2;
    }
    hash
}

/// Recompute the shared FS scalars the prover used at the transition
/// from `prev` to `next`. Auditors call this to bind their own
/// invariance check to the same `(r_index, r_value)`.
pub fn rederive_sharded_fs_scalars<E: Pairing, P: AegonPcs<E>>(
    prev_r_index: E::ScalarField,
    prev_r_value: E::ScalarField,
    next: &ShardedEpochCommitment<E, P>,
) -> (E::ScalarField, E::ScalarField) {
    let index_commits: Vec<P::Commitment> = next
        .per_shard
        .iter()
        .map(|c| c.index_commitment.clone())
        .collect();
    let value_commits: Vec<P::Commitment> = next
        .per_shard
        .iter()
        .map(|c| c.value_commitment.clone())
        .collect();
    let r_index = fs_chain_scalar::<E::ScalarField, P::Commitment>(
        b"aegon.sharded.fs.r_index",
        prev_r_index,
        &index_commits,
    );
    let r_value = fs_chain_scalar::<E::ScalarField, P::Commitment>(
        b"aegon.sharded.fs.r_value",
        prev_r_value,
        &value_commits,
    );
    (r_index, r_value)
}
