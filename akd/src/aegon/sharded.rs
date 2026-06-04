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
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::rand::Rng;
use akd_core::aegon_crypto::pcs::PCSGlobalParam;
use akd_core::aegon_crypto::transcript::IOPTranscript;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use super::audit::verify_chain;
use super::config::{AegonConfig, VerifierContext};
use super::db::{
    key_coord_state, key_epoch_commit, key_history_openings, key_label_placement,
    key_routing, key_slot, key_value, key_value_history, Db, DbOp, DbSource, RedisDb,
};
use super::error::AegonError;
use super::hash::{bool_index_to_point, HashSuite, Sha256Hash};
use super::server::Aegon;
use super::types::{
    AegonPcs, AuditState, EpochCommitment, HistoryOpeningEntry, HistoryOpenings, Label, RandPair,
    Value, ValueChangeEntry,
};

/// Lazily-built thread pool for the blocking gRPC fan-out in
/// `plan_phase_1_batches`. Sized far above the coordinator's CPU
/// count because each task spends nearly all its time blocked on a
/// gRPC RTT, not on CPU — see comment at the call site.
static FALLBACK_POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

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
    /// If non-empty, every in-process shard's [`Aegon`] will fetch its
    /// per-opening masking packages from one of these remote
    /// `aegon_masking_server` endpoints instead of using the default
    /// in-process [`super::masking::MaskingPool`]. With more than one
    /// endpoint, requests are spread across servers in round-robin via
    /// [`super::masking::MaskingClientPool`], so the per-shard masking
    /// ceiling becomes `N * single-server-rate` instead of just
    /// single-server-rate. Only honoured for
    /// [`ShardTransport::InProcess`] — remote shards configure their
    /// own masking source via `aegon_shard_server`'s `--masking-addr`.
    pub masking_addrs: Vec<String>,
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

        let file = std::fs::File::create(path).map_err(|e| {
            AegonError::Config(format!(
                "create srs file '{}': {e}",
                path.display()
            ))
        })?;
        // BufWriter — arkworks' CanonicalSerialize issues many small
        // writes (one per scalar / group-element field). Without
        // buffering each one is a syscall; on a 4–5 GB SRS that is
        // ~250M syscalls and turns serialise into the slow phase.
        //
        // serialize_uncompressed (instead of _compressed) skips the
        // per-point encoding that the reader would have to invert
        // with a sqrt — file is ~2× larger but load drops from
        // ~25 min/4.9GB to ~1 min on n2-standard-16. Each shard
        // writes its own local copy now (no NFS), so the size hit
        // is contained to local disk.
        let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);
        pk.serialize_uncompressed(&mut writer).map_err(|e| {
            AegonError::Config(format!("serialize prover_param: {e}"))
        })?;
        vk.serialize_uncompressed(&mut writer).map_err(|e| {
            AegonError::Config(format!("serialize verifier_param: {e}"))
        })?;
        use std::io::Write;
        writer.flush().map_err(|e| {
            AegonError::Config(format!("flush srs file: {e}"))
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
    let file = std::fs::File::open(path).map_err(|e| {
        AegonError::Config(format!(
            "open srs file '{}': {e}",
            path.display()
        ))
    })?;
    // BufReader — arkworks' CanonicalDeserialize issues one `read`
    // per scalar / group-element field (~50 bytes). Without
    // buffering each call hits the kernel; a 4.9GB SRS produced
    // ~250M read syscalls and took 25min to load on n2-standard-16.
    // A 1MB buffer cuts syscalls ~20000x and load time to ~minutes.
    //
    // deserialize_uncompressed_unchecked: the file format
    // matches `serialize_uncompressed` above. Skips the per-point
    // sqrt (compressed form would need sqrt_in_Fq per element,
    // which is the dominant cost in compressed deserialise at
    // multi-GB scale) and the subgroup check. We trust the file
    // — it was either written by the shard itself or generated
    // by aegon_srs_gen on the same host. Load: ~25min → ~1min.
    let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
    let pk = P::ProverParam::deserialize_uncompressed_unchecked(&mut reader).map_err(|e| {
        AegonError::Config(format!("deserialize prover_param: {e}"))
    })?;
    let vk = P::VerifierParam::deserialize_uncompressed_unchecked(&mut reader).map_err(|e| {
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
    masking_addrs: Vec<String>,
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
            masking_addrs: Vec::new(),
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

    /// Wire every in-process shard to a single remote
    /// `aegon_masking_server` at this endpoint (e.g.
    /// `"http://127.0.0.1:50061"`). Convenience wrapper around
    /// [`Self::masking_addrs`] with a one-element list — see that
    /// method's docs for the multi-server / round-robin path.
    pub fn masking_addr(self, v: impl Into<String>) -> Self {
        self.masking_addrs(vec![v.into()])
    }

    /// Wire every in-process shard to a *set* of remote
    /// `aegon_masking_server` endpoints. Each shard's [`Aegon`] swaps
    /// its default in-process [`super::masking::MaskingPool`] for a
    /// [`super::masking::MaskingClientPool`] that round-robins package
    /// fetches across all endpoints. Passing one element gives the
    /// same behaviour as the old single-endpoint path. Has no effect
    /// when [`ShardTransport::Remote`] is selected (remote shards
    /// configure their own masking source).
    pub fn masking_addrs(mut self, v: Vec<String>) -> Self {
        self.masking_addrs = v;
        self
    }

    /// SRS source. Defaults to `SrsSource::DangerouslyGenerate`
    /// (suitable for tests; **never** production).
    pub fn srs(mut self, v: SrsSource) -> Self {
        self.srs = v;
        self
    }

    /// Coordinator-side label→value KV store. Defaults to
    /// [`DbSource::None`] — set to [`DbSource::Redis`] or
    /// [`DbSource::Rocks`] in cluster deployments so `lookup` can
    /// return the raw value bytes.
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
            masking_addrs: self.masking_addrs,
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
/// fields in declaration order.
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
    /// - `Some(H_F(label))` when the label is brand-new in this
    ///   epoch — the shard writes this hash as the slot's identity,
    ///   which is what the lookup-side open-addressing trail checks
    ///   against.
    /// - `None` when the label already exists (the slot's identity
    ///   was set in a prior epoch and must not change). Tells
    ///   `Aegon::publish_phase_1_at_slots` to skip the index-poly
    ///   write at this slot.
    pub h_label: Option<F>,
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
/// post-publish DB durability barrier can write fresh `aegon:slot:*`
/// and `aegon:routing:*` keys without scanning the routing table.
///
/// (Existing labels — being value-updated — don't need fresh routing /
/// slot writes since those keys were committed in a prior epoch.)
#[derive(Clone, Debug)]
pub(crate) struct NewPlacement {
    pub(crate) label: Label,
    pub(crate) shard_id: u32,
    pub(crate) slot_idx: usize,
    /// Full open-addressing probe trail (length 1 when no collisions, >1
    /// when the label collided on earlier probes). Carried here so
    /// `persist_publish_to_db` can write `routing:{label}` to the DB
    /// without reading from an in-memory `self.routing` HashMap (which
    /// the medium-scale refactor eliminated to keep coord RSS bounded).
    pub(crate) trail: Vec<(u32, Vec<bool>)>,
}

/// Coordinator state recovered from the DB on restart. Built by
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
///
/// ## Single Merkle tree, not four
///
/// Each leaf is `H("aegon.sharded.leaf" || index_com || value_com ||
/// rand_index_com || rand_value_com)` for one shard — so all four
/// per-shard commitments live in the same leaf and the same path
/// covers all four polynomials. That's a deliberate compression
/// (saves 3× the path digests and 3 independent Merkle roots);
/// it does mean opening any of the four polynomials at a slot
/// reveals all four commits for the owning shard.
///
/// ## Cached paths
///
/// `paths[i]` is the sibling-only Merkle path from shard `i`'s leaf
/// to `merkle_root`. Eagerly computed once in
/// [`ShardedEpochCommitment::with_per_shard`] (alongside the root),
/// so the publish + lookup + history paths can read pre-built path
/// data via [`merkle_path`](Self::merkle_path) instead of rebuilding
/// the whole tree on every opening.
///
/// ## Wire format
///
/// The `paths` field is a server-side cache, **not** part of the
/// published bytes — auditors and verifiers can rebuild it from
/// `per_shard` on their own, so we don't pay the ~`n × log(n) × 32 B`
/// cost on the wire. Deserialisation rebuilds the cache from
/// `per_shard` automatically.
///
/// ## Single-shard degenerate case
///
/// When `n_shards = 1`: `paths[0]` is empty and `merkle_root` equals
/// `merkle_leaf(per_shard[0])`. The verifier path-walk loop is a
/// no-op and the leaf hash IS the dictionary commitment. No special
/// branch is needed in callers — empty path is handled uniformly.
#[derive(Debug)]
pub struct ShardedEpochCommitment<E: Pairing, P: AegonPcs<E>> {
    pub epoch: u64,
    pub merkle_root: EpochDigest,
    pub per_shard: Vec<EpochCommitment<E, P>>,
    /// Sibling-only Merkle paths, one per shard. Derived from
    /// `per_shard` and never serialised — see the type-level docs.
    /// Kept `pub(crate)` so the constructor is the only public way
    /// to produce a well-formed instance.
    pub(crate) paths: Vec<Vec<EpochDigest>>,
}

impl<E: Pairing, P: AegonPcs<E>> ShardedEpochCommitment<E, P> {
    /// Construct a commitment for `(epoch, per_shard)` by computing the
    /// root and all sibling paths in a single pass over the Merkle
    /// tree. Use this everywhere a `ShardedEpochCommitment` is built —
    /// it's the only public way to populate the `paths` cache.
    ///
    /// `per_shard.len()` must be a power of two (asserted by the
    /// underlying tree-build helper).
    pub fn with_per_shard(epoch: u64, per_shard: Vec<EpochCommitment<E, P>>) -> Self {
        let (merkle_root, paths) = build_merkle_root_and_paths::<E, P>(&per_shard);
        Self {
            epoch,
            merkle_root,
            per_shard,
            paths,
        }
    }

    /// Sibling-only Merkle path from shard `shard_id`'s leaf up to
    /// `merkle_root`. Pre-built; O(1) lookup at every call site that
    /// previously walked `build_merkle_path`. Returns an empty slice
    /// at `n_shards = 1`.
    pub fn merkle_path(&self, shard_id: usize) -> &[EpochDigest] {
        &self.paths[shard_id]
    }
}

impl<E: Pairing, P: AegonPcs<E>> Clone for ShardedEpochCommitment<E, P> {
    fn clone(&self) -> Self {
        Self {
            epoch: self.epoch,
            merkle_root: self.merkle_root,
            per_shard: self.per_shard.clone(),
            paths: self.paths.clone(),
        }
    }
}

// Manual ark-serialize impls — the `paths` cache is derivable from
// `per_shard`, so the wire format only carries `(epoch, merkle_root,
// per_shard)` and deserialisation rebuilds the cache. This keeps the
// on-disk + bulletin-board encoding byte-identical to what the older
// derive-based impl produced.
impl<E: Pairing, P: AegonPcs<E>> CanonicalSerialize for ShardedEpochCommitment<E, P>
where
    EpochCommitment<E, P>: CanonicalSerialize,
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.epoch.serialize_with_mode(&mut writer, compress)?;
        self.merkle_root
            .serialize_with_mode(&mut writer, compress)?;
        self.per_shard
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.epoch.serialized_size(compress)
            + self.merkle_root.serialized_size(compress)
            + self.per_shard.serialized_size(compress)
    }
}

impl<E: Pairing, P: AegonPcs<E>> Valid for ShardedEpochCommitment<E, P>
where
    EpochCommitment<E, P>: Valid,
{
    fn check(&self) -> Result<(), SerializationError> {
        self.epoch.check()?;
        self.merkle_root.check()?;
        self.per_shard.check()?;
        // `paths` is derived from `per_shard`, no independent validity
        // condition to verify.
        Ok(())
    }
}

impl<E: Pairing, P: AegonPcs<E>> CanonicalDeserialize for ShardedEpochCommitment<E, P>
where
    EpochCommitment<E, P>: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let epoch = u64::deserialize_with_mode(&mut reader, compress, validate)?;
        let merkle_root =
            <EpochDigest>::deserialize_with_mode(&mut reader, compress, validate)?;
        let per_shard =
            Vec::<EpochCommitment<E, P>>::deserialize_with_mode(&mut reader, compress, validate)?;
        // Rebuild the path cache deterministically from per_shard;
        // sanity-check that the stored `merkle_root` matches what
        // per_shard reconstructs (catches truncated reads + tampering
        // at the wire layer).
        let (recomputed_root, paths) = build_merkle_root_and_paths::<E, P>(&per_shard);
        if recomputed_root != merkle_root {
            return Err(SerializationError::InvalidData);
        }
        Ok(Self {
            epoch,
            merkle_root,
            per_shard,
            paths,
        })
    }
}

/// One probe along a sharded open-addressing trail. The verifier
/// recomputes `(shard_id, slot_bits)` from `H(ctr, label)`, then checks
/// that `merkle_path` reconstructs the epoch root from `leaf`, and that
/// the PCS `proof` verifies `evaluation` against `leaf.index_commitment`
/// at the slot.
///
/// When the deployment uses a VRF for index assignment (`EcVrfHash`
/// rather than `Sha256Hash`), `vrf_proof` carries the
/// `VRF_PROOF_BYTES`-long Schnorr-style proof that this probe's
/// `(ctr, label)` was honestly mapped — the verifier consumes it to
/// recover `slot_bits` rather than computing them locally. For the
/// SHA-256 path `vrf_proof` is the empty byte vector and the verifier
/// falls back to `H::h_bits`.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedProbe<E: Pairing, P: AegonPcs<E>> {
    pub shard_id: u32,
    pub leaf: EpochCommitment<E, P>,
    pub merkle_path: Vec<EpochDigest>,
    pub evaluation: E::ScalarField,
    pub proof: P::Proof,
    /// RFC 9381 ECVRF proof for `(ctr, label)` at this probe, or empty
    /// when the deployment uses the non-VRF SHA-256 hash suite. Length
    /// is exactly `VRF_PROOF_BYTES` (80) when present.
    pub vrf_proof: Vec<u8>,
}

// Manual `Clone` impl: `#[derive(Clone)]` would generate
// `where E: Clone, P: Clone`, but `E: Pairing` / `P: AegonPcs<E>`
// don't carry `Clone`. The fields *are* all clonable on their own
// (P::Proof: Clone via the PCS trait, ScalarField: Clone via Field,
// EpochCommitment has a manual Clone impl), so we just spell it out.
impl<E: Pairing, P: AegonPcs<E>> Clone for ShardedProbe<E, P> {
    fn clone(&self) -> Self {
        Self {
            shard_id: self.shard_id,
            leaf: self.leaf.clone(),
            merkle_path: self.merkle_path.clone(),
            evaluation: self.evaluation,
            proof: self.proof.clone(),
            vrf_proof: self.vrf_proof.clone(),
        }
    }
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

/// Where in the cluster a label has been canonically placed —
/// `(shard, per-shard slot bits)`. Returned by `lookup_label` and
/// reconstructed by `verify_lookup_label`. The client caches this
/// once and uses it for any number of subsequent `lookup_value` calls
/// without having to re-prove residency.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize, PartialEq, Eq)]
pub struct LabelSlot {
    pub shard_id: u32,
    pub slot_bits: Vec<bool>,
}

/// Proof that a label resides at a specific `(shard, slot)`. This is
/// the open-addressing half of the original combined `lookup` proof —
/// one probe per `ctr ∈ 0..=ctr0`, ending at the canonical slot. The
/// verifier walks the trail, re-derives each `(shard, slot)` from
/// `H(ctr, label)`, anchors each leaf under the epoch root, verifies
/// each `index_poly` opening, and checks the open-addressing
/// constraints (earlier slots are non-empty + non-`H_F(label)`, final
/// slot equals `H_F(label)`).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedLabelProof<E: Pairing, P: AegonPcs<E>> {
    pub ctr0: u64,
    pub probes: Vec<ShardedProbe<E, P>>,
}

/// Proof that `value_poly` opens to `evaluation` at the given slot.
/// Decoupled from any specific label: a client who already cached
/// the `LabelSlot` from a prior `lookup_label` can call
/// `lookup_value(slot)` indefinitely as the value updates.
///
/// The raw value bytes are out-of-band — the verifier obtains them
/// from the side-channel (the DB, the publisher, wherever) and
/// confirms `evaluation == H_F(value)`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedValueProof<E: Pairing, P: AegonPcs<E>> {
    pub shard_id: u32,
    pub slot_bits: Vec<bool>,
    pub leaf: EpochCommitment<E, P>,
    pub merkle_path: Vec<EpochDigest>,
    pub evaluation: E::ScalarField,
    pub proof: P::Proof,
}

/// Maximum number of value-history entries retained per label. The
/// coordinator's value-history feature maintains a sliding window of
/// at most this many records per label in the DB (LPUSH + LTRIM 0
/// N-1 semantics; both Redis and RocksDB backends implement these
/// via the `Db` trait). When a label changes value more than
/// `HISTORY_WINDOW` times, only the most recent `HISTORY_WINDOW`
/// records are observable via `lookup_history`.
pub const HISTORY_WINDOW: usize = 5;

/// One persisted value-history record. Each `lookup_history(label)`
/// returns up to `HISTORY_WINDOW` of these — one per publish in which
/// the label's value changed (placement is treated as the first
/// value-change). Self-contained: includes the per-shard commitments
/// at both the prior and the new epoch + their merkle paths under the
/// sharded root, so a verifier can re-anchor every opening without
/// any external bulletin-board lookup beyond pinning the *roots*
/// against the user's trusted source.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct StoredValueHistoryEntry<E: Pairing, P: AegonPcs<E>> {
    /// Epoch at which this value-change was published. The entry's
    /// `post_*` commitments are the per-shard commitments at this
    /// epoch; the `prev_*` commitments are at epoch (epoch - 1).
    pub epoch: u64,
    /// Shard that owns the slot.
    pub shard_id: u32,
    /// Slot bits within the shard (low bit first).
    pub slot_bits: Vec<bool>,
    /// Raw value bytes that were published. The verifier hashes these
    /// locally and checks `H_F(value_bytes) == value_post_eval`.
    pub value_bytes: Vec<u8>,
    /// `rand_value_n(slot)` at the prior-epoch rand_value commitment
    /// inside `prev_shard_commit`. Zero on a brand-new placement.
    pub rand_value_pre_eval: E::ScalarField,
    pub rand_value_pre_proof: P::Proof,
    /// `rand_value_{n+1}(slot)` at the new-epoch rand_value commitment.
    pub rand_value_post_eval: E::ScalarField,
    pub rand_value_post_proof: P::Proof,
    /// `value_{n+1}(slot) = H_F(value)` at the new-epoch value
    /// commitment.
    pub value_post_eval: E::ScalarField,
    pub value_post_proof: P::Proof,
    /// Per-shard commitment at epoch (epoch - 1) — the leaf the
    /// `rand_value_pre_proof` anchors against. Carries
    /// `rand_value_commitment` (used) plus the other per-shard
    /// commitments (unused for verification but kept so the leaf hash
    /// reproduces).
    pub prev_shard_commit: EpochCommitment<E, P>,
    /// Merkle path from `prev_shard_commit` (at position `shard_id`)
    /// up to the sharded root at epoch (epoch - 1).
    pub prev_merkle_path: Vec<EpochDigest>,
    /// Per-shard commitment at epoch — anchors the post-update
    /// `rand_value` and `value` openings.
    pub post_shard_commit: EpochCommitment<E, P>,
    /// Merkle path from `post_shard_commit` up to the sharded root at
    /// epoch.
    pub post_merkle_path: Vec<EpochDigest>,
}

/// `lookup_history(label)` response: up to `HISTORY_WINDOW` entries,
/// most-recent first (matches the LPUSH+LRANGE 0 N-1 head-first
/// ordering both backends implement under the `Db` trait).
///
/// `freshness` (when present) attests "no publish has touched this
/// slot since `entries[0].epoch`". It is freshly computed by the
/// owning shard at lookup time — the only part of this bundle that
/// is **not** a pure DB read.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedValueHistory<E: Pairing, P: AegonPcs<E>> {
    pub label: Vec<u8>,
    pub entries: Vec<StoredValueHistoryEntry<E, P>>,
    /// Live opening of the latest `rand_value_poly` at the label's
    /// slot, anchored under the live sharded root. `None` iff
    /// `entries.is_empty()` (nothing to attest freshness against).
    pub freshness: Option<FreshnessAttestation<E, P>>,
}

/// One placement record. Written once per label at the moment the
/// label is first added to the dictionary; read back during
/// `lookup_label_history`. Self-contained: includes the per-shard
/// commitment at the placement epoch + its merkle path under the
/// sharded root, so the verifier can re-anchor the opening without
/// any external bulletin-board lookup beyond pinning the placement
/// root against their trusted source.
///
/// Why just one stored opening (vs. value-history's three per
/// entry): labels are placed exactly once and never mutate
/// (no rename, no delete in the current system). So the only
/// non-trivial check is "the slot's `rand_index` hasn't been
/// disturbed since placement" — which `lookup_label_history` answers
/// by sending this stored placement opening plus a freshly-computed
/// opening of the live `rand_index_poly`. The verifier compares
/// evaluations; if they match, no publish has touched this slot
/// since the placement epoch.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct StoredLabelPlacement<E: Pairing, P: AegonPcs<E>> {
    /// Epoch at which this label was placed.
    pub epoch: u64,
    /// Shard that owns the slot.
    pub shard_id: u32,
    /// Slot bits within the shard (low bit first).
    pub slot_bits: Vec<bool>,
    /// `rand_index_{epoch}(slot)` opened against this shard's
    /// `placement_shard_commit.rand_index_commitment`. Equals
    /// `r_index_{epoch-1} · H_F(label)` on a fresh placement (the
    /// pre-placement value is zero because the slot was empty).
    pub rand_index_eval: E::ScalarField,
    pub rand_index_proof: P::Proof,
    /// Per-shard `EpochCommitment` at the placement epoch — anchors
    /// the placement opening.
    pub placement_shard_commit: EpochCommitment<E, P>,
    /// Merkle path from `placement_shard_commit` (at position
    /// `shard_id`) up to the sharded root at the placement epoch.
    pub placement_merkle_path: Vec<EpochDigest>,
}

/// `lookup_label_history(label)` response. The label-side analog of
/// [`ShardedValueHistory`].
///
/// Asymmetry with `ShardedValueHistory`: labels are placed exactly
/// once, so there's no sliding window of past changes — instead,
/// just the placement record (frozen at placement time) plus a
/// freshness attestation (freshly computed by the owning shard on
/// every call). Together with a separate `lookup_label(label)`
/// (which proves "label is currently at slot S in the live
/// state"), the verifier learns: "this label has been bound to
/// slot S since placement at epoch E" — i.e., the slot has not
/// been disturbed by any subsequent publish, no other label has
/// displaced it, no migration has occurred.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedLabelHistory<E: Pairing, P: AegonPcs<E>> {
    pub label: Vec<u8>,
    /// `None` iff the label is unknown to the coordinator
    /// (placement record has never been written for it).
    pub placement: Option<StoredLabelPlacement<E, P>>,
    /// Live `rand_index(slot)` opening + anchoring under the live
    /// sharded root. `None` iff `placement` is `None`.
    pub freshness: Option<FreshnessAttestationLabel<E, P>>,
}

/// Label-side mirror of [`FreshnessAttestation`]. Same shape, but
/// opens the live `rand_index_poly` against the live shard's
/// `rand_index_commitment` (rather than `rand_value`).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct FreshnessAttestationLabel<E: Pairing, P: AegonPcs<E>> {
    pub shard_id: u32,
    pub slot_bits: Vec<bool>,
    /// `rand_index_live(slot)` — the live shard's rand_index poly
    /// evaluated at the slot.
    pub rand_index_current_eval: E::ScalarField,
    pub rand_index_current_proof: P::Proof,
    /// Live per-shard `EpochCommitment` for `shard_id`.
    pub shard_commit: EpochCommitment<E, P>,
    /// Merkle path from `shard_commit` (at position `shard_id`) up
    /// to the live sharded root.
    pub merkle_path: Vec<EpochDigest>,
}

/// "No-change since the most recent history entry" attestation.
///
/// Each publish updates `rand_value_poly` only at the slots it
/// touched (the rest of the polynomial is invariant under the
/// chain-blinding homomorphism — `delta_value_poly` is zero outside
/// the touched slots, so `new_rand_value = prev_rand_value` outside
/// them). Therefore: if the slot's `rand_value` evaluation right
/// after the most recent value-change in `entries[0]` equals the
/// live evaluation **now**, no publish in between can have written
/// to this slot.
///
/// The verifier in `verify_lookup_history` checks:
///   1. The opening verifies under `shard_commit.rand_value_commitment`.
///   2. The opening's evaluation equals `entries[0].rand_value_post_eval`.
///   3. The Merkle path reconstructs the live sharded root the
///      caller separately trusts (typically: pinned against the
///      coordinator's `current_commitment()`).
///
/// (1) + (2) bind the live state to the latest history entry; (3)
/// is the freshness anchor.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct FreshnessAttestation<E: Pairing, P: AegonPcs<E>> {
    /// Shard that owns the slot. Matches `entries[0].shard_id`.
    pub shard_id: u32,
    /// Slot bits within the shard. Matches `entries[0].slot_bits`.
    pub slot_bits: Vec<bool>,
    /// `rand_value_live(slot)` — the live shard's rand_value poly
    /// evaluated at the slot.
    pub rand_value_current_eval: E::ScalarField,
    pub rand_value_current_proof: P::Proof,
    /// Live per-shard `EpochCommitment` for `shard_id`. Carries
    /// `rand_value_commitment` (used) plus the other per-shard
    /// commitments (unused for verification but kept so the leaf
    /// hash reproduces under the live root).
    pub shard_commit: EpochCommitment<E, P>,
    /// Merkle path from `shard_commit` (at position `shard_id`) up
    /// to the live sharded root.
    pub merkle_path: Vec<EpochDigest>,
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

/// Verifier-side bundle, including the deployment's `log_n_shards`
/// (clients need it to recompute the probe trail) and, when the
/// deployment uses a VRF for index assignment, the verifier
/// configured with the server's published public key.
#[derive(Clone)]
pub struct ShardedVerifierContext<E: Pairing, P: AegonPcs<E>> {
    /// `inner.log_capacity` is the *per-shard* log_capacity.
    pub inner: VerifierContext<E, P>,
    pub log_n_shards: usize,
    /// `Some(verifier)` when the deployment runs with `EcVrfHash`
    /// (RFC 9381 ECVRF) and `verify_lookup_label` should consume
    /// each probe's `vrf_proof`. `None` for the SHA-256 path, in
    /// which case slot bits are re-derived locally via `H::h_bits`.
    pub vrf_verifier: Option<super::hash::VrfVerifier>,
}

impl<E: Pairing, P: AegonPcs<E>> ShardedVerifierContext<E, P> {
    pub fn new(inner: VerifierContext<E, P>, log_n_shards: usize) -> Self {
        Self {
            inner,
            log_n_shards,
            vrf_verifier: None,
        }
    }

    /// Attach a [`VrfVerifier`](super::hash::VrfVerifier) so subsequent
    /// `verify_lookup_label` calls consume the `vrf_proof` field on
    /// each probe instead of re-deriving bits via `H::h_bits`. Used
    /// when the deployment runs with `EcVrfHash` on the server side.
    pub fn with_vrf_verifier(mut self, verifier: super::hash::VrfVerifier) -> Self {
        self.vrf_verifier = Some(verifier);
        self
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
/// routing table to the DB and rebuild on restart.
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

    /// Server-side VRF prover. `Some` when the deployment runs with
    /// `EcVrfHash` — `lookup_label` then computes a VRF proof per
    /// probe and attaches it to the `ShardedProbe.vrf_proof` field.
    /// `None` for the SHA-256 path, in which case `vrf_proof` is left
    /// empty and the verifier re-derives slot bits via `H::h_bits`.
    vrf_prover: Option<super::hash::VrfProver>,
}

impl<E, P, H> ShardedAegon<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam: PCSGlobalParam + CanonicalDeserialize + Send + Sync + 'static,
    P::VerifierParam: PCSGlobalParam + CanonicalDeserialize + Send + Sync + 'static,
    P::Commitment: Clone
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
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
                // If remote masking server endpoints are configured,
                // connect once and share the client(s) across all shards.
                // A one-element list builds a single MaskingClient
                // (same behaviour as before). A multi-element list
                // builds a MaskingClientPool that round-robins fetches
                // across all servers, so the per-shard masking
                // throughput becomes `N * single-server-rate`.
                // Both are Clone-cheap (Arc-wrapped worker threads
                // under the hood), so sharing across shards is fine.
                let masking_source: Option<
                    std::sync::Arc<dyn super::masking::MaskingSource<E, P>>,
                > = match config.masking_addrs.len() {
                    0 => None,
                    1 => {
                        let client = super::masking::MaskingClient::<E, P>::connect(
                            config.masking_addrs[0].clone(),
                        )?;
                        Some(std::sync::Arc::new(client))
                    },
                    _ => {
                        let pool = super::masking::MaskingClientPool::<E, P>::connect_all(
                            &config.masking_addrs,
                        )?;
                        Some(std::sync::Arc::new(pool))
                    },
                };
                let mut shards: Vec<Box<dyn super::shard_grpc::ShardHandle<E, P, H>>> =
                    Vec::with_capacity(n_shards);
                for _ in 0..n_shards {
                    let mut aegon = Aegon::<E, P, H>::init(
                        prover_param.clone(),
                        verifier_param.clone(),
                        &shard_config,
                    )?;
                    if let Some(src) = &masking_source {
                        aegon.set_masking_source(std::sync::Arc::clone(src));
                    }
                    shards.push(Box::new(aegon));
                }
                (shards, dims, vctx)
            },
            ShardTransport::Remote { endpoints } => {
                // Remote shards already loaded their own SRS at boot
                // (via the shard-server binary). The coordinator just
                // fetches the verifier-side projection from shard 0
                // over gRPC — `verifier_param` is ~tens of KB
                // regardless of dictionary size, while the prover-side
                // SRS that produced it can be multi-GB (and the
                // coordinator never needs the prover-side bytes; all
                // prover work happens on the shards).
                //
                // The previous code regenerated the full SRS from
                // `--setup-seed` here just to extract `verifier_param`
                // and `block_dims`. At medium scale (log_cap=27, k=9)
                // that took ~11 min on a 4-vCPU coord — a one-time
                // setup cost that turned a fast-cache-hit run into a
                // slow-cache-miss one the first time `private=true`
                // was used. The fetch path is ~1 second regardless of
                // cache state and matches between zk and nozk modes.
                if endpoints.is_empty() {
                    return Err(AegonError::Config(
                        "Remote shard transport requires at least one endpoint".into(),
                    ));
                }
                let (vctx, _fetched_log_capacity) =
                    super::shard_grpc::fetch_verifier_context_from_endpoint::<E, P>(
                        endpoints[0].clone(),
                    )?;
                let verifier_param = vctx.verifier_param.clone();
                // `block_dims` for KZH-k is a pure function of
                // `(k, log_capacity)` — we use the dummy SRS-free path
                // via `P::block_dims_from_verifier_param` when
                // available; for the generic trait fallback we
                // construct dims via the verifier_param's own getter
                // wrapper. PCSGlobalParam exposes everything we need.
                let dims = P::block_dims_from_verifier_param(
                    &verifier_param,
                    shard_config.log_capacity,
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
        let initial_commit =
            ShardedEpochCommitment::<E, P>::with_per_shard(0, initial_per_shard);

        // Coordinator-side KV store. Connect eagerly so a misconfigured
        // URL fails at setup, not on the first publish.
        let db: Option<Box<dyn Db>> = match &config.db {
            DbSource::None => None,
            DbSource::Redis(url) => Some(Box::new(RedisDb::connect(url)?)),
            DbSource::Rocks(path) => Some(Box::new(crate::aegon::db::RocksDb::open(path)?)),
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
                vrf_prover: None,
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
                vrf_prover: None,
            })
        }
    }

    /// Attach a VRF prover to the coordinator. After this call,
    /// `lookup_label` populates the `vrf_proof` field on each
    /// `ShardedProbe` with a real RFC 9381 ECVRF proof, and
    /// `sharded_verifier_context()` returns a context whose
    /// `vrf_verifier` is set to the matching public key. The prover
    /// stays in effect for the rest of the process's lifetime.
    ///
    /// Production deployments construct the prover once at startup
    /// from `VrfProver::from_env()` (driven by the
    /// `AEGON_VRF_SEED` / `AEGON_VRF_KEY_PATH` environment variables
    /// — see `aegon::hash::vrf_key_source`). Tests and microbenches
    /// can use `VrfProver::from_seed(&BENCH_VRF_SEED)`.
    pub fn set_vrf_prover(&mut self, prover: super::hash::VrfProver) {
        self.vrf_prover = Some(prover);
    }

    /// Current VRF prover, if one has been configured.
    pub fn vrf_prover(&self) -> Option<&super::hash::VrfProver> {
        self.vrf_prover.as_ref()
    }

    /// Read the coordinator's durable state from the DB if any was
    /// previously persisted. Returns `None` on a fresh DB (no
    /// `aegon:coord:state` key), `Some(RecoveredState)` on a restart,
    /// `Err` if a key exists but a downstream get/deserialize fails
    /// — that means the DB is half-written or corrupt and the caller
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
                    "epoch commitment for epoch {e} missing from the DB"
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

        // 3. With DB present, routing is read on-demand from RocksDB
        //    (see `read_routing`). We deliberately do NOT pre-load the
        //    full label→routing map here — at 60M+ entries that's
        //    multiple GB of RAM that would defeat the point of having a
        //    persistent store. RocksDB's bloom filters keep on-demand
        //    `get` cheap (~µs per existence check).
        let routing: HashMap<Label, LabelRouting> = HashMap::new();

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

    /// Convenience: full sharded verifier context. When a VRF prover
    /// has been attached (`set_vrf_prover`), the returned context
    /// carries a matching `VrfVerifier` so `verify_lookup_label` will
    /// consume the `vrf_proof` field on each probe.
    pub fn sharded_verifier_context(&self) -> ShardedVerifierContext<E, P> {
        let mut ctx = ShardedVerifierContext::new(self.verifier_context(), self.log_n_shards);
        if let Some(prover) = self.vrf_prover.as_ref() {
            ctx = ctx.with_vrf_verifier(super::hash::VrfVerifier::new(prover.public_key().clone()));
        }
        ctx
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

    /// Bench-only: populate every (in-process) shard's polynomials with
    /// `total_count` random `(slot, h_label, h_value)` entries, evenly
    /// split across shards, then refresh the cached epoch-0 commit so
    /// the next `publish` sees the prefilled state. Used by
    /// `aegon_publish_bench` to sweep across fill percentages without
    /// having to actually drive the publish flow for hundreds of
    /// thousands of entries.
    ///
    /// Requires every shard's [`ShardHandle::prefill_random_in_place`]
    /// to succeed — which is only true for in-process shards (the
    /// gRPC client transport returns `Err`, since remote shards
    /// prefill at boot time via `aegon_shard_server --prefill-count`).
    /// Distributed-mode benches should prefill via the shard binary
    /// and then run this with `total_count = 0`.
    ///
    /// `seed_base` deterministically distinguishes per-shard prefill:
    /// shard `i` is seeded with `seed_base.wrapping_add(i as u64)`,
    /// mirroring the bench-cluster.sh pattern `PREFILL_SEED + i`.
    pub fn prefill_random_per_shard(
        &mut self,
        total_count: u64,
        seed_base: u64,
    ) -> Result<(), AegonError> {
        let n_shards = self.shards.len();
        if n_shards == 0 {
            return Err(AegonError::Config("prefill: no shards".into()));
        }
        // Even split with leftover assigned to the first `r` shards.
        let per_shard = total_count / n_shards as u64;
        let leftover = (total_count % n_shards as u64) as usize;
        for (i, shard) in self.shards.iter_mut().enumerate() {
            let count_i = per_shard + if i < leftover { 1 } else { 0 };
            let seed_i = seed_base.wrapping_add(i as u64);
            shard.prefill_random_in_place(count_i as usize, seed_i)?;
        }
        // The shards just reset themselves back to epoch-0 — wipe
        // the coordinator's own epoch + epoch_commits chain to match
        // and rebuild the epoch-0 Merkle root from each shard's
        // freshly-prefilled commitment. Without this, a subsequent
        // publish would advance the coordinator's epoch from N+1
        // while the shards thought it was 1, and FS-chain
        // derivation would desync.
        let per_shard: Vec<EpochCommitment<E, P>> = self
            .shards
            .iter()
            .map(|s| s.current_commitment())
            .collect();
        let refreshed = ShardedEpochCommitment::<E, P>::with_per_shard(0, per_shard);
        self.epoch = 0;
        self.epoch_commits.clear();
        self.epoch_commits.push(refreshed);
        Ok(())
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
    ///      into the DB for the lookup path.
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(
            level = "debug",
            skip_all,
            name = "ShardedAegon::Publish",
            fields(num_updates = updates.len())
        )
    )]
    pub fn publish(
        &mut self,
        updates: &[(Label, Value)],
    ) -> Result<ShardedEpochCommitment<E, P>, AegonError> {
        let prof = super::instrument::publish_profile_enabled();
        let t_total = std::time::Instant::now();
        let t = std::time::Instant::now();
        Self::reject_duplicate_labels(updates)?;
        if prof {
            eprintln!(
                "[pub-profile] coord.reject_duplicate_labels: {:.3} ms (updates={})",
                t.elapsed().as_secs_f64() * 1000.0,
                updates.len()
            );
        }
        let t = std::time::Instant::now();
        let (sub_batches, new_placements) = self.plan_phase_1_batches(updates)?;
        if prof {
            eprintln!(
                "[pub-profile] coord.plan_phase_1_batches: {:.3} ms (new_placements={})",
                t.elapsed().as_secs_f64() * 1000.0,
                new_placements.len()
            );
        }
        let t = std::time::Instant::now();
        let (new_index_commits, new_value_commits) = self.run_phase_1(&sub_batches)?;
        if prof {
            eprintln!(
                "[pub-profile] coord.run_phase_1: {:.3} ms (n_shards={})",
                t.elapsed().as_secs_f64() * 1000.0,
                sub_batches.len()
            );
        }
        let t = std::time::Instant::now();
        let (new_r_index, new_r_value) =
            self.derive_chain_scalars(&new_index_commits, &new_value_commits);
        if prof {
            eprintln!(
                "[pub-profile] coord.derive_chain_scalars: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let t = std::time::Instant::now();
        let (per_shard_commits, per_shard_history) =
            self.run_phase_2(new_r_index, new_r_value)?;
        if prof {
            eprintln!(
                "[pub-profile] coord.run_phase_2: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let t = std::time::Instant::now();
        let sharded_commit = self.finalize_epoch(per_shard_commits, new_r_index, new_r_value);
        if prof {
            eprintln!(
                "[pub-profile] coord.finalize_epoch: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let t = std::time::Instant::now();
        self.persist_publish_to_db(
            updates,
            &new_placements,
            &sharded_commit,
            &per_shard_history,
        )?;
        if prof {
            eprintln!(
                "[pub-profile] coord.persist_publish_to_db: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0,
            );
            eprintln!(
                "[pub-profile] COORD_PUBLISH_TOTAL: {:.3} ms (updates={})",
                t_total.elapsed().as_secs_f64() * 1000.0,
                updates.len()
            );
        }
        Ok(sharded_commit)
    }

    /// O(n) scan for `label` appearing twice. Errors out the whole
    /// batch on the first collision so phase 1 never sees a malformed
    /// input.
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(level = "debug", skip_all, name = "ShardedAegon::RejectDuplicates")
    )]
    fn reject_duplicate_labels(updates: &[(Label, Value)]) -> Result<(), AegonError> {
        let mut seen: HashSet<&[u8]> = HashSet::with_capacity(updates.len());
        for (label, _) in updates {
            if !seen.insert(label.as_slice()) {
                return Err(AegonError::DuplicateLabel(label.clone()));
            }
        }
        Ok(())
    }

    /// Look up the routing entry for `label`. Source of truth depends
    /// on whether a DB is attached:
    ///   * DB present (production / bench): always read from RocksDB.
    ///     `self.routing` HashMap is never populated, so coord RSS
    ///     stays bounded as the dictionary grows. Cost: one DB get per
    ///     call (microseconds; RocksDB has bloom filters on existence
    ///     checks).
    ///   * DB absent (in-process tests): fall back to the in-memory
    ///     HashMap. At test scale this is a few hundred entries, so
    ///     keeping it in RAM is fine.
    fn read_routing(&self, label: &[u8]) -> Result<Option<LabelRouting>, AegonError> {
        if let Some(db) = self.db.as_ref() {
            let Some(bytes) = db.get(&key_routing(label))? else {
                return Ok(None);
            };
            let lr = LabelRouting::deserialize_compressed(&bytes[..]).map_err(|e| {
                AegonError::Database(format!("deserialize routing for {label:?}: {e}"))
            })?;
            Ok(Some(lr))
        } else {
            Ok(self.routing.get(label).cloned())
        }
    }

    /// Decide where every `(label, value)` write lands. Returns one
    /// sub-batch per shard, each entry shaped
    /// `ShardWrite { slot_bits, h_label, h_value }`:
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
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(level = "debug", skip_all, name = "ShardedAegon::PlanPhase1")
    )]
    fn plan_phase_1_batches(
        &mut self,
        updates: &[(Label, Value)],
    ) -> Result<(Vec<SubBatch<E::ScalarField>>, Vec<NewPlacement>), AegonError> {
        let n = self.shards.len();
        let mut sub_batches: Vec<SubBatch<E::ScalarField>> =
            (0..n).map(|_| Vec::new()).collect();
        let mut in_batch_claimed: Vec<HashSet<usize>> = (0..n).map(|_| HashSet::new()).collect();
        let mut new_placements: Vec<NewPlacement> = Vec::new();

        // Pass 1: route value-only updates (existing labels) directly,
        // and stash brand-new labels for round-based occupancy probing.
        // `update_idx` carries the original ordering so the in-batch
        // collision semantics match the sequential implementation.
        struct NewLabel<'a, F> {
            update_idx: usize,
            label: &'a [u8],
            h_label: F,
            h_value: F,
            ctr: u64,
            trail: Vec<(u32, Vec<bool>)>,
        }
        let mut new_labels: Vec<NewLabel<'_, E::ScalarField>> = Vec::new();

        for (idx, (label, value)) in updates.iter().enumerate() {
            let h_value = H::h_f(value);
            if let Some(routing) = self.read_routing(label)? {
                let (sid, slot_bits) = routing.final_assignment().clone();
                sub_batches[sid as usize].push(ShardWrite {
                    slot_bits,
                    h_label: None,
                    h_value,
                });
            } else {
                new_labels.push(NewLabel {
                    update_idx: idx,
                    label: label.as_slice(),
                    h_label: H::h_f(label),
                    h_value,
                    ctr: 0,
                    trail: Vec::new(),
                });
            }
        }

        // Pass 2 — assign new labels via open-addressing.
        //
        // Round-based: each round issues ONE pipelined occupancy
        // batch that covers every still-in-flight label at its
        // current probe ctr (one network round-trip per round, not
        // per probe). Then we resolve placements in original-input
        // order — necessary to preserve the sequential
        // implementation's "earlier label wins an in-batch
        // collision" semantics — and advance any losers to ctr+1
        // for the next round. At low load factor almost everyone
        // places on round 0 and the whole pass collapses to a
        // single round-trip.
        //
        // The coord's local DB is now the system-wide authority for
        // slot occupancy — both Redis and RocksDB backends go
        // through the same path. We always batch-query the local
        // DB; for any probes the DB reports as empty we fall back
        // to per-probe gRPC `is_index_slot_occupied` to the owning
        // shard. The fallback handles the bench-cluster scenario
        // where shards started with `--prefill-count N` (which
        // populates the shard's polynomial in memory but
        // deliberately does NOT write slot keys to the coord — see
        // server.rs `prefill_with_random`). In a clean production
        // deployment where every label arrives via a coord publish,
        // the fallback never fires because the coord's local DB
        // sees every placement.
        //
        // The DbSource::None branch falls back to per-label
        // `assign_trail` against the shard's in-memory occupancy
        // set (used by in-process tests).
        let total_capacity = 1u64 << self.log_capacity();
        if let Some(db) = self.db.as_ref() {
            let log_n_shards = self.log_n_shards;
            let shard_log_capacity = self.shard_log_capacity();
            let shard_dims = self.shard_dims.clone();
            while !new_labels.is_empty() {
                // Compute (shard_id, slot_bits, slot_idx, key) for each
                // in-flight label at its current ctr — pure CPU, no I/O.
                let probes: Vec<(u32, Vec<bool>, usize, Vec<u8>)> = new_labels
                    .iter()
                    .map(|nl| {
                        let (shard_id, slot_bits) = probe_at::<H, E::ScalarField>(
                            nl.ctr,
                            nl.label,
                            log_n_shards,
                            shard_log_capacity,
                        );
                        let slot_idx = bool_index_to_usize_dims(&slot_bits, &shard_dims);
                        let key = key_slot(shard_id, slot_idx);
                        (shard_id, slot_bits, slot_idx, key)
                    })
                    .collect();

                // One round-trip for the whole round's DB lookup.
                let keys: Vec<Vec<u8>> =
                    probes.iter().map(|(_, _, _, k)| k.clone()).collect();
                let mut occupied_prev = db.exists_many(&keys)?;
                // For any probe the coord's DB doesn't have, fall
                // back to asking the owning shard. This covers the
                // bench prefill case (shard's polynomial has the
                // slot, but the coord wasn't told). In a fresh-
                // cluster bench the DB starts empty, so this fallback
                // fires for *every* probe — at batch=16384 the
                // serial-RPC loop costs ~8 s per publish. Each call
                // is a blocking gRPC RTT (~2 ms intra-VPC), so we want
                // many more concurrent threads than the coord's CPU
                // count to amortize the RTT. The default rayon pool
                // matches CPU count (4 on n2-standard-4 → still 8 s
                // wall). A dedicated 64-thread pool brings it to
                // ~0.5 s.
                let shards = &self.shards;
                let fallback_pool = FALLBACK_POOL.get_or_init(|| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(64)
                        .thread_name(|i| format!("aegon-occ-fallback-{i}"))
                        .build()
                        .expect("build fallback rayon pool")
                });
                let fallback_updates: Vec<usize> = fallback_pool.install(|| {
                    probes
                        .par_iter()
                        .enumerate()
                        .filter_map(|(i, (shard_id, slot_bits, _slot_idx, _key))| {
                            if occupied_prev[i] {
                                return None;
                            }
                            if shards[*shard_id as usize]
                                .is_index_slot_occupied(slot_bits)
                            {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect()
                });
                for i in fallback_updates {
                    occupied_prev[i] = true;
                }

                // Resolve in input order so in-batch collisions are
                // broken consistently with the sequential reference.
                let mut next_round: Vec<NewLabel<'_, E::ScalarField>> = Vec::new();
                let drained: Vec<NewLabel<'_, E::ScalarField>> =
                    std::mem::take(&mut new_labels);
                // Pair each new label with its probe + DB answer,
                // then sort by original update index. Sort is stable on
                // small Vecs (rayon not needed) and almost always a
                // no-op on round 0.
                let mut zipped: Vec<(
                    NewLabel<'_, E::ScalarField>,
                    (u32, Vec<bool>, usize, Vec<u8>),
                    bool,
                )> = drained
                    .into_iter()
                    .zip(probes.into_iter())
                    .zip(occupied_prev.into_iter())
                    .map(|((nl, probe), occ)| (nl, probe, occ))
                    .collect();
                zipped.sort_by_key(|(nl, _, _)| nl.update_idx);
                for (mut nl, (shard_id, slot_bits, slot_idx, _key), occ_prev) in zipped {
                    nl.trail.push((shard_id, slot_bits.clone()));
                    let occupied_in_batch =
                        in_batch_claimed[shard_id as usize].contains(&slot_idx);
                    if !occ_prev && !occupied_in_batch {
                        in_batch_claimed[shard_id as usize].insert(slot_idx);
                        sub_batches[shard_id as usize].push(ShardWrite {
                            slot_bits,
                            h_label: Some(nl.h_label),
                            h_value: nl.h_value,
                        });
                        let trail = nl.trail;
                        new_placements.push(NewPlacement {
                            label: nl.label.to_vec(),
                            shard_id,
                            slot_idx,
                            trail: trail.clone(),
                        });
                        // Keep an in-memory copy ONLY when there's no DB
                        // (test-only path). In production / bench, the
                        // routing is persisted to RocksDB by
                        // `persist_publish_to_db` later in this call and
                        // looked up from there on future publishes,
                        // keeping coord RSS bounded.
                        if self.db.is_none() {
                            self.routing
                                .insert(nl.label.to_vec(), LabelRouting { trail });
                        }
                    } else {
                        nl.ctr += 1;
                        if nl.ctr >= total_capacity {
                            return Err(AegonError::DictionaryFull {
                                capacity: total_capacity as usize,
                            });
                        }
                        next_round.push(nl);
                    }
                }
                new_labels = next_round;
            }
        } else {
            // No-DB fallback: per-label sequential probing against the
            // shard's in-memory occupancy set, matching the original
            // implementation. Round-based pipelining doesn't apply here
            // since there's no network round-trip to amortise.
            for nl in new_labels {
                let trail = self.assign_trail(nl.label, &mut in_batch_claimed)?;
                let (sid, slot_bits) = trail.final_assignment().clone();
                let slot_idx = bool_index_to_usize_dims(&slot_bits, &self.shard_dims);
                sub_batches[sid as usize].push(ShardWrite {
                    slot_bits,
                    h_label: Some(nl.h_label),
                    h_value: nl.h_value,
                });
                let trail_for_placement = trail.trail.clone();
                new_placements.push(NewPlacement {
                    label: nl.label.to_vec(),
                    shard_id: sid,
                    slot_idx,
                    trail: trail_for_placement,
                });
                // No-DB fallback path: this branch never has a DB to
                // persist routing to, so we MUST keep the in-memory
                // map. With DB, the analogous insert in the round-based
                // path above is gated behind `self.db.is_none()`.
                self.routing.insert(nl.label.to_vec(), trail);
            }
        }
        Ok((sub_batches, new_placements))
    }

    /// Drive every shard's `publish_phase_1_at_slots` in parallel and
    /// transpose the per-shard `(index_com, value_com)` pairs into
    /// two shard-id-ordered vectors. Shards are independent (separate
    /// polynomials + state), so rayon's data-parallel pattern is
    /// safe; only the read-only prover_param is shared.
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(level = "debug", skip_all, name = "ShardedAegon::RunPhase1")
    )]
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
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(
            level = "debug",
            skip_all,
            name = "ShardedAegon::DeriveChainScalars"
        )
    )]
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
    /// HistoryOpenings)` pairs into shard-id-ordered vectors. The
    /// auditor recovers everything it needs from the per-shard
    /// commitments via the commitment-homomorphism check; the
    /// `HistoryOpenings` vec carries the §6.4 history witnesses (one
    /// per shard, possibly empty).
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(level = "debug", skip_all, name = "ShardedAegon::RunPhase2")
    )]
    fn run_phase_2(
        &mut self,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
    ) -> Result<(Vec<EpochCommitment<E, P>>, Vec<HistoryOpenings<E, P>>), AegonError> {
        let phase_2: Vec<(EpochCommitment<E, P>, HistoryOpenings<E, P>)> = self
            .shards
            .par_iter_mut()
            .map(|shard| shard.publish_phase_2(new_r_index, new_r_value))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(phase_2.into_iter().unzip())
    }

    /// Coordinator-side bookkeeping: advance `(r_index, r_value,
    /// epoch)`, build the Merkle root over the per-shard commits, and
    /// append the new `ShardedEpochCommitment` to the history. Returns
    /// the commitment the caller will publish externally.
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(level = "debug", skip_all, name = "ShardedAegon::FinalizeEpoch")
    )]
    fn finalize_epoch(
        &mut self,
        per_shard_commits: Vec<EpochCommitment<E, P>>,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
    ) -> ShardedEpochCommitment<E, P> {
        self.r_index = new_r_index;
        self.r_value = new_r_value;
        self.epoch += 1;
        let sharded_commit =
            ShardedEpochCommitment::<E, P>::with_per_shard(self.epoch, per_shard_commits);
        self.epoch_commits.push(sharded_commit.clone());
        sharded_commit
    }

    /// Durability barrier for one publish: write every key the
    /// coordinator (and a future restarted coordinator) needs to
    /// reconstruct its view, atomically via the `Db` trait's
    /// `write_atomic` (Redis MULTI/EXEC or RocksDB WriteBatch
    /// depending on the configured backend).
    ///
    /// One `aegon:value:{label}` per update (value-update or new).
    /// For brand-new labels, also one `aegon:routing:{label}` and one
    /// `aegon:slot:{shard}:{slot}`. Finally, two coordinator-global
    /// keys: `aegon:coord:state` (epoch + FS scalars) and
    /// `aegon:coord:epoch_commit:{epoch}` (the new sharded epoch
    /// commitment).
    ///
    /// (Historically also wrote `SADD aegon:labels {label}` per
    /// update — pure overhead since nothing in the repo ever reads
    /// that set, and the per-batch 16K writes to a single key kept
    /// triggering RocksDB compaction stalls under load.)
    ///
    /// The polynomial commitment binds `H_F(value)` at the right slot
    /// already — the DB is only the side-channel that lets `lookup`
    /// return the value alongside the proof, and the durability layer
    /// for crash recovery; the verifier still re-hashes everything it
    /// receives. No-op when no DB was configured (`DbSource::None`).
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(level = "debug", skip_all, name = "ShardedAegon::PersistToDb")
    )]
    fn persist_publish_to_db(
        &self,
        updates: &[(Label, Value)],
        new_placements: &[NewPlacement],
        sharded_commit: &ShardedEpochCommitment<E, P>,
        per_shard_history: &[HistoryOpenings<E, P>],
    ) -> Result<(), AegonError> {
        let Some(db) = &self.db else { return Ok(()) };
        let prof = super::instrument::publish_profile_enabled();
        let _persist_t_total = std::time::Instant::now();

        // Pre-size: 1 op per update (value SET) + 2 ops per new
        // placement (routing SET + slot SET) + 2 global ops
        // (coord:state SET + coord:epoch_commit:{epoch} SET) + 1 op
        // per non-empty shard's §6.4 history bundle.
        let non_empty_histories = per_shard_history.iter().filter(|h| !h.entries.is_empty()).count();
        let mut ops: Vec<DbOp> = Vec::with_capacity(
            updates.len() + new_placements.len() * 2 + 2 + non_empty_histories,
        );

        // 1. value:{label} for every update.
        let _step1_t = std::time::Instant::now();
        for (label, value) in updates {
            ops.push(DbOp::Set {
                key: key_value(label),
                value: value.clone(),
            });
        }

        if prof {
            eprintln!(
                "[pub-profile] persist.step1_value_set: {:.3} ms (ops={})",
                _step1_t.elapsed().as_secs_f64() * 1000.0,
                updates.len(),
            );
        }

        // 2. routing:{label} + slot:{shard}:{slot} for new placements.
        //    Trail comes from the `NewPlacement` struct itself — no
        //    lookup against `self.routing` (which is empty when DB is
        //    present; the whole point of carrying the trail in
        //    `NewPlacement` is to avoid that read).
        let _step2_t = std::time::Instant::now();
        for placement in new_placements {
            let routing = LabelRouting {
                trail: placement.trail.clone(),
            };
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

        if prof {
            eprintln!(
                "[pub-profile] persist.step2_routing_and_slot_serialize: {:.3} ms (placements={})",
                _step2_t.elapsed().as_secs_f64() * 1000.0,
                new_placements.len(),
            );
        }

        // 3. coord:state — one key, contains (epoch, r_index, r_value).
        let _step3_t = std::time::Instant::now();
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

        if prof {
            eprintln!(
                "[pub-profile] persist.step3_coord_state_serialize: {:.3} ms",
                _step3_t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        // 4. coord:epoch_commit:{epoch} — the externally-published commitment.
        let _step4_t = std::time::Instant::now();
        let mut commit_bytes = Vec::new();
        sharded_commit
            .serialize_compressed(&mut commit_bytes)
            .map_err(|e| AegonError::Database(format!("serialize epoch commit: {e}")))?;
        ops.push(DbOp::Set {
            key: key_epoch_commit(sharded_commit.epoch),
            value: commit_bytes,
        });

        if prof {
            eprintln!(
                "[pub-profile] persist.step4_epoch_commit_serialize: {:.3} ms",
                _step4_t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        // 5. openings:{epoch}:{shard_id} — §6.4 history witnesses. One
        // key per shard whose batch carried at least one brand-new
        // label. Shards that did only value-updates produced an empty
        // `HistoryOpenings.entries`; persisting an empty bundle would
        // just waste a DB write, so those are skipped here. The
        // value_changes side may still be non-empty for those shards
        // (handled separately below as user-facing per-label history).
        let _step5_t = std::time::Instant::now();
        // Parallel: serialize each shard's HistoryOpenings on its own
        // rayon worker. At batch=16K the per-shard payload is ~410 ms
        // of compressed-serialize work, so going parallel cuts wall
        // time roughly in half for n_shards=2 and scales linearly.
        let step5_ops: Vec<DbOp> = per_shard_history
            .par_iter()
            .enumerate()
            .filter(|(_, h)| !h.entries.is_empty())
            .map(|(shard_id, history)| -> Result<DbOp, AegonError> {
                let mut history_bytes = Vec::new();
                history
                    .serialize_compressed(&mut history_bytes)
                    .map_err(|e| AegonError::Database(format!("serialize history openings: {e}")))?;
                Ok(DbOp::Set {
                    key: key_history_openings(sharded_commit.epoch, shard_id as u32),
                    value: history_bytes,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        ops.extend(step5_ops);

        if prof {
            eprintln!(
                "[pub-profile] persist.step5_history_openings_serialize_compressed: {:.3} ms (shards_with_history={})",
                _step5_t.elapsed().as_secs_f64() * 1000.0,
                per_shard_history.iter().filter(|h| !h.entries.is_empty()).count(),
            );
        }

        let _step6_t = std::time::Instant::now();
        // 6. value_history:{label} — user-facing value-history sliding
        // window. For every slot whose value changed this publish
        // (across all shards), build a `StoredValueHistoryEntry` and
        // LPUSH it onto that label's list, then LTRIM to keep at most
        // `HISTORY_WINDOW` entries. Both ops sit inside the same
        // MULTI/EXEC, so a concurrent reader sees either the full
        // pre-publish or the full post-publish list — never a torn
        // state.
        //
        // Mapping value_changes back to the label that owned the slot.
        // For brand-new placements in this batch, the trail is in
        // `new_placements`. For value-only updates (existing labels),
        // the trail is in the DB (or `self.routing` for the no-DB
        // test path). We pre-build a `(shard_id, slot_bits) -> label`
        // map so the inner loop is O(1).
        let prev_commit = if sharded_commit.epoch == 0 {
            None
        } else {
            self.epoch_commits.get((sharded_commit.epoch - 1) as usize).cloned()
        };
        let mut slot_to_label: std::collections::HashMap<(u32, Vec<bool>), Label> =
            std::collections::HashMap::with_capacity(updates.len());
        // First: register new placements from this batch.
        let new_placement_labels: std::collections::HashSet<&[u8]> = new_placements
            .iter()
            .map(|p| p.label.as_slice())
            .collect();
        for placement in new_placements {
            if let Some((sid, sbits)) = placement.trail.last() {
                slot_to_label.insert((*sid, sbits.clone()), placement.label.clone());
            }
        }
        // Then: value-only updates — look up in DB (or no-DB fallback).
        for (label, _value) in updates {
            if new_placement_labels.contains(label.as_slice()) {
                continue;
            }
            if let Some(routing) = self.read_routing(label)? {
                let (sid, sbits) = routing.final_assignment();
                slot_to_label.insert((*sid, sbits.clone()), label.clone());
            }
        }
        // Also build a (label -> value bytes) map so each entry can
        // carry the raw value_bytes the user later hashes against
        // value_post_eval. Updates list is already that map — just
        // address it by label.
        let mut label_to_value: std::collections::HashMap<&[u8], &[u8]> =
            std::collections::HashMap::with_capacity(updates.len());
        for (label, value) in updates {
            label_to_value.insert(label.as_slice(), value.as_slice());
        }
        // Parallel: flatten all (shard_id, vc) pairs and serialize
        // their value-history entries on rayon. At batch=16K the
        // serialize_uncompressed wall is ~400 ms even though each
        // entry is only ~25 µs — the loop is dominated by serializing
        // ~25 G1 points per entry. par_iter gets the work down to
        // ~50 ms on a 4-vCPU coord.
        let step6_inputs: Vec<(usize, &ValueChangeEntry<E, P>)> = per_shard_history
            .iter()
            .enumerate()
            .filter(|(_, h)| !h.value_changes.is_empty() && prev_commit.is_some())
            .flat_map(|(shard_id, h)| {
                h.value_changes.iter().map(move |vc| (shard_id, vc))
            })
            .collect();
        let step6_ops: Vec<DbOp> = step6_inputs
            .par_iter()
            .map(|(shard_id, vc)| -> Result<[DbOp; 2], AegonError> {
                let prev_sharded = prev_commit
                    .as_ref()
                    .expect("filter above ensures prev_commit is Some");
                let prev_leaf = prev_sharded.per_shard[*shard_id].clone();
                let prev_merkle_path = prev_sharded.merkle_path(*shard_id).to_vec();
                let post_leaf = sharded_commit.per_shard[*shard_id].clone();
                let post_merkle_path = sharded_commit.merkle_path(*shard_id).to_vec();
                let Some(label) = slot_to_label.get(&(*shard_id as u32, vc.slot_bits.clone())) else {
                    return Err(AegonError::Database(format!(
                        "internal: value_change at shard {shard_id} slot {:?} has no matching label in this publish's updates",
                        vc.slot_bits
                    )));
                };
                let value_bytes = label_to_value.get(label.as_slice()).copied().unwrap_or(&[]);
                let entry = StoredValueHistoryEntry::<E, P> {
                    epoch: sharded_commit.epoch,
                    shard_id: *shard_id as u32,
                    slot_bits: vc.slot_bits.clone(),
                    value_bytes: value_bytes.to_vec(),
                    rand_value_pre_eval: vc.rand_value_pre_eval,
                    rand_value_pre_proof: vc.rand_value_pre_proof.clone(),
                    rand_value_post_eval: vc.rand_value_post_eval,
                    rand_value_post_proof: vc.rand_value_post_proof.clone(),
                    value_post_eval: vc.value_post_eval,
                    value_post_proof: vc.value_post_proof.clone(),
                    prev_shard_commit: prev_leaf,
                    prev_merkle_path,
                    post_shard_commit: post_leaf,
                    post_merkle_path,
                };
                let mut entry_bytes = Vec::new();
                // Uncompressed on purpose: each entry is ~30 G1Affine
                // points, and compressed reads pay a Tonelli-Shanks
                // sqrt per point on the lookup_history path
                // (~25 µs/point ≈ 1 ms/entry, dominating that RPC).
                // Uncompressed roughly doubles per-entry DB bytes
                // (~15 KB → ~30 KB at production proof shapes) but
                // cuts deserialize cost ~10×. The gRPC wire to the
                // client still uses compressed encoding — only DB
                // storage changes. See the matching
                // `deserialize_uncompressed_unchecked` in
                // `lookup_history`.
                entry
                    .serialize_uncompressed(&mut entry_bytes)
                    .map_err(|e| AegonError::Database(format!("serialize value history entry: {e}")))?;
                let history_key = key_value_history(label);
                Ok([
                    DbOp::LPush {
                        key: history_key.clone(),
                        member: entry_bytes,
                    },
                    DbOp::LTrim {
                        key: history_key,
                        start: 0,
                        stop: (HISTORY_WINDOW as isize) - 1,
                    },
                ])
            })
            .collect::<Result<Vec<[DbOp; 2]>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        ops.extend(step6_ops);

        if prof {
            eprintln!(
                "[pub-profile] persist.step6_value_history_serialize_uncompressed: {:.3} ms",
                _step6_t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let _step7_t = std::time::Instant::now();
        // 7. label_placement:{label} — exactly one record per
        // newly-placed label. Asymmetric with value_history: labels
        // are placed once and never mutate, so we use a single Set
        // (not LPush/LTrim) and we extract the openings from the
        // §6.4 `entries` bundle (already produced by
        // publish_phase_2) rather than building fresh openings —
        // every placement already paid for a `rand_index_post` open
        // in the §6.4 path. We just lift it into the user-facing
        // placement record + add the anchoring shard commit/path.
        //
        // Pre-build a `(shard_id, slot_idx) -> &HistoryOpeningEntry`
        // map once — the inner lookup is then O(1). The naive
        // alternative (linear scan over `per_shard_history[shard_id].entries`
        // per placement, recomputing `bool_index_to_usize_dims`) is
        // O(placements^2) per shard and dominated publish time at
        // batch=16384 (~6.5 s of 8.7 s persist).
        let mut shard_slot_to_entry: HashMap<
            (u32, usize),
            &HistoryOpeningEntry<E, P>,
        > = HashMap::with_capacity(new_placements.len());
        for (shard_id, history) in per_shard_history.iter().enumerate() {
            for entry in &history.entries {
                let slot_idx = bool_index_to_usize_dims(&entry.slot_bits, &self.shard_dims);
                shard_slot_to_entry.insert((shard_id as u32, slot_idx), entry);
            }
        }
        let step7_ops: Vec<DbOp> = new_placements
            .par_iter()
            .map(|placement| -> Result<DbOp, AegonError> {
                let shard_id = placement.shard_id;
                let entry = shard_slot_to_entry
                    .get(&(shard_id, placement.slot_idx))
                    .copied()
                    .ok_or_else(|| {
                        AegonError::Database(format!(
                            "internal: no §6.4 history entry for placement at shard {shard_id} slot_idx {}",
                            placement.slot_idx
                        ))
                    })?;
                let post_leaf = sharded_commit.per_shard[shard_id as usize].clone();
                let post_merkle_path = sharded_commit.merkle_path(shard_id as usize).to_vec();
                let stored = StoredLabelPlacement::<E, P> {
                    epoch: sharded_commit.epoch,
                    shard_id,
                    slot_bits: entry.slot_bits.clone(),
                    rand_index_eval: entry.rand_index_post_eval,
                    rand_index_proof: entry.rand_index_post_proof.clone(),
                    placement_shard_commit: post_leaf,
                    placement_merkle_path: post_merkle_path,
                };
                let mut bytes = Vec::new();
                // Same uncompressed-on-disk rationale as the value-
                // history side: trades ~2× bytes for ~10× faster reads.
                stored
                    .serialize_uncompressed(&mut bytes)
                    .map_err(|e| AegonError::Database(format!("serialize label placement: {e}")))?;
                Ok(DbOp::Set {
                    key: key_label_placement(&placement.label),
                    value: bytes,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        ops.extend(step7_ops);

        if prof {
            eprintln!(
                "[pub-profile] persist.step7_label_placement_serialize_uncompressed: {:.3} ms (placements={})",
                _step7_t.elapsed().as_secs_f64() * 1000.0,
                new_placements.len(),
            );
        }

        let _write_t = std::time::Instant::now();
        let r = db.write_atomic(&ops);
        if prof {
            eprintln!(
                "[pub-profile] persist.db_write_atomic: {:.3} ms (ops={})",
                _write_t.elapsed().as_secs_f64() * 1000.0,
                ops.len(),
            );
            eprintln!(
                "[pub-profile] PERSIST_TOTAL: {:.3} ms",
                _persist_t_total.elapsed().as_secs_f64() * 1000.0,
            );
        }
        r
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
            // Occupancy check: when a DB is configured, ask the DB
            // (one EXISTS — no gRPC). Otherwise fall back to the shard
            // (in-process test path). The DB is authoritative once
            // it's configured because `persist_publish_to_db` writes
            // `aegon:slot:*` in the same atomic txn as the shard
            // commitments are finalized, so the two never disagree
            // unless we're mid-recovery.
            // Coord's local DB is authoritative for slots the coord
            // itself placed. For slots the coord doesn't know about
            // (bench-prefill case), fall back to the shard's own
            // in-memory occupancy. In production where every label
            // arrives via a coord publish, the fallback never fires.
            let occupied_prev = match self.db.as_ref() {
                Some(db) => {
                    db.exists(&key_slot(shard_id, slot_idx))?
                        || self.shards[shard_id as usize].is_index_slot_occupied(&slot_bits)
                },
                None => self.shards[shard_id as usize].is_index_slot_occupied(&slot_bits),
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

    /// First half of the split lookup design: prove that `label` is
    /// canonically placed at a specific `(shard_id, slot_bits)`.
    ///
    /// Walks the open-addressing trail in `self.routing`, opens
    /// `index_poly` at each probe against the corresponding shard's
    /// current commitment, and packages everything into a
    /// `ShardedLabelProof`. Returns the canonical `LabelSlot` so the
    /// caller can cache it for future `lookup_value` calls without
    /// re-proving residency.
    ///
    /// Does **not** touch `value_poly` and does **not** fetch any
    /// side-channel value bytes — that's `lookup_value`'s job.
    pub fn lookup_label(
        &self,
        label: &Label,
    ) -> Result<(LabelSlot, ShardedLabelProof<E, P>), AegonError> {
        let routing = self
            .read_routing(label)?
            .ok_or_else(|| AegonError::UnknownLabel(label.clone()))?;
        let current = self.current_commitment();

        let mut probes: Vec<ShardedProbe<E, P>> = Vec::with_capacity(routing.trail.len());
        let total_bits = self.log_n_shards + self.shard_log_capacity();
        for (ctr_us, (shard_id, slot_bits)) in routing.trail.iter().enumerate() {
            let (evaluation, proof) =
                self.shards[*shard_id as usize].open_index_at_slot(slot_bits)?;
            let leaf = current.per_shard[*shard_id as usize].clone();
            let merkle_path = current.merkle_path(*shard_id as usize).to_vec();
            // When the deployment runs with a VRF prover (EcVrfHash),
            // recompute the proof for this `(ctr, label)` pair so the
            // client can verify the slot bits independently. Cost is
            // ~150 µs per probe on modern x86; a typical α=4 trail of
            // ~3 probes adds ~0.5 ms to `lookup_label`. For the SHA-
            // 256 path the vector stays empty and the verifier falls
            // back to `H::h_bits`.
            let vrf_proof: Vec<u8> = if let Some(prover) = self.vrf_prover.as_ref() {
                let (_bits, proof_bytes) = prover.prove_h_bits(ctr_us as u64, label, total_bits);
                proof_bytes.to_vec()
            } else {
                Vec::new()
            };
            probes.push(ShardedProbe {
                shard_id: *shard_id,
                leaf,
                merkle_path,
                evaluation,
                proof,
                vrf_proof,
            });
        }

        let (final_shard, final_slot) = routing.final_assignment();
        let slot = LabelSlot {
            shard_id: *final_shard,
            slot_bits: final_slot.clone(),
        };
        Ok((
            slot,
            ShardedLabelProof {
                ctr0: routing.ctr0(),
                probes,
            },
        ))
    }

    /// Second half of the split lookup design: open `value_poly` at a
    /// cached `(shard, slot)` and return just the value-side proof.
    ///
    /// Inputs are unchecked at this layer — the caller is expected to
    /// have obtained `slot` from a verified `lookup_label` (or to be
    /// using it within the trust boundary, like the bench harness).
    /// Verifier-side validation lives in `verify_lookup_value`.
    ///
    /// No side-channel value bytes are fetched here either: the
    /// polynomial commitment binds `H_F(value)`, the raw bytes ride a
    /// separate channel, and the verifier hashes them locally.
    pub fn lookup_value(
        &self,
        slot: &LabelSlot,
    ) -> Result<ShardedValueProof<E, P>, AegonError> {
        if (slot.shard_id as usize) >= self.shards.len() {
            return Err(AegonError::Config(format!(
                "lookup_value: shard_id {} out of range (have {} shards)",
                slot.shard_id,
                self.shards.len()
            )));
        }
        let current = self.current_commitment();
        let leaf = current.per_shard[slot.shard_id as usize].clone();
        let merkle_path = current.merkle_path(slot.shard_id as usize).to_vec();
        let (evaluation, proof) =
            self.shards[slot.shard_id as usize].open_value_at_slot(&slot.slot_bits)?;
        Ok(ShardedValueProof {
            shard_id: slot.shard_id,
            slot_bits: slot.slot_bits.clone(),
            leaf,
            merkle_path,
            evaluation,
            proof,
        })
    }

    /// User-facing value-history fetch. Reads up to `HISTORY_WINDOW`
    /// most-recent `(epoch, value, openings)` records for `label` from
    /// the `aegon:value_history:{label}` DB list, decodes each
    /// entry, and returns them. Order is most-recent first (matches
    /// the underlying LPUSH+LRANGE 0 N-1 semantics).
    ///
    /// Pure read — no shard RPCs, no PCS work on the server side.
    /// Verification is the client's job (see `verify_lookup_history`).
    /// Returns an empty `entries` Vec when:
    ///   * `DbSource::None` is configured (no place to fetch from),
    ///   * the label exists but has never been published (no LPUSH
    ///     has run for it yet), or
    ///   * the label is unknown.
    /// Returning empty (rather than `UnknownLabel`) keeps the API
    /// resilient to races where a client asks for history right after
    /// a label was assigned but before the persist's MULTI/EXEC
    /// landed.
    pub fn lookup_history(
        &self,
        label: &Label,
    ) -> Result<ShardedValueHistory<E, P>, AegonError> {
        let Some(db) = &self.db else {
            return Ok(ShardedValueHistory {
                label: label.clone(),
                entries: Vec::new(),
                freshness: None,
            });
        };
        // Fetch the whole window in one round-trip. HISTORY_WINDOW is
        // small enough that LRANGE 0 -1 would also be fine — we cap
        // explicitly so a stale list that's somehow longer than the
        // window doesn't surprise the client.
        let raw_entries = db.lrange(
            &key_value_history(label),
            0,
            (HISTORY_WINDOW as isize) - 1,
        )?;
        // Parallel across the history window (up to HISTORY_WINDOW
        // entries). Each entry is decoded then sent to its owning
        // shard for remasking — three masking-server calls per entry,
        // themselves parallel internally. Both axes need to be
        // parallel: at HISTORY_WINDOW=5 the previous fully-sequential
        // path did 15 remasks back-to-back.
        //
        // Matches the `serialize_uncompressed` write side in
        // `persist_publish_to_db`. `*_unchecked` skips the
        // group-element subgroup check on each curve point — safe
        // here because we wrote these bytes ourselves at the most
        // recent publish_phase_2 and the entries never leave our
        // own DB until being returned to the verifier (who
        // re-verifies the openings cryptographically anyway).
        //
        // Stored proofs are PLAIN non-ZK (publish_phase_2 stores
        // them as such regardless of `--private`). Under a
        // non-hiding SRS that's the final form. Under a hiding
        // SRS we must mask them via the masking-server protocol
        // before shipping to the verifier — done below by
        // `remask_value_history_entry` on the owning shard
        // (which holds the per-epoch `tau_f` snapshots and has a
        // masking client or inline fallback to produce a
        // `MaskingPackage`). This is the "publish stores non-zk,
        // lookup masks" design — publish-time crypto is now
        // identical in `--private=true` and `--private=false`.
        let entries: Vec<StoredValueHistoryEntry<E, P>> = raw_entries
            .par_iter()
            .map(|bytes| -> Result<StoredValueHistoryEntry<E, P>, AegonError> {
                let entry =
                    StoredValueHistoryEntry::<E, P>::deserialize_uncompressed_unchecked(&bytes[..])
                        .map_err(|e| {
                            AegonError::Database(format!("decode value history entry: {e}"))
                        })?;
                let shard_id = entry.shard_id;
                if (shard_id as usize) >= self.shards.len() {
                    return Err(AegonError::Config(format!(
                        "lookup_history: entry's shard_id {shard_id} out of range \
                         (have {} shards)",
                        self.shards.len()
                    )));
                }
                self.shards[shard_id as usize].remask_value_history_entry(entry)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Freshness attestation: an opening of the LIVE rand_value
        // poly at the slot of the most recent entry, anchored under
        // the live sharded root. The shard's `rand_value` evaluation
        // at a slot is invariant under any publish that does not
        // touch that slot (the chain-blinding delta is zero
        // off-support), so a verifier seeing the same evaluation
        // here as in `entries[0].rand_value_post_eval` learns that
        // no publish has touched the slot since then. This is the
        // only field of the response that isn't a pure DB read —
        // it costs one shard gRPC round-trip + one PCS open.
        let freshness = if let Some(latest) = entries.first() {
            let shard_id = latest.shard_id;
            if (shard_id as usize) >= self.shards.len() {
                return Err(AegonError::Config(format!(
                    "lookup_history: latest entry's shard_id {shard_id} out of range \
                     (have {} shards)",
                    self.shards.len()
                )));
            }
            let current = self.current_commitment();
            let (eval, proof) = self.shards[shard_id as usize]
                .open_rand_value_at_slot_current(&latest.slot_bits)?;
            let merkle_path = current.merkle_path(shard_id as usize).to_vec();
            let shard_commit = current.per_shard[shard_id as usize].clone();
            Some(FreshnessAttestation {
                shard_id,
                slot_bits: latest.slot_bits.clone(),
                rand_value_current_eval: eval,
                rand_value_current_proof: proof,
                shard_commit,
                merkle_path,
            })
        } else {
            None
        };
        Ok(ShardedValueHistory {
            label: label.clone(),
            entries,
            freshness,
        })
    }

    /// User-facing label-history fetch. Label-side mirror of
    /// `lookup_history`. Returns the (single) placement record for
    /// `label` plus a freshly-computed freshness attestation. The
    /// verifier (`verify_lookup_label_history`) checks that:
    ///
    ///   * The stored placement opening verifies under
    ///     `placement_shard_commit.rand_index_commitment`.
    ///   * The live opening verifies under
    ///     `shard_commit.rand_index_commitment`.
    ///   * The two evaluations are equal — i.e. no publish has
    ///     written `index_poly` at this slot since the placement
    ///     epoch (rand_index is invariant off the touched slots
    ///     under chain-blinding).
    ///   * Both merkle paths re-anchor to roots the caller
    ///     separately trusts (the placement-epoch root and the
    ///     live root).
    ///
    /// Together with a separate `lookup_label(label)` (which proves
    /// "this label is at this slot in the live state"), the bundle
    /// proves "this label has been bound to this slot since
    /// `placement.epoch`".
    ///
    /// Returns `placement = None, freshness = None` when:
    ///   * `DbSource::None` is configured (no place to fetch from),
    ///   * the label exists but has never been published (placement
    ///     record has not landed yet — racy window), or
    ///   * the label is unknown.
    /// Returning empty (rather than `UnknownLabel`) keeps the API
    /// symmetric with `lookup_history`.
    pub fn lookup_label_history(
        &self,
        label: &Label,
    ) -> Result<ShardedLabelHistory<E, P>, AegonError> {
        let Some(db) = &self.db else {
            return Ok(ShardedLabelHistory {
                label: label.clone(),
                placement: None,
                freshness: None,
            });
        };
        let raw = db.get(&key_label_placement(label))?;
        let Some(bytes) = raw else {
            return Ok(ShardedLabelHistory {
                label: label.clone(),
                placement: None,
                freshness: None,
            });
        };
        // Matches the `serialize_uncompressed` write side in
        // `persist_publish_to_db`. See the parallel comment on
        // `lookup_history`'s decode loop for the `_unchecked`
        // rationale.
        let placement =
            StoredLabelPlacement::<E, P>::deserialize_uncompressed_unchecked(&bytes[..]).map_err(
                |e| AegonError::Database(format!("decode label placement: {e}")),
            )?;
        let shard_id = placement.shard_id;
        if (shard_id as usize) >= self.shards.len() {
            return Err(AegonError::Config(format!(
                "lookup_label_history: placement shard_id {shard_id} out of range \
                 (have {} shards)",
                self.shards.len()
            )));
        }
        // Live opening — the single piece of non-DB work this RPC
        // does. One shard gRPC + one PCS open of rand_index_poly.
        let current = self.current_commitment();
        let (eval, proof) = self.shards[shard_id as usize]
            .open_rand_index_at_slot_current(&placement.slot_bits)?;
        let merkle_path = current.merkle_path(shard_id as usize).to_vec();
        let shard_commit = current.per_shard[shard_id as usize].clone();
        let freshness = Some(FreshnessAttestationLabel {
            shard_id,
            slot_bits: placement.slot_bits.clone(),
            rand_index_current_eval: eval,
            rand_index_current_proof: proof,
            shard_commit,
            merkle_path,
        });
        Ok(ShardedLabelHistory {
            label: label.clone(),
            placement: Some(placement),
            freshness,
        })
    }

    /// Backward-compat wrapper that does both halves in one call and
    /// also pulls the raw value bytes from the coordinator's KV store
    /// (or returns an empty `Value` when `DbSource::None`). Composes
    /// `lookup_label` and `lookup_value` so the splits and the
    /// combined call always agree.
    ///
    /// New code should prefer the split methods directly: clients
    /// typically only need to look up a label once and want to call
    /// `lookup_value` many times against the cached slot.
    pub fn lookup(
        &self,
        label: &Label,
    ) -> Result<(Value, ShardedLookupProof<E, P>), AegonError> {
        let (slot, label_proof) = self.lookup_label(label)?;
        let value_proof = self.lookup_value(&slot)?;
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
                ctr0: label_proof.ctr0,
                probes: label_proof.probes,
                value_evaluation: value_proof.evaluation,
                value_proof: value_proof.proof,
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
                merkle_path_s0: s0_commit.merkle_path(*shard_id as usize).to_vec(),
                merkle_path_s1: s1_commit.merkle_path(*shard_id as usize).to_vec(),
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
            merkle_path_s0: s0_commit.merkle_path(*final_shard as usize).to_vec(),
            merkle_path_s1: s1_commit.merkle_path(*final_shard as usize).to_vec(),
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
/// Verify the open-addressing chain proves `label` lives at the
/// returned `LabelSlot`. This is the first half of the split lookup:
/// it does **not** touch `value_poly` or any value bytes — it only
/// confirms that the label is canonically placed where the server
/// claims, and surfaces that slot for the client to cache and reuse
/// across future `lookup_value` calls.
///
/// Returns the canonical `LabelSlot` (derived from re-running the
/// open-addressing trail against `H(ctr, label)` for the verified
/// `ctr0`) so the client doesn't have to trust the server for the
/// slot — it falls out of the verified chain.
pub fn verify_lookup_label<E, P, H>(
    ctx: &ShardedVerifierContext<E, P>,
    commit: &ShardedEpochCommitment<E, P>,
    label: &Label,
    proof: &ShardedLabelProof<E, P>,
) -> Result<LabelSlot, AegonError>
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
    let mut final_slot: Option<(u32, Vec<bool>)> = None;

    let total_bits = ctx.log_n_shards + ctx.shard_log_capacity();
    for (ctr_us, probe) in proof.probes.iter().enumerate() {
        let ctr = ctr_us as u64;
        // Recover (shard_id, slot_bits) for this probe. Two paths:
        //   * VRF deployment (`ctx.vrf_verifier == Some`): consume
        //     `probe.vrf_proof`, run `VRF.verify`, slice the VRF
        //     output into total_bits. This is the *only* way the
        //     client can compute these bits — the VRF secret lives
        //     on the server.
        //   * SHA-256 deployment: call `H::h_bits` locally. The hash
        //     is publicly computable, so no proof is needed.
        let (expected_shard, slot_bits) = if let Some(verifier) = ctx.vrf_verifier.as_ref() {
            if probe.vrf_proof.is_empty() {
                return Err(AegonError::Verification(
                    "verifier configured with VRF public key but probe.vrf_proof is empty",
                ));
            }
            let bits = verifier
                .verify_h_bits(ctr, label, &probe.vrf_proof, total_bits)
                .map_err(|e| {
                    // Pre-format the verification error so the
                    // returned static-str variant carries enough
                    // context for a debugger. Distinguishes "bytes
                    // didn't parse" from "proof rejected by the key".
                    match e {
                        super::hash::VrfVerifyError::Malformed(_) => AegonError::Verification(
                            "probe.vrf_proof failed to parse as an RFC 9381 ECVRF proof",
                        ),
                        super::hash::VrfVerifyError::InvalidProof(_) => AegonError::Verification(
                            "probe.vrf_proof did not verify under the deployment's VRF public key",
                        ),
                    }
                })?;
            // Split bits into (shard_id, slot_bits) using the same
            // little-endian convention as probe_at.
            let mut shard_id: u32 = 0;
            for (i, b) in bits[..ctx.log_n_shards].iter().enumerate() {
                if *b {
                    shard_id |= 1u32 << i;
                }
            }
            let slot_bits = bits[ctx.log_n_shards..].to_vec();
            (shard_id, slot_bits)
        } else {
            // Legacy SHA-256 path: re-derive (shard_id, slot_bits)
            // from H(ctr, label) — the server can't lie about which
            // slot any given ctr probes because anyone can hash.
            probe_at::<H, E::ScalarField>(
                ctr,
                label,
                ctx.log_n_shards,
                ctx.shard_log_capacity(),
            )
        };
        let _ = total_bits; // silence dead-let warning on the legacy branch.
        if expected_shard != probe.shard_id {
            return Err(AegonError::Verification(
                "probe shard_id does not match H(ctr, label)",
            ));
        }
        // Merkle anchor: this probe's leaf must hash up to `commit.merkle_root`.
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
            return Err(AegonError::Verification(
                "probe opening did not verify against index commitment",
            ));
        }
        // Open-addressing constraints (paper §6.1, Fig. 4):
        //   * earlier probes must be non-empty and not the label's own hash,
        //     otherwise the server could have stopped at a smaller `ctr`;
        //   * the final probe must hold exactly `H_F(label)`.
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
        } else {
            if probe.evaluation != h_label {
                return Err(AegonError::Verification(
                    "final probe slot does not hold H_F(label)",
                ));
            }
            final_slot = Some((probe.shard_id, slot_bits));
        }
    }

    let (shard_id, slot_bits) = final_slot.ok_or(AegonError::Verification(
        "verify_lookup_label: empty probe trail (ctr0 underflow)",
    ))?;
    Ok(LabelSlot { shard_id, slot_bits })
}

/// Verify the value at a cached `LabelSlot` opens to `value`'s hash.
///
/// The second half of the split lookup. Unlike `verify_lookup_label`,
/// this verifier does **not** know the label — it works purely on the
/// slot, the value bytes, and the value-side proof. That decoupling
/// is the whole point: a client caches a `LabelSlot` once and uses it
/// against value openings forever after, even if the deployment-level
/// label namespace changes shape.
///
/// `slot` must agree with the proof's `(shard_id, slot_bits)` — we
/// re-check that here to prevent a server (or a sloppy client) from
/// answering at a different slot than was requested.
///
/// The hash suite `H` is needed so the verifier can re-hash the
/// out-of-band value bytes and confirm they match the polynomial
/// commitment's bound evaluation `H_F(value)`.
pub fn verify_lookup_value<E, P, H>(
    ctx: &ShardedVerifierContext<E, P>,
    commit: &ShardedEpochCommitment<E, P>,
    slot: &LabelSlot,
    value: &Value,
    proof: &ShardedValueProof<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    // Mismatch between the requested slot and the proof's slot is a
    // hard error — protects against a server that opens at a different
    // slot than the client asked about.
    if proof.shard_id != slot.shard_id || proof.slot_bits != slot.slot_bits {
        return Err(AegonError::Verification(
            "value proof slot does not match requested slot",
        ));
    }
    // Merkle anchor: the proof's leaf must hash up to `commit.merkle_root`.
    let reconstructed = verify_merkle_path::<E, P>(
        &proof.leaf,
        proof.shard_id as usize,
        &proof.merkle_path,
    );
    if reconstructed != commit.merkle_root {
        return Err(AegonError::Verification(
            "value proof merkle path does not reconstruct epoch root",
        ));
    }
    // PCS opening against the shard's value commitment at the slot.
    let value_point = bool_index_to_point::<E::ScalarField>(&proof.slot_bits);
    let mut tr = IOPTranscript::<E::ScalarField>::new(b"aegon.value.open");
    let ok = P::verify(
        &ctx.inner.verifier_param,
        &proof.leaf.value_commitment,
        &value_point,
        &proof.evaluation,
        &proof.proof,
        &mut tr,
    )?;
    if !ok {
        return Ok(false);
    }
    // The polynomial commitment binds `H_F(value)`. The verifier
    // re-hashes the out-of-band value bytes and checks that the hash
    // matches the opened evaluation.
    let expected_value = H::h_f(value);
    if proof.evaluation != expected_value {
        return Err(AegonError::Verification(
            "value opening does not match H_F(value)",
        ));
    }
    Ok(true)
}

/// Verify a `ShardedValueHistory` bundle returned by
/// `ShardedAegon::lookup_history`. Each entry is checked independently:
///
///   1. Per-shard leaf `entry.prev_shard_commit` re-hashes up to a
///      sharded root via `entry.prev_merkle_path` (the "prev root").
///      Per-shard leaf `entry.post_shard_commit` re-hashes up to a
///      sharded root via `entry.post_merkle_path` (the "post root").
///      The caller is responsible for cross-checking these two roots
///      against whatever bulletin-board snapshot they trust for
///      epochs `entry.epoch - 1` and `entry.epoch`. This function
///      returns the reconstructed roots in the `Ok` path so the
///      caller can do that without re-verifying.
///   2. `rand_value_pre_proof` opens at `slot_bits` against
///      `entry.prev_shard_commit.rand_value_commitment` with value
///      `rand_value_pre_eval`.
///   3. `rand_value_post_proof` opens at `slot_bits` against
///      `entry.post_shard_commit.rand_value_commitment` with value
///      `rand_value_post_eval`.
///   4. `value_post_proof` opens at `slot_bits` against
///      `entry.post_shard_commit.value_commitment` with value
///      `value_post_eval`.
///   5. `H::h_f(entry.value_bytes) == entry.value_post_eval`.
///
/// `Ok(verified_anchors)` is a vector of `(prev_root, post_root)`
/// pairs parallel to `history.entries` — handy for the caller's
/// bulletin-board cross-check step. `Err` signals a per-entry failure
/// (caller decides whether to reject the whole bundle or keep going).
/// Output of [`verify_lookup_history`].
///
/// `entry_roots` reconstructs the sharded root at `(epoch-1, epoch)`
/// for each history entry — one tuple per `entries` slot, same order
/// (most recent first). `live_root`, when present, is the sharded
/// root reconstructed from the freshness attestation; pair it with
/// the coordinator's `current_commitment().sharded_root` to confirm
/// "this history is current as of right now".
#[derive(Clone, Debug)]
pub struct VerifiedLookupHistory {
    pub entry_roots: Vec<(EpochDigest, EpochDigest)>,
    pub live_root: Option<EpochDigest>,
}

pub fn verify_lookup_history<E, P, H>(
    ctx: &ShardedVerifierContext<E, P>,
    history: &ShardedValueHistory<E, P>,
) -> Result<VerifiedLookupHistory, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    let mut roots: Vec<(EpochDigest, EpochDigest)> = Vec::with_capacity(history.entries.len());
    for entry in &history.entries {
        // Re-anchor both leaves under their respective sharded roots.
        let prev_root = verify_merkle_path::<E, P>(
            &entry.prev_shard_commit,
            entry.shard_id as usize,
            &entry.prev_merkle_path,
        );
        let post_root = verify_merkle_path::<E, P>(
            &entry.post_shard_commit,
            entry.shard_id as usize,
            &entry.post_merkle_path,
        );

        // Per-slot point (same encoding as lookup_value).
        let point = bool_index_to_point::<E::ScalarField>(&entry.slot_bits);

        // rand_value_pre against prev rand_value commitment.
        let mut tr_pre = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_value.open");
        let ok_pre = P::verify(
            &ctx.inner.verifier_param,
            &entry.prev_shard_commit.rand_value_commitment,
            &point,
            &entry.rand_value_pre_eval,
            &entry.rand_value_pre_proof,
            &mut tr_pre,
        )?;
        if !ok_pre {
            return Err(AegonError::Verification(
                "history entry rand_value_pre opening did not verify",
            ));
        }

        // rand_value_post against post rand_value commitment.
        let mut tr_post = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_value.open");
        let ok_post = P::verify(
            &ctx.inner.verifier_param,
            &entry.post_shard_commit.rand_value_commitment,
            &point,
            &entry.rand_value_post_eval,
            &entry.rand_value_post_proof,
            &mut tr_post,
        )?;
        if !ok_post {
            return Err(AegonError::Verification(
                "history entry rand_value_post opening did not verify",
            ));
        }

        // value_post against post value commitment.
        let mut tr_val = IOPTranscript::<E::ScalarField>::new(b"aegon.value.open");
        let ok_val = P::verify(
            &ctx.inner.verifier_param,
            &entry.post_shard_commit.value_commitment,
            &point,
            &entry.value_post_eval,
            &entry.value_post_proof,
            &mut tr_val,
        )?;
        if !ok_val {
            return Err(AegonError::Verification(
                "history entry value_post opening did not verify",
            ));
        }

        // Value bytes ↔ value_post_eval. The polynomial commitment
        // binds H_F(value), so the verifier re-hashes whatever the
        // server delivered and rejects on mismatch.
        let expected_h = H::h_f(&entry.value_bytes);
        if expected_h != entry.value_post_eval {
            return Err(AegonError::Verification(
                "history entry value_bytes do not hash to value_post_eval",
            ));
        }
        roots.push((prev_root, post_root));
    }

    // Freshness attestation. Must be present iff there's at least
    // one history entry. Cross-checks the live `rand_value(slot)`
    // against the most recent entry's `rand_value_post_eval`.
    let live_root = match (&history.freshness, history.entries.first()) {
        (None, None) => None,
        (None, Some(_)) => {
            return Err(AegonError::Verification(
                "history has entries but no freshness attestation",
            ));
        },
        (Some(_), None) => {
            return Err(AegonError::Verification(
                "history has a freshness attestation but no entries",
            ));
        },
        (Some(fr), Some(latest)) => {
            if fr.shard_id != latest.shard_id || fr.slot_bits != latest.slot_bits {
                return Err(AegonError::Verification(
                    "freshness attestation references a different (shard, slot) than the latest entry",
                ));
            }
            let live_root = verify_merkle_path::<E, P>(
                &fr.shard_commit,
                fr.shard_id as usize,
                &fr.merkle_path,
            );
            // Live `rand_value(slot)` opens under the live shard commit.
            let point = bool_index_to_point::<E::ScalarField>(&fr.slot_bits);
            let mut tr_live = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_value.open");
            let ok_live = P::verify(
                &ctx.inner.verifier_param,
                &fr.shard_commit.rand_value_commitment,
                &point,
                &fr.rand_value_current_eval,
                &fr.rand_value_current_proof,
                &mut tr_live,
            )?;
            if !ok_live {
                return Err(AegonError::Verification(
                    "freshness rand_value opening did not verify",
                ));
            }
            // No-change-since: rand_value at this slot is invariant
            // under any publish that does not touch the slot. Equal
            // evaluations ⇒ no publish has touched this slot since
            // the most recent entry's epoch.
            if fr.rand_value_current_eval != latest.rand_value_post_eval {
                return Err(AegonError::Verification(
                    "freshness check failed: live rand_value differs from latest entry's post-update value — a subsequent publish modified this slot but was not recorded in the history window",
                ));
            }
            Some(live_root)
        },
    };

    Ok(VerifiedLookupHistory {
        entry_roots: roots,
        live_root,
    })
}

/// Output of [`verify_lookup_label_history`].
///
/// `placement_root` reconstructs the sharded root at
/// `placement.epoch` from the placement record's merkle path —
/// pair it with whatever the caller trusts as the bulletin-board
/// snapshot for that epoch. `live_root` is the analogous
/// reconstruction from the freshness attestation; pair it with the
/// coordinator's `current_commitment().merkle_root`.
///
/// Both are `None` exactly when the bundle is empty (no placement
/// record stored, e.g. label unknown or pre-publish race).
#[derive(Clone, Debug)]
pub struct VerifiedLookupLabelHistory {
    pub placement_root: Option<EpochDigest>,
    pub live_root: Option<EpochDigest>,
}

/// Verify a `ShardedLabelHistory` bundle returned by
/// `ShardedAegon::lookup_label_history`. Label-side mirror of
/// `verify_lookup_history`, but simpler:
///
///   1. The stored `rand_index` opening verifies under
///      `placement.placement_shard_commit.rand_index_commitment`.
///   2. The live `rand_index` opening verifies under
///      `freshness.shard_commit.rand_index_commitment`.
///   3. Both evaluations agree — "no publish has touched this slot
///      since the placement epoch". `rand_index` at a slot is
///      invariant under any publish whose `delta_index_poly` is
///      zero at that slot, so equality ⇒ no `index_poly` write at
///      this slot since placement.
///   4. Both merkle paths anchor against their respective sharded
///      roots, which we return for the caller's bulletin-board
///      cross-check.
///
/// Returns empty (all `None`) when the bundle has no placement
/// record (label unknown / racy pre-persist window).
pub fn verify_lookup_label_history<E, P, H>(
    ctx: &ShardedVerifierContext<E, P>,
    history: &ShardedLabelHistory<E, P>,
) -> Result<VerifiedLookupLabelHistory, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    match (&history.placement, &history.freshness) {
        (None, None) => Ok(VerifiedLookupLabelHistory {
            placement_root: None,
            live_root: None,
        }),
        (None, Some(_)) => Err(AegonError::Verification(
            "label history has a freshness attestation but no placement record",
        )),
        (Some(_), None) => Err(AegonError::Verification(
            "label history has a placement record but no freshness attestation",
        )),
        (Some(p), Some(fr)) => {
            if p.shard_id != fr.shard_id || p.slot_bits != fr.slot_bits {
                return Err(AegonError::Verification(
                    "label history freshness references a different (shard, slot) than the placement",
                ));
            }
            // Re-anchor both leaves under their respective roots.
            let placement_root = verify_merkle_path::<E, P>(
                &p.placement_shard_commit,
                p.shard_id as usize,
                &p.placement_merkle_path,
            );
            let live_root = verify_merkle_path::<E, P>(
                &fr.shard_commit,
                fr.shard_id as usize,
                &fr.merkle_path,
            );

            let point = bool_index_to_point::<E::ScalarField>(&p.slot_bits);

            // (1) Placement rand_index opening.
            let mut tr_placement = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_index.open");
            let ok_placement = P::verify(
                &ctx.inner.verifier_param,
                &p.placement_shard_commit.rand_index_commitment,
                &point,
                &p.rand_index_eval,
                &p.rand_index_proof,
                &mut tr_placement,
            )?;
            if !ok_placement {
                return Err(AegonError::Verification(
                    "label history placement rand_index opening did not verify",
                ));
            }

            // (2) Live rand_index opening.
            let mut tr_live = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_index.open");
            let ok_live = P::verify(
                &ctx.inner.verifier_param,
                &fr.shard_commit.rand_index_commitment,
                &point,
                &fr.rand_index_current_eval,
                &fr.rand_index_current_proof,
                &mut tr_live,
            )?;
            if !ok_live {
                return Err(AegonError::Verification(
                    "label history freshness rand_index opening did not verify",
                ));
            }

            // (3) No-change-since-placement: rand_index(slot) is
            // invariant under any publish that does not write
            // index_poly at the slot. Labels are placed exactly
            // once, so equality here is the cryptographic witness
            // that the label is still bound to this slot.
            if p.rand_index_eval != fr.rand_index_current_eval {
                return Err(AegonError::Verification(
                    "label history freshness check failed: live rand_index differs from placement rand_index — a publish has written index_poly at this slot since placement (label has been displaced or otherwise mutated)",
                ));
            }

            // Sanity: H is unused in this verifier (no value-bytes
            // hashing to do — `index_poly`'s data value is bound by
            // the lookup_label proof, not by us). Reference H to
            // keep the type parameter live for the call site.
            let _ = std::marker::PhantomData::<H>;

            Ok(VerifiedLookupLabelHistory {
                placement_root: Some(placement_root),
                live_root: Some(live_root),
            })
        },
    }
}

/// Backward-compat wrapper that verifies both halves of the original
/// combined `lookup` proof in one call. New code should call
/// `verify_lookup_label` + `verify_lookup_value` separately so a
/// client can stash the slot after the first label verification.
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
    // Split the combined proof into its two halves and run each
    // verifier. The value-side leaf is implicitly the final probe's
    // leaf (which is exactly what the splitter would have stored), so
    // we lift it out and rebuild a `ShardedValueProof` on the fly.
    let label_proof = ShardedLabelProof {
        ctr0: proof.ctr0,
        probes: proof.probes.clone(),
    };
    let slot = verify_lookup_label::<E, P, H>(ctx, commit, label, &label_proof)?;

    let final_probe = proof.probes.last().expect("ctr0 + 1 >= 1 probes");
    let value_proof = ShardedValueProof {
        shard_id: slot.shard_id,
        slot_bits: slot.slot_bits.clone(),
        leaf: final_probe.leaf.clone(),
        merkle_path: final_probe.merkle_path.clone(),
        evaluation: proof.value_evaluation,
        proof: proof.value_proof.clone(),
    };
    if !verify_lookup_value::<E, P, H>(ctx, commit, &slot, value, &value_proof)? {
        return Ok(false);
    }
    // `verify_lookup_value` already checks `H_F(value) == evaluation`,
    // so the combined wrapper has nothing left to add.
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

    // (1) Merkle root reconstructs from the announced per-shard commits.
    if merkle_root(&next.per_shard) != next.merkle_root {
        return Err(AegonError::Verification(
            "sharded audit: announced merkle_root does not match per_shard leaves",
        ));
    }

    // (2) Re-derive shared FS scalars from prev_r and the full per-shard tuple.
    let (new_r_index, new_r_value) =
        rederive_sharded_fs_scalars::<E, P>(audit_state.r_index, audit_state.r_value, next);

    // (3) Per-shard chain checks with the *shared* scalars. Every group
    // element the auditor needs is in `prev.per_shard[i]` /
    // `next.per_shard[i]`, and `verify_chain` does the homomorphism
    // check directly on commitments.
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
///
/// Prefer [`ShardedEpochCommitment::with_per_shard`] for new code —
/// it builds the root + every shard's sibling path in one pass and
/// caches the result. This standalone helper stays for the auditor /
/// verifier paths that only have `per_shard` in hand.
pub fn merkle_root<E: Pairing, P: AegonPcs<E>>(
    per_shard: &[EpochCommitment<E, P>],
) -> EpochDigest {
    build_merkle_root_and_paths::<E, P>(per_shard).0
}

/// Build the sibling-only Merkle path for `leaf_index` against
/// `per_shard`. Path length is `log2(per_shard.len())`.
///
/// Prefer [`ShardedEpochCommitment::merkle_path`] for new code — the
/// path is precomputed at commit construction. This helper rebuilds
/// the tree on every call and is kept only for verifier-side code
/// that operates on bare `&[EpochCommitment]` slices.
pub fn build_merkle_path<E: Pairing, P: AegonPcs<E>>(
    per_shard: &[EpochCommitment<E, P>],
    leaf_index: usize,
) -> Vec<EpochDigest> {
    let (_, mut paths) = build_merkle_root_and_paths::<E, P>(per_shard);
    paths
        .get_mut(leaf_index)
        .map(std::mem::take)
        .expect("leaf_index in range (guarded by power-of-two assertion)")
}

/// Build the Merkle root **and** every shard's sibling path in one
/// pass over the tree. `O(n_shards)` SHA256 ops total (each interior
/// digest computed once, each leaf hashed once), versus
/// `O(n_shards × log n_shards)` if [`build_merkle_path`] were called
/// once per shard.
///
/// Returns `(root, paths)` where `paths[i]` is the sibling-only path
/// from leaf `i` to the root. At `n_shards = 1`, `paths = vec![vec![]]`
/// (one shard, vacuous empty path) and `root = merkle_leaf(per_shard[0])`.
pub fn build_merkle_root_and_paths<E: Pairing, P: AegonPcs<E>>(
    per_shard: &[EpochCommitment<E, P>],
) -> (EpochDigest, Vec<Vec<EpochDigest>>) {
    assert!(
        per_shard.len().is_power_of_two(),
        "build_merkle_root_and_paths: leaf count must be a power of two (got {})",
        per_shard.len()
    );
    let n = per_shard.len();
    // Build every layer of the tree, bottom-up, retaining each layer
    // so we can index sibling digests when extracting paths.
    let mut layers: Vec<Vec<EpochDigest>> = Vec::with_capacity(n.trailing_zeros() as usize + 1);
    layers.push(per_shard.iter().map(merkle_leaf).collect());
    while layers.last().expect("at least one layer").len() > 1 {
        let prev = layers.last().expect("just pushed");
        let mut next: Vec<EpochDigest> = Vec::with_capacity(prev.len() / 2);
        for pair in prev.chunks_exact(2) {
            next.push(merkle_parent(&pair[0], &pair[1]));
        }
        layers.push(next);
    }
    let depth = layers.len() - 1; // path length = tree depth
    let root = layers
        .last()
        .expect("non-empty")
        .first()
        .copied()
        .expect("singleton root");

    // Walk each leaf upward, picking the sibling at every layer.
    let mut paths: Vec<Vec<EpochDigest>> = Vec::with_capacity(n);
    for leaf_idx in 0..n {
        let mut path: Vec<EpochDigest> = Vec::with_capacity(depth);
        let mut idx = leaf_idx;
        for d in 0..depth {
            let sibling = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
            path.push(layers[d][sibling]);
            idx /= 2;
        }
        paths.push(path);
    }
    (root, paths)
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
