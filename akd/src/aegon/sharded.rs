// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

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

use std::collections::HashSet;
use std::marker::PhantomData;

use akd_core::aegon_crypto::pcs::PCSGlobalParam;
use akd_core::aegon_crypto::transcript::IOPTranscript;
use ark_ec::pairing::Pairing;
use ark_ff::{Field, Zero};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::rand::Rng;
use rayon::prelude::*;

use sha2::{Digest, Sha256};

use super::audit::verify_chain;
use super::config::{AegonConfig, VerifierContext};
use super::db::{
    key_coord_state, key_epoch_commit, key_shard_fullness, Db, DbOp, DbSource, RedisDb,
};
use super::error::AegonError;
use super::hash::{bool_index_to_point, HashSuite, Sha256Hash};
use super::server::Aegon;
use super::types::{AegonPcs, EpochCommitment, Label, ShardedAuditState, Value};

/// The thread pool that shard fan-out runs on.
///
/// `ShardHandle` methods block the calling thread, and for
/// [`GrpcShardClient`](super::shard_grpc::GrpcShardClient) they block it on a
/// network round trip (`runtime.block_on`). Fanning those out across rayon's
/// *global* pool deadlocks whenever a shard server shares the process --- the
/// gRPC integration tests, and any single-box deployment. Every global worker
/// parks inside `block_on` waiting for a reply, while the server handling that
/// very request calls `rayon::join` and blocks waiting for a global worker to
/// come free. Nothing breaks the cycle.
///
/// How many cores you have decides whether you ever see it: a 2-core CI runner
/// hangs outright, a 12-core laptop has enough spare workers to hide it, and a
/// 16-core coordinator talking to 128 shards is safe only because the shards
/// are in other processes with pools of their own.
///
/// A separate pool breaks the cycle: the fan-out no longer occupies the global
/// pool, so shard-side compute always finds a worker there. The width matches
/// the global pool, so the number of shards in flight at once is unchanged.
fn shard_fanout_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads().max(1))
            .thread_name(|i| format!("aegon-shard-fanout-{i}"))
            .build()
            .expect("build the shard fan-out pool")
    })
}

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
#[derive(Clone, Debug, Default)]
pub enum ShardTransport {
    /// Single-process: shards co-located in the same address space.
    #[default]
    InProcess,
    /// Each shard is a remote `tonic`/gRPC service. `endpoints[i]`
    /// is the address of shard `i` (e.g. `"http://10.0.0.7:50051"`
    /// or `"https://aegon-shard-7.svc.cluster.local:50051"`). Length
    /// must equal `1 << log_n_shards`.
    Remote {
        /// Address of shard `i` at index `i`.
        endpoints: Vec<String>,
    },
}

/// Where the SRS / (prover_param, verifier_param) come from.
///
/// `DangerouslyGenerate` calls `P::gen_srs_for_testing` and is **not**
/// suitable for production — every Directory instance gets a fresh
/// (un-ceremonial) SRS. `Path` reads a previously-serialized SRS
/// from disk (the natural output of a trusted-setup ceremony).
/// `Path` is not yet wired up; using it returns
/// `AegonError::Config(...)` at setup time.
#[derive(Clone, Debug, Default)]
pub enum SrsSource {
    /// Generate a fresh SRS from `P::gen_srs_for_testing`. Test-only.
    #[default]
    DangerouslyGenerate,
    /// Load `(prover_param, verifier_param)` from a previously-
    /// serialized file (canonical SRS, typically a trusted-setup
    /// ceremony output). Not yet implemented.
    Path(std::path::PathBuf),
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
    /// Audit-path Fiat-Shamir derivations, propagated to every shard
    /// so their Schnorr challenges match the coordinator's chain
    /// scalars. Defaults to SHA256; an IVC-audited deployment
    /// installs the Poseidon bundle here and on every verifier.
    ///
    /// **In-process shards only.** With
    /// [`ShardTransport::Remote`] each `aegon_shard_server` builds
    /// its own [`AegonConfig`] from its own flags, so it will not
    /// inherit this setting — its Schnorr challenges would stay on
    /// SHA256 while the coordinator derives chain scalars with
    /// Poseidon, and every audit would fail. Wiring the selection
    /// through the shard-server CLI is still outstanding; until then
    /// IVC auditing is supported on
    /// [`ShardTransport::InProcess`] deployments.
    pub audit_fs: super::audit_fs::AuditFsHooks<E, P>,
    /// Number of independent Fiat-Shamir chains the audit runs. `1`
    /// (the default) derives one `(r_index, r_value)` pair per epoch
    /// from every shard's commitments, which is the behaviour every
    /// existing deployment has.
    ///
    /// A larger `G` partitions the shards into `G` contiguous groups
    /// (see [`GroupPlan`](super::chain_groups::GroupPlan)), each with
    /// its own rolling accumulator. The groups never interact, so a
    /// recursive auditor can fold and compress them in parallel — the
    /// compression step is linear in circuit size, so `G` groups cost
    /// `1/G` each. Must divide `n_shards` exactly.
    ///
    /// Every verifier in the deployment must be configured with the
    /// same value; a mismatch makes every audit fail.
    pub chain_groups: usize,
    /// Ties the config to its pairing without storing one.
    pub _e: PhantomData<E>,
}

impl<E: Pairing, P: AegonPcs<E>> ShardedAegonConfig<E, P> {
    /// Start a fluent builder. Required fields are
    /// `shard_log_capacity`, `log_n_shards`, and either `pcs_config`
    /// (generic) or — for the KZH-k backend — `kzh_k`.
    pub fn builder() -> ShardedAegonConfigBuilder<E, P> {
        ShardedAegonConfigBuilder::new()
    }

    /// Number of shards, `1 << log_n_shards`.
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
        let srs = P::gen_srs_for_testing(self.pcs_config.clone(), rng, self.shard_log_capacity)?;
        let (pk, vk) = P::trim(&srs, None, Some(self.shard_log_capacity))?;

        let file = std::fs::File::create(path).map_err(|e| {
            AegonError::Config(format!("create srs file '{}': {e}", path.display()))
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
        pk.serialize_uncompressed(&mut writer)
            .map_err(|e| AegonError::Config(format!("serialize prover_param: {e}")))?;
        vk.serialize_uncompressed(&mut writer)
            .map_err(|e| AegonError::Config(format!("serialize verifier_param: {e}")))?;
        use std::io::Write;
        writer
            .flush()
            .map_err(|e| AegonError::Config(format!("flush srs file: {e}")))?;
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
    let file = std::fs::File::open(path)
        .map_err(|e| AegonError::Config(format!("open srs file '{}': {e}", path.display())))?;
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
    let pk = P::ProverParam::deserialize_uncompressed_unchecked(&mut reader)
        .map_err(|e| AegonError::Config(format!("deserialize prover_param: {e}")))?;
    let vk = P::VerifierParam::deserialize_uncompressed_unchecked(&mut reader)
        .map_err(|e| AegonError::Config(format!("deserialize verifier_param: {e}")))?;
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
    audit_fs: super::audit_fs::AuditFsHooks<E, P>,
    chain_groups: usize,
    _e: PhantomData<E>,
}

impl<E: Pairing, P: AegonPcs<E>> Default for ShardedAegonConfigBuilder<E, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Pairing, P: AegonPcs<E>> ShardedAegonConfigBuilder<E, P> {
    /// An empty builder with every field unset.
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
            audit_fs: super::audit_fs::AuditFsHooks::sha256(),
            chain_groups: 1,
            _e: PhantomData,
        }
    }

    /// Split the audit's Fiat-Shamir chain into `g` independent
    /// groups. Defaults to `1`. Must divide `n_shards` exactly. See
    /// [`ShardedAegonConfig::chain_groups`].
    pub fn chain_groups(mut self, g: usize) -> Self {
        self.chain_groups = g;
        self
    }

    /// Install a different audit-path Fiat-Shamir bundle. Every
    /// verifier in the deployment must be configured to match.
    pub fn audit_fs(mut self, hooks: super::audit_fs::AuditFsHooks<E, P>) -> Self {
        self.audit_fs = hooks;
        self
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

    /// Validate and finalize. Errors when a required field is missing
    /// or when `chain_groups` does not divide the shard count exactly.
    pub fn build(self) -> Result<ShardedAegonConfig<E, P>, AegonError> {
        let shard_log_capacity = self.shard_log_capacity.ok_or_else(|| {
            AegonError::Config("ShardedAegonConfig: shard_log_capacity is required".into())
        })?;
        let log_n_shards = self.log_n_shards.ok_or_else(|| {
            AegonError::Config("ShardedAegonConfig: log_n_shards is required".into())
        })?;
        // Reject an unusable partition here rather than at the first
        // publish: an audit that derives per-group scalars the
        // verifier cannot reconstruct fails silently and late.
        super::chain_groups::GroupPlan::new(1usize << log_n_shards, self.chain_groups)?;
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
            audit_fs: self.audit_fs,
            chain_groups: self.chain_groups,
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

/// Coordinator state recovered from the DB on restart. Built by
/// `ShardedAegon::try_recover_from_db` and consumed in `setup`.
struct RecoveredState<E: Pairing, P: AegonPcs<E>> {
    epoch: u64,
    /// One rolling accumulator per chain group. Length is the
    /// deployment's `chain_groups`; `1` for the default single-chain
    /// configuration.
    r_index: Vec<E::ScalarField>,
    r_value: Vec<E::ScalarField>,
    epoch_commits: Vec<ShardedEpochCommitment<E, P>>,
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
    /// Epoch this commitment describes.
    pub epoch: u64,
    /// Merkle root over the per-shard leaves; the dictionary commitment.
    pub merkle_root: EpochDigest,
    /// One commitment per shard, indexed by shard id.
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
        self.per_shard.serialize_with_mode(&mut writer, compress)?;
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
        let merkle_root = <EpochDigest>::deserialize_with_mode(&mut reader, compress, validate)?;
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

/// Where in the cluster a label has been canonically placed —
/// `(shard, per-shard slot bits)`. Returned by `lookup_label` and
/// reconstructed by `verify_lookup_label`. The client caches this
/// once and uses it for any number of subsequent `lookup_value` calls
/// without having to re-prove residency.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize, PartialEq, Eq)]
pub struct LabelSlot {
    /// Which shard owns this slot.
    pub shard_id: u32,
    /// Slot address within that shard, low bit first.
    pub slot_bits: Vec<bool>,
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
    /// Which shard owns this slot.
    pub shard_id: u32,
    /// Slot address within that shard, low bit first.
    pub slot_bits: Vec<bool>,
    /// The owning shard's per-shard commitment (the Merkle leaf).
    pub leaf: EpochCommitment<E, P>,
    /// Sibling path from `leaf` up to the sharded root.
    pub merkle_path: Vec<EpochDigest>,
    /// `value(slot)` at the resolved slot.
    pub evaluation: E::ScalarField,
    /// Opening proof for `evaluation`.
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
    /// Opening proof for `rand_value_pre_eval` against the prior epoch.
    pub rand_value_pre_proof: P::Proof,
    /// `rand_value_{n+1}(slot)` at the new-epoch rand_value commitment.
    pub rand_value_post_eval: E::ScalarField,
    /// Opening proof for `rand_value_post_eval` against the new epoch.
    pub rand_value_post_proof: P::Proof,
    /// `value_{n+1}(slot) = H_F(value)` at the new-epoch value
    /// commitment.
    pub value_post_eval: E::ScalarField,
    /// Opening proof for `value_post_eval` against the new epoch.
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
    /// The label these entries belong to.
    pub label: Vec<u8>,
    /// Most recent first, at most `HISTORY_WINDOW` entries.
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
    /// Opening proof for the rand_index evaluation above.
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
    /// The label this placement record belongs to.
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
    /// Which shard owns this slot.
    pub shard_id: u32,
    /// Slot address within that shard, low bit first.
    pub slot_bits: Vec<bool>,
    /// `rand_index_live(slot)` — the live shard's rand_index poly
    /// evaluated at the slot.
    pub rand_index_current_eval: E::ScalarField,
    /// Opening proof for the current-epoch rand_index evaluation.
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
    /// Opening proof for the current-epoch rand_value evaluation.
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

// ===================== two-layer routing proofs =====================
//
// The two-layer routing model replaces the old single cross-shard
// open-addressing trail with TWO trails:
//
//   * inter-shard trail (`H_shard`): for each `shard_ctr` until the
//     routing lands on a non-full shard, one `ShardRoutingProbe`
//     carrying the VRF proof + (for intermediate ctrs) the shard's
//     fullness proof.
//   * intra-shard trail (`H_slot`): within the destination shard,
//     one `ShardSlotProbe` per `slot_ctr` carrying the VRF proof +
//     PCS opening of `index_poly` at the probed slot.
//
// The destination shard's leaf + merkle path are factored out into a
// single (`dest_leaf`, `dest_merkle_path`) pair because every probe in
// the intra-shard trail anchors against the same shard.

/// One inter-shard `H_shard(shard_ctr, label) → shard_id` probe.
///
/// Carries the VRF proof so the verifier can re-derive `shard_id`
/// without the VRF secret. For intermediate probes (the shard was
/// full and routing skipped past it), `fullness_proof` is `Some(_)` —
/// at this revision the bytes are a placeholder (empty); the real
/// soundness proof drops in here without a struct change. For the
/// final probe (routing landed) `fullness_proof` is `None`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardRoutingProbe {
    /// Shard `H_shard(ctr, label)` lands on at this probe.
    pub shard_id: u32,
    /// RFC 9381 ECVRF proof for `(shard_ctr, label)`, length
    /// `VRF_PROOF_BYTES` when the deployment uses `EcVrfHash`; empty
    /// when the deployment uses `Sha256Hash` (the verifier re-derives
    /// the bits locally via `H::h_shard`).
    pub vrf_proof: Vec<u8>,
    /// Fullness proof bytes, `Some(_)` only for intermediate probes.
    /// Placeholder (`Some(vec![])`) at this revision — the real proof
    /// fits in here later without a wire format change.
    pub fullness_proof: Option<Vec<u8>>,
}

/// One intra-shard `H_slot(slot_ctr, label) → slot_bits` probe inside
/// the destination shard. All probes anchor against the same shard
/// leaf + Merkle path; only the slot bits + opening change per probe.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardSlotProbe<E: Pairing, P: AegonPcs<E>> {
    /// Slot bits within the destination shard at this probe. Empty
    /// when `shard_log_capacity = 0` (degenerate single-slot shard).
    pub slot_bits: Vec<bool>,
    /// VRF proof for `(slot_ctr, label)`. Same encoding rule as
    /// `ShardRoutingProbe::vrf_proof`.
    pub vrf_proof: Vec<u8>,
    /// `index_poly(slot_bits)` opened against
    /// `dest_leaf.index_commitment`.
    pub evaluation: E::ScalarField,
    /// Opening proof for the evaluation above.
    pub proof: P::Proof,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for ShardSlotProbe<E, P> {
    fn clone(&self) -> Self {
        Self {
            slot_bits: self.slot_bits.clone(),
            vrf_proof: self.vrf_proof.clone(),
            evaluation: self.evaluation,
            proof: self.proof.clone(),
        }
    }
}

/// One step in the shard's intra-shard `H_slot` probe trail, packaged
/// with its index-polynomial opening. The shard emits this directly
/// from a single call to [`ShardHandle::fetch_label_proof_trail`] —
/// no separate `OpenIndexAtSlot` round-trips per probe.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LabelProofTrailEntry<E: Pairing, P: AegonPcs<E>> {
    /// Slot bits at this probe (equal to `H_slot(ctr, label)`, where
    /// the entry's position in [`LabelProofTrail::entries`] is `ctr`).
    /// The coord re-derives `ctr` from the position and uses VRF to
    /// authenticate the bits.
    pub slot_bits: Vec<bool>,
    /// `index_poly` evaluation at this slot.
    pub evaluation: E::ScalarField,
    /// Opening proof for the evaluation against the shard's current
    /// index commitment.
    pub proof: P::Proof,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for LabelProofTrailEntry<E, P> {
    fn clone(&self) -> Self {
        Self {
            slot_bits: self.slot_bits.clone(),
            evaluation: self.evaluation,
            proof: self.proof.clone(),
        }
    }
}

/// Combined shard response for the intra-shard half of
/// `lookup_label_two_layer`. The shard walks `H_slot(slot_ctr, label)`
/// once and emits the entire trail (including the index opening at
/// every probed slot) in a single call. Replaces the
/// `find_label_slot` + N×`open_index_at_slot` chain.
///
///   * `entries.len() == slot_ctr0 + 1`. The final entry's `slot_bits`
///     equals `final_slot_bits` (sanity-check on the coord side).
///   * Returned only when the label was found. The wire RPC sets
///     `found = false` for the not-in-this-shard case, which the coord
///     surfaces as `UnknownLabel`.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LabelProofTrail<E: Pairing, P: AegonPcs<E>> {
    /// Slot bits at the final (landing) probe.
    pub final_slot_bits: Vec<bool>,
    /// The `H_slot` counter at the final probe.
    pub slot_ctr0: u64,
    /// One entry per probe in `0..=slot_ctr0`, ordered by counter.
    pub entries: Vec<LabelProofTrailEntry<E, P>>,
    /// Precomputed VRF proofs for the inter-shard `H_shard` chain,
    /// indexed by ctr 0 first; `vrf_proofs_shard.len() ==
    /// final_shard_ctr + 1`. Populated by `publish_batch` (which
    /// receives them from the coord during routing). Empty when no
    /// `vrf_prover` is configured cluster-wide — in that case the
    /// coord recomputes the proofs at lookup time as before.
    pub vrf_proofs_shard: Vec<Vec<u8>>,
    /// Precomputed VRF proofs for the intra-shard `H_slot` chain,
    /// indexed by slot_ctr 0 first; `vrf_proofs_slot.len() ==
    /// slot_ctr0 + 1`. Populated by `publish_batch` on the owning
    /// shard. Empty when no `vrf_prover` is configured.
    pub vrf_proofs_slot: Vec<Vec<u8>>,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for LabelProofTrail<E, P> {
    fn clone(&self) -> Self {
        Self {
            final_slot_bits: self.final_slot_bits.clone(),
            slot_ctr0: self.slot_ctr0,
            entries: self.entries.clone(),
            vrf_proofs_shard: self.vrf_proofs_shard.clone(),
            vrf_proofs_slot: self.vrf_proofs_slot.clone(),
        }
    }
}

/// Combined shard response for `ShardedAegon::lookup_history`. The
/// shard reads its value-history sliding window for `label`, remasks
/// every entry, and opens the live `rand_value_poly` at the latest
/// entry's slot — all under one read lock and emitted in a single
/// gRPC RPC. Replaces the
/// `fetch_value_history` + N×`remask_value_history_entry` +
/// `open_rand_value_at_slot_current` chain (≈7 RTTs at
/// `HISTORY_WINDOW = 5`).
///
/// The shard fills in everything it can locally:
///   * `entries[i].shard_id`, `slot_bits`, `value_bytes`, and the four
///     evaluation/proof fields (remasked under live ZK if applicable),
///   * `entries[i].prev_shard_commit`, `post_shard_commit`,
///   * `freshness_eval` + `freshness_proof` (when entries non-empty).
/// The coord still owns the cross-shard merkle paths
/// (`prev_merkle_path` / `post_merkle_path` on each entry; freshness
/// `merkle_path` + `shard_commit`) — it stitches those in from its
/// in-memory `epoch_commits` cache before returning to the caller.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct FullValueHistory<E: Pairing, P: AegonPcs<E>> {
    /// Most-recent first (matches the LPush+LRange order on disk).
    /// Each entry already has remasking applied — the coord does not
    /// re-call `remask_value_history_entry`.
    pub entries: Vec<StoredValueHistoryEntry<E, P>>,
    /// Live `rand_value_poly(slot)` evaluation at `entries[0].slot_bits`.
    /// `None` iff `entries.is_empty()`.
    pub freshness_eval: Option<E::ScalarField>,
    /// Companion opening proof for `freshness_eval` against the
    /// shard's live `rand_value_commitment`.
    pub freshness_proof: Option<P::Proof>,
}

/// Combined shard response for `ShardedAegon::lookup_label_history`.
/// The shard reads its placement record for `label` and opens the
/// live `rand_index_poly` at the slot in the same call — collapsing
/// the prior `fetch_label_placement` + `open_rand_index_at_slot_current`
/// chain into ONE gRPC round-trip.
///
/// The placement carries an EMPTY `placement_merkle_path` (the shard
/// has no view of the cross-shard root); the coord stitches it in
/// from its in-memory `epoch_commits` cache before returning to the
/// caller.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct FullLabelHistory<E: Pairing, P: AegonPcs<E>> {
    /// Placement record for `label`. The wire-level `found = false`
    /// case is handled outside this struct (coord returns
    /// `placement = None` / `freshness = None` to the caller); when
    /// this struct exists, the placement is always populated.
    pub placement: StoredLabelPlacement<E, P>,
    /// Live `rand_index_poly(slot)` evaluation at
    /// `placement.slot_bits`.
    pub freshness_eval: E::ScalarField,
    /// Companion opening proof against the shard's live
    /// `rand_index_commitment`.
    pub freshness_proof: P::Proof,
}

/// Two-layer label-residency proof. Replaces `ShardedLabelProof` in
/// the new routing model.
///
///   * `route` walks `H_shard` from `shard_ctr = 0` until landing on a
///     non-full shard. All intermediate probes carry a `fullness_proof`;
///     the final probe (`route.last()`) does not. `route.len() ==
///     final_shard_ctr + 1` and `route.last().shard_id == dest_shard_id`.
///   * `dest_leaf` + `dest_merkle_path` anchor the destination shard
///     under the published epoch root. Every slot probe verifies
///     against `dest_leaf.index_commitment`.
///   * `slots` walks `H_slot` inside `dest_shard_id` from `slot_ctr = 0`
///     until landing on the slot that holds `H_F(label)`. Intermediate
///     probes must hold a non-zero evaluation that is not `H_F(label)`;
///     the final probe must hold exactly `H_F(label)`.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedLabelProofTwoLayer<E: Pairing, P: AegonPcs<E>> {
    /// Inter-shard routing probes, ctr 0 first.
    pub route: Vec<ShardRoutingProbe>,
    /// Shard the routing walk landed on.
    pub dest_shard_id: u32,
    /// That shard's per-shard commitment (the Merkle leaf).
    pub dest_leaf: EpochCommitment<E, P>,
    /// Sibling path from `dest_leaf` up to the sharded root.
    pub dest_merkle_path: Vec<EpochDigest>,
    /// Intra-shard slot probes on the destination shard.
    pub slots: Vec<ShardSlotProbe<E, P>>,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for ShardedLabelProofTwoLayer<E, P> {
    fn clone(&self) -> Self {
        Self {
            route: self.route.clone(),
            dest_shard_id: self.dest_shard_id,
            dest_leaf: self.dest_leaf.clone(),
            dest_merkle_path: self.dest_merkle_path.clone(),
            slots: self.slots.clone(),
        }
    }
}

/// Combined two-layer lookup proof: label residency + value opening
/// at the canonical slot. Mirrors `ShardedLookupProof` for the
/// two-layer routing model.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedLookupProofTwoLayer<E: Pairing, P: AegonPcs<E>> {
    /// Residency half: which slot this label owns.
    pub label_proof: ShardedLabelProofTwoLayer<E, P>,
    /// `value_poly(final_slot_bits)` opened against
    /// `dest_leaf.value_commitment`.
    pub value_evaluation: E::ScalarField,
    /// Opening proof for the value evaluation.
    pub value_proof: P::Proof,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for ShardedLookupProofTwoLayer<E, P> {
    fn clone(&self) -> Self {
        Self {
            label_proof: self.label_proof.clone(),
            value_evaluation: self.value_evaluation,
            value_proof: self.value_proof.clone(),
        }
    }
}

/// One intra-shard slot probe in a two-layer consistency proof:
/// `rand_index` opened at the same `(slot_bits)` at both `s0` and
/// `s1`. The verifier checks both openings AND that the evaluations
/// agree (which is what attests "this slot's rand_index didn't
/// change between epochs").
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardSlotRandPair<E: Pairing, P: AegonPcs<E>> {
    /// Slot bits derived from `H_slot(slot_ctr, label)`.
    pub slot_bits: Vec<bool>,
    /// RFC 9381 ECVRF proof for `(slot_ctr, label)`, empty in
    /// SHA-256 deployments.
    pub vrf_proof: Vec<u8>,
    /// `rand_index(slot)` at the earlier epoch `s0`.
    pub rand_index_s0_eval: E::ScalarField,
    /// Opening proof for `rand_index_s0_eval`.
    pub rand_index_s0_proof: P::Proof,
    /// `rand_index(slot)` at the later epoch `s1`.
    pub rand_index_s1_eval: E::ScalarField,
    /// Opening proof for `rand_index_s1_eval`.
    pub rand_index_s1_proof: P::Proof,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for ShardSlotRandPair<E, P> {
    fn clone(&self) -> Self {
        Self {
            slot_bits: self.slot_bits.clone(),
            vrf_proof: self.vrf_proof.clone(),
            rand_index_s0_eval: self.rand_index_s0_eval,
            rand_index_s0_proof: self.rand_index_s0_proof.clone(),
            rand_index_s1_eval: self.rand_index_s1_eval,
            rand_index_s1_proof: self.rand_index_s1_proof.clone(),
        }
    }
}

/// Two-layer consistency proof: attests that `label` was at the same
/// `(shard, slot)` at epochs `s0` and `s1`, that no other publish
/// disturbed any slot in the H_slot trail in between, and that the
/// value at the final slot wasn't updated.
///
/// Shape:
///   - `route` is the same inter-shard `H_shard` trail emitted by
///     `lookup_label_two_layer` against the current epoch (`s1`).
///   - `dest_leaf_s0` / `dest_leaf_s1` (+ merkle paths) anchor the
///     destination shard's leaf under each epoch's sharded root.
///   - `slots` carries one `ShardSlotRandPair` per intra-shard probe.
///   - `value_rand_*` opens `rand_value` at the final slot at both
///     epochs (its s0 == s1 equality attests "value unchanged").
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ShardedConsistencyProofTwoLayer<E: Pairing, P: AegonPcs<E>> {
    /// Inter-shard routing probes, ctr 0 first.
    pub route: Vec<ShardRoutingProbe>,
    /// Shard the routing walk landed on.
    pub dest_shard_id: u32,
    /// Destination shard's commitment at the earlier epoch `s0`.
    pub dest_leaf_s0: EpochCommitment<E, P>,
    /// Destination shard's commitment at the later epoch `s1`.
    pub dest_leaf_s1: EpochCommitment<E, P>,
    /// Sibling path from `dest_leaf_s0` up to the `s0` root.
    pub dest_merkle_path_s0: Vec<EpochDigest>,
    /// Sibling path from `dest_leaf_s1` up to the `s1` root.
    pub dest_merkle_path_s1: Vec<EpochDigest>,
    /// Per-slot rand_index opening pairs along the probe walk.
    pub slots: Vec<ShardSlotRandPair<E, P>>,
    /// `rand_value(slot)` at the earlier epoch `s0`.
    pub value_rand_s0_eval: E::ScalarField,
    /// Opening proof for `value_rand_s0_eval`.
    pub value_rand_s0_proof: P::Proof,
    /// `rand_value(slot)` at the later epoch `s1`.
    pub value_rand_s1_eval: E::ScalarField,
    /// Opening proof for `value_rand_s1_eval`.
    pub value_rand_s1_proof: P::Proof,
}

impl<E: Pairing, P: AegonPcs<E>> Clone for ShardedConsistencyProofTwoLayer<E, P> {
    fn clone(&self) -> Self {
        Self {
            route: self.route.clone(),
            dest_shard_id: self.dest_shard_id,
            dest_leaf_s0: self.dest_leaf_s0.clone(),
            dest_leaf_s1: self.dest_leaf_s1.clone(),
            dest_merkle_path_s0: self.dest_merkle_path_s0.clone(),
            dest_merkle_path_s1: self.dest_merkle_path_s1.clone(),
            slots: self.slots.clone(),
            value_rand_s0_eval: self.value_rand_s0_eval,
            value_rand_s0_proof: self.value_rand_s0_proof.clone(),
            value_rand_s1_eval: self.value_rand_s1_eval,
            value_rand_s1_proof: self.value_rand_s1_proof.clone(),
        }
    }
}

/// Verifier-side bundle, including the deployment's `log_n_shards`
/// (clients need it to recompute the probe trail) and, when the
/// deployment uses a VRF for index assignment, the verifier
/// configured with the server's published public key.
#[derive(Clone)]
pub struct ShardedVerifierContext<E: Pairing, P: AegonPcs<E>> {
    /// `inner.log_capacity` is the *per-shard* log_capacity.
    pub inner: VerifierContext<E, P>,
    /// Log2 of the shard count.
    pub log_n_shards: usize,
    /// `Some(verifier)` when the deployment runs with `EcVrfHash`
    /// (RFC 9381 ECVRF) and `verify_lookup_label` should consume
    /// each probe's `vrf_proof`. `None` for the SHA-256 path, in
    /// which case slot bits are re-derived locally via `H::h_bits`.
    pub vrf_verifier: Option<super::hash::VrfVerifier>,
    /// Number of independent Fiat-Shamir chains the audit runs. Must
    /// match the server's
    /// [`ShardedAegonConfig::chain_groups`]. Defaults to `1`.
    pub chain_groups: usize,
}

impl<E: Pairing, P: AegonPcs<E>> ShardedVerifierContext<E, P> {
    /// Build a sharded verifier context over a single shard's context.
    pub fn new(inner: VerifierContext<E, P>, log_n_shards: usize) -> Self {
        Self {
            inner,
            log_n_shards,
            vrf_verifier: None,
            chain_groups: 1,
        }
    }

    /// Track `g` independent chain-scalar accumulators instead of
    /// one. Must match the server's
    /// [`ShardedAegonConfig::chain_groups`] exactly — a mismatch
    /// makes every audit fail, since the verifier would absorb a
    /// different set of commitments than the server did.
    pub fn with_chain_groups(mut self, g: usize) -> Self {
        self.chain_groups = g;
        self
    }

    /// The shard partition this context audits against.
    pub fn group_plan(&self) -> Result<super::chain_groups::GroupPlan, AegonError> {
        super::chain_groups::GroupPlan::new(1usize << self.log_n_shards, self.chain_groups)
    }

    /// Attach a [`VrfVerifier`](super::hash::VrfVerifier) so subsequent
    /// `verify_lookup_label` calls consume the `vrf_proof` field on
    /// each probe instead of re-deriving bits via `H::h_bits`. Used
    /// when the deployment runs with `EcVrfHash` on the server side.
    pub fn with_vrf_verifier(mut self, verifier: super::hash::VrfVerifier) -> Self {
        self.vrf_verifier = Some(verifier);
        self
    }

    /// Log2 of one shard's slot count.
    pub fn shard_log_capacity(&self) -> usize {
        self.inner.log_capacity
    }

    /// Log2 of the whole dictionary's slot count.
    pub fn total_log_capacity(&self) -> usize {
        self.inner.log_capacity + self.log_n_shards
    }
}

// ---------- ShardedAegon -----------------------------------------------

/// Coordinator that owns `n_shards` independent [`Aegon`] instances.
pub struct ShardedAegon<E, P, H = Sha256Hash>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    shards: Vec<Box<dyn super::shard_grpc::ShardHandle<E, P, H>>>,
    // Derivable from the shard config; nothing reads it.
    #[allow(dead_code)]
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
    // hashing prev_r with the new data commits of the shards in its
    // group -- ALL of them when `chain_groups == 1`, which is the
    // default and what every pre-existing deployment does.
    r_index: Vec<E::ScalarField>,
    r_value: Vec<E::ScalarField>,
    // How the shard set is partitioned into independent chains.
    chain_groups: usize,

    // Coordinator's view of past epochs.
    epoch_commits: Vec<ShardedEpochCommitment<E, P>>,

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

    /// Two-layer routing's fullness map: index `i` is `Some(proof_bytes)`
    /// iff shard `i` has reported back as full. Length is `N_shards`.
    /// The coord consults this map at publish time to skip past full
    /// shards via `H_shard(VRF, shard_ctr+1, label)`; at lookup time
    /// the same map drives the same skip rule so the verifier walks
    /// an identical shard-routing trail.
    ///
    /// `proof_bytes` is opaque to the coord — at this revision it's
    /// always an empty `Vec<u8>` placeholder. The real per-shard
    /// soundness proof of fullness drops in here later without any
    /// wire-format change (the gRPC carries arbitrary bytes).
    ///
    /// Persisted at `aegon:coord:shard_fullness` so a coord restart
    /// recovers the map exactly. The map is `O(N_shards)`, never
    /// scales with the label count, so this is the *entire*
    /// label-count-independent slice of coord state added by the
    /// two-layer refactor.
    shard_full_proofs: Vec<Option<Vec<u8>>>,
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
        + std::ops::Sub<Output = P::Commitment>
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
    pub fn setup<R: Rng>(rng: &mut R, config: &ShardedAegonConfig<E, P>) -> Result<Self, AegonError>
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
            // Every shard must derive its Schnorr challenges the same
            // way the coordinator derives the chain scalars, or the
            // audit equation will not close.
            audit_fs: config.audit_fs,
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
                    }
                    SrsSource::Path(path) => read_srs_from_file::<E, P>(path)?,
                };
                let dims = P::block_dims(&prover_param, shard_config.log_capacity);
                let vctx = VerifierContext::new(shard_config.log_capacity, verifier_param.clone());
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
                    }
                    _ => {
                        let pool = super::masking::MaskingClientPool::<E, P>::connect_all(
                            &config.masking_addrs,
                        )?;
                        Some(std::sync::Arc::new(pool))
                    }
                };
                let mut shards: Vec<Box<dyn super::shard_grpc::ShardHandle<E, P, H>>> =
                    Vec::with_capacity(n_shards);
                for i in 0..n_shards {
                    let mut aegon = Aegon::<E, P, H>::init(
                        prover_param.clone(),
                        verifier_param.clone(),
                        &shard_config,
                    )?;
                    if let Some(src) = &masking_source {
                        aegon.set_masking_source(std::sync::Arc::clone(src));
                    }
                    // Give each in-process shard its own store, so an
                    // in-process deployment persists what a gRPC one does
                    // (raw values, the value-history window, placements).
                    // Without this the shard's write-sink discards and its
                    // reads return empty -- silently, as `Ok`.
                    //
                    // Each shard needs a *separate* store, not a shared one:
                    // `key_history_openings_local(epoch)` is
                    // `aegon:openings:{epoch}` with no shard discriminator,
                    // because on a real deployment the store is already
                    // shard-local. N shards sharing one keyspace would
                    // overwrite each other's openings every epoch.
                    match &config.db {
                        DbSource::None => {}
                        DbSource::Rocks(path) => {
                            // Sibling of the coordinator's directory rather
                            // than a child, so nothing nests inside a live
                            // RocksDB directory.
                            let mut shard_path = path.clone();
                            let leaf = shard_path
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_else(|| "aegon".to_string());
                            shard_path.set_file_name(format!("{leaf}.shards"));
                            shard_path.push(format!("{i}"));
                            aegon.set_db(Box::new(crate::aegon::db::RocksDb::open(&shard_path)?));
                        }
                        DbSource::Redis(_) => {
                            return Err(AegonError::Config(format!(
                                "DbSource::Redis is not supported with \
                                 ShardTransport::InProcess ({n_shards} shards would share one \
                                 keyspace, and per-epoch history openings are stored without a \
                                 shard discriminator, so shards would overwrite each other). \
                                 Use DbSource::Rocks, which gives each in-process shard its own \
                                 store, or ShardTransport::Remote, where every shard owns its DB."
                            )));
                        }
                    }
                    shards.push(Box::new(aegon));
                }
                (shards, dims, vctx)
            }
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
                let dims =
                    P::block_dims_from_verifier_param(&verifier_param, shard_config.log_capacity);
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
            }
        };

        let initial_per_shard: Vec<EpochCommitment<E, P>> =
            shards.iter().map(|s| s.current_commitment()).collect();
        let initial_commit = ShardedEpochCommitment::<E, P>::with_per_shard(0, initial_per_shard);

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

        // Fullness map: same length as shards, all `None` at boot.
        // Try to restore the persisted map; missing key (fresh DB or
        // pre-two-layer state) means "no shards reported full yet."
        let shard_full_proofs = match &db {
            Some(db_inner) => Self::try_recover_shard_full_proofs(&**db_inner, shards.len())?
                .unwrap_or_else(|| vec![None; shards.len()]),
            None => vec![None; shards.len()],
        };

        // The verifier context is the single source of truth for the
        // audit-path FS derivations: `derive_chain_scalars` reads it
        // at publish time and `sharded_verifier_context()` hands the
        // same bundle to auditors, so the two can never drift.
        let mut shard_verifier_context = shard_verifier_context;
        shard_verifier_context.audit_fs = config.audit_fs;

        let chain_groups = config.chain_groups;
        if let Some(rec) = recovered {
            // A deployment that changed `chain_groups` between runs
            // would silently start deriving scalars a verifier cannot
            // reproduce, so refuse rather than resume.
            if rec.r_index.len() != chain_groups || rec.r_value.len() != chain_groups {
                return Err(AegonError::Config(format!(
                    "recovered coordinator state has {} chain groups but this configuration \
                     declares {chain_groups}; the Fiat-Shamir chain cannot be resumed across \
                     a change to chain_groups",
                    rec.r_index.len()
                )));
            }
            Ok(Self {
                shards,
                shard_dims,
                shard_verifier_context,
                shard_log_capacity_cached: shard_config.log_capacity,
                log_n_shards: config.log_n_shards,
                epoch: rec.epoch,
                r_index: rec.r_index,
                r_value: rec.r_value,
                chain_groups,
                epoch_commits: rec.epoch_commits,
                db,
                vrf_prover: None,
                shard_full_proofs,
            })
        } else {
            Ok(Self {
                shards,
                shard_dims,
                shard_verifier_context,
                shard_log_capacity_cached: shard_config.log_capacity,
                log_n_shards: config.log_n_shards,
                epoch: 0,
                r_index: vec![E::ScalarField::zero(); chain_groups],
                r_value: vec![E::ScalarField::zero(); chain_groups],
                chain_groups,
                epoch_commits: vec![initial_commit],
                db,
                vrf_prover: None,
                shard_full_proofs,
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
    fn try_recover_from_db(db: &dyn Db) -> Result<Option<RecoveredState<E, P>>, AegonError> {
        let Some(state_bytes) = db.get(key_coord_state())? else {
            return Ok(None);
        };

        // 1. Parse coord:state → (epoch, r_index[], r_value[])
        //
        // The scalars are length-prefixed vectors, one entry per
        // chain group. Deployments predating group-sharding wrote
        // bare scalars here, so their state will not parse — that is
        // deliberate. Silently reading a bare scalar as a one-element
        // vector is not possible to do safely (the encodings are not
        // distinguishable), and mis-parsing the FS chain would break
        // every subsequent audit rather than fail loudly now.
        let mut cursor = &state_bytes[..];
        let epoch: u64 = u64::deserialize_compressed(&mut cursor)
            .map_err(|e| AegonError::Database(format!("deserialize epoch: {e}")))?;
        let r_index: Vec<E::ScalarField> =
            Vec::<E::ScalarField>::deserialize_compressed(&mut cursor)
                .map_err(|e| AegonError::Database(format!("deserialize r_index: {e}")))?;
        let r_value: Vec<E::ScalarField> =
            Vec::<E::ScalarField>::deserialize_compressed(&mut cursor)
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
        Ok(Some(RecoveredState {
            epoch,
            r_index,
            r_value,
            epoch_commits,
        }))
    }

    /// Load the two-layer routing's fullness map from the DB. Returns
    /// `None` when the key isn't present (fresh DB or pre-two-layer
    /// state — caller should treat as "no shard has reported full").
    /// Errors if the key exists but is malformed.
    ///
    /// Wire format: `u32_le n` followed by `n` repetitions of
    /// `u8 has_proof || u32_le proof_len || proof_bytes`. `has_proof
    /// == 0` ⇒ slot is `None` (proof_len follows but is 0).
    fn try_recover_shard_full_proofs(
        db: &dyn Db,
        expected_n: usize,
    ) -> Result<Option<Vec<Option<Vec<u8>>>>, AegonError> {
        let Some(bytes) = db.get(key_shard_fullness())? else {
            return Ok(None);
        };
        if bytes.len() < 4 {
            return Err(AegonError::Database(
                "shard_fullness blob truncated (header)".into(),
            ));
        }
        let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if n != expected_n {
            return Err(AegonError::Database(format!(
                "shard_fullness length ({n}) does not match current shard count ({expected_n})"
            )));
        }
        let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
        let mut off = 4;
        for i in 0..n {
            if off + 5 > bytes.len() {
                return Err(AegonError::Database(format!(
                    "shard_fullness entry {i} truncated"
                )));
            }
            let has_proof = bytes[off];
            off += 1;
            let plen =
                u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
                    as usize;
            off += 4;
            if off + plen > bytes.len() {
                return Err(AegonError::Database(format!(
                    "shard_fullness entry {i} proof bytes truncated"
                )));
            }
            let proof = bytes[off..off + plen].to_vec();
            off += plen;
            out.push(if has_proof == 0 { None } else { Some(proof) });
        }
        Ok(Some(out))
    }

    /// Serialize the in-memory fullness map into the same wire format
    /// `try_recover_shard_full_proofs` decodes. Cheap — the map is
    /// `O(N_shards)` and each entry is small (placeholder proofs are
    /// empty bytes, real ones are at most a few hundred bytes).
    fn encode_shard_full_proofs(&self) -> Vec<u8> {
        let n = self.shard_full_proofs.len();
        let total_bytes_est: usize = 4
            + n * 5
            + self
                .shard_full_proofs
                .iter()
                .filter_map(|p| p.as_ref().map(Vec::len))
                .sum::<usize>();
        let mut out = Vec::with_capacity(total_bytes_est);
        out.extend_from_slice(&(n as u32).to_le_bytes());
        for entry in &self.shard_full_proofs {
            match entry {
                None => {
                    out.push(0);
                    out.extend_from_slice(&0u32.to_le_bytes());
                }
                Some(proof) => {
                    out.push(1);
                    out.extend_from_slice(&(proof.len() as u32).to_le_bytes());
                    out.extend_from_slice(proof);
                }
            }
        }
        out
    }

    /// Public read-only accessors for the fullness map. The publish
    /// routing path consults `is_shard_full`, the verifier-side
    /// shard-trail derivation consults the same to know how far to
    /// advance `H_shard`'s ctr.
    pub fn is_shard_full(&self, shard_id: usize) -> bool {
        self.shard_full_proofs
            .get(shard_id)
            .map(|entry| entry.is_some())
            .unwrap_or(false)
    }

    /// Opaque fullness proof for `shard_id`, when one was published.
    pub fn shard_fullness_proof(&self, shard_id: usize) -> Option<&[u8]> {
        self.shard_full_proofs
            .get(shard_id)
            .and_then(|entry| entry.as_deref())
    }

    /// Number of shards in the cluster.
    pub fn n_shards(&self) -> usize {
        self.shards.len()
    }

    /// Log2 of the shard count.
    pub fn log_n_shards(&self) -> usize {
        self.log_n_shards
    }

    /// Log2 of one shard's slot count.
    pub fn shard_log_capacity(&self) -> usize {
        self.shard_log_capacity_cached
    }

    /// Log2 of the whole dictionary's slot count.
    pub fn log_capacity(&self) -> usize {
        self.shard_log_capacity_cached + self.log_n_shards
    }

    /// The coordinator's current epoch.
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
        // Carries the deployment's chain-group partition, so an
        // auditor handed this context can never absorb a different
        // set of commitments than the coordinator did.
        let mut ctx = ShardedVerifierContext::new(self.verifier_context(), self.log_n_shards)
            .with_chain_groups(self.chain_groups);
        if let Some(prover) = self.vrf_prover.as_ref() {
            ctx = ctx.with_vrf_verifier(super::hash::VrfVerifier::new(prover.public_key().clone()));
        }
        ctx
    }

    /// The sharded commitment at the current epoch.
    pub fn current_commitment(&self) -> ShardedEpochCommitment<E, P> {
        self.epoch_commits
            .last()
            .expect("epoch 0 always retained")
            .clone()
    }

    /// The sharded commitment published at `epoch`, if still retained.
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
        let per_shard: Vec<EpochCommitment<E, P>> =
            self.shards.iter().map(|s| s.current_commitment()).collect();
        let refreshed = ShardedEpochCommitment::<E, P>::with_per_shard(0, per_shard);
        self.epoch = 0;
        self.epoch_commits.clear();
        self.epoch_commits.push(refreshed);
        Ok(())
    }

    /// Wipe the entire sharded dictionary back to its post-setup
    /// state without rebuilding the SRS, tearing down any gRPC
    /// connections, or removing the backing database files. After
    /// the call returns:
    ///
    ///   - every (in-process) shard is byte-identical to the state
    ///     it had right after [`Self::setup`] — see
    ///     [`Aegon::clear_dictionary`];
    ///   - the coordinator's `epoch = 0`, FS chain scalars are zero,
    ///     `epoch_commits` carries only the freshly-rebuilt epoch-0
    ///     entry, and the in-memory `routing` table is empty;
    ///   - the coord-side DB (when configured) has every `aegon:`
    ///     key dropped via a `delete_range`-style sweep, and the
    ///     fresh epoch-0 `aegon:coord:state` + `aegon:coord:epoch_commit:0`
    ///     are re-written so a downstream restart sees the empty
    ///     dictionary, not the pre-clear one.
    ///
    /// Use case: migration-bench's K-sweep. Each K starts from a
    /// truly empty dictionary, but we don't want to pay SRS
    /// generation + cluster respin per K. One process, one cluster,
    /// `clear_dictionary` between iterations.
    ///
    /// Returns `Err` when any shard is remote (gRPC transport): the
    /// `ClearDictionary` RPC isn't wired up yet, so the per-shard
    /// trait method errors out and we surface that.
    pub fn clear_dictionary(&mut self) -> Result<(), AegonError> {
        for shard in self.shards.iter_mut() {
            shard.clear_dictionary()?;
        }
        let per_shard: Vec<EpochCommitment<E, P>> =
            self.shards.iter().map(|s| s.current_commitment()).collect();
        let refreshed = ShardedEpochCommitment::<E, P>::with_per_shard(0, per_shard);
        self.epoch = 0;
        self.r_index = vec![E::ScalarField::zero(); self.chain_groups];
        self.r_value = vec![E::ScalarField::zero(); self.chain_groups];
        self.epoch_commits.clear();
        self.epoch_commits.push(refreshed.clone());
        if let Some(db) = &self.db {
            // Single `delete_range` over [`aegon:`, `aegon;`) wipes
            // every key the coord ever wrote. Cheaper than enumerating
            // keyspaces (`aegon:value:`, `aegon:routing:`, …) and
            // forward-compatible — any new `aegon:` keyspace added
            // later is cleared by the same call without code changes.
            db.delete_prefix(b"aegon:")?;
            // Re-persist the fresh epoch-0 state. A restart between
            // bench iterations would otherwise see no `coord:state`
            // and treat the DB as never-initialized, which is fine
            // for the no-restart bench flow but defends recovery in
            // case the bench harness crashes mid-iteration.
            let mut state_bytes: Vec<u8> = Vec::new();
            self.epoch
                .serialize_compressed(&mut state_bytes)
                .map_err(|e| AegonError::Database(format!("serialize epoch: {e}")))?;
            self.r_index
                .serialize_compressed(&mut state_bytes)
                .map_err(|e| AegonError::Database(format!("serialize r_index: {e}")))?;
            self.r_value
                .serialize_compressed(&mut state_bytes)
                .map_err(|e| AegonError::Database(format!("serialize r_value: {e}")))?;
            let mut commit_bytes: Vec<u8> = Vec::new();
            refreshed
                .serialize_compressed(&mut commit_bytes)
                .map_err(|e| AegonError::Database(format!("serialize epoch commit: {e}")))?;
            db.write_atomic(&[
                DbOp::Set {
                    key: key_coord_state().to_vec(),
                    value: state_bytes,
                },
                DbOp::Set {
                    key: key_epoch_commit(0),
                    value: commit_bytes,
                },
            ])?;
        }
        Ok(())
    }

    /// Two-layer routing's publish entry. Replacement for
    /// [`Self::publish`] under the new architecture where:
    ///   - The coord uses `H_shard(VRF, shard_ctr, label)` to pick a
    ///     destination shard, advancing `shard_ctr` only when the
    ///     destination has reported full.
    ///   - Each chosen shard runs its own `H_slot` open-addressing
    ///     internally to land on a slot — no `is_slot_occupied` RPC
    ///     fan-out from the coord.
    ///   - Coord state stays `O(N_shards)` (the
    ///     [`shard_full_proofs`](Self::shard_full_proofs) map) and
    ///     never grows with the user count.
    ///
    /// Multi-wave routing handles the partial-fullness case: if a
    /// shard reports full mid-batch, the un-placed tail is re-routed
    /// past it (via the same `H_shard` ctr-advance rule) in a
    /// follow-up wave. At the system's design operating point
    /// (per-shard fill ≤ 25%), the wave loop terminates after one
    /// iteration almost surely; full multi-wave runs are vanishingly
    /// rare.
    ///
    /// Once routing completes, the rest of the pipeline
    /// (`derive_chain_scalars`, `run_phase_2`, `finalize_epoch`,
    /// `persist_publish_to_db`) is unchanged — the only delta in
    /// persistence is the additional `aegon:coord:shard_fullness`
    /// write.
    ///
    /// The legacy [`Self::publish`] remains during the transition
    /// for tests + the existing wire-compatible path; once every
    /// caller migrates, the legacy entry is removed.
    pub fn publish_two_layer(
        &mut self,
        updates: &[(Label, Value)],
    ) -> Result<ShardedEpochCommitment<E, P>, AegonError>
    where
        P::Commitment: Clone,
    {
        Self::reject_duplicate_labels(updates)?;
        let n_shards = self.n_shards();
        let mut shard_outcomes: Vec<Option<super::server::PublishBatchOutcome<E, P>>> =
            (0..n_shards).map(|_| None).collect();

        // Routing-wave loop. Each iteration: group still-unplaced
        // labels by destination shard, dispatch a single
        // `publish_batch` call per non-empty group, then queue any
        // unplaced tail for the next wave.
        let mut unplaced: Vec<(Label, Value)> = updates.to_vec();
        let mut waves_executed = 0usize;
        let wave_bound = n_shards + 2;

        while !unplaced.is_empty() {
            waves_executed += 1;
            if waves_executed > wave_bound {
                // Defensive bound. At the protocol's operating
                // point we expect one wave; even pathological
                // load-skew shouldn't need more than `n_shards`.
                return Err(AegonError::Verification(
                    "publish_two_layer: routing failed to converge — every shard reported full",
                ));
            }
            // Route each label to its destination shard in parallel.
            // `route_label_to_shard` takes `&self` — every iteration
            // reads `vrf_prover` + `shard_full_proofs`, no mutation.
            // With 1M-label batches and a VRF evaluate per label this
            // is the single biggest serial hotspot on the coord side;
            // par_iter scales linearly with available cores.
            let drained_unplaced = std::mem::take(&mut unplaced);
            // Route AND collect H_shard proofs in one walk so the
            // destination shard can cache them. With no vrf_prover
            // configured the proof vec is empty and the cache is
            // skipped (lookups fall back to the legacy compute path).
            let routed: Vec<(usize, Label, Value, Vec<Vec<u8>>)> = drained_unplaced
                .into_par_iter()
                .map(|(label, value)| {
                    let (shard_id, _final_ctr, vrf_proofs) =
                        self.route_label_to_shard_with_proofs(&label)?;
                    Ok::<_, AegonError>((shard_id, label, value, vrf_proofs))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut groups: Vec<Vec<(Label, Value)>> = (0..n_shards).map(|_| Vec::new()).collect();
            let mut group_vrf_proofs: Vec<Vec<Vec<Vec<u8>>>> =
                (0..n_shards).map(|_| Vec::new()).collect();
            for (shard_id, label, value, proofs) in routed {
                groups[shard_id].push((label, value));
                group_vrf_proofs[shard_id].push(proofs);
            }

            // Fan out publish_batch across shards that received
            // labels this wave. Each shard's call is independent —
            // with N shards this is the parallelism the sharding
            // architecture is supposed to deliver. Phase-2 already
            // uses the same `par_iter_mut().zip(...)` pattern below.
            //
            // A shard that ran successfully in a prior wave has a
            // pending state that publish_batch wouldn't accept. The
            // wave invariant prevents this: a shard is either (a)
            // not full, never reached → can run now; (b) not full,
            // ran in a prior wave and succeeded → no labels route
            // to it again; (c) marked full → routing skips it.
            // Mid-wave failures (c) is the only way labels get re-
            // routed.
            type WaveItem<E2, P2> = Option<(
                usize,
                super::server::PublishBatchOutcome<E2, P2>,
                Vec<(Label, Value)>,
            )>;
            let shards = &mut self.shards;
            let wave_results: Vec<Result<WaveItem<E, P>, AegonError>> = shard_fanout_pool()
                .install(move || {
                    shards
                        .par_iter_mut()
                        .zip(groups.into_par_iter())
                        .zip(group_vrf_proofs.into_par_iter())
                        .enumerate()
                        .map(|(shard_id, ((shard, group), vrf_proofs))| {
                            if group.is_empty() {
                                return Ok(None);
                            }
                            let outcome = shard.publish_batch(&group, &vrf_proofs)?;
                            Ok(Some((shard_id, outcome, group)))
                        })
                        .collect()
                });

            let mut next_unplaced: Vec<(Label, Value)> = Vec::new();
            for r in wave_results {
                if let Some((shard_id, outcome, group)) = r? {
                    let placed_count = outcome.placed_count;
                    if outcome.fullness_proof.is_some() {
                        // Persist the (placeholder) fullness bytes.
                        let proof_bytes = outcome.fullness_proof.clone().unwrap_or_default();
                        self.shard_full_proofs[shard_id] = Some(proof_bytes);
                        // Re-queue the tail. They'll route to a
                        // different shard in the next wave because
                        // `is_shard_full` now returns true here.
                        for (label, value) in group.into_iter().skip(placed_count) {
                            next_unplaced.push((label, value));
                        }
                    }
                    shard_outcomes[shard_id] = Some(outcome);
                }
            }
            unplaced = next_unplaced;
        }

        // Cleanup pass: every shard that didn't receive any wave
        // still needs to run phase-1 (with an empty batch) so its
        // pending state is set up for phase-2's rand-poly advance.
        // Same fan-out reasoning as the wave loop above.
        let needs_cleanup: Vec<bool> = shard_outcomes.iter().map(|o| o.is_none()).collect();
        if needs_cleanup.iter().any(|&b| b) {
            let cleanup_results: Vec<
                Result<Option<(usize, super::server::PublishBatchOutcome<E, P>)>, AegonError>,
            > = {
                let shards = &mut self.shards;
                let needs_cleanup = &needs_cleanup;
                shard_fanout_pool().install(move || {
                    shards
                        .par_iter_mut()
                        .zip(needs_cleanup.par_iter())
                        .enumerate()
                        .map(|(shard_id, (shard, &needs))| {
                            if !needs {
                                return Ok(None);
                            }
                            let outcome = shard.publish_batch(&[], &[])?;
                            Ok(Some((shard_id, outcome)))
                        })
                        .collect()
                })
            };
            for r in cleanup_results {
                if let Some((shard_id, outcome)) = r? {
                    shard_outcomes[shard_id] = Some(outcome);
                }
            }
        }

        // Collect per-shard commits in shard_id order.
        let new_index_commits: Vec<P::Commitment> = shard_outcomes
            .iter()
            .map(|o| {
                o.as_ref()
                    .expect("cleanup pass guarantees every shard has an outcome")
                    .index_commitment
                    .clone()
            })
            .collect();
        let new_value_commits: Vec<P::Commitment> = shard_outcomes
            .iter()
            .map(|o| {
                o.as_ref()
                    .expect("cleanup pass guarantees every shard has an outcome")
                    .value_commitment
                    .clone()
            })
            .collect();

        // Phase 2 + persist (single RPC per shard).
        let (new_r_index, new_r_value) =
            self.derive_chain_scalars(&new_index_commits, &new_value_commits)?;
        let per_shard_commits = self.run_phase_2(&new_r_index, &new_r_value)?;
        let sharded_commit = self.finalize_epoch(per_shard_commits, new_r_index, new_r_value);

        // Coord-side persist: just `coord:state` + `coord:epoch_commit:{epoch}`.
        // The shard fan-out for dictionary content already happened
        // inside `publish_phase_2_and_persist` — each shard wrote its
        // own RocksDB before returning the new commitment.
        self.persist_publish_to_db(&sharded_commit)?;

        // Persist the (possibly updated) per-shard fullness map.
        if let Some(db) = &self.db {
            db.write_atomic(&[DbOp::Set {
                key: key_shard_fullness().to_vec(),
                value: self.encode_shard_full_proofs(),
            }])?;
        }

        Ok(sharded_commit)
    }

    /// Two-layer routing helper: starting from `shard_ctr_start`,
    /// advance the first-layer ctr until landing on a shard that
    /// has *not* been marked full in
    /// [`Self::shard_full_proofs`]. Returns `(shard_id, final_ctr)`
    /// so the caller can record the `shard_ctr` used (the lookup
    /// proof carries it later for the verifier to re-derive the
    /// same shard).
    ///
    /// Errors with `Verification` when the ctr walk would exceed a
    /// defensive bound — only fires if `is_shard_full` is `true` for
    /// at least `N_shards × 16` distinct `(shard_ctr, label)`
    /// pairings, which empirically can only happen if every shard
    /// has been marked full.
    fn route_label_to_shard(
        &self,
        label: &[u8],
        shard_ctr_start: u64,
    ) -> Result<(usize, u64), AegonError> {
        let mut shard_ctr = shard_ctr_start;
        let bound: u64 = (self.n_shards() as u64).saturating_mul(16).max(64);
        loop {
            if shard_ctr > bound {
                return Err(AegonError::Verification(
                    "H_shard ctr walk exceeded its bound — every shard appears to be full",
                ));
            }
            // Bits-only path: the publish-side routing never serializes
            // the proof, so `evaluate_h_shard` (one scalar mul) is the
            // right call. `prove_h_shard` (three scalar muls plus
            // Fiat-Shamir) is reserved for the lookup-side trail where
            // the proof actually goes to the wire.
            let shard_bits = if let Some(vrf) = &self.vrf_prover {
                vrf.evaluate_h_shard(shard_ctr, label, self.log_n_shards)
            } else {
                H::h_shard(shard_ctr, label, self.log_n_shards)
            };
            // log_n_shards == 0 (single-shard) → shard_bits is
            // empty → shard_id always 0. Loop terminates immediately
            // unless shard 0 is marked full.
            let mut shard_id: usize = 0;
            for (i, b) in shard_bits.iter().enumerate() {
                if *b {
                    shard_id |= 1usize << i;
                }
            }
            if !self.is_shard_full(shard_id) {
                return Ok((shard_id, shard_ctr));
            }
            shard_ctr += 1;
        }
    }

    /// Like [`Self::route_label_to_shard`], but also emits the VRF
    /// proof bytes for every probed ctr in `[0, final_shard_ctr]`. Used
    /// by `publish_two_layer` so the shipped per-label proofs can be
    /// cached on the destination shard and served back at lookup time
    /// without re-running the prove. Returns an empty proof vec when
    /// no VRF prover is configured (the cluster won't bind any proofs
    /// to lookups in that case either).
    fn route_label_to_shard_with_proofs(
        &self,
        label: &[u8],
    ) -> Result<(usize, u64, Vec<Vec<u8>>), AegonError> {
        let mut shard_ctr: u64 = 0;
        let bound: u64 = (self.n_shards() as u64).saturating_mul(16).max(64);
        let mut proofs: Vec<Vec<u8>> = Vec::new();
        loop {
            if shard_ctr > bound {
                return Err(AegonError::Verification(
                    "H_shard ctr walk exceeded its bound — every shard appears to be full",
                ));
            }
            let (shard_bits, proof_opt) = if let Some(vrf) = &self.vrf_prover {
                let (bits, proof) = vrf.prove_h_shard(shard_ctr, label, self.log_n_shards);
                (bits, Some(proof.to_vec()))
            } else {
                (H::h_shard(shard_ctr, label, self.log_n_shards), None)
            };
            if let Some(p) = proof_opt {
                proofs.push(p);
            }
            let mut shard_id: usize = 0;
            for (i, b) in shard_bits.iter().enumerate() {
                if *b {
                    shard_id |= 1usize << i;
                }
            }
            if !self.is_shard_full(shard_id) {
                return Ok((shard_id, shard_ctr, proofs));
            }
            shard_ctr += 1;
        }
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

    /// Derive the shared `(r_index, r_value)` Fiat-Shamir scalars for
    /// this epoch transition. Each scalar is `O(prev_r, every shard's
    /// new data commitment)` — binding to all N commits up front is
    /// what stops a malicious server from re-tuning any one shard's
    /// commit after observing the chain randomness.
    #[cfg_attr(
        feature = "tracing_instrument",
        tracing::instrument(level = "debug", skip_all, name = "ShardedAegon::DeriveChainScalars")
    )]
    fn derive_chain_scalars(
        &self,
        new_index_commits: &[P::Commitment],
        new_value_commits: &[P::Commitment],
    ) -> Result<(Vec<E::ScalarField>, Vec<E::ScalarField>), AegonError> {
        let plan = super::chain_groups::GroupPlan::new(self.n_shards(), self.chain_groups)?;
        let index_groups = plan.split(new_index_commits)?;
        let value_groups = plan.split(new_value_commits)?;

        // One rolling accumulator per group, each absorbing only its
        // own shards' commitments. With `chain_groups == 1` this is
        // byte-for-byte the single-transcript derivation it replaces.
        let new_r_index = index_groups
            .iter()
            .zip(&self.r_index)
            .map(|(commits, prev)| {
                self.shard_verifier_context.audit_fs.chain_scalar(
                    b"aegon.sharded.fs.r_index",
                    *prev,
                    commits,
                )
            })
            .collect();
        let new_r_value = value_groups
            .iter()
            .zip(&self.r_value)
            .map(|(commits, prev)| {
                self.shard_verifier_context.audit_fs.chain_scalar(
                    b"aegon.sharded.fs.r_value",
                    *prev,
                    commits,
                )
            })
            .collect();
        Ok((new_r_index, new_r_value))
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
        new_r_index: &[E::ScalarField],
        new_r_value: &[E::ScalarField],
    ) -> Result<Vec<EpochCommitment<E, P>>, AegonError> {
        // Combined phase-2 + persist. Each shard runs phase-2 crypto
        // locally and writes its dictionary-content DbOps (with empty
        // merkle paths — coord stitches them at lookup time) to its
        // own RocksDB in a single RPC. We collect the new per-shard
        // EpochCommitments and build the cross-shard merkle tree
        // upstream in `finalize_epoch`.
        let shard_ids: Vec<u32> = (0..self.shards.len() as u32).collect();
        // Each shard is handed *its group's* scalars. Because phase 2
        // is already one call per shard, group-sharding costs nothing
        // here and works for remote shards as well as in-process
        // ones -- unlike `audit_fs`, which a remote shard configures
        // for itself.
        let plan = super::chain_groups::GroupPlan::new(self.n_shards(), self.chain_groups)?;
        let shards = &mut self.shards;
        let plan = &plan;
        let results: Vec<Result<EpochCommitment<E, P>, AegonError>> =
            shard_fanout_pool().install(move || {
                shards
                    .par_iter_mut()
                    .zip(shard_ids.into_par_iter())
                    .map(|(shard, shard_id)| {
                        let g = plan.group_of(shard_id as usize);
                        shard.publish_phase_2_and_persist(new_r_index[g], new_r_value[g], shard_id)
                    })
                    .collect()
            });
        results.into_iter().collect::<Result<Vec<_>, _>>()
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
        new_r_index: Vec<E::ScalarField>,
        new_r_value: Vec<E::ScalarField>,
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
        sharded_commit: &ShardedEpochCommitment<E, P>,
    ) -> Result<(), AegonError> {
        let prof = super::instrument::publish_profile_enabled();
        let _persist_t_total = std::time::Instant::now();

        // Post-collapse: every per-label DB op (value, value_history,
        // label_placement, openings) was built + written by the
        // owning shard inside `publish_phase_2_and_persist`. The
        // coord's own DB writes exactly two keys here:
        //   * `coord:state` (epoch + FS scalars)
        //   * `coord:epoch_commit:{epoch}` (the global sharded
        //      commitment, including the cached merkle paths the
        //      lookup-history join needs).
        let mut coord_ops: Vec<DbOp> = Vec::with_capacity(2);

        // 1. coord:state — one key, contains (epoch, r_index, r_value).
        let _step1_t = std::time::Instant::now();
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
        coord_ops.push(DbOp::Set {
            key: key_coord_state().to_vec(),
            value: state_bytes,
        });

        if prof {
            eprintln!(
                "[pub-profile] persist.step1_coord_state_serialize: {:.3} ms",
                _step1_t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        // 2. coord:epoch_commit:{epoch} — the externally-published commitment.
        let _step2_t = std::time::Instant::now();
        let mut commit_bytes = Vec::new();
        sharded_commit
            .serialize_compressed(&mut commit_bytes)
            .map_err(|e| AegonError::Database(format!("serialize epoch commit: {e}")))?;
        coord_ops.push(DbOp::Set {
            key: key_epoch_commit(sharded_commit.epoch),
            value: commit_bytes,
        });

        if prof {
            eprintln!(
                "[pub-profile] persist.step2_epoch_commit_serialize: {:.3} ms",
                _step2_t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        // 3. Coord-side write — at most ~10 KB total per publish.
        let _coord_write_t = std::time::Instant::now();
        if let Some(db) = &self.db {
            db.write_atomic(&coord_ops)?;
        }
        if prof {
            eprintln!(
                "[pub-profile] persist.step3_coord_write: {:.3} ms (ops={})",
                _coord_write_t.elapsed().as_secs_f64() * 1000.0,
                coord_ops.len(),
            );
        }

        if prof {
            eprintln!(
                "[pub-profile] PERSIST_TOTAL: {:.3} ms",
                _persist_t_total.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(())
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
    pub fn lookup_value(&self, slot: &LabelSlot) -> Result<ShardedValueProof<E, P>, AegonError> {
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
    pub fn lookup_history(&self, label: &Label) -> Result<ShardedValueHistory<E, P>, AegonError> {
        // Post-refactor + Patch 7: value_history lives on the owning
        // shard's DB, and the entire {fetch + per-entry remask +
        // freshness opening} chain is collapsed into ONE shard RPC
        // (`fetch_full_value_history`). The shard reads its sliding
        // window, decodes + remasks each entry under one read lock,
        // and opens the live `rand_value_poly` at the latest entry's
        // slot — all in a single call. In-process shards return
        // Ok(empty) via the default trait impl, preserving the prior
        // empty-history behaviour for `DbSource::None`.
        //
        // What the coord still does locally (cheap):
        //   * Validate `entry.shard_id` is in range.
        //   * Stitch in cross-shard merkle paths from the in-memory
        //     `epoch_commits` cache (the shard wrote empty paths).
        //   * Wrap the freshness eval/proof into the full
        //     `FreshnessAttestation` (the shard returned only the
        //     opening; the per-shard commit + merkle path come from
        //     the coord's live commitment).
        //
        // Stored proofs were PLAIN non-ZK at publish_phase_2 time
        // regardless of `--private`. Under a non-hiding SRS the
        // remask is a pass-through; under a hiding SRS the shard
        // applies the masking-server protocol (it holds the per-epoch
        // `tau_f` snapshots and has the masking client or inline
        // fallback). The publish-time crypto cost is identical
        // between `--private=true` and `--private=false`.
        let (dest_shard_id, _) = self.route_label_to_shard(label, 0)?;
        let full = self.shards[dest_shard_id].fetch_full_value_history(label)?;
        let mut entries = full.entries;
        // Stitch in cross-shard merkle paths. The shard cannot
        // produce these (it has no view of the cross-shard root) so
        // it left them empty. `.get()` over `epoch_commits` rather
        // than direct indexing — see legacy comment: a stale entry
        // pointing past `coord.epoch` should surface as
        // `Status::Internal` with the bad epoch numbers, not panic.
        for entry in entries.iter_mut() {
            let shard_id = entry.shard_id;
            if (shard_id as usize) >= self.shards.len() {
                return Err(AegonError::Config(format!(
                    "lookup_history: entry's shard_id {shard_id} out of range \
                     (have {} shards)",
                    self.shards.len()
                )));
            }
            entry.post_merkle_path = self
                .epoch_commits
                .get(entry.epoch as usize)
                .ok_or_else(|| {
                    AegonError::Database(format!(
                        "lookup_history: entry.epoch={} out of bounds \
                     (epoch_commits.len()={}, coord.epoch={})",
                        entry.epoch,
                        self.epoch_commits.len(),
                        self.epoch,
                    ))
                })?
                .merkle_path(shard_id as usize)
                .to_vec();
            if entry.epoch > 0 {
                entry.prev_merkle_path = self
                    .epoch_commits
                    .get((entry.epoch - 1) as usize)
                    .ok_or_else(|| {
                        AegonError::Database(format!(
                            "lookup_history: entry.epoch-1={} out of bounds \
                         (epoch_commits.len()={}, coord.epoch={})",
                            entry.epoch - 1,
                            self.epoch_commits.len(),
                            self.epoch,
                        ))
                    })?
                    .merkle_path(shard_id as usize)
                    .to_vec();
            }
        }
        // Freshness attestation: the shard returned the eval+proof
        // for `entries[0].slot_bits` against its live
        // `rand_value_commitment`; the coord wraps it with the live
        // per-shard commit + merkle path. The shard's `rand_value`
        // evaluation at a slot is invariant under any publish that
        // does not touch that slot, so a verifier seeing the same
        // evaluation here as in `entries[0].rand_value_post_eval`
        // learns that no publish has touched the slot since then.
        let freshness = if let Some(latest) = entries.first() {
            let shard_id = latest.shard_id;
            if (shard_id as usize) >= self.shards.len() {
                return Err(AegonError::Config(format!(
                    "lookup_history: latest entry's shard_id {shard_id} out of range \
                     (have {} shards)",
                    self.shards.len()
                )));
            }
            let eval = full.freshness_eval.ok_or_else(|| {
                AegonError::Database(
                    "lookup_history: shard returned entries but no freshness eval".into(),
                )
            })?;
            let proof = full.freshness_proof.ok_or_else(|| {
                AegonError::Database(
                    "lookup_history: shard returned entries but no freshness proof".into(),
                )
            })?;
            let current = self.current_commitment();
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
        // Post-refactor + Patch 8: label_placement lives on the
        // owning shard's DB, and the {fetch placement + open
        // rand_index_at_slot_current} chain is collapsed into ONE
        // shard RPC (`fetch_full_label_history`). The shard reads its
        // placement and opens the live `rand_index_poly` at the slot
        // in the same call. In-process shards return Ok(None) via
        // the default trait impl, preserving the prior "no
        // placement" behaviour for `DbSource::None`.
        //
        // What the coord still does locally (cheap):
        //   * Validate `placement.shard_id` is in range.
        //   * Stitch `placement_merkle_path` in from
        //     `epoch_commits` (the shard wrote it empty).
        //   * Wrap the freshness eval/proof in a
        //     `FreshnessAttestationLabel` with the live per-shard
        //     commit + merkle path.
        let (dest_shard_id, _) = self.route_label_to_shard(label, 0)?;
        let Some(full) = self.shards[dest_shard_id].fetch_full_label_history(label)? else {
            return Ok(ShardedLabelHistory {
                label: label.clone(),
                placement: None,
                freshness: None,
            });
        };
        let mut placement = full.placement;
        let shard_id = placement.shard_id;
        if (shard_id as usize) >= self.shards.len() {
            return Err(AegonError::Config(format!(
                "lookup_label_history: placement shard_id {shard_id} out of range \
                 (have {} shards)",
                self.shards.len()
            )));
        }
        placement.placement_merkle_path = self
            .epoch_commits
            .get(placement.epoch as usize)
            .ok_or_else(|| {
                AegonError::Database(format!(
                    "lookup_label_history: placement.epoch={} out of bounds \
                 (epoch_commits.len()={}, coord.epoch={})",
                    placement.epoch,
                    self.epoch_commits.len(),
                    self.epoch,
                ))
            })?
            .merkle_path(shard_id as usize)
            .to_vec();
        let current = self.current_commitment();
        let merkle_path = current.merkle_path(shard_id as usize).to_vec();
        let shard_commit = current.per_shard[shard_id as usize].clone();
        let freshness = Some(FreshnessAttestationLabel {
            shard_id,
            slot_bits: placement.slot_bits.clone(),
            rand_index_current_eval: full.freshness_eval,
            rand_index_current_proof: full.freshness_proof,
            shard_commit,
            merkle_path,
        });
        Ok(ShardedLabelHistory {
            label: label.clone(),
            placement: Some(placement),
            freshness,
        })
    }

    // ============= two-layer routing lookup methods ====================
    //
    // Mirror of `lookup_label` / `lookup_value` / `lookup` for the
    // two-layer routing model. The label-residency proof carries:
    //
    //   * an inter-shard `H_shard` trail (the coord's first-layer
    //     routing) advancing `shard_ctr` past full shards;
    //   * a single shard leaf + Merkle path for the destination shard;
    //   * an intra-shard `H_slot` trail produced by the shard itself.
    //
    // Coord state used is `O(N_shards)` (the `shard_full_proofs` map);
    // no per-label state is read or required at lookup time.

    /// Two-layer label-residency proof. The coord walks
    /// `H_shard(shard_ctr, label)` until landing on a non-full shard,
    /// asks that shard for the within-shard trail length via
    /// `find_label_slot`, then collects an index opening per
    /// `H_slot(slot_ctr, label)` probe.
    ///
    /// Returns the canonical `LabelSlot` (so the client can call
    /// `lookup_value_two_layer(slot)` against it without re-proving
    /// residency) together with the proof.
    pub fn lookup_label_two_layer(
        &self,
        label: &Label,
    ) -> Result<(LabelSlot, ShardedLabelProofTwoLayer<E, P>), AegonError> {
        // First-layer routing: walk H_shard until landing on a
        // non-full shard. `route_label_to_shard` does the same walk
        // publish does, so lookup and publish always agree on the
        // destination shard. Uses the cheap `evaluate_h_shard` path
        // (1 scalar mul per step); the matching VRF proofs come from
        // the destination shard's cache below, not from re-proving
        // here.
        let (dest_shard_id, final_shard_ctr) = self.route_label_to_shard(label, 0)?;

        // Second-layer routing: ask the destination shard for the
        // within-shard placement of `label`, the index-polynomial
        // openings along its `H_slot` trail, and (if VRF was attached
        // at publish time) the cached H_shard + H_slot VRF proofs —
        // all in one RPC. Replaces the legacy `find_label_slot` +
        // N × `open_index_at_slot` + per-step `prove_h_*` chain (one
        // network RTT and zero coord-side ECVRF compute instead of
        // `slot_ctr0 + 2` + (final_shard_ctr + 1 + slot_ctr0 + 1)
        // ECVRF proves).
        let trail = self.shards[dest_shard_id]
            .fetch_label_proof_trail(label)?
            .ok_or_else(|| AegonError::UnknownLabel(label.clone()))?;
        let LabelProofTrail {
            final_slot_bits,
            slot_ctr0,
            entries,
            vrf_proofs_shard: cached_shard_proofs,
            vrf_proofs_slot: cached_slot_proofs,
        } = trail;
        if entries.len() != (slot_ctr0 as usize) + 1 {
            return Err(AegonError::Verification(
                "lookup_label_two_layer: trail entries length disagrees with slot_ctr0",
            ));
        }
        // When the shard returned cached proofs they must match the
        // ctr counts we'll iterate below. Empty vectors signal "no
        // cache; fall back to coord-side prove" — the legacy path.
        let have_cached_shard_proofs = !cached_shard_proofs.is_empty();
        if have_cached_shard_proofs && cached_shard_proofs.len() != (final_shard_ctr as usize) + 1 {
            return Err(AegonError::Verification(
                "lookup_label_two_layer: cached vrf_proofs_shard length disagrees with final_shard_ctr",
            ));
        }
        let have_cached_slot_proofs = !cached_slot_proofs.is_empty();
        if have_cached_slot_proofs && cached_slot_proofs.len() != (slot_ctr0 as usize) + 1 {
            return Err(AegonError::Verification(
                "lookup_label_two_layer: cached vrf_proofs_slot length disagrees with slot_ctr0",
            ));
        }

        // Build the inter-shard route trail. For ctr in
        // 0..final_shard_ctr we emit an intermediate probe carrying
        // the shard's fullness proof; for ctr == final_shard_ctr we
        // emit the landing probe (no fullness proof). VRF proofs come
        // from the cache when available, else we fall back to
        // prove_h_shard.
        let mut route: Vec<ShardRoutingProbe> = Vec::with_capacity((final_shard_ctr as usize) + 1);
        for ctr in 0..=final_shard_ctr {
            let (bits, vrf_proof_bytes) = if have_cached_shard_proofs {
                let bits = if let Some(vrf) = &self.vrf_prover {
                    vrf.evaluate_h_shard(ctr, label, self.log_n_shards)
                } else {
                    H::h_shard(ctr, label, self.log_n_shards)
                };
                (bits, cached_shard_proofs[ctr as usize].clone())
            } else if let Some(prover) = self.vrf_prover.as_ref() {
                let (b, p) = prover.prove_h_shard(ctr, label, self.log_n_shards);
                (b, p.to_vec())
            } else {
                let b = H::h_shard(ctr, label, self.log_n_shards);
                (b, Vec::new())
            };
            let mut shard_id: u32 = 0;
            for (i, b) in bits.iter().enumerate() {
                if *b {
                    shard_id |= 1u32 << i;
                }
            }
            let fullness_proof = if ctr < final_shard_ctr {
                let bytes = self
                    .shard_fullness_proof(shard_id as usize)
                    .ok_or({
                        AegonError::Verification(
                            "lookup_label_two_layer: routing skipped a shard not marked full",
                        )
                    })?
                    .to_vec();
                Some(bytes)
            } else {
                None
            };
            route.push(ShardRoutingProbe {
                shard_id,
                vrf_proof: vrf_proof_bytes,
                fullness_proof,
            });
        }

        // Anchor the destination shard's leaf under the live root.
        let current = self.current_commitment();
        let dest_leaf = current.per_shard[dest_shard_id].clone();
        let dest_merkle_path = current.merkle_path(dest_shard_id).to_vec();

        // Walk the intra-shard trail. The shard already produced the
        // openings; the coord adds the VRF proof per probe (from cache
        // when available) and sanity-checks that the shard's
        // `slot_bits` match the H_slot derivation.
        let shard_log_capacity = self.shard_log_capacity();
        let mut slots: Vec<ShardSlotProbe<E, P>> = Vec::with_capacity((slot_ctr0 as usize) + 1);
        for (slot_ctr, entry) in (0..=slot_ctr0).zip(entries.into_iter()) {
            let (expected_bits, vrf_proof_bytes) = if have_cached_slot_proofs {
                let bits = if let Some(vrf) = &self.vrf_prover {
                    vrf.evaluate_h_slot(slot_ctr, label, shard_log_capacity)
                } else {
                    H::h_slot(slot_ctr, label, shard_log_capacity)
                };
                (bits, cached_slot_proofs[slot_ctr as usize].clone())
            } else if let Some(prover) = self.vrf_prover.as_ref() {
                let (b, p) = prover.prove_h_slot(slot_ctr, label, shard_log_capacity);
                (b, p.to_vec())
            } else {
                let b = H::h_slot(slot_ctr, label, shard_log_capacity);
                (b, Vec::new())
            };
            if expected_bits != entry.slot_bits {
                return Err(AegonError::Verification(
                    "lookup_label_two_layer: shard's trail slot_bits disagree with H_slot derivation",
                ));
            }
            if slot_ctr == slot_ctr0 && entry.slot_bits != final_slot_bits {
                return Err(AegonError::Verification(
                    "lookup_label_two_layer: derived final slot_bits disagree with shard's fetch_label_proof_trail",
                ));
            }
            slots.push(ShardSlotProbe {
                slot_bits: entry.slot_bits,
                vrf_proof: vrf_proof_bytes,
                evaluation: entry.evaluation,
                proof: entry.proof,
            });
        }

        let slot = LabelSlot {
            shard_id: dest_shard_id as u32,
            slot_bits: final_slot_bits,
        };
        let proof = ShardedLabelProofTwoLayer {
            route,
            dest_shard_id: dest_shard_id as u32,
            dest_leaf,
            dest_merkle_path,
            slots,
        };
        Ok((slot, proof))
    }

    /// Second half of the two-layer lookup. Identical to
    /// [`Self::lookup_value`] — the value opening is independent of
    /// routing, so we just delegate. Kept as a named wrapper so
    /// callers reading two-layer code have an obvious paired entry
    /// point.
    pub fn lookup_value_two_layer(
        &self,
        slot: &LabelSlot,
    ) -> Result<ShardedValueProof<E, P>, AegonError> {
        self.lookup_value(slot)
    }

    /// Combined two-layer lookup: residency proof + value opening +
    /// raw value bytes (when a DB is attached). Mirror of
    /// [`Self::lookup`] for the two-layer routing model.
    pub fn lookup_two_layer(
        &self,
        label: &Label,
    ) -> Result<(Value, ShardedLookupProofTwoLayer<E, P>), AegonError> {
        let (slot, label_proof) = self.lookup_label_two_layer(label)?;
        let value_proof = self.lookup_value(&slot)?;
        // Post-refactor: raw value bytes live on the owning shard's DB
        // (`slot.shard_id`). In-process shards return Ok(None) via the
        // default trait impl, preserving the prior empty-Value behavior
        // for `DbSource::None` tests.
        let value: Value = match self.shards[slot.shard_id as usize].fetch_value(label)? {
            Some(v) => v,
            None => Vec::new(),
        };
        Ok((
            value,
            ShardedLookupProofTwoLayer {
                label_proof,
                value_evaluation: value_proof.evaluation,
                value_proof: value_proof.proof,
            },
        ))
    }

    /// Two-layer consistency proof. Attests that `label` was at the
    /// same `(shard, slot)` at epochs `s0` and `s1 = self.epoch`,
    /// that no other publish disturbed any slot in the H_slot
    /// trail in between, and that the value at the final slot
    /// wasn't updated.
    pub fn consistency_proof_two_layer(
        &self,
        label: &Label,
        s0: u64,
    ) -> Result<ShardedConsistencyProofTwoLayer<E, P>, AegonError> {
        let s0_commit = self
            .epoch_commitment(s0)
            .ok_or(AegonError::InvalidEpoch(s0))?;
        let s1_commit = self.current_commitment();
        let s1 = self.epoch;

        // (1) Inter-shard route trail — same walk lookup_label_two_layer
        // does (uses self.shard_full_proofs at the current epoch).
        let (dest_shard_id, final_shard_ctr) = self.route_label_to_shard(label, 0)?;
        let mut route: Vec<ShardRoutingProbe> = Vec::with_capacity((final_shard_ctr as usize) + 1);
        for ctr in 0..=final_shard_ctr {
            let (bits, vrf_proof_bytes) = if let Some(prover) = self.vrf_prover.as_ref() {
                let (b, p) = prover.prove_h_shard(ctr, label, self.log_n_shards);
                (b, p.to_vec())
            } else {
                let b = H::h_shard(ctr, label, self.log_n_shards);
                (b, Vec::new())
            };
            let mut shard_id: u32 = 0;
            for (i, b) in bits.iter().enumerate() {
                if *b {
                    shard_id |= 1u32 << i;
                }
            }
            let fullness_proof = if ctr < final_shard_ctr {
                let bytes = self
                    .shard_fullness_proof(shard_id as usize)
                    .ok_or({
                        AegonError::Verification(
                            "consistency_proof_two_layer: routing skipped a shard not marked full",
                        )
                    })?
                    .to_vec();
                Some(bytes)
            } else {
                None
            };
            route.push(ShardRoutingProbe {
                shard_id,
                vrf_proof: vrf_proof_bytes,
                fullness_proof,
            });
        }

        // (2) Destination shard leaves + Merkle paths at both epochs.
        let dest_leaf_s0 = s0_commit.per_shard[dest_shard_id].clone();
        let dest_leaf_s1 = s1_commit.per_shard[dest_shard_id].clone();
        let dest_merkle_path_s0 = s0_commit.merkle_path(dest_shard_id).to_vec();
        let dest_merkle_path_s1 = s1_commit.merkle_path(dest_shard_id).to_vec();

        // (3) Within-shard slot trail. Use the current (s1) shard
        // state to learn slot_ctr0 — by design the rand_index walk
        // must look identical at s0 (else the proof's equality
        // check fails, surfacing the disturbance).
        let (final_slot_bits, slot_ctr0) = self.shards[dest_shard_id]
            .find_label_slot(label)?
            .ok_or_else(|| AegonError::UnknownLabel(label.clone()))?;

        let shard_log_capacity = self.shard_log_capacity();
        let mut slots: Vec<ShardSlotRandPair<E, P>> = Vec::with_capacity((slot_ctr0 as usize) + 1);
        for slot_ctr in 0..=slot_ctr0 {
            let (slot_bits, vrf_proof_bytes) = if let Some(prover) = self.vrf_prover.as_ref() {
                let (b, p) = prover.prove_h_slot(slot_ctr, label, shard_log_capacity);
                (b, p.to_vec())
            } else {
                let b = H::h_slot(slot_ctr, label, shard_log_capacity);
                (b, Vec::new())
            };
            if slot_ctr == slot_ctr0 && slot_bits != final_slot_bits {
                return Err(AegonError::Verification(
                    "consistency_proof_two_layer: derived final slot_bits disagree with shard's find_label_slot",
                ));
            }
            let (eval_s0, proof_s0) =
                self.shards[dest_shard_id].open_rand_index_at_slot_in_epoch(&slot_bits, s0)?;
            let (eval_s1, proof_s1) =
                self.shards[dest_shard_id].open_rand_index_at_slot_in_epoch(&slot_bits, s1)?;
            slots.push(ShardSlotRandPair {
                slot_bits,
                vrf_proof: vrf_proof_bytes,
                rand_index_s0_eval: eval_s0,
                rand_index_s0_proof: proof_s0,
                rand_index_s1_eval: eval_s1,
                rand_index_s1_proof: proof_s1,
            });
        }

        // (4) rand_value at the final slot at both epochs.
        let (value_rand_s0_eval, value_rand_s0_proof) =
            self.shards[dest_shard_id].open_rand_value_at_slot_in_epoch(&final_slot_bits, s0)?;
        let (value_rand_s1_eval, value_rand_s1_proof) =
            self.shards[dest_shard_id].open_rand_value_at_slot_in_epoch(&final_slot_bits, s1)?;

        Ok(ShardedConsistencyProofTwoLayer {
            route,
            dest_shard_id: dest_shard_id as u32,
            dest_leaf_s0,
            dest_leaf_s1,
            dest_merkle_path_s0,
            dest_merkle_path_s1,
            slots,
            value_rand_s0_eval,
            value_rand_s0_proof,
            value_rand_s1_eval,
            value_rand_s1_proof,
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
    let reconstructed =
        verify_merkle_path::<E, P>(&proof.leaf, proof.shard_id as usize, &proof.merkle_path);
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
    /// `(prev_root, post_root)` per verified history entry.
    pub entry_roots: Vec<(EpochDigest, EpochDigest)>,
    /// The live sharded root the freshness opening anchors to.
    pub live_root: Option<EpochDigest>,
}

/// Verify a value-history bundle: every entry's openings, its Merkle
/// anchoring at both epochs, and the freshness attestation.
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
        }
        (Some(_), None) => {
            return Err(AegonError::Verification(
                "history has a freshness attestation but no entries",
            ));
        }
        (Some(fr), Some(latest)) => {
            if fr.shard_id != latest.shard_id || fr.slot_bits != latest.slot_bits {
                return Err(AegonError::Verification(
                    "freshness attestation references a different (shard, slot) than the latest entry",
                ));
            }
            let live_root =
                verify_merkle_path::<E, P>(&fr.shard_commit, fr.shard_id as usize, &fr.merkle_path);
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
        }
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
    /// Root the placement record anchors under, if one was returned.
    pub placement_root: Option<EpochDigest>,
    /// The live sharded root the freshness opening anchors to.
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
            let live_root =
                verify_merkle_path::<E, P>(&fr.shard_commit, fr.shard_id as usize, &fr.merkle_path);

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
        }
    }
}

// ================ two-layer routing verifiers ========================

/// Verify a two-layer label-residency proof. Re-derives the
/// destination shard from the `H_shard` route, anchors the
/// destination shard's leaf under `commit.merkle_root`, then walks
/// the `H_slot` intra-shard trail checking each PCS opening + the
/// open-addressing constraints (intermediate non-zero/non-H_F(label),
/// final == H_F(label)).
///
/// Returns the canonical `LabelSlot` (so the caller can pair it with
/// a `lookup_value_two_layer` call without re-proving residency).
pub fn verify_lookup_label_two_layer<E, P, H>(
    ctx: &ShardedVerifierContext<E, P>,
    commit: &ShardedEpochCommitment<E, P>,
    label: &Label,
    proof: &ShardedLabelProofTwoLayer<E, P>,
) -> Result<LabelSlot, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    if proof.route.is_empty() {
        return Err(AegonError::Verification(
            "two-layer label proof has an empty route trail",
        ));
    }
    if proof.slots.is_empty() {
        return Err(AegonError::Verification(
            "two-layer label proof has an empty slot trail",
        ));
    }

    // (1) Inter-shard routing trail. Re-derive shard_id at each
    // shard_ctr via H_shard (consuming the VRF proof when present),
    // confirm intermediate probes carry a fullness proof and the
    // landing probe doesn't, and that the landing probe's shard_id
    // matches `proof.dest_shard_id`.
    let final_idx = proof.route.len() - 1;
    for (ctr_us, rprobe) in proof.route.iter().enumerate() {
        let ctr = ctr_us as u64;
        let bits = if let Some(verifier) = ctx.vrf_verifier.as_ref() {
            if rprobe.vrf_proof.is_empty() {
                return Err(AegonError::Verification(
                    "two-layer label proof: vrf-mode route probe has empty vrf_proof",
                ));
            }
            verifier
                .verify_h_shard(ctr, label, &rprobe.vrf_proof, ctx.log_n_shards)
                .map_err(|e| match e {
                    super::hash::VrfVerifyError::Malformed(_) => {
                        AegonError::Verification("two-layer route probe vrf_proof failed to parse")
                    }
                    super::hash::VrfVerifyError::InvalidProof(_) => {
                        AegonError::Verification("two-layer route probe vrf_proof did not verify")
                    }
                })?
        } else {
            H::h_shard(ctr, label, ctx.log_n_shards)
        };
        let mut shard_id: u32 = 0;
        for (i, b) in bits.iter().enumerate() {
            if *b {
                shard_id |= 1u32 << i;
            }
        }
        if shard_id != rprobe.shard_id {
            return Err(AegonError::Verification(
                "two-layer route probe shard_id does not match H_shard(ctr, label)",
            ));
        }
        if ctr_us < final_idx {
            // Intermediate probe: must advertise a fullness proof.
            // Placeholder bytes are accepted; real soundness check
            // drops in here when the fullness-proof system lands.
            if rprobe.fullness_proof.is_none() {
                return Err(AegonError::Verification(
                    "two-layer route probe in intermediate position lacks a fullness proof",
                ));
            }
        } else {
            // Landing probe: must not carry a fullness proof, and
            // must announce the same shard_id as proof.dest_shard_id.
            if rprobe.fullness_proof.is_some() {
                return Err(AegonError::Verification(
                    "two-layer route probe in landing position carries a fullness proof",
                ));
            }
            if rprobe.shard_id != proof.dest_shard_id {
                return Err(AegonError::Verification(
                    "two-layer route landing shard_id does not match proof.dest_shard_id",
                ));
            }
        }
    }

    // (2) Anchor the destination shard leaf under commit.merkle_root.
    let dest_root = verify_merkle_path::<E, P>(
        &proof.dest_leaf,
        proof.dest_shard_id as usize,
        &proof.dest_merkle_path,
    );
    if dest_root != commit.merkle_root {
        return Err(AegonError::Verification(
            "two-layer label proof: dest_merkle_path does not reconstruct epoch root",
        ));
    }

    // (3) Intra-shard slot trail. Each probe: re-derive slot_bits via
    // H_slot (consuming VRF proof when present), verify PCS opening
    // against dest_leaf.index_commitment, check open-addressing
    // constraints (intermediate != 0 and != h_label; final == h_label).
    let h_label = H::h_f(label);
    let shard_log_capacity = ctx.shard_log_capacity();
    let final_slot_idx = proof.slots.len() - 1;
    let mut final_slot_bits: Option<Vec<bool>> = None;
    for (slot_ctr_us, sprobe) in proof.slots.iter().enumerate() {
        let slot_ctr = slot_ctr_us as u64;
        let derived_bits = if let Some(verifier) = ctx.vrf_verifier.as_ref() {
            if sprobe.vrf_proof.is_empty() {
                return Err(AegonError::Verification(
                    "two-layer label proof: vrf-mode slot probe has empty vrf_proof",
                ));
            }
            verifier
                .verify_h_slot(slot_ctr, label, &sprobe.vrf_proof, shard_log_capacity)
                .map_err(|e| match e {
                    super::hash::VrfVerifyError::Malformed(_) => {
                        AegonError::Verification("two-layer slot probe vrf_proof failed to parse")
                    }
                    super::hash::VrfVerifyError::InvalidProof(_) => {
                        AegonError::Verification("two-layer slot probe vrf_proof did not verify")
                    }
                })?
        } else {
            H::h_slot(slot_ctr, label, shard_log_capacity)
        };
        if derived_bits != sprobe.slot_bits {
            return Err(AegonError::Verification(
                "two-layer slot probe slot_bits do not match H_slot(slot_ctr, label)",
            ));
        }
        // PCS opening against the destination shard's index_poly.
        let point = bool_index_to_point::<E::ScalarField>(&sprobe.slot_bits);
        let mut tr = IOPTranscript::<E::ScalarField>::new(b"aegon.index.open");
        let ok = P::verify(
            &ctx.inner.verifier_param,
            &proof.dest_leaf.index_commitment,
            &point,
            &sprobe.evaluation,
            &sprobe.proof,
            &mut tr,
        )?;
        if !ok {
            return Err(AegonError::Verification(
                "two-layer slot probe opening did not verify against dest index commitment",
            ));
        }
        if slot_ctr_us < final_slot_idx {
            // Intermediate slot probe: must be non-empty and not
            // already this label (else placement would have stopped
            // earlier in the H_slot walk).
            if sprobe.evaluation.is_zero() {
                return Err(AegonError::Verification(
                    "two-layer intermediate slot is empty: shard placement was not canonical",
                ));
            }
            if sprobe.evaluation == h_label {
                return Err(AegonError::Verification(
                    "two-layer intermediate slot holds H_F(label): label would have been placed earlier",
                ));
            }
        } else {
            // Final probe: must hold this label's hash.
            if sprobe.evaluation != h_label {
                return Err(AegonError::Verification(
                    "two-layer final slot does not hold H_F(label)",
                ));
            }
            final_slot_bits = Some(sprobe.slot_bits.clone());
        }
    }

    let slot_bits = final_slot_bits.ok_or(AegonError::Verification(
        "two-layer label proof: empty slot trail (final probe missing)",
    ))?;
    Ok(LabelSlot {
        shard_id: proof.dest_shard_id,
        slot_bits,
    })
}

/// Combined two-layer lookup verifier. Mirrors `verify_sharded_lookup`
/// but reads from the two-layer proof shape. Returns `Ok(true)` iff
/// the residency proof verifies, the value opening verifies under
/// `dest_leaf.value_commitment`, and `H_F(value) == value_evaluation`.
pub fn verify_sharded_lookup_two_layer<E, P, H>(
    ctx: &ShardedVerifierContext<E, P>,
    commit: &ShardedEpochCommitment<E, P>,
    label: &Label,
    value: &Value,
    proof: &ShardedLookupProofTwoLayer<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    let slot = verify_lookup_label_two_layer::<E, P, H>(ctx, commit, label, &proof.label_proof)?;
    // Rebuild a `ShardedValueProof` from the bundled value parts +
    // the residency-verified dest leaf/path so we can reuse the
    // existing value-side verifier.
    let value_proof = ShardedValueProof {
        shard_id: slot.shard_id,
        slot_bits: slot.slot_bits.clone(),
        leaf: proof.label_proof.dest_leaf.clone(),
        merkle_path: proof.label_proof.dest_merkle_path.clone(),
        evaluation: proof.value_evaluation,
        proof: proof.value_proof.clone(),
    };
    if !verify_lookup_value::<E, P, H>(ctx, commit, &slot, value, &value_proof)? {
        return Ok(false);
    }
    Ok(true)
}

/// Verify a two-layer consistency proof. Returns `Ok(true)` iff the
/// route, intra-shard openings, and rand_value openings all verify
/// AND every `s0 == s1` equality (the per-slot rand_index pair plus
/// the final rand_value pair) holds. Returns `Ok(false)` when the
/// crypto is well-formed but the slot was disturbed (legitimate
/// rejection); `Err` only on malformed proofs.
pub fn verify_sharded_consistency_two_layer<E, P, H>(
    ctx: &ShardedVerifierContext<E, P>,
    s0_commit: &ShardedEpochCommitment<E, P>,
    s1_commit: &ShardedEpochCommitment<E, P>,
    label: &Label,
    proof: &ShardedConsistencyProofTwoLayer<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    if proof.route.is_empty() {
        return Err(AegonError::Verification(
            "two-layer consistency proof has an empty route trail",
        ));
    }
    if proof.slots.is_empty() {
        return Err(AegonError::Verification(
            "two-layer consistency proof has an empty slot trail",
        ));
    }

    // (1) Walk the inter-shard route trail, re-deriving shard_id from
    // H_shard and confirming intermediate vs landing semantics.
    let final_idx = proof.route.len() - 1;
    for (ctr_us, rprobe) in proof.route.iter().enumerate() {
        let ctr = ctr_us as u64;
        let bits = if let Some(verifier) = ctx.vrf_verifier.as_ref() {
            if rprobe.vrf_proof.is_empty() {
                return Err(AegonError::Verification(
                    "two-layer consistency: vrf-mode route probe has empty vrf_proof",
                ));
            }
            verifier
                .verify_h_shard(ctr, label, &rprobe.vrf_proof, ctx.log_n_shards)
                .map_err(|e| match e {
                    super::hash::VrfVerifyError::Malformed(_) => AegonError::Verification(
                        "two-layer consistency route vrf_proof failed to parse",
                    ),
                    super::hash::VrfVerifyError::InvalidProof(_) => AegonError::Verification(
                        "two-layer consistency route vrf_proof did not verify",
                    ),
                })?
        } else {
            H::h_shard(ctr, label, ctx.log_n_shards)
        };
        let mut shard_id: u32 = 0;
        for (i, b) in bits.iter().enumerate() {
            if *b {
                shard_id |= 1u32 << i;
            }
        }
        if shard_id != rprobe.shard_id {
            return Err(AegonError::Verification(
                "two-layer consistency route shard_id does not match H_shard",
            ));
        }
        if ctr_us < final_idx {
            if rprobe.fullness_proof.is_none() {
                return Err(AegonError::Verification(
                    "two-layer consistency intermediate route probe lacks fullness proof",
                ));
            }
        } else {
            if rprobe.fullness_proof.is_some() {
                return Err(AegonError::Verification(
                    "two-layer consistency landing route probe carries fullness proof",
                ));
            }
            if rprobe.shard_id != proof.dest_shard_id {
                return Err(AegonError::Verification(
                    "two-layer consistency route landing shard_id does not match dest_shard_id",
                ));
            }
        }
    }

    // (2) Anchor the destination shard leaves under each epoch root.
    if verify_merkle_path::<E, P>(
        &proof.dest_leaf_s0,
        proof.dest_shard_id as usize,
        &proof.dest_merkle_path_s0,
    ) != s0_commit.merkle_root
    {
        return Err(AegonError::Verification(
            "two-layer consistency dest_merkle_path_s0 does not reconstruct s0 root",
        ));
    }
    if verify_merkle_path::<E, P>(
        &proof.dest_leaf_s1,
        proof.dest_shard_id as usize,
        &proof.dest_merkle_path_s1,
    ) != s1_commit.merkle_root
    {
        return Err(AegonError::Verification(
            "two-layer consistency dest_merkle_path_s1 does not reconstruct s1 root",
        ));
    }

    // (3) Intra-shard slot trail. Each probe must verify openings
    // under both s0 and s1 dest_leaf.rand_index_commitment AND its
    // s0 evaluation must match its s1 evaluation (slot undisturbed).
    let shard_log_capacity = ctx.shard_log_capacity();
    let mut final_slot_bits: Option<Vec<bool>> = None;
    for (slot_ctr_us, sprobe) in proof.slots.iter().enumerate() {
        let slot_ctr = slot_ctr_us as u64;
        let derived_bits = if let Some(verifier) = ctx.vrf_verifier.as_ref() {
            if sprobe.vrf_proof.is_empty() {
                return Err(AegonError::Verification(
                    "two-layer consistency: vrf-mode slot probe has empty vrf_proof",
                ));
            }
            verifier
                .verify_h_slot(slot_ctr, label, &sprobe.vrf_proof, shard_log_capacity)
                .map_err(|e| match e {
                    super::hash::VrfVerifyError::Malformed(_) => AegonError::Verification(
                        "two-layer consistency slot vrf_proof failed to parse",
                    ),
                    super::hash::VrfVerifyError::InvalidProof(_) => AegonError::Verification(
                        "two-layer consistency slot vrf_proof did not verify",
                    ),
                })?
        } else {
            H::h_slot(slot_ctr, label, shard_log_capacity)
        };
        if derived_bits != sprobe.slot_bits {
            return Err(AegonError::Verification(
                "two-layer consistency slot bits disagree with H_slot",
            ));
        }
        let point = bool_index_to_point::<E::ScalarField>(&sprobe.slot_bits);
        let mut tr_s0 = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_index.open");
        let ok_s0 = P::verify(
            &ctx.inner.verifier_param,
            &proof.dest_leaf_s0.rand_index_commitment,
            &point,
            &sprobe.rand_index_s0_eval,
            &sprobe.rand_index_s0_proof,
            &mut tr_s0,
        )?;
        let mut tr_s1 = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_index.open");
        let ok_s1 = P::verify(
            &ctx.inner.verifier_param,
            &proof.dest_leaf_s1.rand_index_commitment,
            &point,
            &sprobe.rand_index_s1_eval,
            &sprobe.rand_index_s1_proof,
            &mut tr_s1,
        )?;
        if !(ok_s0 && ok_s1) {
            return Ok(false);
        }
        if sprobe.rand_index_s0_eval != sprobe.rand_index_s1_eval {
            return Ok(false);
        }
        if slot_ctr_us == proof.slots.len() - 1 {
            final_slot_bits = Some(sprobe.slot_bits.clone());
        }
    }
    let final_slot = final_slot_bits.ok_or(AegonError::Verification(
        "two-layer consistency: missing final slot",
    ))?;

    // (4) rand_value at the final slot.
    let value_point = bool_index_to_point::<E::ScalarField>(&final_slot);
    let mut tr_v_s0 = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_value.open");
    let ok_v_s0 = P::verify(
        &ctx.inner.verifier_param,
        &proof.dest_leaf_s0.rand_value_commitment,
        &value_point,
        &proof.value_rand_s0_eval,
        &proof.value_rand_s0_proof,
        &mut tr_v_s0,
    )?;
    let mut tr_v_s1 = IOPTranscript::<E::ScalarField>::new(b"aegon.rand_value.open");
    let ok_v_s1 = P::verify(
        &ctx.inner.verifier_param,
        &proof.dest_leaf_s1.rand_value_commitment,
        &value_point,
        &proof.value_rand_s1_eval,
        &proof.value_rand_s1_proof,
        &mut tr_v_s1,
    )?;
    if !(ok_v_s0 && ok_v_s1) {
        return Ok(false);
    }
    if proof.value_rand_s0_eval != proof.value_rand_s1_eval {
        return Ok(false);
    }
    Ok(true)
}

/// Verify a sharded invariance proof for the transition `prev → next`.
///
/// Steps:
///   1. The announced `next.merkle_root` must reconstruct from
///      `next.per_shard` (auditor independently hashes the leaves).
///   2. Re-derive the `(r_index, r_value)` Fiat-Shamir scalars from
///      `audit_state.r_*` and the new data commits of each chain
///      group's shards, in shard-id order — the same hash the
///      coordinator used at publish time. With the default single
///      group that absorbs *all* shards, which is what every
///      deployment predating
///      [`chain_groups`](super::chain_groups) does.
///   3. For every shard `i`, run the standard single-shard invariance
///      chain check (paper Fig. 4) on `(prev.per_shard[i],
///      next.per_shard[i])` using *that shard's group's* scalars —
///      *not* the single-shard derivation, since the coordinator
///      committed every shard in a group to the same `r`.
///
/// On success, `audit_state` is advanced to the new chain scalars,
/// ready for the next transition.
pub fn verify_sharded_invariance<E, P>(
    ctx: &ShardedVerifierContext<E, P>,
    audit_state: &mut ShardedAuditState<E::ScalarField>,
    prev: &ShardedEpochCommitment<E, P>,
    next: &ShardedEpochCommitment<E, P>,
) -> Result<bool, AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    P::VerifierParam: akd_core::aegon_crypto::pcs::PCSGlobalParam,
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

    // (2) Re-derive each group's FS scalars from its own prev_r and
    // its own slice of the per-shard tuple.
    let plan = ctx.group_plan()?;
    if audit_state.groups() != plan.groups() {
        return Err(AegonError::Verification(
            "sharded audit: audit state's chain-group count does not match the verifier context",
        ));
    }
    let (new_r_index, new_r_value) = rederive_sharded_fs_scalars::<E, P>(
        &audit_state.r_index,
        &audit_state.r_value,
        next,
        plan,
        &ctx.inner.audit_fs,
    )?;

    // (3) Per-shard chain checks with that shard's group's scalars. Every group
    // element the auditor needs is in `prev.per_shard[i]` /
    // `next.per_shard[i]`, and `verify_chain` does the homomorphism
    // check directly on commitments.
    // Every shard runs against the same SRS by construction (one
    // (prover_param, verifier_param) is cloned out to all shards in
    // `ShardedAegon::setup`), so the audit-path sigma proof verifies
    // against the single shared verifier_param exposed on
    // `ctx.inner`. The index chain stays exact (non-zk per shard);
    // the value chain consumes each shard's own Schnorr proof.
    let vk = &ctx.inner.verifier_param;
    let zk_srs = akd_core::aegon_crypto::pcs::PCSGlobalParam::is_zk(vk);
    for i in 0..next.per_shard.len() {
        let prev_i = &prev.per_shard[i];
        let next_i = &next.per_shard[i];
        let g = plan.group_of(i);

        let index_ok = verify_chain::<E, P>(
            new_r_index[g],
            &prev_i.index_commitment,
            &next_i.index_commitment,
            &prev_i.rand_index_commitment,
            &next_i.rand_index_commitment,
            None,
            vk,
            &ctx.inner.audit_fs,
        );
        if !index_ok {
            return Ok(false);
        }
        // Paper §7 policy: under a hiding SRS every shard's value
        // chain MUST carry its own Schnorr proof. A shard skipping
        // re-randomisation would defeat the zk simulator argument
        // even if all other shards play by the rules — the audit
        // rejects globally on any shard's missing proof.
        if zk_srs && next_i.audit_value_blinding_proof.is_none() {
            return Ok(false);
        }
        let value_ok = verify_chain::<E, P>(
            new_r_value[g],
            &prev_i.value_commitment,
            &next_i.value_commitment,
            &prev_i.rand_value_commitment,
            &next_i.rand_value_commitment,
            next_i.audit_value_blinding_proof.as_ref(),
            vk,
            &ctx.inner.audit_fs,
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

// Superseded during the per-shard-DB / two-layer refactors. Kept for reference rather than deleted; nothing calls it.
#[allow(dead_code)]
/// Re-export of `aegon::hash::bool_index_to_usize` under a name that
/// doesn't collide with the field above. Used internally to dedup
/// in-batch slot claims by their canonical PCS index.
fn bool_index_to_usize_dims(bits: &[bool], dims: &[usize]) -> usize {
    super::hash::bool_index_to_usize(bits, dims)
}

// Superseded by the per-group `derive_chain_scalars`.
#[allow(dead_code)]
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
pub fn merkle_root<E: Pairing, P: AegonPcs<E>>(per_shard: &[EpochCommitment<E, P>]) -> EpochDigest {
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
        hash = if idx.is_multiple_of(2) {
            merkle_parent(&hash, sibling)
        } else {
            merkle_parent(sibling, &hash)
        };
        idx /= 2;
    }
    hash
}

/// Recompute the FS chain scalars the prover used at the transition
/// from `prev` to `next`. Auditors call this to bind their own
/// invariance check to the same scalars the server derived.
///
/// Returns one `(r_index, r_value)` pair per chain group, in group
/// order. `plan` must be the deployment's partition — with
/// [`GroupPlan::single`](super::chain_groups::GroupPlan::single) this
/// is the original directory-wide derivation.
pub fn rederive_sharded_fs_scalars<E: Pairing, P: AegonPcs<E>>(
    prev_r_index: &[E::ScalarField],
    prev_r_value: &[E::ScalarField],
    next: &ShardedEpochCommitment<E, P>,
    plan: super::chain_groups::GroupPlan,
    audit_fs: &super::audit_fs::AuditFsHooks<E, P>,
) -> Result<(Vec<E::ScalarField>, Vec<E::ScalarField>), AegonError> {
    if prev_r_index.len() != plan.groups() || prev_r_value.len() != plan.groups() {
        return Err(AegonError::Verification(
            "sharded audit: audit state does not carry one accumulator per chain group",
        ));
    }
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
    let index_groups = plan.split(&index_commits)?;
    let value_groups = plan.split(&value_commits)?;

    let r_index = index_groups
        .iter()
        .zip(prev_r_index)
        .map(|(commits, prev)| audit_fs.chain_scalar(b"aegon.sharded.fs.r_index", *prev, commits))
        .collect();
    let r_value = value_groups
        .iter()
        .zip(prev_r_value)
        .map(|(commits, prev)| audit_fs.chain_scalar(b"aegon.sharded.fs.r_value", *prev, commits))
        .collect();
    Ok((r_index, r_value))
}
