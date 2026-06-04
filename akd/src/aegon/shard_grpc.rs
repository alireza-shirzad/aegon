//! gRPC transport for sharded Aegon.
//!
//! The in-process `ShardedAegon` coordinator owns a `Vec<Aegon<E,P,H>>`
//! and dispatches the seven low-level shard methods locally. To make
//! shards live on different machines, those local calls become RPCs
//! against this module's `ShardService`. Everything is bytes-tunneled
//! over a tiny proto schema; the heavy lifting is arkworks
//! `CanonicalSerialize`/`Deserialize` on both sides.
//!
//! Three things live here:
//!   * [`ShardHandle`] — the trait the coordinator dispatches through.
//!     Implemented by `Aegon` (in-process) and [`GrpcShardClient`]
//!     (remote).
//!   * [`ShardServer`] — wraps an `Aegon` and exposes it over tonic.
//!     The `aegon_shard_server` binary just plumbs CLI args into
//!     `ShardServer::serve`.
//!   * [`GrpcShardClient`] — implements `ShardHandle` by making
//!     blocking RPCs from a private tokio runtime. Sync surface so
//!     the rayon-parallel publish loop in `ShardedAegon` stays the
//!     same shape regardless of transport.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tokio::runtime::Runtime;
use tokio::sync::{Mutex as AsyncMutex, RwLock as AsyncRwLock};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Identity, Server, ServerTlsConfig};
use tonic::{Request, Response, Status};

use super::config::VerifierContext;
use super::error::AegonError;
use super::hash::{HashSuite, Sha256Hash};
use super::server::Aegon;
use super::db::{key_shard_state, Db, DbOp, DbSource, RedisDb};
use super::server::AegonCheckpoint;
use super::sharded::ShardWrite;
use super::types::{AegonPcs, EpochCommitment, HistoryOpenings, Label, Value};

// Generated tonic code lives in this module. `tonic-build` emits one
// rust module per proto package; ours is `aegon.shard.v1`.
pub mod proto {
    tonic::include_proto!("aegon.shard.v1");
}

use proto::shard_service_client::ShardServiceClient;
use proto::shard_service_server::{ShardService, ShardServiceServer};
use proto::{
    CommitmentResponse, Empty, OpenResponse, PublishPhase1Request, PublishPhase1Response,
    PublishPhase2Request, PublishPhase2Response, ReconfigurePrefillRequest,
    ReconfigurePrefillResponse, SlotEpochRequest, SlotOccupiedResponse, SlotRequest,
};

// ---------- wire encoding helpers --------------------------------------

// Use uncompressed (un)serialization on the gRPC wire. Compressed
// format halves payload size but each G1Affine decode pays a Tonelli–
// Shanks square root in Fp (~5 µs/pt on Bn254); for a 10k-update
// publish the coordinator decodes ~1.8M points, which alone burned
// ~2 s of single-thread CPU per shard (×4 shards via rayon ≈ 2 s wall).
// Uncompressed is a straight memcpy and skips the curve-membership
// check, since both sides of the wire run our own binaries and we
// trust the points.
fn encode<T: CanonicalSerialize>(t: &T) -> Result<Vec<u8>, AegonError> {
    let mut buf = Vec::with_capacity(t.uncompressed_size());
    t.serialize_uncompressed(&mut buf)
        .map_err(|e| AegonError::Config(format!("encode: {e}")))?;
    Ok(buf)
}

fn decode<T: CanonicalDeserialize>(bytes: &[u8]) -> Result<T, AegonError> {
    T::deserialize_uncompressed_unchecked(bytes)
        .map_err(|e| AegonError::Config(format!("decode: {e}")))
}

fn err_to_status(e: AegonError) -> Status {
    Status::internal(format!("{e}"))
}

fn status_to_err(s: Status) -> AegonError {
    AegonError::Config(format!("grpc: {} ({})", s.message(), s.code()))
}

// ---------- ShardHandle trait ------------------------------------------

/// What the `ShardedAegon` coordinator needs from each shard. The
/// in-process implementation (`Aegon<E, P, H>`) just forwards each
/// method; the remote implementation ([`GrpcShardClient`]) makes a
/// blocking RPC. The trait is intentionally sync — callers like
/// `ShardedAegon::publish` use rayon's `par_iter_mut`, which is
/// incompatible with `async fn`.
pub trait ShardHandle<E, P, H>: Send + Sync
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    fn publish_phase_1_at_slots(
        &mut self,
        batch: &[ShardWrite<E::ScalarField>],
    ) -> Result<(P::Commitment, P::Commitment), AegonError>;

    fn publish_phase_2(
        &mut self,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
    ) -> Result<(EpochCommitment<E, P>, HistoryOpenings<E, P>), AegonError>;

    fn is_index_slot_occupied(&self, slot_bits: &[bool]) -> bool;

    fn open_index_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    fn open_value_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    fn open_rand_index_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    fn open_rand_value_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    /// Open the live `rand_value_poly` at `slot_bits`. Mirrors
    /// `Aegon::open_rand_value_at_slot_current`. Used by
    /// `ShardedAegon::lookup_history` to attach a freshness
    /// attestation to the history bundle.
    fn open_rand_value_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    /// Open the live `rand_index_poly` at `slot_bits`. Label-side
    /// mirror; used by `ShardedAegon::lookup_label_history`.
    fn open_rand_index_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    /// Re-mask a publish-time non-ZK value-history entry into a
    /// hiding one. The coordinator calls this during
    /// `ShardedAegon::lookup_history` for every stored entry it
    /// reads from the DB before serving the bundle to a user — the
    /// masking-server protocol is applied at user-facing time, not
    /// at publish time.
    fn remask_value_history_entry(
        &self,
        entry: super::sharded::StoredValueHistoryEntry<E, P>,
    ) -> Result<super::sharded::StoredValueHistoryEntry<E, P>, AegonError>;

    fn current_commitment(&self) -> EpochCommitment<E, P>;

    /// Per-shard verifier context. Only really needed at coordinator
    /// setup time; we read it from shard 0 to build the public
    /// `ShardedVerifierContext`.
    fn verifier_context(&self) -> VerifierContext<E, P>;

    fn log_capacity(&self) -> usize;

    /// Pre-populate this shard's polynomials with `count` random
    /// `(slot, h_label, h_value)` entries. Drives the publish bench's
    /// fill-percentage sweep.
    ///
    /// Only callable on a freshly-`init`-ed shard at epoch 0 with no
    /// pending publish (matches `Aegon::prefill_random`). Default
    /// returns `AegonError::Config(...)` so the gRPC `ShardClient`
    /// impl, which has no path to call into a remote shard's prefill,
    /// can keep the no-op default — remote prefill is driven at boot
    /// time via `aegon_shard_server --prefill-count`.
    fn prefill_random_in_place(
        &mut self,
        _count: usize,
        _seed: u64,
    ) -> Result<(), AegonError> {
        Err(AegonError::Config(
            "prefill_random_in_place not supported via this transport — set --prefill-count at shard boot time"
                .into(),
        ))
    }

    /// Wipe this shard back to its post-setup empty state without
    /// regenerating the SRS or dropping the transport. The in-process
    /// impl calls [`Aegon::clear_dictionary`] (cheap; restores from a
    /// stashed setup baseline). The default returns `Err` so the
    /// remote-transport impl, which would need a `ClearDictionary`
    /// RPC, fails loudly until that RPC is wired up.
    fn clear_dictionary(&mut self) -> Result<(), AegonError> {
        Err(AegonError::Config(
            "clear_dictionary not supported via this transport — wire up the ClearDictionary RPC"
                .into(),
        ))
    }
}

// ---------- in-process impl: Aegon directly is a ShardHandle -----------

impl<E, P, H> ShardHandle<E, P, H> for Aegon<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync,
    P::ProverParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Send + Sync,
    P::VerifierParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Send + Sync,
    P::Commitment: Clone
        + Send
        + Sync
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Sub<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    P::Proof: Clone + Send + Sync,
    P::State: Send + Sync,
    P::Polynomial: Send + Sync,
    P::Point: Send + Sync,
    P::Evaluation: Send + Sync,
    H: HashSuite<E::ScalarField> + Send + Sync,
{
    fn publish_phase_1_at_slots(
        &mut self,
        batch: &[ShardWrite<E::ScalarField>],
    ) -> Result<(P::Commitment, P::Commitment), AegonError> {
        Aegon::publish_phase_1_at_slots(self, batch)
    }

    fn publish_phase_2(
        &mut self,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
    ) -> Result<(EpochCommitment<E, P>, HistoryOpenings<E, P>), AegonError> {
        Aegon::publish_phase_2(self, new_r_index, new_r_value)
    }

    fn is_index_slot_occupied(&self, slot_bits: &[bool]) -> bool {
        Aegon::is_index_slot_occupied(self, slot_bits)
    }

    fn open_index_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_index_at_slot(self, slot_bits)
    }

    fn open_value_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_value_at_slot(self, slot_bits)
    }

    fn open_rand_index_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_rand_index_at_slot_in_epoch(self, slot_bits, epoch)
    }

    fn open_rand_value_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_rand_value_at_slot_in_epoch(self, slot_bits, epoch)
    }

    fn open_rand_value_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_rand_value_at_slot_current(self, slot_bits)
    }

    fn open_rand_index_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_rand_index_at_slot_current(self, slot_bits)
    }

    fn remask_value_history_entry(
        &self,
        entry: super::sharded::StoredValueHistoryEntry<E, P>,
    ) -> Result<super::sharded::StoredValueHistoryEntry<E, P>, AegonError> {
        // Under a non-hiding SRS, publish_phase_2 already produced
        // plain non-ZK openings and the stored entries verify
        // directly. Pass-through.
        if !self.is_zk_srs() {
            return Ok(entry);
        }
        // Under a hiding SRS:
        //   - publish_phase_2 stored plain non-ZK proofs (cheap)
        //   - this shard has the per-epoch `tau_f` for value-side
        //     polys (kept in `EpochSnapshot.{value_tau,
        //     rand_value_tau}` independently of `retain_epoch_polys`)
        //   - the masking server (or an inline fallback) provides a
        //     fresh `MaskingPackage`
        // Combine the three to mint a hiding opening per stored
        // proof. Each `remask_value_side_proof` call internally
        // fetches/generates ONE masking package; we pay 3 packages
        // per entry. That's still ~5× cheaper than the inline-ZK
        // path at publish time because we only pay it on the
        // (presumably rare) history-lookup path, not on every batch
        // of 16K labels.
        let prev_epoch = entry.epoch.saturating_sub(1);
        let post_epoch = entry.epoch;
        // Fetch all three tau snapshots up front (cheap reads).
        let rand_value_pre_tau =
            self.rand_value_tau_at_epoch(prev_epoch).ok_or_else(|| {
                AegonError::Config(format!(
                    "remask_value_history_entry: rand_value tau missing for epoch {prev_epoch}"
                ))
            })?;
        let rand_value_post_tau =
            self.rand_value_tau_at_epoch(post_epoch).ok_or_else(|| {
                AegonError::Config(format!(
                    "remask_value_history_entry: rand_value tau missing for epoch {post_epoch}"
                ))
            })?;
        let value_post_tau = self.value_tau_at_epoch(post_epoch).ok_or_else(|| {
            AegonError::Config(format!(
                "remask_value_history_entry: value tau missing for epoch {post_epoch}"
            ))
        })?;

        // The three remasks are independent — each takes its own
        // masking package, has its own commitment + tau, and writes
        // its own output proof. `remask_value_side_proof` is `&self`
        // and the three commitments / evaluations / proofs / taus are
        // disjoint owned/borrowed slices, so rayon can run them in
        // parallel. With an inline-generated masking package this
        // saves ~2/3 of the wall (~50 ms → ~17 ms per entry); with a
        // real masking server it cuts the fetch+apply RTT × 3 down to
        // a single concurrent burst.
        let ((rand_value_pre_proof, rand_value_post_proof), value_post_proof) =
            rayon::join(
                || {
                    rayon::join(
                        || {
                            self.remask_value_side_proof(
                                &entry.prev_shard_commit.rand_value_commitment,
                                &entry.slot_bits,
                                &entry.rand_value_pre_eval,
                                entry.rand_value_pre_proof.clone(),
                                &rand_value_pre_tau,
                                b"aegon.rand_value.open",
                            )
                        },
                        || {
                            self.remask_value_side_proof(
                                &entry.post_shard_commit.rand_value_commitment,
                                &entry.slot_bits,
                                &entry.rand_value_post_eval,
                                entry.rand_value_post_proof.clone(),
                                &rand_value_post_tau,
                                b"aegon.rand_value.open",
                            )
                        },
                    )
                },
                || {
                    self.remask_value_side_proof(
                        &entry.post_shard_commit.value_commitment,
                        &entry.slot_bits,
                        &entry.value_post_eval,
                        entry.value_post_proof.clone(),
                        &value_post_tau,
                        b"aegon.value.open",
                    )
                },
            );
        let rand_value_pre_proof = rand_value_pre_proof?;
        let rand_value_post_proof = rand_value_post_proof?;
        let value_post_proof = value_post_proof?;
        Ok(super::sharded::StoredValueHistoryEntry {
            rand_value_pre_proof,
            rand_value_post_proof,
            value_post_proof,
            ..entry
        })
    }

    fn current_commitment(&self) -> EpochCommitment<E, P> {
        Aegon::current_commitment(self)
    }

    fn verifier_context(&self) -> VerifierContext<E, P> {
        Aegon::verifier_context(self)
    }

    fn prefill_random_in_place(
        &mut self,
        count: usize,
        seed: u64,
    ) -> Result<(), AegonError> {
        use ark_std::rand::SeedableRng;
        // Reset back to epoch-0 first so multi-stage fill-percent
        // sweeps can call this repeatedly on the same Aegon —
        // matches the new gRPC ReconfigurePrefill semantics
        // ("wipe and refill"). For a freshly-init-ed Aegon, this is
        // a no-op (zero polys, zero commitments stay zero).
        Aegon::reset_state(self)?;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        // `shard_id`/`db_source` arguments to Aegon::prefill_random are
        // legacy — the implementation ignores both (the comment on
        // server.rs:573 spells it out). Pass `DbSource::None` + `0`.
        Aegon::prefill_random(self, &mut rng, count, &super::DbSource::None, 0)
    }

    fn clear_dictionary(&mut self) -> Result<(), AegonError> {
        Aegon::clear_dictionary(self)
    }

    fn log_capacity(&self) -> usize {
        Aegon::log_capacity(self)
    }
}

// ---------- gRPC server: wraps an Aegon, exposes 8 RPCs ----------------

/// Server-side TLS material. Build via [`Self::from_pem_files`].
#[derive(Clone)]
pub struct ShardServerTlsConfig {
    inner: ServerTlsConfig,
}

impl ShardServerTlsConfig {
    /// Load a PEM-encoded server certificate + matching private key
    /// from disk. The cert may be a chain (concatenated PEM blocks);
    /// the key must be the matching leaf.
    pub fn from_pem_files(
        cert_path: &Path,
        key_path: &Path,
    ) -> Result<Self, AegonError> {
        let cert = std::fs::read(cert_path).map_err(|e| {
            AegonError::Config(format!("read tls cert '{}': {e}", cert_path.display()))
        })?;
        let key = std::fs::read(key_path).map_err(|e| {
            AegonError::Config(format!("read tls key '{}': {e}", key_path.display()))
        })?;
        let identity = Identity::from_pem(cert, key);
        Ok(Self {
            inner: ServerTlsConfig::new().identity(identity),
        })
    }
}

/// Server-side adapter. Wraps an `Aegon` instance behind an async
/// `RwLock` so the read-only RPCs (open_*, is_index_slot_occupied,
/// current_commitment) execute concurrently. Mutating RPCs
/// (publish_phase_1/phase_2) take the write lock and serialize as
/// expected.
pub struct ShardServer<E, P, H = Sha256Hash>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    aegon: Arc<AsyncRwLock<Aegon<E, P, H>>>,
    /// Optional DB client (Redis or RocksDB, depending on
    /// `DbSource`). When set, the shard writes its
    /// [`AegonCheckpoint`] to `aegon:shard:{shard_id}:state` after
    /// every successful `publish_phase_2`. Powers shard-restart
    /// recovery (`aegon_shard_server --db-url` or `--db-path`).
    db: Option<Arc<dyn Db>>,
    shard_id: u32,
}

impl<E, P, H> ShardServer<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Send + Sync + 'static,
    P::VerifierParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Clone + Send + Sync + 'static,
    P::Commitment: CanonicalSerialize
        + CanonicalDeserialize
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Sub<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    P::Proof: CanonicalSerialize + Send + Sync + 'static,
    P::State: Send + Sync + 'static,
    P::Polynomial: Send + Sync + 'static,
    P::Point: Send + Sync + 'static,
    P::Evaluation: Send + Sync + 'static,
    H: HashSuite<E::ScalarField> + Send + Sync + 'static,
    EpochCommitment<E, P>: CanonicalSerialize + Send + Sync + 'static,
{
    pub fn new(aegon: Aegon<E, P, H>) -> Self {
        Self {
            aegon: Arc::new(AsyncRwLock::new(aegon)),
            db: None,
            shard_id: 0,
        }
    }

    /// Same as `new`, but also wires up the DB-backed durability
    /// barrier — every successful `publish_phase_2` writes the new
    /// `AegonCheckpoint` to `aegon:shard:{shard_id}:state`. The
    /// shard server binary calls this when started with `--db-url`
    /// (Redis) or `--db-path` (RocksDB).
    pub fn new_with_checkpoint(
        aegon: Aegon<E, P, H>,
        db_source: DbSource,
        shard_id: u32,
    ) -> Result<Self, AegonError> {
        let db: Arc<dyn Db> = match db_source {
            DbSource::None => {
                return Err(AegonError::Config(
                    "ShardServer::new_with_checkpoint requires a Redis or Rocks DbSource".into(),
                ));
            },
            DbSource::Redis(url) => Arc::new(RedisDb::connect(&url)?),
            DbSource::Rocks(path) => Arc::new(crate::aegon::db::RocksDb::open(&path)?),
        };
        Ok(Self {
            aegon: Arc::new(AsyncRwLock::new(aegon)),
            db: Some(db),
            shard_id,
        })
    }

    /// Bind and serve indefinitely on `addr` (plaintext HTTP/2).
    pub async fn serve(
        self,
        addr: std::net::SocketAddr,
    ) -> Result<(), tonic::transport::Error> {
        let service = Self::wrap_service(self);
        Server::builder().add_service(service).serve(addr).await
    }

    /// Bind and serve indefinitely on `addr` with TLS. The
    /// coordinator-side client connects via
    /// [`GrpcShardClientConfig::with_tls_ca`].
    pub async fn serve_with_tls(
        self,
        addr: std::net::SocketAddr,
        tls: ShardServerTlsConfig,
    ) -> Result<(), tonic::transport::Error> {
        let service = Self::wrap_service(self);
        Server::builder()
            .tls_config(tls.inner)?
            .add_service(service)
            .serve(addr)
            .await
    }

    /// Wrap `self` as a tonic service with the message-size limits the
    /// cluster needs. Tonic defaults to 4 MiB, which is fine for tiny
    /// batches but breaks publish_phase_1 once §6.4 openings per shard
    /// cross that threshold (happens around batch=10k for a 4-shard
    /// cluster). Set to 1 GiB on both directions so the limit is
    /// effectively never reached.
    fn wrap_service(this: Self) -> ShardServiceServer<Self> {
        const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;
        ShardServiceServer::new(this)
            .max_decoding_message_size(MAX_MSG_BYTES)
            .max_encoding_message_size(MAX_MSG_BYTES)
    }
}

#[tonic::async_trait]
impl<E, P, H> ShardService for ShardServer<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Send + Sync + 'static,
    P::VerifierParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Clone + Send + Sync + 'static,
    P::Commitment: CanonicalSerialize
        + CanonicalDeserialize
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Sub<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    P::Proof: CanonicalSerialize + Send + Sync + 'static,
    P::State: Send + Sync + 'static,
    P::Polynomial: Send + Sync + 'static,
    P::Point: Send + Sync + 'static,
    P::Evaluation: Send + Sync + 'static,
    H: HashSuite<E::ScalarField> + Send + Sync + 'static,
    EpochCommitment<E, P>: CanonicalSerialize + Send + Sync + 'static,
    HistoryOpenings<E, P>: CanonicalSerialize + Send + Sync + 'static,
    AegonCheckpoint<E, P>: CanonicalSerialize + Send + Sync + 'static,
{
    async fn publish_phase1_at_slots(
        &self,
        req: Request<PublishPhase1Request>,
    ) -> Result<Response<PublishPhase1Response>, Status> {
        let batch: Vec<ShardWrite<E::ScalarField>> =
            decode(&req.into_inner().batch_bytes).map_err(err_to_status)?;
        let mut aegon = self.aegon.write().await;
        let (index_com, value_com) = aegon
            .publish_phase_1_at_slots(&batch)
            .map_err(err_to_status)?;
        Ok(Response::new(PublishPhase1Response {
            index_commitment: encode(&index_com).map_err(err_to_status)?,
            value_commitment: encode(&value_com).map_err(err_to_status)?,
        }))
    }

    async fn publish_phase2(
        &self,
        req: Request<PublishPhase2Request>,
    ) -> Result<Response<PublishPhase2Response>, Status> {
        let r = req.into_inner();
        let r_index: E::ScalarField = decode(&r.r_index).map_err(err_to_status)?;
        let r_value: E::ScalarField = decode(&r.r_value).map_err(err_to_status)?;
        let mut aegon = self.aegon.write().await;
        let (commit, history) = aegon
            .publish_phase_2(r_index, r_value)
            .map_err(err_to_status)?;

        // TODO(shard-checkpoint-fault-tolerance): per-publish
        // checkpoint write into the shard's own local DB is
        // **intentionally disabled** in the current architecture.
        // The system-wide DB now belongs to the coordinator alone;
        // shards don't communicate with any DB during steady-state
        // operation. The code structure below is preserved for the
        // future fault-tolerance feature: when re-enabled, each
        // shard would persist its own `(rand_index_poly,
        // rand_value_poly, KZH aux states)` delta to a private
        // local RocksDB after every phase_2, slot-keyed for fast
        // delta writes (~few hundred bytes per publish per shard
        // at 2^34 scale) and prefix-scan recovery on restart.
        //
        // Until that's enabled, a shard process restart loses its
        // polynomial state and the shard has to be re-driven from
        // scratch by the coord. Acceptable for bench/research where
        // shards rarely restart mid-run; not acceptable for prod.
        const SHARD_CHECKPOINT_ENABLED: bool = false;
        if SHARD_CHECKPOINT_ENABLED {
            if let Some(db) = &self.db {
                let ckpt = aegon.capture_checkpoint();
                let bytes = encode(&ckpt).map_err(err_to_status)?;
                db.write_atomic(&[DbOp::Set {
                    key: key_shard_state(self.shard_id),
                    value: bytes,
                }])
                .map_err(err_to_status)?;
            }
        }
        drop(aegon);

        Ok(Response::new(PublishPhase2Response {
            epoch_commitment: encode(&commit).map_err(err_to_status)?,
            history_openings: encode(&history).map_err(err_to_status)?,
        }))
    }

    async fn is_index_slot_occupied(
        &self,
        req: Request<SlotRequest>,
    ) -> Result<Response<SlotOccupiedResponse>, Status> {
        let slot: Vec<bool> = decode(&req.into_inner().slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        Ok(Response::new(SlotOccupiedResponse {
            occupied: aegon.is_index_slot_occupied(&slot),
        }))
    }

    async fn open_index_at_slot(
        &self,
        req: Request<SlotRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let slot: Vec<bool> = decode(&req.into_inner().slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon.open_index_at_slot(&slot).map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn open_value_at_slot(
        &self,
        req: Request<SlotRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let slot: Vec<bool> = decode(&req.into_inner().slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon.open_value_at_slot(&slot).map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn open_rand_index_at_slot_in_epoch(
        &self,
        req: Request<SlotEpochRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let r = req.into_inner();
        let slot: Vec<bool> = decode(&r.slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon
            .open_rand_index_at_slot_in_epoch(&slot, r.epoch)
            .map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn open_rand_value_at_slot_in_epoch(
        &self,
        req: Request<SlotEpochRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let r = req.into_inner();
        let slot: Vec<bool> = decode(&r.slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon
            .open_rand_value_at_slot_in_epoch(&slot, r.epoch)
            .map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn open_rand_value_at_slot_current(
        &self,
        req: Request<SlotRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let slot: Vec<bool> = decode(&req.into_inner().slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon
            .open_rand_value_at_slot_current(&slot)
            .map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn open_rand_index_at_slot_current(
        &self,
        req: Request<SlotRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let slot: Vec<bool> = decode(&req.into_inner().slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon
            .open_rand_index_at_slot_current(&slot)
            .map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn remask_value_history_entry(
        &self,
        req: Request<proto::RemaskValueHistoryEntryRequest>,
    ) -> Result<Response<proto::RemaskValueHistoryEntryResponse>, Status> {
        let entry: super::sharded::StoredValueHistoryEntry<E, P> =
            decode(&req.into_inner().entry_uncompressed).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let remasked = ShardHandle::<E, P, H>::remask_value_history_entry(&*aegon, entry)
            .map_err(err_to_status)?;
        Ok(Response::new(proto::RemaskValueHistoryEntryResponse {
            entry_uncompressed: encode(&remasked).map_err(err_to_status)?,
        }))
    }

    async fn current_commitment(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<CommitmentResponse>, Status> {
        let aegon = self.aegon.read().await;
        let commit = aegon.current_commitment();
        Ok(Response::new(CommitmentResponse {
            epoch_commitment: encode(&commit).map_err(err_to_status)?,
        }))
    }

    async fn get_verifier_context(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<proto::GetVerifierContextResponse>, Status> {
        let aegon = self.aegon.read().await;
        let vp = ShardHandle::<E, P, H>::verifier_context(&*aegon).verifier_param;
        let log_capacity = ShardHandle::<E, P, H>::log_capacity(&*aegon);
        Ok(Response::new(proto::GetVerifierContextResponse {
            verifier_param: encode(&vp).map_err(err_to_status)?,
            log_capacity: log_capacity as u64,
        }))
    }

    async fn reconfigure_prefill(
        &self,
        req: Request<ReconfigurePrefillRequest>,
    ) -> Result<Response<ReconfigurePrefillResponse>, Status> {
        let r = req.into_inner();
        let mut aegon = self.aegon.write().await;
        // Reset wipes the in-memory Aegon back to a fresh epoch-0
        // and recomputes the empty-poly commitments. Then prefill
        // sprays `count` random entries into the data polynomials.
        aegon.reset_state().map_err(err_to_status)?;
        if r.count > 0 {
            use ark_std::rand::SeedableRng;
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(r.seed);
            // shard_id / db_source are ignored by Aegon::prefill_random.
            aegon
                .prefill_random(&mut rng, r.count as usize, &super::DbSource::None, 0)
                .map_err(err_to_status)?;
        }
        Ok(Response::new(ReconfigurePrefillResponse {}))
    }

    async fn clear_dictionary(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<Empty>, Status> {
        let mut aegon = self.aegon.write().await;
        // Restores from the in-memory `setup_baseline` — see
        // `Aegon::clear_dictionary` on the server side. Cheap relative
        // to `reset_state` because the empty-poly commitments are
        // already stashed rather than re-derived. Fails if a publish
        // is in flight on this shard.
        aegon.clear_dictionary().map_err(err_to_status)?;
        Ok(Response::new(Empty {}))
    }
}

// ---------- gRPC client: implements ShardHandle ------------------------

/// Blocking client adapter. Holds a dedicated tokio runtime + a
/// tonic client. Each `ShardHandle` method is a sync wrapper that
/// `block_on`s the underlying async RPC. The runtime is owned by the
/// adapter; we never share it with the application's runtime, so
/// there's no "block_on inside a runtime" deadlock risk.
///
/// `verifier_context` is cached at construction time: it doesn't
/// change over the shard's lifetime, and the coordinator queries it
/// once at setup.
/// Retry policy for the client. Every RPC retries on transport-level
/// failures (server unreachable, broken connection); the body of a
/// `Status::internal` reply — which is what surfaces a protocol-level
/// error from the shard's `Aegon` — is **not** retried, since those
/// errors are deterministic.
#[derive(Clone, Debug)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            initial_backoff: Duration::from_millis(50),
            max_backoff: Duration::from_secs(2),
        }
    }
}

impl RetryPolicy {
    pub fn off() -> Self {
        Self {
            max_attempts: 1,
            initial_backoff: Duration::from_millis(0),
            max_backoff: Duration::from_millis(0),
        }
    }
}

/// Connection options for [`GrpcShardClient::connect_with`].
pub struct GrpcShardClientConfig<E: Pairing, P: AegonPcs<E>> {
    pub endpoint: String,
    pub verifier_context: VerifierContext<E, P>,
    pub log_capacity: usize,
    /// PEM-encoded CA bundle used to verify the shard's server cert.
    /// `None` → plaintext HTTP/2 (matches the server side default).
    pub tls_ca_pem: Option<Vec<u8>>,
    /// Optional SNI / domain name override for TLS. Useful when the
    /// endpoint URL holds an IP but the cert is for a hostname.
    pub tls_domain: Option<String>,
    pub retry: RetryPolicy,
    pub connect_timeout: Option<Duration>,
}

impl<E: Pairing, P: AegonPcs<E>> GrpcShardClientConfig<E, P> {
    pub fn new(
        endpoint: String,
        verifier_context: VerifierContext<E, P>,
        log_capacity: usize,
    ) -> Self {
        Self {
            endpoint,
            verifier_context,
            log_capacity,
            tls_ca_pem: None,
            tls_domain: None,
            retry: RetryPolicy::default(),
            connect_timeout: Some(Duration::from_secs(10)),
        }
    }

    pub fn with_tls_ca(mut self, ca_pem: Vec<u8>) -> Self {
        self.tls_ca_pem = Some(ca_pem);
        self
    }

    pub fn with_tls_domain(mut self, domain: impl Into<String>) -> Self {
        self.tls_domain = Some(domain.into());
        self
    }

    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }
}

pub struct GrpcShardClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    runtime: Arc<Runtime>,
    client: Arc<AsyncMutex<ShardServiceClient<Channel>>>,
    cached_verifier_context: VerifierContext<E, P>,
    cached_log_capacity: usize,
    retry: RetryPolicy,
}

impl<E, P> GrpcShardClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync,
    P::Commitment: CanonicalDeserialize + Send + Sync,
    P::VerifierParam: Clone + Send + Sync,
{
    /// Convenience: plaintext connect with default retry.
    pub fn connect(
        endpoint: String,
        verifier_context: VerifierContext<E, P>,
        log_capacity: usize,
    ) -> Result<Self, AegonError> {
        Self::connect_with(GrpcShardClientConfig::new(
            endpoint,
            verifier_context,
            log_capacity,
        ))
    }

    /// Connect with full configurability (TLS, timeouts, retries).
    pub fn connect_with(cfg: GrpcShardClientConfig<E, P>) -> Result<Self, AegonError> {
        // The coord's plan_phase_1_batches RPC fan-out (~16K
        // is_index_slot_occupied calls per fresh-cluster publish) is
        // I/O-bound — each call is a sub-ms gRPC RTT, not CPU work.
        // Default Runtime::new() sizes workers to CPU count, which on
        // an n2-standard-4 coord caps in-flight concurrency at 4 and
        // turns the fan-out into an 8 s serial wall. 64 workers
        // multiplex the same connection without contention.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(64)
            .enable_all()
            .thread_name("aegon-grpc-rt")
            .build()
            .map_err(|e| AegonError::Config(format!("tokio runtime: {e}")))?;

        let mut endpoint = tonic::transport::Endpoint::from_shared(cfg.endpoint.clone())
            .map_err(|e| AegonError::Config(format!("endpoint '{}': {e}", cfg.endpoint)))?;
        if let Some(t) = cfg.connect_timeout {
            endpoint = endpoint.connect_timeout(t);
        }
        if let Some(ca_pem) = &cfg.tls_ca_pem {
            let mut tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(ca_pem));
            if let Some(domain) = &cfg.tls_domain {
                tls = tls.domain_name(domain);
            }
            endpoint = endpoint.tls_config(tls).map_err(|e| {
                AegonError::Config(format!("tls config: {e}"))
            })?;
        }

        let client = runtime
            .block_on(endpoint.connect())
            .map_err(|e| AegonError::Config(format!("connect '{}': {e}", cfg.endpoint)))?;
        // See `ShardServer::wrap_service` for the rationale on the 1 GiB
        // limit — same trigger (batch=10k publish_phase_1 response).
        const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;
        let client = ShardServiceClient::new(client)
            .max_decoding_message_size(MAX_MSG_BYTES)
            .max_encoding_message_size(MAX_MSG_BYTES);

        Ok(Self {
            runtime: Arc::new(runtime),
            client: Arc::new(AsyncMutex::new(client)),
            cached_verifier_context: cfg.verifier_context,
            cached_log_capacity: cfg.log_capacity,
            retry: cfg.retry,
        })
    }
}

/// One-shot fetch of a shard's `(VerifierContext, log_capacity)` over
/// gRPC. Used by the coordinator at setup time so it doesn't have to
/// regenerate the (multi-GB, multi-minute) full SRS just to derive
/// its small verifier-side projection. Each call opens its own
/// throwaway runtime + channel — cheap because this runs once per
/// cluster boot.
///
/// Plaintext only — the bench/intra-cluster traffic is already
/// implicitly trusted (same VPC), matching the masking-server
/// rationale.
pub fn fetch_verifier_context_from_endpoint<E, P>(
    endpoint: String,
) -> Result<(VerifierContext<E, P>, usize), AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    P::VerifierParam: CanonicalDeserialize + Clone,
{
    let runtime = Runtime::new()
        .map_err(|e| AegonError::Config(format!("tokio runtime: {e}")))?;
    let ep = tonic::transport::Endpoint::from_shared(endpoint.clone())
        .map_err(|e| AegonError::Config(format!("endpoint '{endpoint}': {e}")))?
        .connect_timeout(Duration::from_secs(10));
    let channel = runtime
        .block_on(ep.connect())
        .map_err(|e| AegonError::Config(format!("connect '{endpoint}': {e}")))?;
    const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;
    let mut client = ShardServiceClient::new(channel)
        .max_decoding_message_size(MAX_MSG_BYTES)
        .max_encoding_message_size(MAX_MSG_BYTES);
    let resp = runtime
        .block_on(client.get_verifier_context(Request::new(Empty {})))
        .map_err(status_to_err)?
        .into_inner();
    let verifier_param: P::VerifierParam = decode(&resp.verifier_param)?;
    let log_capacity = resp.log_capacity as usize;
    let vctx = VerifierContext::new(log_capacity, verifier_param);
    Ok((vctx, log_capacity))
}

// `impl GrpcShardClient` continues below with `with_retry` and
// the rest of the client methods.
impl<E, P> GrpcShardClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync,
    P::Commitment: CanonicalDeserialize + Send + Sync,
    P::VerifierParam: Clone + Send + Sync,
{

    /// Run `op` against the client, retrying on transport-level
    /// failures only (`Status::code() == Unavailable | Unknown`). The
    /// `op` closure is async, awaited from this method's owned tokio
    /// runtime. Retries use exponential backoff capped at
    /// `retry.max_backoff`.
    fn with_retry<T, Fut, F>(&self, mut op: F) -> Result<T, AegonError>
    where
        F: FnMut(Arc<AsyncMutex<ShardServiceClient<Channel>>>) -> Fut,
        Fut: std::future::Future<Output = Result<T, Status>>,
    {
        self.runtime.block_on(async {
            let mut backoff = self.retry.initial_backoff;
            let mut last_err: Option<Status> = None;
            for attempt in 0..self.retry.max_attempts {
                match op(self.client.clone()).await {
                    Ok(v) => return Ok(v),
                    Err(s) => {
                        let retriable = matches!(
                            s.code(),
                            tonic::Code::Unavailable | tonic::Code::Unknown
                        );
                        last_err = Some(s);
                        if !retriable || attempt + 1 == self.retry.max_attempts {
                            break;
                        }
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(self.retry.max_backoff);
                    },
                }
            }
            Err(status_to_err(last_err.expect("at least one attempt")))
        })
    }
}

impl<E, P, H> ShardHandle<E, P, H> for GrpcShardClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync,
    P::ProverParam: Send + Sync,
    P::VerifierParam: Clone + Send + Sync,
    P::Commitment: CanonicalSerialize
        + CanonicalDeserialize
        + Clone
        + Send
        + Sync
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Sub<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    P::Proof: CanonicalDeserialize + Clone + Send + Sync,
    P::State: Send + Sync,
    P::Polynomial: Send + Sync,
    P::Point: Send + Sync,
    P::Evaluation: Send + Sync,
    H: HashSuite<E::ScalarField> + Send + Sync,
    EpochCommitment<E, P>: CanonicalDeserialize + Send + Sync,
    HistoryOpenings<E, P>: CanonicalDeserialize + Send + Sync,
{
    fn publish_phase_1_at_slots(
        &mut self,
        batch: &[ShardWrite<E::ScalarField>],
    ) -> Result<(P::Commitment, P::Commitment), AegonError> {
        let req = PublishPhase1Request {
            batch_bytes: encode(&batch.to_vec())?,
        };
        let resp = self.runtime.block_on(async {
            self.client
                .lock()
                .await
                .publish_phase1_at_slots(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        let idx: P::Commitment = decode(&inner.index_commitment)?;
        let val: P::Commitment = decode(&inner.value_commitment)?;
        Ok((idx, val))
    }

    fn publish_phase_2(
        &mut self,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
    ) -> Result<(EpochCommitment<E, P>, HistoryOpenings<E, P>), AegonError> {
        let req = PublishPhase2Request {
            r_index: encode(&new_r_index)?,
            r_value: encode(&new_r_value)?,
        };
        let resp = self.runtime.block_on(async {
            self.client
                .lock()
                .await
                .publish_phase2(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        let commit: EpochCommitment<E, P> = decode(&inner.epoch_commitment)?;
        let history: HistoryOpenings<E, P> = decode(&inner.history_openings)?;
        Ok((commit, history))
    }

    fn is_index_slot_occupied(&self, slot_bits: &[bool]) -> bool {
        // Hot path: plan_phase_1_batches calls this once per probe
        // (~16K times per fresh-cluster publish). Using `with_retry`
        // would hold the client's AsyncMutex across the whole RPC,
        // serializing every concurrent call onto a single in-flight
        // request and turning the fan-out into a serial wall. Take
        // the lock just long enough to clone the (cheap, Channel-
        // backed) client, then drive the RPC without holding it.
        let req = SlotRequest {
            slot_bits: encode(&slot_bits.to_vec()).expect("slot_bits encode"),
        };
        let client = self.client.clone();
        let result = self.runtime.block_on(async move {
            let mut client = client.lock().await.clone();
            client.is_index_slot_occupied(req).await
        });
        match result {
            Ok(resp) => resp.into_inner().occupied,
            // RPC unreachable; default to "occupied" so the
            // coordinator's open-addressing loop doesn't claim a slot
            // we can't actually verify.
            Err(_) => true,
        }
    }

    fn open_index_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
        };
        let resp = self.with_retry(move |client| {
            let req = req.clone();
            async move { client.lock().await.open_index_at_slot(req).await }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn open_value_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
        };
        let resp = self.with_retry(move |client| {
            let req = req.clone();
            async move { client.lock().await.open_value_at_slot(req).await }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn open_rand_index_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotEpochRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
            epoch,
        };
        let resp = self.with_retry(move |client| {
            let req = req.clone();
            async move {
                client
                    .lock()
                    .await
                    .open_rand_index_at_slot_in_epoch(req)
                    .await
            }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn open_rand_value_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotEpochRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
            epoch,
        };
        let resp = self.with_retry(move |client| {
            let req = req.clone();
            async move {
                client
                    .lock()
                    .await
                    .open_rand_value_at_slot_in_epoch(req)
                    .await
            }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn open_rand_value_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
        };
        let resp = self.with_retry(move |client| {
            let req = req.clone();
            async move {
                client
                    .lock()
                    .await
                    .open_rand_value_at_slot_current(req)
                    .await
            }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn open_rand_index_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
        };
        let resp = self.with_retry(move |client| {
            let req = req.clone();
            async move {
                client
                    .lock()
                    .await
                    .open_rand_index_at_slot_current(req)
                    .await
            }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn remask_value_history_entry(
        &self,
        entry: super::sharded::StoredValueHistoryEntry<E, P>,
    ) -> Result<super::sharded::StoredValueHistoryEntry<E, P>, AegonError> {
        let req = proto::RemaskValueHistoryEntryRequest {
            entry_uncompressed: encode(&entry)?,
        };
        let resp = self.with_retry(move |client| {
            let req = req.clone();
            async move {
                client
                    .lock()
                    .await
                    .remask_value_history_entry(req)
                    .await
            }
        })?;
        decode(&resp.into_inner().entry_uncompressed)
    }

    fn current_commitment(&self) -> EpochCommitment<E, P> {
        self.with_retry(|client| async move {
            client.lock().await.current_commitment(Empty {}).await
        })
        .and_then(|r| decode(&r.into_inner().epoch_commitment))
        .expect("current_commitment RPC")
    }

    fn verifier_context(&self) -> VerifierContext<E, P> {
        self.cached_verifier_context.clone()
    }

    fn log_capacity(&self) -> usize {
        self.cached_log_capacity
    }

    fn prefill_random_in_place(
        &mut self,
        count: usize,
        seed: u64,
    ) -> Result<(), AegonError> {
        let req = ReconfigurePrefillRequest {
            count: count as u64,
            seed,
        };
        let _ = self.with_retry(move |client| {
            let req = req.clone();
            async move { client.lock().await.reconfigure_prefill(req).await }
        })?;
        Ok(())
    }

    fn clear_dictionary(&mut self) -> Result<(), AegonError> {
        let _ = self.with_retry(|client| async move {
            client.lock().await.clear_dictionary(Empty {}).await
        })?;
        Ok(())
    }
}

// Unused but useful to anchor the type aliases at module-scope.
#[allow(dead_code)]
type _Label = Label;
#[allow(dead_code)]
type _Value = Value;
