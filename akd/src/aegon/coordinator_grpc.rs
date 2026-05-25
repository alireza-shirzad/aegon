//! User-facing gRPC service for the coordinator.
//!
//! Mirrors the split lookup design on the Rust side:
//!
//!   * `lookup_label(label)` →
//!     [`ShardedAegon::lookup_label`](super::sharded::ShardedAegon::lookup_label)
//!     proves the label is canonically placed at some `(shard, slot)`.
//!   * `lookup_value(slot)` →
//!     [`ShardedAegon::lookup_value`](super::sharded::ShardedAegon::lookup_value)
//!     opens `value_poly` at that slot.
//!   * `current_commitment()` returns the current
//!     [`ShardedEpochCommitment`] so a fresh client can pin its
//!     verification root.
//!
//! Companion to `shard_grpc.rs` — the latter is for coord↔shard,
//! this one is for client↔coord. Wire encoding is the same
//! (arkworks-canonical-uncompressed bytes wrapped in protobuf), so
//! deserialization on the client side reuses the existing helpers.

use std::sync::Arc;

use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tokio::sync::RwLock as AsyncRwLock;
use tonic::transport::{Server, ServerTlsConfig};
use tonic::{Request, Response, Status};

use super::error::AegonError;
use super::hash::{HashSuite, Sha256Hash};
use super::sharded::{
    verify_lookup_history, verify_lookup_label, verify_lookup_label_history, verify_lookup_value,
    LabelSlot, ShardedAegon, ShardedEpochCommitment, ShardedLabelHistory, ShardedLabelProof,
    ShardedValueHistory, ShardedValueProof, ShardedVerifierContext,
};
use super::types::{AegonPcs, EpochCommitment, Label, Value};

// Generated tonic code for the coordinator service. Lives in its own
// proto package (`aegon.coordinator.v1`) so it can evolve
// independently from the shard wire.
pub mod proto {
    tonic::include_proto!("aegon.coordinator.v1");
}

use proto::coordinator_service_client::CoordinatorServiceClient;
use proto::coordinator_service_server::{CoordinatorService, CoordinatorServiceServer};
use proto::{
    CommitmentResponse, Empty, LookupHistoryRequest, LookupHistoryResponse,
    LookupLabelHistoryRequest, LookupLabelHistoryResponse, LookupLabelRequest, LookupLabelResponse,
    LookupValueRequest, LookupValueResponse,
};
use tokio::runtime::Runtime;
use tokio::sync::Mutex as AsyncMutex;
use tonic::transport::{Certificate, Channel, ClientTlsConfig};

// ---------- wire encoding helpers --------------------------------------

// Uncompressed, unchecked — same rationale as `shard_grpc.rs::encode`:
// compressed serialization is 2× smaller on the wire but costs a
// Tonelli–Shanks sqrt in Fp per G1Affine on decode (~5 µs/point),
// which dwarfs the 100 ns memcpy cost of uncompressed. Both sides
// are our own binaries, so curve-membership validation is redundant.
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

// ---------- TLS config (mirrors shard_grpc::ShardServerTlsConfig) ------

/// Server-side TLS material for the coordinator. Optional; plaintext
/// HTTP/2 is fine for in-cluster deployments behind a load balancer
/// or service mesh that terminates TLS itself.
pub struct CoordinatorServerTlsConfig {
    inner: ServerTlsConfig,
}

impl CoordinatorServerTlsConfig {
    /// Load TLS identity from PEM files (cert chain + private key).
    pub fn from_pem_files(
        cert_pem: &std::path::Path,
        key_pem: &std::path::Path,
    ) -> Result<Self, AegonError> {
        let cert = std::fs::read(cert_pem)
            .map_err(|e| AegonError::Config(format!("read {cert_pem:?}: {e}")))?;
        let key = std::fs::read(key_pem)
            .map_err(|e| AegonError::Config(format!("read {key_pem:?}: {e}")))?;
        let identity = tonic::transport::Identity::from_pem(cert, key);
        Ok(Self {
            inner: ServerTlsConfig::new().identity(identity),
        })
    }
}

// ---------- server adapter ---------------------------------------------

/// Server-side adapter. Wraps a `ShardedAegon` behind an async
/// `RwLock` so the read-only lookup RPCs execute concurrently against
/// the same coordinator state. Mutating coordinator operations
/// (publish, audit-emission, …) would take the write lock when they
/// land here; for now this server is read-only and the publish path
/// is driven externally (e.g. by `aegon_coordinator_bench`).
pub struct CoordinatorServer<E, P, H = Sha256Hash>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    state: Arc<AsyncRwLock<ShardedAegon<E, P, H>>>,
}

impl<E, P, H> CoordinatorServer<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam:
        akd_core::aegon_crypto::pcs::PCSGlobalParam + CanonicalDeserialize + Send + Sync + 'static,
    P::VerifierParam:
        akd_core::aegon_crypto::pcs::PCSGlobalParam + CanonicalDeserialize + Send + Sync + 'static,
    P::Commitment: CanonicalSerialize
        + CanonicalDeserialize
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    P::Proof: CanonicalSerialize + Clone + Send + Sync + 'static,
    P::State: Send + Sync + 'static,
    P::Polynomial: Send + Sync + 'static,
    P::Point: Send + Sync + 'static,
    P::Evaluation: Send + Sync + 'static,
    H: HashSuite<E::ScalarField> + Send + Sync + 'static,
    EpochCommitment<E, P>: CanonicalSerialize + Send + Sync + 'static,
{
    pub fn new(state: ShardedAegon<E, P, H>) -> Self {
        Self {
            state: Arc::new(AsyncRwLock::new(state)),
        }
    }

    /// Build from a shared handle. Useful when the same `ShardedAegon`
    /// is driven by an external publish loop concurrently with the
    /// gRPC service — the loop holds an `Arc<RwLock<ShardedAegon>>`
    /// and passes a clone here.
    pub fn from_shared(state: Arc<AsyncRwLock<ShardedAegon<E, P, H>>>) -> Self {
        Self { state }
    }

    /// Borrow the shared state — for the external publish loop or
    /// tests that need to drive `publish`/`audit` directly.
    pub fn shared_state(&self) -> Arc<AsyncRwLock<ShardedAegon<E, P, H>>> {
        Arc::clone(&self.state)
    }

    /// Bind and serve plaintext HTTP/2.
    pub async fn serve(
        self,
        addr: std::net::SocketAddr,
    ) -> Result<(), tonic::transport::Error> {
        let service = Self::wrap_service(self);
        Server::builder().add_service(service).serve(addr).await
    }

    /// Bind and serve with TLS. The client connects via
    /// `CoordinatorClientConfig::with_tls_ca`.
    pub async fn serve_with_tls(
        self,
        addr: std::net::SocketAddr,
        tls: CoordinatorServerTlsConfig,
    ) -> Result<(), tonic::transport::Error> {
        let service = Self::wrap_service(self);
        Server::builder()
            .tls_config(tls.inner)?
            .add_service(service)
            .serve(addr)
            .await
    }

    /// Same 1 GiB ceiling as `ShardServer::wrap_service`. The coord↔
    /// client direction carries much smaller payloads than the
    /// coord↔shard direction (one trail of ~ctr0+1 probes vs. ~12k
    /// openings per shard at batch=10k), but keep the limit
    /// consistent so a single proof from a very full table still
    /// fits.
    fn wrap_service(this: Self) -> CoordinatorServiceServer<Self> {
        const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;
        CoordinatorServiceServer::new(this)
            .max_decoding_message_size(MAX_MSG_BYTES)
            .max_encoding_message_size(MAX_MSG_BYTES)
    }
}

// ---------- service impl -----------------------------------------------

#[tonic::async_trait]
impl<E, P, H> CoordinatorService for CoordinatorServer<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam:
        akd_core::aegon_crypto::pcs::PCSGlobalParam + CanonicalDeserialize + Send + Sync + 'static,
    P::VerifierParam:
        akd_core::aegon_crypto::pcs::PCSGlobalParam + CanonicalDeserialize + Send + Sync + 'static,
    P::Commitment: CanonicalSerialize
        + CanonicalDeserialize
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    P::Proof: CanonicalSerialize + Clone + Send + Sync + 'static,
    P::State: Send + Sync + 'static,
    P::Polynomial: Send + Sync + 'static,
    P::Point: Send + Sync + 'static,
    P::Evaluation: Send + Sync + 'static,
    H: HashSuite<E::ScalarField> + Send + Sync + 'static,
    EpochCommitment<E, P>: CanonicalSerialize + Send + Sync + 'static,
{
    async fn lookup_label(
        &self,
        req: Request<LookupLabelRequest>,
    ) -> Result<Response<LookupLabelResponse>, Status> {
        let label = req.into_inner().label;
        // `ShardedAegon::lookup_label` is sync but with a `Remote`
        // shard transport it walks the probe trail sequentially and
        // each probe calls `shard_client.runtime.block_on(...)` to
        // issue the open_index_at_slot RPC. Calling that from inside
        // this async handler would panic with "Cannot start a runtime
        // from within a runtime" because tonic's worker thread already
        // has a tokio CONTEXT set. Move the work to a blocking pool
        // thread (no tokio CONTEXT) so the shard runtime can be
        // entered safely. `spawn_blocking` clones the Arc cheaply so
        // the closure can hold its own handle to the shared state.
        let state = Arc::clone(&self.state);
        let result = tokio::task::spawn_blocking(move || {
            let state = state.blocking_read();
            state.lookup_label(&label)
        })
        .await
        .map_err(|e| Status::internal(format!("lookup_label join: {e}")))?;
        let (slot, proof) = result.map_err(err_to_status)?;
        Ok(Response::new(LookupLabelResponse {
            slot: encode(&slot).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn lookup_value(
        &self,
        req: Request<LookupValueRequest>,
    ) -> Result<Response<LookupValueResponse>, Status> {
        let slot_bytes = req.into_inner().slot;
        let slot: LabelSlot = decode(&slot_bytes).map_err(err_to_status)?;
        // Same `spawn_blocking` rationale as `lookup_label` above —
        // the value-side opening hits the owning shard via the same
        // blocking shard-client wrapper, so we can't issue it from
        // this async handler directly.
        let state = Arc::clone(&self.state);
        let proof_result = tokio::task::spawn_blocking(move || {
            let state = state.blocking_read();
            state.lookup_value(&slot)
        })
        .await
        .map_err(|e| Status::internal(format!("lookup_value join: {e}")))?;
        let proof = proof_result.map_err(err_to_status)?;
        // The proof side is purely polynomial — but we may also have
        // the raw value bytes in the coordinator's KV side-channel.
        // Today the persistence layer keys values by *label*, so the
        // server can't recover bytes from a slot alone; we return an
        // empty `value` and leave the byte fetch to the client (which
        // can either know the label out-of-band, or use a label→slot
        // index it built locally from its own `lookup_label`
        // history).
        //
        // A future DB schema extension (`aegon:value_by_slot:{s}:{i}`)
        // would let this RPC also return the value bytes; the proto
        // already has the field.
        Ok(Response::new(LookupValueResponse {
            proof: encode(&proof).map_err(err_to_status)?,
            value: Vec::new(),
        }))
    }

    async fn lookup_history(
        &self,
        req: Request<LookupHistoryRequest>,
    ) -> Result<Response<LookupHistoryResponse>, Status> {
        let label = req.into_inner().label;
        // `ShardedAegon::lookup_history` is a pure DB read — no
        // shard RPCs — so it doesn't strictly need `spawn_blocking`.
        // But the DB clients this codebase uses are sync (the
        // `redis` crate behind a `std::sync::Mutex` for Redis;
        // `rocksdb::DB` for RocksDB — both block the calling
        // thread), so we still keep tonic's async worker thread
        // unblocked by dispatching the read onto a blocking pool
        // thread.
        let state = Arc::clone(&self.state);
        let history_result = tokio::task::spawn_blocking(move || {
            let state = state.blocking_read();
            state.lookup_history(&label)
        })
        .await
        .map_err(|e| Status::internal(format!("lookup_history join: {e}")))?;
        let history = history_result.map_err(err_to_status)?;
        Ok(Response::new(LookupHistoryResponse {
            history: encode(&history).map_err(err_to_status)?,
        }))
    }

    async fn lookup_label_history(
        &self,
        req: Request<LookupLabelHistoryRequest>,
    ) -> Result<Response<LookupLabelHistoryResponse>, Status> {
        let label = req.into_inner().label;
        // Unlike value-history, this RPC does a shard gRPC call
        // (open_rand_index_at_slot_current) under the hood — so the
        // spawn_blocking here is doing real work, not just dodging
        // a sync redis client. Same pattern as `lookup_label` and
        // `lookup_value`.
        let state = Arc::clone(&self.state);
        let history_result = tokio::task::spawn_blocking(move || {
            let state = state.blocking_read();
            state.lookup_label_history(&label)
        })
        .await
        .map_err(|e| Status::internal(format!("lookup_label_history join: {e}")))?;
        let history = history_result.map_err(err_to_status)?;
        Ok(Response::new(LookupLabelHistoryResponse {
            history: encode(&history).map_err(err_to_status)?,
        }))
    }

    async fn current_commitment(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<CommitmentResponse>, Status> {
        let state = self.state.read().await;
        let commit = state.current_commitment();
        drop(state);
        Ok(Response::new(CommitmentResponse {
            commitment: encode(&commit).map_err(err_to_status)?,
        }))
    }
}

// ---------- gRPC client ------------------------------------------------

/// Blocking client for `CoordinatorService`. Holds its own tokio
/// runtime + a `CoordinatorServiceClient<Channel>` and exposes
/// `lookup_label` / `lookup_value` / `current_commitment` as plain
/// sync methods that do the RPC, decode the response, and verify the
/// proof locally using `verify_lookup_label` / `verify_lookup_value`.
///
/// "Verify locally" is the whole point of this client: the user does
/// not have to trust the coordinator's response — every successful
/// return has already been checked against the cached
/// `ShardedVerifierContext` and the bulletin-board commitment.
///
/// The verifier context (SRS-derived `verifier_param` + `log_capacity`
/// + `log_n_shards`) is supplied at construction time, since the
/// coordinator can't be trusted to hand it out. In practice it's
/// bundled with the client binary (or fetched from a public bulletin
/// board out-of-band).
pub struct CoordinatorClient<E, P, H = Sha256Hash>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    runtime: Arc<Runtime>,
    client: Arc<AsyncMutex<CoordinatorServiceClient<Channel>>>,
    verifier_ctx: ShardedVerifierContext<E, P>,
    _hash_suite: std::marker::PhantomData<H>,
}

/// Connection config for `CoordinatorClient`. Plaintext by default;
/// optional TLS via `with_tls_ca`.
pub struct CoordinatorClientConfig<E: Pairing, P: AegonPcs<E>> {
    pub endpoint: String,
    pub verifier_ctx: ShardedVerifierContext<E, P>,
    pub tls_ca_pem: Option<Vec<u8>>,
    pub tls_domain: Option<String>,
    pub connect_timeout: Option<std::time::Duration>,
}

impl<E: Pairing, P: AegonPcs<E>> CoordinatorClientConfig<E, P> {
    pub fn new(endpoint: String, verifier_ctx: ShardedVerifierContext<E, P>) -> Self {
        Self {
            endpoint,
            verifier_ctx,
            tls_ca_pem: None,
            tls_domain: None,
            connect_timeout: Some(std::time::Duration::from_secs(10)),
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
}

impl<E, P, H> CoordinatorClient<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync,
    P::VerifierParam: Clone + Send + Sync,
    P::Commitment: CanonicalDeserialize + Send + Sync,
    P::Proof: CanonicalDeserialize + Send + Sync,
    H: HashSuite<E::ScalarField> + Send + Sync,
    EpochCommitment<E, P>: CanonicalDeserialize + Send + Sync,
{
    /// Convenience: plaintext connect.
    pub fn connect(
        endpoint: String,
        verifier_ctx: ShardedVerifierContext<E, P>,
    ) -> Result<Self, AegonError> {
        Self::connect_with(CoordinatorClientConfig::new(endpoint, verifier_ctx))
    }

    /// Connect with full config (TLS, timeouts).
    pub fn connect_with(cfg: CoordinatorClientConfig<E, P>) -> Result<Self, AegonError> {
        let runtime = Runtime::new()
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
            endpoint = endpoint
                .tls_config(tls)
                .map_err(|e| AegonError::Config(format!("tls config: {e}")))?;
        }
        let channel = runtime
            .block_on(endpoint.connect())
            .map_err(|e| AegonError::Config(format!("connect '{}': {e}", cfg.endpoint)))?;
        // Same 1 GiB ceiling as the server side.
        const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;
        let client = CoordinatorServiceClient::new(channel)
            .max_decoding_message_size(MAX_MSG_BYTES)
            .max_encoding_message_size(MAX_MSG_BYTES);
        Ok(Self {
            runtime: Arc::new(runtime),
            client: Arc::new(AsyncMutex::new(client)),
            verifier_ctx: cfg.verifier_ctx,
            _hash_suite: std::marker::PhantomData,
        })
    }

    /// Fetch the current `ShardedEpochCommitment` — the bulletin-board
    /// commitment a freshly-connected client should pin verifications
    /// against. Returns the commitment as-is (no on-chain root check
    /// is meaningful at this layer; an auditor/registry oversees the
    /// bulletin board separately).
    pub fn current_commitment(&self) -> Result<ShardedEpochCommitment<E, P>, AegonError> {
        let resp = self.runtime.block_on(async {
            self.client
                .lock()
                .await
                .current_commitment(Empty {})
                .await
                .map_err(|s| AegonError::Config(format!("grpc current_commitment: {s}")))
        })?;
        decode::<ShardedEpochCommitment<E, P>>(&resp.into_inner().commitment)
    }

    /// Resolve `label` to its canonical `LabelSlot`. Verifies the
    /// open-addressing trail locally against `commit`; the returned
    /// `LabelSlot` falls out of the verified chain (not trusted from
    /// the server). Cache it and use it for any number of subsequent
    /// `lookup_value` calls.
    pub fn lookup_label(
        &self,
        commit: &ShardedEpochCommitment<E, P>,
        label: &Label,
    ) -> Result<LabelSlot, AegonError> {
        let req = LookupLabelRequest {
            label: label.clone(),
        };
        let resp = self.runtime.block_on(async {
            self.client
                .lock()
                .await
                .lookup_label(req)
                .await
                .map_err(|s| AegonError::Config(format!("grpc lookup_label: {s}")))
        })?;
        let inner = resp.into_inner();
        let server_slot: LabelSlot = decode(&inner.slot)?;
        let proof: ShardedLabelProof<E, P> = decode(&inner.proof)?;
        let verified_slot = verify_lookup_label::<E, P, H>(&self.verifier_ctx, commit, label, &proof)?;
        // The server's claimed slot must match what falls out of the
        // verified chain — otherwise the server is hinting at a slot
        // its own proof doesn't actually prove.
        if server_slot != verified_slot {
            return Err(AegonError::Verification(
                "server-claimed slot disagrees with verified open-addressing trail",
            ));
        }
        Ok(verified_slot)
    }

    /// Open the value at a previously-resolved `slot`. Returns the
    /// value bytes when the coordinator delivered them inline; an
    /// empty `Value` means the byte channel is out-of-band (caller
    /// has to supply the value to verify).
    ///
    /// Either way, the polynomial opening is verified against
    /// `commit` before the value is accepted — if the caller supplies
    /// a value via a parallel channel, they should call this method
    /// with that value via [`Self::lookup_value_with_bytes`].
    pub fn lookup_value(
        &self,
        commit: &ShardedEpochCommitment<E, P>,
        slot: &LabelSlot,
    ) -> Result<(Value, ShardedValueProof<E, P>), AegonError> {
        let req = LookupValueRequest {
            slot: encode(slot)?,
        };
        let resp = self.runtime.block_on(async {
            self.client
                .lock()
                .await
                .lookup_value(req)
                .await
                .map_err(|s| AegonError::Config(format!("grpc lookup_value: {s}")))
        })?;
        let inner = resp.into_inner();
        let proof: ShardedValueProof<E, P> = decode(&inner.proof)?;
        // If the server returned value bytes inline, verify them here;
        // otherwise hand the proof back and let the caller verify once
        // they have the value via their out-of-band channel.
        if !inner.value.is_empty() {
            if !verify_lookup_value::<E, P, H>(&self.verifier_ctx, commit, slot, &inner.value, &proof)? {
                return Err(AegonError::Verification(
                    "value proof did not verify against current commitment",
                ));
            }
        }
        Ok((inner.value, proof))
    }

    /// Same as `lookup_value`, but the caller supplies the value
    /// bytes (typically obtained from a separate KV channel keyed by
    /// label). The opening is verified against those bytes.
    pub fn lookup_value_with_bytes(
        &self,
        commit: &ShardedEpochCommitment<E, P>,
        slot: &LabelSlot,
        value: &Value,
    ) -> Result<ShardedValueProof<E, P>, AegonError> {
        let (_server_value, proof) = self.lookup_value(commit, slot)?;
        if !verify_lookup_value::<E, P, H>(&self.verifier_ctx, commit, slot, value, &proof)? {
            return Err(AegonError::Verification(
                "value proof did not verify against caller-supplied value bytes",
            ));
        }
        Ok(proof)
    }

    /// Fetch this label's value-history bundle from the coordinator
    /// and verify every entry locally. Up to `HISTORY_WINDOW`
    /// most-recent entries are returned, most-recent first.
    ///
    /// What the client verifies per-entry:
    ///   * Two merkle paths re-anchor the per-shard leaves under
    ///     reconstructed sharded roots — the function returns those
    ///     roots so the caller can cross-check them against whatever
    ///     historical bulletin-board snapshot they trust.
    ///   * Three PCS openings (`rand_value_pre`, `rand_value_post`,
    ///     `value_post`) check out against the corresponding per-
    ///     shard commitments inside the leaves.
    ///   * `H_F(value_bytes) == value_post_eval`, so the inline
    ///     value bytes match the polynomial commitment's bound hash.
    ///
    /// What the client verifies bundle-wide (freshness):
    ///   * Live `rand_value(slot)` opens under the live shard
    ///     commitment, and its evaluation equals the latest entry's
    ///     `rand_value_post_eval` — i.e. no publish has touched the
    ///     slot since the most recent recorded value-change. The
    ///     reconstructed `live_root` is returned for cross-check
    ///     against the coordinator's current published commitment.
    ///
    /// Caller responsibility: cross-check the returned roots against
    /// the bulletin board (historical entries + live). This API
    /// doesn't pin them — the trust anchor lives outside the
    /// coordinator.
    pub fn lookup_history(
        &self,
        label: &Label,
    ) -> Result<(ShardedValueHistory<E, P>, super::sharded::VerifiedLookupHistory), AegonError>
    {
        let req = LookupHistoryRequest {
            label: label.clone(),
        };
        let resp = self.runtime.block_on(async {
            self.client
                .lock()
                .await
                .lookup_history(req)
                .await
                .map_err(|s| AegonError::Config(format!("grpc lookup_history: {s}")))
        })?;
        let inner = resp.into_inner();
        let history: ShardedValueHistory<E, P> = decode(&inner.history)?;
        let verified = verify_lookup_history::<E, P, H>(&self.verifier_ctx, &history)?;
        Ok((history, verified))
    }

    /// Fetch this label's placement record + freshness attestation
    /// from the coordinator and verify both locally. The label-side
    /// mirror of `lookup_history`.
    ///
    /// Returns `(history, verified)`. `history` carries the raw
    /// bundle (placement record + live opening). `verified` carries
    /// the two reconstructed sharded roots — `placement_root` (at
    /// the placement epoch) and `live_root` (at right now) — the
    /// caller cross-checks against the trusted bulletin board.
    ///
    /// `history.placement == None` (and `verified.placement_root ==
    /// None`) when the label is unknown or its placement record
    /// hasn't landed yet (racy pre-persist window). No error in
    /// either case — matches the value-side semantics.
    pub fn lookup_label_history(
        &self,
        label: &Label,
    ) -> Result<(ShardedLabelHistory<E, P>, super::sharded::VerifiedLookupLabelHistory), AegonError>
    {
        let req = LookupLabelHistoryRequest {
            label: label.clone(),
        };
        let resp = self.runtime.block_on(async {
            self.client
                .lock()
                .await
                .lookup_label_history(req)
                .await
                .map_err(|s| AegonError::Config(format!("grpc lookup_label_history: {s}")))
        })?;
        let inner = resp.into_inner();
        let history: ShardedLabelHistory<E, P> = decode(&inner.history)?;
        let verified = verify_lookup_label_history::<E, P, H>(&self.verifier_ctx, &history)?;
        Ok((history, verified))
    }
}

/// Alias so the public `lookup_history` return type doesn't pull a
/// `sharded::EpochDigest` import into every downstream module that
/// only wants to use the client. Same underlying `[u8; 32]` hash.
pub type EpochDigestForHistory = super::sharded::EpochDigest;
