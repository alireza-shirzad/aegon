// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Masking-server transport for ZK opening packages.
//!
//! The shard server's hiding (value-side) openings require an
//! opening-point-agnostic *masking package* (`KZHKMaskingPackage`) per
//! opening. Sampling + committing one on the shard's critical path
//! adds Pedersen-MSM work to every lookup; the masking server lifts
//! that work out of the request path by keeping a hot, pre-built
//! queue of packages and handing them out to shards on demand. Two
//! concurrent workers:
//!
//! * **Producer** — N background tasks call
//!   [`P::generate_masking_package`] in a loop, each push blocking on
//!   the shared bounded channel. Throughput auto-scales: when shards
//!   stop drawing, producers idle on `send`; when shards drain,
//!   producers fill at line rate.
//! * **Responder** — the gRPC server handler pops one package per
//!   `GetMaskingPackage` call and ships it on the wire. Each package
//!   is consumed exactly once.
//!
//! The masking server never sees the opening point, the polynomial,
//! the polynomial's commitment, or any shard secret — every package
//! is opening-point-agnostic by construction. The server's only
//! configuration is `(prover_param, num_vars)` plus queue size.
//!
//! Wire encoding is `serialize_uncompressed` end-to-end to match the
//! shard/coordinator transports — the package is consumed
//! immediately, so deserialise speed dominates wire size.

use std::sync::Arc;
use std::time::Duration;

use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tokio::sync::mpsc;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use super::error::AegonError;
use super::types::AegonPcs;

// Generated tonic code for `aegon.masking.v1`.
// Generated code carries no rustdoc; the crate-level `warn(missing_docs)`
// cannot be satisfied for types we do not author.
#[allow(missing_docs)]
pub mod proto {
    tonic::include_proto!("aegon.masking.v1");
}

use proto::masking_service_client::MaskingServiceClient;
use proto::masking_service_server::{MaskingService, MaskingServiceServer};
use proto::{MaskingPackageRequest, MaskingPackageResponse};

// ---------- wire encoding helpers --------------------------------------

fn encode<T: CanonicalSerialize>(t: &T) -> Result<Vec<u8>, AegonError> {
    let mut buf = Vec::with_capacity(t.uncompressed_size());
    t.serialize_uncompressed(&mut buf)
        .map_err(|e| AegonError::Config(format!("masking encode: {e}")))?;
    Ok(buf)
}

fn decode<T: CanonicalDeserialize>(bytes: &[u8]) -> Result<T, AegonError> {
    T::deserialize_uncompressed_unchecked(bytes)
        .map_err(|e| AegonError::Config(format!("masking decode: {e}")))
}

fn err_to_status(e: AegonError) -> Status {
    Status::internal(format!("{e}"))
}

fn status_to_err(s: Status) -> AegonError {
    AegonError::Config(format!("masking grpc: {} ({})", s.message(), s.code()))
}

// Match the shard/coordinator services — 1 GiB caps for both
// directions. Masking packages are smaller (KZH-k sparse `r` with
// ~`k * N^{1/k}` non-zero coefficients) but the same cap removes
// any ambient size pressure as `num_vars` grows.
const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;

// ---------- masking server --------------------------------------------

/// Server-side adapter. Holds a producer queue + a tonic
/// `MaskingService` handler. Build via [`Self::new`] and bind via
/// [`Self::serve`].
pub struct MaskingServer<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    rx: Arc<tokio::sync::Mutex<mpsc::Receiver<P::MaskingPackage>>>,
    expected_num_vars: u32,
}

impl<E, P> MaskingServer<E, P>
where
    E: Pairing + Send + Sync + 'static,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam: Send + Sync + 'static,
    P::MaskingPackage: CanonicalSerialize + Send + Sync + 'static,
    E::ScalarField: Send + Sync + 'static,
{
    /// Spawn `producer_count` background producer tasks against
    /// `prover_param`/`num_vars` and start a bounded queue of
    /// `queue_size` pre-built packages. Returns a configured server
    /// that you can `serve(...)` on a `SocketAddr`.
    ///
    /// Producers run on `tokio::task::spawn_blocking` because
    /// [`P::generate_masking_package`] does Pedersen-MSM work and
    /// would otherwise block the runtime's reactor threads. Each
    /// producer holds an `Arc<ProverParam>` clone — the param itself
    /// is large (gigabytes of `H_1` for KZH-k at `log_capacity=29`),
    /// so we hand out arc-clones rather than deep copies.
    pub fn new(
        prover_param: Arc<P::ProverParam>,
        num_vars: usize,
        queue_size: usize,
        producer_count: usize,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<P::MaskingPackage>(queue_size.max(1));
        for worker in 0..producer_count.max(1) {
            let pp = Arc::clone(&prover_param);
            let tx = tx.clone();
            tokio::spawn(async move {
                loop {
                    let pp = Arc::clone(&pp);
                    let pkg = match tokio::task::spawn_blocking(move || {
                        <P as akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme<E>>::
                            generate_masking_package(pp.as_ref(), num_vars)
                    })
                    .await
                    {
                        Ok(Ok(pkg)) => pkg,
                        Ok(Err(e)) => {
                            eprintln!("masking producer {worker}: PCS error {e:?}");
                            tokio::time::sleep(Duration::from_millis(250)).await;
                            continue;
                        }
                        Err(e) => {
                            eprintln!("masking producer {worker}: join error {e}");
                            tokio::time::sleep(Duration::from_millis(250)).await;
                            continue;
                        }
                    };
                    // Back-pressure: this `send` waits when the queue is
                    // full, so producers idle naturally if consumers
                    // aren't draining. A closed channel means the server
                    // is shutting down — bail.
                    if tx.send(pkg).await.is_err() {
                        eprintln!("masking producer {worker}: queue closed, exiting");
                        return;
                    }
                }
            });
        }
        Self {
            rx: Arc::new(tokio::sync::Mutex::new(rx)),
            expected_num_vars: num_vars as u32,
        }
    }

    /// Bind and serve plaintext gRPC on `addr` until the runtime is
    /// shut down. The masking server is intentionally intra-cluster
    /// only — TLS would just add latency on a path that already
    /// trusts its callers (the shards) by virtue of running inside
    /// the same VPC.
    pub async fn serve(self, addr: std::net::SocketAddr) -> Result<(), tonic::transport::Error> {
        let svc = MaskingServiceServer::new(self)
            .max_decoding_message_size(MAX_MSG_BYTES)
            .max_encoding_message_size(MAX_MSG_BYTES);
        Server::builder().add_service(svc).serve(addr).await
    }
}

#[tonic::async_trait]
impl<E, P> MaskingService for MaskingServer<E, P>
where
    E: Pairing + Send + Sync + 'static,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::MaskingPackage: CanonicalSerialize + Send + Sync + 'static,
    E::ScalarField: Send + Sync + 'static,
{
    async fn get_masking_package(
        &self,
        req: Request<MaskingPackageRequest>,
    ) -> Result<Response<MaskingPackageResponse>, Status> {
        let req = req.into_inner();
        if req.num_vars != self.expected_num_vars {
            return Err(Status::failed_precondition(format!(
                "masking server serves num_vars={} but client requested {}",
                self.expected_num_vars, req.num_vars
            )));
        }
        // The handler holds the receiver across this `recv` — only
        // one in-flight pop at a time. With a single-receiver mpsc
        // this is the natural fit; if we ever want N-way concurrent
        // pops we'd switch to `tokio::sync::broadcast` or N shadow
        // queues, but the producer queue is the bottleneck either
        // way.
        let pkg = {
            let mut rx = self.rx.lock().await;
            rx.recv()
                .await
                .ok_or_else(|| Status::unavailable("masking queue closed"))?
        };
        let bytes = encode(&pkg).map_err(err_to_status)?;
        Ok(Response::new(MaskingPackageResponse {
            package_uncompressed: bytes,
        }))
    }
}

// ---------- masking client --------------------------------------------

/// Client-side adapter. Each shard holds one [`MaskingClient`] for
/// the cluster's masking server.
///
/// The client owns a dedicated worker thread that runs its own tokio
/// runtime; all RPCs flow through a sync `std::sync::mpsc` request
/// channel + per-call `std::sync::mpsc` response channel. `fetch_package`
/// never calls `block_on`, so it is safe to invoke from **any** caller
/// thread — rayon workers, tokio runtime workers, tokio blocking pool,
/// or plain std threads. (A naive `self.rt.block_on(...)` would panic
/// with "Cannot start a runtime from within a runtime" when the
/// shard's gRPC handlers — which run on the shard's main tokio
/// runtime — call into the value-side opening path.)
pub struct MaskingClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    tx: std::sync::mpsc::Sender<FetchRequest>,
    _worker: Arc<MaskingWorker>,
    _e: std::marker::PhantomData<(E, P)>,
}

type FetchResponse = Result<Vec<u8>, AegonError>;

struct FetchRequest {
    num_vars: u32,
    resp: std::sync::mpsc::SyncSender<FetchResponse>,
}

/// Joinable handle for the worker thread + its runtime. The runtime
/// is shut down when the worker is dropped (which happens when the
/// last `MaskingClient` clone is dropped).
struct MaskingWorker {
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for MaskingWorker {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            // The worker exits once the request channel is closed
            // (sender drops trigger recv() to return Err). Join with a
            // short window so we don't hang shutdown.
            let _ = h.join();
        }
    }
}

impl<E, P> MaskingClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
    P::MaskingPackage: CanonicalDeserialize,
{
    /// Connect to a masking server at `endpoint` (e.g. `"http://10.0.0.5:5505"`).
    pub fn connect(endpoint: impl Into<String>) -> Result<Self, AegonError> {
        let endpoint: String = endpoint.into();
        let (req_tx, req_rx) = std::sync::mpsc::channel::<FetchRequest>();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), AegonError>>(1);
        let endpoint_for_worker = endpoint.clone();

        let handle = std::thread::Builder::new()
            .name("aegon-masking-client".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .thread_name("aegon-masking-rt")
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = ready_tx
                            .send(Err(AegonError::Config(format!("build masking rt: {e}"))));
                        return;
                    }
                };
                // Connect synchronously, once. After this, we never
                // call block_on again — incoming requests are dispatched
                // via rt.spawn(...) so this worker thread is free to
                // park on the std::mpsc::Receiver between requests.
                let client = match rt.block_on(async {
                    let ep = tonic::transport::Endpoint::from_shared(endpoint_for_worker.clone())
                        .map_err(|e| AegonError::Config(format!("masking endpoint: {e}")))?;
                    let ch = ep.connect().await.map_err(|e| {
                        AegonError::Config(format!("masking connect '{endpoint_for_worker}': {e}"))
                    })?;
                    Ok::<_, AegonError>(
                        MaskingServiceClient::new(ch)
                            .max_decoding_message_size(MAX_MSG_BYTES)
                            .max_encoding_message_size(MAX_MSG_BYTES),
                    )
                }) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };

                if ready_tx.send(Ok(())).is_err() {
                    return;
                }
                drop(ready_tx);

                // Pump the sync request channel from this std thread
                // (no tokio context → recv won't panic). Each request
                // is dispatched as an async task on `rt`; the task
                // awaits the RPC and sync-sends the response back to
                // the caller. Multiple in-flight requests run
                // concurrently on the runtime's worker.
                while let Ok(req) = req_rx.recv() {
                    let mut client = client.clone();
                    rt.spawn(async move {
                        let result = client
                            .get_masking_package(Request::new(MaskingPackageRequest {
                                num_vars: req.num_vars,
                            }))
                            .await
                            .map_err(status_to_err)
                            .map(|r| r.into_inner().package_uncompressed);
                        let _ = req.resp.send(result);
                    });
                }
                // Sender dropped → shutdown. Drop rt to stop the runtime.
                drop(rt);
            })
            .map_err(|e| AegonError::Config(format!("spawn masking worker: {e}")))?;

        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(AegonError::Config(
                    "masking worker died during connect".into(),
                ))
            }
        }

        Ok(Self {
            tx: req_tx,
            _worker: Arc::new(MaskingWorker {
                handle: Some(handle),
            }),
            _e: std::marker::PhantomData,
        })
    }

    /// Fetch one pre-built masking package for `num_vars` from the
    /// server. Safe to call from any thread — including tokio runtime
    /// workers, tokio blocking pool, and rayon workers — because the
    /// underlying RPC runs on the dedicated worker thread, never on
    /// the caller's thread.
    pub fn fetch_package(&self, num_vars: usize) -> Result<P::MaskingPackage, AegonError> {
        let (resp_tx, resp_rx) = std::sync::mpsc::sync_channel::<FetchResponse>(1);
        self.tx
            .send(FetchRequest {
                num_vars: num_vars as u32,
                resp: resp_tx,
            })
            .map_err(|_| AegonError::Config("masking worker has stopped".into()))?;
        let bytes = resp_rx
            .recv()
            .map_err(|_| AegonError::Config("masking response channel dropped".into()))??;
        decode::<P::MaskingPackage>(&bytes)
    }
}

impl<E, P> Clone for MaskingClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            _worker: Arc::clone(&self._worker),
            _e: std::marker::PhantomData,
        }
    }
}

// ---------- MaskingSource trait ---------------------------------------
//
// Unified interface over (a) the remote gRPC client used by cluster
// shards and (b) the in-process pool used by tests / single-shard
// in-process bench binaries. Aegon stores `Arc<dyn MaskingSource>`
// and asks the source for a package on every value-side opening.

/// A source of pre-built [`P::MaskingPackage`]s. Implementations:
/// - [`MaskingClient`]: fetches one package per RPC from a remote
///   masking server (cluster path).
/// - [`MaskingPool`]: pops one package from an in-process queue
///   filled by background producer threads (tests + small-regime
///   in-process bench path).
pub trait MaskingSource<E, P>: Send + Sync
where
    E: Pairing,
    P: AegonPcs<E>,
{
    /// Fetch one masking package. `num_vars` must match the
    /// `log_capacity` the source was built for; mismatched values are
    /// rejected by remote servers and ignored by the local pool (which
    /// is sized at construction time).
    fn fetch_package(&self, num_vars: usize) -> Result<P::MaskingPackage, AegonError>;
}

impl<E, P> MaskingSource<E, P> for MaskingClient<E, P>
where
    E: Pairing + Send + Sync,
    P: AegonPcs<E> + Send + Sync,
    P::MaskingPackage: CanonicalDeserialize,
{
    fn fetch_package(&self, num_vars: usize) -> Result<P::MaskingPackage, AegonError> {
        MaskingClient::fetch_package(self, num_vars)
    }
}

// ---------- MaskingClientPool (round-robin over many servers) ---------
//
// One shard backed by a single remote masking server caps at that
// server's per-package generation rate (~120 pkg/s for nv=27/k=9,
// ~185 pkg/s for nv=22/k=7). The small/medium regimes use only 1-2
// shards, so per-shard masking assignment can't spread load across
// more than 1-2 servers, which means provisioning extra masking VMs
// is wasted unless we round-robin requests across them at the shard.
//
// `MaskingClientPool` wraps `Vec<MaskingClient>` and a shared atomic
// cursor: every `fetch_package` increments the cursor and dispatches
// to `clients[cursor % N]`. Concurrent calls race on the cursor (this
// is fine — `Ordering::Relaxed` is enough because we only care that
// the increment is atomic, not its global ordering). Under a uniform
// arrival rate this gives even load across all N servers; under a
// burst it still spreads cleanly because every individual fetch picks
// the next slot.
pub struct MaskingClientPool<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    clients: Vec<MaskingClient<E, P>>,
    cursor: Arc<std::sync::atomic::AtomicUsize>,
}

impl<E, P> MaskingClientPool<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
    P::MaskingPackage: CanonicalDeserialize,
{
    /// Connect to every endpoint in `endpoints` and wrap the resulting
    /// clients in a round-robin pool. Returns an error if any
    /// individual connection fails (we don't degrade silently — the
    /// caller asked for N servers).
    pub fn connect_all(endpoints: &[String]) -> Result<Self, AegonError> {
        if endpoints.is_empty() {
            return Err(AegonError::Config(
                "MaskingClientPool::connect_all: endpoints is empty".into(),
            ));
        }
        let mut clients = Vec::with_capacity(endpoints.len());
        for ep in endpoints {
            clients.push(MaskingClient::<E, P>::connect(ep.clone())?);
        }
        Ok(Self {
            clients,
            cursor: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    pub fn fetch_package(&self, num_vars: usize) -> Result<P::MaskingPackage, AegonError> {
        let i = self
            .cursor
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.clients.len();
        self.clients[i].fetch_package(num_vars)
    }
}

impl<E, P> Clone for MaskingClientPool<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    fn clone(&self) -> Self {
        Self {
            clients: self.clients.clone(),
            cursor: Arc::clone(&self.cursor),
        }
    }
}

impl<E, P> MaskingSource<E, P> for MaskingClientPool<E, P>
where
    E: Pairing + Send + Sync,
    P: AegonPcs<E> + Send + Sync,
    P::MaskingPackage: CanonicalDeserialize,
{
    fn fetch_package(&self, num_vars: usize) -> Result<P::MaskingPackage, AegonError> {
        MaskingClientPool::fetch_package(self, num_vars)
    }
}

// ---------- MaskingPool (in-process source) ---------------------------

/// In-process masking-package pool. Mirrors the cluster
/// [`MaskingServer`]'s producer/queue design but skips the gRPC
/// transport entirely — `fetch_package` does a sync `recv()` on a
/// `std::sync::mpsc` channel filled by background producer threads.
///
/// Use this when Aegon runs in the same process as its callers
/// (tests, small-regime in-process bench binaries, dev tools). Cluster
/// shards use [`MaskingClient`] against the dedicated masking VM
/// instead.
pub struct MaskingPool<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    rx: Arc<std::sync::Mutex<std::sync::mpsc::Receiver<P::MaskingPackage>>>,
    /// Producer threads are joined here on drop (after `rx` drops,
    /// `send` fails, and the loop bails).
    workers: Arc<MaskingPoolWorkers>,
    expected_num_vars: usize,
    _e: std::marker::PhantomData<(E, P)>,
}

struct MaskingPoolWorkers {
    handles: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Drop for MaskingPoolWorkers {
    fn drop(&mut self) {
        // The pool's `rx` has been dropped (since `Drop` on Self runs
        // after fields are dropped — actually Rust drops fields after
        // Drop::drop returns, so we rely on the rx being dropped *after*
        // this method but before the workers join. To force shutdown
        // here we close the channel by taking the handles out and
        // joining them; the `send` in the workers will fail once the
        // outer pool is dropped because they each hold a clone of `tx`
        // and the receiver-side is in the pool. Joining is best-effort.
        if let Ok(mut hs) = self.handles.lock() {
            for h in hs.drain(..) {
                let _ = h.join();
            }
        }
    }
}

impl<E, P> MaskingPool<E, P>
where
    E: Pairing + Send + Sync + 'static,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam: Send + Sync + 'static,
    P::MaskingPackage: Send + 'static,
    E::ScalarField: Send + Sync + 'static,
{
    /// Spawn `producer_count` background threads producing
    /// [`P::MaskingPackage`] for `num_vars` against `prover_param`.
    /// Threads push into a bounded `std::sync::mpsc::sync_channel(queue_size)` —
    /// when the queue is full producers naturally idle on the
    /// blocking `send`, and when the pool is dropped they exit (since
    /// `send` returns `Err` once `rx` is dropped).
    pub fn new(
        prover_param: Arc<P::ProverParam>,
        num_vars: usize,
        queue_size: usize,
        producer_count: usize,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<P::MaskingPackage>(queue_size.max(1));
        let producer_count = producer_count.max(1);
        let mut handles = Vec::with_capacity(producer_count);
        for worker in 0..producer_count {
            let pp = Arc::clone(&prover_param);
            let tx = tx.clone();
            let h = std::thread::Builder::new()
                .name(format!("aegon-masking-pool-{worker}"))
                .spawn(move || loop {
                    let pkg = match <P as akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme<
                        E,
                    >>::generate_masking_package(
                        pp.as_ref(), num_vars
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("masking pool producer {worker}: PCS error {e:?}");
                            std::thread::sleep(Duration::from_millis(250));
                            continue;
                        }
                    };
                    if tx.send(pkg).is_err() {
                        return; // receiver dropped → pool shutting down
                    }
                })
                .expect("spawn masking pool producer thread");
            handles.push(h);
        }
        // tx is moved into each spawn; we also hold a copy here so the
        // channel doesn't close prematurely if all producers exit. Drop
        // it explicitly — we want producers to be the only senders so
        // that when they exit, the channel closes naturally for the
        // receiver.
        drop(tx);
        Self {
            rx: Arc::new(std::sync::Mutex::new(rx)),
            workers: Arc::new(MaskingPoolWorkers {
                handles: std::sync::Mutex::new(handles),
            }),
            expected_num_vars: num_vars,
            _e: std::marker::PhantomData,
        }
    }

    /// Pop one pre-built package from the queue. Blocks if the queue
    /// is empty; returns an error only when the pool has been shut
    /// down (all producer threads have exited).
    pub fn fetch_package(&self, num_vars: usize) -> Result<P::MaskingPackage, AegonError> {
        if num_vars != self.expected_num_vars {
            return Err(AegonError::Config(format!(
                "masking pool was built for num_vars={} but caller requested {num_vars}",
                self.expected_num_vars,
            )));
        }
        let rx = self
            .rx
            .lock()
            .map_err(|_| AegonError::Config("masking pool mutex poisoned".into()))?;
        rx.recv()
            .map_err(|_| AegonError::Config("masking pool shut down".into()))
    }
}

impl<E, P> Clone for MaskingPool<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    fn clone(&self) -> Self {
        Self {
            rx: Arc::clone(&self.rx),
            workers: Arc::clone(&self.workers),
            expected_num_vars: self.expected_num_vars,
            _e: std::marker::PhantomData,
        }
    }
}

impl<E, P> MaskingSource<E, P> for MaskingPool<E, P>
where
    E: Pairing + Send + Sync + 'static,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam: Send + Sync + 'static,
    P::MaskingPackage: Send + 'static,
    E::ScalarField: Send + Sync + 'static,
{
    fn fetch_package(&self, num_vars: usize) -> Result<P::MaskingPackage, AegonError> {
        MaskingPool::fetch_package(self, num_vars)
    }
}
