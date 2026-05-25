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
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tonic::transport::{Channel, Server};
use tonic::{Request, Response, Status};

use super::error::AegonError;
use super::types::AegonPcs;

// Generated tonic code for `aegon.masking.v1`.
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
                        },
                        Err(e) => {
                            eprintln!("masking producer {worker}: join error {e}");
                            tokio::time::sleep(Duration::from_millis(250)).await;
                            continue;
                        },
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
    pub async fn serve(
        self,
        addr: std::net::SocketAddr,
    ) -> Result<(), tonic::transport::Error> {
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
/// the cluster's masking server. The client runs its own private
/// tokio runtime so that shard code on rayon worker threads can use
/// it via blocking calls (mirrors `GrpcShardClient`'s sync surface).
pub struct MaskingClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    rt: Arc<Runtime>,
    inner: Arc<tokio::sync::Mutex<MaskingServiceClient<Channel>>>,
    _e: std::marker::PhantomData<(E, P)>,
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
        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .thread_name("aegon-masking-client")
                .build()
                .map_err(|e| AegonError::Config(format!("build masking rt: {e}")))?,
        );
        let channel = rt
            .block_on(async {
                tonic::transport::Endpoint::from_shared(endpoint.clone())
                    .map_err(|e| AegonError::Config(format!("masking endpoint: {e}")))?
                    .connect()
                    .await
                    .map_err(|e| AegonError::Config(format!("masking connect '{endpoint}': {e}")))
            })?;
        let inner = MaskingServiceClient::new(channel)
            .max_decoding_message_size(MAX_MSG_BYTES)
            .max_encoding_message_size(MAX_MSG_BYTES);
        Ok(Self {
            rt,
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
            _e: std::marker::PhantomData,
        })
    }

    /// Fetch one pre-built masking package for `num_vars` from the
    /// server. Synchronous wrapper over the gRPC call — safe to call
    /// from rayon worker threads.
    pub fn fetch_package(&self, num_vars: usize) -> Result<P::MaskingPackage, AegonError> {
        let inner = Arc::clone(&self.inner);
        let bytes = self.rt.block_on(async move {
            let mut client = inner.lock().await;
            client
                .get_masking_package(Request::new(MaskingPackageRequest {
                    num_vars: num_vars as u32,
                }))
                .await
                .map_err(status_to_err)
                .map(|r| r.into_inner().package_uncompressed)
        })?;
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
            rt: Arc::clone(&self.rt),
            inner: Arc::clone(&self.inner),
            _e: std::marker::PhantomData,
        }
    }
}
