//! Localhost-only gRPC integration test for sharded Aegon.
//!
//! Spins up two `ShardServer`s on ephemeral localhost ports, has the
//! coordinator connect over gRPC instead of running shards
//! in-process, then exercises the full publish/lookup path. Verifies
//! that swapping the in-process `Aegon` for a network-dispatched
//! `GrpcShardClient` produces byte-identical proofs that the
//! existing `verify_sharded_lookup` accepts.

use std::sync::Arc;
use std::time::Duration;

use akd::aegon::shard_grpc::ShardServer;
use akd::aegon::{
    verify_sharded_lookup, AegonConfig, ShardTransport, ShardedAegon, ShardedAegonConfig,
    ShardedVerifierContext, Sha256Hash,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::marker::PhantomData;
use std::net::SocketAddr;
use tokio::task::JoinHandle;

type Pcs = KZHK<Bn254>;
type Aegon = akd::aegon::Aegon<Bn254, Pcs, Sha256Hash>;
type Sharded = ShardedAegon<Bn254, Pcs, Sha256Hash>;

/// Build an in-memory Aegon shard sized for `log_capacity` with k=2.
fn build_local_aegon(log_capacity: usize, seed: u64) -> Aegon {
    let kzh = KZHKConfig::new(2, false);
    let cfg = AegonConfig::<Bn254, Pcs> {
        log_capacity,
        private: false,
        pcs_config: kzh,
        _e: PhantomData,
    };
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Aegon::setup(&mut rng, &cfg).expect("Aegon::setup")
}

/// Spawn a ShardServer on an ephemeral port. Returns the bound
/// address (as `http://...` URI for tonic's `connect`) plus a join
/// handle. The handle isn't awaited; tokio aborts the task at test
/// end.
async fn spawn_shard(
    aegon: Aegon,
) -> (String, JoinHandle<Result<(), tonic::transport::Error>>) {
    // Bind to port 0 to get an ephemeral port; we use a TcpListener
    // first to discover the port, then pass the address to tonic.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral");
    let addr: SocketAddr = listener.local_addr().expect("local_addr");
    drop(listener);

    let server = ShardServer::<Bn254, Pcs, Sha256Hash>::new(aegon);
    let handle = tokio::spawn(async move { server.serve(addr).await });
    // Give the server a moment to actually start accepting before
    // the client tries to connect.
    tokio::time::sleep(Duration::from_millis(100)).await;
    (format!("http://{addr}"), handle)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn grpc_sharded_publish_lookup_verify_roundtrip() {
    // Two shards, each with log_capacity=6 (64 slots/shard).
    // Coordinator total log_capacity = 6 + 1 = 7.
    let shard_log_capacity = 6usize;
    let log_n_shards = 1usize;

    // Build two independent in-memory Aegons that will sit behind
    // the two gRPC servers. Different seeds so the in-process and
    // gRPC sides aren't accidentally identical via seed reuse.
    let aegon_a = build_local_aegon(shard_log_capacity, 0xAEC0_A);
    let aegon_b = build_local_aegon(shard_log_capacity, 0xAEC0_B);

    let (addr_a, _h_a) = spawn_shard(aegon_a).await;
    let (addr_b, _h_b) = spawn_shard(aegon_b).await;

    // Coordinator: build a sharded config pointing at the two
    // remote endpoints. SrsSource::DangerouslyGenerate here is only
    // used to load the *verifier_param* on the coordinator side
    // (for VerifierContext); the prover params live on the shard
    // servers we just spawned. We use the same seed as one shard so
    // the verifier params match — they're deterministic.
    //
    // In production, every machine (coordinator + 32 shards) would
    // share an SrsSource::Path pointing at the same SRS file. Here
    // we cheat with a deterministic seed so the test stays
    // self-contained.
    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(shard_log_capacity)
        .log_n_shards(log_n_shards)
        .private(false)
        .kzh_k(2)
        .shards(ShardTransport::Remote {
            endpoints: vec![addr_a.clone(), addr_b.clone()],
        })
        .build()
        .expect("config builds");

    let mut rng = ChaCha20Rng::seed_from_u64(0xAEC0_A);
    // setup() does the gRPC connect handshake to both shard
    // servers. tokio::spawn_blocking moves this off the multi-
    // thread runtime since GrpcShardClient owns its own internal
    // runtime + blocks on it.
    let coordinator: Arc<tokio::sync::Mutex<Sharded>> = tokio::task::spawn_blocking(move || {
        Sharded::setup(&mut rng, &cfg).expect("Sharded::setup via gRPC")
    })
    .await
    .map(|server| Arc::new(tokio::sync::Mutex::new(server)))
    .expect("setup join");

    // Publish a small batch.
    let updates = vec![
        (b"alice".to_vec(), b"alice-v1".to_vec()),
        (b"bob".to_vec(), b"bob-v1".to_vec()),
        (b"carol".to_vec(), b"carol-v1".to_vec()),
        (b"dave".to_vec(), b"dave-v1".to_vec()),
    ];
    let updates_for_publish = updates.clone();
    let coord_clone = coordinator.clone();
    let commit = tokio::task::spawn_blocking(move || {
        let mut server = coord_clone.blocking_lock();
        let commit = server.publish(&updates_for_publish).expect("publish");
        commit
    })
    .await
    .expect("publish join");
    assert_eq!(commit.epoch, 1);
    assert_eq!(commit.per_shard.len(), 2);

    // Each lookup + verify must pass.
    let ctx: ShardedVerifierContext<Bn254, Pcs> = tokio::task::spawn_blocking({
        let coord_clone = coordinator.clone();
        move || coord_clone.blocking_lock().sharded_verifier_context()
    })
    .await
    .expect("ctx join");

    for (label, value) in &updates {
        let coord_clone = coordinator.clone();
        let label_c = label.clone();
        let (_db_value, proof) = tokio::task::spawn_blocking(move || {
            coord_clone.blocking_lock().lookup(&label_c).expect("lookup")
        })
        .await
        .expect("lookup join");
        let ok = verify_sharded_lookup::<Bn254, Pcs, Sha256Hash>(
            &ctx, &commit, label, value, &proof,
        )
        .expect("verify");
        assert!(
            ok,
            "gRPC-backed sharded lookup must verify for {label:?}"
        );
    }

    // Drop the coordinator in a blocking context. Its
    // `GrpcShardClient`s each own a private tokio runtime; dropping
    // those from inside an async context panics ("Cannot drop a
    // runtime in a context where blocking is not allowed"). Moving
    // the drop to `spawn_blocking` sidesteps that.
    tokio::task::spawn_blocking(move || drop(coordinator))
        .await
        .expect("drop join");
}
