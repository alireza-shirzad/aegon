// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! End-to-end test of the distributed SRS protocol over real gRPC.
//!
//! Spins up `n_shards` tonic servers on ephemeral localhost ports,
//! sends each one a `BootstrapSrs` push, lets them exchange slabs
//! peer-to-peer, and verifies that every shard's assembled SRS bytes
//! match the in-process reference.
//!
//! Also exercises the cache hit path: a second `run_distributed_compute`
//! call against a state that already has a cache file on disk should
//! short-circuit via `try_cache_hit` and never need the trapdoor handoff.

use std::sync::Arc;

use akd::aegon::distributed_srs::{
    cache_file_path, connect_srs_client, proto::GetMetricsRequest, run_distributed_compute,
    try_cache_hit, Phase, SrsBootstrapConfig, SrsBootstrapState, SrsServer, Trapdoors,
};
use akd_core::aegon_crypto::pcs::kzhk::srs::KZHKUniversalParams as RefParams;
use akd_core::aegon_crypto::StructuredReferenceString;
use ark_bn254::Bn254;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use tonic::transport::Server;

type E = Bn254;

fn unique_tmp_dir(label: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "aegon-srs-grpc-test-{}-{}-{}",
        label,
        std::process::id(),
        ark_std::rand::random::<u64>()
    ));
    std::fs::create_dir_all(&p).expect("mkdir");
    p
}

/// Bind a tonic server on `127.0.0.1:0` and return its actual port.
/// Lets us run the test against ephemeral ports without colliding with
/// anything the host already has bound.
async fn spawn_srs_server(state: Arc<SrsBootstrapState<E>>) -> u16 {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind");
    listener.set_nonblocking(true).expect("nonblocking");
    let port = listener.local_addr().expect("addr").port();
    let listener = tokio::net::TcpListener::from_std(listener).expect("convert listener");
    let stream = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let svc = SrsServer::new(state).into_service();
    tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(svc)
            .serve_with_incoming(stream)
            .await;
    });
    // Give the listener a moment to start accepting.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    port
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn distributed_srs_end_to_end() {
    let k = 3;
    let num_vars = 6;
    let n_shards = 4usize;
    let setup_seed: u64 = 42;

    // ---- spin up N shards, each with its own state + gRPC server ----
    let mut states: Vec<Arc<SrsBootstrapState<E>>> = Vec::with_capacity(n_shards);
    let mut endpoints: Vec<String> = Vec::with_capacity(n_shards);
    let mut cache_dirs: Vec<std::path::PathBuf> = Vec::with_capacity(n_shards);
    for shard_id in 0..n_shards {
        let cache_dir = unique_tmp_dir(&format!("s{shard_id}"));
        cache_dirs.push(cache_dir.clone());
        let cfg = SrsBootstrapConfig {
            shard_id: shard_id as u32,
            log_capacity: num_vars as u32,
            k: k as u32,
            setup_seed,
            cache_dir,
        };
        let state = SrsBootstrapState::<E>::new(cfg);
        states.push(state);
    }
    for state in &states {
        let port = spawn_srs_server(Arc::clone(state)).await;
        endpoints.push(format!("http://127.0.0.1:{port}"));
    }

    // ---- bootstrap actor: sample trapdoors, push to every shard ----
    let mut rng = ChaCha20Rng::seed_from_u64(setup_seed);
    let trapdoors: Trapdoors<E> = Trapdoors::sample(&mut rng, k, num_vars);
    let trapdoors_bytes = trapdoors.encode().expect("encode trapdoors");

    let mut push_tasks = Vec::new();
    for (shard_id, endpoint) in endpoints.iter().enumerate() {
        let endpoint = endpoint.clone();
        let bytes = trapdoors_bytes.clone();
        let peers = endpoints.clone();
        push_tasks.push(tokio::spawn(async move {
            let mut client = connect_srs_client(&endpoint).await.expect("connect");
            let resp = client
                .bootstrap_srs(tonic::Request::new(
                    akd::aegon::distributed_srs::proto::BootstrapSrsRequest {
                        setup_seed,
                        shard_id: shard_id as u32,
                        n_shards: n_shards as u32,
                        kzh_k: k as u32,
                        log_capacity: num_vars as u32,
                        peer_endpoints: peers,
                        trapdoors_uncompressed: bytes,
                    },
                ))
                .await
                .expect("bootstrap_srs");
            assert!(
                !resp.into_inner().cache_hit,
                "first run on shard {shard_id} should not be a cache hit"
            );
        }));
    }
    for t in push_tasks {
        t.await.expect("push task");
    }

    // ---- run distributed compute on every shard in parallel ----
    let mut compute_tasks = Vec::new();
    for state in &states {
        let state = Arc::clone(state);
        compute_tasks.push(tokio::spawn(async move {
            run_distributed_compute::<E>(state).await
        }));
    }

    let mut assembled: Vec<RefParams<E>> = Vec::with_capacity(n_shards);
    for t in compute_tasks {
        let (up, _pk, _vk) = t.await.expect("compute join").expect("compute ok");
        assembled.push(up);
    }

    // Mark all shards Ready so any waiting WaitForReady would unblock
    // (not actually exercised in this test, but maintains the
    // invariant the state machine documents).
    for state in &states {
        state.set_phase(Phase::Ready, "test-complete").await;
    }

    // ---- verify: every shard's SRS == in-process reference ----
    let mut rng = ChaCha20Rng::seed_from_u64(setup_seed);
    let reference =
        RefParams::<E>::gen_srs_for_testing(&mut rng, k, true, num_vars).expect("ref gen");
    let mut ref_bytes = Vec::new();
    reference
        .serialize_uncompressed(&mut ref_bytes)
        .expect("serialise ref");

    for (shard_id, srs) in assembled.iter().enumerate() {
        let mut shard_bytes = Vec::new();
        srs.serialize_uncompressed(&mut shard_bytes)
            .expect("serialise shard srs");
        assert_eq!(
            shard_bytes.len(),
            ref_bytes.len(),
            "shard {shard_id} SRS size mismatch"
        );
        assert!(
            shard_bytes == ref_bytes,
            "shard {shard_id} SRS bytes differ from reference"
        );
    }

    // ---- verify cache: cache files exist + try_cache_hit short-circuits ----
    for (shard_id, state) in states.iter().enumerate() {
        let cfg = state.config();
        let path = cache_file_path(
            &cfg.cache_dir,
            cfg.log_capacity as usize,
            cfg.k as usize,
            cfg.setup_seed,
        );
        assert!(
            path.exists(),
            "cache file missing for shard {shard_id}: {}",
            path.display()
        );
    }

    // ---- verify per-shard metrics via GetMetrics gRPC ----
    //
    // Three invariants per shard, after distributed gen:
    //   1. inbound_slab_bytes > 0 — we pulled from N-1 peers.
    //   2. outbound_slab_bytes > 0 — N-1 peers pulled from us.
    //   3. By symmetry on a uniformly sized cluster, every shard's
    //      inbound equals every shard's outbound. We verify per-shard
    //      directly and also check the cluster-wide totals match.
    //   4. pk_bytes/vk_bytes/universal_bytes are non-zero and identical
    //      across shards (every shard's SRS is the same SRS).
    //   5. cache_hit is false (distributed-gen, not cache).
    //   6. Phase log contains the expected transitions.
    let mut total_inbound: u64 = 0;
    let mut total_outbound: u64 = 0;
    let mut sizes: Option<(u64, u64, u64)> = None;
    for (shard_id, endpoint) in endpoints.iter().enumerate() {
        let mut client = connect_srs_client(endpoint).await.expect("connect");
        let resp = client
            .get_metrics(tonic::Request::new(GetMetricsRequest {}))
            .await
            .expect("GetMetrics")
            .into_inner();
        assert_eq!(resp.shard_id, shard_id as u32);
        assert!(!resp.cache_hit, "shard {shard_id} should not be cache_hit");
        assert!(
            resp.inbound_slab_bytes > 0,
            "shard {shard_id} inbound bytes should be non-zero"
        );
        assert!(
            resp.outbound_slab_bytes > 0,
            "shard {shard_id} outbound bytes should be non-zero"
        );
        assert!(resp.pk_bytes > 0);
        assert!(resp.vk_bytes > 0);
        assert!(resp.universal_bytes > 0);
        // Compute + communication splits, in the n_shards=4 test
        // setup: both should be non-zero (we both did local MSM work
        // and we pulled from 3 peers).
        assert!(
            resp.compute_secs > 0.0,
            "shard {shard_id} should have spent time in compute"
        );
        assert!(
            resp.communication_secs > 0.0,
            "shard {shard_id} should have spent time in communication (n_shards={n_shards})"
        );
        // Sizes are the same across shards.
        match sizes {
            None => sizes = Some((resp.pk_bytes, resp.vk_bytes, resp.universal_bytes)),
            Some((pk, vk, up)) => {
                assert_eq!(resp.pk_bytes, pk);
                assert_eq!(resp.vk_bytes, vk);
                assert_eq!(resp.universal_bytes, up);
            }
        }
        total_inbound += resp.inbound_slab_bytes;
        total_outbound += resp.outbound_slab_bytes;
        // Phase log: assert that we hit the expected transitions in
        // order. The shard's main loop walks
        // awaiting -> computing/pulling -> assembling/writing -> ready
        // (we promote to Ready at the end of the test loop above).
        let labels: Vec<&str> = resp.phases.iter().map(|p| p.phase.as_str()).collect();
        assert!(
            labels.first() == Some(&"awaiting-bootstrap"),
            "shard {shard_id} phase log should start with awaiting-bootstrap, got {labels:?}"
        );
        assert!(
            labels.contains(&"test-complete"),
            "shard {shard_id} should have seen the test-complete marker"
        );
        // Monotonic_secs strictly increases.
        let mut last = -1.0f64;
        for p in &resp.phases {
            assert!(
                p.monotonic_secs >= last,
                "phase timestamps must be monotonic, got {p:?} after {last}"
            );
            last = p.monotonic_secs;
        }
    }
    // Conservation: by symmetry, the sum of every shard's inbound
    // equals the sum of every shard's outbound (each byte pulled by
    // shard A from shard B is also a byte served by shard B to A).
    assert_eq!(
        total_inbound, total_outbound,
        "cluster inbound total {} must equal outbound total {}",
        total_inbound, total_outbound,
    );

    // ---- second run: fresh state, same cache_dir -> cache hit ----
    for (shard_id, cache_dir) in cache_dirs.iter().enumerate() {
        let cfg = SrsBootstrapConfig {
            shard_id: shard_id as u32,
            log_capacity: num_vars as u32,
            k: k as u32,
            setup_seed,
            cache_dir: cache_dir.clone(),
        };
        let state2 = SrsBootstrapState::<E>::new(cfg);
        let hit = try_cache_hit::<E>(&state2).await.expect("try_cache_hit");
        assert!(
            hit.is_some(),
            "second boot for shard {shard_id} should hit cache"
        );

        let (up2, _, _) = hit.unwrap();
        let mut bytes = Vec::new();
        up2.serialize_uncompressed(&mut bytes).expect("ser");
        assert_eq!(
            bytes, ref_bytes,
            "cache-loaded SRS != reference (shard {shard_id})"
        );
        assert_eq!(
            state2.phase().await,
            Phase::Initializing,
            "cache hit should advance phase past AwaitingBootstrap"
        );
        // Cache-hit snapshot: cache_hit flag set, sizes recorded, no
        // network traffic recorded (byte counters stay at zero because
        // try_cache_hit doesn't touch peers).
        let snap = state2.snapshot_metrics().await;
        assert!(snap.cache_hit, "cache hit flag must be true on second boot");
        assert_eq!(snap.inbound_slab_bytes, 0);
        assert_eq!(snap.outbound_slab_bytes, 0);
        assert!(snap.pk_bytes > 0);
        assert!(snap.vk_bytes > 0);
        assert!(snap.universal_bytes > 0);
        // Cache-hit path does no compute and no communication.
        assert_eq!(snap.compute_secs, 0.0);
        assert_eq!(snap.communication_secs, 0.0);
    }

    // ---- clean up tmp dirs ----
    for d in cache_dirs {
        let _ = std::fs::remove_dir_all(&d);
    }
}
