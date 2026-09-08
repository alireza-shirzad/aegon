// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Localhost gRPC integration test for the full ECVRF wiring.
//!
//! Spins up a `CoordinatorServer` on an ephemeral port with
//! `EcVrfHash` + `VrfProver` attached, connects with a
//! `CoordinatorClient`, and exercises publish + lookup + verify
//! across the wire. The bootstrap handshake (`CurrentCommitment`)
//! must transport a non-empty `vrf_pubkey`; the client must
//! construct a `VrfVerifier` from it, and every lookup proof must
//! verify under that verifier.
//!
//! This is the end-to-end test the punch list called for. It
//! exercises every piece introduced in this commit: the proto field,
//! the server-side handler, the client-side handshake, and the
//! cross-process VRF cryptography.

use std::sync::Arc;
use std::time::Duration;

use akd::aegon::coordinator_grpc::{CoordinatorClient, CoordinatorServer};
use akd::aegon::{
    DbSource, EcVrfHash, ShardTransport, ShardedAegon, ShardedAegonConfig, ShardedVerifierContext,
    VrfProver, BENCH_VRF_SEED,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use tokio::sync::RwLock as AsyncRwLock;

type Pcs = KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, EcVrfHash>;

fn build_cfg(shard_log_capacity: usize, log_n_shards: usize) -> ShardedAegonConfig<Bn254, Pcs> {
    ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(shard_log_capacity)
        .log_n_shards(log_n_shards)
        .private(false)
        .kzh_k(2)
        .shards(ShardTransport::InProcess)
        .db(DbSource::None)
        .build()
        .expect("config builds")
}

fn must(label: &str, ok: bool) {
    if !ok {
        eprintln!("FAIL: {label}");
        std::process::exit(1);
    }
    println!("  ok  {label}");
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    println!("=== Aegon ECVRF localhost gRPC end-to-end ===\n");

    // ---- 1. Build the coordinator state. ----
    let shard_log_capacity = 8usize;
    let log_n_shards = 1usize;
    let cfg = build_cfg(shard_log_capacity, log_n_shards);

    let mut rng = ChaCha20Rng::seed_from_u64(0xAEC0_E2E);
    let mut state = tokio::task::spawn_blocking(move || {
        Sharded::setup(&mut rng, &cfg).expect("Sharded::setup with EcVrfHash")
    })
    .await
    .expect("setup join");

    state.set_vrf_prover(VrfProver::from_seed(&BENCH_VRF_SEED));
    let verifier_ctx_template: ShardedVerifierContext<Bn254, Pcs> =
        state.sharded_verifier_context();
    must(
        "server-side ctx carries vrf_verifier",
        verifier_ctx_template.vrf_verifier.is_some(),
    );

    // Publish a small batch.
    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..6)
        .map(|i| {
            (
                format!("alice-{i}").into_bytes(),
                format!("pk-{i}").into_bytes(),
            )
        })
        .collect();
    let updates_for_publish = updates.clone();
    let commit_epoch = {
        let commit = state
            .publish_two_layer(&updates_for_publish)
            .expect("publish_two_layer");
        commit.epoch
    };
    println!(
        "  published {} labels @ epoch {commit_epoch}.",
        updates.len()
    );

    // ---- 2. Wrap in CoordinatorServer and bind on an ephemeral port. ----
    let shared = Arc::new(AsyncRwLock::new(state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 127.0.0.1:0");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);
    let endpoint = format!("http://{addr}");

    let server = CoordinatorServer::<Bn254, Pcs, EcVrfHash>::from_shared(Arc::clone(&shared));
    let _server_task = tokio::spawn(async move {
        let _ = server.serve(addr).await;
    });
    tokio::time::sleep(Duration::from_millis(150)).await;
    println!("  CoordinatorServer listening at {endpoint}");

    // ---- 3. Client connects. The handshake fetches CurrentCommitment
    //         and pins the deployment's VRF pubkey on the verifier_ctx.
    // The caller-supplied verifier_ctx_template has the prover-derived
    // pubkey pre-attached too, but a real client wouldn't — it would
    // pass a ctx with `vrf_verifier: None` and rely on the bootstrap.
    // To exercise the bootstrap, we deliberately strip the pubkey from
    // the template here.
    let bare_ctx = {
        let mut c = verifier_ctx_template.clone();
        c.vrf_verifier = None;
        c
    };
    let endpoint_for_client = endpoint.clone();
    let client = tokio::task::spawn_blocking(move || {
        CoordinatorClient::<Bn254, Pcs, EcVrfHash>::connect(endpoint_for_client, bare_ctx)
            .expect("CoordinatorClient::connect")
    })
    .await
    .expect("connect join");
    println!("  CoordinatorClient connected; bootstrap handshake done.");

    // The client-side verifier_ctx now carries the verifier. We can't
    // observe it directly through public API, but the proof of life
    // is that the lookup verifies (it would fail if the verifier
    // weren't attached, because every probe carries vrf_proof bytes
    // and verify_lookup_label requires the verifier in that case).

    // ---- 4. Fetch current commitment over the wire. ----
    let client_arc = Arc::new(client);
    let client_for_commit = Arc::clone(&client_arc);
    let commit = tokio::task::spawn_blocking(move || {
        client_for_commit
            .current_commitment()
            .expect("current_commitment")
    })
    .await
    .expect("commit join");
    println!(
        "  fetched current commitment over gRPC (epoch {}).",
        commit.epoch
    );

    // ---- 5. Lookup each label + verify across the wire. ----
    let mut ok_count = 0;
    for (label, value) in &updates {
        let label_c = label.clone();
        let value_c = value.clone();
        let commit_c = commit.clone();
        let client_c = Arc::clone(&client_arc);

        let result = tokio::task::spawn_blocking(move || {
            let slot = client_c
                .lookup_label(&commit_c, &label_c)
                .expect("lookup_label across gRPC");
            let proof = client_c
                .lookup_value_with_bytes(&commit_c, &slot, &value_c)
                .expect("lookup_value_with_bytes");
            (slot, proof)
        })
        .await
        .expect("lookup join");
        let _ = result;
        ok_count += 1;
    }
    must(
        &format!("{ok_count}/{} lookups verified over gRPC", updates.len()),
        ok_count == updates.len(),
    );

    // ---- 6. Negative: wrong-value verify fails. ----
    let probe_label = updates[0].0.clone();
    let bad_value = b"this-is-not-pk-0".to_vec();
    let commit_c = commit.clone();
    let client_c = Arc::clone(&client_arc);
    let err = tokio::task::spawn_blocking(move || {
        let slot = client_c
            .lookup_label(&commit_c, &probe_label)
            .expect("lookup_label");
        client_c.lookup_value_with_bytes(&commit_c, &slot, &bad_value)
    })
    .await
    .expect("join");
    must("wrong-value verify rejected", err.is_err());

    println!("\nAll checks passed. ECVRF is fully wired through the gRPC layer.");

    // Drop the client in a blocking context. `CoordinatorClient`
    // owns its own internal tokio runtime; dropping that runtime
    // from inside another tokio runtime panics ("Cannot drop a
    // runtime in a context where blocking is not allowed"). Mirrors
    // the same teardown shim in tests/grpc_sharded.rs.
    tokio::task::spawn_blocking(move || drop(client_arc))
        .await
        .expect("drop join");
}
