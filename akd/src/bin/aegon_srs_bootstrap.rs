//! `aegon_srs_bootstrap` — one-shot bootstrap actor for the distributed
//! SRS generation flow.
//!
//! Samples trapdoors deterministically from `--setup-seed`, pushes them
//! to every shard's `SrsService.BootstrapSrs` endpoint, then polls
//! `WaitForReady` on every shard until they all report `ready = true`
//! (SRS assembled + Aegon init + prefill complete). Exits 0 once the
//! cluster is fully ready; 1 on any shard error.
//!
//! Typical use:
//!
//!   aegon_srs_bootstrap \
//!     --shard-log-capacity 29 \
//!     --kzh-k 10 \
//!     --setup-seed 42 \
//!     --shard-endpoints http://10.0.0.5:50052,http://10.0.0.6:50052,...
//!
//! Idempotent against caches: if every shard already has a matching
//! `.cache` file on disk, the BootstrapSrs RPC returns `cache_hit =
//! true` and the cluster reports ready almost immediately.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use akd::aegon::distributed_srs::{
    connect_srs_client,
    proto::{BootstrapSrsRequest, WaitForReadyRequest},
    Trapdoors,
};
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;

type E = Bn254;

#[derive(Debug, Parser)]
#[command(
    name = "aegon_srs_bootstrap",
    about = "Sample trapdoors and push them to every shard's SrsService, then wait for ready. Run ONCE per cluster lifetime (or per --setup-seed)."
)]
struct Args {
    /// log_2 of slots per shard. Must match every shard's
    /// `--shard-log-capacity`.
    #[arg(long)]
    shard_log_capacity: usize,

    /// KZH-k block parameter. Must match every shard's `--kzh-k`.
    #[arg(long)]
    kzh_k: usize,

    /// Deterministic seed for trapdoor sampling. Must match every
    /// shard's `--setup-seed`. The seed is hashed into the cache
    /// filename on each shard, so a different seed produces a different
    /// cache file (no silent reuse).
    #[arg(long)]
    setup_seed: u64,

    /// Comma-separated list of shard `SrsService` endpoints (e.g.
    /// `http://10.0.0.5:50052,http://10.0.0.6:50052,...`). The order
    /// IS the shard_id ordering — endpoint at index `i` is the gRPC URL
    /// for shard `i`. The full list is also pushed to every shard so
    /// they know where their peers' GetSrsSlab endpoints live.
    #[arg(long, value_delimiter = ',')]
    shard_endpoints: Vec<String>,

    /// Maximum wall-clock to wait for all shards to report ready.
    /// Default 30 minutes — leaves headroom for the SRS-gen + slab
    /// exchange + prefill at log_cap=29 on a 32-shard cluster.
    #[arg(long, default_value_t = 1800)]
    timeout_secs: u64,

    /// How often to log progress (e.g. "shards ready: 17/32") while
    /// polling `WaitForReady`. Default 10s.
    #[arg(long, default_value_t = 10)]
    poll_log_interval_secs: u64,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> ExitCode {
    let args = Args::parse();
    let n_shards = args.shard_endpoints.len();
    if n_shards == 0 {
        eprintln!("error: --shard-endpoints must be non-empty");
        return ExitCode::from(2);
    }

    eprintln!(
        "[bootstrap] n_shards={} shard_log_capacity={} kzh_k={} seed={:#018x}",
        n_shards, args.shard_log_capacity, args.kzh_k, args.setup_seed,
    );

    // ---- sample trapdoors -------------------------------------------
    // Sampling matches `KZHKUniversalParams::gen_srs_for_testing`
    // exactly so seed=42 here produces byte-identical trapdoors to
    // seed=42 in any other path that uses ChaCha20Rng.
    eprintln!("[bootstrap] sampling trapdoors from seed");
    let mut rng = ChaCha20Rng::seed_from_u64(args.setup_seed);
    let trapdoors: Trapdoors<E> = Trapdoors::sample(&mut rng, args.kzh_k, args.shard_log_capacity);
    let trapdoors_bytes = match trapdoors.encode() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[bootstrap] error encoding trapdoors: {e}");
            return ExitCode::from(1);
        },
    };
    eprintln!(
        "[bootstrap] trapdoors ready ({} bytes uncompressed)",
        trapdoors_bytes.len()
    );

    // ---- push to all shards in parallel ------------------------------
    eprintln!("[bootstrap] pushing BootstrapSrs to {n_shards} shard(s) in parallel");
    let mut push_tasks = Vec::with_capacity(n_shards);
    for (shard_id, endpoint) in args.shard_endpoints.iter().enumerate() {
        let endpoint = endpoint.clone();
        let bytes = trapdoors_bytes.clone();
        let peers = args.shard_endpoints.clone();
        let kzh_k = args.kzh_k as u32;
        let log_capacity = args.shard_log_capacity as u32;
        let seed = args.setup_seed;
        push_tasks.push(tokio::spawn(async move {
            let mut client = match connect_srs_client(&endpoint).await {
                Ok(c) => c,
                Err(e) => return Err(format!("connect '{endpoint}': {e}")),
            };
            let resp = client
                .bootstrap_srs(tonic::Request::new(BootstrapSrsRequest {
                    setup_seed: seed,
                    shard_id: shard_id as u32,
                    n_shards: n_shards as u32,
                    kzh_k,
                    log_capacity,
                    peer_endpoints: peers,
                    trapdoors_uncompressed: bytes,
                }))
                .await
                .map_err(|s| format!("BootstrapSrs '{endpoint}': {s}"))?;
            Ok::<_, String>((shard_id, endpoint, resp.into_inner().cache_hit))
        }));
    }

    let mut cache_hits = 0usize;
    for t in push_tasks {
        match t.await {
            Ok(Ok((shard_id, endpoint, cache_hit))) => {
                if cache_hit {
                    cache_hits += 1;
                    eprintln!("[bootstrap] shard {shard_id} ({endpoint}): cache hit");
                } else {
                    eprintln!("[bootstrap] shard {shard_id} ({endpoint}): bootstrapping");
                }
            },
            Ok(Err(e)) => {
                eprintln!("[bootstrap] push failed: {e}");
                return ExitCode::from(1);
            },
            Err(e) => {
                eprintln!("[bootstrap] push task join error: {e}");
                return ExitCode::from(1);
            },
        }
    }
    eprintln!(
        "[bootstrap] all {n_shards} shard(s) acked ({cache_hits} cache hit, {} bootstrapping)",
        n_shards - cache_hits
    );

    // ---- wait until every shard reports ready ------------------------
    eprintln!(
        "[bootstrap] polling WaitForReady (timeout {}s)",
        args.timeout_secs
    );
    let deadline = Instant::now() + Duration::from_secs(args.timeout_secs);

    // Per-poll timeout we pass to each shard so it can return status
    // strings while we wait. Cap below the global deadline so we
    // re-poll periodically and log progress.
    let per_poll_timeout_secs = args.poll_log_interval_secs.max(1);

    loop {
        let mut wait_tasks = Vec::with_capacity(n_shards);
        for (shard_id, endpoint) in args.shard_endpoints.iter().enumerate() {
            let endpoint = endpoint.clone();
            wait_tasks.push(tokio::spawn(async move {
                let mut client = match connect_srs_client(&endpoint).await {
                    Ok(c) => c,
                    Err(e) => return Err((shard_id, endpoint, format!("connect: {e}"))),
                };
                match client
                    .wait_for_ready(tonic::Request::new(WaitForReadyRequest {
                        timeout_secs: per_poll_timeout_secs,
                    }))
                    .await
                {
                    Ok(r) => {
                        let inner = r.into_inner();
                        Ok((shard_id, endpoint, inner.ready, inner.status))
                    },
                    Err(s) => Err((shard_id, endpoint, format!("WaitForReady: {s}"))),
                }
            }));
        }

        let mut ready_count = 0usize;
        let mut status_summary: Vec<String> = Vec::new();
        for t in wait_tasks {
            match t.await {
                Ok(Ok((shard_id, _endpoint, ready, status))) => {
                    if ready {
                        ready_count += 1;
                    } else {
                        status_summary.push(format!("s{shard_id}:{status}"));
                    }
                },
                Ok(Err((shard_id, endpoint, e))) => {
                    eprintln!("[bootstrap] shard {shard_id} ({endpoint}) poll error: {e}");
                    // Don't bail immediately on transient errors — log and
                    // re-poll. The cluster might still be coming up.
                    status_summary.push(format!("s{shard_id}:poll-err"));
                },
                Err(e) => {
                    eprintln!("[bootstrap] poll task join: {e}");
                    return ExitCode::from(1);
                },
            }
        }

        if ready_count == n_shards {
            eprintln!("[bootstrap] all {n_shards} shard(s) ready");
            return ExitCode::SUCCESS;
        }

        // Trim status_summary for log brevity (show first few non-ready shards).
        let preview: Vec<&str> = status_summary.iter().take(5).map(|s| s.as_str()).collect();
        let truncated = if status_summary.len() > 5 {
            format!(" (+{} more)", status_summary.len() - 5)
        } else {
            String::new()
        };
        eprintln!(
            "[bootstrap] ready: {ready_count}/{n_shards} — pending: [{}]{truncated}",
            preview.join(", "),
        );

        if Instant::now() >= deadline {
            eprintln!(
                "[bootstrap] TIMEOUT after {}s with {ready_count}/{n_shards} shards ready",
                args.timeout_secs
            );
            return ExitCode::from(1);
        }
    }
}
