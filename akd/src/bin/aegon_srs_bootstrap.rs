// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

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

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use akd::aegon::distributed_srs::{
    connect_srs_client,
    proto::{BootstrapSrsRequest, GetMetricsRequest, GetMetricsResponse, WaitForReadyRequest},
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

    /// Optional path. When set, after every shard reports Ready the
    /// bootstrap actor calls `GetMetrics` on each shard, aggregates
    /// the per-shard records, and writes one JSON file to this path
    /// (schema mirrors `aegon_setup_bench` for the size fields, plus
    /// per-shard inbound/outbound bytes + phase timings). Used by
    /// `bench-cluster.sh setup-bench` to drive the planetary
    /// regime's setup-time + comm-bytes measurement.
    #[arg(long)]
    metrics_out: Option<PathBuf>,
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
        }
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
            }
            Ok(Err(e)) => {
                eprintln!("[bootstrap] push failed: {e}");
                return ExitCode::from(1);
            }
            Err(e) => {
                eprintln!("[bootstrap] push task join error: {e}");
                return ExitCode::from(1);
            }
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
                    }
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
                }
                Ok(Err((shard_id, endpoint, e))) => {
                    eprintln!("[bootstrap] shard {shard_id} ({endpoint}) poll error: {e}");
                    // Don't bail immediately on transient errors — log and
                    // re-poll. The cluster might still be coming up.
                    status_summary.push(format!("s{shard_id}:poll-err"));
                }
                Err(e) => {
                    eprintln!("[bootstrap] poll task join: {e}");
                    return ExitCode::from(1);
                }
            }
        }

        if ready_count == n_shards {
            eprintln!("[bootstrap] all {n_shards} shard(s) ready");
            // Aggregate metrics if asked. Done after Ready so the
            // per-phase timings are complete and the cache-write
            // sizes are recorded.
            if let Some(path) = args.metrics_out.as_ref() {
                if let Err(e) = gather_and_write_metrics(
                    &args.shard_endpoints,
                    args.shard_log_capacity as u32,
                    args.kzh_k as u32,
                    args.setup_seed,
                    path,
                )
                .await
                {
                    eprintln!("[bootstrap] metrics gather failed: {e}");
                    return ExitCode::from(1);
                }
            }
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

/// Call `GetMetrics` on every shard and write one consolidated JSON
/// record to `out_path`. Per-shard records are concatenated into a
/// `shards` array; cluster-wide aggregates (total inbound/outbound,
/// max/min per-shard ready time) sit at the top level alongside the
/// invariant params (log_cap, kzh_k, setup_seed). Schema mirrors
/// `aegon_setup_bench`'s output for size fields so downstream
/// analysis can plot the two on the same axes.
async fn gather_and_write_metrics(
    endpoints: &[String],
    shard_log_capacity: u32,
    kzh_k: u32,
    setup_seed: u64,
    out_path: &std::path::Path,
) -> Result<(), String> {
    eprintln!("[bootstrap] gathering per-shard metrics");
    let n_shards = endpoints.len();
    let mut tasks = Vec::with_capacity(n_shards);
    for (i, ep) in endpoints.iter().enumerate() {
        let ep = ep.clone();
        tasks.push(tokio::spawn(async move {
            let mut client = connect_srs_client(&ep)
                .await
                .map_err(|e| format!("shard {i} connect '{ep}': {e}"))?;
            let resp = client
                .get_metrics(tonic::Request::new(GetMetricsRequest {}))
                .await
                .map_err(|s| format!("shard {i} GetMetrics: {s}"))?;
            Ok::<_, String>((i, resp.into_inner()))
        }));
    }
    let mut per_shard: Vec<(usize, GetMetricsResponse)> = Vec::with_capacity(n_shards);
    for t in tasks {
        let (i, resp) = t.await.map_err(|e| format!("metrics join: {e}"))??;
        per_shard.push((i, resp));
    }
    per_shard.sort_by_key(|(i, _)| *i);

    // Aggregates over the cluster.
    let mut total_inbound: u64 = 0;
    let mut total_outbound: u64 = 0;
    let mut max_ready_secs: f64 = 0.0;
    let mut min_ready_secs: f64 = f64::INFINITY;
    // Shards do their setup work in parallel, so the cluster's
    // effective compute / communication time is the max across
    // shards (not the sum). We report both max and min so an outlier
    // straggler is visible.
    let mut max_compute_secs: f64 = 0.0;
    let mut max_communication_secs: f64 = 0.0;
    let mut min_compute_secs: f64 = f64::INFINITY;
    let mut min_communication_secs: f64 = f64::INFINITY;
    // Picked from the first shard — all shards see the same SRS, so
    // pk/vk/universal sizes are identical. Asserting that across all
    // shards is left to downstream consistency checks.
    let (pk_bytes, vk_bytes, universal_bytes) = per_shard
        .first()
        .map(|(_, r)| (r.pk_bytes, r.vk_bytes, r.universal_bytes))
        .unwrap_or((0, 0, 0));
    for (_, r) in &per_shard {
        total_inbound += r.inbound_slab_bytes;
        total_outbound += r.outbound_slab_bytes;
        max_compute_secs = max_compute_secs.max(r.compute_secs);
        max_communication_secs = max_communication_secs.max(r.communication_secs);
        min_compute_secs = min_compute_secs.min(r.compute_secs);
        min_communication_secs = min_communication_secs.min(r.communication_secs);
        if let Some(ready_entry) = r.phases.iter().find(|p| p.phase == "ready") {
            max_ready_secs = max_ready_secs.max(ready_entry.monotonic_secs);
            min_ready_secs = min_ready_secs.min(ready_entry.monotonic_secs);
        }
    }
    if !min_ready_secs.is_finite() {
        min_ready_secs = 0.0;
    }
    if !min_compute_secs.is_finite() {
        min_compute_secs = 0.0;
    }
    if !min_communication_secs.is_finite() {
        min_communication_secs = 0.0;
    }

    let mut shards_json: Vec<String> = Vec::with_capacity(n_shards);
    for (_, r) in &per_shard {
        let phases_csv = r
            .phases
            .iter()
            .map(|p| {
                format!(
                    "      {{\"phase\": \"{}\", \"monotonic_secs\": {:.6}}}",
                    p.phase, p.monotonic_secs
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        shards_json.push(format!(
            concat!(
                "    {{\n",
                "      \"shard_id\": {sid},\n",
                "      \"cache_hit\": {ch},\n",
                "      \"compute_secs\": {cs:.6},\n",
                "      \"communication_secs\": {ms:.6},\n",
                "      \"inbound_slab_bytes\": {in_b},\n",
                "      \"outbound_slab_bytes\": {out_b},\n",
                "      \"pk_bytes\": {pkb},\n",
                "      \"vk_bytes\": {vkb},\n",
                "      \"universal_bytes\": {upb},\n",
                "      \"phases\": [\n{phases}\n      ]\n",
                "    }}"
            ),
            sid = r.shard_id,
            ch = r.cache_hit,
            cs = r.compute_secs,
            ms = r.communication_secs,
            in_b = r.inbound_slab_bytes,
            out_b = r.outbound_slab_bytes,
            pkb = r.pk_bytes,
            vkb = r.vk_bytes,
            upb = r.universal_bytes,
            phases = phases_csv,
        ));
    }

    let json = format!(
        concat!(
            "{{\n",
            "  \"regime\": \"distributed\",\n",
            "  \"n_shards\": {n_shards},\n",
            "  \"shard_log_capacity\": {slc},\n",
            "  \"kzh_k\": {k},\n",
            "  \"setup_seed\": {seed},\n",
            "  \"prover_param_bytes\": {pkb},\n",
            "  \"verifier_param_bytes\": {vkb},\n",
            "  \"universal_params_bytes\": {upb},\n",
            "  \"total_inbound_slab_bytes\": {tot_in},\n",
            "  \"total_outbound_slab_bytes\": {tot_out},\n",
            "  \"max_shard_ready_secs\": {max_ready:.6},\n",
            "  \"min_shard_ready_secs\": {min_ready:.6},\n",
            "  \"max_shard_compute_secs\": {max_cs:.6},\n",
            "  \"min_shard_compute_secs\": {min_cs:.6},\n",
            "  \"max_shard_communication_secs\": {max_ms:.6},\n",
            "  \"min_shard_communication_secs\": {min_ms:.6},\n",
            "  \"shards\": [\n{shards}\n  ]\n",
            "}}\n"
        ),
        n_shards = n_shards,
        slc = shard_log_capacity,
        k = kzh_k,
        seed = setup_seed,
        pkb = pk_bytes,
        vkb = vk_bytes,
        upb = universal_bytes,
        tot_in = total_inbound,
        tot_out = total_outbound,
        max_ready = max_ready_secs,
        min_ready = min_ready_secs,
        max_cs = max_compute_secs,
        min_cs = min_compute_secs,
        max_ms = max_communication_secs,
        min_ms = min_communication_secs,
        shards = shards_json.join(",\n"),
    );

    std::fs::write(out_path, &json).map_err(|e| format!("write '{}': {e}", out_path.display()))?;
    eprintln!(
        "[bootstrap] metrics: total_inbound={:.2} GiB total_outbound={:.2} GiB max_ready={:.2}s \
         max_compute={:.2}s max_communication={:.2}s",
        total_inbound as f64 / (1024.0 * 1024.0 * 1024.0),
        total_outbound as f64 / (1024.0 * 1024.0 * 1024.0),
        max_ready_secs,
        max_compute_secs,
        max_communication_secs,
    );
    eprintln!("[bootstrap] wrote {}", out_path.display());
    Ok(())
}
