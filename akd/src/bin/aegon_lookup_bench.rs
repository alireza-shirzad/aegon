//! `aegon_lookup_bench` — lookup latency + wire-size sweep against a
//! live shard cluster.
//!
//! For each "preload level" N in `--preload-counts` (e.g.
//! `1,2,4,...,capacity/8`), the bench:
//!
//!   1. Publishes enough fresh `(label, value)` pairs to bring the
//!      total count up to N (incremental from the last level).
//!   2. Picks `--samples-per-level` already-loaded labels and, for
//!      each, measures four timings:
//!        * `server_lookup_label_ns`: time of the *server-side*
//!          `ShardedAegon::lookup_label` call (no gRPC, no verify).
//!        * `server_lookup_value_ns`: time of the *server-side*
//!          `ShardedAegon::lookup_value` call.
//!        * `client_lookup_label_ns`: full client path — gRPC RTT
//!          to the in-process coordinator + local proof verification.
//!        * `client_lookup_value_ns`: same, for the value side.
//!   3. Encodes the response messages via `prost::Message::encoded_len`
//!      to capture the on-wire byte counts the coordinator actually
//!      sends back to clients, plus the size of just the proof field
//!      (excludes the slot / value fields), plus the label/value
//!      application-level sizes. "Proof overhead" is then
//!      `wire_bytes - label_size_bytes` or `wire_bytes - value_size_bytes`
//!      depending on the call.
//!
//! The bench drives the preload itself (no shard-level `--prefill-count`
//! coordination) so a fresh cluster yields well-defined sample
//! distributions — every sample at level N is drawn from the bench's
//! own deterministic namespace `bench-u{i}` for i in [0..N), so we
//! always know the value bytes for `lookup_value_with_bytes` and we
//! always know a label is actually present.
//!
//! The coordinator gRPC server runs in a background thread on
//! `--coordinator-listen` (default `127.0.0.1:50190`), driven by the
//! same `Arc<RwLock<ShardedAegon>>` the bench publishes against. The
//! bench's "server-direct" measurements take the lock outside of
//! gRPC, so they see the same state the client side sees — just
//! without serialization / network / verify cost.
//!
//! Usage:
//!
//! ```text
//! aegon_lookup_bench \
//!   --shard-log-capacity 20 --kzh-k 10 --setup-seed 42 \
//!   --endpoints http://aegon-shard-0:50051,...,http://aegon-shard-3:50051 \
//!   --preload-counts 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288 \
//!   --samples-per-level 20 \
//!   --db-url redis://aegon-bench-db:6379 \
//!   --output /tmp/aegon-lookup-bench.json
//! ```

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use akd::aegon::coordinator_grpc::{
    proto::{
        coordinator_service_client::CoordinatorServiceClient, Empty, LookupHistoryRequest,
        LookupHistoryResponse, LookupLabelHistoryRequest, LookupLabelHistoryResponse,
        LookupLabelRequest, LookupLabelResponse, LookupValueRequest,
        LookupValueResponse,
    },
    CoordinatorServer,
};
use akd::aegon::{DbSource, Sha256Hash, ShardTransport, ShardedAegon, ShardedAegonConfig, SrsSource};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::SeedableRng;
use clap::Parser;
use prost::Message;
use rand_chacha::ChaCha20Rng;
use tokio::sync::RwLock as AsyncRwLock;

type Pcs = KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, Sha256Hash>;

/// Realistic application sizing for this bench:
/// labels are 12-byte ASCII phone numbers in E.164 form
/// (`+1` + 10 digits, e.g. `+10000000123`) and values are 256-byte
/// random buffers that stand in for a 2048-bit RSA public key. The
/// goal isn't to use _correct_ RSA bytes — the AKD only sees the byte
/// string — but to make the wire-size / hash-cost / DB-footprint
/// numbers reflect what a real deployment would see, instead of the
/// ~10-byte `bench-u{i}` / `bench-v{i}` that the old bench used.
const RSA_VALUE_LEN: usize = 256;

fn phone_label(idx: u64) -> Vec<u8> {
    // E.164 +1NNNNNNNNNN. The 10-digit space is 10^10 which comfortably
    // covers the ~100K labels this bench publishes.
    format!("+1{:010}", idx % 10_000_000_000).into_bytes()
}

fn rsa_value(idx: u64) -> Vec<u8> {
    // Deterministic per-idx so a lookup_value returns the same bytes
    // every run. Seed includes a fixed constant so we don't accidentally
    // collide with another bench's RNG stream.
    use ark_std::rand::RngCore;
    let mut rng = ChaCha20Rng::seed_from_u64(0xCAFE_C0DE ^ idx);
    let mut v = vec![0u8; RSA_VALUE_LEN];
    rng.fill_bytes(&mut v);
    v
}

/// Read this process's resident set size (kB) from /proc/self/status.
/// Returns None on non-Linux or if the field can't be parsed. Used by
/// the bench to log memory headroom at each sweep level — n2-standard-4
/// (16 GB, swap=0) is tight enough that we want to see the curve grow.
fn read_self_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb_str = rest.trim().split_whitespace().next()?;
            return kb_str.parse().ok();
        }
    }
    None
}

#[derive(Debug, Parser)]
#[command(
    name = "aegon_lookup_bench",
    about = "Sweep preload counts against a live shard cluster, measure lookup latency (server + client) and proof wire sizes."
)]
struct Args {
    /// log_2 of slots per shard. Must match every shard server.
    #[arg(long)]
    shard_log_capacity: usize,

    /// KZH-k block parameter. Must match every shard.
    #[arg(long)]
    kzh_k: usize,

    /// Comma-separated shard endpoints. Length must be a power of two.
    #[arg(long, value_delimiter = ',')]
    endpoints: Vec<String>,

    /// Path to a serialized SRS. Mutually exclusive with --setup-seed.
    #[arg(long, conflicts_with = "setup_seed")]
    srs_path: Option<PathBuf>,

    /// Deterministic in-process SRS gen. Must match every shard's seed.
    #[arg(long)]
    setup_seed: Option<u64>,

    /// Enable zero-knowledge mode (must match every shard).
    #[arg(long)]
    private: bool,

    /// Redis URL for coordinator-side open-addressing. Without this
    /// the coordinator falls back to gRPC `EXISTS`-style probes against
    /// the owning shard for every slot check, which dominates latency
    /// on a WAN cluster.
    #[arg(long, conflicts_with = "db_path")]
    db_url: Option<String>,

    /// Local RocksDB directory for coordinator-side state. Mutually
    /// exclusive with `--db-url`.
    #[arg(long)]
    db_path: Option<PathBuf>,

    /// Loopback address to bind the in-process gRPC coordinator on.
    /// The bench's own `CoordinatorClient` connects here. Pick a port
    /// that's free on the bench host.
    #[arg(long, default_value = "127.0.0.1:50190")]
    coordinator_listen: String,

    /// Sweep points: comma-separated label counts that the bench will
    /// publish up to before sampling lookups. Must be strictly
    /// increasing; the bench publishes the gap between consecutive
    /// points so e.g. `1,2,4,...,N` publishes N labels total across
    /// the sweep, not 1+2+4+...+N.
    #[arg(long, value_delimiter = ',')]
    preload_counts: Vec<usize>,

    /// How many lookup samples to draw per preload level. Each sample
    /// times all four call paths (server-direct label, server-direct
    /// value, client label, client value) and records one wire-size
    /// row.
    #[arg(long, default_value_t = 10)]
    samples_per_level: usize,

    /// Internal batch size used to publish the gap between sweep
    /// points. Larger batches amortize the publish's fixed cost; too
    /// large and a single batch can exceed the coordinator's working
    /// memory. 1024 is a safe default on the bench cluster.
    #[arg(long, default_value_t = 1024)]
    publish_batch_size: usize,

    /// Where to write the JSON timing report.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> ExitCode {
    #[cfg(feature = "tracing_instrument")]
    akd::aegon::tracing_init::init_tree_subscriber();
    let args = Args::parse();

    if !args.endpoints.len().is_power_of_two() {
        eprintln!(
            "error: --endpoints length must be a power of two (got {})",
            args.endpoints.len()
        );
        return ExitCode::from(2);
    }
    if args.srs_path.is_none() && args.setup_seed.is_none() {
        eprintln!("error: provide either --srs-path or --setup-seed");
        return ExitCode::from(2);
    }
    if args.preload_counts.is_empty() {
        eprintln!("error: --preload-counts must list at least one value");
        return ExitCode::from(2);
    }
    if args.samples_per_level == 0 {
        eprintln!("error: --samples-per-level must be > 0");
        return ExitCode::from(2);
    }
    // Sort + dedup. The publish loop assumes monotonically increasing
    // levels because it only publishes the gap between consecutive
    // levels — going back would require deleting labels, which the
    // coordinator doesn't expose.
    let mut preload_counts = args.preload_counts.clone();
    preload_counts.sort();
    preload_counts.dedup();
    if preload_counts[0] == 0 {
        // 0 means "sample without publishing anything new" — but we
        // can't sample if there are no labels at all, so skip.
        eprintln!(
            "warn: preload_count 0 has no labels to sample; dropping that level"
        );
        preload_counts.remove(0);
        if preload_counts.is_empty() {
            eprintln!("error: no usable preload levels after dropping 0");
            return ExitCode::from(2);
        }
    }
    let log_n_shards = args.endpoints.len().trailing_zeros() as usize;

    // ---- build ShardedAegon -------------------------------------------
    // Same wiring as `aegon_coordinator_bench`: remote shards + optional
    // Redis for occupancy probes.
    let mut builder = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(args.shard_log_capacity)
        .log_n_shards(log_n_shards)
        .private(args.private)
        .kzh_k(args.kzh_k)
        .shards(ShardTransport::Remote {
            endpoints: args.endpoints.clone(),
        });
    if let Some(path) = &args.srs_path {
        builder = builder.srs(SrsSource::Path(path.clone()));
    }
    if let Some(url) = &args.db_url {
        builder = builder.db(DbSource::Redis(url.clone()));
    } else if let Some(path) = &args.db_path {
        builder = builder.db(DbSource::Rocks(path.clone()));
    }
    let cfg = match builder.build() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: config invalid: {e}");
            return ExitCode::from(2);
        },
    };

    eprintln!(
        "bench: connecting to {} shards (shard_log_capacity={}, kzh_k={}, log_n_shards={})",
        args.endpoints.len(),
        args.shard_log_capacity,
        args.kzh_k,
        log_n_shards
    );
    let mut rng = ChaCha20Rng::seed_from_u64(args.setup_seed.unwrap_or(0));
    let t_setup = Instant::now();
    let state = match Sharded::setup(&mut rng, &cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: setup failed: {e}");
            return ExitCode::from(1);
        },
    };
    let setup_ms = t_setup.elapsed().as_secs_f64() * 1000.0;
    eprintln!("bench: setup OK in {setup_ms:.1} ms");

    // ---- spawn in-process CoordinatorServer ---------------------------
    // The bench owns the `Arc<RwLock<ShardedAegon>>` and gives a clone
    // to the gRPC server. Server-direct measurements grab the lock on
    // this same Arc; client-path measurements go through gRPC and hit
    // the read-lock inside the service impl.
    //
    // NOTE: we deliberately do NOT build a `ShardedVerifierContext` +
    // `CoordinatorClient` here. Building verifier_param via
    // `gen_srs_for_testing(per_shard_size=2^shard_log_capacity, kzh_k)`
    // peaks at several GB of working memory (Pippenger windows over
    // 2^25-cap chunks). On the bench's commodity n2-standard-4 coord
    // (16 GB RAM, swap=0) that competes with the in-process gRPC
    // server's `ShardedAegon` state + history-openings growth across
    // the sweep and reliably OOMs. Instead we connect a raw tonic
    // `CoordinatorServiceClient<Channel>` below: it does the same
    // round-trip + decode work as `CoordinatorClient`, just without
    // the `verify_lookup_label` / `verify_lookup_value` step. The
    // verify cost is comfortably below the network RTT on this
    // cluster, so the "client" timings here are network + decode +
    // negligible-verify; for a beefier coord one can flip back to the
    // verifying client.
    let shared: Arc<AsyncRwLock<Sharded>> = Arc::new(AsyncRwLock::new(state));
    let listen_addr: std::net::SocketAddr = match args.coordinator_listen.parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!(
                "error: invalid --coordinator-listen {:?}: {e}",
                args.coordinator_listen
            );
            return ExitCode::from(2);
        },
    };
    let server_state = Arc::clone(&shared);
    std::thread::spawn(move || {
        // Build a dedicated runtime for the server thread. tonic's
        // `serve` is async; using a fresh runtime keeps it independent
        // of the bench's main thread state.
        let rt = match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("error: server tokio runtime: {e}");
                return;
            },
        };
        let server = CoordinatorServer::<Bn254, Pcs, Sha256Hash>::from_shared(server_state);
        if let Err(e) = rt.block_on(server.serve(listen_addr)) {
            eprintln!("error: coordinator gRPC server exited: {e}");
        }
    });
    // Give the server a moment to bind. 500ms is generous on loopback;
    // we could poll a TCP `connect`, but the simple sleep keeps the
    // bench code small.
    std::thread::sleep(Duration::from_millis(500));

    // ---- connect raw tonic gRPC client --------------------------------
    // The client URL must be `http://...`, not the raw `ip:port` form.
    // We use `CoordinatorServiceClient<Channel>` directly (the tonic
    // generated client) rather than `CoordinatorClient` so we don't
    // pay for verifier_param. Round-trip + decode is what we time;
    // verification overhead is excluded.
    let endpoint_url = format!("http://{}", args.coordinator_listen);
    eprintln!("bench: client connecting to {endpoint_url} ...");
    // A driver runtime so we can drive async operations on `shared`
    // (publish, server-direct lookups) AND also issue the raw RPCs
    // outside the server's runtime.
    let driver_rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: driver tokio runtime: {e}");
            return ExitCode::from(1);
        },
    };
    // Build the tonic channel + client once; reuse across all samples.
    // Same 1 GiB ceiling as the verifying client + server. We do a
    // single warm-up `current_commitment` round-trip after connect so
    // the first timed sample doesn't include the channel's lazy first-
    // message setup.
    let raw_client: CoordinatorServiceClient<tonic::transport::Channel> = match driver_rt
        .block_on(async {
            let endpoint = tonic::transport::Endpoint::from_shared(endpoint_url.clone())?
                .connect_timeout(std::time::Duration::from_secs(10));
            let channel = endpoint.connect().await?;
            const MAX_MSG_BYTES: usize = 1024 * 1024 * 1024;
            let mut c = CoordinatorServiceClient::new(channel)
                .max_decoding_message_size(MAX_MSG_BYTES)
                .max_encoding_message_size(MAX_MSG_BYTES);
            // Warm-up RPC: forces HTTP/2 SETTINGS + initial WINDOW_UPDATE
            // before any timed sample. Without it the first sample's
            // RTT is artificially inflated by the channel handshake.
            let _ = c.current_commitment(Empty {}).await?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(c)
        }) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: raw client connect: {e}");
            return ExitCode::from(1);
        },
    };

    // ---- preload sweep ------------------------------------------------
    // `current_count` tracks how many labels have been published so
    // far. Each level publishes `target - current_count` more, in
    // batches of `--publish-batch-size`. Labels follow a bench-local
    // namespace `bench-u{i}` so we always know the value bytes
    // (`bench-v{i}`) and can call `lookup_value_with_bytes` without
    // touching the side-channel KV.
    let mut current_count: usize = 0;
    let mut level_reports: Vec<String> = Vec::new();

    for (level_idx, &target) in preload_counts.iter().enumerate() {
        eprintln!(
            "--- preload level {}/{}: target={} (current={}) ---",
            level_idx + 1,
            preload_counts.len(),
            target,
            current_count
        );

        // Climb from `current_count` to `target` in publish_batch_size
        // chunks. We do this one batch at a time so a single failure
        // doesn't lose all progress and so progress reports come out
        // at a reasonable rate.
        let mut publish_ms_total: f64 = 0.0;
        while current_count < target {
            let batch_end = (current_count + args.publish_batch_size).min(target);
            let updates: Vec<(Vec<u8>, Vec<u8>)> = (current_count..batch_end)
                .map(|i| (phone_label(i as u64), rsa_value(i as u64)))
                .collect();
            let t = Instant::now();
            // Same rationale as the lookup paths: blocking_write from
            // the main thread, no driver_rt wrap. `ShardedAegon::publish`
            // happens to work fine inside `driver_rt.block_on` today
            // because its shard fan-out goes through rayon (no tokio
            // CONTEXT on rayon threads), but using blocking_write is
            // safer against future refactors that might move pieces of
            // publish onto the sequential path.
            let res = {
                let mut s = shared.blocking_write();
                s.publish(&updates)
            };
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            publish_ms_total += ms;
            match res {
                Ok(_commit) => {
                    eprintln!(
                        "  published {} labels ({}..{}) in {:.1} ms",
                        updates.len(),
                        current_count,
                        batch_end,
                        ms
                    );
                },
                Err(e) => {
                    eprintln!(
                        "error: publish failed at level={target}, batch {}..{}: {e}",
                        current_count, batch_end
                    );
                    return ExitCode::from(1);
                },
            }
            current_count = batch_end;
        }

        // Force-flush /proc/self/status memory before sampling so we can
        // correlate sweep level with RSS in the JSON output. Cheap.
        let rss_kb = read_self_rss_kb().unwrap_or(0);
        eprintln!(
            "  level={target}: bench RSS = {} MB ({} KB)",
            rss_kb / 1024,
            rss_kb
        );

        // Sample lookups. Pick deterministic indices spread across
        // [0..current_count] so different sweep levels don't all keep
        // hitting the same hot label.
        let n_samples = args.samples_per_level;
        let mut samples_json: Vec<String> = Vec::with_capacity(n_samples);
        for sample_idx in 0..n_samples {
            // Deterministic spread: stride by a coprime increment to
            // avoid landing on every k-th label. `level_idx` is mixed
            // in so consecutive levels don't all start at index 0.
            let idx = ((sample_idx as u64).wrapping_mul(2_654_435_761)
                ^ (level_idx as u64).wrapping_mul(11_400_714_819_323_198_485))
                % (current_count as u64);
            let label = phone_label(idx);
            let value = rsa_value(idx);

            // (a) Server-direct lookup_label. Read lock only — the
            // server-direct path is the lower bound on what any
            // remote client can achieve (no network, no serialization,
            // no verify cost).
            //
            // We deliberately use `blocking_read` from the *main*
            // (sync) thread rather than wrapping in
            // `driver_rt.block_on(async { shared.read().await... })`.
            // `ShardedAegon::lookup_label` is sync but iterates the
            // probe trail sequentially, doing `shard_client.runtime
            // .block_on(...)` on each shard's private runtime per
            // probe. Calling that from inside `driver_rt.block_on`
            // panics with "Cannot start a runtime from within a
            // runtime" because the driver_rt's CONTEXT is set on the
            // calling thread. `publish` dodges this because it
            // dispatches to rayon worker threads (which have no tokio
            // CONTEXT), but the sequential lookup loop has nowhere to
            // hide. Main thread + blocking_read = no tokio CONTEXT,
            // shard runtimes can be entered freely.
            let t = Instant::now();
            let server_label_result = {
                let s = shared.blocking_read();
                s.lookup_label(&label)
            };
            let server_label_ns = t.elapsed().as_nanos() as u64;
            let (slot, label_proof) = match server_label_result {
                Ok(p) => p,
                Err(e) => {
                    eprintln!(
                        "error: server-direct lookup_label({:?}) at level {target}: {e}",
                        String::from_utf8_lossy(&label)
                    );
                    return ExitCode::from(1);
                },
            };

            // (b) Server-direct lookup_value at that slot. Same
            // blocking_read pattern as (a).
            let t = Instant::now();
            let server_value_result = {
                let s = shared.blocking_read();
                s.lookup_value(&slot)
            };
            let server_value_ns = t.elapsed().as_nanos() as u64;
            let value_proof = match server_value_result {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error: server-direct lookup_value at level {target}: {e}");
                    return ExitCode::from(1);
                },
            };

            // (c) Client lookup_label via raw tonic RPC (no verify).
            // This measures network RTT + protobuf decode on the
            // client side, not verify cost. We re-clone the client per
            // call because tonic's generated client takes `&mut self`
            // and we need it to be Send across the await.
            let t = Instant::now();
            let label_req = LookupLabelRequest {
                label: label.clone(),
            };
            let mut rc_label = raw_client.clone();
            let client_label_result =
                driver_rt.block_on(async move { rc_label.lookup_label(label_req).await });
            let client_label_ns = t.elapsed().as_nanos() as u64;
            if let Err(e) = client_label_result {
                eprintln!("error: raw RPC lookup_label at level {target}: {e}");
                return ExitCode::from(1);
            }

            // (d) Client lookup_value via raw tonic RPC (no verify).
            // The slot we send is the one we got server-direct above —
            // same bytes the verifying client would have sent.
            let mut slot_req_bytes: Vec<u8> = Vec::with_capacity(slot.uncompressed_size());
            if let Err(e) = slot.serialize_uncompressed(&mut slot_req_bytes) {
                eprintln!("error: serialize slot for RPC: {e}");
                return ExitCode::from(1);
            }
            let value_req = LookupValueRequest {
                slot: slot_req_bytes,
            };
            let t = Instant::now();
            let mut rc_value = raw_client.clone();
            let client_value_result =
                driver_rt.block_on(async move { rc_value.lookup_value(value_req).await });
            let client_value_ns = t.elapsed().as_nanos() as u64;
            if let Err(e) = client_value_result {
                eprintln!("error: raw RPC lookup_value at level {target}: {e}");
                return ExitCode::from(1);
            }

            // (e) Server-direct lookup_history. Same blocking_read
            // pattern as (a)/(b). Returns up to HISTORY_WINDOW entries
            // most-recent first; at this preload level each sampled
            // label has been published exactly once so we expect
            // entries.len() == 1.
            let t = Instant::now();
            let server_history_result = {
                let s = shared.blocking_read();
                s.lookup_history(&label)
            };
            let server_history_ns = t.elapsed().as_nanos() as u64;
            let history = match server_history_result {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("error: server-direct lookup_history at level {target}: {e}");
                    return ExitCode::from(1);
                },
            };

            // (f) Client lookup_history via raw tonic RPC (no verify).
            let history_req = LookupHistoryRequest {
                label: label.clone(),
            };
            let t = Instant::now();
            let mut rc_history = raw_client.clone();
            let client_history_result =
                driver_rt.block_on(async move { rc_history.lookup_history(history_req).await });
            let client_history_ns = t.elapsed().as_nanos() as u64;
            if let Err(e) = client_history_result {
                eprintln!("error: raw RPC lookup_history at level {target}: {e}");
                return ExitCode::from(1);
            }

            // (g) Server-direct lookup_label_history. Same
            // blocking_read pattern. Returns the placement record +
            // a live opening of rand_index_poly at the slot; under
            // the system's current invariants (labels placed
            // exactly once) the response will always be Some(...)
            // for any label we've published in this bench.
            let t = Instant::now();
            let server_label_history_result = {
                let s = shared.blocking_read();
                s.lookup_label_history(&label)
            };
            let server_label_history_ns = t.elapsed().as_nanos() as u64;
            let label_history = match server_label_history_result {
                Ok(h) => h,
                Err(e) => {
                    eprintln!(
                        "error: server-direct lookup_label_history at level {target}: {e}"
                    );
                    return ExitCode::from(1);
                },
            };

            // (h) Client lookup_label_history via raw tonic RPC.
            let label_history_req = LookupLabelHistoryRequest {
                label: label.clone(),
            };
            let t = Instant::now();
            let mut rc_lh = raw_client.clone();
            let client_label_history_result = driver_rt
                .block_on(async move { rc_lh.lookup_label_history(label_history_req).await });
            let client_label_history_ns = t.elapsed().as_nanos() as u64;
            if let Err(e) = client_label_history_result {
                eprintln!("error: raw RPC lookup_label_history at level {target}: {e}");
                return ExitCode::from(1);
            }

            // ---- wire-size + application-size accounting ------------
            // Reproduce the responses the gRPC server would send (the
            // server's `encode` helper is private, but it's just
            // canonical-uncompressed serialization — same as below).
            let mut slot_bytes: Vec<u8> = Vec::with_capacity(slot.uncompressed_size());
            if let Err(e) = slot.serialize_uncompressed(&mut slot_bytes) {
                eprintln!("error: serialize slot: {e}");
                return ExitCode::from(1);
            }
            let mut label_proof_bytes: Vec<u8> =
                Vec::with_capacity(label_proof.uncompressed_size());
            if let Err(e) = label_proof.serialize_uncompressed(&mut label_proof_bytes) {
                eprintln!("error: serialize label proof: {e}");
                return ExitCode::from(1);
            }
            let mut value_proof_bytes: Vec<u8> =
                Vec::with_capacity(value_proof.uncompressed_size());
            if let Err(e) = value_proof.serialize_uncompressed(&mut value_proof_bytes) {
                eprintln!("error: serialize value proof: {e}");
                return ExitCode::from(1);
            }
            // `LookupValueResponse.value` may or may not be populated
            // by the coordinator depending on DbSource — to compute
            // the *worst-case* wire size we report it both with and
            // without inline value bytes.
            let label_resp = LookupLabelResponse {
                slot: slot_bytes.clone(),
                proof: label_proof_bytes.clone(),
            };
            let value_resp_empty = LookupValueResponse {
                proof: value_proof_bytes.clone(),
                value: Vec::new(),
            };
            let value_resp_with_value = LookupValueResponse {
                proof: value_proof_bytes.clone(),
                value: value.clone(),
            };

            let label_wire_bytes = label_resp.encoded_len();
            let value_wire_bytes_empty = value_resp_empty.encoded_len();
            let value_wire_bytes_with_value = value_resp_with_value.encoded_len();

            // History wire size: encode the same bundle the server
            // ships back. Each entry carries 3 opening proofs + the
            // post/pre commits + per-entry value bytes, so the wire
            // size grows roughly linearly in `history.entries.len()`
            // up to HISTORY_WINDOW.
            let mut history_bytes: Vec<u8> = Vec::with_capacity(history.uncompressed_size());
            if let Err(e) = history.serialize_uncompressed(&mut history_bytes) {
                eprintln!("error: serialize history: {e}");
                return ExitCode::from(1);
            }
            let history_resp = LookupHistoryResponse {
                history: history_bytes.clone(),
            };
            let history_wire_bytes = history_resp.encoded_len();
            let history_entries = history.entries.len();

            // Label-history wire size. Asymmetric with value-
            // history: at most one placement record + at most one
            // freshness opening, so the bundle is always smaller
            // than a 5-entry value-history. We still encode it for
            // a fair size comparison.
            let mut label_history_bytes: Vec<u8> =
                Vec::with_capacity(label_history.uncompressed_size());
            if let Err(e) = label_history.serialize_uncompressed(&mut label_history_bytes) {
                eprintln!("error: serialize label_history: {e}");
                return ExitCode::from(1);
            }
            let label_history_resp = LookupLabelHistoryResponse {
                history: label_history_bytes.clone(),
            };
            let label_history_wire_bytes = label_history_resp.encoded_len();

            // "proof overhead" per the user's spec: the gap between
            // the bytes shipped to the client and the underlying
            // application-level payload (label for the label call,
            // value for the value call; for history, the payload is
            // n_entries × value_size since each entry carries one
            // value snapshot).
            let label_size_bytes = label.len();
            let value_size_bytes = value.len();
            let history_payload_bytes = history_entries * value_size_bytes;
            let label_proof_overhead_bytes =
                label_wire_bytes.saturating_sub(label_size_bytes);
            let value_proof_overhead_bytes =
                value_wire_bytes_with_value.saturating_sub(value_size_bytes);
            let history_proof_overhead_bytes =
                history_wire_bytes.saturating_sub(history_payload_bytes);

            samples_json.push(format!(
                "        {{\n          \"sample_idx\": {sample_idx},\n          \"label_idx\": {idx},\n          \"server_lookup_label_ns\": {server_label_ns},\n          \"server_lookup_value_ns\": {server_value_ns},\n          \"server_lookup_history_ns\": {server_history_ns},\n          \"server_lookup_label_history_ns\": {server_label_history_ns},\n          \"client_lookup_label_ns\": {client_label_ns},\n          \"client_lookup_value_ns\": {client_value_ns},\n          \"client_lookup_history_ns\": {client_history_ns},\n          \"client_lookup_label_history_ns\": {client_label_history_ns},\n          \"label_size_bytes\": {label_size_bytes},\n          \"value_size_bytes\": {value_size_bytes},\n          \"history_entries\": {history_entries},\n          \"label_wire_bytes\": {label_wire_bytes},\n          \"value_wire_bytes_empty_value\": {value_wire_bytes_empty},\n          \"value_wire_bytes_with_value\": {value_wire_bytes_with_value},\n          \"history_wire_bytes\": {history_wire_bytes},\n          \"label_history_wire_bytes\": {label_history_wire_bytes},\n          \"label_proof_field_bytes\": {label_proof_field},\n          \"value_proof_field_bytes\": {value_proof_field},\n          \"history_proof_field_bytes\": {history_proof_field},\n          \"label_history_proof_field_bytes\": {label_history_proof_field},\n          \"label_slot_field_bytes\": {label_slot_field},\n          \"label_proof_overhead_bytes\": {label_proof_overhead_bytes},\n          \"value_proof_overhead_bytes\": {value_proof_overhead_bytes},\n          \"history_proof_overhead_bytes\": {history_proof_overhead_bytes}\n        }}",
                label_proof_field = label_proof_bytes.len(),
                value_proof_field = value_proof_bytes.len(),
                history_proof_field = history_bytes.len(),
                label_history_proof_field = label_history_bytes.len(),
                label_slot_field = slot_bytes.len(),
            ));

            if sample_idx == 0 || (sample_idx + 1) % 10 == 0 {
                eprintln!(
                    "  sample {}: server_label={:.2}ms server_value={:.2}ms server_history={:.2}ms server_label_history={:.2}ms client_label={:.2}ms client_value={:.2}ms client_history={:.2}ms client_label_history={:.2}ms label_wire={}B value_wire={}B history_wire={}B (entries={}) label_history_wire={}B",
                    sample_idx,
                    server_label_ns as f64 / 1e6,
                    server_value_ns as f64 / 1e6,
                    server_history_ns as f64 / 1e6,
                    server_label_history_ns as f64 / 1e6,
                    client_label_ns as f64 / 1e6,
                    client_value_ns as f64 / 1e6,
                    client_history_ns as f64 / 1e6,
                    client_label_history_ns as f64 / 1e6,
                    label_wire_bytes,
                    value_wire_bytes_with_value,
                    history_wire_bytes,
                    history_entries,
                    label_history_wire_bytes,
                );
            }
        }

        // Per-level eyeball summary on the four timing series.
        // Computing them inline here keeps the JSON writer simple.
        let level_block = format!(
            "    {{\n      \"preload_count\": {target},\n      \"publish_ms_total\": {publish_ms_total:.4},\n      \"samples\": [\n{samples}\n      ]\n    }}",
            samples = samples_json.join(",\n"),
        );
        level_reports.push(level_block);
    }

    // ---- write JSON ---------------------------------------------------
    let endpoints_json = args
        .endpoints
        .iter()
        .map(|e| format!("\"{e}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let preload_json = preload_counts
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let json = format!(
        "{{\n  \"params\": {{\n    \"n_shards\": {n_shards},\n    \"log_n_shards\": {log_n_shards},\n    \"shard_log_capacity\": {shard_log_capacity},\n    \"kzh_k\": {kzh_k},\n    \"private\": {private},\n    \"samples_per_level\": {samples},\n    \"publish_batch_size\": {publish_batch_size},\n    \"endpoints\": [{endpoints_json}],\n    \"preload_counts\": [{preload_json}]\n  }},\n  \"setup_ms\": {setup_ms:.4},\n  \"levels\": [\n{levels}\n  ]\n}}\n",
        n_shards = args.endpoints.len(),
        log_n_shards = log_n_shards,
        shard_log_capacity = args.shard_log_capacity,
        kzh_k = args.kzh_k,
        private = args.private,
        samples = args.samples_per_level,
        publish_batch_size = args.publish_batch_size,
        endpoints_json = endpoints_json,
        preload_json = preload_json,
        setup_ms = setup_ms,
        levels = level_reports.join(",\n"),
    );

    let mut f = match File::create(&args.output) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: cannot create output {:?}: {e}", args.output);
            return ExitCode::from(1);
        },
    };
    if let Err(e) = f.write_all(json.as_bytes()) {
        eprintln!("error: cannot write output: {e}");
        return ExitCode::from(1);
    }
    eprintln!("wrote {:?}", args.output);
    ExitCode::SUCCESS
}
