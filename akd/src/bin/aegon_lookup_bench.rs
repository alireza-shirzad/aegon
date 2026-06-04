//! `aegon_lookup_bench` — combined lookup + publish bench, sweeping
//! preloaded dictionary sizes.
//!
//! At each "preload level" N (specified either directly via
//! `--preload-counts` or as a fraction of true capacity via
//! `--fill-percents` + `--true-log-capacity`), the bench:
//!
//!   1. **Top-up**: publishes fresh `(label, value)` pairs in the
//!      bench's known namespace (`phone_label(idx)` for `idx` in
//!      `[0..N)`) until the sampleable namespace has N entries. The
//!      growth is incremental — going from level N₁ to level N₂
//!      only publishes the gap N₂ − N₁.
//!   2. **Lookup bench**: picks `--samples-per-level` already-loaded
//!      labels and, for each, measures four server-direct +
//!      four client-via-gRPC timings:
//!        * label lookup        — opens index_poly at the label's slot.
//!        * value lookup        — opens value_poly at that slot.
//!        * value history       — DB read of value-history entries +
//!                                a live freshness opening.
//!        * label history       — DB read of placement record + a
//!                                live freshness opening.
//!      For each lookup type, reports three size numbers:
//!        * `*_data_bytes`      — literal application payload (label
//!                                bytes for label/label-history;
//!                                value bytes for value/value-history,
//!                                summed across entries for the latter).
//!        * `*_proof_bytes`     — uncompressed-serialised size of the
//!                                cryptographic proof object the server
//!                                returns. For value-history this
//!                                excludes the in-band value_bytes (we
//!                                subtract them out so data + proof
//!                                doesn't double-count).
//!        * `*_total_bytes`     — `data + proof`, the total payload
//!                                metric per the user's spec.
//!   3. **Publish bench (optional)**: if `--publish-batch-sizes` is
//!      provided, sweeps each batch size and runs `--publish-samples-
//!      per-batch` publishes per size, recording per-publish wall
//!      time + the commit-class byte sizes (index / value / rand-
//!      index / rand-value, summed across shards, plus the total
//!      `ShardedEpochCommitment` size). The publish bench uses a
//!      disjoint namespace from the lookup-sampleable one, so its
//!      samples don't pollute future-stage lookup measurements.
//!
//! ## Modes
//!
//! * **Local in-process** (no `--endpoints`): builds an in-process
//!   `ShardedAegon` with `--n-shards` (default 1). The `--initial-
//!   prefill-count` flag bulk-loads anonymous filler via
//!   `prefill_random_per_shard` (only valid at epoch 0, before any
//!   publish), so the dict can land at a realistic fill level
//!   without paying for hundreds of thousands of `publish` calls.
//!   Used for the **small** (`shard_log_cap=22`, `true_log_cap=20`)
//!   and **medium** (`shard_log_cap=28`, `true_log_cap=26`) regimes.
//!
//! * **Remote / distributed** (`--endpoints http://...,...`):
//!   connects to a running cluster. The cluster is expected to be
//!   prefilled to the desired baseline via
//!   `aegon_shard_server --prefill-count` (the lookup namespace's
//!   labels are then published on top by this bench). Used for the
//!   **planetary** (`shard_log_cap=29`, `true_log_cap=32`,
//!   `n_shards=32`) regime.
//!
//! The coordinator gRPC server runs in a background thread on
//! `--coordinator-listen` (default `127.0.0.1:50190`), driven by the
//! same `Arc<RwLock<ShardedAegon>>` the bench publishes against. The
//! bench's "server-direct" measurements take the lock outside of
//! gRPC, so they see the same state the client side sees — just
//! without serialization / network / verify cost.
//!
//! ## Usage
//!
//! Local single-shard small regime:
//! ```text
//! aegon_lookup_bench \
//!   --shard-log-capacity 22 --true-log-capacity 20 --kzh-k 10 \
//!   --setup-seed 42 --n-shards 1 \
//!   --fill-percents 1,30,60,90 \
//!   --samples-per-level 20 \
//!   --publish-batch-sizes 2,4,8,16,32,64 --publish-samples-per-batch 3 \
//!   --output /tmp/aegon-lookup-bench.json
//! ```
//!
//! Remote planetary regime (cluster prefilled externally):
//! ```text
//! aegon_lookup_bench \
//!   --shard-log-capacity 29 --kzh-k 10 --setup-seed 42 \
//!   --endpoints http://aegon-shard-0:50051,...,http://aegon-shard-31:50051 \
//!   --preload-counts 1000 \
//!   --samples-per-level 20 \
//!   --publish-batch-sizes 4096,8192,16384,32768,65536,131072 \
//!   --publish-samples-per-batch 3 \
//!   --db-url redis://aegon-bench-db:6379 \
//!   --output /tmp/aegon-lookup-bench.json
//! ```

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "mimalloc_alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use akd::aegon::coordinator_grpc::{
    proto::{
        coordinator_service_client::CoordinatorServiceClient, Empty, LookupHistoryRequest,
        LookupHistoryResponse, LookupLabelHistoryRequest, LookupLabelHistoryResponse,
        LookupLabelRequest, LookupLabelResponse, LookupValueRequest,
        LookupValueResponse,
    },
    CoordinatorServer,
};
use akd::aegon::{
    optimal_kzh_k, verify_sharded_invariance, AuditState, DbSource, EcVrfHash, ShardTransport,
    ShardedAegon, ShardedAegonConfig, SrsSource, VrfProver,
};
use ark_ec::pairing::Pairing;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::SeedableRng;
use clap::Parser;
use prost::Message;
use rand_chacha::ChaCha20Rng;
use tokio::sync::RwLock as AsyncRwLock;

type Pcs = KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, EcVrfHash>;

/// Realistic application sizing for this bench:
/// labels are 12-byte ASCII phone numbers in E.164 form
/// (`+1` + 10 digits, e.g. `+10000000123`) and values are 256-byte
/// random buffers that stand in for a 2048-bit RSA public key. The
/// goal isn't to use _correct_ RSA bytes — the AKD only sees the byte
/// string — but to make the wire-size / hash-cost / DB-footprint
/// numbers reflect what a real deployment would see, instead of the
/// ~10-byte `bench-u{i}` / `bench-v{i}` that the old bench used.
const RSA_VALUE_LEN: usize = 256;

/// Index-namespace offset reserved for the publish bench. Lookup-
/// sampleable labels live in `phone_label(i)` for `i in [0..2^33)`;
/// publish-bench labels live at `phone_label(PUBLISH_BENCH_NAMESPACE_OFFSET
/// + j)` for `j in [0..)`. The two namespaces never overlap, so
/// publish-bench samples cannot accidentally become lookup-bench
/// sampleable. 2^33 is comfortably above the planetary regime's
/// `true_log_capacity = 32`. Both namespaces share the `phone_label`
/// + `rsa_value` byte format so wire-size measurements stay
/// apples-to-apples.
const PUBLISH_BENCH_NAMESPACE_OFFSET: u64 = 1u64 << 33;

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
    about = "Sweep preload counts (or fill_percents) and measure lookup latency (server + client), \
             proof wire sizes, and optional per-stage publish bench."
)]
struct Args {
    /// log_2 of slots per shard. Must match every shard server.
    #[arg(long)]
    shard_log_capacity: usize,

    /// log_2 of the TRUE total dictionary size — fill percentages are
    /// computed against this, not the over-provisioned shard size.
    /// Required when `--fill-percents` is used.
    #[arg(long)]
    true_log_capacity: Option<usize>,

    /// KZH-k block parameter. Defaults to
    /// `optimal_kzh_k(shard_log_capacity)`. Must match every shard
    /// in distributed mode.
    #[arg(long)]
    kzh_k: Option<usize>,

    /// Number of shards. Defaults to 1 (in-process local mode).
    /// `n_shards > 1` requires `--endpoints` and switches to remote
    /// (distributed) mode.
    #[arg(long, default_value_t = 1)]
    n_shards: usize,

    /// Comma-separated shard endpoints. Length must be a power of two
    /// and match `--n-shards`. When empty, runs in local in-process
    /// mode with `--n-shards` (default 1) in-memory shards.
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

    /// Comma-separated list of remote `aegon_masking_server` endpoints
    /// (e.g. `http://127.0.0.1:50061,http://127.0.0.1:50062`). When
    /// set, every in-process shard's value-side opening fetches its
    /// masking package from the server(s) instead of using the default
    /// in-process pool. Multiple endpoints are dispatched round-robin
    /// via `MaskingClientPool`, matching the cluster shard
    /// architecture and unlocking >1 masking server's worth of
    /// throughput per shard. May be repeated, or pass a single
    /// comma-separated string.
    #[arg(long = "masking-addr", value_delimiter = ',')]
    masking_addr: Vec<String>,

    /// Sweep points: comma-separated **lookup-sampleable** label
    /// counts that the bench will publish up to before sampling
    /// lookups. Must be strictly increasing; the bench publishes the
    /// gap between consecutive points so e.g. `1,2,4,...,N` publishes
    /// N labels total across the sweep, not 1+2+4+...+N. Mutually
    /// exclusive with `--fill-percents`.
    #[arg(long, value_delimiter = ',')]
    preload_counts: Vec<usize>,

    /// Alternative sweep specification: comma-separated fill
    /// percentages against `true_log_capacity`. Each percentage `p`
    /// produces a sweep target of `floor(2^true_log_capacity * p /
    /// 100)`. Requires `--true-log-capacity`. Mutually exclusive with
    /// `--preload-counts`. The lookup-sampleable count at each stage
    /// is `target - initial_prefill_count` (clamped to ≥ 0).
    #[arg(long, value_delimiter = ',')]
    fill_percents: Vec<u32>,

    /// **Local mode only**: bulk-prefill the in-process shards with
    /// `count` random `(slot, h_label, h_value)` entries via
    /// `prefill_random_per_shard` BEFORE any publish. These entries
    /// are anonymous (no label↔slot mapping in routing) so they are
    /// NOT lookup-sampleable — they just pad the dict to a realistic
    /// fill level so lookup latencies reflect probe-trail depth at
    /// scale. Must be called at epoch 0 (no prior publishes).
    #[arg(long, default_value_t = 0)]
    initial_prefill_count: u64,

    /// Prefill seed base (local mode bulk-prefill only). Shard `i`
    /// is seeded with `prefill_seed + i`, mirroring the cluster
    /// pattern.
    #[arg(long, default_value_t = 1)]
    prefill_seed: u64,

    /// How many lookup samples to draw per preload level. Each sample
    /// times all eight call paths (server + client × {label, value,
    /// value-history, label-history}) and records one wire-size row.
    #[arg(long, default_value_t = 10)]
    samples_per_level: usize,

    /// Internal batch size used to publish the gap between sweep
    /// points. Larger batches amortize the publish's fixed cost; too
    /// large and a single batch can exceed the coordinator's working
    /// memory. 1024 is a safe default on the bench cluster.
    #[arg(long, default_value_t = 1024)]
    publish_batch_size: usize,

    /// **Publish bench**: comma-separated batch sizes to sweep at
    /// every preload level (after the lookup samples). When empty,
    /// the publish bench is skipped. Each sample uses a disjoint
    /// namespace so it doesn't pollute the lookup-sampleable space.
    #[arg(long, value_delimiter = ',')]
    publish_batch_sizes: Vec<usize>,

    /// Samples per publish-bench batch size. Default 3 — enough for a
    /// stable median.
    #[arg(long, default_value_t = 3)]
    publish_samples_per_batch: usize,

    /// **Audit bench**: number of consecutive epoch-transition audits
    /// to time per stage via `verify_sharded_invariance`. Starts from
    /// a fresh `AuditState` at epoch 0 and walks forward, so the chain
    /// state is correct at every step. Each sample times one
    /// `verify_sharded_invariance(prev_commit, next_commit)` call and
    /// records `audit_proof_bytes` = `next.uncompressed_size()` (the
    /// bulletin-board fetch the auditor pays per epoch).
    ///
    /// Requires at least `audit_samples + 1` published epochs by the
    /// time the audit phase runs (which happens AFTER the lookup and
    /// publish-bench phases, so the publish bench's own batches count
    /// toward the available epoch chain). Set to 0 to skip auditing.
    #[arg(long, default_value_t = 5)]
    audit_samples: usize,

    /// Concurrency levels for the per-stage **throughput sweep**.
    /// Comma-separated list, e.g. `1,4,16,64,256`. Empty list (the
    /// default) skips the sweep entirely. Runs after lookup samples
    /// + audit, before publish_bench, so the cluster state is exactly
    /// what audit just saw. Each level spawns N concurrent client
    /// tasks that loop on a single chosen RPC kind
    /// (`--throughput-lookup-kind`) for `--throughput-window-secs`
    /// seconds; the achieved QPS at each level + p50/p90/p99 latency
    /// land in the JSON's `concurrency_sweep` block.
    #[arg(long, value_delimiter = ',', default_value = "")]
    throughput_concurrencies: Vec<usize>,

    /// Measurement window per concurrency level, in seconds.
    #[arg(long, default_value_t = 20)]
    throughput_window_secs: u64,

    /// Warmup window per concurrency level (latencies discarded).
    /// Lets the tonic channels reach steady state and the masking
    /// queue fill before we count.
    #[arg(long, default_value_t = 3)]
    throughput_warmup_secs: u64,

    /// Which lookup RPC the throughput sweep drives. One of:
    /// `label`, `value` (label+value chain — the realistic client
    /// flow), `history`, `label_history`.
    #[arg(long, default_value = "value")]
    throughput_lookup_kind: String,

    /// Per-RPC timeout in the throughput sweep, in seconds. Bounds
    /// the resource footprint of a stalled request (gRPC buffer +
    /// pending future) so high concurrency can't pile up forever.
    /// Requests that exceed this deadline count as errors.
    #[arg(long, default_value_t = 10)]
    throughput_rpc_timeout_secs: u64,

    /// Early-stop trigger: error rate (errors / attempts) at which
    /// we abort the sweep at the current fill level. Lets us bail
    /// before the sweep cascades a cluster failure. 0 = disabled.
    #[arg(long, default_value_t = 0.25)]
    throughput_max_error_rate: f64,

    /// Early-stop trigger: p99 latency (ms) at which we stop
    /// climbing concurrency at the current fill level. The sweep
    /// completes the current level's measurement and skips the
    /// remaining (higher) concurrencies. 0 = disabled.
    #[arg(long, default_value_t = 60000)]
    throughput_max_p99_ms: u64,

    /// Health check before each concurrency level: a single
    /// sequential lookup with a tight timeout. If it fails, abort
    /// the sweep at this fill level entirely (preserves the climb
    /// for remaining fill levels). 0 = disabled.
    #[arg(long, default_value_t = 5)]
    throughput_health_check_timeout_secs: u64,

    /// Where to write the JSON timing report.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> ExitCode {
    #[cfg(feature = "tracing_instrument")]
    akd::aegon::tracing_init::init_tree_subscriber();
    let args = Args::parse();

    // ---- mode + arg validation ----
    let local_mode = args.endpoints.is_empty();
    let effective_n_shards = if local_mode { args.n_shards } else { args.endpoints.len() };
    if !effective_n_shards.is_power_of_two() || effective_n_shards == 0 {
        eprintln!(
            "error: n_shards must be a power of two (got {effective_n_shards})"
        );
        return ExitCode::from(2);
    }
    if !local_mode && args.endpoints.len() != args.n_shards {
        eprintln!(
            "error: --endpoints length ({}) must equal --n-shards ({})",
            args.endpoints.len(),
            args.n_shards
        );
        return ExitCode::from(2);
    }
    if args.srs_path.is_none() && args.setup_seed.is_none() {
        eprintln!("error: provide either --srs-path or --setup-seed");
        return ExitCode::from(2);
    }
    if !args.preload_counts.is_empty() && !args.fill_percents.is_empty() {
        eprintln!("error: --preload-counts and --fill-percents are mutually exclusive");
        return ExitCode::from(2);
    }
    if args.preload_counts.is_empty() && args.fill_percents.is_empty() {
        eprintln!("error: provide either --preload-counts or --fill-percents");
        return ExitCode::from(2);
    }
    if !args.fill_percents.is_empty() && args.true_log_capacity.is_none() {
        eprintln!("error: --fill-percents requires --true-log-capacity");
        return ExitCode::from(2);
    }
    if args.samples_per_level == 0 {
        eprintln!("error: --samples-per-level must be > 0");
        return ExitCode::from(2);
    }
    if !args.publish_batch_sizes.is_empty() && args.publish_samples_per_batch == 0 {
        eprintln!("error: --publish-samples-per-batch must be > 0 when --publish-batch-sizes is set");
        return ExitCode::from(2);
    }

    let log_n_shards = effective_n_shards.trailing_zeros() as usize;
    let k = args.kzh_k.unwrap_or_else(|| optimal_kzh_k(args.shard_log_capacity));

    // Resolve preload_counts. Either taken directly from --preload-counts
    // or derived from --fill-percents × true_capacity, minus the initial
    // bulk-prefill (those entries are anonymous and not lookup-sampleable).
    let preload_counts: Vec<usize> = if !args.preload_counts.is_empty() {
        let mut v = args.preload_counts.clone();
        v.sort();
        v.dedup();
        v
    } else {
        let true_cap: u128 = 1u128 << args.true_log_capacity.unwrap();
        let mut pcts = args.fill_percents.clone();
        pcts.sort();
        pcts.dedup();
        pcts.into_iter()
            .map(|pct| {
                let target_total = ((true_cap * pct as u128) / 100u128) as i128;
                let sampleable = target_total - args.initial_prefill_count as i128;
                sampleable.max(0) as usize
            })
            .collect()
    };
    // Sort + dedup once more (fill_percents=0 might map to preload=0
    // alongside other 0s if initial_prefill is large).
    let mut preload_counts = preload_counts;
    preload_counts.sort();
    preload_counts.dedup();
    // We keep a leading 0 — it means "no lookup samples at this stage,
    // but the stage may still run a publish bench against the
    // initial-prefilled state". Without a publish-bench config we
    // skip it.
    let publish_enabled = !args.publish_batch_sizes.is_empty();
    if preload_counts.is_empty() {
        eprintln!("error: empty sweep after dedup");
        return ExitCode::from(2);
    }
    if preload_counts[0] == 0 && !publish_enabled {
        eprintln!(
            "warn: preload_count 0 has no labels to sample and publish bench is disabled; \
             dropping that level"
        );
        preload_counts.remove(0);
        if preload_counts.is_empty() {
            eprintln!("error: no usable preload levels after dropping 0");
            return ExitCode::from(2);
        }
    }

    // ---- build ShardedAegon -------------------------------------------
    let mut builder = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(args.shard_log_capacity)
        .log_n_shards(log_n_shards)
        .private(args.private)
        .kzh_k(k);
    if local_mode {
        builder = builder.shards(ShardTransport::InProcess);
    } else {
        builder = builder.shards(ShardTransport::Remote {
            endpoints: args.endpoints.clone(),
        });
    }
    if let Some(path) = &args.srs_path {
        builder = builder.srs(SrsSource::Path(path.clone()));
    }
    if let Some(url) = &args.db_url {
        builder = builder.db(DbSource::Redis(url.clone()));
    } else if let Some(path) = &args.db_path {
        builder = builder.db(DbSource::Rocks(path.clone()));
    }
    if !args.masking_addr.is_empty() {
        eprintln!(
            "bench: in-process shards will fetch masking packages from {} server(s): {:?}",
            args.masking_addr.len(),
            args.masking_addr,
        );
        builder = builder.masking_addrs(args.masking_addr.clone());
    }
    let cfg = match builder.build() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: config invalid: {e}");
            return ExitCode::from(2);
        },
    };

    eprintln!(
        "bench: mode={} n_shards={} (shard_log_capacity={}, kzh_k={}, log_n_shards={})",
        if local_mode { "local" } else { "remote" },
        effective_n_shards,
        args.shard_log_capacity,
        k,
        log_n_shards
    );
    let mut rng = ChaCha20Rng::seed_from_u64(args.setup_seed.unwrap_or(0));
    let t_setup = Instant::now();
    let mut state = match Sharded::setup(&mut rng, &cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: setup failed: {e}");
            return ExitCode::from(1);
        },
    };
    state.set_vrf_prover(VrfProver::from_env());
    let setup_ms = t_setup.elapsed().as_secs_f64() * 1000.0;
    eprintln!("bench: setup OK in {setup_ms:.1} ms (ECVRF prover attached)");

    // Initial bulk-prefill. Anonymous filler that pads the dict to
    // a realistic fill level without going through publish — keeps
    // the lookup-bench's preload climb cheap even at planetary
    // scale. Must be done before any publish call: `prefill_random`
    // errors at epoch != 0. Works for both local (in-process) and
    // distributed (gRPC ReconfigurePrefill) modes.
    let mut initial_prefill_ms: f64 = 0.0;
    if args.initial_prefill_count > 0 {
        eprintln!(
            "bench: initial prefill of {} anonymous entries (seed_base={})",
            args.initial_prefill_count, args.prefill_seed
        );
        let t = Instant::now();
        if let Err(e) =
            state.prefill_random_per_shard(args.initial_prefill_count, args.prefill_seed)
        {
            eprintln!("error: initial prefill failed: {e}");
            return ExitCode::from(1);
        }
        initial_prefill_ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "bench: initial prefill done in {:.1} s",
            initial_prefill_ms / 1000.0
        );
    }

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
        let server = CoordinatorServer::<Bn254, Pcs, EcVrfHash>::from_shared(server_state);
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
            const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;
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
    // `current_count` tracks how many labels have been published in the
    // **lookup-sampleable** namespace so far (i.e. via `phone_label(i)`
    // for `i in [0..current_count)`). Each stage publishes
    // `target - current_count` more in chunks of `--publish-batch-size`.
    //
    // The bench also tracks a separate publish-bench namespace
    // (`PUBLISH_BENCH_NAMESPACE_OFFSET` upward) so the publish-bench
    // samples never collide with the lookup-bench namespace.
    let mut current_count: usize = 0;
    let mut publish_bench_idx: u64 = PUBLISH_BENCH_NAMESPACE_OFFSET;
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
        let mut preload_publish_ms_total: f64 = 0.0;
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
            preload_publish_ms_total += ms;
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
        //
        // Skip lookup sampling entirely if `current_count == 0` (no
        // labels in the sampleable namespace yet). The stage still
        // runs the publish bench below — useful as a baseline at
        // fill_pct=0 stages.
        let n_samples = if current_count == 0 { 0 } else { args.samples_per_level };
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
            // We re-clone the client per call because tonic's
            // generated client takes `&mut self` and we need it to be
            // Send across the await. Latency is read out of the
            // server-reported `server_processing_micros` field on the
            // response, so it excludes network RTT and client-side
            // protobuf decode — what we actually report is "server
            // handler wall time" (which under load includes tokio
            // queue-wait).
            let label_req = LookupLabelRequest {
                label: label.clone(),
            };
            let mut rc_label = raw_client.clone();
            let client_label_result =
                driver_rt.block_on(async move { rc_label.lookup_label(label_req).await });
            let client_label_ns = match &client_label_result {
                Ok(resp) => resp.get_ref().server_processing_micros.saturating_mul(1000),
                Err(_) => 0,
            };
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
            let mut rc_value = raw_client.clone();
            let client_value_result =
                driver_rt.block_on(async move { rc_value.lookup_value(value_req).await });
            let client_value_ns = match &client_value_result {
                Ok(resp) => resp.get_ref().server_processing_micros.saturating_mul(1000),
                Err(_) => 0,
            };
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
            let mut rc_history = raw_client.clone();
            let client_history_result =
                driver_rt.block_on(async move { rc_history.lookup_history(history_req).await });
            let client_history_ns = match &client_history_result {
                Ok(resp) => resp.get_ref().server_processing_micros.saturating_mul(1000),
                Err(_) => 0,
            };
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
            let mut rc_lh = raw_client.clone();
            let client_label_history_result = driver_rt
                .block_on(async move { rc_lh.lookup_label_history(label_history_req).await });
            let client_label_history_ns = match &client_label_history_result {
                Ok(resp) => resp.get_ref().server_processing_micros.saturating_mul(1000),
                Err(_) => 0,
            };
            if let Err(e) = client_label_history_result {
                eprintln!("error: raw RPC lookup_label_history at level {target}: {e}");
                return ExitCode::from(1);
            }

            // ---- size accounting -------------------------------------
            //
            // Per the user's spec, each lookup type reports three
            // numbers: `*_data_bytes`, `*_proof_bytes`,
            // `*_total_bytes = data + proof`. "Data" is the literal
            // application-level payload (label/value bytes); "proof" is
            // the cryptographic-opening byte cost; "total" is their
            // sum.
            //
            //   * label lookup: data = label bytes (the user's queried
            //     identifier); proof = `serialize_uncompressed(label_
            //     proof)` (does NOT contain the label bytes). The
            //     full gRPC response also ships back the slot bytes —
            //     reported as `label_slot_bytes` for transparency but
            //     NOT included in proof (the user spec sums only data
            //     + proof).
            //
            //   * value lookup: data = value bytes; proof =
            //     `serialize_uncompressed(value_proof)` (does NOT
            //     contain the value bytes — the protobuf sends them
            //     in a separate `value` field).
            //
            //   * value history: data = sum of `value_bytes.len()`
            //     across entries (each entry carries one value
            //     snapshot). proof = serialize_uncompressed length of
            //     the full `ShardedValueHistory` struct MINUS data
            //     (the struct embeds value_bytes inline, so we
            //     subtract to avoid double-counting in
            //     total = data + proof). For HISTORY_WINDOW = 1
            //     entry × 256-byte values the proof dwarfs data, and
            //     `total` matches the full struct's
            //     `serialize_uncompressed` length.
            //
            //   * label history: data = label bytes (per the user
            //     spec — "the one and only label in label history").
            //     proof = serialize_uncompressed length of the full
            //     `ShardedLabelHistory` struct MINUS the label bytes
            //     it inlines (the struct has its own `label: Vec<u8>`
            //     field; subtracting keeps total = data + proof from
            //     double-counting). `total` then equals the struct's
            //     `serialize_uncompressed` length.
            //
            // Also recorded for cross-checking: the actual protobuf-
            // encoded gRPC response sizes (`*_wire_bytes`), so a
            // reader can confirm `data + proof + framing ≈ wire`.

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
            let mut history_bytes: Vec<u8> = Vec::with_capacity(history.uncompressed_size());
            if let Err(e) = history.serialize_uncompressed(&mut history_bytes) {
                eprintln!("error: serialize history: {e}");
                return ExitCode::from(1);
            }
            let mut label_history_bytes: Vec<u8> =
                Vec::with_capacity(label_history.uncompressed_size());
            if let Err(e) = label_history.serialize_uncompressed(&mut label_history_bytes) {
                eprintln!("error: serialize label_history: {e}");
                return ExitCode::from(1);
            }

            // Protobuf wire sizes (for cross-checks / `wire == data +
            // proof + framing` sanity).
            let label_resp = LookupLabelResponse {
                slot: slot_bytes.clone(),
                proof: label_proof_bytes.clone(),
                server_processing_micros: 0,
            };
            let value_resp_with_value = LookupValueResponse {
                proof: value_proof_bytes.clone(),
                value: value.clone(),
                server_processing_micros: 0,
            };
            let history_resp = LookupHistoryResponse {
                history: history_bytes.clone(),
                server_processing_micros: 0,
            };
            let label_history_resp = LookupLabelHistoryResponse {
                history: label_history_bytes.clone(),
                server_processing_micros: 0,
            };
            let label_wire_bytes = label_resp.encoded_len();
            let value_wire_bytes = value_resp_with_value.encoded_len();
            let history_wire_bytes = history_resp.encoded_len();
            let label_history_wire_bytes = label_history_resp.encoded_len();

            let history_entries = history.entries.len();
            let value_history_value_bytes_sum: usize = history
                .entries
                .iter()
                .map(|e| e.value_bytes.len())
                .sum();

            // Per-type data/proof/total triples.
            let label_lookup_data_bytes = label.len();
            let label_lookup_proof_bytes = label_proof_bytes.len();
            let label_lookup_total_bytes =
                label_lookup_data_bytes + label_lookup_proof_bytes;

            let value_lookup_data_bytes = value.len();
            let value_lookup_proof_bytes = value_proof_bytes.len();
            let value_lookup_total_bytes =
                value_lookup_data_bytes + value_lookup_proof_bytes;

            let value_history_lookup_data_bytes = value_history_value_bytes_sum;
            let value_history_lookup_proof_bytes = history_bytes
                .len()
                .saturating_sub(value_history_lookup_data_bytes);
            let value_history_lookup_total_bytes =
                value_history_lookup_data_bytes + value_history_lookup_proof_bytes;

            let label_history_lookup_data_bytes = label.len();
            let label_history_lookup_proof_bytes = label_history_bytes
                .len()
                .saturating_sub(label_history_lookup_data_bytes);
            let label_history_lookup_total_bytes =
                label_history_lookup_data_bytes + label_history_lookup_proof_bytes;

            samples_json.push(format!(
                concat!(
                    "        {{\n",
                    "          \"sample_idx\": {sample_idx},\n",
                    "          \"label_idx\": {idx},\n",
                    "          \"server_lookup_label_ns\": {server_label_ns},\n",
                    "          \"server_lookup_value_ns\": {server_value_ns},\n",
                    "          \"server_lookup_history_ns\": {server_history_ns},\n",
                    "          \"server_lookup_label_history_ns\": {server_label_history_ns},\n",
                    "          \"client_lookup_label_ns\": {client_label_ns},\n",
                    "          \"client_lookup_value_ns\": {client_value_ns},\n",
                    "          \"client_lookup_history_ns\": {client_history_ns},\n",
                    "          \"client_lookup_label_history_ns\": {client_label_history_ns},\n",
                    "          \"history_entries\": {history_entries},\n",
                    "          \"label_lookup_data_bytes\": {l_d},\n",
                    "          \"label_lookup_proof_bytes\": {l_p},\n",
                    "          \"label_lookup_total_bytes\": {l_t},\n",
                    "          \"value_lookup_data_bytes\": {v_d},\n",
                    "          \"value_lookup_proof_bytes\": {v_p},\n",
                    "          \"value_lookup_total_bytes\": {v_t},\n",
                    "          \"value_history_lookup_data_bytes\": {vh_d},\n",
                    "          \"value_history_lookup_proof_bytes\": {vh_p},\n",
                    "          \"value_history_lookup_total_bytes\": {vh_t},\n",
                    "          \"label_history_lookup_data_bytes\": {lh_d},\n",
                    "          \"label_history_lookup_proof_bytes\": {lh_p},\n",
                    "          \"label_history_lookup_total_bytes\": {lh_t},\n",
                    "          \"label_slot_bytes\": {slot_b},\n",
                    "          \"label_response_wire_bytes\": {l_w},\n",
                    "          \"value_response_wire_bytes\": {v_w},\n",
                    "          \"history_response_wire_bytes\": {h_w},\n",
                    "          \"label_history_response_wire_bytes\": {lh_w}\n",
                    "        }}"
                ),
                sample_idx = sample_idx,
                idx = idx,
                server_label_ns = server_label_ns,
                server_value_ns = server_value_ns,
                server_history_ns = server_history_ns,
                server_label_history_ns = server_label_history_ns,
                client_label_ns = client_label_ns,
                client_value_ns = client_value_ns,
                client_history_ns = client_history_ns,
                client_label_history_ns = client_label_history_ns,
                history_entries = history_entries,
                l_d = label_lookup_data_bytes,
                l_p = label_lookup_proof_bytes,
                l_t = label_lookup_total_bytes,
                v_d = value_lookup_data_bytes,
                v_p = value_lookup_proof_bytes,
                v_t = value_lookup_total_bytes,
                vh_d = value_history_lookup_data_bytes,
                vh_p = value_history_lookup_proof_bytes,
                vh_t = value_history_lookup_total_bytes,
                lh_d = label_history_lookup_data_bytes,
                lh_p = label_history_lookup_proof_bytes,
                lh_t = label_history_lookup_total_bytes,
                slot_b = slot_bytes.len(),
                l_w = label_wire_bytes,
                v_w = value_wire_bytes,
                h_w = history_wire_bytes,
                lh_w = label_history_wire_bytes,
            ));

            if sample_idx == 0 || (sample_idx + 1) % 10 == 0 {
                eprintln!(
                    "  sample {}: server[lbl/val/vh/lh]={:.2}/{:.2}/{:.2}/{:.2}ms \
                     client[lbl/val/vh/lh]={:.2}/{:.2}/{:.2}/{:.2}ms \
                     totals[lbl/val/vh/lh]={}/{}/{}/{}B (vh_entries={})",
                    sample_idx,
                    server_label_ns as f64 / 1e6,
                    server_value_ns as f64 / 1e6,
                    server_history_ns as f64 / 1e6,
                    server_label_history_ns as f64 / 1e6,
                    client_label_ns as f64 / 1e6,
                    client_value_ns as f64 / 1e6,
                    client_history_ns as f64 / 1e6,
                    client_label_history_ns as f64 / 1e6,
                    label_lookup_total_bytes,
                    value_lookup_total_bytes,
                    value_history_lookup_total_bytes,
                    label_history_lookup_total_bytes,
                    history_entries,
                );
            }
        }

        // ---- per-stage audit bench --------------------------------
        // Run `verify_sharded_invariance` on consecutive epoch
        // transitions starting from a fresh `AuditState` at epoch 0.
        // The auditor's per-epoch cost is `O(n_shards × 2)` group
        // equations (no PCS openings, no proofs to ship — the only
        // bytes pulled per audit step are the next epoch's
        // `ShardedEpochCommitment`).
        //
        // Runs BEFORE the publish-bench so the audit chain length
        // reflects the preload's target fill exactly, not
        // `target + publish_bench_overhead`. The preload climb has
        // already produced plenty of transitions to sample.
        //
        // The whole audit is verifier-side work — the "server" cost is
        // just the `epoch_commitment(epoch)` clone (cached in
        // `epoch_commits`). We record that fetch time separately as
        // `server_fetch_ns` so the auditor + bulletin-board sides are
        // both visible.
        let mut audit_json: Option<String> = None;
        if args.audit_samples > 0 {
            let current_epoch = shared.blocking_read().current_commitment().epoch;
            if current_epoch < 1 {
                eprintln!(
                    "  audit: skipped — only epoch 0 available (no transitions to audit)"
                );
            } else {
                let max_audit = args.audit_samples.min(current_epoch as usize);
                let verifier_ctx = shared.blocking_read().sharded_verifier_context();
                let mut audit_state =
                    AuditState::<<Bn254 as Pairing>::ScalarField>::default();
                let prev0 = match shared.blocking_read().epoch_commitment(0) {
                    Some(c) => c,
                    None => {
                        eprintln!("error: epoch 0 commitment missing");
                        return ExitCode::from(1);
                    },
                };
                let mut prev_commit = prev0;
                let mut audit_blocks: Vec<String> = Vec::with_capacity(max_audit);
                for i in 0..max_audit {
                    let next_epoch = i as u64 + 1;
                    // Time the server-side commit fetch (cached clone)
                    // and the auditor-side verify separately. Auditors
                    // pull `next` from a bulletin board; the bench
                    // approximates that with the local `epoch_commitment`
                    // accessor (a clone of a cached `Vec` entry).
                    let t_fetch = Instant::now();
                    let next_commit = match shared.blocking_read().epoch_commitment(next_epoch) {
                        Some(c) => c,
                        None => {
                            eprintln!(
                                "error: epoch_commitment({next_epoch}) missing during audit"
                            );
                            return ExitCode::from(1);
                        },
                    };
                    let server_fetch_ns = t_fetch.elapsed().as_nanos() as u64;

                    let t_audit = Instant::now();
                    let ok = verify_sharded_invariance::<Bn254, Pcs>(
                        &verifier_ctx,
                        &mut audit_state,
                        &prev_commit,
                        &next_commit,
                    );
                    let audit_ns = t_audit.elapsed().as_nanos() as u64;
                    match ok {
                        Ok(true) => {},
                        Ok(false) => {
                            eprintln!(
                                "error: verify_sharded_invariance returned false at \
                                 epoch transition {i} -> {next_epoch}"
                            );
                            return ExitCode::from(1);
                        },
                        Err(e) => {
                            eprintln!(
                                "error: verify_sharded_invariance failed at epoch transition \
                                 {i} -> {next_epoch}: {e}"
                            );
                            return ExitCode::from(1);
                        },
                    }
                    let audit_proof_bytes = next_commit.uncompressed_size() as u64;
                    audit_blocks.push(format!(
                        concat!(
                            "          {{\n",
                            "            \"sample_idx\": {idx},\n",
                            "            \"prev_epoch\": {pe},\n",
                            "            \"next_epoch\": {ne},\n",
                            "            \"server_fetch_ns\": {fetch_ns},\n",
                            "            \"audit_invariance_ns\": {audit_ns},\n",
                            "            \"audit_proof_bytes\": {bytes}\n",
                            "          }}"
                        ),
                        idx = i,
                        pe = i,
                        ne = next_epoch,
                        fetch_ns = server_fetch_ns,
                        audit_ns = audit_ns,
                        bytes = audit_proof_bytes,
                    ));
                    if i == 0 || (i + 1) % 5 == 0 {
                        eprintln!(
                            "  audit sample {i}: epoch {i}->{next_epoch} verify={:.3}ms \
                             fetch={:.3}ms proof_bytes={audit_proof_bytes}",
                            audit_ns as f64 / 1e6,
                            server_fetch_ns as f64 / 1e6,
                        );
                    }
                    prev_commit = next_commit;
                }
                audit_json = Some(format!(
                    "        \"samples\": [\n{samples}\n        ]",
                    samples = audit_blocks.join(",\n"),
                ));
            }
        }

        // CHECKPOINT after audit, before the (potentially destructive)
        // throughput sweep. The sweep can stress shards / coord / the
        // masking queue at high concurrency — if it triggers an OOM
        // or cascade failure we don't want to lose this level's
        // already-paid lookup + audit samples. The flush is cheap
        // (single JSON write) so we do it unconditionally; sweeps and
        // publish_bench will overwrite the same file with richer data
        // at the end of the level.
        {
            let lookup_block_so_far = format!(
                "      \"lookup\": {{\n        \"sample_count\": {sc},\n        \"samples\": [\n{samples}\n        ]\n      }}",
                sc = samples_json.len(),
                samples = samples_json.join(",\n"),
            );
            let audit_block_so_far = match &audit_json {
                Some(b) => format!(",\n      \"audit\": {{\n{b}\n      }}"),
                None => String::new(),
            };
            let partial_level_block = format!(
                "    {{\n      \"preload_count\": {target},\n      \"current_count_after_topup\": {current_count},\n      \"preload_publish_ms_total\": {preload_publish_ms_total:.4},\n      \"rss_kb\": {rss},\n{lookup_block}{audit_block}\n    }}",
                target = target,
                current_count = current_count,
                preload_publish_ms_total = preload_publish_ms_total,
                rss = rss_kb,
                lookup_block = lookup_block_so_far,
                audit_block = audit_block_so_far,
            );
            let mut tmp_reports = level_reports.clone();
            tmp_reports.push(partial_level_block);
            let json = render_levels_json(
                &args,
                &preload_counts,
                local_mode,
                effective_n_shards,
                log_n_shards,
                k,
                setup_ms,
                initial_prefill_ms,
                &tmp_reports,
            );
            match File::create(&args.output)
                .and_then(|mut f| f.write_all(json.as_bytes()))
            {
                Ok(()) => eprintln!(
                    "[lookup-bench] CHECKPOINT: persisted level {}/{} lookup+audit ({})",
                    level_idx + 1,
                    preload_counts.len(),
                    args.output.display(),
                ),
                Err(e) => eprintln!(
                    "[lookup-bench] WARN: checkpoint write to {:?} failed: {e}",
                    args.output
                ),
            }
        }

        // ---- per-stage throughput sweep (optional) ---------------
        // For each concurrency N in `--throughput-concurrencies`,
        // spawn N tokio tasks that pound the coord gRPC endpoint
        // with the chosen lookup kind, then record achieved QPS +
        // latency quantiles. Runs after audit and BEFORE
        // publish_bench so the cluster state is exactly what audit
        // measured — the sweep itself doesn't add epochs.
        //
        // Notes on design:
        //   - The "value" kind does the realistic two-RPC client
        //     flow: lookup_label → lookup_value with the resolved
        //     slot. The full chain is timed so the per-request
        //     latency reflects what a real client experiences.
        //   - Other kinds are single-RPC.
        //   - Each task uses an independent linear-congruential
        //     sequence over [0, current_count), so they cover the
        //     label space without coordinating.
        //   - The hot loop continues running through warmup; only
        //     latencies recorded after `measuring = true` count.
        let mut concurrency_sweep_json: Option<String> = None;
        if !args.throughput_concurrencies.is_empty() && current_count > 0 {
            use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
            let kind = args.throughput_lookup_kind.clone();
            let valid_kinds = ["label", "value", "history", "label_history"];
            if !valid_kinds.contains(&kind.as_str()) {
                eprintln!(
                    "error: invalid --throughput-lookup-kind {kind:?} (expected one of {valid_kinds:?})"
                );
                return ExitCode::from(2);
            }

            let rpc_timeout = Duration::from_secs(args.throughput_rpc_timeout_secs.max(1));
            let mut sweep_blocks: Vec<String> = Vec::with_capacity(args.throughput_concurrencies.len());
            let mut aborted = false;
            for &concurrency in &args.throughput_concurrencies {
                if concurrency == 0 {
                    continue;
                }

                // Health probe before each new concurrency level: a
                // single sequential lookup with a tight timeout. If
                // it fails, the cluster is in trouble — bail the
                // sweep so we don't make it worse.
                if args.throughput_health_check_timeout_secs > 0 {
                    let probe_timeout = Duration::from_secs(args.throughput_health_check_timeout_secs);
                    let probe_label = phone_label(0);
                    let mut probe_client = raw_client.clone();
                    let probe_ok = driver_rt.block_on(async {
                        tokio::time::timeout(
                            probe_timeout,
                            probe_client.lookup_label(LookupLabelRequest { label: probe_label }),
                        )
                        .await
                        .map(|r| r.is_ok())
                        .unwrap_or(false)
                    });
                    if !probe_ok {
                        eprintln!(
                            "  throughput_sweep: HEALTH PROBE FAILED before concurrency={concurrency} — \
                             aborting sweep at this fill level to protect the cluster"
                        );
                        aborted = true;
                        break;
                    }
                }

                eprintln!(
                    "  throughput_sweep: kind={kind} concurrency={concurrency} window={}s warmup={}s rpc_timeout={}s",
                    args.throughput_window_secs, args.throughput_warmup_secs, rpc_timeout.as_secs(),
                );

                let stop = Arc::new(AtomicBool::new(false));
                let measuring = Arc::new(AtomicBool::new(false));
                let err_count = Arc::new(AtomicU64::new(0));
                let attempt_count = Arc::new(AtomicU64::new(0));

                let (all_latencies, elapsed_s) = driver_rt.block_on(async {
                    let mut handles: Vec<tokio::task::JoinHandle<Vec<f64>>> =
                        Vec::with_capacity(concurrency);
                    for task_id in 0..concurrency {
                        let mut client = raw_client.clone();
                        let stop = Arc::clone(&stop);
                        let measuring = Arc::clone(&measuring);
                        let err_count = Arc::clone(&err_count);
                        let attempt_count = Arc::clone(&attempt_count);
                        let kind = kind.clone();
                        let current_count_u64 = current_count as u64;
                        let h = tokio::spawn(async move {
                            let mut lat: Vec<f64> = Vec::new();
                            // Per-task LCG seed so tasks pick
                            // disjoint-ish label sequences.
                            let mut seq: u64 = (task_id as u64)
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                ^ 0xDEAD_BEEF_CAFE_F00D;
                            while !stop.load(Ordering::Relaxed) {
                                seq = seq
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                                let idx = seq % current_count_u64.max(1);
                                let label = phone_label(idx);
                                attempt_count.fetch_add(1, Ordering::Relaxed);
                                // Latency is read from the response's
                                // `server_processing_micros` field, not
                                // a client-side wall-clock measurement.
                                // This isolates server-induced latency
                                // (incl. queue-wait inside the handler
                                // under load) from network RTT and
                                // client-side decode overhead.
                                let server_micros: Option<u64> = match kind.as_str() {
                                    "label" => {
                                        let req = LookupLabelRequest { label };
                                        match tokio::time::timeout(rpc_timeout, client.lookup_label(req)).await {
                                            Ok(Ok(resp)) => Some(resp.into_inner().server_processing_micros),
                                            _ => None,
                                        }
                                    },
                                    "value" => {
                                        // Realistic two-step: label
                                        // probe, then value at slot.
                                        // Both legs inside the same
                                        // timeout budget. Sum the two
                                        // server-side processing
                                        // intervals to get the total
                                        // server work for the chained
                                        // operation.
                                        let chained = async {
                                            let lbl_req = LookupLabelRequest { label: label.clone() };
                                            let lbl_resp = client.lookup_label(lbl_req).await?;
                                            let lbl_inner = lbl_resp.into_inner();
                                            let slot = lbl_inner.slot;
                                            let val_req = LookupValueRequest { slot };
                                            let val_resp = client.lookup_value(val_req).await?;
                                            let val_inner = val_resp.into_inner();
                                            Ok::<u64, tonic::Status>(
                                                lbl_inner.server_processing_micros
                                                    + val_inner.server_processing_micros,
                                            )
                                        };
                                        match tokio::time::timeout(rpc_timeout, chained).await {
                                            Ok(Ok(us)) => Some(us),
                                            _ => None,
                                        }
                                    },
                                    "history" => {
                                        let req = LookupHistoryRequest { label };
                                        match tokio::time::timeout(rpc_timeout, client.lookup_history(req)).await {
                                            Ok(Ok(resp)) => Some(resp.into_inner().server_processing_micros),
                                            _ => None,
                                        }
                                    },
                                    "label_history" => {
                                        let req = LookupLabelHistoryRequest { label };
                                        match tokio::time::timeout(rpc_timeout, client.lookup_label_history(req)).await {
                                            Ok(Ok(resp)) => Some(resp.into_inner().server_processing_micros),
                                            _ => None,
                                        }
                                    },
                                    _ => unreachable!(),
                                };
                                let Some(us) = server_micros else {
                                    err_count.fetch_add(1, Ordering::Relaxed);
                                    continue;
                                };
                                let dur_ms = us as f64 / 1000.0;
                                if measuring.load(Ordering::Relaxed) {
                                    lat.push(dur_ms);
                                }
                            }
                            lat
                        });
                        handles.push(h);
                    }

                    // Warmup
                    tokio::time::sleep(Duration::from_secs(args.throughput_warmup_secs)).await;
                    // Measure
                    measuring.store(true, Ordering::Relaxed);
                    let start = Instant::now();
                    tokio::time::sleep(Duration::from_secs(args.throughput_window_secs)).await;
                    measuring.store(false, Ordering::Relaxed);
                    let elapsed = start.elapsed().as_secs_f64();
                    // Stop
                    stop.store(true, Ordering::Relaxed);
                    let mut all = Vec::new();
                    for h in handles {
                        if let Ok(l) = h.await {
                            all.extend(l);
                        }
                    }
                    (all, elapsed)
                });

                let n_requests = all_latencies.len() as u64;
                let mut sorted = all_latencies.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let q = |frac: f64| -> f64 {
                    if sorted.is_empty() {
                        return f64::NAN;
                    }
                    let idx =
                        ((sorted.len() - 1) as f64 * frac).round() as usize;
                    sorted[idx.min(sorted.len() - 1)]
                };
                let p50 = q(0.50);
                let p90 = q(0.90);
                let p99 = q(0.99);
                let qps = n_requests as f64 / elapsed_s.max(1e-9);
                let errs = err_count.load(Ordering::Relaxed);
                let attempts = attempt_count.load(Ordering::Relaxed);
                let err_rate = if attempts > 0 { errs as f64 / attempts as f64 } else { 0.0 };
                eprintln!(
                    "    qps={qps:.0}  n={n_requests}  attempts={attempts}  errs={errs} ({err_pct:.1}%)  p50={p50:.1}ms  p90={p90:.1}ms  p99={p99:.1}ms",
                    err_pct = err_rate * 100.0,
                );
                sweep_blocks.push(format!(
                    concat!(
                        "          {{\n",
                        "            \"concurrency\": {c},\n",
                        "            \"n_requests\": {n},\n",
                        "            \"errors\": {errs},\n",
                        "            \"window_s\": {ws:.3},\n",
                        "            \"qps\": {qps:.3},\n",
                        "            \"latency_ms_p50\": {p50:.3},\n",
                        "            \"latency_ms_p90\": {p90:.3},\n",
                        "            \"latency_ms_p99\": {p99:.3}\n",
                        "          }}"
                    ),
                    c = concurrency,
                    n = n_requests,
                    errs = errs,
                    ws = elapsed_s,
                    qps = qps,
                    p50 = p50,
                    p90 = p90,
                    p99 = p99,
                ));

                // Early-stop: bail before climbing concurrency
                // further if either signal says we're in trouble.
                if args.throughput_max_error_rate > 0.0 && err_rate >= args.throughput_max_error_rate {
                    eprintln!(
                        "  throughput_sweep: error rate {err_pct:.1}% ≥ threshold {thr:.1}% — \
                         stopping sweep at this fill level",
                        err_pct = err_rate * 100.0,
                        thr = args.throughput_max_error_rate * 100.0,
                    );
                    aborted = true;
                    break;
                }
                if args.throughput_max_p99_ms > 0 && p99 >= args.throughput_max_p99_ms as f64 {
                    eprintln!(
                        "  throughput_sweep: p99 latency {p99:.0}ms ≥ threshold {thr}ms — \
                         stopping sweep at this fill level",
                        p99 = p99,
                        thr = args.throughput_max_p99_ms,
                    );
                    aborted = true;
                    break;
                }
            }

            if aborted {
                eprintln!(
                    "  throughput_sweep: aborted at this fill level (cluster will continue to next level)"
                );
            }

            if !sweep_blocks.is_empty() {
                concurrency_sweep_json = Some(format!(
                    concat!(
                        "        \"lookup_kind\": \"{kind}\",\n",
                        "        \"samples\": [\n{samples}\n        ]"
                    ),
                    kind = args.throughput_lookup_kind,
                    samples = sweep_blocks.join(",\n"),
                ));
            }
        }

        // ---- per-stage publish bench (optional) -------------------
        // After lookups + audit, sweep `--publish-batch-sizes`
        // (each batch run `--publish-samples-per-batch` times) and
        // record per-publish wall time + commit-class byte sizes.
        // Runs LAST in the stage because each sample adds an epoch
        // to the chain — running audit first means the audit sees
        // the preload's target fill exactly, not
        // `target + publish_bench_overhead`. Disjoint namespace from
        // the lookup-sampleable space so the bench doesn't
        // accidentally turn its own publish-samples into sample
        // candidates for the next stage.
        let mut publish_bench_json: Option<String> = None;
        if publish_enabled {
            let mut batch_blocks: Vec<String> = Vec::with_capacity(args.publish_batch_sizes.len());
            for &batch_size in &args.publish_batch_sizes {
                eprintln!(
                    "  publish_bench: batch={batch_size} samples={}",
                    args.publish_samples_per_batch
                );
                let mut samples_ms: Vec<f64> = Vec::with_capacity(args.publish_samples_per_batch);
                // Commit-class sizes are invariant in batch_size for a
                // given n_shards; we still record them per batch so a
                // buggy invariant shows up in the JSON.
                let mut commit_sizes: Option<(u64, u64, u64, u64, u64)> = None;
                for sample_idx in 0..args.publish_samples_per_batch {
                    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..batch_size as u64)
                        .map(|_| {
                            let idx = publish_bench_idx;
                            publish_bench_idx += 1;
                            (phone_label(idx), rsa_value(idx))
                        })
                        .collect();
                    let t = Instant::now();
                    let res = {
                        let mut s = shared.blocking_write();
                        s.publish(&updates)
                    };
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    samples_ms.push(ms);
                    let commit = match res {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!(
                                "error: publish_bench publish failed (level={target}, \
                                 batch={batch_size}, sample={sample_idx}): {e}"
                            );
                            return ExitCode::from(1);
                        },
                    };
                    if commit_sizes.is_none() {
                        let (mut ic, mut vc, mut ric, mut rvc) = (0u64, 0u64, 0u64, 0u64);
                        for shard_commit in &commit.per_shard {
                            ic += shard_commit.index_commitment.uncompressed_size() as u64;
                            vc += shard_commit.value_commitment.uncompressed_size() as u64;
                            ric += shard_commit.rand_index_commitment.uncompressed_size() as u64;
                            rvc += shard_commit.rand_value_commitment.uncompressed_size() as u64;
                        }
                        let total = commit.uncompressed_size() as u64;
                        commit_sizes = Some((ic, vc, ric, rvc, total));
                    }
                }
                let mut sorted = samples_ms.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let fastest = sorted[0];
                let slowest = sorted[sorted.len() - 1];
                let median = sorted[sorted.len() / 2];
                let mean = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
                let (ic, vc, ric, rvc, total_commit) =
                    commit_sizes.expect("commit_sizes set in publish_bench loop");
                let samples_str = samples_ms
                    .iter()
                    .map(|t| format!("{t:.4}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                batch_blocks.push(format!(
                    concat!(
                        "          {{\n",
                        "            \"batch_size\": {bs},\n",
                        "            \"samples_ms\": [{samples_str}],\n",
                        "            \"fastest_ms\": {fastest:.4},\n",
                        "            \"slowest_ms\": {slowest:.4},\n",
                        "            \"median_ms\": {median:.4},\n",
                        "            \"mean_ms\": {mean:.4},\n",
                        "            \"index_commitment_bytes\": {ic},\n",
                        "            \"value_commitment_bytes\": {vc},\n",
                        "            \"rand_index_commitment_bytes\": {ric},\n",
                        "            \"rand_value_commitment_bytes\": {rvc},\n",
                        "            \"total_commit_bytes\": {tot}\n",
                        "          }}"
                    ),
                    bs = batch_size,
                    samples_str = samples_str,
                    fastest = fastest,
                    slowest = slowest,
                    median = median,
                    mean = mean,
                    ic = ic,
                    vc = vc,
                    ric = ric,
                    rvc = rvc,
                    tot = total_commit,
                ));
            }
            publish_bench_json = Some(format!(
                "        \"batches\": [\n{batches}\n        ]",
                batches = batch_blocks.join(",\n"),
            ));
        }

        // Per-stage JSON block. `lookup.samples` is empty when
        // current_count==0; `publish_bench` is absent when
        // --publish-batch-sizes was not provided.
        let lookup_block = format!(
            "      \"lookup\": {{\n        \"sample_count\": {sc},\n        \"samples\": [\n{samples}\n        ]\n      }}",
            sc = samples_json.len(),
            samples = samples_json.join(",\n"),
        );
        let publish_block = match publish_bench_json {
            Some(b) => format!(",\n      \"publish_bench\": {{\n{b}\n      }}"),
            None => String::new(),
        };
        let audit_block = match audit_json {
            Some(b) => format!(",\n      \"audit\": {{\n{b}\n      }}"),
            None => String::new(),
        };
        let concurrency_sweep_block = match concurrency_sweep_json {
            Some(b) => format!(",\n      \"concurrency_sweep\": {{\n{b}\n      }}"),
            None => String::new(),
        };
        let level_block = format!(
            "    {{\n      \"preload_count\": {target},\n      \"current_count_after_topup\": {current_count},\n      \"preload_publish_ms_total\": {preload_publish_ms_total:.4},\n      \"rss_kb\": {rss},\n{lookup_block}{publish_block}{audit_block}{concurrency_sweep_block}\n    }}",
            target = target,
            current_count = current_count,
            preload_publish_ms_total = preload_publish_ms_total,
            rss = rss_kb,
            lookup_block = lookup_block,
            publish_block = publish_block,
            audit_block = audit_block,
            concurrency_sweep_block = concurrency_sweep_block,
        );
        level_reports.push(level_block);

        // Incremental JSON flush after EVERY completed stage. Cheap
        // (the JSON is rendered from in-memory state), and means a
        // mid-run crash at a later stage doesn't destroy the data
        // we've already paid for. Errors here are non-fatal; the next
        // stage's flush (or the final write below) retries.
        let json = render_levels_json(
            &args,
            &preload_counts,
            local_mode,
            effective_n_shards,
            log_n_shards,
            k,
            setup_ms,
            initial_prefill_ms,
            &level_reports,
        );
        match File::create(&args.output)
            .and_then(|mut f| f.write_all(json.as_bytes()))
        {
            Ok(()) => eprintln!(
                "[lookup-bench] flushed {} ({} level(s) so far)",
                args.output.display(),
                level_reports.len()
            ),
            Err(e) => eprintln!(
                "[lookup-bench] WARN: incremental write to {:?} failed: {e}",
                args.output
            ),
        }
    }

    // ---- final write (also serves as the success exit signal) -------
    let json = render_levels_json(
        &args,
        &preload_counts,
        local_mode,
        effective_n_shards,
        log_n_shards,
        k,
        setup_ms,
        initial_prefill_ms,
        &level_reports,
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

/// Render the full bench JSON for whatever level_reports we currently
/// have. Called both from inside the per-stage loop (for the
/// incremental crash-survival flush) and once more after the loop
/// exits. Pulled out into a free function rather than a closure so
/// it can borrow `args` immutably while the loop body still
/// owns mutable state elsewhere.
fn render_levels_json(
    args: &Args,
    preload_counts: &[usize],
    local_mode: bool,
    effective_n_shards: usize,
    log_n_shards: usize,
    k: usize,
    setup_ms: f64,
    initial_prefill_ms: f64,
    level_reports: &[String],
) -> String {
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
    let fill_percents_json = args
        .fill_percents
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let publish_batch_sizes_json = args
        .publish_batch_sizes
        .iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let true_log_capacity_json = match args.true_log_capacity {
        Some(v) => v.to_string(),
        None => "null".to_string(),
    };
    format!(
        concat!(
            "{{\n",
            "  \"params\": {{\n",
            "    \"mode\": \"{mode}\",\n",
            "    \"n_shards\": {n_shards},\n",
            "    \"log_n_shards\": {log_n_shards},\n",
            "    \"shard_log_capacity\": {shard_log_capacity},\n",
            "    \"true_log_capacity\": {true_log_capacity},\n",
            "    \"kzh_k\": {kzh_k},\n",
            "    \"private\": {private},\n",
            "    \"samples_per_level\": {samples},\n",
            "    \"publish_batch_size\": {publish_batch_size},\n",
            "    \"publish_batch_sizes\": [{publish_batch_sizes}],\n",
            "    \"publish_samples_per_batch\": {publish_samples_per_batch},\n",
            "    \"audit_samples\": {audit_samples},\n",
            "    \"initial_prefill_count\": {initial_prefill_count},\n",
            "    \"prefill_seed\": {prefill_seed},\n",
            "    \"endpoints\": [{endpoints_json}],\n",
            "    \"preload_counts\": [{preload_json}],\n",
            "    \"fill_percents\": [{fill_percents_json}]\n",
            "  }},\n",
            "  \"setup_ms\": {setup_ms:.4},\n",
            "  \"initial_prefill_ms\": {initial_prefill_ms:.4},\n",
            "  \"levels\": [\n{levels}\n  ]\n",
            "}}\n"
        ),
        mode = if local_mode { "local" } else { "remote" },
        n_shards = effective_n_shards,
        log_n_shards = log_n_shards,
        shard_log_capacity = args.shard_log_capacity,
        true_log_capacity = true_log_capacity_json,
        kzh_k = k,
        private = args.private,
        samples = args.samples_per_level,
        publish_batch_size = args.publish_batch_size,
        publish_batch_sizes = publish_batch_sizes_json,
        publish_samples_per_batch = args.publish_samples_per_batch,
        audit_samples = args.audit_samples,
        initial_prefill_count = args.initial_prefill_count,
        prefill_seed = args.prefill_seed,
        endpoints_json = endpoints_json,
        preload_json = preload_json,
        fill_percents_json = fill_percents_json,
        setup_ms = setup_ms,
        initial_prefill_ms = initial_prefill_ms,
        levels = level_reports.join(",\n"),
    )
}
