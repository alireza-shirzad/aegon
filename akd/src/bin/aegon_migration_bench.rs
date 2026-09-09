// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! `aegon_migration_bench` — measure end-to-end migration time from
//! an empty Aegon dictionary up to a target fill, parameterised by a
//! sweep of `--chunk-sizes` K values.
//!
//! One invocation = one regime = many K's. The binary holds a single
//! `ShardedAegon` instance across the entire sweep: between K's it
//! calls `ShardedAegon::clear_dictionary`, which wipes every shard
//! back to its post-setup empty state without rebuilding the SRS,
//! tearing down the cluster, or removing the backing database.
//!
//! What "migration" means here: starting from epoch 0 (clean
//! dictionary), publish N users in chunks of K until the dictionary
//! reaches `--target-fill-percent` of `2^true_log_capacity`. Wall-clock
//! elapsed is recorded at every `--milestone-fill` so a single K
//! yields the full migration curve, not just an endpoint.
//!
//! Why a separate binary from `aegon_publish_bench`: that bench
//! answers "at fill = N%, how fast is a publish of batch size B?" —
//! steady-state latency for production traffic, after migration is
//! over. This bench answers "how long does the one-time migration
//! take?" — total cumulative time across the full climb. Different
//! questions, different best K, different JSON shape, so a clean
//! split keeps each plot honest.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use akd::aegon::{
    optimal_kzh_k, AegonError, DbSource, EcVrfHash, ShardTransport, ShardedAegon,
    ShardedAegonConfig, SrsSource, VrfProver,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_std::rand::{RngCore, SeedableRng};
use clap::Parser;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, EcVrfHash>;

/// Matches the per-user wire size that the publish + lookup benches
/// use. Same RSA-sized values, so migration throughput numbers are
/// directly comparable.
const RSA_VALUE_LEN: usize = 256;

fn phone_label(idx: u64) -> Vec<u8> {
    format!("+1{:010}", idx % 10_000_000_000).into_bytes()
}

fn rsa_value(idx: u64) -> Vec<u8> {
    let mut rng = ChaCha20Rng::seed_from_u64(0xCAFE_C0DE ^ idx);
    let mut v = vec![0u8; RSA_VALUE_LEN];
    rng.fill_bytes(&mut v);
    v
}

#[derive(Debug, Parser)]
#[command(
    name = "aegon_migration_bench",
    about = "Multi-K migration-time benchmark from epoch 0 to a target fill."
)]
struct Args {
    /// Per-shard polynomial log capacity (over-provisioned). E.g.,
    /// small=22, medium=28.
    #[arg(long)]
    shard_log_capacity: usize,

    /// log_2 of the TRUE dictionary capacity; the target fill is
    /// computed against this, not the over-provisioned shard size.
    #[arg(long)]
    true_log_capacity: usize,

    /// KZH-k block parameter. Defaults to `optimal_kzh_k(shard_log_capacity)`.
    #[arg(long)]
    kzh_k: Option<usize>,

    /// Number of shards. Defaults to 1 (in-process).
    #[arg(long, default_value_t = 1)]
    n_shards: usize,

    /// Remote shard endpoints (comma-separated). When set, switches
    /// to distributed mode against an already-running cluster.
    /// Multi-K sweeps fan out the `ClearDictionary` RPC to every
    /// shard between climbs.
    #[arg(long, value_delimiter = ',')]
    endpoints: Vec<String>,

    /// Target fill (% of true capacity) to climb to. The migration
    /// ends when `current_count >= target_fill_percent / 100 *
    /// 2^true_log_capacity`. Ignored in `--single-batch-per-k` mode.
    #[arg(long, default_value_t = 90)]
    target_fill_percent: u32,

    /// Skip the full climb: for each K, publish exactly one batch of
    /// K updates against the freshly-cleared empty dictionary and
    /// measure that publish's wall-clock. Picks the best K by
    /// throughput. Cheap relative to climbing to e.g. 90% fill (one
    /// publish per K instead of ~10⁶/K). The assumption is that K
    /// ordering doesn't flip much across fill levels — if you suspect
    /// it does for your regime, run with `--target-fill-percent` set
    /// to the fill range you actually care about instead.
    #[arg(long, default_value_t = false)]
    single_batch_per_k: bool,

    /// Comma-separated list of chunk sizes to sweep. Each value is a
    /// distinct climb (epoch 0 → target fill) with that constant
    /// chunk size, and the dictionary is cleared between K's.
    /// Constant within a climb (no adaptive sizing) so each run
    /// produces a single line on the migration plot.
    #[arg(long, value_delimiter = ',', required = true)]
    chunk_sizes: Vec<u64>,

    /// Output path template. Must contain the literal substring
    /// `{K}`, which is replaced with each chunk size before writing
    /// the per-K JSON. For example,
    /// `bench-results/migration/small_K{K}.json` writes
    /// `bench-results/migration/small_K1024.json`,
    /// `bench-results/migration/small_K4096.json`, etc.
    #[arg(long)]
    out_template: PathBuf,

    /// Fill percentages at which to record an elapsed-time
    /// milestone. The migration plot uses these to draw the climb
    /// curve. Each milestone <= target_fill_percent is honoured;
    /// milestones above the target are silently dropped.
    #[arg(long, value_delimiter = ',', default_values_t = vec![1u32, 5, 10, 30, 60, 90])]
    milestone_fills: Vec<u32>,

    /// SRS RNG seed. Reproducibility hook.
    #[arg(long, default_value_t = 42)]
    setup_seed: u64,

    /// User-data RNG seed base. Per-user `(label, value)` pairs are
    /// reproducible across runs at the same K.
    #[arg(long, default_value_t = 1)]
    prefill_seed: u64,

    /// Optional Redis URL for coordinator-side open-addressing
    /// occupancy checks. Mirrors `aegon_publish_bench`.
    #[arg(long, conflicts_with = "db_path")]
    db_url: Option<String>,

    /// Local RocksDB path for coord-side state. Mutually exclusive
    /// with `--db-url`.
    #[arg(long)]
    db_path: Option<PathBuf>,

    /// Enable hiding (`private=true`) mode. Must match the SRS the
    /// cluster was set up with.
    #[arg(
        long,
        default_value_t = true,
        num_args = 0..=1,
        default_missing_value = "true",
    )]
    private: bool,
}

/// One recorded crossing of a milestone fill during the climb.
struct MilestoneRecord {
    /// Configured milestone (e.g. 30 — meaning "30% of capacity").
    fill_percent: u32,
    /// Actual `current_count` at the time the milestone first
    /// triggered. Always `>= target_for(fill_percent)`; the small
    /// overshoot is the last chunk that crossed the boundary.
    users_migrated: u64,
    /// Wall-clock since the climb's t0, including all publish() calls
    /// since the per-K dictionary was empty. NOT including the
    /// SRS-generation / shard-setup phase before the very first
    /// climb, NOR the `clear_dictionary` time between climbs (those
    /// are reported separately).
    elapsed_ms: f64,
    /// Convenience: `users_migrated * 1000 / elapsed_ms`. Cumulative
    /// average throughput from epoch 0 to this milestone.
    throughput_users_per_sec: f64,
}

fn build_server(args: &Args, k: usize, log_n_shards: usize) -> Result<Sharded, AegonError> {
    let mut builder = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(args.shard_log_capacity)
        .log_n_shards(log_n_shards)
        .private(args.private)
        .kzh_k(k);
    if args.endpoints.is_empty() {
        builder = builder.shards(ShardTransport::InProcess);
    } else {
        builder = builder.shards(ShardTransport::Remote {
            endpoints: args.endpoints.clone(),
        });
    }
    builder = builder.srs(SrsSource::DangerouslyGenerate);
    if let Some(url) = &args.db_url {
        builder = builder.db(DbSource::Redis(url.clone()));
    } else if let Some(path) = &args.db_path {
        builder = builder.db(DbSource::Rocks(path.clone()));
    }
    let cfg = builder.build()?;
    let mut rng = ChaCha20Rng::seed_from_u64(args.setup_seed);
    Sharded::setup(&mut rng, &cfg)
}

fn render_output_path(template: &Path, k: u64) -> Result<PathBuf, String> {
    let s = template
        .to_str()
        .ok_or_else(|| format!("--out-template is not valid UTF-8: {template:?}"))?;
    if !s.contains("{K}") {
        return Err(format!("--out-template must contain '{{K}}', got: {s:?}"));
    }
    Ok(PathBuf::from(s.replace("{K}", &k.to_string())))
}

/// One K's worth of climb output, also collected for the trailing
/// summary line.
struct ClimbResult {
    chunk_size: u64,
    users_migrated: u64,
    elapsed_ms: f64,
    throughput: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_one_climb(
    server: &mut Sharded,
    args: &Args,
    k: usize,
    log_n_shards: usize,
    chunk_size: u64,
    target_count: u64,
    milestone_targets: &[(u32, u64)],
    out_path: &Path,
    setup_secs: f64,
) -> Result<ClimbResult, String> {
    // Label namespace dedicated to migration prefill. Disjoint from
    // any namespace the lookup/publish benches use, so a downstream
    // run on the same shard won't collide.
    let migration_namespace_base: u64 = 2_000_000_000_000_000_000;

    let climb_t0 = Instant::now();
    let mut current_count: u64 = 0;
    let mut next_milestone_idx: usize = 0;
    let mut milestone_records: Vec<MilestoneRecord> = Vec::with_capacity(milestone_targets.len());

    while current_count < target_count {
        let chunk = std::cmp::min(chunk_size, target_count - current_count);
        let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..chunk)
            .map(|i| {
                let idx = migration_namespace_base + current_count + i;
                (phone_label(idx), rsa_value(idx))
            })
            .collect();
        server.publish_two_layer(&updates).map_err(|e| {
            format!("publish_two_layer error at current_count={current_count} chunk={chunk}: {e}")
        })?;
        current_count += chunk;

        // Cross zero or more milestones with this chunk.
        while next_milestone_idx < milestone_targets.len()
            && current_count >= milestone_targets[next_milestone_idx].1
        {
            let (pct, _tgt) = milestone_targets[next_milestone_idx];
            let elapsed_ms = climb_t0.elapsed().as_secs_f64() * 1000.0;
            let throughput = if elapsed_ms > 0.0 {
                current_count as f64 * 1000.0 / elapsed_ms
            } else {
                f64::INFINITY
            };
            milestone_records.push(MilestoneRecord {
                fill_percent: pct,
                users_migrated: current_count,
                elapsed_ms,
                throughput_users_per_sec: throughput,
            });
            eprintln!(
                "[migration-bench K={chunk_size}] milestone fill={pct}% \
                 users={current_count} elapsed={:.1}s throughput={:.0} users/sec",
                elapsed_ms / 1000.0,
                throughput,
            );
            next_milestone_idx += 1;
        }
    }

    let total_elapsed_ms = climb_t0.elapsed().as_secs_f64() * 1000.0;
    let total_throughput = if total_elapsed_ms > 0.0 {
        current_count as f64 * 1000.0 / total_elapsed_ms
    } else {
        f64::INFINITY
    };
    eprintln!(
        "[migration-bench K={chunk_size}] complete: users={current_count} \
         elapsed={:.1}s throughput={:.0} users/sec",
        total_elapsed_ms / 1000.0,
        total_throughput,
    );

    let json = render_json(
        args,
        k,
        log_n_shards,
        chunk_size,
        target_count,
        setup_secs,
        &milestone_records,
        current_count,
        total_elapsed_ms,
        total_throughput,
    );
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir -p '{}': {e}", parent.display()))?;
        }
    }
    // Atomic write: a mid-write OOM-kill on the next K (or a
    // crash in this process for any other reason) would otherwise
    // leave a half-written file that downstream pick_best_k logic
    // can't parse. Write to a sibling temp file then rename — on
    // POSIX, rename(2) is atomic w.r.t. crashes.
    let tmp_path = {
        let mut p = out_path.to_path_buf().into_os_string();
        p.push(".tmp");
        std::path::PathBuf::from(p)
    };
    std::fs::write(&tmp_path, json.as_bytes())
        .map_err(|e| format!("write '{}': {e}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, out_path).map_err(|e| {
        format!(
            "rename '{}' -> '{}': {e}",
            tmp_path.display(),
            out_path.display()
        )
    })?;
    eprintln!(
        "[migration-bench K={chunk_size}] wrote {}",
        out_path.display()
    );

    Ok(ClimbResult {
        chunk_size,
        users_migrated: current_count,
        elapsed_ms: total_elapsed_ms,
        throughput: total_throughput,
    })
}

fn main() -> ExitCode {
    #[cfg(feature = "tracing_instrument")]
    akd::aegon::tracing_init::init_tree_subscriber();
    let args = Args::parse();

    // ---- validation ------------------------------------------------
    if args.chunk_sizes.is_empty() {
        eprintln!("error: --chunk-sizes must list at least one value");
        return ExitCode::from(2);
    }
    if args.chunk_sizes.contains(&0) {
        eprintln!("error: every value in --chunk-sizes must be > 0");
        return ExitCode::from(2);
    }
    if args.target_fill_percent == 0 || args.target_fill_percent > 100 {
        eprintln!("error: --target-fill-percent must be in (0, 100]");
        return ExitCode::from(2);
    }
    if args.true_log_capacity > args.shard_log_capacity + 6 {
        eprintln!(
            "error: true_log_capacity={} is implausibly larger than shard_log_capacity={}",
            args.true_log_capacity, args.shard_log_capacity
        );
        return ExitCode::from(2);
    }
    // local_mode iff no endpoints given. In-process supports
    // n_shards>1 by spinning up multiple in-process shards — useful
    // for local medium-regime smoke runs on a beefy box, even though
    // production-scale medium/large normally run distributed via
    // bench-cluster.sh.
    let local_mode = args.endpoints.is_empty();
    if !local_mode && args.endpoints.len() != args.n_shards {
        eprintln!(
            "error: --endpoints length ({}) must equal --n-shards ({})",
            args.endpoints.len(),
            args.n_shards
        );
        return ExitCode::from(2);
    }
    let log_n_shards = (args.n_shards as f64).log2() as usize;
    if (1usize << log_n_shards) != args.n_shards {
        eprintln!(
            "error: --n-shards ({}) must be a power of two",
            args.n_shards
        );
        return ExitCode::from(2);
    }
    let k = args
        .kzh_k
        .unwrap_or_else(|| optimal_kzh_k(args.shard_log_capacity));

    // Validate the template before doing any work — fast-fail.
    for &chunk_size in &args.chunk_sizes {
        if let Err(e) = render_output_path(&args.out_template, chunk_size) {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    }

    // ---- compute the climb plan ------------------------------------
    let true_capacity: u64 = 1u64 << args.true_log_capacity;
    // In single-batch mode, target_count + milestones are computed
    // per-K inside the loop (target_count = chunk_size). In climb
    // mode, both are shared across K's and computed once.
    let climb_target_count: u64 = if args.single_batch_per_k {
        0
    } else {
        ((true_capacity as u128) * (args.target_fill_percent as u128) / 100u128) as u64
    };
    let climb_milestone_targets: Vec<(u32, u64)> = if args.single_batch_per_k {
        Vec::new()
    } else {
        if climb_target_count == 0 {
            eprintln!(
                "error: target_count is 0 (target_fill_percent={} * 2^{} / 100); use a larger fill",
                args.target_fill_percent, args.true_log_capacity,
            );
            return ExitCode::from(2);
        }
        let mut milestones: Vec<u32> = args
            .milestone_fills
            .iter()
            .copied()
            .filter(|p| *p > 0 && *p <= args.target_fill_percent)
            .collect();
        milestones.sort();
        milestones.dedup();
        if milestones.is_empty() {
            eprintln!(
                "error: no milestone_fills survive the target_fill_percent={} filter",
                args.target_fill_percent
            );
            return ExitCode::from(2);
        }
        milestones
            .iter()
            .map(|p| {
                (
                    *p,
                    ((true_capacity as u128) * (*p as u128) / 100u128) as u64,
                )
            })
            .collect()
    };

    if args.single_batch_per_k {
        eprintln!(
            "[migration-bench] mode={} shard_log_capacity={} true_log_capacity={} kzh_k={} \
             n_shards={} sweep=single-batch-per-K chunk_sizes={:?}",
            if local_mode { "local" } else { "distributed" },
            args.shard_log_capacity,
            args.true_log_capacity,
            k,
            args.n_shards,
            args.chunk_sizes,
        );
    } else {
        eprintln!(
            "[migration-bench] mode={} shard_log_capacity={} true_log_capacity={} kzh_k={} \
             n_shards={} target_fill={}% target_count={} chunk_sizes={:?}",
            if local_mode { "local" } else { "distributed" },
            args.shard_log_capacity,
            args.true_log_capacity,
            k,
            args.n_shards,
            args.target_fill_percent,
            climb_target_count,
            args.chunk_sizes,
        );
    }

    // ---- setup -----------------------------------------------------
    let setup_t0 = Instant::now();
    let mut server = match build_server(&args, k, log_n_shards) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[migration-bench] setup error: {e}");
            return ExitCode::from(1);
        }
    };
    server.set_vrf_prover(VrfProver::from_env());
    let setup_secs = setup_t0.elapsed().as_secs_f64();
    eprintln!("[migration-bench] setup done in {setup_secs:.1}s");

    // ---- per-K climbs ---------------------------------------------
    let mut results: Vec<ClimbResult> = Vec::with_capacity(args.chunk_sizes.len());
    for (idx, &chunk_size) in args.chunk_sizes.iter().enumerate() {
        if idx > 0 {
            // Clear between K's: shards + coord-side caches + coord DB
            // are reset back to epoch-0 empty without touching the SRS
            // or transport. Cheap relative to a fresh setup at scale,
            // and lets every K start from the same baseline.
            let clear_t0 = Instant::now();
            if let Err(e) = server.clear_dictionary() {
                eprintln!("[migration-bench K={chunk_size}] clear_dictionary error: {e}");
                return ExitCode::from(1);
            }
            let clear_ms = clear_t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[migration-bench] cleared dictionary between K's in {clear_ms:.1} ms (next K={chunk_size})"
            );
        }
        let out_path = match render_output_path(&args.out_template, chunk_size) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[migration-bench] {e}");
                return ExitCode::from(1);
            }
        };
        // single-batch mode: each K publishes exactly chunk_size
        // users (one publish) to the cleared empty dictionary. The
        // "milestone" we emit is a single 100% marker, so the JSON
        // shape is unchanged but the climb is one batch.
        let (target_count, milestone_targets): (u64, Vec<(u32, u64)>) = if args.single_batch_per_k {
            (chunk_size, vec![(100, chunk_size)])
        } else {
            (climb_target_count, climb_milestone_targets.clone())
        };
        eprintln!("[migration-bench K={chunk_size}] starting climb to target_count={target_count}");
        let result = match run_one_climb(
            &mut server,
            &args,
            k,
            log_n_shards,
            chunk_size,
            target_count,
            &milestone_targets,
            &out_path,
            // setup_secs is recorded only once, on the first K — the
            // remaining K's amortize the same setup. Persist it on
            // every JSON for downstream tooling that reads any single
            // file in isolation; the value is the same across all.
            setup_secs,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[migration-bench K={chunk_size}] {e}");
                return ExitCode::from(1);
            }
        };
        results.push(result);
    }

    // ---- summary --------------------------------------------------
    eprintln!("[migration-bench] sweep complete:");
    for r in &results {
        eprintln!(
            "  K={:>8} users={:>12} elapsed={:>8.1}s throughput={:>10.0} users/sec",
            r.chunk_size,
            r.users_migrated,
            r.elapsed_ms / 1000.0,
            r.throughput,
        );
    }
    // Rank by throughput (users/sec) — correct in both modes. In
    // climb mode every K hits the same target_count, so throughput
    // and 1/elapsed give the same ordering; in single-batch mode
    // each K publishes its own K users, so elapsed scales with K
    // and the only meaningful ordering is throughput.
    if let Some(best) = results
        .iter()
        .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
    {
        eprintln!(
            "[migration-bench] best K={} (elapsed={:.1}s, throughput={:.0} users/sec)",
            best.chunk_size,
            best.elapsed_ms / 1000.0,
            best.throughput,
        );
    }
    ExitCode::SUCCESS
}

/// Hand-rolled JSON. Same shape conventions as the other Aegon
/// benches — no serde_json dependency. One JSON per K.
#[allow(clippy::too_many_arguments)]
fn render_json(
    args: &Args,
    k: usize,
    log_n_shards: usize,
    chunk_size: u64,
    target_count: u64,
    setup_secs: f64,
    milestones: &[MilestoneRecord],
    total_users: u64,
    total_elapsed_ms: f64,
    total_throughput: f64,
) -> String {
    let endpoints_json = args
        .endpoints
        .iter()
        .map(|e| format!("\"{e}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let mut milestones_json: Vec<String> = Vec::with_capacity(milestones.len());
    for m in milestones {
        milestones_json.push(format!(
            concat!(
                "    {{\n",
                "      \"fill_percent\": {pct},\n",
                "      \"users_migrated\": {users},\n",
                "      \"elapsed_ms\": {elapsed:.4},\n",
                "      \"throughput_users_per_sec\": {tp:.2}\n",
                "    }}"
            ),
            pct = m.fill_percent,
            users = m.users_migrated,
            elapsed = m.elapsed_ms,
            tp = m.throughput_users_per_sec,
        ));
    }
    format!(
        concat!(
            "{{\n",
            "  \"params\": {{\n",
            "    \"shard_log_capacity\": {slc},\n",
            "    \"true_log_capacity\": {tlc},\n",
            "    \"kzh_k\": {k},\n",
            "    \"n_shards\": {n_shards},\n",
            "    \"log_n_shards\": {log_n_shards},\n",
            "    \"target_fill_percent\": {target_fill},\n",
            "    \"target_count\": {target_count},\n",
            "    \"chunk_size\": {chunk_size},\n",
            "    \"setup_seed\": {seed},\n",
            "    \"prefill_seed\": {pseed},\n",
            "    \"private\": {private},\n",
            "    \"endpoints\": [{endpoints}]\n",
            "  }},\n",
            "  \"setup_secs\": {setup_secs:.4},\n",
            "  \"milestones\": [\n{milestones}\n  ],\n",
            "  \"total\": {{\n",
            "    \"users_migrated\": {total_users},\n",
            "    \"elapsed_ms\": {total_elapsed:.4},\n",
            "    \"throughput_users_per_sec\": {total_tp:.2}\n",
            "  }}\n",
            "}}\n"
        ),
        slc = args.shard_log_capacity,
        tlc = args.true_log_capacity,
        k = k,
        n_shards = args.n_shards,
        log_n_shards = log_n_shards,
        target_fill = args.target_fill_percent,
        target_count = target_count,
        chunk_size = chunk_size,
        seed = args.setup_seed,
        pseed = args.prefill_seed,
        private = args.private,
        endpoints = endpoints_json,
        setup_secs = setup_secs,
        milestones = milestones_json.join(",\n"),
        total_users = total_users,
        total_elapsed = total_elapsed_ms,
        total_tp = total_throughput,
    )
}
