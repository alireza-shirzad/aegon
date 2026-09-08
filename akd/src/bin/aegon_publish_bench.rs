// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! `aegon_publish_bench` — publish-time + commitment-size benchmark.
//!
//! Sweeps `(fill_percent × batch_size)` for one regime and writes a
//! single JSON record.
//!
//! ## Two operating modes
//!
//! * **Local in-process** (`--n-shards 1`, no `--endpoints`): builds a
//!   single-shard in-process `ShardedAegon`, walks every
//!   `--fill-percents` value in one process — prefilling the shard via
//!   `ShardedAegon::prefill_random_per_shard` between sweep stages.
//!   Used for the **small** (`shard_log_cap=22`, `true_log_cap=20`)
//!   and **medium** (`shard_log_cap=28`, `true_log_cap=26`) regimes.
//!
//! * **Distributed remote** (`--endpoints http://...,http://...`):
//!   connects to a running cluster. Cluster must be pre-loaded to the
//!   target fill percentage externally (via `bench-cluster.sh
//!   restart-shards` with the right per-shard prefill_count). The
//!   bench takes a single `--fill-percent` value as a metadata tag.
//!   Used for the **planetary** (`shard_log_cap=29`, `true_log_cap=32`,
//!   `n_shards=32`) regime.
//!
//! ## What's measured
//!
//! Per `(fill_percent, batch_size)`:
//!   * `samples_ms[]` — per-publish wall-clock, one entry per sample.
//!   * `fastest_ms` / `slowest_ms` / `median_ms` / `mean_ms`.
//!   * `commit_bytes` — uncompressed serialised size of the
//!     `ShardedEpochCommitment` produced. Broken into
//!     `index_commitment_bytes` / `value_commitment_bytes` /
//!     `rand_index_commitment_bytes` / `rand_value_commitment_bytes`
//!     (sum across all shards) plus `total_bytes` (full
//!     `ShardedEpochCommitment`). Each per-shard `P::Commitment` is a
//!     single G1 element (~64 B on BN254 uncompressed), so totals are
//!     `n_shards × 4 × ~64 B + merkle_root (32 B) + epoch (8 B)`. The
//!     numbers are invariant in batch_size — they're recorded per
//!     batch_size anyway so downstream analysis can verify the
//!     invariant.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use akd::aegon::{
    optimal_kzh_k, AegonError, DbSource, EcVrfHash, ShardTransport, ShardedAegon,
    ShardedAegonConfig, SrsSource, VrfProver,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::{RngCore, SeedableRng};
use clap::Parser;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, EcVrfHash>;

/// Same per-label / per-value bytes the coordinator + lookup benches
/// use, so the wire-size measurements line up across the three bench
/// binaries.
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
    name = "aegon_publish_bench",
    about = "Publish-time + commitment-size benchmark across (fill_percent x batch_size)."
)]
struct Args {
    /// Per-shard polynomial log capacity (over-provisioned). E.g.,
    /// small=22, medium=28, planetary-per-shard=29.
    #[arg(long)]
    shard_log_capacity: usize,

    /// log_2 of the TRUE total dictionary size — fill percentages are
    /// computed against this, not the over-provisioned shard size.
    /// E.g., small=20, medium=26, planetary=32.
    #[arg(long)]
    true_log_capacity: usize,

    /// KZH-k block parameter. Defaults to `optimal_kzh_k(shard_log_capacity)`.
    #[arg(long)]
    kzh_k: Option<usize>,

    /// Number of shards. Defaults to 1 (in-process local mode).
    /// `n_shards > 1` requires `--endpoints` and switches to remote
    /// (distributed) mode.
    #[arg(long, default_value_t = 1)]
    n_shards: usize,

    /// Remote shard endpoints (comma-separated). When set, switches
    /// to distributed mode — the cluster must already be prefilled
    /// to the target fill percentage (`--fill-percents` is then a
    /// metadata tag, not a directive).
    #[arg(long, value_delimiter = ',')]
    endpoints: Vec<String>,

    /// Fill percentages to bench. Stages are walked in sorted order;
    /// between stages we run *real* publishes (not the in-memory
    /// prefill_random shortcut) until the dictionary reaches the
    /// target count. That populates Redis open-addressing state,
    /// history openings, and the FS chain — so each measured
    /// publish runs against the same kind of state a production
    /// cluster would have. The cost of the warmup grows with the
    /// fill ceiling, so the default caps at 30%.
    #[arg(long, value_delimiter = ',', default_values_t = vec![0u32, 30])]
    fill_percents: Vec<u32>,

    /// Comma-separated batch sizes to sweep at every fill level.
    #[arg(long, value_delimiter = ',')]
    batch_sizes: Vec<usize>,

    /// Samples per batch_size. Default 3 — enough for a stable
    /// median at the regime scales we run.
    #[arg(long, default_value_t = 3)]
    samples_per_batch: usize,

    /// Batch size used by the inter-stage warmup publishes. Large
    /// batches amortise the per-publish protocol overhead so the
    /// warmup finishes in a sensible time at large fill levels.
    /// Pick a value at the high end of `--batch-sizes` for the
    /// regime (e.g. 2048 for small/medium, 131072 for large).
    ///
    /// In a two-phase bench workflow this is set by reading
    /// `bench-results/migration/{regime}_best_k.txt`, which the
    /// migration bench writes after sweeping K.
    #[arg(long, default_value_t = 16384)]
    warmup_batch_size: u64,

    /// Setup seed. Must match the seed used by the cluster's shards
    /// when running in distributed mode (so the SRS is identical).
    #[arg(long, default_value_t = 42)]
    setup_seed: u64,

    /// Prefill seed base. Kept on the CLI for back-compat with
    /// existing scripts; the bench now uses real publishes for
    /// warmup so this only flavours per-stage label namespaces.
    #[arg(long, default_value_t = 1)]
    prefill_seed: u64,

    /// Optional Redis URL for coordinator-side open-addressing
    /// occupancy checks. Same flag shape as
    /// `aegon_coordinator_bench`. Mutually exclusive with `--db-path`.
    #[arg(long, conflicts_with = "db_path")]
    db_url: Option<String>,

    /// Local RocksDB path for coord-side state. Mutually exclusive
    /// with `--db-url`.
    #[arg(long)]
    db_path: Option<PathBuf>,

    /// Enable hiding (`private=true`) mode. Must match the SRS the
    /// cluster was set up with. Defaults to true now that the
    /// benches are run against the production-style hiding SRS.
    /// Accepts: `--private`, `--private true`, `--private false`,
    /// or omit it to take the default.
    #[arg(
        long,
        default_value_t = true,
        num_args = 0..=1,
        default_missing_value = "true",
    )]
    private: bool,

    /// Output JSON path.
    #[arg(long)]
    out: PathBuf,
}

/// Per-(batch_size) record after a sample sweep. Aggregates into the
/// stage's `batches` map.
struct BatchRecord {
    batch_size: usize,
    samples_ms: Vec<f64>,
    fastest_ms: f64,
    slowest_ms: f64,
    median_ms: f64,
    mean_ms: f64,
    /// Sum across all shards of each commit-class's uncompressed
    /// serialised bytes. Each `P::Commitment` is one G1 element.
    index_commitment_bytes: u64,
    value_commitment_bytes: u64,
    rand_index_commitment_bytes: u64,
    rand_value_commitment_bytes: u64,
    /// Total `ShardedEpochCommitment` uncompressed size, including
    /// the Merkle root + epoch counter overhead.
    total_commit_bytes: u64,
}

/// Per-fill-percentage stage record.
struct StageRecord {
    fill_percent: u32,
    /// Effective number of entries prefilled. In local mode this is
    /// `round(true_capacity * pct / 100)`. In distributed mode it's
    /// just the same arithmetic — the cluster's actual fill must
    /// match externally.
    prefilled_count: u64,
    batches: Vec<BatchRecord>,
}

fn main() -> ExitCode {
    #[cfg(feature = "tracing_instrument")]
    akd::aegon::tracing_init::init_tree_subscriber();
    let args = Args::parse();

    if args.batch_sizes.is_empty() {
        eprintln!("error: --batch-sizes must list at least one value");
        return ExitCode::from(2);
    }
    if args.fill_percents.is_empty() {
        eprintln!("error: --fill-percents must list at least one value");
        return ExitCode::from(2);
    }
    if args.true_log_capacity > args.shard_log_capacity + 6 {
        // 6 = log_2(64). At more than 64x over-provisioning the
        // numbers stop being meaningful; bail before doing work.
        eprintln!(
            "error: true_log_capacity={} is implausibly larger than shard_log_capacity={}",
            args.true_log_capacity, args.shard_log_capacity
        );
        return ExitCode::from(2);
    }
    // local_mode iff no endpoints given. In-process supports
    // n_shards>1 by spinning up multiple in-process shards — useful
    // for local medium-regime smoke runs.
    let local_mode = args.endpoints.is_empty();
    if !local_mode && args.endpoints.len() != args.n_shards {
        eprintln!(
            "error: --endpoints length ({}) must equal --n-shards ({})",
            args.endpoints.len(),
            args.n_shards
        );
        return ExitCode::from(2);
    }

    let k = args
        .kzh_k
        .unwrap_or_else(|| optimal_kzh_k(args.shard_log_capacity));
    let log_n_shards = {
        let n = args.n_shards;
        if !n.is_power_of_two() {
            eprintln!("error: --n-shards must be a power of two (got {n})");
            return ExitCode::from(2);
        }
        n.trailing_zeros() as usize
    };

    eprintln!(
        "[publish-bench] mode={} shard_log_capacity={} true_log_capacity={} kzh_k={} n_shards={} samples_per_batch={}",
        if local_mode { "local" } else { "distributed" },
        args.shard_log_capacity,
        args.true_log_capacity,
        k,
        args.n_shards,
        args.samples_per_batch
    );

    // ---- build the ShardedAegon ----
    let mut server = match build_server(&args, k, log_n_shards) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[publish-bench] setup error: {e}");
            return ExitCode::from(1);
        }
    };
    server.set_vrf_prover(VrfProver::from_env());

    // ---- sweep ----
    // Stages are walked in sorted order. Between stages we run real
    // publishes (the same code path the measurement uses) until the
    // dict reaches the next target_count. Each real publish updates:
    //   * each shard's index/value/rand polynomials and their commits
    //   * the FS chain (r_index, r_value)
    //   * the coordinator's Redis open-addressing state
    //   * §6.4 history openings for every new label
    //
    // So when we then measure a publish batch onto the warmed dict,
    // it's running against the same state shape a production cluster
    // would have at that fill level — no shortcuts.
    let mut stages_to_run: Vec<u32> = args.fill_percents.clone();
    stages_to_run.sort();
    stages_to_run.dedup();

    let mut stage_records: Vec<StageRecord> = Vec::with_capacity(stages_to_run.len());
    // Tracks total entries inserted across all stages so far —
    // warmup at stage N=30 picks up where measurement at stage N=0
    // left off. Measurements add a small amount per sample (we
    // increment after each publish below); the warmup loop will
    // top up from current_count to target_count.
    let mut current_count: u64 = 0;
    // Disjoint label namespace for warmup vs measurement: warmup
    // uses indices in [1e18, 2e18), measurements use < 1e14.
    let warmup_namespace_base: u64 = 1_000_000_000_000_000_000;
    for &fill_pct in &stages_to_run {
        let true_capacity: u64 = 1u64 << args.true_log_capacity;
        let target_count: u64 = ((true_capacity as u128) * (fill_pct as u128) / 100u128) as u64;

        if current_count < target_count {
            let to_add = target_count - current_count;
            eprintln!(
                "[publish-bench] warmup: publishing {to_add} entries via real publish \
                 (batches of {}) to reach {fill_pct}% of 2^{} \
                 (current={current_count} -> target={target_count}) [{}]",
                args.warmup_batch_size,
                args.true_log_capacity,
                if local_mode { "local" } else { "distributed" },
            );
            let warmup_t0 = Instant::now();
            while current_count < target_count {
                let chunk = std::cmp::min(args.warmup_batch_size, target_count - current_count);
                let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..chunk)
                    .map(|i| {
                        let idx = warmup_namespace_base + current_count + i;
                        (phone_label(idx), rsa_value(idx))
                    })
                    .collect();
                match server.publish_two_layer(&updates) {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("[publish-bench] warmup publish error: {e}");
                        return ExitCode::from(1);
                    }
                }
                current_count += chunk;
                if current_count.is_multiple_of(args.warmup_batch_size * 10)
                    || current_count >= target_count
                {
                    let elapsed = warmup_t0.elapsed().as_secs_f64();
                    eprintln!(
                        "[publish-bench]   warmup progress: {current_count}/{target_count} ({:.1}% of target) in {:.1}s",
                        100.0 * current_count as f64 / target_count.max(1) as f64,
                        elapsed,
                    );
                }
            }
            let warmup_secs = warmup_t0.elapsed().as_secs_f64();
            eprintln!(
                "[publish-bench] warmup to {fill_pct}% done in {:.1}s",
                warmup_secs
            );
        } else if current_count > target_count {
            eprintln!(
                "[publish-bench] note: current_count={current_count} already past fill_pct={fill_pct}% target={target_count} \
                 — skipping warmup, but measurement is recorded at the actual current fill"
            );
        }

        let mut batches: Vec<BatchRecord> = Vec::with_capacity(args.batch_sizes.len());
        for &batch_size in &args.batch_sizes {
            eprintln!(
                "[publish-bench] fill={fill_pct}% batch={batch_size} samples={}",
                args.samples_per_batch
            );
            let mut samples_ms: Vec<f64> = Vec::with_capacity(args.samples_per_batch);
            // Commitment sizes are constant in batch_size for a
            // fixed n_shards, but we re-measure on every batch so a
            // bug in that invariant shows up clearly in the JSON.
            let mut commit_sizes: Option<(u64, u64, u64, u64, u64)> = None;
            for sample_idx in 0..args.samples_per_batch {
                // Disjoint namespace per (fill_pct, batch_size, sample, i)
                // so the open-addressing trail never collides.
                let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..batch_size as u64)
                    .map(|i| {
                        let idx = (fill_pct as u64) * 1_000_000_000_000
                            + (batch_size as u64) * 10_000_000
                            + (sample_idx as u64) * 100_000
                            + i;
                        (phone_label(idx), rsa_value(idx))
                    })
                    .collect();
                let t = Instant::now();
                let commit = match server.publish_two_layer(&updates) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!(
                            "[publish-bench] publish failed (fill={fill_pct}, batch={batch_size}, sample={sample_idx}): {e}"
                        );
                        return ExitCode::from(1);
                    }
                };
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                samples_ms.push(ms);
                // Each measurement publish adds batch_size entries to
                // the dict; track so the next stage's warmup math is
                // correct ("how much further from here to target").
                current_count += batch_size as u64;

                // Record commit sizes on the first sample of each
                // batch. Sum per-class across shards.
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
            let (ic, vc, ric, rvc, total) = commit_sizes.expect("commit_sizes set in loop");
            batches.push(BatchRecord {
                batch_size,
                samples_ms,
                fastest_ms: fastest,
                slowest_ms: slowest,
                median_ms: median,
                mean_ms: mean,
                index_commitment_bytes: ic,
                value_commitment_bytes: vc,
                rand_index_commitment_bytes: ric,
                rand_value_commitment_bytes: rvc,
                total_commit_bytes: total,
            });
        }
        stage_records.push(StageRecord {
            fill_percent: fill_pct,
            prefilled_count: target_count,
            batches,
        });

        // Incremental JSON flush: re-render and overwrite `--out` after
        // EVERY completed stage so a crash on the next stage doesn't
        // discard the work we just spent minutes/hours producing.
        // Cheap (the JSON is small and we re-render from in-memory
        // state). Errors here are non-fatal — we log them and keep
        // going, since the next stage's flush will retry the write.
        let json = render_json(&args, k, log_n_shards, &stage_records);
        match std::fs::write(&args.out, json.as_bytes()) {
            Ok(()) => eprintln!(
                "[publish-bench] flushed {} ({} stage(s) so far)",
                args.out.display(),
                stage_records.len()
            ),
            Err(e) => eprintln!(
                "[publish-bench] WARN: incremental write to '{}' failed: {e}",
                args.out.display()
            ),
        }
    }

    // ---- final emit (also serves as the success exit signal) ----
    let json = render_json(&args, k, log_n_shards, &stage_records);
    match std::fs::write(&args.out, json.as_bytes()) {
        Ok(()) => eprintln!("[publish-bench] wrote {}", args.out.display()),
        Err(e) => {
            eprintln!("[publish-bench] write '{}': {e}", args.out.display());
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}

/// Build the `ShardedAegon` server. Pure in-process for local mode;
/// remote-shards transport otherwise. The setup-seed path is used
/// for in-process gen so the bench is reproducible without needing
/// a pre-generated SRS file.
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
    // The setup-seed path is always available; --srs-path is not
    // exposed on this bench to keep the CLI tight. If a deployer ever
    // wants a ceremony-SRS run, swap `SrsSource::DangerouslyGenerate`
    // for `Path(path)` below — same shape.
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

/// Hand-rolled JSON. Hand-roll for the same reason every other Aegon
/// bench does — no serde_json in the mandatory deps.
fn render_json(args: &Args, k: usize, log_n_shards: usize, stages: &[StageRecord]) -> String {
    let endpoints_json = args
        .endpoints
        .iter()
        .map(|e| format!("\"{e}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let mut stages_json: Vec<String> = Vec::with_capacity(stages.len());
    for stage in stages {
        let mut batches_json: Vec<String> = Vec::with_capacity(stage.batches.len());
        for b in &stage.batches {
            let samples = b
                .samples_ms
                .iter()
                .map(|t| format!("{t:.4}"))
                .collect::<Vec<_>>()
                .join(", ");
            batches_json.push(format!(
                concat!(
                    "      \"{bs}\": {{\n",
                    "        \"samples_ms\": [{samples}],\n",
                    "        \"fastest_ms\": {fastest:.4},\n",
                    "        \"slowest_ms\": {slowest:.4},\n",
                    "        \"median_ms\": {median:.4},\n",
                    "        \"mean_ms\": {mean:.4},\n",
                    "        \"index_commitment_bytes\": {ic},\n",
                    "        \"value_commitment_bytes\": {vc},\n",
                    "        \"rand_index_commitment_bytes\": {ric},\n",
                    "        \"rand_value_commitment_bytes\": {rvc},\n",
                    "        \"total_commit_bytes\": {tot}\n",
                    "      }}"
                ),
                bs = b.batch_size,
                samples = samples,
                fastest = b.fastest_ms,
                slowest = b.slowest_ms,
                median = b.median_ms,
                mean = b.mean_ms,
                ic = b.index_commitment_bytes,
                vc = b.value_commitment_bytes,
                ric = b.rand_index_commitment_bytes,
                rvc = b.rand_value_commitment_bytes,
                tot = b.total_commit_bytes,
            ));
        }
        stages_json.push(format!(
            concat!(
                "    {{\n",
                "      \"fill_percent\": {pct},\n",
                "      \"prefilled_count\": {pcount},\n",
                "      \"batches\": {{\n{batches}\n      }}\n",
                "    }}"
            ),
            pct = stage.fill_percent,
            pcount = stage.prefilled_count,
            batches = batches_json.join(",\n"),
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
            "    \"samples_per_batch\": {samples},\n",
            "    \"setup_seed\": {seed},\n",
            "    \"prefill_seed\": {pseed},\n",
            "    \"endpoints\": [{endpoints}]\n",
            "  }},\n",
            "  \"stages\": [\n{stages}\n  ]\n",
            "}}\n"
        ),
        slc = args.shard_log_capacity,
        tlc = args.true_log_capacity,
        k = k,
        n_shards = args.n_shards,
        log_n_shards = log_n_shards,
        samples = args.samples_per_batch,
        seed = args.setup_seed,
        pseed = args.prefill_seed,
        endpoints = endpoints_json,
        stages = stages_json.join(",\n"),
    )
}
