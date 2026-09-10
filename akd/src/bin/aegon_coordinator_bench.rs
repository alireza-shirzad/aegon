// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! `aegon_coordinator_bench` — cluster-level publish-batch benchmark.
//!
//! Runs on the coordinator machine after every shard server is up
//! (and, for production-scale benches, after each shard was started
//! with `--prefill-count` so its polynomials are already populated).
//! Connects to all shards via gRPC, sweeps a list of batch sizes, and
//! measures wall-clock time per `ShardedAegon::publish` call.
//!
//! Setup cost (SRS load, handshake to every shard) is paid once and
//! reported separately. Each batch_size in `--batch-sizes` is run
//! `--samples-per-batch` times; samples within one batch_size use
//! disjoint label namespaces so they don't collide on the
//! open-addressing trail.
//!
//! Output is a JSON file with the per-sample times and the params
//! they were collected under — easy to plot, easy to diff across
//! commits.
//!
//! Usage (after `cluster.sh up && cluster.sh deploy`):
//!
//! ```text
//! aegon_coordinator_bench \
//!   --shard-log-capacity 29 --kzh-k 10 \
//!   --setup-seed 42 \
//!   --endpoints http://aegon-shard-0:50051,...,http://aegon-shard-31:50051 \
//!   --batch-sizes 10,100,1000,10000 \
//!   --samples-per-batch 5 \
//!   --output /tmp/aegon-bench.json
//! ```
//!
//! Note: `setup` here is not the SRS generation (each shard generated
//! its own from the seed) — it's the coordinator's `verifier_param`
//! derivation + one gRPC `current_commitment` round-trip per shard.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use akd::aegon::{
    DbSource, EcVrfHash, ShardTransport, ShardedAegon, ShardedAegonConfig, SrsSource, VrfProver,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, EcVrfHash>;

/// Realistic application sizing: labels are 12-byte ASCII phone
/// numbers (`+1` + 10 digits), values are 256-byte RSA-pubkey-sized
/// random buffers. The AKD doesn't care about content — these are
/// just bytes — but the wire-size / DB-footprint numbers reflect a
/// real deployment instead of the old `b{...}-u{i}` / `v-{i}`
/// placeholders.
const RSA_VALUE_LEN: usize = 256;

fn phone_label(idx: u64) -> Vec<u8> {
    format!("+1{:010}", idx % 10_000_000_000).into_bytes()
}

fn rsa_value(idx: u64) -> Vec<u8> {
    use ark_std::rand::RngCore;
    let mut rng = ChaCha20Rng::seed_from_u64(0xCAFE_C0DE ^ idx);
    let mut v = vec![0u8; RSA_VALUE_LEN];
    rng.fill_bytes(&mut v);
    v
}

#[derive(Debug, Parser)]
#[command(
    name = "aegon_coordinator_bench",
    about = "Sweep publish-batch sizes against a live shard cluster and emit JSON timings."
)]
struct Args {
    /// log_2 of slots in each shard's polynomial. Must match every
    /// `aegon_shard_server`'s `--shard-log-capacity`.
    #[arg(long)]
    shard_log_capacity: usize,

    /// KZH-k block parameter. Must match every shard.
    #[arg(long)]
    kzh_k: usize,

    /// Comma-separated shard endpoints. Length must be a power of two.
    #[arg(long, value_delimiter = ',')]
    endpoints: Vec<String>,

    /// Path to a serialized SRS (output of `aegon_srs_gen`). The
    /// coordinator only needs `verifier_param` from it — the prover
    /// side lives on the shard machines.
    #[arg(long, conflicts_with = "setup_seed")]
    srs_path: Option<PathBuf>,

    /// Deterministic in-process SRS gen. **Must match the seed every
    /// shard server used** (otherwise the prover and verifier params
    /// disagree). Mutually exclusive with `--srs-path`.
    #[arg(long)]
    setup_seed: Option<u64>,

    /// Enable zero-knowledge mode (must match every shard).
    #[arg(long)]
    private: bool,

    /// Comma-separated batch sizes to sweep, e.g. `10,100,1000,10000`.
    #[arg(long, value_delimiter = ',')]
    batch_sizes: Vec<usize>,

    /// How many timed publishes per batch_size. Default 5 — enough for
    /// a median to stabilise without dragging the wall time too far.
    #[arg(long, default_value_t = 5)]
    samples_per_batch: usize,

    /// Where to write the JSON timing report.
    #[arg(long)]
    output: PathBuf,

    /// Optional Redis URL for coordinator-side open-addressing. When
    /// set, occupancy checks during slot probing become `EXISTS` calls
    /// against `aegon:slot:{shard_id}:{slot_idx}` instead of gRPC
    /// round-trips to the owning shard — typically a 10–100x speedup
    /// on WAN clusters. Form: `redis://host[:port][/db]`. For results
    /// to be correct against a prefilled cluster, every shard must
    /// have been started with the matching `--db-url` so the prefill
    /// also populated these keys.
    #[arg(long, conflicts_with = "db_path")]
    db_url: Option<String>,

    /// Local RocksDB directory for coordinator-side state. Mutually
    /// exclusive with `--db-url`. When selected, slot-occupancy
    /// probes fall back to per-probe gRPC calls to the owning shard.
    #[arg(long)]
    db_path: Option<PathBuf>,
    /// Fiat-Shamir transcript for the audit path: `poseidon`
    /// (default) or `sha256`.
    ///
    /// Every node in a deployment must agree, and the choice is baked
    /// into each published epoch, so it cannot change on a live chain.
    #[arg(long, default_value_t = akd::aegon::audit_fs::AuditFs::Poseidon)]
    audit_fs: akd::aegon::audit_fs::AuditFs,
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
    if args.batch_sizes.is_empty() {
        eprintln!("error: --batch-sizes must list at least one value");
        return ExitCode::from(2);
    }
    let log_n_shards = args.endpoints.len().trailing_zeros() as usize;

    let mut builder = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .audit_fs(akd::aegon::ivc::adapter::hooks_for(args.audit_fs))
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
    // With `--db-url`, the coordinator uses Redis for open-addressing
    // occupancy checks (one `EXISTS` per probe) instead of gRPC
    // round-trips to the owning shard. On a WAN cluster this dominates
    // the wall-clock for any non-trivial batch, so toggling DB on is
    // typically how you get from "end-to-end network-bound timing" to
    // "what's actually crypto-bound". Without `--db-url` the bench
    // falls back to gRPC occupancy checks (same correctness).
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
        }
    };

    eprintln!(
        "connecting to {} shards (shard_log_capacity={}, kzh_k={}, log_n_shards={})",
        args.endpoints.len(),
        args.shard_log_capacity,
        args.kzh_k,
        log_n_shards
    );
    let mut rng = ChaCha20Rng::seed_from_u64(args.setup_seed.unwrap_or(0));
    let t_setup = Instant::now();
    let mut server = match Sharded::setup(&mut rng, &cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: setup failed: {e}");
            return ExitCode::from(1);
        }
    };
    server.set_vrf_prover(VrfProver::from_env());
    let setup_ms = t_setup.elapsed().as_secs_f64() * 1000.0;
    eprintln!("coordinator setup OK in {setup_ms:.1} ms (ECVRF prover attached)");

    // Sweep. Each sample uses a disjoint label namespace
    // (`b{batch_size}-s{sample_idx}-u{i}`) so labels from earlier
    // samples don't collide with later ones on the open-addressing
    // trail.
    let mut json_batches: Vec<String> = Vec::new();
    for &batch_size in &args.batch_sizes {
        eprintln!(
            "--- batch_size = {batch_size}, samples = {} ---",
            args.samples_per_batch
        );
        let mut samples_ms: Vec<f64> = Vec::with_capacity(args.samples_per_batch);
        for sample_idx in 0..args.samples_per_batch {
            // Per-(batch_size, sample_idx) disjoint namespace mapped
            // into the E.164 phone-number space. The arithmetic gives
            // each (batch_size, sample_idx) pair a 100K-entry window
            // — enough headroom for batch_size up to ~65K — and the
            // batch_size component spaces different sizes far apart
            // so a sweep never collides on the open-addressing trail.
            let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..batch_size as u64)
                .map(|i| {
                    let idx = (batch_size as u64) * 10_000_000 + (sample_idx as u64) * 100_000 + i;
                    (phone_label(idx), rsa_value(idx))
                })
                .collect();
            let t = Instant::now();
            match server.publish_two_layer(&updates) {
                Ok(_commit) => {}
                Err(e) => {
                    eprintln!("error: publish_two_layer failed (batch={batch_size}, sample={sample_idx}): {e}");
                    return ExitCode::from(1);
                }
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            samples_ms.push(ms);
            eprintln!("  sample {sample_idx}: {ms:.2} ms");
        }
        // Compute fastest/slowest/median/mean for the eyeball summary.
        let mut sorted = samples_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let fastest = sorted[0];
        let slowest = sorted[sorted.len() - 1];
        let median = sorted[sorted.len() / 2];
        let mean = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
        eprintln!(
            "  batch={batch_size}: fastest={fastest:.2} ms, median={median:.2} ms, slowest={slowest:.2} ms, mean={mean:.2} ms"
        );

        let samples_json = samples_ms
            .iter()
            .map(|t| format!("{t:.4}"))
            .collect::<Vec<_>>()
            .join(", ");
        json_batches.push(format!(
            "    \"{batch_size}\": {{\n      \"samples_ms\": [{samples_json}],\n      \"fastest_ms\": {fastest:.4},\n      \"slowest_ms\": {slowest:.4},\n      \"median_ms\": {median:.4},\n      \"mean_ms\": {mean:.4}\n    }}"
        ));
    }

    // Hand-rolled JSON to avoid pulling in serde_json just for this.
    // Schema mirrors the in-process `publish_sweep` divan output so
    // downstream plotters can treat them uniformly.
    let endpoints_json = args
        .endpoints
        .iter()
        .map(|e| format!("\"{e}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let json = format!(
        "{{\n  \"params\": {{\n    \"n_shards\": {n_shards},\n    \"log_n_shards\": {log_n_shards},\n    \"shard_log_capacity\": {shard_log_capacity},\n    \"kzh_k\": {kzh_k},\n    \"private\": {private},\n    \"samples_per_batch\": {samples},\n    \"endpoints\": [{endpoints_json}]\n  }},\n  \"setup_ms\": {setup_ms:.4},\n  \"publish_batch\": {{\n{batches}\n  }}\n}}\n",
        n_shards = args.endpoints.len(),
        log_n_shards = log_n_shards,
        shard_log_capacity = args.shard_log_capacity,
        kzh_k = args.kzh_k,
        private = args.private,
        samples = args.samples_per_batch,
        endpoints_json = endpoints_json,
        setup_ms = setup_ms,
        batches = json_batches.join(",\n"),
    );

    let mut f = match File::create(&args.output) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: cannot create output {:?}: {e}", args.output);
            return ExitCode::from(1);
        }
    };
    if let Err(e) = f.write_all(json.as_bytes()) {
        eprintln!("error: cannot write output: {e}");
        return ExitCode::from(1);
    }
    eprintln!("wrote {:?}", args.output);
    ExitCode::SUCCESS
}
