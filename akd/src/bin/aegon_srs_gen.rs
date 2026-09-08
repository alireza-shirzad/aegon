// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! `aegon_srs_gen` — one-shot SRS generator. Run **once** on a setup
//! machine; the resulting file is the input to `--srs-path` on every
//! shard server and on the coordinator.
//!
//! Usage:
//!   aegon_srs_gen \
//!     --shard-log-capacity 29 \
//!     --kzh-k 10 \
//!     --out /tmp/shard.srs \
//!     --seed 42
//!
//! For real production this binary is a placeholder for "load the
//! trusted-setup ceremony output" — the on-wire shape of the SRS
//! file is the same either way, so swapping the underlying source
//! doesn't change the rest of the deployment.

use std::path::PathBuf;
use std::process::ExitCode;

use akd::aegon::ShardedAegonConfig;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;

#[derive(Debug, Parser)]
#[command(
    name = "aegon_srs_gen",
    about = "Generate the shared SRS file used by every shard + the coordinator. Run ONCE on a setup machine, then distribute the output to all participants."
)]
struct Args {
    /// log_2 of slots in one shard's polynomial. Every shard in the
    /// cluster shares this value; the SRS is sized to it.
    #[arg(long)]
    shard_log_capacity: usize,

    /// KZH-k block parameter. Use `optimal_kzh_k(shard_log_capacity)`
    /// (k=10 for shard_log_capacity=29) unless you have a specific
    /// reason to override.
    #[arg(long)]
    kzh_k: usize,

    /// Number of shards in the eventual cluster. The SRS itself
    /// doesn't depend on this — it's per-shard — but log_n_shards
    /// is part of the `ShardedAegonConfig` so we capture it for
    /// documentation and to keep the builder happy.
    #[arg(long, default_value = "0")]
    log_n_shards: usize,

    /// Output path for the serialized SRS file.
    #[arg(long)]
    out: PathBuf,

    /// RNG seed for `gen_srs_for_testing`. This is **not** a trusted
    /// setup; the seed is here so two operators with the same seed
    /// produce byte-identical SRS files (useful during testing and
    /// for reproducibility). For real deployments, replace this
    /// binary with a loader that pulls the ceremony output.
    #[arg(long)]
    seed: u64,

    /// Enable zero-knowledge mode (KZH-k `zk = true`).
    #[arg(long)]
    private: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let cfg = match ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(args.shard_log_capacity)
        .log_n_shards(args.log_n_shards)
        .private(args.private)
        .kzh_k(args.kzh_k)
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: config invalid: {e}");
            return ExitCode::from(2);
        }
    };

    eprintln!(
        "generating SRS: shard_log_capacity={}, kzh_k={}, private={}, seed={}",
        args.shard_log_capacity, args.kzh_k, args.private, args.seed,
    );
    let t0 = std::time::Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    if let Err(e) = cfg.generate_srs_to_file(&mut rng, &args.out) {
        eprintln!("error: SRS gen failed: {e}");
        return ExitCode::from(1);
    }
    let dt = t0.elapsed();

    let size = std::fs::metadata(&args.out).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "wrote {} ({} bytes) in {:.2} s",
        args.out.display(),
        size,
        dt.as_secs_f64()
    );
    eprintln!();
    eprintln!("Next steps:");
    eprintln!(
        "  1. Distribute the file: gsutil cp {} gs://your-bucket/",
        args.out.display()
    );
    eprintln!(
        "  2. On each shard: aegon_shard_server --srs-path {} ...",
        args.out.display()
    );
    eprintln!(
        "  3. On the coordinator: configure ShardedAegonConfig with SrsSource::Path({:?})",
        args.out
    );
    ExitCode::SUCCESS
}
