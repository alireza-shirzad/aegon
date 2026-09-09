// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! In-process driver for diagnosing publish-time memory spikes.
//!
//! Builds a single-shard `ShardedAegon`, climbs to `--fill-count`
//! entries with `--climb-batch`-sized publishes, then sweeps a list of
//! small batch sizes with `--probe-samples` publishes each — the same
//! pattern `aegon_lookup_bench`'s `publish-bench` section does. Each
//! `publish_phase_1` / `publish_phase_2` call prints VmRSS at every
//! intermediate sub-step (see `aegon::instrument::log_rss`).
//!
//! Designed for log_cap ~ 22 so the whole thing fits in a few GB and
//! finishes in minutes — small enough to iterate fast, big enough that
//! the structural memory characteristics (aux table density, prev/new
//! state clones, allocator churn across many epochs) reproduce.

use std::process::ExitCode;
use std::time::Instant;

#[cfg(feature = "mimalloc_alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use akd::aegon::config::AegonConfig;
use akd::aegon::server::Aegon;
use akd::aegon::{optimal_kzh_k, EcVrfHash};
use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;
use std::marker::PhantomData;

type Pcs = akd_core::aegon_crypto::pcs::kzhk::KZHK<Bn254>;
type SingleShard = Aegon<Bn254, Pcs, EcVrfHash>;

#[derive(Parser, Debug)]
#[command(
    name = "aegon_rss_probe",
    about = "Drive Aegon publish in-process while logging VmRSS at every publish sub-step"
)]
struct Args {
    #[arg(long, default_value_t = 22)]
    shard_log_capacity: usize,

    /// KZH-k k parameter. 0 = use optimal_kzh_k(shard_log_capacity).
    #[arg(long, default_value_t = 0)]
    kzh_k: usize,

    /// Match the medium-cluster shard server: drop per-epoch snapshot
    /// rand polys + states after each publish, keep only the 4
    /// commitments. Set this to mirror the GCP shard config.
    #[arg(long)]
    no_retain_epoch_polys: bool,

    /// Total entries to publish during the climb.
    #[arg(long, default_value_t = 1_258_291)]
    fill_count: usize,

    /// Per-batch size during the climb phase.
    #[arg(long, default_value_t = 16384)]
    climb_batch: usize,

    /// Comma-separated batch sizes to sweep after the climb.
    #[arg(long, default_value = "64,128,256,512,1024,2048")]
    probe_batches: String,

    /// Number of publish samples per probe batch size.
    #[arg(long, default_value_t = 3)]
    probe_samples: usize,

    /// Setup seed for ChaCha20Rng. Affects SRS + per-batch label hashing.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> ExitCode {
    let args = Args::parse();
    akd::aegon::instrument::set_rss_log(true);
    akd::aegon::instrument::log_rss("startup");

    let kzh_k = if args.kzh_k == 0 {
        optimal_kzh_k(args.shard_log_capacity)
    } else {
        args.kzh_k
    };
    eprintln!(
        "rss_probe: log_cap={} kzh_k={} retain_epoch_polys={} fill_count={} climb_batch={} probe_batches={} probe_samples={}",
        args.shard_log_capacity,
        kzh_k,
        !args.no_retain_epoch_polys,
        args.fill_count,
        args.climb_batch,
        args.probe_batches,
        args.probe_samples,
    );

    let cfg = AegonConfig::<Bn254, Pcs> {
        log_capacity: args.shard_log_capacity,
        private: false,
        pcs_config: KZHKConfig::new(kzh_k, false),
        audit_fs: Default::default(),
        _e: PhantomData,
    };

    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    let t_setup = Instant::now();
    let mut server: SingleShard = match SingleShard::setup(&mut rng, &cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("setup failed: {e}");
            return ExitCode::from(1);
        }
    };
    if args.no_retain_epoch_polys {
        server.set_retain_epoch_polys(false);
    }
    eprintln!("setup OK in {:.2}s", t_setup.elapsed().as_secs_f64());
    akd::aegon::instrument::log_rss("post_setup");

    let probe_batches: Vec<usize> = args
        .probe_batches
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    if probe_batches.is_empty() {
        eprintln!("--probe-batches parsed empty");
        return ExitCode::from(2);
    }

    let mut published = 0usize;
    if args.fill_count > 0 {
        eprintln!(
            "--- climb start: target={} batch={} ---",
            args.fill_count, args.climb_batch
        );
        let t0 = Instant::now();
        while published < args.fill_count {
            let remaining = args.fill_count - published;
            let this_batch = remaining.min(args.climb_batch);
            let batch: Vec<(Vec<u8>, Vec<u8>)> = (0..this_batch)
                .map(|i| {
                    let idx = published + i;
                    (
                        format!("climb-{idx}").into_bytes(),
                        format!("v-{idx}").into_bytes(),
                    )
                })
                .collect();
            let t_pub = Instant::now();
            if let Err(e) = server.publish(&batch) {
                eprintln!("climb publish failed at published={published}: {e}");
                return ExitCode::from(1);
            }
            let dt = t_pub.elapsed().as_secs_f64();
            published += this_batch;
            eprintln!(
                "climb: published {this_batch} in {dt:.2}s (total {published}/{})",
                args.fill_count
            );
        }
        eprintln!("climb OK in {:.2}s", t0.elapsed().as_secs_f64());
    }
    akd::aegon::instrument::log_rss("post_climb");

    // Probe sweep: same pattern as publish-bench in aegon_lookup_bench.
    for &batch_size in &probe_batches {
        eprintln!(
            "--- probe batch={batch_size} samples={} ---",
            args.probe_samples
        );
        for sample in 0..args.probe_samples {
            let batch: Vec<(Vec<u8>, Vec<u8>)> = (0..batch_size)
                .map(|i| {
                    let idx = published + sample * batch_size + i;
                    (
                        format!("probe-{idx}").into_bytes(),
                        format!("v-{idx}").into_bytes(),
                    )
                })
                .collect();
            let t = Instant::now();
            if let Err(e) = server.publish(&batch) {
                eprintln!("probe publish failed (batch={batch_size}, sample={sample}): {e}");
                return ExitCode::from(1);
            }
            let dt = t.elapsed().as_secs_f64() * 1000.0;
            akd::aegon::instrument::log_rss_ctx(
                "probe.after_publish",
                &format!("batch={batch_size} sample={sample} ms={dt:.1}"),
            );
        }
        published += batch_size * args.probe_samples;
    }
    akd::aegon::instrument::log_rss("done");

    ExitCode::SUCCESS
}
