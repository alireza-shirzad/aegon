//! `aegon_masking_throughput_bench` — measure the steady-state
//! production rate of `generate_masking_package` for a given (num_vars,
//! kzh_k) config.
//!
//! Skips the gRPC server + queue entirely. Mirrors the producer-loop
//! pattern from `aegon::masking::MaskingServer::new` ([masking.rs:122])
//! so the per-package work is exactly what the production server pays.
//! A hot drainer keeps the bounded channel near-empty so producers
//! never block on back-pressure — the measured rate is therefore the
//! peak production rate the masking server can deliver.
//!
//! Reports aggregate pkg/s, per-producer pkg/s, and per-package time
//! quantiles. Intended to inform the cost model: at large scale, if
//! lookup QPS exceeds the masking server's production rate, the queue
//! drains and value-side lookups stall.

use std::process::ExitCode;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use akd::aegon::sharded::read_srs_from_file;
use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme;
use ark_bn254::Bn254;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;
use tokio::sync::mpsc;

type Pcs = KZHK<Bn254>;

#[derive(Debug, Parser)]
#[command(
    name = "aegon_masking_throughput_bench",
    about = "Measure peak masking-package production rate for a given (num_vars, kzh_k) config."
)]
struct Args {
    /// log_2 of the polynomial size. Must match the SRS.
    #[arg(long)]
    num_vars: usize,

    /// KZH-k block parameter. Must match the SRS.
    #[arg(long)]
    kzh_k: usize,

    /// Path to the serialized hiding SRS. Either this or
    /// `--setup-seed` is required.
    #[arg(long, conflicts_with = "setup_seed")]
    srs_path: Option<PathBuf>,

    /// Generate SRS in-process from this seed. Test-mode only.
    #[arg(long)]
    setup_seed: Option<u64>,

    /// Number of concurrent producer tasks. Mirrors the
    /// `--producers` flag of `aegon_masking_server`. Defaults to the
    /// number of logical CPUs.
    #[arg(long)]
    producers: Option<usize>,

    /// Bounded channel size. Should not affect throughput as long as
    /// it's larger than `producers` — the drainer keeps it near-empty.
    #[arg(long, default_value_t = 1024)]
    queue_size: usize,

    /// Warmup window before measurement starts. Lets producers reach
    /// steady-state (caches warm, allocator stable).
    #[arg(long, default_value_t = 5)]
    warmup_secs: u64,

    /// Measurement window in seconds.
    #[arg(long, default_value_t = 60)]
    duration_secs: u64,

    /// Emit a one-line JSON summary on stdout after the
    /// human-readable report. Lets a wrapper script aggregate runs.
    #[arg(long)]
    json: bool,
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();

    if args.srs_path.is_none() && args.setup_seed.is_none() {
        eprintln!("error: provide either --srs-path or --setup-seed");
        return ExitCode::from(2);
    }

    let pcs_config = KZHKConfig::new(args.kzh_k, /*zk =*/ true);

    eprintln!(
        "preparing SRS (num_vars={}, kzh_k={}, hiding=true)",
        args.num_vars, args.kzh_k
    );
    let srs_t0 = Instant::now();
    let prover_param = match (&args.srs_path, args.setup_seed) {
        (Some(path), _) => {
            eprintln!("  loading SRS from {}", path.display());
            match read_srs_from_file::<Bn254, Pcs>(path) {
                Ok((pk, _vk)) => pk,
                Err(e) => {
                    eprintln!("error loading SRS: {e}");
                    return ExitCode::from(1);
                },
            }
        },
        (None, Some(seed)) => {
            eprintln!("  generating SRS in-process from seed {seed}");
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let srs = match <Pcs as PolynomialCommitmentScheme<Bn254>>::gen_srs_for_testing(
                pcs_config,
                &mut rng,
                args.num_vars,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error generating SRS: {e:?}");
                    return ExitCode::from(1);
                },
            };
            match <Pcs as PolynomialCommitmentScheme<Bn254>>::trim(&srs, None, Some(args.num_vars)) {
                Ok((pk, _vk)) => pk,
                Err(e) => {
                    eprintln!("error trimming SRS: {e:?}");
                    return ExitCode::from(1);
                },
            }
        },
        (None, None) => unreachable!("checked above"),
    };
    eprintln!("  SRS ready in {:.1}s", srs_t0.elapsed().as_secs_f64());

    let prover_param = Arc::new(prover_param);
    let producers = args
        .producers
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let num_vars = args.num_vars;
    let kzh_k = args.kzh_k;

    eprintln!(
        "spawning {} producers, queue={}, warmup={}s, measurement={}s",
        producers, args.queue_size, args.warmup_secs, args.duration_secs
    );

    let (tx, mut rx) = mpsc::channel::<<Pcs as PolynomialCommitmentScheme<Bn254>>::MaskingPackage>(
        args.queue_size.max(1),
    );
    let stop = Arc::new(AtomicBool::new(false));
    let measuring = Arc::new(AtomicBool::new(false));

    // Producers — each records per-call durations + per-package
    // serialized sizes into local Vecs while `measuring` is true.
    // Size is computed as `uncompressed_size()` (matches what the
    // gRPC server would put on the wire — `encode()` in masking.rs
    // is `serialize_uncompressed` into a Vec of that exact size).
    type ProducerOut = (Vec<f64>, Vec<u64>);
    let mut producer_handles: Vec<tokio::task::JoinHandle<ProducerOut>> =
        Vec::with_capacity(producers);
    for _ in 0..producers {
        let pp = Arc::clone(&prover_param);
        let tx = tx.clone();
        let stop = Arc::clone(&stop);
        let measuring = Arc::clone(&measuring);
        let h = tokio::spawn(async move {
            let mut durs: Vec<f64> = Vec::new();
            let mut sizes: Vec<u64> = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let pp2 = Arc::clone(&pp);
                let t0 = Instant::now();
                let res = tokio::task::spawn_blocking(move || {
                    <Pcs as PolynomialCommitmentScheme<Bn254>>::generate_masking_package(
                        pp2.as_ref(),
                        num_vars,
                    )
                })
                .await;
                let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let pkg = match res {
                    Ok(Ok(pkg)) => pkg,
                    Ok(Err(e)) => {
                        eprintln!("producer: PCS error {e:?}");
                        continue;
                    },
                    Err(e) => {
                        eprintln!("producer: join error {e}");
                        continue;
                    },
                };
                if measuring.load(Ordering::Relaxed) {
                    durs.push(dur_ms);
                    sizes.push(pkg.uncompressed_size() as u64);
                }
                if tx.send(pkg).await.is_err() {
                    break;
                }
            }
            (durs, sizes)
        });
        producer_handles.push(h);
    }
    drop(tx);

    // Hot drainer — keeps the channel near-empty so producers never
    // block on back-pressure. Just drop the packages.
    let drainer = tokio::spawn(async move {
        let mut count: usize = 0;
        while rx.recv().await.is_some() {
            count += 1;
        }
        count
    });

    // Warmup
    eprintln!("warming up ({}s) ...", args.warmup_secs);
    tokio::time::sleep(Duration::from_secs(args.warmup_secs)).await;

    // Begin measurement
    eprintln!("measuring ({}s) ...", args.duration_secs);
    measuring.store(true, Ordering::Relaxed);
    let measure_start = Instant::now();
    tokio::time::sleep(Duration::from_secs(args.duration_secs)).await;
    measuring.store(false, Ordering::Relaxed);
    let measure_elapsed = measure_start.elapsed().as_secs_f64();

    // Stop producers and collect their durations + sizes.
    stop.store(true, Ordering::Relaxed);
    let mut all_durs: Vec<f64> = Vec::new();
    let mut all_sizes: Vec<u64> = Vec::new();
    for h in producer_handles {
        if let Ok((d, s)) = h.await {
            all_durs.extend(d);
            all_sizes.extend(s);
        }
    }
    // The drainer ends when the channel closes (all tx clones dropped
    // — which happens when the producers exit).
    let _drained_total = drainer.await.unwrap_or(0);

    all_durs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pkgs_in_window = all_durs.len();
    let aggregate = pkgs_in_window as f64 / measure_elapsed;
    let per_producer = if producers > 0 {
        aggregate / producers as f64
    } else {
        0.0
    };
    let p50 = percentile(&all_durs, 50.0);
    let p10 = percentile(&all_durs, 10.0);
    let p90 = percentile(&all_durs, 90.0);
    let mean: f64 = if pkgs_in_window > 0 {
        all_durs.iter().sum::<f64>() / pkgs_in_window as f64
    } else {
        0.0
    };

    // Package size + bandwidth. The serialized size is the
    // CanonicalSerialize::uncompressed_size() of the generated
    // package — i.e. the byte count the masking server would write
    // into a MaskingPackageResponse (see encode() in masking.rs).
    // gRPC framing/HTTP-2 headers add a small per-message envelope
    // not captured here.
    let (size_min, size_max, size_mean, total_bytes) = if all_sizes.is_empty() {
        (0u64, 0u64, 0.0f64, 0u64)
    } else {
        let mn = *all_sizes.iter().min().unwrap();
        let mx = *all_sizes.iter().max().unwrap();
        let sum: u64 = all_sizes.iter().sum();
        let mean = sum as f64 / all_sizes.len() as f64;
        (mn, mx, mean, sum)
    };
    let bytes_per_sec = total_bytes as f64 / measure_elapsed;

    println!();
    println!("=== masking-package production throughput ===");
    println!("config:           num_vars={num_vars}, kzh_k={kzh_k}, producers={producers}");
    println!("measurement:      {pkgs_in_window} packages in {measure_elapsed:.2}s");
    println!("aggregate:        {aggregate:.1} pkg/s");
    println!("per producer:     {per_producer:.2} pkg/s");
    println!(
        "per-pkg time:     median={p50:.1} ms, p10={p10:.1} ms, p90={p90:.1} ms, mean={mean:.1} ms"
    );
    println!(
        "per-pkg bytes:    mean={mean_kb:.1} KB, min={mn_kb:.1} KB, max={mx_kb:.1} KB",
        mean_kb = size_mean / 1024.0,
        mn_kb = size_min as f64 / 1024.0,
        mx_kb = size_max as f64 / 1024.0,
    );
    println!(
        "wire bandwidth:   {bw_mb:.1} MB/s  ({bw_gb:.2} GB/s) — per masking server",
        bw_mb = bytes_per_sec / (1024.0 * 1024.0),
        bw_gb = bytes_per_sec / (1024.0 * 1024.0 * 1024.0),
    );

    if args.json {
        println!(
            "{{\"num_vars\":{num_vars},\"kzh_k\":{kzh_k},\"producers\":{producers},\
             \"window_s\":{ws:.3},\"pkgs\":{pkgs_in_window},\"aggregate_pkg_s\":{agg:.3},\
             \"per_producer_pkg_s\":{pp:.3},\"per_pkg_ms_p50\":{p50:.3},\
             \"per_pkg_ms_p10\":{p10:.3},\"per_pkg_ms_p90\":{p90:.3},\
             \"per_pkg_ms_mean\":{mean:.3},\
             \"per_pkg_bytes_mean\":{sm:.1},\"per_pkg_bytes_min\":{smin},\
             \"per_pkg_bytes_max\":{smax},\"total_bytes\":{tb},\
             \"bytes_per_sec\":{bps:.1}}}",
            ws = measure_elapsed,
            agg = aggregate,
            pp = per_producer,
            sm = size_mean,
            smin = size_min,
            smax = size_max,
            tb = total_bytes,
            bps = bytes_per_sec,
        );
    }

    ExitCode::SUCCESS
}
