// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! `aegon_masking_server` — pre-builds opening-point-agnostic
//! `KZHKMaskingPackage`s and serves them over gRPC. One per cluster.
//!
//! Shards request packages on every value-side opening (lookup and
//! history lookup); this server lifts the Pedersen-MSM work out of
//! the lookup critical path by running it in the background and
//! handing pre-built packages out on demand.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use akd::aegon::masking::MaskingServer;
use akd::aegon::sharded::read_srs_from_file;
use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use akd_core::aegon_crypto::pcs::PolynomialCommitmentScheme;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;

/// Aegon ZK masking server.
#[derive(Debug, Parser)]
#[command(
    name = "aegon_masking_server",
    about = "gRPC server that pre-builds ZK masking packages for shard value-side openings."
)]
struct Args {
    /// Socket address to bind on (e.g. `0.0.0.0:50061`).
    #[arg(long)]
    bind: SocketAddr,

    /// log_2 of the polynomial size. Must match the shards' value-
    /// /rand_value-poly `num_vars`.
    #[arg(long)]
    num_vars: usize,

    /// KZH-k block parameter. Must match the shards.
    #[arg(long)]
    kzh_k: usize,

    /// Path to the serialized SRS. The SRS must be the hiding
    /// (`zk = true`) variant — opening packages are meaningless from
    /// a non-hiding SRS.
    #[arg(long, conflicts_with = "setup_seed")]
    srs_path: Option<PathBuf>,

    /// Generate SRS in-process from a deterministic seed. Test-mode
    /// only — never use in production. Hiding variant is forced.
    #[arg(long)]
    setup_seed: Option<u64>,

    /// Bounded queue size of pre-built packages. Each waiting package
    /// pins a Pedersen commitment (one G1 affine + a sparse `r` with
    /// `k * N^{1/k}` non-zero coefficients) in memory; size to match
    /// expected lookup burstiness rather than steady-state QPS.
    #[arg(long, default_value_t = 256)]
    queue_size: usize,

    /// Number of concurrent producer tasks. Each does its own
    /// Pedersen-MSM, so this sets the masking server's peak fill
    /// rate. Defaults to the number of logical CPUs.
    #[arg(long)]
    producers: Option<usize>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();

    if args.srs_path.is_none() && args.setup_seed.is_none() {
        eprintln!("error: provide either --srs-path or --setup-seed");
        return ExitCode::from(2);
    }

    let pcs_config = KZHKConfig::new(args.kzh_k, /*zk =*/ true);

    let prover_param = match (&args.srs_path, args.setup_seed) {
        (Some(path), _) => {
            eprintln!("loading SRS from {}", path.display());
            match read_srs_from_file::<Bn254, Pcs>(path) {
                Ok((pk, _vk)) => pk,
                Err(e) => {
                    eprintln!("error loading SRS: {e}");
                    return ExitCode::from(1);
                }
            }
        }
        (None, Some(seed)) => {
            eprintln!(
                "WARNING: generating hiding SRS in-process from seed {seed} (test mode only)"
            );
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
                }
            };
            match <Pcs as PolynomialCommitmentScheme<Bn254>>::trim(&srs, None, Some(args.num_vars))
            {
                Ok((pk, _vk)) => pk,
                Err(e) => {
                    eprintln!("error trimming SRS: {e:?}");
                    return ExitCode::from(1);
                }
            }
        }
        (None, None) => unreachable!("checked above"),
    };

    let producers = args.producers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });

    eprintln!(
        "aegon_masking_server listening on {} (num_vars={}, kzh_k={}, queue_size={}, producers={})",
        args.bind, args.num_vars, args.kzh_k, args.queue_size, producers,
    );

    let server = MaskingServer::<Bn254, Pcs>::new(
        Arc::new(prover_param),
        args.num_vars,
        args.queue_size,
        producers,
    );

    if let Err(e) = server.serve(args.bind).await {
        eprintln!("masking server exited with error: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
