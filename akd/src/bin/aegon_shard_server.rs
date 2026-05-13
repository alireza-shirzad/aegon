//! `aegon_shard_server` — one gRPC shard for the sharded Aegon
//! cluster. Operators deploy one per shard machine; the coordinator
//! connects to N of them over the network.
//!
//! See the workspace README for the full deployment workflow
//! (generate SRS once, distribute, run one server per machine, point
//! a coordinator at all of them).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use akd::aegon::shard_grpc::{ShardServer, ShardServerTlsConfig};
use akd::aegon::sharded::read_srs_from_file;
use akd::aegon::{AegonConfig, Sha256Hash};
use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;
use std::marker::PhantomData;

type Pcs = KZHK<Bn254>;
type Aegon = akd::aegon::Aegon<Bn254, Pcs, Sha256Hash>;

/// One shard of a sharded-Aegon deployment.
#[derive(Debug, Parser)]
#[command(
    name = "aegon_shard_server",
    about = "gRPC server for one Aegon shard. Deploy one per machine; the coordinator connects to N of them."
)]
struct Args {
    /// Socket address to bind on (e.g. `0.0.0.0:50051`).
    #[arg(long)]
    bind: SocketAddr,

    /// log_2 of the number of slots this shard's polynomial holds.
    /// Must match the value the coordinator uses on its side.
    #[arg(long)]
    shard_log_capacity: usize,

    /// KZH-k block parameter. Operator should set this to the
    /// `optimal_kzh_k(shard_log_capacity)` value (k=10 for
    /// shard_log_capacity=29).
    #[arg(long)]
    kzh_k: usize,

    /// Path to the serialized SRS (output of
    /// `ShardedAegonConfig::generate_srs_to_file`). Mutually exclusive
    /// with `--setup-seed`.
    #[arg(long, conflicts_with = "setup_seed")]
    srs_path: Option<PathBuf>,

    /// Generate SRS in-process from a deterministic seed. Test-mode
    /// only — never use in production.
    #[arg(long)]
    setup_seed: Option<u64>,

    /// Enable zero-knowledge mode (KZH-k `zk = true`).
    #[arg(long)]
    private: bool,

    /// Path to a PEM-encoded server certificate for TLS. Requires
    /// `--tls-key`. If unset, the server runs in plaintext HTTP/2.
    #[arg(long, requires = "tls_key")]
    tls_cert: Option<PathBuf>,

    /// Path to the PEM-encoded private key matching `--tls-cert`.
    #[arg(long, requires = "tls_cert")]
    tls_key: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();

    if args.srs_path.is_none() && args.setup_seed.is_none() {
        eprintln!("error: provide either --srs-path or --setup-seed");
        return ExitCode::from(2);
    }

    let aegon_cfg = AegonConfig::<Bn254, Pcs> {
        log_capacity: args.shard_log_capacity,
        private: args.private,
        pcs_config: KZHKConfig::new(args.kzh_k, args.private),
        _e: PhantomData,
    };

    let aegon = match (&args.srs_path, args.setup_seed) {
        (Some(path), _) => {
            eprintln!("loading SRS from {}", path.display());
            let (pk, vk) = match read_srs_from_file::<Bn254, Pcs>(path) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error loading SRS: {e}");
                    return ExitCode::from(1);
                },
            };
            match Aegon::init(pk, vk, &aegon_cfg) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("error initializing Aegon: {e}");
                    return ExitCode::from(1);
                },
            }
        },
        (None, Some(seed)) => {
            eprintln!("WARNING: generating SRS in-process from seed {seed} (test mode only)");
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            match Aegon::setup(&mut rng, &aegon_cfg) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(1);
                },
            }
        },
        (None, None) => unreachable!("checked above"),
    };

    let tls_config = match (&args.tls_cert, &args.tls_key) {
        (Some(cert), Some(key)) => match ShardServerTlsConfig::from_pem_files(cert, key) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("error loading TLS material: {e}");
                return ExitCode::from(1);
            },
        },
        _ => None,
    };

    eprintln!(
        "aegon_shard_server listening on {}{} (shard_log_capacity={}, kzh_k={}, private={})",
        args.bind,
        if tls_config.is_some() { " [TLS]" } else { "" },
        args.shard_log_capacity,
        args.kzh_k,
        args.private,
    );

    let server = ShardServer::<Bn254, Pcs, Sha256Hash>::new(aegon);
    let result = if let Some(tls) = tls_config {
        server.serve_with_tls(args.bind, tls).await
    } else {
        server.serve(args.bind).await
    };
    if let Err(e) = result {
        eprintln!("server exited with error: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
