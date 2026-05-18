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

use akd::aegon::server::load_aegon_checkpoint_from_db;
use akd::aegon::shard_grpc::{ShardServer, ShardServerTlsConfig};
use akd::aegon::sharded::read_srs_from_file;
use akd::aegon::{AegonConfig, DbSource, Sha256Hash};
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

    /// **Currently a no-op** — reserved for the fault-tolerance
    /// feature. The shard's per-publish checkpoint writer is
    /// disabled in the current build (see
    /// `SHARD_CHECKPOINT_ENABLED` in `shard_grpc.rs`), so neither
    /// this URL nor `--db-path` are actually used during steady-
    /// state operation. The flag is kept on the CLI so existing
    /// deploy scripts don't have to change shape when fault
    /// tolerance lands; the read path (`load_aegon_checkpoint_from_db`)
    /// remains compiled so flipping the feature on is one constant
    /// flip + the bench-cluster bring-up.
    ///
    /// In the meantime: shards run with in-memory state only. A
    /// shard process restart loses its polynomial state and must be
    /// re-driven by the coord. Acceptable for bench/research, not
    /// for production.
    #[arg(long, requires = "shard_id", conflicts_with = "db_path")]
    db_url: Option<String>,

    /// **Currently a no-op** — same status as `--db-url`. See that
    /// flag's doc for the rationale. When fault tolerance is
    /// enabled this will point at a private per-shard RocksDB
    /// directory used solely for the shard's own checkpoint
    /// (coord never reads it). Until then, leave unset.
    #[arg(long, requires = "shard_id")]
    db_path: Option<std::path::PathBuf>,

    /// Identifier this shard registers under in the coordinator's
    /// routing namespace. Required with `--db-url`; otherwise unused
    /// but accepted for documentation. Set to the same `i` the
    /// coordinator uses for `endpoints[i]` (typically the position in
    /// the cluster's shard list, 0..N-1).
    #[arg(long)]
    shard_id: Option<u32>,

    /// **Benchmark-only**: after setup, locally populate this many
    /// random `(slot, h_label, h_value)` entries directly into the
    /// shard's polynomials (via `Aegon::prefill_random`) and recommit.
    /// Lets a benchmark cluster start at a "dictionary already has N
    /// users" state without paying for `N` round-trip coordinator
    /// publishes. **Do not use in production** — the FS-chain has no
    /// history covering these entries, so an auditor walking the chain
    /// could only validate from this state forward, not into it.
    #[arg(long)]
    prefill_count: Option<usize>,

    /// Seed for the deterministic RNG that drives `--prefill-count`'s
    /// slot/value generation. Default-zero gives a fixed prefill
    /// across runs; vary it to get a different fill pattern.
    #[arg(long, default_value_t = 0)]
    prefill_seed: u64,

    /// Optional address of the cluster's masking server (e.g.
    /// `http://10.0.0.5:50061`). When set, the shard fetches a one-
    /// shot `KZHKMaskingPackage` from the masking server before every
    /// value-side opening (lookup, freshness attestation, history
    /// re-mask) and produces a hiding opening via
    /// `P::open_zk_with_package` / `P::remask_with_package`. Leave
    /// unset for tests; value-side openings then fall back to
    /// generating a fresh masking package inline.
    #[arg(long)]
    masking_addr: Option<String>,
}

#[tokio::main]
async fn main() -> ExitCode {
    #[cfg(feature = "tracing_instrument")]
    akd::aegon::tracing_init::init_tree_subscriber();
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

    let db_source = match (&args.db_url, &args.db_path) {
        (Some(url), None) => DbSource::Redis(url.clone()),
        (None, Some(path)) => DbSource::Rocks(path.clone()),
        (None, None) => DbSource::None,
        (Some(_), Some(_)) => {
            eprintln!("error: --db-url and --db-path are mutually exclusive");
            return ExitCode::from(2);
        },
    };
    let shard_id = args.shard_id.unwrap_or(0);

    let mut aegon = match (&args.srs_path, args.setup_seed) {
        (Some(path), _) => {
            eprintln!("loading SRS from {}", path.display());
            let (pk, vk) = match read_srs_from_file::<Bn254, Pcs>(path) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error loading SRS: {e}");
                    return ExitCode::from(1);
                },
            };
            // Checkpoint-recovery path: if a previous incarnation of
            // this shard left a checkpoint behind, resume from it
            // instead of starting fresh at epoch 0.
            let recovered = match load_aegon_checkpoint_from_db::<Bn254, Pcs>(&db_source, shard_id) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error reading shard checkpoint: {e}");
                    return ExitCode::from(1);
                },
            };
            match recovered {
                Some(ckpt) => {
                    eprintln!(
                        "resuming shard {shard_id} from Redis checkpoint at epoch {}",
                        ckpt.epoch
                    );
                    match Aegon::restore_from_checkpoint(pk, vk, &aegon_cfg, ckpt) {
                        Ok(a) => a,
                        Err(e) => {
                            eprintln!("error restoring Aegon from checkpoint: {e}");
                            return ExitCode::from(1);
                        },
                    }
                },
                None => match Aegon::init(pk, vk, &aegon_cfg) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("error initializing Aegon: {e}");
                        return ExitCode::from(1);
                    },
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

    // Wire up the cluster's masking server (if any) so the shard's
    // value-side openings fetch one-shot ZK packages from it instead
    // of generating them inline. Done before prefill / serve so the
    // very first opening that hits the shard uses the masking
    // server.
    if let Some(addr) = &args.masking_addr {
        eprintln!("connecting to masking server at {addr}");
        match akd::aegon::masking::MaskingClient::<Bn254, Pcs>::connect(addr.clone()) {
            Ok(client) => aegon.set_masking_client(std::sync::Arc::new(client)),
            Err(e) => {
                eprintln!("error connecting to masking server '{addr}': {e}");
                return ExitCode::from(1);
            },
        }
    }

    // Benchmark-only prefill. Runs after setup but before binding the
    // gRPC socket — we want the polynomials populated by the time the
    // coordinator connects, but the prefill itself takes time at large
    // counts (each entry = 1 RNG draw + 1 HashMap insert, plus one
    // commit + update_state pass at the end).
    if let Some(count) = args.prefill_count {
        if count > 0 {
            eprintln!(
                "prefilling shard with {count} random entries (seed={})",
                args.prefill_seed
            );
            let mut prefill_rng = ChaCha20Rng::seed_from_u64(args.prefill_seed);
            if let Err(e) =
                aegon.prefill_random(&mut prefill_rng, count, &db_source, shard_id)
            {
                eprintln!("error prefilling shard: {e}");
                return ExitCode::from(1);
            }
        }
    }

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

    let has_db = !matches!(db_source, DbSource::None);
    let server = if has_db {
        match ShardServer::<Bn254, Pcs, Sha256Hash>::new_with_checkpoint(
            aegon,
            db_source.clone(),
            shard_id,
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error wiring shard checkpoint sink: {e}");
                return ExitCode::from(1);
            },
        }
    } else {
        ShardServer::<Bn254, Pcs, Sha256Hash>::new(aegon)
    };
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
