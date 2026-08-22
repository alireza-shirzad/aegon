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
use std::sync::Arc;

#[cfg(feature = "mimalloc_alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use akd::aegon::distributed_srs::{
    run_distributed_compute, try_cache_hit, Phase, SrsBootstrapConfig, SrsBootstrapState,
    SrsServer as SrsGrpcServer,
};
use akd::aegon::server::load_aegon_checkpoint_from_db;
use akd::aegon::shard_grpc::{ShardServer, ShardServerTlsConfig};
use akd::aegon::sharded::read_srs_from_file;
use akd::aegon::hash::VrfProver;
use akd::aegon::{AegonConfig, DbSource, EcVrfHash};
use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;
use std::marker::PhantomData;
use tonic::transport::Server;

type Pcs = KZHK<Bn254>;
type Aegon = akd::aegon::Aegon<Bn254, Pcs, EcVrfHash>;

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

    /// Comma-separated list of masking-server endpoints (e.g.
    /// `http://10.0.0.5:50061,http://10.0.0.6:50061`). When set, the
    /// shard fetches a one-shot `KZHKMaskingPackage` from one of these
    /// servers before every value-side opening (lookup, freshness
    /// attestation, history re-mask), produced via
    /// `P::open_zk_with_package` / `P::remask_with_package`. Multiple
    /// endpoints are dispatched round-robin via
    /// `MaskingClientPool` — useful when a single masking server is
    /// CPU-bound and only this one shard is driving load (e.g. the
    /// small/medium regimes where N_SHARDS ≤ 2). Leave unset for
    /// tests; value-side openings then fall back to generating a
    /// fresh masking package inline. May be repeated, or pass a
    /// single comma-separated string.
    #[arg(long = "masking-addr", value_delimiter = ',')]
    masking_addr: Vec<String>,

    /// Bind address for the distributed-SRS-bootstrap gRPC service
    /// (separate port from the main shard service). When set, the
    /// shard:
    ///
    ///   1. Binds `SrsService` on this address before doing any heavy
    ///      work,
    ///   2. Tries to load a cached SRS from `--srs-cache-dir` keyed on
    ///      `(shard_log_capacity, kzh_k, hash(setup_seed))`,
    ///   3. On cache miss, blocks until an external bootstrap actor
    ///      (`aegon_srs_bootstrap`) pushes trapdoors + peer endpoints
    ///      via `BootstrapSrs`, then runs the distributed compute /
    ///      slab exchange protocol with its peers,
    ///   4. Writes the assembled SRS to the cache and continues to
    ///      Aegon init + prefill + the main shard service.
    ///
    /// Requires `--setup-seed` (used both as the cache key and as a
    /// safety check against the seed pushed by the bootstrap actor) and
    /// `--shard-id`. Mutually exclusive with `--srs-path` (file-backed
    /// SRS path) — `--srs-bind` triggers the distributed-gen path
    /// exclusively.
    #[arg(long, requires = "setup_seed", requires = "shard_id", conflicts_with = "srs_path")]
    srs_bind: Option<SocketAddr>,

    /// Cache directory for the distributed-gen SRS path. Each shard
    /// writes its assembled SRS here so subsequent boots short-circuit
    /// the distributed exchange. Honoured only when `--srs-bind` is
    /// set. Default: `$HOME/.cache/aegon-srs`.
    #[arg(long)]
    srs_cache_dir: Option<PathBuf>,

    /// Skip retaining the rand polynomials + PCS state in
    /// `epoch_history` per publish. Saves ~50 MB / epoch on shards
    /// driving long publish chains (bench cluster) at the cost of
    /// breaking `consistency_proof(label, old_epoch)` — the four
    /// commitments per epoch ARE still kept, so
    /// `verify_sharded_invariance` (audits) and `epoch_commitment` keep
    /// working. Off by default; production deployments leave it off.
    #[arg(long)]
    no_retain_epoch_polys: bool,

    /// Emit `[rss] <stage>: <GiB>` lines on stdout at SRS-load, Aegon-
    /// init, and every publish phase-1 / phase-2 sub-step. Cheap to
    /// leave off (one relaxed atomic load per call site); useful for
    /// diagnosing publish-time memory spikes.
    #[arg(long)]
    log_rss: bool,
}

#[tokio::main]
async fn main() -> ExitCode {
    #[cfg(feature = "tracing_instrument")]
    akd::aegon::tracing_init::init_tree_subscriber();
    let args = Args::parse();

    akd::aegon::instrument::set_rss_log(args.log_rss);
    akd::aegon::instrument::log_rss("startup");

    if args.srs_path.is_none() && args.setup_seed.is_none() {
        eprintln!("error: provide either --srs-path or --setup-seed");
        return ExitCode::from(2);
    }

    let aegon_cfg = AegonConfig::<Bn254, Pcs> {
        log_capacity: args.shard_log_capacity,
        private: args.private,
        pcs_config: KZHKConfig::new(args.kzh_k, args.private),
        audit_fs: Default::default(),
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

    // ---- SRS bootstrap state (only used in distributed-gen mode) ----
    //
    // Bound on its own port before any heavy work so the bootstrap
    // actor can push trapdoors as soon as it's run. The state object
    // is shared with the gRPC service (which fields incoming
    // `BootstrapSrs` + `GetSrsSlab` + `WaitForReady` calls) and with
    // the main control flow below (which polls it for trapdoors and
    // publishes local slabs as they're computed).
    let srs_state: Option<Arc<SrsBootstrapState<Bn254>>> = if let Some(addr) = args.srs_bind {
        let seed = args.setup_seed.expect("--srs-bind requires --setup-seed (checked by clap)");
        let cache_dir = args
            .srs_cache_dir
            .clone()
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|h| PathBuf::from(h).join(".cache").join("aegon-srs"))
            })
            .unwrap_or_else(|| PathBuf::from(".aegon-srs-cache"));
        eprintln!(
            "[distributed-gen] cache_dir={} srs_bind={}",
            cache_dir.display(),
            addr,
        );
        let cfg = SrsBootstrapConfig {
            shard_id,
            log_capacity: args.shard_log_capacity as u32,
            k: args.kzh_k as u32,
            setup_seed: seed,
            cache_dir,
        };
        let state = SrsBootstrapState::<Bn254>::new(cfg);

        // Bind the SrsService gRPC server on `--srs-bind` in a
        // background task. Keep it alive for the full process lifetime
        // — bootstrap actor late-arriving `WaitForReady` polls keep
        // working after the shard is Ready, and the cost of holding
        // the listener is negligible.
        let svc = SrsGrpcServer::new(Arc::clone(&state)).into_service();
        tokio::spawn(async move {
            if let Err(e) = Server::builder().add_service(svc).serve(addr).await {
                eprintln!("srs service exited: {e}");
            }
        });
        // Allow the listener a beat to come up before we start fielding
        // peer connections.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        Some(state)
    } else {
        None
    };

    let mut aegon = match (&args.srs_path, args.setup_seed, &srs_state) {
        // --srs-bind path: distributed gen (or cache hit) via the
        // bootstrap actor + peers. This is the production-grade path.
        (None, Some(_seed), Some(state)) => {
            // Cache hit?
            match try_cache_hit::<Bn254>(state).await {
                Ok(Some((_up, pk, vk))) => {
                    eprintln!("[distributed-gen] cache hit; skipping bootstrap handshake");
                    build_aegon_from_srs(pk, vk, &aegon_cfg, &db_source, shard_id)
                },
                Ok(None) => {
                    eprintln!(
                        "[distributed-gen] cache miss — awaiting BootstrapSrs from bootstrap actor"
                    );
                    match run_distributed_compute::<Bn254>(Arc::clone(state)).await {
                        Ok((_up, pk, vk)) => {
                            eprintln!("[distributed-gen] SRS assembled + cached");
                            build_aegon_from_srs(pk, vk, &aegon_cfg, &db_source, shard_id)
                        },
                        Err(e) => {
                            eprintln!("[distributed-gen] error: {e}");
                            return ExitCode::from(1);
                        },
                    }
                },
                Err(e) => {
                    eprintln!("[distributed-gen] cache read error: {e}");
                    return ExitCode::from(1);
                },
            }
        },
        // --srs-path path: load a pre-existing SRS file (production
        // path without distributed gen — typically a trusted-setup
        // ceremony output).
        (Some(path), _, _) => {
            eprintln!("loading SRS from {}", path.display());
            let (pk, vk) = match read_srs_from_file::<Bn254, Pcs>(path) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error loading SRS: {e}");
                    return ExitCode::from(1);
                },
            };
            build_aegon_from_srs(pk, vk, &aegon_cfg, &db_source, shard_id)
        },
        // Legacy in-process gen via seed (no --srs-bind). Test mode.
        (None, Some(seed), None) => {
            eprintln!("WARNING: generating SRS in-process from seed {seed} (test mode only)");
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            match Aegon::setup(&mut rng, &aegon_cfg) {
                Ok(a) => Ok(a),
                Err(e) => Err(format!("setup: {e}")),
            }
        },
        (None, None, _) => unreachable!("checked above"),
    };
    let mut aegon = match aegon {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error initializing Aegon: {e}");
            return ExitCode::from(1);
        },
    };
    akd::aegon::instrument::log_rss("post_aegon_init");

    // Honor `--no-retain-epoch-polys` before any publish runs. Bench
    // shards set this to skip the per-epoch poly+state clones (~50 MB
    // each at shard_log_capacity=27) that `consistency_proof` would
    // need but the bench doesn't exercise. Affects only future
    // publishes — the epoch-0 snapshot that Aegon::setup just inserted
    // already has its polys populated (cheap; both are empty there).
    if args.no_retain_epoch_polys {
        eprintln!("shard config: retain_epoch_polys=false (consistency_proof at old epochs disabled)");
        aegon.set_retain_epoch_polys(false);
    }

    // DIAGNOSTIC v15c: VRF prover RE-attached (cache will populate
    // during publish_batch). The coord's lookup_label_two_layer was
    // reverted to v14-style structure for this iteration so we can
    // tell whether the v15 regression came from the cache-aware
    // lookup branches.
    aegon.set_vrf_prover(VrfProver::from_env());

    // Distributed-gen path advances phase as we work through init +
    // prefill so any `WaitForReady` poll has a useful status string.
    if let Some(state) = &srs_state {
        state
            .set_phase(Phase::Initializing, "post-init")
            .await;
    }

    // Cluster path: if --masking-addr is set, swap out Aegon's
    // default in-process MaskingPool for a remote MaskingClient (one
    // endpoint) or MaskingClientPool (multiple endpoints, dispatched
    // round-robin). Aegon always has *some* masking source in hiding
    // mode (built by `Aegon::init_with_arc`); the local pool is only
    // kept when no remote endpoint is configured (e.g. single-shard
    // dev).
    match args.masking_addr.len() {
        0 => {},
        1 => {
            let addr = &args.masking_addr[0];
            eprintln!("connecting to masking server at {addr}");
            match akd::aegon::masking::MaskingClient::<Bn254, Pcs>::connect(addr.clone()) {
                Ok(client) => aegon.set_masking_source(std::sync::Arc::new(client)),
                Err(e) => {
                    eprintln!("error connecting to masking server '{addr}': {e}");
                    return ExitCode::from(1);
                },
            }
        },
        n => {
            eprintln!("connecting to {n} masking servers (round-robin): {:?}", args.masking_addr);
            match akd::aegon::masking::MaskingClientPool::<Bn254, Pcs>::connect_all(
                &args.masking_addr,
            ) {
                Ok(pool) => aegon.set_masking_source(std::sync::Arc::new(pool)),
                Err(e) => {
                    eprintln!("error connecting to masking pool {:?}: {e}", args.masking_addr);
                    return ExitCode::from(1);
                },
            }
        },
    }

    // Benchmark-only prefill. Runs after setup but before binding the
    // gRPC socket — we want the polynomials populated by the time the
    // coordinator connects, but the prefill itself takes time at large
    // counts (each entry = 1 RNG draw + 1 HashMap insert, plus one
    // commit + update_state pass at the end).
    if let Some(count) = args.prefill_count {
        if count > 0 {
            if let Some(state) = &srs_state {
                state
                    .set_phase(Phase::Prefilling, "prefilling")
                    .await;
            }
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

    // Releases any `WaitForReady` polls held by the bootstrap actor.
    // Done before we bind ShardService so a poll racing the bind
    // observes Ready first, not "service available but not really
    // initialized."
    if let Some(state) = &srs_state {
        state.set_phase(Phase::Ready, "ready").await;
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
        match ShardServer::<Bn254, Pcs, EcVrfHash>::new_with_checkpoint(
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
        ShardServer::<Bn254, Pcs, EcVrfHash>::new(aegon)
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

/// Build an `Aegon` from prover/verifier params, honouring the checkpoint-
/// recovery path. Centralised here so the file-load, distributed-gen,
/// and (future) cache-hit paths all share the same checkpoint logic.
fn build_aegon_from_srs(
    pk: akd_core::aegon_crypto::pcs::kzhk::srs::KZHKProverParam<Bn254>,
    vk: akd_core::aegon_crypto::pcs::kzhk::srs::KZHKVerifierParam<Bn254>,
    aegon_cfg: &AegonConfig<Bn254, Pcs>,
    db_source: &DbSource,
    shard_id: u32,
) -> Result<Aegon, String> {
    let recovered = load_aegon_checkpoint_from_db::<Bn254, Pcs>(db_source, shard_id)
        .map_err(|e| format!("read shard checkpoint: {e}"))?;
    match recovered {
        Some(ckpt) => {
            eprintln!(
                "resuming shard {shard_id} from checkpoint at epoch {}",
                ckpt.epoch
            );
            Aegon::restore_from_checkpoint(pk, vk, aegon_cfg, ckpt)
                .map_err(|e| format!("restore_from_checkpoint: {e}"))
        },
        None => Aegon::init(pk, vk, aegon_cfg).map_err(|e| format!("init: {e}")),
    }
}
