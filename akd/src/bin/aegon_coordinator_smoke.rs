//! `aegon_coordinator_smoke` — end-to-end smoke test for a remote
//! shard cluster. Run on the coordinator machine after all shard
//! servers are up. Connects to every shard listed in `--endpoints`,
//! publishes a batch of users, looks them all up, and verifies the
//! proofs.
//!
//! This is the cluster-bringup equivalent of the in-process
//! `grpc_sharded.rs` test. Output is wall-clock timings + a
//! pass/fail summary; intended for operators eyeballing whether the
//! cluster is healthy.
//!
//! Usage:
//!   aegon_coordinator_smoke \
//!     --shard-log-capacity 20 --kzh-k 6 \
//!     --srs-path /etc/aegon/shard.srs \
//!     --endpoints http://aegon-shard-0:50051,http://aegon-shard-1:50051,\
//!                 http://aegon-shard-2:50051,http://aegon-shard-3:50051 \
//!     --n-users 64

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use akd::aegon::{
    verify_sharded_lookup, DbSource, Sha256Hash, ShardTransport, ShardedAegon,
    ShardedAegonConfig, SrsSource,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, Sha256Hash>;

#[derive(Debug, Parser)]
#[command(
    name = "aegon_coordinator_smoke",
    about = "End-to-end smoke test against a remote shard cluster. Publishes some users, looks them up, verifies."
)]
struct Args {
    /// log_2 of slots in each shard's polynomial. Must match what
    /// every `aegon_shard_server` was started with.
    #[arg(long)]
    shard_log_capacity: usize,

    /// KZH-k block parameter. Must match every shard server.
    #[arg(long)]
    kzh_k: usize,

    /// Comma-separated list of shard endpoints. Length must be a
    /// power of two (it determines `log_n_shards`).
    #[arg(long, value_delimiter = ',')]
    endpoints: Vec<String>,

    /// Path to the SRS file (same file every shard loads). Used here
    /// only to derive the coordinator's `verifier_param` —
    /// `prover_param` is large and stays on the shard machines.
    /// Mutually exclusive with `--setup-seed`.
    #[arg(long, conflicts_with = "setup_seed")]
    srs_path: Option<PathBuf>,

    /// Test-only: derive `verifier_param` from a deterministic seed.
    /// Useful when shards were also started with `--setup-seed` and
    /// you don't want to bother with file distribution.
    #[arg(long)]
    setup_seed: Option<u64>,

    /// Enable zero-knowledge mode (must match every shard).
    #[arg(long)]
    private: bool,

    /// How many users to publish. Each gets a deterministic
    /// `user-{i}` / `v-{i}` pair.
    #[arg(long, default_value = "64")]
    n_users: u32,

    /// Optional Redis URL where the coordinator stores raw
    /// `(label, value)` bytes. With this set, `lookup` returns the
    /// value from Redis alongside the proof; without it, the smoke
    /// test falls back to the pre-known values it just published
    /// (single-process style). URL form: `redis://host[:port][/db]`.
    #[arg(long)]
    db_url: Option<String>,
}

fn main() -> ExitCode {
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
    let log_n_shards = args.endpoints.len().trailing_zeros() as usize;

    let mut builder = ShardedAegonConfig::<Bn254, Pcs>::builder()
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
    if let Some(url) = &args.db_url {
        builder = builder.db(DbSource::Redis(url.clone()));
    }
    let cfg = match builder.build() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: config invalid: {e}");
            return ExitCode::from(2);
        },
    };

    // Setup: this is the only step that contacts every shard up-front
    // (handshake + current_commitment per shard).
    eprintln!(
        "connecting to {} shards (shard_log_capacity={}, kzh_k={}, log_n_shards={})",
        args.endpoints.len(),
        args.shard_log_capacity,
        args.kzh_k,
        log_n_shards
    );
    let mut rng = ChaCha20Rng::seed_from_u64(args.setup_seed.unwrap_or(0));
    let t0 = Instant::now();
    let mut server = match Sharded::setup(&mut rng, &cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: setup failed: {e}");
            return ExitCode::from(1);
        },
    };
    let setup_ms = t0.elapsed().as_millis();
    eprintln!("setup OK in {setup_ms} ms");

    // Publish a batch.
    let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..args.n_users)
        .map(|i| (format!("user-{i}").into_bytes(), format!("v-{i}").into_bytes()))
        .collect();
    let t0 = Instant::now();
    let (commit, _audit) = match server.publish(&updates) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: publish failed: {e}");
            return ExitCode::from(1);
        },
    };
    let publish_ms = t0.elapsed().as_millis();
    eprintln!(
        "publish {} users OK in {publish_ms} ms ({:.2} ms/user)",
        updates.len(),
        publish_ms as f64 / updates.len() as f64
    );

    // Lookup + verify every user. When --db-url is set, the value
    // comes back from the DB tier and we also check it matches what
    // we published; without it, lookup returns an empty value and we
    // fall back to the pre-known one (single-process style).
    let ctx = server.sharded_verifier_context();
    let using_db = args.db_url.is_some();
    let t0 = Instant::now();
    let mut failures = 0usize;
    for (label, value) in &updates {
        let (db_value, proof) = match server.lookup(label) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: lookup {label:?} failed: {e}");
                failures += 1;
                continue;
            },
        };
        if using_db && &db_value != value {
            eprintln!(
                "error: DB returned wrong value for {label:?}: got {db_value:?}, expected {value:?}"
            );
            failures += 1;
            continue;
        }
        let verify_value = if using_db { &db_value } else { value };
        match verify_sharded_lookup::<Bn254, Pcs, Sha256Hash>(
            &ctx, &commit, label, verify_value, &proof,
        ) {
            Ok(true) => {},
            Ok(false) => {
                eprintln!("error: verify_sharded_lookup REJECTED for {label:?}");
                failures += 1;
            },
            Err(e) => {
                eprintln!("error: verify {label:?} failed: {e}");
                failures += 1;
            },
        }
    }
    let lookup_ms = t0.elapsed().as_millis();
    let n = updates.len();
    eprintln!(
        "{} lookup+verify pairs in {lookup_ms} ms ({:.2} ms/pair)",
        n,
        lookup_ms as f64 / n as f64
    );

    if failures == 0 {
        eprintln!("ALL {n} LOOKUPS VERIFIED — cluster healthy");
        ExitCode::SUCCESS
    } else {
        eprintln!("FAILED: {failures}/{n} lookups did not verify");
        ExitCode::from(1)
    }
}
