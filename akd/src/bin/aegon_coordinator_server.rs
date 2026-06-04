//! `aegon_coordinator_server` — user-facing coordinator binary.
//!
//! Sets up a `ShardedAegon`, connects to every shard via gRPC, and
//! exposes the split lookup API (`LookupLabel`, `LookupValue`,
//! `CurrentCommitment`) on a TCP listen address. End users hit this
//! instead of the shards directly.
//!
//! Optionally pre-populates the dictionary with `--seed-batch-size`
//! deterministic `(label, value)` pairs (same namespace the
//! coordinator bench uses) so the server has something to look up
//! against right after startup. Drop the flag for a pristine
//! coordinator.
//!
//! Companion to `aegon_coordinator_bench`: bench drives publishes
//! and measures, this serves lookups. They share the same
//! `ShardedAegon` setup path — only the post-setup phase differs.
//!
//! Usage:
//!
//! ```text
//! aegon_coordinator_server \
//!   --listen 0.0.0.0:50100 \
//!   --shard-log-capacity 20 --kzh-k 10 --setup-seed 42 \
//!   --endpoints http://aegon-shard-0:50051,...,http://aegon-shard-3:50051 \
//!   --db-url redis://aegon-bench-db:6379 \
//!   --seed-batch-size 100
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use akd::aegon::{
    coordinator_grpc::CoordinatorServer, DbSource, EcVrfHash, ShardTransport, ShardedAegon,
    ShardedAegonConfig, SrsSource, VrfProver,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;
use tokio::sync::RwLock as AsyncRwLock;

type Pcs = KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, EcVrfHash>;

#[derive(Debug, Parser)]
#[command(
    name = "aegon_coordinator_server",
    about = "Run the user-facing Aegon coordinator gRPC service."
)]
struct Args {
    /// Address to bind the gRPC service on.
    #[arg(long, default_value = "0.0.0.0:50100")]
    listen: String,

    /// log_2 of slots per shard. Must match every `aegon_shard_server`.
    #[arg(long)]
    shard_log_capacity: usize,

    /// KZH-k block parameter. Must match every shard.
    #[arg(long)]
    kzh_k: usize,

    /// Comma-separated shard endpoints. Length must be a power of two.
    #[arg(long, value_delimiter = ',')]
    endpoints: Vec<String>,

    /// Path to a serialized SRS. Mutually exclusive with --setup-seed.
    #[arg(long, conflicts_with = "setup_seed")]
    srs_path: Option<PathBuf>,

    /// Deterministic in-process SRS gen. Must match every shard's seed.
    #[arg(long)]
    setup_seed: Option<u64>,

    /// Enable zero-knowledge mode (must match every shard).
    #[arg(long)]
    private: bool,

    /// Redis URL for coordinator-side open-addressing. Required when
    /// the shards were started with `--db-url` so the slot-occupancy
    /// keys are populated. Mutually exclusive with `--db-path`.
    #[arg(long, conflicts_with = "db_path")]
    db_url: Option<String>,

    /// Local RocksDB directory for coordinator-side state (value,
    /// routing, history, epoch commits, checkpoints). Process-local;
    /// when set, slot-occupancy probes fall back to per-probe gRPC
    /// `is_index_slot_occupied` calls to the owning shard instead of
    /// pipelined `EXISTS` against shared Redis. Use this when the
    /// dataset is too large to fit Redis RAM (target ~2^34 scale).
    #[arg(long)]
    db_path: Option<std::path::PathBuf>,

    /// If non-zero, publish `--seed-batch-size` deterministic
    /// `(label, value)` pairs immediately after setup so the
    /// coordinator has known labels to look up. Labels follow the
    /// same `b{batch}-s0-u{i}` namespace `aegon_coordinator_bench`
    /// uses, so a hand-driven client can predict label names without
    /// out-of-band coordination.
    #[arg(long, default_value_t = 0)]
    seed_batch_size: usize,

    /// Write the coordinator's ECVRF public key (hex, 64 chars + LF)
    /// to this path after setup, before binding the listener. Useful
    /// for out-of-band distribution to auditors / clients that don't
    /// want to call `CurrentCommitment` just to learn the key. Write
    /// is atomic via `write-then-rename` so a downstream watcher only
    /// ever sees a complete file. Refuses to overwrite an existing
    /// file (delete it first if you really mean to clobber).
    #[arg(long)]
    vrf_pubkey_out: Option<PathBuf>,
}

/// Atomic write: stage the bytes in a sibling `*.tmp` file in the same
/// directory, then `rename(2)` into place. Refuses to clobber an
/// existing target so a stale pubkey from a prior run is never
/// silently overwritten — the operator has to delete it on purpose.
fn write_pubkey_atomic(path: &std::path::Path, pk_hex: &str) -> std::io::Result<()> {
    use std::io::Write;
    if path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("{} already exists; remove it first", path.display()),
        ));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "--vrf-pubkey-out path has no file name",
        )
    })?;
    let mut tmp = parent.to_path_buf();
    tmp.push(format!("{}.tmp", file_name.to_string_lossy()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(pk_hex.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
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
    } else if let Some(path) = &args.db_path {
        builder = builder.db(DbSource::Rocks(path.clone()));
    }
    let cfg = match builder.build() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: config invalid: {e}");
            return ExitCode::from(2);
        },
    };

    eprintln!(
        "coordinator: connecting to {} shards (shard_log_capacity={}, kzh_k={}, log_n_shards={})",
        args.endpoints.len(),
        args.shard_log_capacity,
        args.kzh_k,
        log_n_shards
    );
    let mut rng = ChaCha20Rng::seed_from_u64(args.setup_seed.unwrap_or(0));
    let t_setup = Instant::now();
    let mut state = match Sharded::setup(&mut rng, &cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: setup failed: {e}");
            return ExitCode::from(1);
        },
    };
    eprintln!(
        "coordinator: setup OK in {:.1} ms",
        t_setup.elapsed().as_secs_f64() * 1000.0
    );

    // Initialise the ECVRF prover from the configured key source:
    //   AEGON_VRF_SEED      — hex-encoded 32-byte Ed25519 secret;
    //   AEGON_VRF_KEY_PATH  — sealed-file path (auto-generated on
    //                         first run from /dev/urandom, persisted
    //                         at mode 0600);
    //   otherwise           — the published benchmark seed, useful
    //                         only for tests and CI.
    // The matching public key is automatically advertised in every
    // `CurrentCommitment` response so clients can build a
    // `VrfVerifier` without an out-of-band fetch.
    let prover = VrfProver::from_env();
    let pk_hex = hex::encode(prover.public_key().as_bytes());
    state.set_vrf_prover(prover);
    eprintln!(
        "coordinator: ECVRF prover attached (pubkey {pk_hex}); clients fetch via CurrentCommitment"
    );

    if let Some(path) = &args.vrf_pubkey_out {
        if let Err(e) = write_pubkey_atomic(path, &pk_hex) {
            eprintln!(
                "error: failed to write VRF public key to {}: {e}",
                path.display()
            );
            return ExitCode::from(1);
        }
        eprintln!("coordinator: ECVRF pubkey written to {}", path.display());
    }

    // Optional seeding so a fresh server has something to look up.
    // Single publish at most — keeps the binary's responsibility
    // narrow (it's a server, not a benchmark).
    if args.seed_batch_size > 0 {
        let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..args.seed_batch_size as u32)
            .map(|i| {
                (
                    format!("b{}-s0-u{i}", args.seed_batch_size).into_bytes(),
                    format!("v-{i}").into_bytes(),
                )
            })
            .collect();
        let t = Instant::now();
        match state.publish(&updates) {
            Ok(_commit) => eprintln!(
                "coordinator: seeded {} labels in {:.1} ms",
                args.seed_batch_size,
                t.elapsed().as_secs_f64() * 1000.0
            ),
            Err(e) => {
                eprintln!("error: seed publish failed: {e}");
                return ExitCode::from(1);
            },
        }
    }

    let addr: std::net::SocketAddr = match args.listen.parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: invalid --listen {:?}: {e}", args.listen);
            return ExitCode::from(2);
        },
    };

    // Wrap the (possibly already-seeded) `ShardedAegon` in the gRPC
    // adapter and serve. Drops into a tokio runtime here because the
    // setup phase is sync but the serve path is async.
    let server = CoordinatorServer::<Bn254, Pcs, EcVrfHash>::from_shared(Arc::new(
        AsyncRwLock::new(state),
    ));
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to build tokio runtime: {e}");
            return ExitCode::from(1);
        },
    };
    eprintln!("coordinator: serving on {addr}");
    if let Err(e) = runtime.block_on(server.serve(addr)) {
        eprintln!("error: coordinator gRPC server exited: {e}");
        return ExitCode::from(1);
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::write_pubkey_atomic;
    use std::path::PathBuf;

    fn unique_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        p.push(format!("aegon-vrf-pubkey-{name}-{pid}-{nanos}.hex"));
        p
    }

    #[test]
    fn writes_hex_plus_newline_and_no_tmp_left_behind() {
        let path = unique_path("ok");
        let hex = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
        write_pubkey_atomic(&path, hex).expect("write succeeds");
        let bytes = std::fs::read(&path).expect("read back");
        assert_eq!(bytes.len(), hex.len() + 1, "exactly hex + LF");
        assert_eq!(&bytes[..hex.len()], hex.as_bytes());
        assert_eq!(bytes[hex.len()], b'\n');
        // Tmp sibling must not survive a successful rename.
        let mut tmp = path.clone();
        let fname = format!(
            "{}.tmp",
            path.file_name().unwrap().to_string_lossy()
        );
        tmp.set_file_name(fname);
        assert!(
            !tmp.exists(),
            "tmp file {tmp:?} should be gone after rename"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn refuses_to_clobber_existing_file() {
        let path = unique_path("clobber");
        std::fs::write(&path, b"stale").expect("seed existing file");
        let err = write_pubkey_atomic(&path, "deadbeef")
            .expect_err("must refuse existing file");
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        // Existing content untouched.
        let bytes = std::fs::read(&path).expect("read back");
        assert_eq!(bytes, b"stale");
        let _ = std::fs::remove_file(&path);
    }
}
