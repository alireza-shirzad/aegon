//! Sweep the publish batch size and record per-batch wall time.
//!
//! What this measures: the wall-clock cost of one
//! `ShardedAegon::publish(batch)` call as a function of `batch.len()`.
//! Setup (SRS generation, polynomial initialization) is done inside
//! `with_inputs` and is **not** included in the timed region — only
//! the publish call itself is timed.
//!
//! ## Running
//!
//! Default sweep:
//!
//! ```text
//! cargo bench --bench publish_sweep
//! ```
//!
//! JSON output (this codebase uses the alireza-shirzad/divan fork,
//! which adds a JSON writer; see `-- --help` for the exact flag the
//! fork exposes):
//!
//! ```text
//! cargo bench --bench publish_sweep -- --output json --output-file bench.json
//! ```
//!
//! ## Tuning the sweep without recompiling
//!
//! These env vars are read at bench-binary startup:
//!
//! | Var                          | Default          | Meaning                                       |
//! | :--                          | :--              | :--                                            |
//! | `AEGON_BENCH_BATCH_SIZES`    | `1,4,16,64,256`  | comma-separated batch sizes to sweep           |
//! | `AEGON_BENCH_SHARD_LOG_CAP`  | `16`             | log_2(slots per shard)                         |
//! | `AEGON_BENCH_LOG_N_SHARDS`   | `0`              | log_2(num shards). 0 → single-shard            |
//! | `AEGON_BENCH_PRELOAD_USERS`  | `0`              | users to pre-publish before each timed batch   |
//! | `AEGON_BENCH_PRIVATE`        | `0`              | non-zero → zk-KZH mode                          |
//!
//! `AEGON_BENCH_PRELOAD_USERS` simulates a dictionary that already has
//! `N` users when the timed batch arrives. The preload runs inside
//! `with_inputs` and is **not** in the timed region — only the
//! incremental batch is. Use this to measure how publish cost scales
//! as the dictionary fills up (which affects open-addressing probe
//! lengths and sparse-poly support sizes).
//!
//! Both the preload and the timed batch must fit into the total
//! dictionary, so keep
//! `PRELOAD_USERS + max(BATCH_SIZES) << 2^(SHARD_LOG_CAPACITY + LOG_N_SHARDS)`.
//! At high load factors open-addressing trails grow long and the bench
//! itself slows down — keep the combined occupancy below ~50% for
//! reproducible numbers.
//!
//! ## Why single-shard by default
//!
//! With `LOG_N_SHARDS=0` we exercise exactly one shard's worth of
//! polynomial work — the curve is the pure-CPU publish cost without
//! the rayon-parallel-across-shards confound. Crank `LOG_N_SHARDS` up
//! to see how sharding changes the picture.

use std::sync::OnceLock;

use akd::aegon::{optimal_kzh_k, Sha256Hash, ShardedAegon, ShardedAegonConfig};
use ark_bn254::Bn254;
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Pcs = akd_core::aegon_crypto::pcs::kzhk::KZHK<Bn254>;
type Sharded = ShardedAegon<Bn254, Pcs, Sha256Hash>;

/// One environment-resolved set of bench parameters. Built once at
/// startup and shared across every sample so we don't re-parse env
/// vars or pay clamping logic per iteration.
struct BenchParams {
    batch_sizes: Vec<usize>,
    shard_log_capacity: usize,
    log_n_shards: usize,
    kzh_k: usize,
    preload_users: usize,
    private: bool,
}

fn params() -> &'static BenchParams {
    static P: OnceLock<BenchParams> = OnceLock::new();
    P.get_or_init(|| {
        let batch_sizes = std::env::var("AEGON_BENCH_BATCH_SIZES")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or_else(|| vec![1, 4, 16, 64, 256]);
        let shard_log_capacity = env_usize("AEGON_BENCH_SHARD_LOG_CAP", 16);
        let log_n_shards = env_usize("AEGON_BENCH_LOG_N_SHARDS", 0);
        let preload_users = env_usize("AEGON_BENCH_PRELOAD_USERS", 0);
        let private = env_usize("AEGON_BENCH_PRIVATE", 0) != 0;
        let kzh_k = optimal_kzh_k(shard_log_capacity);
        BenchParams {
            batch_sizes,
            shard_log_capacity,
            log_n_shards,
            kzh_k,
            preload_users,
            private,
        }
    })
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Fresh `ShardedAegon` server, optionally pre-populated with
/// `params().preload_users` brand-new users (the "dictionary already
/// has N users" scenario). Per-iteration cost is the SRS gen +
/// polynomial init + preload publish — substantial, but it lives in
/// `with_inputs` and is NOT in the timed region.
///
/// Preload uses its own `preload-{i}` label namespace so the timed
/// batch's `user-{i}` labels never collide with preloaded ones.
fn build_server() -> Sharded {
    let p = params();
    let cfg = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(p.shard_log_capacity)
        .log_n_shards(p.log_n_shards)
        .private(p.private)
        .kzh_k(p.kzh_k)
        .build()
        .expect("config builds");
    let mut rng = ChaCha20Rng::seed_from_u64(0xA56_5);
    let mut server = Sharded::setup(&mut rng, &cfg).expect("setup");
    if p.preload_users > 0 {
        let preload: Vec<(Vec<u8>, Vec<u8>)> = (0..p.preload_users as u32)
            .map(|i| {
                (
                    format!("preload-{i}").into_bytes(),
                    format!("pv-{i}").into_bytes(),
                )
            })
            .collect();
        server.publish(&preload).expect("preload publish");
    }
    server
}

/// `batch_size` fresh `(label, value)` pairs. The labels are unique so
/// every entry forces a brand-new open-addressing placement (worst
/// case for `plan_phase_1_batches`).
fn build_batch(batch_size: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..batch_size as u32)
        .map(|i| (format!("user-{i}").into_bytes(), format!("v-{i}").into_bytes()))
        .collect()
}

#[divan::bench(args = params().batch_sizes.iter().copied().collect::<Vec<_>>(), sample_count = 5)]
fn publish_batch(bencher: divan::Bencher, batch_size: usize) {
    bencher
        .with_inputs(|| (build_server(), build_batch(batch_size)))
        .bench_local_values(|(mut server, batch)| {
            server.publish(&batch).expect("publish")
        });
}

fn main() {
    let p = params();
    eprintln!(
        "publish_sweep params: shard_log_capacity={}, log_n_shards={}, kzh_k={}, preload_users={}, private={}, batch_sizes={:?}",
        p.shard_log_capacity, p.log_n_shards, p.kzh_k, p.preload_users, p.private, p.batch_sizes,
    );
    divan::main();
}
