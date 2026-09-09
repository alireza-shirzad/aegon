// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! `aegon_setup_bench` — single-process setup benchmark.
//!
//! Measures three things at one `(shard_log_capacity, kzh_k)`
//! configuration:
//!
//!   1. SRS generation wall-clock (`gen_srs_for_testing` with zk=true,
//!      matching the universal-SRS we use everywhere in Aegon).
//!   2. Trim wall-clock (universal → prover_param + verifier_param).
//!   3. Canonical-uncompressed serialised sizes of universal_params,
//!      prover_param, verifier_param. These are the bytes that go on
//!      disk in the SRS cache and on the wire in `--srs-path` deploys.
//!
//! Emits JSON to `--out` (or stdout) so the three-regime sweep can
//! consume it without parsing text.
//!
//! Sized for the **small** (`log_cap=22`, ~256 MiB SRS) and **medium**
//! (`log_cap=28`, ~16 GiB SRS) regimes on a single machine. The
//! **planetary** regime's distributed setup time needs the cluster
//! (`bench-cluster.sh setup-bench`); but the per-shard pk/vk sizes at
//! `log_cap=29, kzh_k=10` ARE deterministic, so running this bench at
//! the per-shard parameters from any beefy machine (~40 GiB RAM)
//! produces the sizes the cluster would observe.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use akd::aegon::optimal_kzh_k;
use akd_core::aegon_crypto::pcs::kzhk::srs::KZHKUniversalParams;
use akd_core::aegon_crypto::StructuredReferenceString;
use ark_bn254::Bn254;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;

type E = Bn254;

#[derive(Debug, Parser)]
#[command(
    name = "aegon_setup_bench",
    about = "Single-process SRS setup benchmark: gen time + pk/vk uncompressed sizes."
)]
struct Args {
    /// log_2 of the dictionary slot count this SRS will commit. For
    /// the sharded path this is `shard_log_capacity` (per-shard), not
    /// the cluster-wide log capacity. For the unsharded path it's the
    /// directory's full log capacity.
    #[arg(long)]
    shard_log_capacity: usize,

    /// KZH-k block parameter. Defaults to `optimal_kzh_k` for the
    /// given `shard_log_capacity` — same default that
    /// `bench-cluster.sh` uses, so per-regime numbers line up.
    #[arg(long)]
    kzh_k: Option<usize>,

    /// Deterministic seed for the SRS gen RNG. The seed only matters
    /// to reproduce the exact same SRS across runs — wall-clock and
    /// sizes are seed-invariant.
    #[arg(long, default_value_t = 0)]
    setup_seed: u64,

    /// Optional label for the JSON record (e.g. "small", "medium",
    /// "planetary-per-shard"). Pure metadata.
    #[arg(long)]
    label: Option<String>,

    /// JSON output path. If unset, writes to stdout.
    #[arg(long)]
    out: Option<PathBuf>,
}

/// Render the run record as a JSON string. Hand-rolled to match the
/// rest of Aegon's bench binaries (avoids pulling serde_json into
/// `akd`'s mandatory deps just for one bench). Schema: see the field
/// names in the format string + the doc-comments here.
///
/// `compute_secs` and `communication_secs` are populated here to keep
/// the local-single-shard schema aligned with the planetary
/// distributed-bench JSON emitted by `aegon_srs_bootstrap
/// --metrics-out`. On the local bench every wall-clock second is
/// compute (no peers to talk to), so `communication_secs = 0` by
/// construction.
#[allow(clippy::too_many_arguments)]
fn render_json(
    label: &str,
    shard_log_capacity: usize,
    kzh_k: usize,
    setup_seed: u64,
    gen_duration_secs: f64,
    trim_duration_secs: f64,
    universal_params_bytes: u64,
    prover_param_bytes: u64,
    verifier_param_bytes: u64,
    h_t_entries: &[u64],
) -> String {
    let h_t_csv = h_t_entries
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let compute_secs = gen_duration_secs + trim_duration_secs;
    format!(
        concat!(
            "{{\n",
            "  \"label\": \"{label}\",\n",
            "  \"shard_log_capacity\": {slc},\n",
            "  \"kzh_k\": {k},\n",
            "  \"setup_seed\": {seed},\n",
            "  \"gen_duration_secs\": {gen:.6},\n",
            "  \"trim_duration_secs\": {trim:.6},\n",
            "  \"compute_secs\": {compute:.6},\n",
            "  \"communication_secs\": 0.0,\n",
            "  \"universal_params_bytes\": {up},\n",
            "  \"prover_param_bytes\": {pk},\n",
            "  \"verifier_param_bytes\": {vk},\n",
            "  \"h_t_entries\": [{h_t_csv}]\n",
            "}}\n"
        ),
        label = label,
        slc = shard_log_capacity,
        k = kzh_k,
        seed = setup_seed,
        gen = gen_duration_secs,
        trim = trim_duration_secs,
        compute = compute_secs,
        up = universal_params_bytes,
        pk = prover_param_bytes,
        vk = verifier_param_bytes,
        h_t_csv = h_t_csv,
    )
}

fn main() -> ExitCode {
    let args = Args::parse();
    let k = args
        .kzh_k
        .unwrap_or_else(|| optimal_kzh_k(args.shard_log_capacity));
    let label = args
        .label
        .clone()
        .unwrap_or_else(|| format!("logcap{}-k{}", args.shard_log_capacity, k));

    eprintln!(
        "[setup-bench] {label}: shard_log_capacity={} kzh_k={} seed={:#018x}",
        args.shard_log_capacity, k, args.setup_seed
    );

    // ---- generate SRS ----
    let mut rng = ChaCha20Rng::seed_from_u64(args.setup_seed);
    let gen_start = Instant::now();
    let universal = match KZHKUniversalParams::<E>::gen_srs_for_testing(
        &mut rng,
        k,
        /* zk */ true,
        args.shard_log_capacity,
    ) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[setup-bench] gen failed: {e}");
            return ExitCode::from(1);
        }
    };
    let gen_duration = gen_start.elapsed();
    eprintln!(
        "[setup-bench] gen done in {:.3}s",
        gen_duration.as_secs_f64()
    );

    // ---- trim ----
    let trim_start = Instant::now();
    let (pk, vk) = match universal.trim(args.shard_log_capacity) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[setup-bench] trim failed: {e}");
            return ExitCode::from(1);
        }
    };
    let trim_duration = trim_start.elapsed();

    // ---- measure sizes ----
    let universal_bytes = universal.uncompressed_size() as u64;
    let pk_bytes = pk.uncompressed_size() as u64;
    let vk_bytes = vk.uncompressed_size() as u64;

    // Per-H_t entry counts. We don't have a direct getter for tensor
    // shapes here, but the dimensions come from universal_params and
    // H_t at index `t` has product_{j>=t} 2^{d_j} entries.
    let dims = universal.get_dimensions().clone();
    let mut h_t_entries: Vec<u64> = Vec::with_capacity(dims.len());
    for t in 0..dims.len() {
        let prod: u64 = dims[t..].iter().map(|&d| 1u64 << d).product();
        h_t_entries.push(prod);
    }

    let json = render_json(
        &label,
        args.shard_log_capacity,
        k,
        args.setup_seed,
        gen_duration.as_secs_f64(),
        trim_duration.as_secs_f64(),
        universal_bytes,
        pk_bytes,
        vk_bytes,
        &h_t_entries,
    );

    eprintln!(
        "[setup-bench] sizes: universal={:.2} GiB pk={:.2} GiB vk={:.2} MiB",
        universal_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        pk_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        vk_bytes as f64 / (1024.0 * 1024.0),
    );

    if let Some(path) = &args.out {
        match std::fs::write(path, &json) {
            Ok(()) => eprintln!("[setup-bench] wrote {}", path.display()),
            Err(e) => {
                eprintln!("[setup-bench] write '{}' failed: {e}", path.display());
                return ExitCode::from(1);
            }
        }
    } else {
        println!("{json}");
    }
    ExitCode::SUCCESS
}
