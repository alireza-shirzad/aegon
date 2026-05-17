//! `aegon_client` — minimal demo client for the coordinator.
//!
//! Connects to an `aegon_coordinator_server` over gRPC and walks the
//! split lookup design:
//!
//!   1. `current_commitment` → pin verification root.
//!   2. `lookup_label(label)` → verify the open-addressing trail,
//!      recover the canonical `LabelSlot` (don't trust the server's
//!      hint).
//!   3. `lookup_value(slot)` → verify the value-poly opening; if the
//!      caller knows the value bytes out-of-band, supply them via
//!      `--expected-value` and we'll verify the opening matches them.
//!
//! Useful both as an end-to-end smoke test and as a copy-pasteable
//! template for a real client. The verifier context is reconstructed
//! locally from `--setup-seed` (matches the bench cluster) or
//! `--srs-path`; in production a client would receive these from a
//! trusted bulletin board, not from the coordinator itself.
//!
//! Usage:
//!
//! ```text
//! aegon_client \
//!   --coordinator http://aegon-bench-coord:50100 \
//!   --shard-log-capacity 20 --kzh-k 10 --log-n-shards 2 --setup-seed 42 \
//!   --label "b100-s0-u0" \
//!   --expected-value "v-0"
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use akd::aegon::{
    coordinator_grpc::CoordinatorClient, Sha256Hash, ShardedVerifierContext, VerifierContext,
};
use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use akd_core::aegon_crypto::pcs::{PolynomialCommitmentScheme, StructuredReferenceString};
use ark_bn254::Bn254;
use ark_serialize::CanonicalDeserialize;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;
use std::marker::PhantomData;

type Pcs = KZHK<Bn254>;

#[derive(Debug, Parser)]
#[command(
    name = "aegon_client",
    about = "Connect to an Aegon coordinator, look up a label, verify both proofs locally."
)]
struct Args {
    /// Coordinator gRPC endpoint URL.
    #[arg(long)]
    coordinator: String,

    /// log_2 of slots per shard — must match the deployment.
    #[arg(long)]
    shard_log_capacity: usize,

    /// log_2 of the shard count — must match the deployment.
    #[arg(long)]
    log_n_shards: usize,

    /// KZH-k block parameter — must match the deployment.
    #[arg(long)]
    kzh_k: usize,

    /// SRS file (preferred). Mutually exclusive with --setup-seed.
    #[arg(long, conflicts_with = "setup_seed")]
    srs_path: Option<PathBuf>,

    /// Deterministic in-process SRS gen — same seed every shard /
    /// coordinator used. Only safe for test deployments.
    #[arg(long)]
    setup_seed: Option<u64>,

    /// Whether the deployment is in zero-knowledge mode.
    #[arg(long)]
    private: bool,

    /// Label to look up.
    #[arg(long)]
    label: String,

    /// Optional expected value. When set, the value proof is verified
    /// against this exact byte string (the coordinator's KV channel
    /// may not return value bytes inline for a slot-only request).
    #[arg(long)]
    expected_value: Option<String>,
}

fn main() -> ExitCode {
    let args = Args::parse();

    if args.srs_path.is_none() && args.setup_seed.is_none() {
        eprintln!("error: provide either --srs-path or --setup-seed");
        return ExitCode::from(2);
    }

    // Build the verifier param. The prover side lives on the shards;
    // we only need verifier_param. The setup_seed path mirrors what
    // each shard does in-process so a test deployment can be probed
    // without an out-of-band SRS file.
    let per_shard_size: usize = 1 << args.shard_log_capacity;
    let pcs_config = KZHKConfig::new(args.kzh_k, args.private);
    let verifier_param = if let Some(path) = &args.srs_path {
        // Read the SRS file once and trim the verifier param.
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: read --srs-path {path:?}: {e}");
                return ExitCode::from(1);
            },
        };
        match <Pcs as PolynomialCommitmentScheme<Bn254>>::SRS::deserialize_uncompressed_unchecked(
            &bytes[..],
        ) {
            Ok(srs) => srs.extract_verifier_param(args.shard_log_capacity),
            Err(e) => {
                eprintln!("error: deserialize SRS: {e}");
                return ExitCode::from(1);
            },
        }
    } else {
        let seed = args.setup_seed.unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        match <Pcs as PolynomialCommitmentScheme<Bn254>>::gen_srs_for_testing(
            pcs_config,
            &mut rng,
            per_shard_size,
        ) {
            Ok(srs) => srs.extract_verifier_param(args.shard_log_capacity),
            Err(e) => {
                eprintln!("error: gen_srs_for_testing: {e}");
                return ExitCode::from(1);
            },
        }
    };

    let verifier_inner = VerifierContext::<Bn254, Pcs> {
        log_capacity: args.shard_log_capacity,
        verifier_param,
        _e: PhantomData,
    };
    let verifier_ctx = ShardedVerifierContext::new(verifier_inner, args.log_n_shards);

    eprintln!("client: connecting to {} ...", args.coordinator);
    let client = match CoordinatorClient::<Bn254, Pcs, Sha256Hash>::connect(
        args.coordinator.clone(),
        verifier_ctx,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: connect: {e}");
            return ExitCode::from(1);
        },
    };

    // 1. Pin the verification root.
    let commit = match client.current_commitment() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: current_commitment: {e}");
            return ExitCode::from(1);
        },
    };
    eprintln!("client: current commitment OK");

    // 2. Look up the label and verify the open-addressing trail.
    let label_bytes = args.label.as_bytes().to_vec();
    let t = Instant::now();
    let slot = match client.lookup_label(&commit, &label_bytes) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: lookup_label({:?}): {e}", args.label);
            return ExitCode::from(1);
        },
    };
    eprintln!(
        "client: lookup_label OK in {:.1} ms → shard {}, slot len {}",
        t.elapsed().as_secs_f64() * 1000.0,
        slot.shard_id,
        slot.slot_bits.len()
    );

    // 3. Open the value at that slot.
    let t = Instant::now();
    let proof = if let Some(expected) = &args.expected_value {
        let expected_bytes = expected.as_bytes().to_vec();
        match client.lookup_value_with_bytes(&commit, &slot, &expected_bytes) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: lookup_value_with_bytes: {e}");
                return ExitCode::from(1);
            },
        }
    } else {
        match client.lookup_value(&commit, &slot) {
            Ok((value, p)) => {
                if value.is_empty() {
                    eprintln!(
                        "client: lookup_value returned no inline value bytes — pass --expected-value to verify"
                    );
                } else {
                    eprintln!(
                        "client: lookup_value returned {} bytes of inline value (verified)",
                        value.len()
                    );
                }
                p
            },
            Err(e) => {
                eprintln!("error: lookup_value: {e}");
                return ExitCode::from(1);
            },
        }
    };
    eprintln!(
        "client: lookup_value OK in {:.1} ms → eval matches H_F(value), proof verified",
        t.elapsed().as_secs_f64() * 1000.0
    );
    let _ = proof; // proof is verified; keep around for downstream auditing if a caller wants.

    eprintln!("client: done.");
    ExitCode::SUCCESS
}
