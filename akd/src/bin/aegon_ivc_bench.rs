//! Benchmark the IVC audit path (paper §3, fast-forwarding).
//!
//! The claim under test is a *shape* claim, not just a speed one:
//!
//! * the classic auditor's total work grows **linearly in the number
//!   of epochs**, and it must be online for every one of them;
//! * the IVC auditor's work is **constant in the number of epochs**,
//!   and it need never have been online at all.
//!
//! So the interesting output is the crossover: how many epochs of
//! being offline it takes before one recursive verification is
//! cheaper than catching up, and what the per-epoch proving cost is
//! that buys it.
//!
//! Uses synthetic epoch transitions (see
//! [`akd::aegon::ivc::synthetic`]). That is sound here because the
//! audit's cost depends on the shard count and not on directory fill
//! — which is itself one of the properties being demonstrated — so a
//! 128-shard measurement needs no planetary-scale SRS. The
//! `ivc_audit_e2e` example covers the real publish path.
//!
//! ```text
//! cargo run --release -p akd --features ivc_audit --bin aegon_ivc_bench -- --shards 1,2,4,8
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use akd::aegon::ivc::circuit::SigmaWitness;
use akd::aegon::ivc::fs_poseidon::{
    domain, poseidon_chain_scalar, poseidon_sigma_challenge, ro_constants, FsParams,
    ShardCommitments,
};
use akd::aegon::ivc::prover::{
    compress, compress_pp, TerminalSnark, compressed_pp_proof_size_bytes, compressed_proof_size_bytes,
    IvcAuditParams, IvcAuditProver,
};
use akd::aegon::ivc::grouped::{
    verify_grouped_folding_proofs, verify_grouped_ivc_audit, GroupPlan, GroupedIvcAuditParams,
    GroupedIvcAuditProver,
};
use akd::aegon::ivc::synthetic::{
    as_sharded_epoch_commitment, genesis, honest_chain, honest_grouped_chain, rand_point, rng,
    Pcs,
};
use akd::aegon::ivc::verifier::verify_compressed_ivc_audit;
use akd::aegon::{verify_sharded_invariance, ShardedAuditState, ShardedVerifierContext, VerifierContext};
use akd_core::aegon_crypto::pcs::kzhk::srs::KZHKUniversalParams;
use akd_core::aegon_crypto::pcs::StructuredReferenceString;
use ark_bn254::{Bn254, Fr, G1Affine};
use ark_ec::{CurveGroup, AffineRepr};
use ark_std::UniformRand;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::SeedableRng;
use clap::Parser;
use rand_chacha::ChaCha20Rng;

#[derive(Parser, Debug)]
#[command(about = "Benchmark Aegon's IVC (Nova) audit path")]
struct Args {
    /// Comma-separated shard counts to sweep.
    #[arg(long, default_value = "1,2,4,8")]
    shards: String,
    /// Epoch transitions to fold per configuration.
    #[arg(long, default_value_t = 4)]
    epochs: usize,
    /// Polynomial variable count bound into the transcripts. Does not
    /// affect audit cost; only the SRS used for the classic-auditor
    /// comparison.
    #[arg(long, default_value_t = 10)]
    num_vars: usize,
    /// Instead of the sweep, sweep the number of independent folding
    /// groups at a fixed shard count. Group-sharding splits the shard
    /// set into `G` chains that fold and compress in parallel; this
    /// reports what one host in a distributed deployment pays, and
    /// what all `G` groups cost if run on a single box.
    #[arg(long)]
    group_sweep: bool,
    /// Compare the two terminal SNARKs -- Spartan (`spartan::snark`)
    /// against MicroNova (`spartan::ppsnark`) -- on setup, prove,
    /// proof size and, above all, VERIFY time.
    #[arg(long)]
    snark_compare: bool,
    /// Isolate what the Poseidon audit-FS costs the *classic*
    /// auditor, against the SHA256 derivation it replaces, on this
    /// machine. Times only the Fiat-Shamir work, so the answer is not
    /// confounded by hardware or dictionary fill.
    #[arg(long)]
    fs_compare: bool,
    /// Number of independent Fiat-Shamir chain groups to audit under.
    ///
    /// Applies to every mode. `1` (the default) is a single chain
    /// over every shard -- the original behaviour. A larger value
    /// splits the shards into that many contiguous groups that fold,
    /// compress and verify independently; it must divide `--shards`.
    ///
    /// `--group-sweep` accepts a comma-separated list here and walks
    /// it; the other modes use the first value.
    ///
    /// This must match the server's `chain_groups`: the folding
    /// circuit re-derives the chain scalars in-circuit, so a
    /// mismatch makes every proof fail to verify.
    #[arg(long, default_value = "1")]
    groups: String,
    /// Instead of the sweep, fold epoch by epoch at a single shard
    /// count and report the proof size after each one. Tests the
    /// central claim directly: that neither proof grows with the
    /// number of epochs folded.
    #[arg(long)]
    size_sweep: bool,
}

/// The server's per-epoch cost under the *classic* (non-IVC) audit:
/// producing the epoch's audit object from the epoch's data
/// commitments. That is the two Fiat-Shamir chain derivations, the
/// homomorphic chain update for every shard, and one Schnorr proof of
/// knowledge of each shard's fresh value blinding (paper §7).
///
/// Deliberately takes `new_index`/`new_value` already computed: those
/// come out of the publish path and are not audit cost. Everything
/// inside is.
fn produce_audit_object(
    p: FsParams,
    h: G1Affine,
    prev: &[ShardCommitments],
    new_index: &[G1Affine],
    new_value: &[G1Affine],
    blinds: &[Fr],
    nonces: &[Fr],
) -> (Vec<ShardCommitments>, Vec<SigmaWitness>) {
    let c = ro_constants();
    let r_index = poseidon_chain_scalar(&c, domain::CHAIN_INDEX, p, Fr::from(0u64), new_index);
    let r_value = poseidon_chain_scalar(&c, domain::CHAIN_VALUE, p, Fr::from(0u64), new_value);

    let mut next = Vec::with_capacity(prev.len());
    let mut sigma = Vec::with_capacity(prev.len());
    for (j, sc) in prev.iter().enumerate() {
        let d_index = (new_index[j].into_group() - sc.index.into_group()) * r_index;
        let rand_index = (sc.rand_index.into_group() + d_index).into_affine();
        let d_value = (new_value[j].into_group() - sc.value.into_group()) * r_value;
        let rand_value =
            (sc.rand_value.into_group() + d_value + h.into_group() * blinds[j]).into_affine();
        let nc = ShardCommitments {
            index: new_index[j],
            value: new_value[j],
            rand_index,
            rand_value,
        };
        let r_commit = (h.into_group() * nonces[j]).into_affine();
        let e = poseidon_sigma_challenge(
            &c,
            p.num_vars,
            &sc.value,
            &nc.value,
            &sc.rand_value,
            &nc.rand_value,
            r_value,
            &r_commit,
        );
        sigma.push(SigmaWitness {
            r_commit,
            response: nonces[j] + e * blinds[j],
        });
        next.push(nc);
    }
    (next, sigma)
}

/// The group counts named by `--groups`.
fn group_counts(args: &Args) -> Vec<usize> {
    args.groups
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|g: &usize| *g > 0)
        .collect()
}

/// The single group count for the modes that take just one.
fn single_group_count(args: &Args) -> usize {
    group_counts(args).first().copied().unwrap_or(1)
}

fn mean(ds: &[Duration]) -> Duration {
    if ds.is_empty() {
        return Duration::ZERO;
    }
    ds.iter().sum::<Duration>() / ds.len() as u32
}

fn main() {
    let args = Args::parse();
    let shard_counts: Vec<usize> = args
        .shards
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|n: &usize| n.is_power_of_two() && *n > 0)
        .collect();
    assert!(!shard_counts.is_empty(), "no valid shard counts given");

    println!("Aegon IVC audit benchmark");
    println!("  epochs folded per config: {}", args.epochs);
    println!("  num_vars: {}", args.num_vars);
    println!("  chain groups: {}\n", single_group_count(&args));

    // One small SRS, reused for the classic-auditor comparison. The
    // audit is independent of dictionary size, so a small SRS gives
    // the same per-epoch verify cost a planetary one would.
    let mut srs_rng = ChaCha20Rng::seed_from_u64(0x5252_5252);
    let srs = <KZHKUniversalParams<Bn254> as StructuredReferenceString<Bn254>>::gen_srs_for_testing(
        &mut srs_rng,
        2,
        true,
        args.num_vars,
    )
    .expect("srs gen");
    let (_pp, vk) =
        <KZHKUniversalParams<Bn254> as StructuredReferenceString<Bn254>>::trim(&srs, args.num_vars)
            .expect("trim");
    let h = vk.get_h();

    if args.size_sweep {
        size_sweep(&args, h, &vk);
        return;
    }

    if args.group_sweep {
        group_sweep(&args, h);
        return;
    }

    if args.snark_compare {
        snark_compare(&args, h);
        return;
    }

    if args.fs_compare {
        fs_compare(&args);
        return;
    }

    println!(
        "{:>7} {:>12} {:>11} {:>11} {:>11} {:>11} {:>10} {:>12} {:>10}",
        "shards",
        "constraints",
        "setup",
        "fold/epoch",
        "fold proof",
        "compress",
        "pub proof",
        "verify",
        "classic/ep",
    );
    println!("{}", "-".repeat(104));

    let groups = single_group_count(&args);
    for &n in &shard_counts {
        let plan = match GroupPlan::new(n, groups) {
            Ok(pl) => pl,
            Err(e) => {
                println!("{n:>7}  skipped: {e}");
                continue;
            },
        };
        let mut r = rng(0xBE_11C4 ^ n as u64);
        // Everything below runs through the grouped types. At
        // `groups == 1` that is the single-chain path exactly -- see
        // the `single_group_matches_the_ungrouped_prover` test --
        // so one code path serves both and `--groups` needs no
        // special-casing anywhere.
        let (epochs, sigmas) = honest_grouped_chain(plan, args.num_vars, h, args.epochs, &mut r);

        // ---- Nova parameter setup (one-time, per deployment) ----
        let t = Instant::now();
        let ivc = GroupedIvcAuditParams::setup(plan, args.num_vars, h).expect("IVC setup");
        let setup_time = t.elapsed();
        let constraints = ivc.constraints_per_step();

        // ---- folding: the prover's per-epoch cost ----
        let mut prover =
            GroupedIvcAuditProver::new(&ivc, &epochs[0]).expect("prover init");
        let mut fold_times = Vec::with_capacity(args.epochs);
        for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
            let t = Instant::now();
            prover.fold_epoch(next, sigma).expect("fold");
            fold_times.push(t.elapsed());
        }

        // ---- verification: the auditor's cost, once, for everything ----
        let latest = epochs.last().expect("non-empty");
        // The folding state verifies directly, with no compression
        // step. This is the auditor's cost if the prover ships the
        // `RecursiveSNARK` itself rather than a compressed proof --
        // fast to check, but a 69 MB download.
        let t = Instant::now();
        verify_grouped_folding_proofs(
            &ivc,
            &prover.proofs().expect("proofs"),
            prover.num_steps(),
            &epochs[0],
            latest,
        )
        .expect("IVC audit verifies");
        let recursive_verify = t.elapsed();

        let recursive_bytes = prover.folding_state_bytes();

        // ---- compression: what actually gets published ----
        let (cpk, cvk) = ivc.compression_keys().expect("compression keys");
        let t = Instant::now();
        let compressed = prover.compress_all(&cpk).expect("compress");
        let compress_time = t.elapsed();
        let compressed_bytes = compressed.size_bytes();

        let t = Instant::now();
        verify_grouped_ivc_audit(
            &ivc,
            &cvk,
            &compressed,
            prover.num_steps(),
            &epochs[0],
            latest,
        )
        .expect("compressed audit verifies");
        let compressed_verify = t.elapsed();

        // ---- the classic per-epoch auditor, on identical data ----
        let vctx = VerifierContext::<Bn254, Pcs>::new(args.num_vars, vk.clone())
            .with_audit_fs(akd::aegon::ivc::adapter::poseidon_audit_fs());
        let sctx = ShardedVerifierContext::new(vctx, n.trailing_zeros() as usize)
            .with_chain_groups(groups);
        let published: Vec<_> = epochs
            .iter()
            .enumerate()
            .map(|(i, shards)| {
                as_sharded_epoch_commitment(
                    i as u64,
                    shards,
                    if i == 0 { None } else { Some(&sigmas[i - 1]) },
                    args.num_vars,
                )
            })
            .collect();
        let mut classic_times = Vec::with_capacity(args.epochs);
        let mut audit_state = ShardedAuditState::<Fr>::with_groups(groups);
        for w in published.windows(2) {
            let t = Instant::now();
            let ok = verify_sharded_invariance(&sctx, &mut audit_state, &w[0], &w[1])
                .expect("classic audit");
            classic_times.push(t.elapsed());
            assert!(ok, "synthetic chain must pass the classic audit too");
        }

        // ---- the classic *server* cost: producing one audit object ----
        let mut r2 = rng(0xA0D17 ^ n as u64);
        let base = &epochs[0];
        let new_index: Vec<G1Affine> = epochs[1].iter().map(|s| s.index).collect();
        let new_value: Vec<G1Affine> = epochs[1].iter().map(|s| s.value).collect();
        let blinds: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut r2)).collect();
        let nonces: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut r2)).collect();
        let mut server_times = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            let out = produce_audit_object(
                FsParams { num_vars: args.num_vars, n_shards: n },
                h,
                base,
                &new_index,
                &new_value,
                &blinds,
                &nonces,
            );
            server_times.push(t.elapsed());
            std::hint::black_box(out);
        }
        let server_classic = mean(&server_times);

        // Bytes the auditor must download for one epoch's commitment tuple.
        let tuple_bytes = as_sharded_epoch_commitment(
            args.epochs as u64,
            latest,
            Some(sigmas.last().expect("non-empty")),
            args.num_vars,
        )
        .compressed_size();

        let fold = mean(&fold_times);
        let classic = mean(&classic_times);
        println!(
            "{n:>7} {constraints:>12} {:>11} {:>11} {:>11} {:>11} {:>10} {:>12} {:>10}",
            format!("{:.2?}", setup_time),
            format!("{:.2?}", fold),
            format!("{:.1} MB", recursive_bytes as f64 / (1024.0 * 1024.0)),
            format!("{:.2?}", compress_time),
            format!("{:.1} KB", compressed_bytes as f64 / 1024.0),
            format!("{:.2?}", compressed_verify),
            format!("{:.2?}", classic),
        );

        println!(
            "\n  --- what each party pays, at {n} shards ---\n\
             \x20 {:>34} {:>14} {:>14} {:>14}\n\
             \x20 {}\n\
             \x20 {:>34} {:>14} {:>14} {:>14}\n\
             \x20 {:>34} {:>14} {:>14} {:>14}\n\
             \x20 {:>34} {:>14} {:>14} {:>14}",
            "scenario", "server/epoch", "client audit", "client bytes",
            "-".repeat(78),
            "1. classic (no IVC)",
            format!("{:.2?}", server_classic),
            format!("{:.2?}/ep", classic),
            format!("{:.1} KB/ep", tuple_bytes as f64 / 1024.0),
            "2. IVC, fold only",
            format!("{:.2?}", fold),
            format!("{:.2?}", recursive_verify),
            format!("{:.1} MB once", (recursive_bytes + tuple_bytes) as f64 / (1024.0 * 1024.0)),
            "3. IVC + compression",
            format!("{:.2?}+{:.2?}", fold, compress_time),
            format!("{:.2?}", compressed_verify),
            format!("{:.1} KB once", (compressed_bytes + tuple_bytes) as f64 / 1024.0),
        );
    }

    println!(
        "\n`classic/ep` is what the existing auditor pays for EVERY epoch, having been online for\n\
         each one and having downloaded each one\'s commitments. `verify` is what the IVC auditor\n\
         pays ONCE, for the whole chain, however long it is and whether or not it was ever online.\n\
         Both `pub proof` and `verify` are independent of the number of epochs folded -- that\n\
         independence, not the constant, is the point.\n\
         `fold proof` is the prover\'s working state (it carries the running R1CS witnesses) and is\n\
         not published; `pub proof` is the compressed proof that is."
    );
}

/// Fold one epoch at a time, sizing the proof after each.
///
/// The point of the IVC path is that an auditor's cost does not grow
/// with how long it has been away. That is a claim about *shape*, so
/// it is worth measuring rather than asserting: if either proof grew
/// per epoch, the whole design would be pointless.
///
/// Also reports what the auditor must fetch alongside the proof — the
/// epoch's per-shard commitment tuple, which it needs to bind the
/// proof to a concrete epoch (Poseidon digest) and to the bulletin
/// board (SHA256 Merkle root). That is a fixed cost paid once,
/// against the classic auditor paying it every epoch.
fn size_sweep(
    args: &Args,
    h: ark_bn254::G1Affine,
    _vk: &akd_core::aegon_crypto::pcs::kzhk::srs::KZHKVerifierParam<Bn254>,
) {
    let n: usize = args
        .shards
        .split(',')
        .next()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(2);
    let groups = single_group_count(args);
    let plan = GroupPlan::new(n, groups).expect("--groups must divide --shards");
    let mut r = rng(0x51_2E);
    let (epochs, sigmas) = honest_grouped_chain(plan, args.num_vars, h, args.epochs, &mut r);

    let ivc = GroupedIvcAuditParams::setup(plan, args.num_vars, h).expect("IVC setup");
    let (cpk, cvk) = ivc.compression_keys().expect("compression keys");
    let mut prover = GroupedIvcAuditProver::new(&ivc, &epochs[0]).expect("prover init");

    // What the auditor downloads alongside the proof, per epoch, in
    // the classic scheme -- and exactly once, in the IVC scheme.
    let tuple_bytes = as_sharded_epoch_commitment(1, &epochs[1], Some(&sigmas[0]), args.num_vars)
        .compressed_size();

    println!(
        "proof size vs epochs folded  (shards={n}, groups={groups}, num_vars={})\n",
        args.num_vars
    );
    println!(
        "{:>7} {:>10} {:>14} {:>12} {:>14}",
        "epochs", "fold", "folding state", "published", "verify"
    );
    println!("{}", "-".repeat(64));

    for (i, (next, sigma)) in epochs[1..].iter().zip(&sigmas).enumerate() {
        let t_fold = Instant::now();
        prover.fold_epoch(next, sigma).expect("fold");
        let fold_t = t_fold.elapsed();
        let steps = prover.num_steps();
        let state_bytes = prover.folding_state_bytes();
        let compressed = prover.compress_all(&cpk).expect("compress");
        let t = Instant::now();
        verify_grouped_ivc_audit(&ivc, &cvk, &compressed, steps, &epochs[0], next)
            .expect("verifies");
        let vt = t.elapsed();
        println!(
            "{:>7} {:>10} {:>14} {:>12} {:>14}",
            i + 1,
            format!("{:.2?}", fold_t),
            format!("{:.1} MB", state_bytes as f64 / (1024.0 * 1024.0)),
            format!("{:.2} KB", compressed.size_bytes() as f64 / 1024.0),
            format!("{:.2?}", vt),
        );
    }

    println!(
        "\nPer-epoch commitment tuple: {:.1} KB.\n\
         The classic auditor downloads that EVERY epoch and verifies each one; the IVC auditor\n\
         downloads it ONCE, for whichever epoch it is checking, alongside a proof whose size does\n\
         not move.",
        tuple_bytes as f64 / 1024.0
    );
}

/// Sweep the number of independent folding groups at a fixed shard
/// count.
///
/// Group-sharding derives each group's Fiat-Shamir chain scalars from
/// that group's commitments alone, so the groups never interact and
/// every phase parallelises. Two numbers matter and they are not the
/// same:
///
/// * **per host** -- what one machine folding one group pays. In a
///   deployment that already runs one process per shard, this is the
///   wall-clock, because the groups run concurrently on separate
///   hosts.
/// * **one box** -- what all `G` groups cost run together on this
///   machine. Close to flat, because Nova already saturates the cores
///   with rayon-parallel MSMs; `G` chains just share them.
///
/// Reporting only the first would overstate the win on a single
/// machine; reporting only the second would hide it entirely for the
/// deployment the design targets.
fn group_sweep(args: &Args, h: G1Affine) {
    let group_counts: Vec<usize> = group_counts(args);
    let n: usize = args
        .shards
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .next()
        .expect("--shards must name a shard count");

    println!("group-sharded audit  (shards={n}, epochs={}, num_vars={})\n", args.epochs, args.num_vars);
    println!(
        "{:>7} {:>8} {:>12} {:>9} {:>13} {:>13} {:>13} {:>13} {:>11} {:>10}",
        "groups", "shards/g", "constraints", "setup",
        "fold/host", "fold/1box", "compr/host", "compr/1box",
        "pub proof", "verify",
    );
    println!("{}", "-".repeat(122));

    for &g in &group_counts {
        let plan = match GroupPlan::new(n, g) {
            Ok(p) => p,
            Err(e) => {
                println!("{g:>7}  skipped: {e}");
                continue;
            },
        };
        let mut r = rng(0x9C0DE ^ (n as u64) ^ ((g as u64) << 32));
        let (epochs, sigmas) = honest_grouped_chain(plan, args.num_vars, h, args.epochs, &mut r);

        // Setup is paid ONCE regardless of `g`: every group folds the
        // same circuit shape, so they share one PublicParams.
        let t = Instant::now();
        let gp = GroupedIvcAuditParams::setup(plan, args.num_vars, h).expect("grouped setup");
        let setup = t.elapsed();
        let constraints = gp.constraints_per_step();

        // ---- what all `g` groups cost together on this machine ----
        let mut prover = GroupedIvcAuditProver::new(&gp, &epochs[0]).expect("prover");
        let mut folds = Vec::new();
        for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
            let t = Instant::now();
            prover.fold_epoch(next, sigma).expect("fold");
            folds.push(t.elapsed());
        }
        let fold_1box = mean(&folds);

        // ---- what ONE host folding ONE group pays ----
        //
        // Measured on a plain single-chain prover over group 0's
        // shards, i.e. with the whole machine to itself -- which is
        // the situation a dedicated host is actually in.
        let g0: Vec<_> = epochs.iter().map(|e| e[plan.range(0)].to_vec()).collect();
        let s0: Vec<_> = sigmas.iter().map(|s| s[plan.range(0)].to_vec()).collect();
        let mut solo = IvcAuditProver::new(gp.inner().clone(), &g0[0]).expect("solo prover");
        let mut solo_folds = Vec::new();
        for (next, sigma) in g0[1..].iter().zip(&s0) {
            let t = Instant::now();
            solo.fold_epoch(next, sigma).expect("solo fold");
            solo_folds.push(t.elapsed());
        }
        let fold_host = mean(&solo_folds);

        let (pk, vk) = gp.compression_keys().expect("compression keys");

        let t = Instant::now();
        let solo_compressed = compress(gp.inner(), &pk, solo.proof().expect("proof"))
            .expect("solo compress");
        let compress_host = t.elapsed();
        let per_group_bytes = compressed_proof_size_bytes(&solo_compressed);

        let t = Instant::now();
        let published = prover.compress_all(&pk).expect("compress all");
        let compress_1box = t.elapsed();

        let latest = epochs.last().expect("non-empty");
        let t = Instant::now();
        verify_grouped_ivc_audit(&gp, &vk, &published, prover.num_steps(), &epochs[0], latest)
            .expect("grouped audit verifies");
        let verify = t.elapsed();

        println!(
            "{g:>7} {:>8} {constraints:>12} {:>9} {:>13} {:>13} {:>13} {:>13} {:>11} {:>10}",
            plan.shards_per_group(),
            format!("{:.2?}", setup),
            format!("{:.2?}", fold_host),
            format!("{:.2?}", fold_1box),
            format!("{:.2?}", compress_host),
            format!("{:.2?}", compress_1box),
            format!("{:.1} KB", published.size_bytes() as f64 / 1024.0),
            format!("{:.2?}", verify),
        );
        let _ = per_group_bytes;
    }

    println!(
        "\n`fold/host` and `compr/host` are what ONE machine folding ONE group pays, with the box to\n\
         itself -- the wall-clock in a deployment that runs each group on its own host. `/1box` is\n\
         all {} groups run together here, which is near-flat because Nova already saturates the\n\
         cores. `setup` is paid once however many groups there are: all groups fold the same shape.\n\
         `pub proof` is the TOTAL an auditor downloads -- one ~11 KB proof per group -- and `verify`\n\
         checks every group.",
        "G",
    );
}

/// Spartan vs MicroNova as the terminal SNARK.
///
/// MicroNova (`nova_snark::spartan::ppsnark`) is a *preprocessing*
/// Spartan: the verifier holds a commitment to the R1CS matrices
/// instead of re-deriving them, which removes a term from the
/// verifier's work. Its own docs qualify the benefit -- it pays off
/// "when using a polynomial commitment scheme in which the verifier's
/// costs is succinct."
///
/// Our primary instance is on Grumpkin, which is not pairing-friendly,
/// so its PCS is IPA and the IPA verifier is linear in the committed
/// vector. Preprocessing cannot remove that term. This mode measures
/// how much is left over once it is the dominant one.
fn snark_compare(args: &Args, h: G1Affine) {
    let shard_counts: Vec<usize> = args
        .shards
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("terminal SNARK: Spartan vs MicroNova  (num_vars={})\n", args.num_vars);
    println!(
        "{:>7} {:>12}   {:>9} {:>9} {:>9} {:>9}   {:>9} {:>9} {:>9} {:>9}",
        "shards", "constraints",
        "sp setup", "sp prove", "sp size", "sp VERIFY",
        "mn setup", "mn prove", "mn size", "mn VERIFY",
    );
    println!("{}", "-".repeat(112));

    for &n in &shard_counts {
        let p = FsParams { num_vars: args.num_vars, n_shards: n };
        let mut r = rng(0x5A17A ^ n as u64);
        let (epochs, sigmas) = honest_chain(p, h, args.epochs, &mut r);

        let latest = epochs.last().expect("non-empty");

        // The two terminal SNARKs need differently-sized commitment
        // keys, so each gets its own parameters and its own fold.
        let ivc = Arc::new(IvcAuditParams::setup(p, h).expect("IVC setup"));
        let mut prover = IvcAuditProver::new(ivc.clone(), &genesis(n)).expect("prover init");
        for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
            prover.fold_epoch(next, sigma).expect("fold");
        }
        let folded = prover.proof().expect("proof");
        let steps = prover.num_steps();

        // ---- Spartan (what we ship today) ----
        let t = Instant::now();
        let (pk, vk) = ivc.compression_keys().expect("spartan keys");
        let sp_setup = t.elapsed();
        let t = Instant::now();
        let sp_proof = compress(&ivc, &pk, folded).expect("spartan prove");
        let sp_prove = t.elapsed();
        let sp_size = compressed_proof_size_bytes(&sp_proof);
        let t = Instant::now();
        verify_compressed_ivc_audit(&ivc, &vk, &sp_proof, steps, prover.z0(), latest)
            .expect("spartan verify");
        let sp_verify = t.elapsed();

        // ---- MicroNova ----
        let ivc2 = Arc::new(
            IvcAuditParams::setup_with(p, h, TerminalSnark::MicroNova).expect("micronova params"),
        );
        let mut prover2 = IvcAuditProver::new(ivc2.clone(), &genesis(n)).expect("prover init");
        for (next, sigma) in epochs[1..].iter().zip(&sigmas) {
            prover2.fold_epoch(next, sigma).expect("fold");
        }
        let folded2 = prover2.proof().expect("proof");
        let t = Instant::now();
        let (pk2, vk2) = ivc2.compression_keys_pp().expect("micronova keys");
        let mn_setup = t.elapsed();
        let t = Instant::now();
        let mn_proof = compress_pp(&ivc2, &pk2, folded2).expect("micronova prove");
        let mn_prove = t.elapsed();
        let mn_size = compressed_pp_proof_size_bytes(&mn_proof);
        let t = Instant::now();
        let z = mn_proof
            .verify(&vk2, steps, prover2.z0())
            .expect("micronova verify");
        let mn_verify = t.elapsed();
        assert_eq!(
            z[0],
            <akd::aegon::ivc::prover::E1 as nova_snark::traits::Engine>::Scalar::from(steps as u64),
            "MicroNova must certify the same epoch count"
        );

        println!(
            "{n:>7} {:>12}   {:>9} {:>9} {:>9} {:>9}   {:>9} {:>9} {:>9} {:>9}",
            ivc.constraints_per_step(),
            format!("{:.2?}", sp_setup),
            format!("{:.2?}", sp_prove),
            format!("{:.1} KB", sp_size as f64 / 1024.0),
            format!("{:.2?}", sp_verify),
            format!("{:.2?}", mn_setup),
            format!("{:.2?}", mn_prove),
            format!("{:.1} KB", mn_size as f64 / 1024.0),
            format!("{:.2?}", mn_verify),
        );
    }

    println!(
        "\nMicroNova preprocesses the R1CS matrices so the verifier never touches them. What it\n\
         cannot remove is the polynomial-commitment opening check, and ours is IPA -- linear in\n\
         the circuit -- because the primary curve is Grumpkin, which has no pairing. Watch whether\n\
         `mn VERIFY` is flat in `shards` or still tracks it."
    );
}

/// What the Poseidon audit-FS costs, measured against the SHA256
/// derivation it replaces, on one machine.
///
/// The IVC path needs its Fiat-Shamir recomputed inside the folding
/// circuit, and SHA256 there would cost ~4M constraints per epoch. So
/// the audit-path derivations moved to Poseidon. That is a change to
/// the *existing* auditor too, and its cost belongs in the paper next
/// to the IVC numbers rather than buried.
///
/// Per epoch the audit performs two chain-scalar derivations (index
/// and value, each absorbing every shard's commitment) and one sigma
/// challenge per shard. Everything else about the audit -- the group
/// arithmetic, the SHA256 Merkle root -- is untouched by the switch,
/// so timing just these isolates the delta.
fn fs_compare(args: &Args) {
    use akd::aegon::audit_fs::AuditFsHooks;
    use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKCommitment;

    let n: usize = args
        .shards
        .split(',')
        .next()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(128);

    let mut r = rng(0xF5_C0FF);
    let commits: Vec<KZHKCommitment<Bn254>> = (0..n)
        .map(|_| KZHKCommitment::new(rand_point(&mut r), args.num_vars))
        .collect();
    let one = KZHKCommitment::new(rand_point(&mut r), args.num_vars);
    let prev = Fr::rand(&mut r);

    let sha = AuditFsHooks::<Bn254, Pcs>::sha256();
    let pos = akd::aegon::ivc::adapter::poseidon_audit_fs();

    // Enough repetitions that a single epoch's worth of work is well
    // above timer resolution.
    const REPS: usize = 20;

    // Warm up. The Poseidon constants are generated once per process
    // behind a `OnceLock`, and that generation is expensive enough
    // that amortising it over REPS shows up as a fixed offset of
    // several milliseconds -- which would be charged to the FS
    // derivation it is not part of. A real auditor pays it once at
    // startup, never per epoch.
    for _ in 0..3 {
        std::hint::black_box(pos.chain_scalar(b"aegon.sharded.fs.r_index", prev, &commits));
        std::hint::black_box(pos.sigma_challenge(&one, &one, &one, &one, prev, &one));
        std::hint::black_box(sha.chain_scalar(b"aegon.sharded.fs.r_index", prev, &commits));
        std::hint::black_box(sha.sigma_challenge(&one, &one, &one, &one, prev, &one));
    }

    let bench_chain = |hooks: &AuditFsHooks<Bn254, Pcs>| {
        let t = Instant::now();
        for _ in 0..REPS {
            std::hint::black_box(hooks.chain_scalar(
                b"aegon.sharded.fs.r_index",
                prev,
                &commits,
            ));
            std::hint::black_box(hooks.chain_scalar(
                b"aegon.sharded.fs.r_value",
                prev,
                &commits,
            ));
        }
        t.elapsed() / REPS as u32
    };

    let bench_sigma = |hooks: &AuditFsHooks<Bn254, Pcs>| {
        let t = Instant::now();
        for _ in 0..REPS {
            for c in &commits {
                std::hint::black_box(hooks.sigma_challenge(c, c, &one, &one, prev, &one));
            }
        }
        t.elapsed() / REPS as u32
    };

    let sha_chain = bench_chain(&sha);
    let pos_chain = bench_chain(&pos);
    let sha_sigma = bench_sigma(&sha);
    let pos_sigma = bench_sigma(&pos);

    let sha_total = sha_chain + sha_sigma;
    let pos_total = pos_chain + pos_sigma;

    println!("audit-path Fiat-Shamir: SHA256 vs Poseidon  (shards={n}, num_vars={})\n", args.num_vars);
    println!("{:>34} {:>13} {:>13} {:>10}", "per epoch", "SHA256", "Poseidon", "ratio");
    println!("{}", "-".repeat(74));
    let row = |label: &str, a: Duration, b: Duration| {
        println!(
            "{:>34} {:>13} {:>13} {:>10}",
            label,
            format!("{:.3?}", a),
            format!("{:.3?}", b),
            format!("{:.2}x", b.as_secs_f64() / a.as_secs_f64().max(1e-12)),
        );
    };
    row("2 chain scalars (all shards)", sha_chain, pos_chain);
    row(&format!("{n} sigma challenges"), sha_sigma, pos_sigma);
    row("TOTAL FS work per epoch", sha_total, pos_total);

    println!(
        "\nThis is the whole delta the Poseidon switch introduces for the classic auditor:\n\
         everything else it does -- the commitment-homomorphism group arithmetic and the\n\
         SHA256 Merkle root -- is byte-for-byte unchanged. Subtract the SHA256 column from\n\
         `classic/ep` and add the Poseidon one to convert between the two configurations."
    );
}
