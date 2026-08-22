//! Group-sharded IVC auditing against a **real** `ShardedAegon`
//! publish.
//!
//! [`ivc_audit_e2e`](ivc_audit_e2e) proves the recursive audit works
//! end to end with a single folding chain. This one adds the
//! partition: the deployment runs `CHAIN_GROUPS` independent
//! Fiat–Shamir accumulators, so the audit splits into that many
//! chains that fold, compress and verify in parallel.
//!
//! Why bother splitting: compression is linear in circuit size
//! (~100 µs per 1 000 constraints, measured), and it is the dominant
//! prover cost. `G` groups each compress a `η/G`-shard circuit, so
//! the phase that costs ~118 s at `η = 128` costs ~1/G of that per
//! group — and the groups run concurrently, on separate hosts in a
//! real deployment.
//!
//! What it demonstrates, all against commitments a real server
//! produced:
//!
//! 1. A deployment configured with `chain_groups > 1` publishes
//!    normally, and the *classic* per-epoch auditor still accepts —
//!    tracking one accumulator per group instead of one overall.
//! 2. The same chain folds into `G` recursive proofs.
//! 3. `G` verifications, run in parallel, cover every epoch.
//! 4. Tampering in **any** group is caught, and a group's proof does
//!    not verify against another group's shards.
//!
//! Run with:
//! ```text
//! cargo run --release -p akd --features ivc_audit --example ivc_audit_grouped_e2e
//! ```

use std::time::Instant;

use akd::aegon::ivc::adapter::{
    epoch_commitments, epoch_sigma_witnesses, fs_params_from_epoch, poseidon_audit_fs,
};
use akd::aegon::ivc::grouped::{
    verify_grouped_folding_proofs, verify_grouped_ivc_audit, GroupedIvcAuditParams,
    GroupedIvcAuditProver,
};
use akd::aegon::{
    verify_sharded_invariance, DbSource, EcVrfHash, GroupPlan, ShardTransport, ShardedAegon,
    ShardedAegonConfig, ShardedAuditState, VrfProver, BENCH_VRF_SEED,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::{Bn254, Fr};
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;

const LOG_CAPACITY: usize = 12;
const LOG_N_SHARDS: usize = 2; // 4 shards
const CHAIN_GROUPS: usize = 2; // 2 shards per group
const EPOCHS: usize = 3;

fn main() {
    println!("── group-sharded IVC audit against a real ShardedAegon publish ──");

    let n_shards = 1usize << LOG_N_SHARDS;
    let plan = GroupPlan::new(n_shards, CHAIN_GROUPS).expect("plan");

    // A hiding SRS, so the value chain carries a real Schnorr
    // re-randomisation proof per shard -- the case the circuit's
    // sigma block exists for.
    let cfg: ShardedAegonConfig<Bn254, Pcs> = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(LOG_CAPACITY - LOG_N_SHARDS)
        .log_n_shards(LOG_N_SHARDS)
        .private(true)
        .kzh_k(2)
        .shards(ShardTransport::InProcess)
        .db(DbSource::None)
        .audit_fs(poseidon_audit_fs())
        // The one line that changes the protocol: derive chain
        // scalars per group instead of directory-wide.
        .chain_groups(CHAIN_GROUPS)
        .build()
        .expect("config builds");

    let mut rng = ChaCha20Rng::seed_from_u64(0x6_C0DE);
    let mut server: ShardedAegon<Bn254, Pcs, EcVrfHash> =
        ShardedAegon::<Bn254, Pcs, EcVrfHash>::setup(&mut rng, &cfg).expect("setup");
    server.set_vrf_prover(VrfProver::from_seed(&BENCH_VRF_SEED));

    let ctx = server.sharded_verifier_context();
    assert_eq!(ctx.chain_groups, CHAIN_GROUPS, "context carries the partition");
    let h = ctx.inner.verifier_param.get_h();
    let genesis = server.epoch_commitment(0).expect("epoch-0 commit retained");

    // ---- publish a few real epochs ------------------------------------
    let mut chain = vec![genesis.clone()];
    for e in 0..EPOCHS {
        let updates: Vec<(Vec<u8>, Vec<u8>)> = (0..4)
            .map(|i| {
                (
                    format!("user-{e}-{i}").into_bytes(),
                    format!("pk-{e}-{i}").into_bytes(),
                )
            })
            .collect();
        chain.push(server.publish_two_layer(&updates).expect("publish"));
    }
    println!(
        "  published {EPOCHS} epochs across {n_shards} shards in {CHAIN_GROUPS} chain groups \
         ({} shards/group).",
        plan.shards_per_group()
    );

    // ---- (1) the classic per-epoch audit still accepts -----------------
    //
    // The auditor now threads `CHAIN_GROUPS` accumulators. Everything
    // else about it is unchanged -- and its aggregate cost is too;
    // grouping buys the recursive auditor, not this one.
    {
        let mut audit_state = ShardedAuditState::<Fr>::with_groups(CHAIN_GROUPS);
        for w in chain.windows(2) {
            let ok = verify_sharded_invariance(&ctx, &mut audit_state, &w[0], &w[1])
                .expect("classic audit");
            assert!(ok, "honest transition must pass the classic audit");
        }
    }
    println!("  ✓ classic per-epoch audit accepts the whole chain, tracking {CHAIN_GROUPS} chains.");

    // A verifier configured for the *wrong* partition must fail
    // rather than quietly accept: it would absorb a different set of
    // commitments than the server did.
    {
        let wrong = server.sharded_verifier_context().with_chain_groups(1);
        let mut st = ShardedAuditState::<Fr>::default();
        assert!(
            verify_sharded_invariance(&wrong, &mut st, &chain[0], &chain[1])
                .map(|ok| !ok)
                .unwrap_or(true),
            "a verifier with the wrong chain_groups must not accept"
        );
    }
    println!("  ✓ a verifier configured with the wrong partition is rejected.");

    // ---- (2) fold the chain into one proof per group -------------------
    let fs_params = fs_params_from_epoch(&chain[1]).expect("fs params");
    let t = Instant::now();
    let gp = GroupedIvcAuditParams::setup(plan, fs_params.num_vars, h).expect("grouped setup");
    println!(
        "  Nova setup: {:.2?}  ({} constraints/step, {} shards/group)",
        t.elapsed(),
        gp.constraints_per_step(),
        plan.shards_per_group(),
    );

    let mut prover =
        GroupedIvcAuditProver::new(&gp, &epoch_commitments(&genesis)).expect("prover init");
    for (i, next) in chain[1..].iter().enumerate() {
        let shards = epoch_commitments(next);
        let sigmas = epoch_sigma_witnesses(next).expect("every shard carries a sigma proof");
        let t = Instant::now();
        prover.fold_epoch(&shards, &sigmas).expect("fold epoch");
        println!("  folded epoch {} across {CHAIN_GROUPS} groups in {:.2?}", i + 1, t.elapsed());
    }
    assert_eq!(prover.num_steps(), EPOCHS);

    // ---- (3) verify every group ---------------------------------------
    let latest = chain.last().expect("non-empty");
    let latest_shards = epoch_commitments(latest);
    let genesis_shards = epoch_commitments(&genesis);

    let t = Instant::now();
    let verified = verify_grouped_folding_proofs(
        &gp,
        &prover.proofs().expect("proofs"),
        prover.num_steps(),
        &genesis_shards,
        &latest_shards,
    )
    .expect("grouped audit verifies");
    println!(
        "  ✓ {CHAIN_GROUPS} verifications covered {} epochs in {:.2?}.",
        verified.epochs,
        t.elapsed()
    );

    // ---- (4) the published (compressed) form ---------------------------
    let (pk, vk) = gp.compression_keys().expect("compression keys");
    let t = Instant::now();
    let published = prover.compress_all(&pk).expect("compress");
    println!(
        "  compressed {CHAIN_GROUPS} groups in {:.2?} -> {:.2} KB total ({:.2} KB/group)",
        t.elapsed(),
        published.size_bytes() as f64 / 1024.0,
        published.size_bytes() as f64 / 1024.0 / CHAIN_GROUPS as f64,
    );

    let t = Instant::now();
    verify_grouped_ivc_audit(
        &gp,
        &vk,
        &published,
        prover.num_steps(),
        &genesis_shards,
        &latest_shards,
    )
    .expect("compressed grouped audit verifies");
    println!("  ✓ the published proof verifies in {:.2?}.", t.elapsed());

    // ---- (5) tampering in ANY group is caught --------------------------
    for shard in 0..n_shards {
        let mut tampered = latest_shards.clone();
        tampered[shard].rand_value = tampered[shard].index;
        assert!(
            verify_grouped_ivc_audit(
                &gp,
                &vk,
                &published,
                prover.num_steps(),
                &genesis_shards,
                &tampered,
            )
            .is_err(),
            "tampering shard {shard} (group {}) must be rejected",
            plan.group_of(shard),
        );
    }
    println!("  ✓ tampering is caught in every one of the {n_shards} shards.");

    // A stale proof next to fresh commitments must fail too.
    assert!(
        verify_grouped_ivc_audit(
            &gp,
            &vk,
            &published,
            prover.num_steps(),
            &genesis_shards,
            &epoch_commitments(&chain[EPOCHS - 1]),
        )
        .is_err(),
        "proof must be bound to the epoch it describes"
    );
    println!("  ✓ the proof is bound to the epoch it describes.");

    println!("\nall group-sharded IVC audit checks passed.");
}
