// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! End-to-end IVC auditing against a **real** `ShardedAegon` publish.
//!
//! The unit tests in `aegon::ivc` synthesise epoch transitions from
//! pure group arithmetic, which exercises every constraint but takes
//! the server's word for what an honest transition looks like. This
//! example closes that gap: it stands up an actual sharded directory
//! with a zk SRS, publishes real batches, and folds the commitments
//! the server genuinely produced.
//!
//! What it demonstrates:
//!
//! 1. A deployment configured with the Poseidon audit-FS bundle
//!    publishes normally and still passes the *classic* per-epoch
//!    `verify_sharded_invariance`.
//! 2. The same epoch chain folds into a single Nova proof.
//! 3. One `verify` call covers every epoch — the auditor never has to
//!    have been online, which is the whole point (paper §3,
//!    "fast-forwarding via recursive proofs").
//! 4. Tampering is caught.
//!
//! Run with:
//! ```text
//! cargo run --release -p akd --features ivc_audit --example ivc_audit_e2e
//! ```
//!
//! It lives as an example rather than a `cargo test` for the same
//! reason `audit_sigma_e2e` does: the crate's `--lib --tests` path has
//! a pre-existing, unrelated breakage in `tests/test_errors.rs`
//! (reproduces on `main`), so integration coverage that must actually
//! run lives here.

use std::sync::Arc;
use std::time::Instant;

use akd::aegon::ivc::adapter::{
    epoch_commitments, epoch_sigma_witnesses, fs_params_from_epoch, poseidon_audit_fs,
};
use akd::aegon::ivc::prover::{IvcAuditParams, IvcAuditProver};
use akd::aegon::ivc::verifier::{verify_against_merkle_root, verify_ivc_audit};
use akd::aegon::{
    merkle_root, verify_sharded_invariance, DbSource, EcVrfHash, ShardTransport, ShardedAegon,
    ShardedAegonConfig, ShardedAuditState, VrfProver, BENCH_VRF_SEED,
};
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_bn254::{Bn254, Fr};
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Pcs = KZHK<Bn254>;

const LOG_CAPACITY: usize = 12;
const LOG_N_SHARDS: usize = 1;
const EPOCHS: usize = 3;

fn main() {
    println!("── IVC audit against a real ShardedAegon publish ──");

    // A hiding SRS: the value chain then carries a real blinding
    // shift per epoch and a genuine Schnorr proof, which is the
    // interesting case for the circuit.
    let cfg: ShardedAegonConfig<Bn254, Pcs> = ShardedAegonConfig::<Bn254, Pcs>::builder()
        .shard_log_capacity(LOG_CAPACITY - LOG_N_SHARDS)
        .log_n_shards(LOG_N_SHARDS)
        .private(true)
        .kzh_k(2)
        .shards(ShardTransport::InProcess)
        .db(DbSource::None)
        // Both the server and every verifier must agree on this.
        .audit_fs(poseidon_audit_fs())
        .build()
        .expect("config builds");

    let mut rng = ChaCha20Rng::seed_from_u64(0x1_5C_1);
    let mut server: ShardedAegon<Bn254, Pcs, EcVrfHash> =
        ShardedAegon::<Bn254, Pcs, EcVrfHash>::setup(&mut rng, &cfg).expect("setup");
    server.set_vrf_prover(VrfProver::from_seed(&BENCH_VRF_SEED));

    let ctx = server.sharded_verifier_context();
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
        let next = server.publish_two_layer(&updates).expect("publish");
        chain.push(next);
    }
    println!(
        "  published {EPOCHS} epochs across {} shards.",
        1 << LOG_N_SHARDS
    );

    // ---- (1) the classic per-epoch audit still accepts -----------------
    //
    // Confirms the Poseidon FS bundle is wired consistently through
    // the publish path: the server derived its chain scalars and
    // Schnorr challenges with it, and the auditor recomputes them the
    // same way.
    {
        let mut audit_state = ShardedAuditState::<Fr>::default();
        for w in chain.windows(2) {
            let ok = verify_sharded_invariance(&ctx, &mut audit_state, &w[0], &w[1])
                .expect("classic audit");
            assert!(ok, "honest transition must pass the classic audit");
        }
    }
    println!("  ✓ classic per-epoch verify_sharded_invariance accepts the whole chain.");

    // ---- (2) fold the same chain into one recursive proof --------------
    let fs_params = fs_params_from_epoch(&chain[1]).expect("fs params");
    println!(
        "  IVC shape: num_vars={} n_shards={}",
        fs_params.num_vars, fs_params.n_shards
    );

    let t = Instant::now();
    let ivc = Arc::new(IvcAuditParams::setup(fs_params, h).expect("IVC setup"));
    println!(
        "  Nova setup: {:.2?}  ({} constraints/step)",
        t.elapsed(),
        ivc.constraints_per_step()
    );

    let mut prover =
        IvcAuditProver::new(ivc.clone(), &epoch_commitments(&genesis)).expect("prover init");
    for (i, next) in chain[1..].iter().enumerate() {
        let shards = epoch_commitments(next);
        let sigmas = epoch_sigma_witnesses(next).expect("every shard carries a sigma proof");
        let t = Instant::now();
        prover.fold_epoch(&shards, &sigmas).expect("fold epoch");
        println!("  folded epoch {} in {:.2?}", i + 1, t.elapsed());
    }
    assert_eq!(prover.num_steps(), EPOCHS);

    // ---- (3) one verification covers every epoch -----------------------
    let latest = chain.last().expect("non-empty");
    let latest_shards = epoch_commitments(latest);
    let t = Instant::now();
    let verified = verify_against_merkle_root(
        &ivc,
        prover.proof().expect("proof"),
        prover.num_steps(),
        prover.z0(),
        &latest_shards,
        latest.merkle_root,
        || merkle_root(&latest.per_shard),
    )
    .expect("IVC audit verifies");
    println!(
        "  ✓ ONE verify covered {} epochs in {:.2?} (and matched the published Merkle root).",
        verified.epochs,
        t.elapsed()
    );

    // ---- (4) tampering is caught ---------------------------------------
    {
        let mut tampered = latest_shards.clone();
        tampered[0].rand_value = tampered[0].index;
        assert!(
            verify_ivc_audit(
                &ivc,
                prover.proof().expect("proof"),
                prover.num_steps(),
                prover.z0(),
                &tampered,
            )
            .is_err(),
            "tampered epoch commitments must not verify"
        );
    }
    println!("  ✓ tampered epoch commitments are rejected.");

    // A stale proof served next to fresh commitments must fail too.
    {
        let earlier = epoch_commitments(&chain[EPOCHS - 1]);
        assert!(
            verify_ivc_audit(
                &ivc,
                prover.proof().expect("proof"),
                prover.num_steps(),
                prover.z0(),
                &earlier,
            )
            .is_err(),
            "proof must be bound to the epoch it describes"
        );
    }
    println!("  ✓ the proof is bound to the epoch it describes.");

    println!("\nall IVC audit checks passed.");
}
