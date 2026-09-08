// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Configuration for an Aegon dictionary.
//!
//! Two bundles, one per role:
//!
//! * [`AegonConfig`] — what the *server* needs at setup time. Holds the
//!   dictionary size, the PCS-specific block layout (so insertions and
//!   openings agree on encoding), and the PCS's own configuration
//!   (e.g. `k` and the zero-knowledge flag for KZH-k).
//!
//! * [`VerifierContext`] — what *clients* and *auditors* need at
//!   verification time. Just the dictionary's `log_capacity` (drives
//!   the random eval point in the invariance check, and the probe
//!   count for index-consistency checks) and the PCS verifier key.
//!   Verifiers do not need `dims`: they never index the BTreeMap
//!   directly, only call `P::verify`.
//!
//! The server creates an `AegonConfig`, hands the resulting Aegon's
//! `verifier_context()` to clients/auditors, and that's all the wiring
//! either side needs.

use std::marker::PhantomData;

use ark_ec::pairing::Pairing;

use super::types::AegonPcs;

/// Over-provisioning factor `α`: the ratio between the underlying
/// polynomial's hypercube size (`2^shard_log_capacity`) and the
/// dictionary's *true* user-facing capacity. With `α = 4`, a shard's
/// polynomial holds 4× as many slots as the dictionary will ever
/// store entries (load factor ≤ 0.25) — open-addressing probe lengths
/// stay short at peak fill, which keeps the per-publish KZH-k opening
/// count bounded.
///
/// Standing regime sizes (all use 4× over-provisioning):
///
/// | regime | true dict   | total dict (= true × 4) | n_shards | per-shard log cap |
/// |--------|-------------|-------------------------|----------|-------------------|
/// | small  | 2²⁰ entries | 2²² slots               | 1        | 22                |
/// | medium | 2²⁶ entries | 2²⁸ slots               | 2        | 27                |
/// | large  | 2³² entries | 2³⁴ slots               | 128      | 27                |
///
/// In the two-layer routing model, `H_shard` spreads the total
/// `OVER_PROVISIONING_FACTOR × 2^true_log_capacity` slots across
/// `2^log_n_shards` shards, so each shard's polynomial has
/// `log_n_shards` fewer variables than the single-shard equivalent.
/// See [`shard_log_capacity_for_two_layer`] for the multi-shard form.
///
/// All sizing computations in the benches and the cluster scripts go
/// through this constant (or [`LOG2_OVER_PROVISIONING_FACTOR`]) so a
/// single edit retunes the whole system.
pub const OVER_PROVISIONING_FACTOR: usize = 4;

/// `log2(OVER_PROVISIONING_FACTOR)`. Defined separately as a
/// compile-time constant so we can do `shard_log_capacity =
/// true_log_capacity + LOG2_OVER_PROVISIONING_FACTOR` without runtime
/// `ilog2` calls. The `const _:` assertion below keeps the two in
/// sync at compile time.
pub const LOG2_OVER_PROVISIONING_FACTOR: usize = 2;

const _: () = {
    assert!(
        1usize << LOG2_OVER_PROVISIONING_FACTOR == OVER_PROVISIONING_FACTOR,
        "LOG2_OVER_PROVISIONING_FACTOR must equal log2(OVER_PROVISIONING_FACTOR)"
    );
};

/// Convert a dictionary's *true* (user-facing) log capacity to the
/// corresponding *shard* (over-provisioned, hypercube) log capacity:
/// `shard_log_capacity = true_log_capacity + LOG2_OVER_PROVISIONING_FACTOR`.
pub const fn shard_log_capacity_from_true(true_log_capacity: usize) -> usize {
    true_log_capacity + LOG2_OVER_PROVISIONING_FACTOR
}

/// Inverse of [`shard_log_capacity_from_true`]: derive the user-facing
/// log capacity from the over-provisioned shard hypercube size.
/// Saturates at 0 (a shard with `shard_log_capacity <
/// LOG2_OVER_PROVISIONING_FACTOR` is degenerate but we don't panic).
pub const fn true_log_capacity_from_shard(shard_log_capacity: usize) -> usize {
    shard_log_capacity.saturating_sub(LOG2_OVER_PROVISIONING_FACTOR)
}

/// Two-layer sizing rule: pick `shard_log_capacity` given the
/// dictionary's *true* log capacity and the number of shards
/// (`log_n_shards = log₂(N_shards)`).
///
/// ```text
/// shard_log_capacity = true_log_capacity + LOG2_OVER_PROVISIONING_FACTOR − log_n_shards
///                    = true_log_capacity + 2 − log_n_shards               // at OPF = 4
/// ```
///
/// This is the natural multi-shard extension of
/// [`shard_log_capacity_from_true`]: total polynomial slots stay at
/// `OVER_PROVISIONING_FACTOR × 2^true_log_capacity` (load factor
/// `1/OVER_PROVISIONING_FACTOR`), but the slots are spread across
/// `2^log_n_shards` shards instead of concentrated in one. Each
/// shard's polynomial therefore has `log_n_shards` fewer variables,
/// which is the only way to keep per-shard SRS and per-publish commit
/// work bounded as N grows.
///
/// Saturates at 1 — a `log_n_shards` larger than `true_log_capacity +
/// LOG2_OVER_PROVISIONING_FACTOR` would produce a degenerate
/// zero-variable shard polynomial; the builder rejects that anyway
/// (`shard_log_capacity ≥ 1`).
///
/// # Examples
///
/// Standing regimes (all OPF = 4):
///
/// ```text
/// shard_log_capacity_for_two_layer(20, 0) = 20 + 2 − 0 = 22   // small   (1 shard)
/// shard_log_capacity_for_two_layer(26, 1) = 26 + 2 − 1 = 27   // medium  (2 shards)
/// shard_log_capacity_for_two_layer(32, 7) = 32 + 2 − 7 = 27   // large   (128 shards)
/// ```
pub const fn shard_log_capacity_for_two_layer(
    true_log_capacity: usize,
    log_n_shards: usize,
) -> usize {
    let total = true_log_capacity + LOG2_OVER_PROVISIONING_FACTOR;
    let v = total.saturating_sub(log_n_shards);
    if v == 0 {
        1
    } else {
        v
    }
}

/// Server-side configuration. Built once, consumed by `Aegon::setup`
/// and `Aegon::init`.
///
/// Three knobs:
///
/// * `log_capacity` — dictionary size.
/// * `private` — whether the system should hide the contents of the
///   dictionary beyond what's necessary for soundness. At setup,
///   `Aegon::init` cross-checks this against the PCS's own
///   `PCSGlobalParam::is_zk()` so the user-level intent and the
///   actual PCS state cannot disagree.
/// * `pcs_config` — backend PCS's own config (e.g. `KZHKConfig{k, zk}`).
///
/// PCS-specific block layout is *not* a user choice; it is queried
/// from the prover param via
/// `PolynomialCommitmentScheme::block_dims` inside `init`.
///
/// Most users should construct configs through the `presets` module,
/// which keeps `private` and the backend-specific zk flag in sync
/// automatically. `AegonConfig::new` is a lower-level escape hatch
/// for backends without a preset.
#[derive(Clone, Debug)]
pub struct AegonConfig<E: Pairing, P: AegonPcs<E>> {
    /// `log_2(N)` where `N` is the maximum number of distinct slots
    /// the dictionary supports. Determines the number of variables of
    /// the underlying multilinear polynomials.
    pub log_capacity: usize,
    /// Whether the system runs in privacy-preserving mode. Translates
    /// to the backend PCS's own zk flag (e.g. `zk=true` for KZH-k).
    pub private: bool,
    /// Backend PCS's own configuration. For KZH-k this is
    /// `KZHKConfig { k, zk }`; for other PCSs it's whatever associated
    /// `Config` type they declare.
    pub pcs_config: P::Config,
    /// Audit-path Fiat-Shamir derivations. Defaults to SHA256; a
    /// deployment audited via IVC installs the Poseidon bundle here
    /// and on the verifier side alike.
    pub audit_fs: super::audit_fs::AuditFsHooks<E, P>,
    pub _e: PhantomData<E>,
}

impl<E: Pairing, P: AegonPcs<E>> AegonConfig<E, P> {
    /// Lower-level constructor — caller is responsible for ensuring
    /// `pcs_config` matches `private` (e.g. for KZH-k, that
    /// `pcs_config.zk == private`). Prefer `presets::*` constructors
    /// where available; they enforce this invariant by construction.
    pub fn new(log_capacity: usize, private: bool, pcs_config: P::Config) -> Self {
        assert!(log_capacity > 0, "log_capacity must be positive");
        Self {
            log_capacity,
            private,
            pcs_config,
            audit_fs: super::audit_fs::AuditFsHooks::sha256(),
            _e: PhantomData,
        }
    }

    /// Install a different audit-path Fiat-Shamir bundle.
    pub fn with_audit_fs(mut self, hooks: super::audit_fs::AuditFsHooks<E, P>) -> Self {
        self.audit_fs = hooks;
        self
    }

    /// Hypercube size — the number of slots a fully-loaded dictionary
    /// can hold before open-addressing starts colliding.
    ///
    /// Note: `2^log_capacity` is the *theoretical* capacity. In
    /// practice you want a load factor ≤ `1/OVER_PROVISIONING_FACTOR`,
    /// so plan for `2^log_capacity / OVER_PROVISIONING_FACTOR` actual
    /// entries before probe lengths degrade. See
    /// [`OVER_PROVISIONING_FACTOR`].
    pub fn dictionary_capacity(&self) -> u64 {
        1u64 << self.log_capacity
    }

    /// True (user-facing) capacity = `dictionary_capacity /
    /// OVER_PROVISIONING_FACTOR`. The dictionary is sized to comfortably
    /// hold this many entries before open-addressing trails grow.
    pub fn true_capacity(&self) -> u64 {
        self.dictionary_capacity() >> LOG2_OVER_PROVISIONING_FACTOR
    }
}

/// Verifier-side bundle: everything a client or auditor needs to call
/// the verification functions. Issued by the server via
/// `Aegon::verifier_context()`.
#[derive(Debug)]
pub struct VerifierContext<E: Pairing, P: AegonPcs<E>> {
    pub log_capacity: usize,
    pub verifier_param: P::VerifierParam,
    /// Which Fiat-Shamir derivations the audit path uses. Must match
    /// what the server was configured with, or every audit fails.
    /// Defaults to SHA256, i.e. the original behaviour.
    pub audit_fs: super::audit_fs::AuditFsHooks<E, P>,
    pub _e: PhantomData<E>,
}

// Manual Clone — the `#[derive]` would impose `P: Clone` rather than
// the actually-required `P::VerifierParam: Clone`.
impl<E: Pairing, P: AegonPcs<E>> Clone for VerifierContext<E, P>
where
    P::VerifierParam: Clone,
{
    fn clone(&self) -> Self {
        Self {
            log_capacity: self.log_capacity,
            verifier_param: self.verifier_param.clone(),
            audit_fs: self.audit_fs,
            _e: PhantomData,
        }
    }
}

impl<E: Pairing, P: AegonPcs<E>> VerifierContext<E, P> {
    pub fn new(log_capacity: usize, verifier_param: P::VerifierParam) -> Self {
        Self {
            log_capacity,
            verifier_param,
            audit_fs: super::audit_fs::AuditFsHooks::sha256(),
            _e: PhantomData,
        }
    }

    /// Swap in a different audit-path Fiat-Shamir bundle — in
    /// practice the Poseidon one, when the deployment is audited via
    /// the IVC path. Must mirror the server's configuration.
    pub fn with_audit_fs(mut self, hooks: super::audit_fs::AuditFsHooks<E, P>) -> Self {
        self.audit_fs = hooks;
        self
    }
}
