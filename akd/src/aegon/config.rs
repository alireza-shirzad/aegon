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
            _e: PhantomData,
        }
    }

    /// Hypercube size — the number of slots a fully-loaded dictionary
    /// can hold before open-addressing starts colliding.
    ///
    /// Note: `2^log_capacity` is the *theoretical* capacity. In
    /// practice you want a load factor ≤ 1/4 (paper §5.1), so plan
    /// for `2^log_capacity / 4` actual entries before probe lengths
    /// degrade.
    pub fn dictionary_capacity(&self) -> u64 {
        1u64 << self.log_capacity
    }
}

/// Verifier-side bundle: everything a client or auditor needs to call
/// the verification functions. Issued by the server via
/// `Aegon::verifier_context()`.
#[derive(Debug)]
pub struct VerifierContext<E: Pairing, P: AegonPcs<E>> {
    pub log_capacity: usize,
    pub verifier_param: P::VerifierParam,
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
            _e: PhantomData,
        }
    }
}

impl<E: Pairing, P: AegonPcs<E>> VerifierContext<E, P> {
    pub fn new(log_capacity: usize, verifier_param: P::VerifierParam) -> Self {
        Self {
            log_capacity,
            verifier_param,
            _e: PhantomData,
        }
    }
}
