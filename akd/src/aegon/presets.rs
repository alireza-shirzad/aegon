//! Convenience constructors for `AegonConfig` over specific PCS
//! backends. Importing this module pulls in the relevant PCS crate;
//! the rest of the Aegon library does not.

use ark_ec::pairing::Pairing;
use std::marker::PhantomData;
use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;

use super::config::AegonConfig;

/// Build an `AegonConfig` for KZH-k. Sets `log_capacity` plus the
/// `KZHKConfig{k, zk = private}`; KZH-k decides its own internal
/// block split at SRS-gen time, and Aegon picks it up via
/// `P::block_dims` inside `init`. The user does not have to touch
/// dims, and the system-level `private` flag is propagated as KZH-k's
/// `zk` parameter so the two cannot drift.
///
/// # Arguments
///
/// * `log_capacity` — `log_2` of the dictionary's hypercube size.
/// * `k` — KZH-k's block count. Must be in `1..=log_capacity`.
///   `k=2` is the classical KZH (`O(√N)` proofs). Larger `k` shrinks
///   proofs at the cost of more server work.
/// * `private` — whether to run in privacy-preserving mode. Wires
///   `zk = private` into `KZHKConfig`.
pub fn kzh<E: Pairing>(
    log_capacity: usize,
    k: usize,
    private: bool,
) -> AegonConfig<E, KZHK<E>> {
    assert!(log_capacity > 0, "log_capacity must be positive");
    assert!(k > 0 && k <= log_capacity, "k must be in 1..=log_capacity");
    AegonConfig {
        log_capacity,
        private,
        pcs_config: KZHKConfig::new(k, private),
        _e: PhantomData,
    }
}
