// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Convenience constructors for `AegonConfig` over specific PCS
//! backends. Importing this module pulls in the relevant PCS crate;
//! the rest of the Aegon library does not.

use akd_core::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
use akd_core::aegon_crypto::pcs::kzhk::KZHK;
use ark_ec::pairing::Pairing;
use std::marker::PhantomData;

use super::config::AegonConfig;

/// Return the KZH-k `k` parameter that minimizes the aux-precomputation
/// cost `f(k) = k(k − 1) · 2^(N/k)` for a polynomial of `log_capacity`
/// variables. Empirically tabulated values for `N ∈ [20, 35]`; the
/// underlying function is convex in `k`, so the surrounding ranges
/// extrapolate sensibly. For very small `N` (below the tabulated
/// range) we fall back to small `k` values that satisfy KZH-k's
/// internal constraint `k ≤ N`.
///
/// | N    | k*   | min work |
/// |------|------|----------|
/// | 20   | 6    | ~302     |
/// | 21   | 7    | ~336     |
/// | 22   | 7    | ~371     |
/// | 23   | 7    | ~410     |
/// | 24   | 8    | ~448     |
/// | 25   | 8    | ~488     |
/// | 26   | 8–9  | ~533     |
/// | 27   | 9    | ~576     |
/// | 28   | 9    | ~622     |
/// | 29   | 10   | ~671     |
/// | 30   | 10   | ~720     |
/// | 31   | 10   | ~771     |
/// | 32   | 11   | ~826     |
/// | 33   | 11   | ~880     |
/// | 34   | 11   | ~937     |
/// | 35   | 12   | ~997     |
///
/// Operators who want a different objective (e.g. minimum proof size
/// rather than minimum prover work) should set `.kzh_k(...)` on the
/// builder explicitly rather than relying on this helper.
pub fn optimal_kzh_k(log_capacity: usize) -> usize {
    let k = match log_capacity {
        0 => 1,
        1..=5 => 2,
        6..=14 => 3,
        15..=19 => 5,
        20 => 6,
        21..=23 => 7,
        24..=26 => 8, // n=26 is tied between 8 and 9; pick 8
        27..=28 => 9,
        29..=31 => 10,
        32..=34 => 11,
        _ => 12,
    };
    // KZHKConfig enforces 1 ≤ k ≤ log_capacity; clamp for safety.
    k.min(log_capacity.max(1))
}

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
pub fn kzh<E: Pairing>(log_capacity: usize, k: usize, private: bool) -> AegonConfig<E, KZHK<E>> {
    assert!(log_capacity > 0, "log_capacity must be positive");
    assert!(k > 0 && k <= log_capacity, "k must be in 1..=log_capacity");
    AegonConfig {
        log_capacity,
        private,
        pcs_config: KZHKConfig::new(k, private),
        audit_fs: crate::aegon::audit_fs::AuditFsHooks::sha256(),
        _e: PhantomData,
    }
}
