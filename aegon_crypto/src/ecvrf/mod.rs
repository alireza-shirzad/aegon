// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! This module contains an implementation of a
//! [verifiable random function](https://en.wikipedia.org/wiki/Verifiable_random_function)
//! (ECVRF). Aegon uses it to derive each label's shard and slot placement: the
//! server can compute the mapping and prove it, and a client can verify it,
//! but nobody without the secret key can compute placements for other labels.
//!
//! This module implements an instantiation of a verifiable random function known as
//! [ECVRF-EDWARDS25519-SHA512-TAI from RFC9381](https://www.ietf.org/rfc/rfc9381.html).
//!
//! Taken from [facebook/akd](https://github.com/facebook/akd), which adapted it from
//! Diem's NextGen Crypto module available [here](https://github.com/diem/diem/blob/502936fbd59e35276e2cf455532b143796d68a16/crypto/nextgen_crypto/src/vrf/ecvrf.rs).
//! Only the raw ECVRF is retained; AKD's key-storage trait is not.

mod ecvrf_impl;
// export the functionality we want visible
pub use crate::ecvrf::ecvrf_impl::{
    Output, Proof, VRFExpandedPrivateKey, VRFPrivateKey, VRFPublicKey, OUTPUT_LENGTH, PROOF_LENGTH,
};

#[cfg(test)]
mod tests;

/// A error related to verifiable random functions
#[derive(Debug, Eq, PartialEq)]
pub enum VrfError {
    /// A problem retrieving or decoding the VRF public key
    PublicKey(String),
    /// A problem retrieving or decoding the VRF signing key
    SigningKey(String),
    /// A problem verifying the VRF proof
    Verification(String),
}

impl core::fmt::Display for VrfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let code = match &self {
            VrfError::PublicKey(msg) => format!("(Public Key) - {msg}"),
            VrfError::SigningKey(msg) => format!("(Signing Key) - {msg}"),
            VrfError::Verification(msg) => format!("(Verification) - {msg}"),
        };
        write!(f, "Verifiable random function error {code}")
    }
}
