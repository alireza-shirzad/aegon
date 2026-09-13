// Copyright (c) Meta Platforms, Inc. and affiliates.
// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Cryptographic primitives that back the Aegon engine.
//!
//! * [`pcs`] — the KZH-k multilinear polynomial commitment scheme, its
//!   SRS, and the MSM wrapper it runs on.
//! * [`arithmetic`] and [`poly`] — multilinear polynomial arithmetic.
//! * [`transcript`] — the Fiat–Shamir transcript.
//! * [`ecvrf`] — ECVRF-EDWARDS25519-SHA512-TAI (RFC 9381), used to derive
//!   verifiable label placements.
//!
//! The engine itself lives in the `aegon` crate.

#![allow(missing_docs)]
#![allow(clippy::non_canonical_clone_impl)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod arithmetic;
pub mod ecvrf;
pub mod pcs;
pub mod poly;
pub mod transcript;

// Convenience re-exports so external code can write
// `aegon_crypto::IOPTranscript`, etc.
pub use pcs::prelude::*;
