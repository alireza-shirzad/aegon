// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! Cryptographic primitives that back the Aegon engine.
//!
//! These were originally separate workspace crates (`arithmetic`,
//! `subroutines`, `transcript`). They are now folded into `akd_core`
//! so the AKD-on-Aegon system lives in two crates total
//! (`akd_core` + `akd`).
//!
//! The public surface is preserved verbatim so internal references
//! using `akd_core::aegon_crypto::pcs::...` / `arithmetic::...` /
//! `transcript::...` resolve at the new locations.

#![allow(missing_docs)]
#![allow(clippy::non_canonical_clone_impl)]

pub mod arithmetic;
pub mod pcs;
pub mod poly;
pub mod transcript;

// Convenience re-exports so external code can write
// `akd_core::aegon_crypto::IOPTranscript`, etc.
pub use pcs::prelude::*;
