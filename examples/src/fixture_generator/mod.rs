// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! A CLI tool for generating directory fixtures for debug and testing purposes.
//! Run cargo run -p examples -- fixture-generator --help for options. Example command:
//!
//!   cargo run -- fixture-generator \
//!     --user "User1: 1, (9, 'abc'), (10, 'def')" \
//!     --user "User2: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10" \
//!     --epochs 10 \
//!     --max_updates 5 \
//!     --capture_states 9 10 \
//!     --capture_deltas 10
//!

mod examples;
mod generator;
mod parser;
pub mod reader;
pub mod writer;

pub(crate) use parser::Args;

/// Re-export generator run function.
pub(crate) use generator::run;

const YAML_SEPARATOR: &str = "---";
