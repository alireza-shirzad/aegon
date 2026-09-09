// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Defines the configuration trait and implementations for various configurations

mod traits;
pub use traits::{Configuration, DomainLabel, ExampleLabel};

#[cfg(feature = "public_tests")]
pub use traits::NamedConfiguration;

// Note(new_config): Update this when adding a new configuration

#[cfg(feature = "whatsapp_v1")]
pub(crate) mod whatsapp_v1;
#[cfg(feature = "whatsapp_v1")]
pub use whatsapp_v1::WhatsAppV1Configuration;

#[cfg(feature = "experimental")]
pub(crate) mod experimental;
#[cfg(feature = "experimental")]
pub use experimental::ExperimentalConfiguration;
