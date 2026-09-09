// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Installs a `tracing-tree` subscriber for the Aegon bins.
//!
//! Compiled only with the `tracing_instrument` feature. The bins call
//! [`init_tree_subscriber`] once at startup; it picks up `RUST_LOG`
//! when set, otherwise defaults to `akd=debug,akd_core=debug` so every
//! span we annotated (`Aegon::PublishPhase1`, `KZH::FMAState`,
//! `ShardedAegon::*`, etc.) actually emits.

use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Install a flat subscriber that prints one line per span close with
/// the elapsed wall time. Re-invocation is silently no-op'd
/// (idempotent across binaries that call it twice).
///
/// Default filter is `akd=debug,akd_core::aegon=debug,akd_core::aegon_crypto::pcs::kzhk=info`
/// — the top-level Aegon spans show, but the per-opening `KZH::Open*`
/// trio (which fires thousands of times per publish at large batch
/// sizes) is suppressed so it doesn't drown out the phase summaries.
/// Override via `RUST_LOG=...` to see them.
///
/// `FmtSpan::CLOSE` is the bit that produces lines like:
///   `INFO close time.busy=12.3ms time.idle=2.1ms: ShardedAegon::RunPhase1`
/// at each span exit. We previously used `tracing-tree`, but its close
/// hook turned out to be no-op'd in our version unless extra options
/// were toggled in just the right combination.
pub fn init_tree_subscriber() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("akd=debug,akd_core::aegon=debug,akd_core::aegon_crypto::pcs::kzhk=info")
    });
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_writer(std::io::stderr)
        .with_span_events(FmtSpan::CLOSE);
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .try_init();
}
