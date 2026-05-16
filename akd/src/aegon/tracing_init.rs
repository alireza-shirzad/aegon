//! Installs a `tracing-tree` subscriber for the Aegon bins.
//!
//! Compiled only with the `tracing_instrument` feature. The bins call
//! [`init_tree_subscriber`] once at startup; it picks up `RUST_LOG`
//! when set, otherwise defaults to `akd=debug,akd_core=debug` so every
//! span we annotated (`Aegon::PublishPhase1`, `KZH::FMAState`,
//! `ShardedAegon::*`, etc.) actually emits.

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;
use tracing_tree::HierarchicalLayer;

/// Install a hierarchical-tree subscriber. Indents nested spans, prints
/// elapsed wall time at each span close. Re-invocation is silently
/// no-op'd (idempotent across binaries that call it twice).
///
/// Default filter is `akd=debug,akd_core::aegon=debug,akd_core::aegon_crypto::pcs::kzhk=info`
/// — the top-level Aegon spans show, but the per-opening `KZH::Open*`
/// trio (which fires thousands of times per publish at large batch
/// sizes) is suppressed so it doesn't drown out the phase summaries.
/// Override via `RUST_LOG=...` to see them.
pub fn init_tree_subscriber() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "akd=debug,akd_core::aegon=debug,akd_core::aegon_crypto::pcs::kzhk=info",
        )
    });
    let layer = HierarchicalLayer::new(2)
        .with_targets(true)
        .with_bracketed_fields(true)
        .with_thread_ids(false)
        .with_indent_lines(true)
        .with_verbose_exit(true);
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .try_init();
}
