// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Runtime RSS instrumentation for diagnosing shard memory spikes.
//!
//! Cheap when disabled (one relaxed atomic load). Enable via
//! `set_rss_log(true)` once at startup (`aegon_shard_server`'s
//! `--log-rss` flag). When enabled, each call reads `/proc/self/status`
//! and prints `VmRSS` in GiB alongside a caller-supplied stage label.

use std::sync::atomic::{AtomicBool, Ordering};

static RSS_LOG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn peak-RSS logging on or off for this process.
pub fn set_rss_log(enabled: bool) {
    RSS_LOG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether peak-RSS logging is currently on.
pub fn rss_log_enabled() -> bool {
    RSS_LOG_ENABLED.load(Ordering::Relaxed)
}

/// Print VmRSS in GiB with `stage` label. No-op when disabled.
pub fn log_rss(stage: &str) {
    if !rss_log_enabled() {
        return;
    }
    match read_vmrss_kb() {
        Some(kb) => println!(
            "[rss] {stage}: {:.3} GiB ({} kB)",
            kb as f64 / (1024.0 * 1024.0),
            kb
        ),
        None => println!("[rss] {stage}: <unavailable>"),
    }
}

/// Same as `log_rss` but with extra key=value context appended.
pub fn log_rss_ctx(stage: &str, ctx: &str) {
    if !rss_log_enabled() {
        return;
    }
    match read_vmrss_kb() {
        Some(kb) => println!(
            "[rss] {stage} {ctx}: {:.3} GiB ({} kB)",
            kb as f64 / (1024.0 * 1024.0),
            kb
        ),
        None => println!("[rss] {stage} {ctx}: <unavailable>"),
    }
}

/// Publish-time profiling. Reads `AEGON_PROFILE_PUBLISH` once on
/// first call; when set to "1" / "true", `pub_profile_*` helpers
/// stream per-stage wall times to stderr. Off by default — cheap when
/// disabled (one relaxed atomic load).
static PUBLISH_PROFILE_INIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether per-phase publish profiling is on, read once from the
/// environment and cached. Gates the `[pub-profile]` timing lines.
pub fn publish_profile_enabled() -> bool {
    *PUBLISH_PROFILE_INIT.get_or_init(|| {
        matches!(
            std::env::var("AEGON_PROFILE_PUBLISH").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes")
        )
    })
}

fn read_vmrss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let Some(kb_str) = parts.first() {
                return kb_str.parse().ok();
            }
        }
    }
    None
}
