//! Helpers for installing a `tracing-subscriber` that prints span
//! durations. Enabled via the `tracing` feature of this crate.
//!
//! Library crates in the workspace emit `tracing` spans (e.g. via
//! `#[tracing::instrument]` or `tracing::debug_span!`). They do not
//! install a subscriber. Binary / bench / test entry points should
//! call [`init`] once at startup so those spans are rendered.
//!
//! Output format:
//! ```text
//! <uptime>  <span::chain>  <fields>
//! ```
//! with the timestamp colored green, the span chain yellow, and the rest
//! in the terminal's default color. The `RUST_LOG` env var still controls
//! filtering (default: `warn,subroutines=debug,arithmetic=debug,transcript=debug`).

use std::fmt::{self, Write as _};
use std::time::Instant;
use tracing::{
    field::{Field, Visit},
    Event, Subscriber,
};
use tracing_subscriber::{
    fmt::{
        format::{FmtSpan, Writer},
        FmtContext, FormatEvent, FormatFields,
    },
    registry::LookupSpan,
    EnvFilter,
};

// ANSI escape codes. We emit them unconditionally because the subscriber is
// only installed by bench/test entry points that write to stderr.
const C_RESET: &str = "\x1b[0m";
const C_GREEN: &str = "\x1b[32m";
const C_YELLOW: &str = "\x1b[33m";
const C_DIM: &str = "\x1b[2m";

/// Custom event formatter: `<green uptime>  <yellow span chain>  <fields>`.
struct ColoredFormat {
    start: Instant,
}

impl<S, N> FormatEvent<S, N> for ColoredFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        // Timestamp (green).
        let elapsed = self.start.elapsed();
        write!(
            writer,
            "{}{:>10.3?}{}  ",
            C_GREEN, elapsed, C_RESET,
        )?;

        // Indent by span depth so nested spans are visually offset from
        // their parents (2 spaces per level).
        let depth = ctx
            .event_scope()
            .map(|scope| scope.from_root().count())
            .unwrap_or(0);
        let indent = depth.saturating_sub(1) * 2;
        for _ in 0..indent {
            write!(writer, " ")?;
        }

        // Innermost span name only (yellow) — the chain is implicit in
        // the indentation.
        if let Some(scope) = ctx.event_scope() {
            if let Some(leaf) = scope.from_root().last() {
                write!(writer, "{}{}{}  ", C_YELLOW, leaf.name(), C_RESET)?;
            }
        }

        // Fields (dim). Skip `time.idle` — we only care about busy time.
        let mut visitor = FilteredFields::default();
        event.record(&mut visitor);
        if !visitor.out.is_empty() {
            write!(writer, "{}{}{}", C_DIM, visitor.out, C_RESET)?;
        }
        writeln!(writer)
    }
}

/// A field visitor that renders all fields except `time.idle` into an
/// internal string. Used so we can drop the `time.idle=...` noise without
/// reimplementing the default field formatter.
#[derive(Default)]
struct FilteredFields {
    out: String,
}

impl Visit for FilteredFields {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        let name = field.name();
        if name == "time.idle" {
            return;
        }
        if !self.out.is_empty() {
            self.out.push(' ');
        }
        if name == "message" {
            let _ = write!(self.out, "{:?}", value);
        } else {
            let _ = write!(self.out, "{}={:?}", name, value);
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "time.idle" {
            return;
        }
        if !self.out.is_empty() {
            self.out.push(' ');
        }
        if field.name() == "message" {
            self.out.push_str(value);
        } else {
            let _ = write!(self.out, "{}={}", field.name(), value);
        }
    }
}

/// Installs a global fmt subscriber that prints each span's duration when
/// it closes.
///
/// Idempotent: if a global subscriber is already installed, this returns
/// `Ok(())` without replacing it. Safe to call from multiple bench / test
/// entry points.
pub fn init() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Default to silent when RUST_LOG is unset. Set e.g. RUST_LOG=debug or
    // RUST_LOG=subroutines=debug to enable span output.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("off"));

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .with_writer(std::io::stderr)
        .event_format(ColoredFormat {
            start: Instant::now(),
        })
        .finish();

    let _ = tracing::subscriber::set_global_default(subscriber);
    Ok(())
}
