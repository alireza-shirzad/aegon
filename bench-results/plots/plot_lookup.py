#!/usr/bin/env python3
"""Plot Aegon benchmark results to PDFs.

Reads bench-results/lookup/{small_fill*.json, medium-combined.json} and
emits PDFs under bench-results/plots/.

Plots so far:
  server_lookup_vs_fill.pdf
      Server-side lookup time vs preload fill_percent, one panel per
      lookup operation (label, value, value-history, label-history).
      Small and medium plotted on the same axes for direct comparison.
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
LOOKUP_DIR = REPO_ROOT / "bench-results" / "lookup"
SMALL_LOOKUP_DIR = REPO_ROOT / "bench-results" / "_remote-small" / "lookup"
PUBLISH_DIR = REPO_ROOT / "bench-results" / "publish"
MIGRATION_DIR = REPO_ROOT / "bench-results" / "migration"
PLOTS_DIR = REPO_ROOT / "bench-results" / "plots"
# Facebook AKD (MySQL backend) overlay data. Same bench harness wrote
# these, so the JSON layout matches aegon's exactly; load_akd_* just
# rebrands the result so the plot helpers can overlay it as a distinct
# "system" alongside the aegon regimes.
AKD_DIR = REPO_ROOT / "bench-results" / "akd-mysql"
# irondict (single-shard VKD with KZH-K commitments) overlay. The
# irondict bench has no fill_percent dimension — every operation is
# measured on a near-empty tree — so irondict overlays render as
# horizontal lines spanning the fill axis. Layout below in
# IRONDICT_STYLES / IronStats / load_irondict_*().
IRONDICT_DIR = REPO_ROOT / "bench-results" / "irondict"

# Lookup operations and the JSON field stems for each.
LOOKUP_KINDS = [
    # Ordered for 2×2 panel layout (row-major flatten). Column 1 is the
    # "label" stack (label on top, label-history below); column 2 is the
    # "value" stack (value on top, value-history below). Reordering the
    # two history entries only affects the three `zip(axes,
    # LOOKUP_KINDS)` panel-builder sites — every other use just walks
    # the list to build dicts and is order-agnostic.
    ("label", "label", "label lookup"),
    ("value", "value", "value lookup"),
    ("label_history", "label-history", "label-history lookup"),
    ("history", "value-history", "value-history lookup"),
]

# Proof-size JSON field per kind. The latency stems above don't match
# the proof-size stems 1:1 (history → value_history), so we keep the
# mapping explicit.
PROOF_FIELDS = {
    "label":         "label_lookup_proof_bytes",
    "value":         "value_lookup_proof_bytes",
    "history":       "value_history_lookup_proof_bytes",
    "label_history": "label_history_lookup_proof_bytes",
}

REGIME_STYLES = {
    "small": {"color": "#1f77b4", "marker": "o", "label": "aegon small"},
    "medium": {"color": "#d62728", "marker": "o", "label": "aegon medium"},
    "large": {"color": "#2ca02c", "marker": "^", "label": "aegon large"},
}

# AKD overlay styles. Color matches the corresponding aegon regime so
# the eye groups by capacity tier; the dashed line + hollow marker
# encodes "system = AKD-MySQL" so a reader can read aegon-vs-AKD on
# linestyle alone. Keys match the regime they pair with so plot
# helpers can look them up by the same name as the aegon regime.
AKD_STYLES = {
    "small": {
        "color": REGIME_STYLES["small"]["color"],
        "marker": "D", "linestyle": "-",
        "label": "AKD small",
    },
    "medium": {
        "color": REGIME_STYLES["medium"]["color"],
        "marker": "D", "linestyle": "-",
        "label": "AKD medium",
    },
    "large": {
        "color": REGIME_STYLES["large"]["color"],
        "marker": "D", "linestyle": "-",
        "label": "AKD large",
    },
}

# irondict overlay styles. Dash-DOT (vs AKD's dash) so a glance at the
# linestyle alone tells aegon / AKD / irondict apart; square marker so
# it doesn't collide with AKD's diamond. Color matches the
# corresponding aegon regime, same logic as AKD_STYLES.
IRONDICT_STYLES = {
    "small": {
        "color": REGIME_STYLES["small"]["color"],
        "marker": "s", "linestyle": "-",
        "label": "irondict small",
    },
    "medium": {
        "color": REGIME_STYLES["medium"]["color"],
        "marker": "s", "linestyle": "-",
        "label": "irondict medium",
    },
    "large": {
        "color": REGIME_STYLES["large"]["color"],
        "marker": "s", "linestyle": "-",
        "label": "irondict large",
    },
}

# Regime subsets we render. Every top-level plot function is called
# once per subset from `main()`, so the small+medium figure is fully
# decoupled from the large figure — important when large is
# qualitatively different (different shard count, different ceiling
# behaviour) and shouldn't squish small/medium's y-axis.
SMALL_MEDIUM: tuple[str, ...] = ("small", "medium")
LARGE_ONLY: tuple[str, ...] = ("large",)
# All three regimes on a single figure — used only by the merged
# migration_curves plot, since throughput-vs-K sweeps are
# qualitatively similar across regimes (same shape, different
# absolute scale) and benefit from a side-by-side comparison.
ALL_REGIMES: tuple[str, ...] = ("small", "medium", "large")


def _subset_suffix(subset: tuple[str, ...]) -> str:
    """Filename suffix derived from a regime subset, e.g.
    ('small','medium') -> '_small_medium', ('large',) -> '_large'."""
    return "_" + "_".join(subset)


# Aggregations the top-level plot functions know about. "median"
# keeps the historical (50th/10th/90th percentile) view; "max"
# swaps the central line to the maximum measured sample and the
# shaded band to the full min..max envelope. Filename suffix is
# empty for median (so existing artefacts keep their names) and
# "_max" for the worst-case view.
AGG_MEDIAN = "median"
AGG_MAX = "max"


def _percentiles_for(agg: str) -> tuple[float, float, float]:
    """(low, central, high) percentiles. Median line + min..max band."""
    return 0.0, 50.0, 100.0


def _agg_suffix(agg: str) -> str:
    """Empty suffix for median (current behaviour); '_max' for max."""
    return "" if agg == AGG_MEDIAN else f"_{agg}"


def _agg_label(agg: str) -> str:
    """Human label used in axis text: 'median' or 'max'."""
    return agg


def _reducer_for(agg: str):
    """Per-sample reducer used by the extrapolation helpers when
    collapsing a fill level's sample list to a single anchor value.
    Median is robust to one-off slow samples (the natural choice for
    typical-case extrapolation); max anchors on the worst observed
    sample (the natural choice for the worst-case view)."""
    return np.max if agg == AGG_MAX else np.median


def _label_axis_endpoints(ax: plt.Axes, formatter=None) -> None:
    """Ensure both endpoints of each axis show a numeric label.

    matplotlib's auto-tick placement chooses round multiples (e.g. 10⁻³, 10⁻²
    on log scales) and stops *inside* the data range, so the visible upper
    and lower bounds are often unlabeled — the reader has to estimate where
    the curve actually peaks.

    Strategy: for each axis, check whether both endpoints are *already*
    within 2% of an existing major tick. If so, leave the tick set alone
    (the plot author chose a careful set and the endpoints are covered).
    Otherwise, add the missing endpoint(s) explicitly, dropping any existing
    tick that would crowd into the new endpoint label (~18% of the log span,
    or ~5% on linear scales). Pass `formatter` to override the default
    decimal renderer (e.g. mathtext scientific for plots whose endpoints
    fall in the sub-millisecond or sub-percent range).
    """
    import math
    if formatter is None:
        formatter = _format_axis_value
    for axis in (ax.xaxis, ax.yaxis):
        scale = axis.get_scale()
        lo, hi = axis.get_view_interval()
        if lo > hi:
            lo, hi = hi, lo
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            continue
        existing = sorted(t for t in axis.get_majorticklocs() if lo <= t <= hi)
        is_log = scale == "log" and lo > 0 and hi > 0
        if is_log:
            span = math.log10(hi) - math.log10(lo)
            def dist(a: float, b: float) -> float:
                return abs(math.log10(a) - math.log10(b))
        else:
            span = hi - lo
            def dist(a: float, b: float) -> float:
                return abs(a - b)
        # If an endpoint is essentially AT an existing tick (within ~6% of
        # the axis span), treat it as already labelled. This covers
        # matplotlib's default 5% auto-margin: a plot with set_xticks([1, 90])
        # has view interval (-3.45, 94.5), but ticks 1 and 90 are the
        # *intended* endpoints — we should keep them, not add -3.45/94.5.
        endpoint_tol = 0.06 * span
        # Crowding tolerance for dropping existing ticks that sit too close
        # to a new endpoint label.
        crowd_tol = 0.18 * span if is_log else 0.05 * span
        endpoints_to_add: list[float] = []
        for e in (lo, hi):
            if is_log and e <= 0:
                continue
            if not any(dist(t, e) < endpoint_tol for t in existing):
                endpoints_to_add.append(e)
        if not endpoints_to_add:
            # Both endpoints already covered — leave the existing tick set
            # untouched (avoids clobbering audit_vs_fill's carefully-chosen
            # [0.1, 0.2, ..., 50] tick list with our endpoint-only override).
            continue
        kept_existing = [
            t for t in existing
            if all(dist(t, e) > crowd_tol for e in endpoints_to_add)
        ]
        new_ticks = sorted(set(kept_existing + endpoints_to_add))
        if not new_ticks:
            continue
        axis.set_ticks(new_ticks)
        axis.set_major_formatter(plt.FuncFormatter(formatter))
        # Hide minor tick labels — matplotlib auto-labels minor ticks
        # (e.g. "8.2×10³", "4×10⁰") when a log axis spans less than one
        # full decade, which then overlaps visually with our new endpoint
        # labels and clutters the panel. The major endpoint labels carry
        # the bounds information the reader needs.
        axis.set_minor_formatter(plt.NullFormatter())


def _format_axis_value(v: float, _pos: int = 0) -> str:
    """Format a tick value as a compact decimal for the paper-readable range.

    %g (which we used initially) silently switches to scientific notation
    once a number reaches 1e+05 or drops below 1e-04. For our plots this
    showed up as "5.43e+03 QPS" and "9.87×10³ bytes", both of which read
    worse than the literal "5430" / "9870". This formatter forces decimal
    rendering for the typical paper range and falls back to %g only at
    truly extreme magnitudes.
    """
    import math
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e6 or av < 1e-4:
        return f"{v:.3g}"
    # 3 significant figures, decimal form, no trailing zeros.
    mag = math.floor(math.log10(av))
    digits = max(0, 2 - mag)
    s = f"{v:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _format_log_tick(v: float, _pos: int = 0) -> str:
    """Compact log-axis tick formatter.

    Decimal in [0.01, 100) so 0.05 / 0.5 / 2 / 50 render naturally;
    mathtext sci outside that band so very small or very large values
    (e.g. 5×10⁻⁵, 10⁻³) stay narrow and uniform. Used by the publish
    plot together with `LogLocator(subs=(1, 2, 5))` to give three
    labelled ticks per decade.
    """
    import math
    if v == 0:
        return "0"
    av = abs(v)
    if av < 0.01 or av >= 100:
        mag = math.floor(math.log10(av))
        mantissa = round(v / (10 ** mag))
        if mantissa == 1:
            return rf"$10^{{{mag}}}$"
        if mantissa == -1:
            return rf"$-10^{{{mag}}}$"
        return rf"${mantissa}\times 10^{{{mag}}}$"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:g}"


def _format_axis_value_sci(v: float, _pos: int = 0) -> str:
    """Compact mathtext scientific notation for log-axis endpoints.

    The decimal formatter renders 0.00363 as "0.00363" — five characters
    wider than "3.6×10⁻³" once you account for matplotlib's tick padding.
    On the publish plot that extra width pushes the small/medium panels
    around between renders. This formatter falls back to plain decimal in
    the [0.01, 1000) range where it isn't any wider, and switches to
    mathtext sci form outside that band.
    """
    import math
    if v == 0:
        return "0"
    av = abs(v)
    if 0.01 <= av < 1000:
        return _format_axis_value(v)
    mag = math.floor(math.log10(av))
    mantissa = v / (10 ** mag)
    s = f"{mantissa:.1f}".rstrip("0").rstrip(".")
    if s == "1":
        return rf"$10^{{{mag}}}$"
    if s == "-1":
        return rf"$-10^{{{mag}}}$"
    return rf"${s}\times 10^{{{mag}}}$"


@dataclass
class LevelStats:
    """Per-level lookup stats for one regime."""

    fill_percent: float
    true_log_capacity: int               # log2 of the dictionary's true capacity
    server_ms: dict[str, list[float]]    # kind -> list of per-sample server ms
    client_ms: dict[str, list[float]]    # kind -> list of per-sample client (verify) ms
    proof_bytes: dict[str, list[int]]    # kind -> list of per-sample proof bytes
    publish_ms: dict[int, list[float]]   # batch_size -> list of per-sample publish ms
    audit_ms: list[float]                # per-sample client audit-verify ms
    audit_bytes: list[int]               # per-sample audit proof bytes
    # Concurrency sweep: present only if the bench was run with
    # `--throughput-concurrencies`. Each entry is one (concurrency,
    # qps, p50/p90/p99 latency_ms) row.
    concurrency_sweep_kind: str = ""
    concurrency_sweep: list[dict] = None  # type: ignore


def _extract_concurrency_sweep(lvl: dict) -> tuple[str, list[dict]]:
    """Pull (lookup_kind, samples) from a level dict's concurrency_sweep
    block. Returns ("", []) when the field is absent."""
    block = lvl.get("concurrency_sweep") or {}
    if not isinstance(block, dict):
        return "", []
    return str(block.get("lookup_kind", "")), list(block.get("samples", []))


@dataclass
class MigrationMilestone:
    """One (fill_percent, elapsed) point from a migration-bench run."""
    fill_percent: float
    users_migrated: int
    elapsed_ms: float
    throughput_users_per_sec: float


@dataclass
class MigrationRun:
    """All milestones from a single (regime, K) migration run."""
    regime: str
    chunk_size: int
    target_fill_percent: float
    total_users: int
    total_elapsed_ms: float
    total_throughput_users_per_sec: float
    milestones: list[MigrationMilestone]


def load_migration_runs(regime: str) -> list[MigrationRun]:
    """Load every `{regime}_K*.json` from `bench-results/migration/`.
    Returns the runs sorted by chunk_size. Empty if no runs exist
    (migration bench never executed for this regime)."""
    runs: list[MigrationRun] = []
    if not MIGRATION_DIR.exists():
        return runs
    for path in sorted(MIGRATION_DIR.glob(f"{regime}_K*.json")):
        try:
            with path.open() as fh:
                d = json.load(fh)
        except json.JSONDecodeError:
            continue
        params = d.get("params", {})
        total = d.get("total", {})
        ms_list = d.get("milestones", [])
        milestones = [
            MigrationMilestone(
                fill_percent=float(m.get("fill_percent", 0)),
                users_migrated=int(m.get("users_migrated", 0)),
                elapsed_ms=float(m.get("elapsed_ms", 0.0)),
                throughput_users_per_sec=float(m.get("throughput_users_per_sec", 0.0)),
            )
            for m in ms_list
        ]
        runs.append(MigrationRun(
            regime=regime,
            chunk_size=int(params.get("chunk_size", 0)),
            target_fill_percent=float(params.get("target_fill_percent", 0)),
            total_users=int(total.get("users_migrated", 0)),
            total_elapsed_ms=float(total.get("elapsed_ms", 0.0)),
            total_throughput_users_per_sec=float(total.get("throughput_users_per_sec", 0.0)),
            milestones=sorted(milestones, key=lambda m: m.fill_percent),
        ))
    runs.sort(key=lambda r: r.chunk_size)
    return runs


def load_best_k(regime: str) -> int | None:
    """Read `{regime}_best_k.txt` written by migration-bench.sh."""
    f = MIGRATION_DIR / f"{regime}_best_k.txt"
    if not f.exists():
        return None
    try:
        return int(f.read_text().strip())
    except ValueError:
        return None


def load_small() -> list[LevelStats]:
    # Prefer the new cluster-mode output (small-combined.json) which
    # matches the medium/large code path exactly — real-publish prefill,
    # populated history records. Fall back to the legacy local-mode
    # per-fill files only if the cluster output is absent.
    cluster_path = LOOKUP_DIR / "small-combined.json"
    if cluster_path.exists():
        with cluster_path.open() as fh:
            d = json.load(fh)
        fill_pcts = d["params"]["fill_percents"]
        levels = d["levels"]
        tlc = int(d["params"]["true_log_capacity"])
        out: list[LevelStats] = []
        for fp, lvl in zip(fill_pcts, levels):
            samples = lvl.get("lookup", {}).get("samples", [])
            if not samples:
                continue
            server_ms = {
                k: [s[f"server_lookup_{k}_ns"] / 1e6 for s in samples]
                for k, _label, _title in LOOKUP_KINDS
            }
            client_ms = {
                k: [s[f"client_lookup_{k}_ns"] / 1e6 for s in samples]
                for k, _label, _title in LOOKUP_KINDS
            }
            proof_bytes = {
                k: [s[PROOF_FIELDS[k]] for s in samples]
                for k, _label, _title in LOOKUP_KINDS
            }
            publish_ms: dict[int, list[float]] = {}
            for b in lvl.get("publish_bench", {}).get("batches", []):
                publish_ms[int(b["batch_size"])] = list(b.get("samples_ms", []))
            audit_samples = lvl.get("audit", {}).get("samples", [])
            audit_ms = [s["audit_invariance_ns"] / 1e6 for s in audit_samples]
            audit_bytes = [s["audit_proof_bytes"] for s in audit_samples]
            cs_kind, cs_samples = _extract_concurrency_sweep(lvl)
            out.append(LevelStats(
                fill_percent=float(fp),
                true_log_capacity=tlc,
                server_ms=server_ms,
                client_ms=client_ms,
                proof_bytes=proof_bytes,
                publish_ms=publish_ms,
                audit_ms=audit_ms,
                audit_bytes=audit_bytes,
                concurrency_sweep_kind=cs_kind,
                concurrency_sweep=cs_samples,
            ))
        out.sort(key=lambda x: x.fill_percent)
        return out

    out: list[LevelStats] = []
    for pct in (1, 30, 60, 90):
        path = SMALL_LOOKUP_DIR / f"small_fill{pct}.json"
        if not path.exists():
            continue
        with path.open() as fh:
            d = json.load(fh)
        tlc = int(d["params"]["true_log_capacity"])
        for lvl in d["levels"]:
            samples = lvl.get("lookup", {}).get("samples", [])
            if not samples:
                continue
            server_ms = {
                k: [s[f"server_lookup_{k}_ns"] / 1e6 for s in samples]
                for k, _label, _title in LOOKUP_KINDS
            }
            client_ms = {
                k: [s[f"client_lookup_{k}_ns"] / 1e6 for s in samples]
                for k, _label, _title in LOOKUP_KINDS
            }
            proof_bytes = {
                k: [s[PROOF_FIELDS[k]] for s in samples]
                for k, _label, _title in LOOKUP_KINDS
            }
            publish_ms: dict[int, list[float]] = {}
            for b in lvl.get("publish_bench", {}).get("batches", []):
                publish_ms[int(b["batch_size"])] = list(b.get("samples_ms", []))
            audit_samples = lvl.get("audit", {}).get("samples", [])
            audit_ms = [s["audit_invariance_ns"] / 1e6 for s in audit_samples]
            audit_bytes = [s["audit_proof_bytes"] for s in audit_samples]
            cs_kind, cs_samples = _extract_concurrency_sweep(lvl)
            out.append(LevelStats(
                fill_percent=float(pct),
                true_log_capacity=tlc,
                server_ms=server_ms,
                client_ms=client_ms,
                proof_bytes=proof_bytes,
                publish_ms=publish_ms,
                audit_ms=audit_ms,
                audit_bytes=audit_bytes,
                concurrency_sweep_kind=cs_kind,
                concurrency_sweep=cs_samples,
            ))
    out.sort(key=lambda x: x.fill_percent)
    return out


def load_medium() -> list[LevelStats]:
    path = LOOKUP_DIR / "medium-combined.json"
    if not path.exists():
        return []
    with path.open() as fh:
        d = json.load(fh)
    fill_pcts = d["params"]["fill_percents"]
    levels = d["levels"]
    tlc = int(d["params"]["true_log_capacity"])
    out: list[LevelStats] = []
    for fp, lvl in zip(fill_pcts, levels):
        samples = lvl.get("lookup", {}).get("samples", [])
        if not samples:
            continue
        server_ms = {
            k: [s[f"server_lookup_{k}_ns"] / 1e6 for s in samples]
            for k, _label, _title in LOOKUP_KINDS
        }
        client_ms = {
            k: [s[f"client_lookup_{k}_ns"] / 1e6 for s in samples]
            for k, _label, _title in LOOKUP_KINDS
        }
        proof_bytes = {
            k: [s[PROOF_FIELDS[k]] for s in samples]
            for k, _label, _title in LOOKUP_KINDS
        }
        publish_ms: dict[int, list[float]] = {}
        for b in lvl.get("publish_bench", {}).get("batches", []):
            publish_ms[int(b["batch_size"])] = list(b.get("samples_ms", []))
        audit_samples = lvl.get("audit", {}).get("samples", [])
        audit_ms = [s["audit_invariance_ns"] / 1e6 for s in audit_samples]
        audit_bytes = [s["audit_proof_bytes"] for s in audit_samples]
        cs_kind, cs_samples = _extract_concurrency_sweep(lvl)
        out.append(LevelStats(
            fill_percent=float(fp),
            true_log_capacity=tlc,
            server_ms=server_ms,
            client_ms=client_ms,
            proof_bytes=proof_bytes,
            publish_ms=publish_ms,
            audit_ms=audit_ms,
            audit_bytes=audit_bytes,
            concurrency_sweep_kind=cs_kind,
            concurrency_sweep=cs_samples,
        ))
    out.sort(key=lambda x: x.fill_percent)
    return out


def load_large() -> list[LevelStats]:
    """Large-regime cluster data. Reads the partial JSON pulled off the
    coordinator mid-run (one record per completed fill level), so we
    derive each level's fill_percent from preload_count / true capacity
    rather than the params.fill_percents list (which enumerates all
    planned levels, not just the completed ones)."""
    # Prefer the final combined file if it exists; fall back to the
    # mid-run partial snapshot.
    path = LOOKUP_DIR / "large-combined.json"
    if not path.exists():
        path = LOOKUP_DIR / "large-combined-partial.json"
    if not path.exists():
        return []
    with path.open() as fh:
        d = json.load(fh)
    tlc = int(d["params"]["true_log_capacity"])
    true_cap = 1 << tlc
    out: list[LevelStats] = []
    for lvl in d["levels"]:
        samples = lvl.get("lookup", {}).get("samples", [])
        if not samples:
            continue
        server_ms = {
            k: [s[f"server_lookup_{k}_ns"] / 1e6 for s in samples]
            for k, _label, _title in LOOKUP_KINDS
        }
        client_ms = {
            k: [s[f"client_lookup_{k}_ns"] / 1e6 for s in samples]
            for k, _label, _title in LOOKUP_KINDS
        }
        proof_bytes = {
            k: [s[PROOF_FIELDS[k]] for s in samples]
            for k, _label, _title in LOOKUP_KINDS
        }
        publish_ms: dict[int, list[float]] = {}
        for b in lvl.get("publish_bench", {}).get("batches", []):
            publish_ms[int(b["batch_size"])] = list(b.get("samples_ms", []))
        audit_samples = lvl.get("audit", {}).get("samples", [])
        audit_ms = [s["audit_invariance_ns"] / 1e6 for s in audit_samples]
        audit_bytes = [s["audit_proof_bytes"] for s in audit_samples]
        # Derive fill_percent from the actual prefill count.
        fill_percent = 100.0 * lvl["preload_count"] / true_cap
        cs_kind, cs_samples = _extract_concurrency_sweep(lvl)
        out.append(LevelStats(
            fill_percent=fill_percent,
            true_log_capacity=tlc,
            server_ms=server_ms,
            client_ms=client_ms,
            proof_bytes=proof_bytes,
            publish_ms=publish_ms,
            audit_ms=audit_ms,
            audit_bytes=audit_bytes,
            concurrency_sweep_kind=cs_kind,
            concurrency_sweep=cs_samples,
        ))
    out.sort(key=lambda x: x.fill_percent)
    return out


def _load_akd_json(path: Path) -> list[LevelStats]:
    """Load one AKD-MySQL JSON (same layout as aegon's lookup JSON;
    written by the same harness) into LevelStats. Differs from
    load_medium() only in path and the fact that AKD reports the same
    proof-size value for each (regime, fill) — i.e. percentile bands
    collapse to a point — but the helpers handle that already."""
    if not path.exists():
        return []
    with path.open() as fh:
        d = json.load(fh)
    fill_pcts = d["params"]["fill_percents"]
    levels = d["levels"]
    tlc = int(d["params"]["true_log_capacity"])
    out: list[LevelStats] = []
    for fp, lvl in zip(fill_pcts, levels):
        samples = lvl.get("lookup", {}).get("samples", [])
        publish_batches = lvl.get("publish_bench", {}).get("batches", [])
        # Keep a level if it has lookup OR publish data — the
        # extrapolated AKD-large JSON has only publish_bench entries.
        if not samples and not publish_batches:
            continue
        server_ms = {
            k: [s[f"server_lookup_{k}_ns"] / 1e6 for s in samples]
            for k, _label, _title in LOOKUP_KINDS
        }
        client_ms = {
            k: [s[f"client_lookup_{k}_ns"] / 1e6 for s in samples]
            for k, _label, _title in LOOKUP_KINDS
        }
        proof_bytes = {
            k: [s[PROOF_FIELDS[k]] for s in samples]
            for k, _label, _title in LOOKUP_KINDS
        }
        publish_ms: dict[int, list[float]] = {}
        for b in publish_batches:
            publish_ms[int(b["batch_size"])] = list(b.get("samples_ms", []))
        audit_samples = lvl.get("audit", {}).get("samples", [])
        audit_ms = [s["audit_invariance_ns"] / 1e6 for s in audit_samples]
        audit_bytes = [s["audit_proof_bytes"] for s in audit_samples]
        cs_kind, cs_samples = _extract_concurrency_sweep(lvl)
        out.append(LevelStats(
            fill_percent=float(fp),
            true_log_capacity=tlc,
            server_ms=server_ms,
            client_ms=client_ms,
            proof_bytes=proof_bytes,
            publish_ms=publish_ms,
            audit_ms=audit_ms,
            audit_bytes=audit_bytes,
            concurrency_sweep_kind=cs_kind,
            concurrency_sweep=cs_samples,
        ))
    out.sort(key=lambda x: x.fill_percent)
    return out


def load_akd_small() -> list[LevelStats]:
    return _load_akd_json(AKD_DIR / "small.json")


def load_akd_medium() -> list[LevelStats]:
    return _load_akd_json(AKD_DIR / "medium.json")


def load_akd_large() -> list[LevelStats]:
    """AKD-large publish numbers are extrapolated from small/medium
    (see akd-mysql/large.json `_note`). Bench was not measurable
    end-to-end before credits ran out."""
    return _load_akd_json(AKD_DIR / "large.json")


@dataclass
class IronStats:
    """Per-regime irondict measurements. irondict has no fill axis,
    so each regime is summarised by a single set of scalars (lookup
    time, audit time, proof bytes, …) and a per-batch-size publish
    table — much flatter than aegon's `LevelStats`. Plot helpers
    overlay these as horizontal lines on the fill-axis plots and as
    direct curves on the publish-vs-batch plot.
    """
    log_capacity: int
    regime: str
    setup_ms: float | None
    server_lookup_ms: float | None
    client_lookup_ms: float | None
    proof_bytes: int | None
    client_key_bytes: int | None
    audit_ms: float | None
    audit_bytes: int | None
    publish_keys: list[tuple[int, float, int | None]]  # (batch_size, median_ms, bulletin_bytes)
    publish_reg: list[tuple[int, float, int | None]]


def _load_irondict_json(path: Path) -> IronStats | None:
    """Load one irondict JSON written by `bench-results/irondict/parse_divan.py`.
    Returns None if the file is missing or has no measured rows yet
    (e.g. the large.json file is empty until the m1-megamem bench
    finishes). The plot helpers skip None overlays gracefully.
    """
    if not path.exists():
        return None
    with path.open() as fh:
        d = json.load(fh)
    params = d.get("params", {})
    lookup = d.get("lookup", {}) or {}
    audit = d.get("audit", {}) or {}
    pk_rows = (d.get("publish_keys") or {}).get("by_batch", []) or []
    pr_rows = (d.get("publish_reg") or {}).get("by_batch", []) or []
    pk = [
        (int(r["batch_size"]), float(r["median_ms"]), r.get("bulletin_bytes"))
        for r in pk_rows if r.get("median_ms") is not None
    ]
    pr = [
        (int(r["batch_size"]), float(r["median_ms"]), r.get("bulletin_bytes"))
        for r in pr_rows if r.get("median_ms") is not None
    ]
    # If literally nothing is populated, treat as "no irondict data
    # for this regime" — so the plot helpers can short-circuit.
    has_any = any([
        d.get("setup_ms") is not None,
        audit.get("audit_ms") is not None,
        lookup.get("server_ms") is not None,
        lookup.get("client_ms") is not None,
        pk, pr,
    ])
    if not has_any:
        return None
    return IronStats(
        log_capacity=int(params.get("log_capacity", 0)),
        regime=str(params.get("regime", "")),
        setup_ms=d.get("setup_ms"),
        server_lookup_ms=lookup.get("server_ms"),
        client_lookup_ms=lookup.get("client_ms"),
        proof_bytes=lookup.get("proof_bytes"),
        client_key_bytes=lookup.get("client_key_bytes"),
        audit_ms=audit.get("audit_ms"),
        audit_bytes=audit.get("audit_bytes"),
        publish_keys=pk,
        publish_reg=pr,
    )


def load_irondict_small() -> IronStats | None:
    return _load_irondict_json(IRONDICT_DIR / "small.json")


def load_irondict_medium() -> IronStats | None:
    return _load_irondict_json(IRONDICT_DIR / "medium.json")


def load_irondict_large() -> IronStats | None:
    return _load_irondict_json(IRONDICT_DIR / "large.json")


def percentile(vs: list[float], q: float) -> float:
    if not vs:
        return float("nan")
    return float(np.percentile(vs, q))


def plot_server_lookup_vs_fill(small: list[LevelStats], medium: list[LevelStats], large: list[LevelStats]) -> list[Path]:
    """One panel per lookup kind; small + medium + large.

    Solid lines are measured (median, 10th–90th percentile band).
    For the large regime we also draw a dashed extrapolation at
    medium's high fills (30/60/90) using medium's growth ratio
    anchored at large's measured 1% — same approach as the proof-size
    plot, since medium and large share identical per-probe structure.
    """
    basic_kinds = [k for k in LOOKUP_KINDS if k[0] in {"label", "value"}]
    history_kinds = [k for k in LOOKUP_KINDS if k[0] in {"label_history", "history"}]
    return [
        _plot_server_lookup_panels(small, medium, large, basic_kinds,
                                   "server_lookup_vs_fill.pdf"),
        _plot_server_lookup_panels(small, medium, large, history_kinds,
                                   "server_lookup_history_vs_fill.pdf"),
    ]


def _plot_server_lookup_panels(
    small: list[LevelStats],
    medium: list[LevelStats],
    large: list[LevelStats],
    kinds: list[tuple[str, str, str]],
    out_filename: str,
) -> Path:
    """Render a single server-lookup figure with one panel per kind.

    Used to split the four lookup operations across two figures (basic
    label/value, and history-flavoured operations) while keeping the
    plot logic in one place.
    """
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.2 * len(kinds), 3), sharex=True)
    if len(kinds) == 1:
        axes = [axes]
    large_style = REGIME_STYLES["large"]
    for ax, (key, _short, title) in zip(axes, kinds):
        for regime_name, regime_levels in [("small", small), ("medium", medium), ("large", large)]:
            if not regime_levels:
                continue
            style = REGIME_STYLES[regime_name]
            xs = np.array([lvl.fill_percent for lvl in regime_levels])
            p50 = np.array([percentile(lvl.server_ms[key], 50) for lvl in regime_levels])
            p10 = np.array([percentile(lvl.server_ms[key], 10) for lvl in regime_levels])
            p90 = np.array([percentile(lvl.server_ms[key], 90) for lvl in regime_levels])
            eps = 1e-4
            p50 = np.maximum(p50, eps)
            p10 = np.maximum(p10, eps)
            p90 = np.maximum(p90, eps)
            ax.fill_between(xs, p10, p90, color=style["color"], alpha=0.18, linewidth=0)
            ax.plot(
                xs,
                p50,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=6,
                label=style["label"],
            )

        ext = _large_metric_theoretical_extrapolation(
            large, lambda lvl, k=key: lvl.server_ms.get(k, [])
        )
        if ext is not None:
            xs_ext, ys_ext = ext
            measured_large = [l for l in large if l.fill_percent <= 10.5]
            if measured_large:
                last = measured_large[-1]
                last_med = float(np.median(last.server_ms[key]))
                xs_full = np.concatenate(([last.fill_percent], xs_ext))
                ys_full = np.concatenate(([last_med], ys_ext))
            else:
                xs_full, ys_full = xs_ext, ys_ext
            ys_full = np.maximum(ys_full, 1e-4)
            ax.plot(
                xs_full,
                ys_full,
                color=large_style["color"],
                marker=large_style["marker"],
                markerfacecolor="none",
                linestyle="--",
                linewidth=1.4,
                markersize=6,
                alpha=0.85,
                label="aegon large (extrapolated)",
            )

        ax.set_xlabel("preload fill (% of true capacity)")
        ax.set_ylabel(f"median {title}\nlatency (ms)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks([1, 30, 60, 90])

    handles, labels = axes[0].get_legend_handles_labels()
    pairs = list(zip(handles, labels))
    row1 = [(h, l) for h, l in pairs if not l.startswith("large")]
    row2 = [(h, l) for h, l in pairs if l.startswith("large")]
    for ax in fig.axes:
        _label_axis_endpoints(ax)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    if row1:
        fig.legend(
            [h for h, _ in row1], [l for _, l in row1],
            loc="lower center", ncol=len(row1),
            bbox_to_anchor=(0.5, 0.08), frameon=False,
        )
    if row2:
        fig.legend(
            [h for h, _ in row2], [l for _, l in row2],
            loc="lower center", ncol=len(row2),
            bbox_to_anchor=(0.5, 0.0), frameon=False,
        )

    out_path = PLOTS_DIR / out_filename
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_audit_vs_fill(
    small: list[LevelStats],
    medium: list[LevelStats],
    large: list[LevelStats],
    subset: tuple[str, ...],
    agg: str = AGG_MEDIAN,
    akd: dict[str, list[LevelStats]] | None = None,
    irondict: dict[str, IronStats] | None = None,
) -> Path:
    """Audit figure: 2 side-by-side panels.

    Left:  auditor verify latency vs preload fill (one line per regime).
    Right: audit proof size (bytes) vs preload fill, same regimes.

    Both are expected to be ~flat with fill — the audit proof structure
    is per-epoch and per-shard, independent of dictionary state. What
    changes between regimes is the shard count: large's proof carries
    ~37 KB of cross-shard merkle data vs medium's 640 B and small's
    344 B.

    `agg` is 'median' or 'max'. Large extrapolation runs only in the
    median view (the helper uses np.median internally), so the max
    view is measured-only.

    `subset` filters which regimes are RENDERED as lines, but the
    large-extrapolation predictor internally needs `medium` as a
    reference, so we always pass medium to that helper regardless of
    rendering.
    """
    q_lo, q_mid, q_hi = _percentiles_for(agg)
    reducer = _reducer_for(agg)
    fig, (ax_time, ax_size) = plt.subplots(1, 2, figsize=(6.4, 3))

    large_style = REGIME_STYLES["large"]
    in_subset = set(subset)
    render_small = "small" in in_subset
    render_medium = "medium" in in_subset
    render_large = "large" in in_subset

    line_regimes: list[tuple[str, list[LevelStats]]] = []
    if render_small:
        line_regimes.append(("small", small))
    if render_medium:
        line_regimes.append(("medium", medium))
    if render_large:
        line_regimes.append(("large", large))

    # ---- LEFT: verify latency ----
    for regime_name, regime_levels in line_regimes:
        regime_levels = [lvl for lvl in regime_levels if lvl.audit_ms]
        if not regime_levels:
            continue
        style = REGIME_STYLES[regime_name]
        xs = np.array([lvl.fill_percent for lvl in regime_levels])
        central = np.array([percentile(lvl.audit_ms, q_mid) for lvl in regime_levels])
        low = np.array([percentile(lvl.audit_ms, q_lo) for lvl in regime_levels])
        high = np.array([percentile(lvl.audit_ms, q_hi) for lvl in regime_levels])
        ax_time.fill_between(xs, low, high, color=style["color"], alpha=0.18, linewidth=0)
        ax_time.plot(
            xs, central,
            color=style["color"], marker=style["marker"], linewidth=1.8,
            markersize=6, label=style["label"],
        )

    if render_large:
        ext = _large_metric_extrapolation(large, medium, lambda lvl: lvl.audit_ms, reducer=reducer)
        if ext is not None:
            xs_ext, ys_ext = ext
            measured_large = [l for l in large if l.fill_percent <= 10.5 and l.audit_ms]
            if measured_large:
                last = measured_large[-1]
                last_anchor = float(reducer(last.audit_ms))
                xs_full = np.concatenate(([last.fill_percent], xs_ext))
                ys_full = np.concatenate(([last_anchor], ys_ext))
            else:
                xs_full, ys_full = xs_ext, ys_ext
            ax_time.plot(
                xs_full, ys_full,
                color=large_style["color"], marker=large_style["marker"],
                markerfacecolor="none", linestyle="--", linewidth=1.4,
                markersize=6, alpha=0.85, label="aegon large (extrapolated)",
            )

    ax_time.set_xlabel("preload fill (% of true capacity)")
    ax_time.set_ylabel(f"{_agg_label(agg)} auditor verify\nlatency (ms)")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.set_xticks([1, 30, 60, 90])

    # ---- RIGHT: proof size in KB ----
    for regime_name, regime_levels in line_regimes:
        regime_levels = [lvl for lvl in regime_levels if lvl.audit_bytes]
        if not regime_levels:
            continue
        style = REGIME_STYLES[regime_name]
        xs = np.array([lvl.fill_percent for lvl in regime_levels])
        # Convert bytes → KB. The proof size is identical per sample at
        # a given regime+fill, so median/p10/p90/max all collapse to a
        # single value — no shaded band needed even under `agg=max`.
        central = np.array([percentile(lvl.audit_bytes, q_mid) for lvl in regime_levels]) / 1024.0
        ax_size.plot(
            xs, central,
            color=style["color"], marker=style["marker"], linewidth=1.8,
            markersize=6, label=style["label"],
        )

    # Large proof-size extrapolation. The audit proof structure carries
    # one merkle commitment per shard and is fill-independent by
    # construction, so the projected curve is simply the measured
    # large value carried over to 30/60/90% (medium's growth ratio is
    # 1.0 — also flat — so _large_metric_extrapolation produces this
    # naturally and we stay consistent with the other panels).
    ext = _large_metric_extrapolation(large, medium, lambda lvl: lvl.audit_bytes, reducer=reducer) if render_large else None
    if ext is not None:
        xs_ext, ys_ext = ext
        measured_large = [l for l in large if l.fill_percent <= 10.5 and l.audit_bytes]
        if measured_large:
            last = measured_large[-1]
            last_anchor = float(reducer(last.audit_bytes))
            xs_full = np.concatenate(([last.fill_percent], xs_ext))
            ys_full = np.concatenate(([last_anchor], ys_ext))
        else:
            xs_full, ys_full = xs_ext, ys_ext
        ax_size.plot(
            xs_full, ys_full / 1024.0,
            color=large_style["color"], marker=large_style["marker"],
            markerfacecolor="none", linestyle="--", linewidth=1.4,
            markersize=6, alpha=0.85, label="aegon large (extrapolated)",
        )

    ax_size.set_xlabel("preload fill (% of true capacity)")
    ax_size.set_ylabel(f"{_agg_label(agg)} audit proof\nsize (KB)")
    ax_size.grid(True, which="both", alpha=0.3)
    ax_size.set_xticks([1, 30, 60, 90])

    # AKD-MySQL overlay. Audit proofs in AKD are an entire epoch's
    # cross-shard merkle data (~13 MB) vs aegon's per-shard
    # commitments (~640 B for medium), so the y-axis switches to log
    # when AKD is present to keep both readable.
    if akd:
        akd_regimes = [(n, akd[n]) for n in subset if n in akd and akd[n]]
        for akd_name, akd_levels in akd_regimes:
            astyle = AKD_STYLES[akd_name]
            lvls_time = [lvl for lvl in akd_levels if lvl.audit_ms]
            if lvls_time:
                xs = np.array([lvl.fill_percent for lvl in lvls_time])
                central = np.array([percentile(lvl.audit_ms, q_mid) for lvl in lvls_time])
                ax_time.plot(
                    xs, central,
                    color=astyle["color"], marker=astyle["marker"],
                    linestyle=astyle["linestyle"], markerfacecolor="none",
                    linewidth=1.6, markersize=6, label=astyle["label"],
                )
            lvls_size = [lvl for lvl in akd_levels if lvl.audit_bytes]
            if lvls_size:
                xs = np.array([lvl.fill_percent for lvl in lvls_size])
                central = np.array([percentile(lvl.audit_bytes, q_mid) for lvl in lvls_size]) / 1024.0
                ax_size.plot(
                    xs, central,
                    color=astyle["color"], marker=astyle["marker"],
                    linestyle=astyle["linestyle"], markerfacecolor="none",
                    linewidth=1.6, markersize=6, label=astyle["label"],
                )
        # Aegon audit costs are O(shards) bytes / ~1 ms; AKD's are
        # O(epoch delta) MB / ~100 ms. Linear axes would collapse the
        # aegon curves into the x-axis — log lets both regimes read.
        ax_time.set_yscale("log")
        ax_size.set_yscale("log")

    # irondict overlay. irondict has no fill axis — every audit is
    # measured on a near-empty tree — so each metric reads as a
    # horizontal dash-dot line spanning the panel. The audit
    # payload size is derived from the publish bulletin sizes
    # (verify_update checks one publish_reg msg + one publish_keys
    # msg per epoch; see parse_divan.py _derive_audit_bytes).
    if irondict:
        iron_xlim_time = ax_time.get_xlim()
        iron_xlim_size = ax_size.get_xlim()
        # Same as the lookup plot: irondict has one measurement and
        # we render it at each of aegon's fill x-positions so the
        # marker reads on the panel and the legend handle matches.
        iron_xs = np.array([1.0, 30.0, 60.0, 90.0])
        any_iron_time = False
        any_iron_size = False
        for iron_name in subset:
            iron = irondict.get(iron_name)
            if iron is None:
                continue
            istyle = IRONDICT_STYLES[iron_name]
            if iron.audit_ms is not None:
                ax_time.plot(
                    iron_xs, np.full_like(iron_xs, iron.audit_ms),
                    color=istyle["color"], linestyle=istyle["linestyle"],
                    marker=istyle["marker"], markerfacecolor="none",
                    linewidth=1.6, markersize=6, label=istyle["label"],
                )
                any_iron_time = True
            if iron.audit_bytes is not None:
                v_kb = iron.audit_bytes / 1024.0
                ax_size.plot(
                    iron_xs, np.full_like(iron_xs, v_kb),
                    color=istyle["color"], linestyle=istyle["linestyle"],
                    marker=istyle["marker"], markerfacecolor="none",
                    linewidth=1.6, markersize=6, label=istyle["label"],
                )
                any_iron_size = True
        ax_time.set_xlim(iron_xlim_time)
        ax_size.set_xlim(iron_xlim_size)
        if any_iron_time:
            ax_time.set_yscale("log")
        if any_iron_size:
            ax_size.set_yscale("log")

    # Two-row shared legend (matches the server / client / proof-size
    # layout): small + medium on top, large + large-extrapolated below.
    handles, labels = ax_time.get_legend_handles_labels()
    pairs = list(zip(handles, labels))
    row1 = [(h, l) for h, l in pairs if not l.startswith("large")]
    row2 = [(h, l) for h, l in pairs if l.startswith("large")]
    for ax in fig.axes:
        _label_axis_endpoints(ax)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    if row1:
        fig.legend(
            [h for h, _ in row1], [l for _, l in row1],
            loc="lower center", ncol=len(row1),
            bbox_to_anchor=(0.5, 0.08), frameon=False,
        )
    if row2:
        fig.legend(
            [h for h, _ in row2], [l for _, l in row2],
            loc="lower center", ncol=len(row2),
            bbox_to_anchor=(0.5, 0.0), frameon=False,
        )

    out_path = PLOTS_DIR / f"audit_vs_fill{_subset_suffix(subset)}{_agg_suffix(agg)}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _publish_log_ticks(axes_with_regimes) -> None:
    """Configure publish-plot log axes per regime.

    Drops endpoint labels (their varying width was making the three
    panels render at different sizes). x-axis uses decade-only ticks
    (wide mathtext sci labels overlap on the narrow small panel).
    y-axis uses an explicit per-regime tick set so the bottom-most
    subdecade tick is dropped — it sits right at the bottom of each
    panel's data range and reads as visual noise.
    """
    y_ticks_per_regime = {
        "small":  [0.01, 0.02, 0.05],
        "medium": [0.1, 0.2, 0.5, 1.0],
    }
    for ax, regime_name in axes_with_regimes:
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=20))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_log_tick))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        explicit = y_ticks_per_regime.get(regime_name)
        if explicit is not None:
            ax.set_yticks(explicit)
        else:
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=20))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_log_tick))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())


def plot_publish_vs_batch(
    small: list[LevelStats],
    medium: list[LevelStats],
    large: list[LevelStats],
    subset: tuple[str, ...],
    agg: str = AGG_MEDIAN,
    akd: dict[str, list[LevelStats]] | None = None,
    irondict: dict[str, IronStats] | None = None,
) -> Path:
    """Publish time vs. batch size, faceted by regime, one curve per
    fill level (color = fill %).

    `subset` selects which regimes to render: ('small', 'medium') for
    the small+medium figure, ('large',) for the large-alone figure.
    `agg` is 'median' or 'max' — the central line traces the median
    (50th percentile) or the slowest sample respectively; the band
    shows p10..p90 or min..max. The large-regime extrapolation curve
    is median-anchored so it's drawn only in the median view; the
    max view shows measured data only.

    x-axis is the absolute publish batch size (number of updates per
    epoch). Panels keep independent x-axes (sharex=False) since
    regime batch ranges don't overlap.
    """
    q_lo, q_mid, q_hi = _percentiles_for(agg)
    all_regimes = {"small": small, "medium": medium, "large": large}
    regimes = [(n, all_regimes[n]) for n in subset if n in all_regimes]
    populated = [(n, lvls) for n, lvls in regimes if lvls]
    out_path = PLOTS_DIR / f"publish_vs_batch{_subset_suffix(subset)}{_agg_suffix(agg)}.pdf"
    if not populated:
        fig, _ = plt.subplots(figsize=(4, 2.8))
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return out_path

    # Per-panel y-axis (sharey=False) so the large panel can show its
    # extrapolated values without squishing small/medium.
    fig, axes = plt.subplots(1, len(populated), figsize=(3.2 * len(populated), 3), sharey=False)
    if len(populated) == 1:
        axes = [axes]
    # Softer rainbow palette: red at fill=0, blue at fill=100.
    # `Spectral` is the diverging red→yellow→green→blue ramp —
    # rainbow-shaped but desaturated, so it doesn't visually
    # overpower the latency band/markers the way `jet_r` does.
    cmap = plt.get_cmap("Spectral")

    # Empirical power-law extrapolation for large publish.
    # The earlier open-addressing theoretical scaling (probe count ~
    # 1/(1-α), capped at 1.29× from 1→90%) DRAMATICALLY underpredicts
    # what large actually does: at batch=131072, theory says 90% ≈
    # 25s but we measured 42.5s already at 4% fill. The gap is
    # RocksDB compaction — write-amp climbs as working set crosses
    # coord RAM, and that effect is missing from the algorithmic
    # cost model.
    #
    # Fit log(time) = a + b·log(fill_pct) per batch size on large's
    # measured 1-4% data, extrapolate to 30/60/90%. This is the
    # most-honest available prediction given the data we have —
    # acknowledged caveat: the trend may overshoot if RocksDB
    # eventually stabilizes at higher fills (we couldn't test past
    # 4% due to bench-client OOM at 5%).
    EXTRAP_FILLS_PUBLISH = (30.0, 60.0, 90.0)
    def _empirical_publish_fit(
        anchor_points: list[tuple[float, float]],  # (fill_pct, median_ms)
        target_fill: float,
    ) -> float | None:
        if len(anchor_points) < 2:
            return None
        xs = np.log(np.array([p[0] for p in anchor_points]))
        ys = np.log(np.array([p[1] for p in anchor_points]))
        slope, intercept = np.polyfit(xs, ys, 1)
        return float(np.exp(intercept + slope * np.log(target_fill)))

    # Per-figure rank-based fill→color map. The smallest fill in this
    # figure pins to deepest red, the largest to deepest blue, and
    # intermediate fills are evenly distributed along the colormap.
    # The user's complaint with the previous `fill/100` mapping was
    # that the large plot's measured fills (1%, 2%, …, 10%) all
    # collapsed into the deep-red end and were indistinguishable;
    # rank-based guarantees a clean visual step between every line.
    # Cost: a given fill_percent (e.g. 30%) reads as a different
    # colour across the small+medium and large figures, since each
    # figure's available fill set is different. The legend in every
    # panel labels lines by fill % so this is unambiguous.
    # Aegon-large derives fill_percent from preload_count / true_cap,
    # so values arrive as e.g. 0.9999999776 instead of exactly 1.0.
    # Round every fill key to 4 decimals so AKD's exact 1.0/30.0/etc.
    # find the matching colormap slot.
    def _fkey(f: float) -> float:
        return round(float(f), 4)

    all_fills_global: set[float] = set()
    for _name, _lvls in populated:
        for _lvl in _lvls:
            if _lvl.publish_ms:
                all_fills_global.add(_fkey(_lvl.fill_percent))
        if _name == "large":
            all_fills_global.update(_fkey(f) for f in EXTRAP_FILLS_PUBLISH)
    global_fills_sorted = sorted(all_fills_global)
    n_fills = len(global_fills_sorted)
    if n_fills <= 1:
        fill_to_color = {f: 0.0 for f in global_fills_sorted}
    else:
        fill_to_color = {f: i / (n_fills - 1) for i, f in enumerate(global_fills_sorted)}

    for ax, (regime_name, regime_levels) in zip(axes, populated):
        if subset == SMALL_MEDIUM:
            ax.set_title(regime_name.capitalize())
        levels_sorted = sorted(regime_levels, key=lambda l: l.fill_percent)
        measured_fills = [lvl.fill_percent for lvl in levels_sorted if lvl.publish_ms]
        if not measured_fills:
            continue
        # Add extrapolated fills only for large (open-addressing
        # theory scaling, projected to 30/60/90 %).
        extra_fills = list(EXTRAP_FILLS_PUBLISH) if regime_name == "large" else []

        for lvl in [l for l in levels_sorted if l.publish_ms]:
            color = cmap(fill_to_color[_fkey(lvl.fill_percent)])
            batch_sizes = np.array(sorted(lvl.publish_ms))
            xs = batch_sizes.astype(float)
            central = np.array([percentile(lvl.publish_ms[b], q_mid) for b in batch_sizes]) / 1000.0
            low = np.array([percentile(lvl.publish_ms[b], q_lo) for b in batch_sizes]) / 1000.0
            high = np.array([percentile(lvl.publish_ms[b], q_hi) for b in batch_sizes]) / 1000.0
            ax.fill_between(xs, low, high, color=color, alpha=0.18, linewidth=0)
            ax.plot(
                xs,
                central,
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=f"aegon {lvl.fill_percent:g}%",
            )

        # Theoretical open-addressing extrapolation for large.
        # The measured 1–10% curves are well-fit by linear models
        # `latency = m·batch_pct + c` (R² ≥ 0.998). The slope m is
        # proportional to per-element insertion cost, which under
        # open-addressing scales as 1/(1−α), where α is the
        # polynomial-side load factor — `fill_pct / (100·OVER_PROV)`
        # with the paper's OVER_PROV = 4. We strip the 1/(1−α) factor
        # from each measured slope and average across the 10 measured
        # fills to get a robust per-element cost `slope_0`. Each
        # extrapolated curve is then a line through the origin:
        # `y = slope_0 / (1 − α_target) · x`.
        #
        # Both median and max views extrapolate. The anchor fit is
        # derived from whichever percentile matches `agg` (p50 for
        # median, p100 for max) so the dashed projection stays
        # consistent with the measured solid line beneath it.
        if regime_name == "large" and extra_fills:
            OVER_PROV = 4.0
            measured = sorted(
                [l for l in levels_sorted if l.publish_ms and l.fill_percent > 0.0],
                key=lambda l: l.fill_percent,
            )
            if measured:
                # Anchor at the HIGHEST measured fill (10%) — closest to
                # the extrapolation targets so the theoretical line stays
                # monotonically above all measured curves, which a global
                # average doesn't guarantee in the presence of 1–10%
                # slope noise. We carry the anchor's intercept across to
                # the extrapolated lines because the constant overhead
                # (gRPC, serialisation, fixed compaction work) doesn't
                # scale with α; only the per-element probe count does.
                anchor = measured[-1]
                bs_anchor = sorted(anchor.publish_ms.keys())
                xs_anchor = np.array(bs_anchor, dtype=float)
                ys_anchor = np.array([
                    percentile(anchor.publish_ms[b], q_mid) / 1000.0
                    for b in bs_anchor
                ])
                m_anchor, c_anchor = np.polyfit(xs_anchor, ys_anchor, 1)
                alpha_anchor = anchor.fill_percent / (100.0 * OVER_PROV)
                slope_zero = m_anchor * (1.0 - alpha_anchor)

                union_batches = sorted(
                    set().union(*(set(l.publish_ms.keys()) for l in measured))
                )
                xs = np.array(union_batches, dtype=float)
                for f in extra_fills:
                    alpha = f / (100.0 * OVER_PROV)
                    slope_f = slope_zero / (1.0 - alpha)
                    ys = slope_f * xs + c_anchor
                    color = cmap(fill_to_color[_fkey(f)])
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        marker="o",
                        markerfacecolor="none",
                        linewidth=1.4,
                        markersize=5,
                        linestyle="--",
                        alpha=0.9,
                        label=f"aegon {f:g}% (extrapolated)",
                    )

        # irondict overlay for this panel's regime. irondict has no
        # fill axis — every publish is measured on a near-empty tree
        # — so we get ONE curve per regime spanning the batch range.
        # Plotted in neutral grey (since aegon already paints every
        # fill in the rainbow palette and there's no fill it
        # naturally pairs with), dash-dot + open square so a single
        # sentinel handle below the panel loop explains the line.
        #
        # irondict's batches span 16..131072 — wider than any single
        # aegon regime's range (small: 2..64, medium: 64..2048,
        # large: 4096..131072) — so we clip to the union of aegon's
        # measured batches in THIS panel, with one batch of padding
        # on each side. Without the clip, the small panel's x-axis
        # auto-scales from 2 → 131072 and the aegon curves collapse
        # into the leftmost ~5% of the panel.
        if irondict and regime_name in irondict:
            iron = irondict[regime_name]
            if iron is not None and iron.publish_keys:
                aegon_batches = set()
                for lvl in levels_sorted:
                    aegon_batches.update(lvl.publish_ms.keys())
                if aegon_batches:
                    bs_min = min(aegon_batches)
                    bs_max = max(aegon_batches)
                    pts = [
                        (bs, ms) for bs, ms, _ in iron.publish_keys
                        if bs_min <= bs <= bs_max
                    ]
                else:
                    pts = [(bs, ms) for bs, ms, _ in iron.publish_keys]
                if pts:
                    xs_iron = np.array([bs for bs, _ in pts], dtype=float)
                    ys_iron = np.array([ms for _, ms in pts]) / 1000.0
                    istyle = IRONDICT_STYLES[regime_name]
                    ax.plot(
                        xs_iron, ys_iron,
                        color="#444444",
                        marker=istyle["marker"], markerfacecolor="none",
                        linestyle=istyle["linestyle"],
                        linewidth=1.6, markersize=5,
                        label="irondict 0%",
                    )

        # AKD-MySQL overlay for this panel's regime. Color matches the
        # corresponding fill in the aegon panel (so a reader can
        # vertically align AKD's fill=30% curve with aegon's fill=30%
        # curve), linestyle is dashed and marker hollow to encode
        # "system = AKD-MySQL". No per-fill labels — they'd duplicate
        # aegon's legend rows; a single sentinel handle is appended
        # below the panel loop instead.
        if akd and regime_name in akd:
            # AKD-large is extrapolated from small+medium (see
            # akd-mysql/large.json _note); flag it in the legend so
            # the reader can distinguish from measured AKD curves.
            akd_suffix = " (extrapolation)" if regime_name == "large" else ""
            for lvl in sorted(akd[regime_name], key=lambda l: l.fill_percent):
                if not lvl.publish_ms:
                    continue
                if _fkey(lvl.fill_percent) not in fill_to_color:
                    continue
                color = cmap(fill_to_color[_fkey(lvl.fill_percent)])
                batch_sizes = np.array(sorted(lvl.publish_ms))
                xs_a = batch_sizes.astype(float)
                ys_a = np.array([percentile(lvl.publish_ms[b], q_mid) for b in batch_sizes]) / 1000.0
                ax.plot(
                    xs_a, ys_a,
                    color=color, marker="D", markerfacecolor="none",
                    linestyle="-", linewidth=1.4, markersize=5,
                    label=f"AKD {lvl.fill_percent:g}%{akd_suffix}",
                )

        ax.set_xlabel("publish batch size (updates)")
        ax.grid(True, which="both", alpha=0.3)
        # No per-panel legend — handles roll up into one shared legend
        # below all panels (see fig.legend after the loop).

    for ax in axes:
        ax.set_ylabel("publish latency (s)")
    # Large publish spans ~5 orders of magnitude with the manual
    # irondict 15-min entry; linear y collapses the aegon curves to
    # the x-axis. Use log y for the large subset.
    if "large" in subset:
        for ax in axes:
            ax.set_yscale("log")

    # One shared legend below the row of panels. Collect handles from
    # every axis, then dedupe by label so a fill that appears in two
    # regimes is only shown once. Labels are now system-prefixed
    # ("aegon 30%", "AKD 30%", "irondict 0%") so they group by system
    # row and sort by fill within each row.
    seen: dict[str, "matplotlib.artist.Artist"] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)

    def _parse_label(lbl: str) -> tuple[int, float, int]:
        """(system_rank, fill_percent, extrap_tiebreak). Rows are
        keyed by system_rank (aegon=0, AKD=1, irondict=2)."""
        if lbl.startswith("aegon"):
            sys_rank = 0
        elif lbl.startswith("AKD"):
            sys_rank = 1
        elif lbl.startswith("irondict"):
            sys_rank = 2
        else:
            sys_rank = 9
        extrap = 1 if "extrap" in lbl else 0
        # Pull the first "<num>%" token.
        import re as _re
        m = _re.search(r"(\d+(?:\.\d+)?)%", lbl)
        num = float(m.group(1)) if m else float("inf")
        return (sys_rank, num, extrap)

    ordered = sorted(seen.items(), key=lambda kv: _parse_label(kv[0]))
    if ordered:
        # One row per system (aegon, AKD, irondict). Then chunk each
        # row into pieces of at most MAX_NCOL so a system with many
        # entries (e.g. aegon-large has 10 measured + 3 extrapolated)
        # wraps cleanly instead of overflowing horizontally.
        MAX_NCOL = 7
        per_system: list[list[tuple[str, object]]] = [[], [], []]
        for label, handle in ordered:
            sys_rank, _, _ = _parse_label(label)
            if sys_rank <= 2:
                per_system[sys_rank].append((label, handle))
        rows: list[list[tuple[str, object]]] = []
        for system_row in per_system:
            if not system_row:
                continue
            for i in range(0, len(system_row), MAX_NCOL):
                rows.append(system_row[i : i + MAX_NCOL])
        rect_map = {1: 0.10, 2: 0.18, 3: 0.26, 4: 0.32, 5: 0.38}
        fig.tight_layout(rect=(0, rect_map.get(len(rows), 0.10), 1, 1))
        y_per_row = 0.08
        y_start = 0.02 + y_per_row * (len(rows) - 1)
        for i, row in enumerate(rows):
            row_labels, row_handles = zip(*row)
            fig.legend(
                row_handles, row_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, y_start - i * y_per_row),
                ncol=len(row_labels), frameon=False, fontsize=9,
            )
    else:
        fig.tight_layout()

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_client_lookup_vs_fill(small: list[LevelStats], medium: list[LevelStats], large: list[LevelStats]) -> list[Path]:
    basic_kinds = [k for k in LOOKUP_KINDS if k[0] in {"label", "value"}]
    history_kinds = [k for k in LOOKUP_KINDS if k[0] in {"label_history", "history"}]
    return [
        _plot_client_lookup_panels(small, medium, large, basic_kinds,
                                   "client_lookup_vs_fill.pdf"),
        _plot_client_lookup_panels(small, medium, large, history_kinds,
                                   "client_lookup_history_vs_fill.pdf"),
    ]


def _plot_client_lookup_panels(
    small: list[LevelStats],
    medium: list[LevelStats],
    large: list[LevelStats],
    kinds: list[tuple[str, str, str]],
    out_filename: str,
) -> Path:
    """Render a single client-lookup figure with one panel per kind.

    Used to split the four lookup operations across two figures (basic
    label/value, and history-flavoured operations) while keeping the
    plot logic in one place. Each curve is the median client-side
    verify time; the shaded band spans the 10th–90th percentile.
    """
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.2 * len(kinds), 3), sharex=True)
    if len(kinds) == 1:
        axes = [axes]
    large_style = REGIME_STYLES["large"]
    for ax, (key, _short, title) in zip(axes, kinds):
        for regime_name, regime_levels in [("small", small), ("medium", medium), ("large", large)]:
            if not regime_levels:
                continue
            style = REGIME_STYLES[regime_name]
            xs = np.array([lvl.fill_percent for lvl in regime_levels])
            p50 = np.array([percentile(lvl.client_ms[key], 50) for lvl in regime_levels])
            p10 = np.array([percentile(lvl.client_ms[key], 10) for lvl in regime_levels])
            p90 = np.array([percentile(lvl.client_ms[key], 90) for lvl in regime_levels])
            eps = 1e-4
            p50 = np.maximum(p50, eps)
            p10 = np.maximum(p10, eps)
            p90 = np.maximum(p90, eps)
            ax.fill_between(xs, p10, p90, color=style["color"], alpha=0.18, linewidth=0)
            ax.plot(
                xs,
                p50,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=6,
                label=style["label"],
            )

        ext = _large_metric_theoretical_extrapolation(
            large, lambda lvl, k=key: lvl.client_ms.get(k, [])
        )
        if ext is not None:
            xs_ext, ys_ext = ext
            measured_large = [l for l in large if l.fill_percent <= 10.5]
            if measured_large:
                last = measured_large[-1]
                last_med = float(np.median(last.client_ms[key]))
                xs_full = np.concatenate(([last.fill_percent], xs_ext))
                ys_full = np.concatenate(([last_med], ys_ext))
            else:
                xs_full, ys_full = xs_ext, ys_ext
            ys_full = np.maximum(ys_full, 1e-4)
            ax.plot(
                xs_full,
                ys_full,
                color=large_style["color"],
                marker=large_style["marker"],
                markerfacecolor="none",
                linestyle="--",
                linewidth=1.4,
                markersize=6,
                alpha=0.85,
                label="aegon large (extrapolated)",
            )

        ax.set_xlabel("preload fill (% of true capacity)")
        ax.set_ylabel(f"median {title}\nclient verify latency (ms)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks([1, 30, 60, 90])

    handles, labels = axes[0].get_legend_handles_labels()
    pairs = list(zip(handles, labels))
    row1 = [(h, l) for h, l in pairs if not l.startswith("large")]
    row2 = [(h, l) for h, l in pairs if l.startswith("large")]
    for ax in fig.axes:
        _label_axis_endpoints(ax)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    if row1:
        fig.legend(
            [h for h, _ in row1], [l for _, l in row1],
            loc="lower center", ncol=len(row1),
            bbox_to_anchor=(0.5, 0.08), frameon=False,
        )
    if row2:
        fig.legend(
            [h for h, _ in row2], [l for _, l in row2],
            loc="lower center", ncol=len(row2),
            bbox_to_anchor=(0.5, 0.0), frameon=False,
        )

    out_path = PLOTS_DIR / out_filename
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _project_via_medium_ratio(
    large: list[LevelStats],
    medium: list[LevelStats],
    getter,
    reducer=np.median,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Medium-ratio projection: apply medium's X(fill)/X(1%) to large's 1%.

    Medium and large share kzh_k=9, shard_log_capacity=27, and
    over-prov=4 — same per-probe opening structure and same open-
    addressing load factor (alpha = fill/400). So medium's relative
    growth above 4% is the best single-regime predictor for large at
    the same fills.

    `reducer` collapses each fill's sample list to a single scalar;
    `np.median` for typical-case curves, `np.max` for worst-case.
    """
    if not large or not medium:
        return None
    large_anchor = min(
        (l for l in large if l.fill_percent < 4.5),
        key=lambda l: abs(l.fill_percent - 1.0),
        default=None,
    )
    medium_anchor = next((l for l in medium if abs(l.fill_percent - 1.0) < 0.01), None)
    if large_anchor is None or medium_anchor is None:
        return None
    m_anchor_vals = getter(medium_anchor)
    l_anchor_vals = getter(large_anchor)
    if not m_anchor_vals or not l_anchor_vals:
        return None
    m1 = float(reducer(m_anchor_vals))
    l1 = float(reducer(l_anchor_vals))
    if m1 == 0:
        return None
    xs, ys = [], []
    for lvl in medium:
        if lvl.fill_percent <= 10.5:
            continue
        vals = getter(lvl)
        if not vals:
            continue
        mf = float(reducer(vals))
        xs.append(lvl.fill_percent)
        ys.append(l1 * mf / m1)
    if not xs:
        return None
    return np.array(xs), np.array(ys)


def _project_via_large_fit(
    large: list[LevelStats], getter, target_fills: np.ndarray,
    reducer=np.median,
) -> np.ndarray | None:
    """Log-log fit through large's measured 1-4% data, projected to target fills.

    Uses large's *own* measurements at its actual scale. The 1-4%
    fits a power law `log(y) = a + b * log(fill)`; the projection at
    `target_fills` assumes that local growth rate continues. Returns
    None when fewer than 2 valid anchor points are available or when
    any value is non-positive (log undefined).
    """
    measured = [l for l in large if l.fill_percent <= 10.5]
    valid: list[tuple[float, float]] = []
    for lvl in measured:
        vals = getter(lvl)
        if not vals:
            continue
        val = float(reducer(vals))
        if val <= 0:
            continue
        valid.append((lvl.fill_percent, val))
    if len(valid) < 2:
        return None
    xs_log = np.log(np.array([v[0] for v in valid]))
    ys_log = np.log(np.array([v[1] for v in valid]))
    slope, intercept = np.polyfit(xs_log, ys_log, 1)
    return np.exp(intercept + slope * np.log(target_fills))


def _large_metric_extrapolation(
    large: list[LevelStats],
    medium: list[LevelStats],
    getter,
    reducer=np.median,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Combine medium-ratio and large's own 1-4% fit via geometric mean.

    Two complementary predictors disagree gracefully:
      - medium-ratio captures the SHAPE of fill-vs-metric at the same
        per-probe scale as large, but ignores any large-specific drift
        observed below 4%.
      - large's own log-log fit captures the LOCAL drift but assumes
        it continues — which it might not if the metric is approaching
        a steady-state asymptote.

    Geometric mean (sqrt of product) is the right way to "average"
    multiplicative predictions: it gives equal weight to each source
    in log space. Falls back to whichever single predictor is
    available if the other is missing.

    `getter(LevelStats) -> list[float]` selects the metric (per-sample
    list of proof bytes or latencies).

    Returns (xs, ys) for medium's high fills, or None if neither
    source has enough data.
    """
    medium_pred = _project_via_medium_ratio(large, medium, getter, reducer=reducer)
    if medium_pred is None:
        return None
    xs_target, ys_medium = medium_pred
    ys_large = _project_via_large_fit(large, getter, xs_target, reducer=reducer)
    if ys_large is None:
        return xs_target, ys_medium
    return xs_target, np.sqrt(ys_medium * ys_large)


def _large_metric_theoretical_extrapolation(
    large: list[LevelStats],
    getter,
    target_fills: tuple[float, ...] = (30.0, 60.0, 90.0),
    over_prov: float = 4.0,
    reducer=np.median,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Open-addressing theoretical extrapolation for lookup metrics.

    Assumes cost(fill) = C / (1 − ρ), where ρ = fill / (100·over_prov)
    is the polynomial-side load factor (paper's over-provisioning
    factor is 4 → ρ = fill/400). Anchors at the highest measured fill
    (10%) so any constant overhead is absorbed into C, then scales by
    (1 − ρ_anchor) / (1 − ρ_target). Same model and same anchor
    discipline as the publish extrapolation — guarantees the
    extrapolated 30/60/90% are monotonically above the measured 10%.
    """
    measured = sorted(
        [l for l in large if l.fill_percent <= 10.5],
        key=lambda l: l.fill_percent,
    )
    anchor_fill: float | None = None
    anchor_val: float | None = None
    for lvl in reversed(measured):
        vals = getter(lvl)
        if not vals:
            continue
        val = float(reducer(vals))
        if val <= 0:
            continue
        anchor_fill, anchor_val = lvl.fill_percent, val
        break
    if anchor_fill is None or anchor_val is None:
        return None
    rho_anchor = anchor_fill / (100.0 * over_prov)
    cost_per_probe = anchor_val * (1.0 - rho_anchor)
    xs = np.array(target_fills, dtype=float)
    ys = cost_per_probe / (1.0 - xs / (100.0 * over_prov))
    return xs, ys


def _large_proof_extrapolation(
    large: list[LevelStats], medium: list[LevelStats], key: str,
    reducer=np.median,
) -> tuple[np.ndarray, np.ndarray] | None:
    return _large_metric_extrapolation(
        large, medium, lambda lvl: lvl.proof_bytes.get(key, []),
        reducer=reducer,
    )


def plot_proof_size_vs_fill(small: list[LevelStats], medium: list[LevelStats], large: list[LevelStats]) -> list[Path]:
    """One panel per lookup kind; proof bytes for small + medium + large.

    Solid lines are measured. For the large regime we also draw a
    dashed extrapolation at fills 30/60/90% computed by transferring
    medium's growth ratio (medium has identical per-probe structure;
    only the Merkle-path depth differs, captured in large's 1% anchor).
    Each measured curve is the median; the shaded band is 10th-90th percentile.
    """
    basic_kinds = [k for k in LOOKUP_KINDS if k[0] in {"label", "value"}]
    history_kinds = [k for k in LOOKUP_KINDS if k[0] in {"label_history", "history"}]
    return [
        _plot_proof_size_panels(small, medium, large, basic_kinds,
                                "proof_size_vs_fill.pdf"),
        _plot_proof_size_panels(small, medium, large, history_kinds,
                                "proof_size_history_vs_fill.pdf"),
    ]


def _plot_proof_size_panels(
    small: list[LevelStats],
    medium: list[LevelStats],
    large: list[LevelStats],
    kinds: list[tuple[str, str, str]],
    out_filename: str,
) -> Path:
    """Render a single proof-size figure with one panel per kind.

    Used to split the four lookup kinds across two figures (basic vs.
    history) while keeping the plot logic in one place. Label/value
    proofs render in KB; history proofs render in bytes so the
    small-regime empty-history floor stays visible.
    """
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.2 * len(kinds), 3), sharex=True)
    if len(kinds) == 1:
        axes = [axes]
    large_style = REGIME_STYLES["large"]
    KB_KINDS = {"label", "value"}
    for ax, (key, _short, title) in zip(axes, kinds):
        scale = 1.0 / 1024.0 if key in KB_KINDS else 1.0
        unit = "KB" if key in KB_KINDS else "bytes"
        for regime_name, regime_levels in [("small", small), ("medium", medium), ("large", large)]:
            if not regime_levels:
                continue
            style = REGIME_STYLES[regime_name]
            xs = np.array([lvl.fill_percent for lvl in regime_levels])
            p50 = np.array([percentile(lvl.proof_bytes[key], 50) for lvl in regime_levels]) * scale
            p10 = np.array([percentile(lvl.proof_bytes[key], 10) for lvl in regime_levels]) * scale
            p90 = np.array([percentile(lvl.proof_bytes[key], 90) for lvl in regime_levels]) * scale
            ax.fill_between(xs, p10, p90, color=style["color"], alpha=0.18, linewidth=0)
            ax.plot(
                xs,
                p50,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=6,
                label=style["label"],
            )

        ext = _large_proof_extrapolation(large, medium, key)
        if ext is not None:
            xs_ext, ys_ext = ext
            ys_ext = ys_ext * scale
            measured_large = [l for l in large if l.fill_percent <= 10.5]
            if measured_large:
                last = measured_large[-1]
                last_med = float(np.median(last.proof_bytes[key])) * scale
                xs_full = np.concatenate(([last.fill_percent], xs_ext))
                ys_full = np.concatenate(([last_med], ys_ext))
            else:
                xs_full, ys_full = xs_ext, ys_ext
            ax.plot(
                xs_full,
                ys_full,
                color=large_style["color"],
                marker=large_style["marker"],
                markerfacecolor="none",
                linestyle="--",
                linewidth=1.4,
                markersize=6,
                alpha=0.85,
                label="aegon large (extrapolated)",
            )

        ax.set_xlabel("preload fill (% of true capacity)")
        ax.set_ylabel(f"median {title}\nproof size ({unit})")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks([1, 30, 60, 90])

    handles, labels = axes[0].get_legend_handles_labels()
    pairs = list(zip(handles, labels))
    row1 = [(h, l) for h, l in pairs if not l.startswith("large")]
    row2 = [(h, l) for h, l in pairs if l.startswith("large")]
    for ax in fig.axes:
        _label_axis_endpoints(ax)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    if row1:
        fig.legend(
            [h for h, _ in row1], [l for _, l in row1],
            loc="lower center", ncol=len(row1),
            bbox_to_anchor=(0.5, 0.08), frameon=False,
        )
    if row2:
        fig.legend(
            [h for h, _ in row2], [l for _, l in row2],
            loc="lower center", ncol=len(row2),
            bbox_to_anchor=(0.5, 0.0), frameon=False,
        )

    out_path = PLOTS_DIR / out_filename
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_lookup_per_operation(
    small: list[LevelStats],
    medium: list[LevelStats],
    large: list[LevelStats],
    subset: tuple[str, ...],
    agg: str = AGG_MEDIAN,
    akd: dict[str, list[LevelStats]] | None = None,
    irondict: dict[str, IronStats] | None = None,
) -> list[Path]:
    """One figure per lookup kind, with server-time, client-time, and
    proof-size panels side by side (same audit-style layout: all
    metrics for the SAME operation, rather than the same metric for
    different operations).

    Produces one PDF per `LOOKUP_KINDS` entry, suffixed by `subset`
    and `agg` (e.g. `lookup_label_small_medium.pdf` for the median
    view, `lookup_label_small_medium_max.pdf` for the max view).
    """
    paths: list[Path] = []
    for key, _short, title in LOOKUP_KINDS:
        paths.append(_plot_one_lookup_op(small, medium, large, key, title, subset, agg, akd, irondict))
    return paths


def _plot_one_lookup_op(
    small: list[LevelStats],
    medium: list[LevelStats],
    large: list[LevelStats],
    key: str,
    title: str,
    subset: tuple[str, ...],
    agg: str = AGG_MEDIAN,
    akd: dict[str, list[LevelStats]] | None = None,
    irondict: dict[str, IronStats] | None = None,
) -> Path:
    """Render a 3-panel figure (server time | client time | proof size)
    for a single lookup operation. Mirrors the audit_vs_fill layout but
    extended with a third panel for the client-side latency.

    `subset` filters which regimes are drawn as lines. `agg` is
    'median' or 'max'. Large extrapolations are median-anchored so
    they're suppressed under `agg=max`."""
    q_lo, q_mid, q_hi = _percentiles_for(agg)
    reducer = _reducer_for(agg)
    fig, (ax_srv, ax_cli, ax_size) = plt.subplots(1, 3, figsize=(9.6, 3), sharex=True)
    large_style = REGIME_STYLES["large"]
    HISTORY_KINDS = {"label_history", "history"}
    proof_scale = 1.0 if key in HISTORY_KINDS else (1.0 / 1024.0)
    proof_unit = "bytes" if key in HISTORY_KINDS else "KB"

    in_subset = set(subset)
    render_large = "large" in in_subset
    line_regimes: list[tuple[str, list[LevelStats]]] = []
    if "small" in in_subset:
        line_regimes.append(("small", small))
    if "medium" in in_subset:
        line_regimes.append(("medium", medium))
    if render_large:
        line_regimes.append(("large", large))

    extrap_large = render_large
    agg_lbl = _agg_label(agg)

    # ---- Server-time panel ----
    for regime_name, regime_levels in line_regimes:
        if not regime_levels:
            continue
        style = REGIME_STYLES[regime_name]
        xs = np.array([lvl.fill_percent for lvl in regime_levels])
        central = np.array([percentile(lvl.server_ms[key], q_mid) for lvl in regime_levels])
        low = np.array([percentile(lvl.server_ms[key], q_lo) for lvl in regime_levels])
        high = np.array([percentile(lvl.server_ms[key], q_hi) for lvl in regime_levels])
        eps = 1e-4
        central, low, high = np.maximum(central, eps), np.maximum(low, eps), np.maximum(high, eps)
        ax_srv.fill_between(xs, low, high, color=style["color"], alpha=0.18, linewidth=0)
        ax_srv.plot(xs, central, color=style["color"], marker=style["marker"],
                    linewidth=1.8, markersize=6, label=style["label"])
    ext = _large_metric_theoretical_extrapolation(large, lambda lvl, k=key: lvl.server_ms.get(k, []), reducer=reducer) if extrap_large else None
    if ext is not None:
        xs_ext, ys_ext = ext
        measured_large = [l for l in large if l.fill_percent <= 10.5]
        if measured_large:
            last = measured_large[-1]
            last_anchor = float(reducer(last.server_ms[key]))
            xs_full = np.concatenate(([last.fill_percent], xs_ext))
            ys_full = np.concatenate(([last_anchor], ys_ext))
        else:
            xs_full, ys_full = xs_ext, ys_ext
        ax_srv.plot(xs_full, np.maximum(ys_full, 1e-4),
                    color=large_style["color"], marker=large_style["marker"],
                    markerfacecolor="none", linestyle="--", linewidth=1.4,
                    markersize=6, alpha=0.85, label="aegon large (extrapolated)")
    ax_srv.set_xlabel("preload fill (% of true capacity)")
    ax_srv.set_ylabel("server latency (ms)")
    ax_srv.grid(True, which="both", alpha=0.3)
    ax_srv.set_xticks([1, 30, 60, 90])

    # ---- Client-time panel ----
    for regime_name, regime_levels in line_regimes:
        if not regime_levels:
            continue
        style = REGIME_STYLES[regime_name]
        xs = np.array([lvl.fill_percent for lvl in regime_levels])
        central = np.array([percentile(lvl.client_ms[key], q_mid) for lvl in regime_levels])
        low = np.array([percentile(lvl.client_ms[key], q_lo) for lvl in regime_levels])
        high = np.array([percentile(lvl.client_ms[key], q_hi) for lvl in regime_levels])
        eps = 1e-4
        central, low, high = np.maximum(central, eps), np.maximum(low, eps), np.maximum(high, eps)
        ax_cli.fill_between(xs, low, high, color=style["color"], alpha=0.18, linewidth=0)
        ax_cli.plot(xs, central, color=style["color"], marker=style["marker"],
                    linewidth=1.8, markersize=6, label=style["label"])
    ext = _large_metric_theoretical_extrapolation(large, lambda lvl, k=key: lvl.client_ms.get(k, []), reducer=reducer) if extrap_large else None
    if ext is not None:
        xs_ext, ys_ext = ext
        measured_large = [l for l in large if l.fill_percent <= 10.5]
        if measured_large:
            last = measured_large[-1]
            last_anchor = float(reducer(last.client_ms[key]))
            xs_full = np.concatenate(([last.fill_percent], xs_ext))
            ys_full = np.concatenate(([last_anchor], ys_ext))
        else:
            xs_full, ys_full = xs_ext, ys_ext
        ax_cli.plot(xs_full, np.maximum(ys_full, 1e-4),
                    color=large_style["color"], marker=large_style["marker"],
                    markerfacecolor="none", linestyle="--", linewidth=1.4,
                    markersize=6, alpha=0.85, label="aegon large (extrapolated)")
    ax_cli.set_xlabel("preload fill (% of true capacity)")
    ax_cli.set_ylabel("client verify latency (ms)")
    ax_cli.grid(True, which="both", alpha=0.3)
    ax_cli.set_xticks([1, 30, 60, 90])

    # ---- Proof-size panel ----
    for regime_name, regime_levels in line_regimes:
        if not regime_levels:
            continue
        style = REGIME_STYLES[regime_name]
        xs = np.array([lvl.fill_percent for lvl in regime_levels])
        central = np.array([percentile(lvl.proof_bytes[key], q_mid) for lvl in regime_levels]) * proof_scale
        low = np.array([percentile(lvl.proof_bytes[key], q_lo) for lvl in regime_levels]) * proof_scale
        high = np.array([percentile(lvl.proof_bytes[key], q_hi) for lvl in regime_levels]) * proof_scale
        ax_size.fill_between(xs, low, high, color=style["color"], alpha=0.18, linewidth=0)
        ax_size.plot(xs, central, color=style["color"], marker=style["marker"],
                     linewidth=1.8, markersize=6, label=style["label"])
    ext = _large_proof_extrapolation(large, medium, key, reducer=reducer) if extrap_large else None
    if ext is not None:
        xs_ext, ys_ext = ext
        ys_ext = ys_ext * proof_scale
        measured_large = [l for l in large if l.fill_percent <= 10.5]
        if measured_large:
            last = measured_large[-1]
            last_anchor = float(reducer(last.proof_bytes[key])) * proof_scale
            xs_full = np.concatenate(([last.fill_percent], xs_ext))
            ys_full = np.concatenate(([last_anchor], ys_ext))
        else:
            xs_full, ys_full = xs_ext, ys_ext
        ax_size.plot(xs_full, ys_full,
                     color=large_style["color"], marker=large_style["marker"],
                     markerfacecolor="none", linestyle="--", linewidth=1.4,
                     markersize=6, alpha=0.85, label="aegon large (extrapolated)")
    ax_size.set_xlabel("preload fill (% of true capacity)")
    ax_size.set_ylabel(f"proof size ({proof_unit})")
    ax_size.grid(True, which="both", alpha=0.3)
    ax_size.set_xticks([1, 30, 60, 90])

    # AKD-MySQL overlay across all three panels. Matches the aegon
    # regime's color (so the eye groups small-AKD with small-aegon),
    # dashed + hollow marker for "system = AKD". Proof-size axis
    # switches to log because AKD value proofs (~14 KB) dwarf aegon
    # small (~340 B) by ~40×; linear would collapse aegon to zero.
    #
    # AKD only physically measures two operations — `Directory::lookup`
    # (binds label↔value) and `Directory::key_history` (value history)
    # — so in the AKD JSON `label` ≡ `value` and `label_history` ≡
    # `history`. We overlay on every panel anyway (matching the
    # irondict convention) so the reader can compare like-for-like
    # against aegon on whichever operation they care about; the AKD
    # numbers on label vs value panels are identical by construction.
    if akd:
        akd_in_subset = [(n, akd[n]) for n in subset if n in akd and akd[n]]
        for akd_name, akd_levels in akd_in_subset:
            astyle = AKD_STYLES[akd_name]
            # AKD-large lookup samples are extrapolated from small+
            # medium (see akd-mysql/large.json _lookup_note); flag it
            # in the legend so the reader can tell apart from measured.
            akd_label = astyle["label"] + (
                " (extrapolation)" if akd_name == "large" else ""
            )
            lvls = sorted(akd_levels, key=lambda l: l.fill_percent)
            xs = np.array([lvl.fill_percent for lvl in lvls])
            srv = np.array([percentile(lvl.server_ms.get(key, []), q_mid) for lvl in lvls])
            cli = np.array([percentile(lvl.client_ms.get(key, []), q_mid) for lvl in lvls])
            sz = np.array([percentile(lvl.proof_bytes.get(key, []), q_mid) for lvl in lvls]) * proof_scale
            srv = np.maximum(srv, 1e-4)
            cli = np.maximum(cli, 1e-4)
            for ax, ys in ((ax_srv, srv), (ax_cli, cli), (ax_size, sz)):
                ax.plot(
                    xs, ys,
                    color=astyle["color"], marker=astyle["marker"],
                    linestyle=astyle["linestyle"], markerfacecolor="none",
                    linewidth=1.6, markersize=6, label=akd_label,
                )
        if akd_in_subset:
            ax_size.set_yscale("log")

    # irondict overlay. irondict doesn't distinguish label/value/
    # history (one single `lookup_prove` method) — the same measured
    # numbers apply to all four aegon kinds. We overlay on every
    # panel so the reader can compare like-for-like on whichever
    # operation they care about; the value on every panel is the
    # same irondict measurement.
    if irondict:
        srv_xlim = ax_srv.get_xlim()
        cli_xlim = ax_cli.get_xlim()
        size_xlim = ax_size.get_xlim()
        # Plot irondict's single measurement at each of aegon's
        # measured fill x-positions so the marker is visible (and so
        # the reader sees a row of markers rather than a continuous
        # curve — irondict has no fill axis, so the value is the same
        # at every x).
        iron_xs = np.array([1.0, 30.0, 60.0, 90.0])
        any_iron_value = False
        for iron_name in subset:
            iron = irondict.get(iron_name)
            if iron is None:
                continue
            istyle = IRONDICT_STYLES[iron_name]
            if iron.server_lookup_ms is not None:
                ax_srv.plot(
                    iron_xs, np.full_like(iron_xs, iron.server_lookup_ms),
                    color=istyle["color"], linestyle=istyle["linestyle"],
                    marker=istyle["marker"], markerfacecolor="none",
                    linewidth=1.6, markersize=6, label=istyle["label"],
                )
                any_iron_value = True
            if iron.client_lookup_ms is not None:
                ax_cli.plot(
                    iron_xs, np.full_like(iron_xs, iron.client_lookup_ms),
                    color=istyle["color"], linestyle=istyle["linestyle"],
                    marker=istyle["marker"], markerfacecolor="none",
                    linewidth=1.6, markersize=6, label=istyle["label"],
                )
                any_iron_value = True
            if iron.proof_bytes is not None:
                v = iron.proof_bytes * proof_scale
                ax_size.plot(
                    iron_xs, np.full_like(iron_xs, v),
                    color=istyle["color"], linestyle=istyle["linestyle"],
                    marker=istyle["marker"], markerfacecolor="none",
                    linewidth=1.6, markersize=6, label=istyle["label"],
                )
                any_iron_value = True
        ax_srv.set_xlim(srv_xlim)
        ax_cli.set_xlim(cli_xlim)
        ax_size.set_xlim(size_xlim)
        if any_iron_value:
            # irondict's lookup proof at small (~4.8 KB) is ~10× the
            # smallest aegon `value` proof (~340 B), so without log
            # the small aegon curve gets compressed.
            ax_size.set_yscale("log")

    # Two-row shared legend (small/medium on top, large/extrapolated below).
    handles, labels = ax_srv.get_legend_handles_labels()
    pairs = list(zip(handles, labels))
    row1 = [(h, l) for h, l in pairs if not l.startswith("large")]
    row2 = [(h, l) for h, l in pairs if l.startswith("large")]
    for ax in fig.axes:
        _label_axis_endpoints(ax)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    if row1:
        fig.legend([h for h, _ in row1], [l for _, l in row1],
                   loc="lower center", ncol=len(row1),
                   bbox_to_anchor=(0.5, 0.08), frameon=False)
    if row2:
        fig.legend([h for h, _ in row2], [l for _, l in row2],
                   loc="lower center", ncol=len(row2),
                   bbox_to_anchor=(0.5, 0.0), frameon=False)

    out_path = PLOTS_DIR / f"lookup_{key}{_subset_suffix(subset)}{_agg_suffix(agg)}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_latency_knee(
    small: list[LevelStats],
    medium: list[LevelStats],
    large: list[LevelStats],
    subset: tuple[str, ...],
    agg: str = AGG_MEDIAN,
    akd: dict[str, list[LevelStats]] | None = None,
) -> Path:
    """Latency-knee figure — one panel per regime in `subset`.

    Side-by-side panels with linear axes and one shared legend below,
    grouped by fill magnitude. Each panel shows latency vs achieved
    QPS, one curve per fill level. The vertical red dashed line in
    each panel marks the sustained QPS ceiling.

    `agg` selects the latency percentile drawn against QPS: 'median'
    uses `latency_ms_p50`, 'max' uses `latency_ms_p99` (the highest
    precomputed percentile in `concurrency_sweep` samples; raw
    per-request latencies aren't retained, so p99 is the worst the
    data can show).
    """
    lat_key = "latency_ms_p99" if agg == AGG_MAX else "latency_ms_p50"
    all_regimes = {"small": small, "medium": medium, "large": large}
    regimes = [(n, all_regimes[n]) for n in subset if n in all_regimes]
    populated = [(n, [l for l in lvls if l.concurrency_sweep]) for n, lvls in regimes]
    populated = [(n, lvls) for n, lvls in populated if lvls]
    out_path = PLOTS_DIR / f"latency_knee{_subset_suffix(subset)}{_agg_suffix(agg)}.pdf"
    if not populated:
        fig, _ = plt.subplots(figsize=(4, 2.8))
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return out_path

    fig, axes = plt.subplots(1, len(populated), figsize=(3.2 * len(populated), 3), sharey=False)
    if len(populated) == 1:
        axes = [axes]
    # Same softer rainbow palette as `plot_publish_vs_batch`:
    # `Spectral` (red→yellow→green→blue, desaturated).
    cmap = plt.get_cmap("Spectral")

    # Same per-figure rank-based mapping as `plot_publish_vs_batch`:
    # smallest fill → deepest red, largest → deepest blue, evenly
    # spaced colours in between. Guarantees visual separation when
    # the available fills cluster (as in the large plot's 1%–10%
    # measured range) at the cost of "same fill % = same colour"
    # across figures.
    all_fills_global: set[float] = set()
    for _name, lvls in populated:
        for lvl in lvls:
            all_fills_global.add(lvl.fill_percent)
    global_fills_sorted = sorted(all_fills_global)
    n_fills = len(global_fills_sorted)
    if n_fills <= 1:
        fill_to_color = {f: 0.0 for f in global_fills_sorted}
    else:
        fill_to_color = {f: i / (n_fills - 1) for i, f in enumerate(global_fills_sorted)}

    for ax, (regime_name, lvls) in zip(axes, populated):
        for lvl in sorted(lvls, key=lambda l: l.fill_percent):
            samples = sorted(lvl.concurrency_sweep, key=lambda s: s["concurrency"])
            qps = np.array([s["qps"] for s in samples])
            ys = np.array([s[lat_key] for s in samples])
            color = cmap(fill_to_color[lvl.fill_percent])
            ax.plot(
                qps, ys,
                color=color, marker="o", linewidth=1.8,
                markersize=5, label=f"{lvl.fill_percent:g}%",
            )

        # AKD-MySQL overlay for this regime. Same fill→color mapping,
        # dashed line with diamond marker for the "system = AKD"
        # encoding. Single sentinel handle added below the loop.
        if akd and regime_name in akd:
            for lvl in sorted(akd[regime_name], key=lambda l: l.fill_percent):
                if not lvl.concurrency_sweep:
                    continue
                if lvl.fill_percent not in fill_to_color:
                    continue
                samples = sorted(lvl.concurrency_sweep, key=lambda s: s["concurrency"])
                qps = np.array([s["qps"] for s in samples])
                ys = np.array([s[lat_key] for s in samples])
                color = cmap(fill_to_color[lvl.fill_percent])
                ax.plot(
                    qps, ys,
                    color=color, marker="D", markerfacecolor="none",
                    linestyle="-", linewidth=1.4, markersize=5,
                )

        ax.set_xlabel("achieved QPS")
        ax.grid(True, which="both", alpha=0.3)

    ylabel_metric = "p99 latency" if agg == AGG_MAX else "median latency"
    for ax in axes:
        ax.set_ylabel(f"{ylabel_metric} (ms)")

    # Shared legend with the same 3-row grouping as publish (top:
    # 10/30/60/90, middle: extrapolated, bottom: 1–9). Knee has no
    # extrapolated entries, so the middle row is dropped and the
    # remaining rows get compacted.
    seen: dict[str, object] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    if akd and any(akd.get(n) for n in subset):
        from matplotlib.lines import Line2D
        seen["AKD-MySQL"] = Line2D(
            [0], [0], color="#555555", linestyle="-", marker="D",
            markerfacecolor="none", markersize=6, linewidth=1.6,
        )

    def _label_sort_key(lbl: str) -> tuple[float, int]:
        try:
            num = float(lbl.split("%")[0])
        except ValueError:
            num = float("inf")
        return (num, 0)
    ordered = sorted(seen.items(), key=lambda kv: _label_sort_key(kv[0]))
    if ordered:
        # Two rows: measured fill percentages (ascending) on top, then
        # extrapolated + system sentinels below.
        def _row_index(lbl: str) -> int:
            if "extrap" in lbl or lbl.startswith("AKD"):
                return 1
            try:
                float(lbl.split("%")[0])
            except ValueError:
                return -1
            return 0
        rows: list[list[tuple[str, object]]] = [[], []]
        for label, handle in ordered:
            idx = _row_index(label)
            if idx >= 0:
                rows[idx].append((label, handle))
        non_empty = [row for row in rows if row]
        n_rows = len(non_empty)
        rect_map = {1: 0.10, 2: 0.18, 3: 0.26}
        fig.tight_layout(rect=(0, rect_map.get(n_rows, 0.10), 1, 1))
        y_per_row = 0.08
        y_start = 0.02 + y_per_row * (n_rows - 1)
        for i, row in enumerate(non_empty):
            row_labels, row_handles = zip(*row)
            fig.legend(
                row_handles, row_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, y_start - i * y_per_row),
                ncol=len(row_labels), frameon=False, fontsize=9,
            )
    else:
        fig.tight_layout()

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_migration_curves(
    small_runs: list[MigrationRun],
    medium_runs: list[MigrationRun],
    large_runs: list[MigrationRun],
    subset: tuple[str, ...],
) -> Path:
    """Throughput vs publish chunk-size K. The migration bench
    produces one summary point per K (a full climb to target fill
    timed end-to-end), so the right view is a 1-D throughput
    sweep, not a time-vs-fill scatter. One line per regime, log-x
    on K, throughput on y. The peak per regime is annotated and
    the `{regime}_best_k.txt` value gets a vertical guide so the
    reader can see at a glance that the cluster bench picks the
    same K the sweep found.

    Why this beats the old time-vs-fill rendering: each K in the
    sweep is a single timed milestone, so the old plot rendered
    one dot per K. With dots-only the throughput optimum was
    invisible. A line connecting the dots in K-space puts the
    peak right in the middle of the figure.
    """
    all_runs = {"small": small_runs, "medium": medium_runs, "large": large_runs}
    selected: list[tuple[str, list[MigrationRun]]] = []
    for name in subset:
        runs = all_runs.get(name, [])
        if runs:
            selected.append((name, runs))

    out_path = PLOTS_DIR / f"migration_curves{_subset_suffix(subset)}.pdf"
    if not selected:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.text(
            0.5, 0.5,
            "No migration data found.\n"
            "Run `scripts/migration-bench.sh` first.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#555555",
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    for regime_name, runs in selected:
        # Sort by K so the line connects in K order, not insertion
        # order from the file system.
        runs_sorted = sorted(runs, key=lambda r: r.chunk_size)
        xs = np.array([r.chunk_size for r in runs_sorted], dtype=float)
        # Use total throughput (users/sec) as the y-axis — same
        # units as `users_per_sec` the migration JSON reports.
        # Divide by 1000 so the y-axis numbers read at human scale
        # (k users/s).
        ys = np.array(
            [r.total_throughput_users_per_sec / 1000.0 for r in runs_sorted]
        )
        style = REGIME_STYLES.get(regime_name, {"color": "#666666", "marker": "o"})
        color = style["color"]
        marker = style["marker"]
        label = style.get("label", regime_name)
        ax.plot(
            xs, ys,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=7,
            label=label,
        )
        # `{regime}_best_k.txt` is what the cluster bench reads to
        # pick PUBLISH_WARMUP_BATCH_SIZE. Drop a dashed guide at
        # that K so a reader can confirm the chosen K matches the
        # sweep's peak (or see by how much it deviates).
        best_k_file = load_best_k(regime_name)
        if best_k_file is not None:
            ax.axvline(
                best_k_file, color=color, linestyle="--",
                linewidth=1.0, alpha=0.6,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("publish chunk size K (log scale)")
    ax.set_ylabel("end-to-end throughput (×10³ users / sec)")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    ax.set_axisbelow(True)
    # Format x ticks as raw K values (the default 2^N labels are
    # less readable than e.g. "65,536" when comparing to batch
    # sizes named in the rest of the pipeline).
    all_ks = sorted({r.chunk_size for _, runs in selected for r in runs})
    if all_ks:
        ax.set_xticks(all_ks)
        ax.set_xticklabels([f"{k:,}" for k in all_ks], rotation=30, ha="right")
    ax.legend(frameon=False, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    small = load_small()
    medium = load_medium()
    large = load_large()
    if not small and not medium and not large:
        raise SystemExit(f"no lookup JSONs found under {LOOKUP_DIR}")

    # Migration runs live in a separate JSON tree
    # (`bench-results/migration/{regime}_K*.json`) populated by
    # `scripts/migration-bench.sh`. Loaded once, fed only to the
    # migration-curves plot — every other plot consumes `LevelStats`
    # from the lookup JSONs as before.
    small_runs = load_migration_runs("small")
    medium_runs = load_migration_runs("medium")
    large_runs = load_migration_runs("large")

    # Facebook AKD (MySQL backend) overlay. Same JSON shape as aegon's
    # lookup JSONs, so it threads through every plot helper via the
    # `akd=` kwarg. Only the small+medium subset gets the overlay —
    # we haven't benched AKD at large yet.
    akd_overlay: dict[str, list[LevelStats]] = {}
    akd_small = load_akd_small()
    akd_medium = load_akd_medium()
    akd_large = load_akd_large()
    if akd_small:
        akd_overlay["small"] = akd_small
    if akd_medium:
        akd_overlay["medium"] = akd_medium
    if akd_large:
        akd_overlay["large"] = akd_large

    # irondict overlay. Same kwarg shape as AKD; threads through
    # plot_audit_vs_fill / plot_publish_vs_batch / lookup_per_operation
    # via `irondict=`. Per-regime — small.json, medium.json, large.json
    # are written by bench-results/irondict/parse_divan.py.
    irondict_overlay: dict[str, IronStats] = {}
    iron_small = load_irondict_small()
    iron_medium = load_irondict_medium()
    iron_large = load_irondict_large()
    if iron_small is not None:
        irondict_overlay["small"] = iron_small
    if iron_medium is not None:
        irondict_overlay["medium"] = iron_medium
    if iron_large is not None:
        irondict_overlay["large"] = iron_large

    # Every per-data-point figure is rendered TWICE — once per
    # subset (SMALL+MEDIUM vs LARGE) to keep the qualitatively-
    # different large regime from compressing small/medium's y-axis.
    # The central line is the median; the band is min..max.
    #   <name>_small_medium.pdf       <- small+medium
    #   <name>_large.pdf              <- large
    written: list[Path] = []
    for subset in (SMALL_MEDIUM, LARGE_ONLY):
        # AKD and irondict are filtered to just the regimes in the
        # current subset, so the small_medium figure doesn't try to
        # render "AKD large" and vice-versa.
        akd_for_subset = {n: akd_overlay[n] for n in subset if n in akd_overlay} or None
        iron_for_subset = {n: irondict_overlay[n] for n in subset if n in irondict_overlay} or None
        written.extend(plot_lookup_per_operation(small, medium, large, subset, AGG_MEDIAN, akd=akd_for_subset, irondict=iron_for_subset))
        written.append(plot_publish_vs_batch(small, medium, large, subset, AGG_MEDIAN, akd=akd_for_subset, irondict=iron_for_subset))
        written.append(plot_audit_vs_fill(small, medium, large, subset, AGG_MEDIAN, akd=akd_for_subset, irondict=iron_for_subset))
        written.append(plot_latency_knee(small, medium, large, subset, AGG_MEDIAN, akd=akd_for_subset))
        written.append(plot_migration_curves(small_runs, medium_runs, large_runs, subset))

    # Merged migration-curves figure: small + medium + large on one
    # plot for a single-glance throughput-vs-K comparison across
    # regimes. The per-subset variants above are kept for paper
    # sections that prefer the regime-separated view.
    written.append(plot_migration_curves(small_runs, medium_runs, large_runs, ALL_REGIMES))

    for p in written:
        size = p.stat().st_size
        rel = p.relative_to(REPO_ROOT)
        print(f"wrote {rel} ({size} bytes)")


if __name__ == "__main__":
    main()
