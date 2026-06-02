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
from matplotlib.ticker import ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
LOOKUP_DIR = REPO_ROOT / "bench-results" / "lookup"
SMALL_LOOKUP_DIR = REPO_ROOT / "bench-results" / "_remote-small" / "lookup"
PLOTS_DIR = REPO_ROOT / "bench-results" / "plots"

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
    "small": {"color": "#1f77b4", "marker": "o", "label": "small (1 shard, log_cap=22)"},
    "medium": {"color": "#d62728", "marker": "s", "label": "medium (2 shards, log_cap=27)"},
    "large": {"color": "#2ca02c", "marker": "^", "label": "large (128 shards, log_cap=27)"},
}


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


def percentile(vs: list[float], q: float) -> float:
    if not vs:
        return float("nan")
    return float(np.percentile(vs, q))


def plot_server_lookup_vs_fill(small: list[LevelStats], medium: list[LevelStats], large: list[LevelStats]) -> Path:
    """One panel per lookup kind; small + medium + large.

    Solid lines are measured (median, 10th–90th percentile band).
    For the large regime we also draw a dashed extrapolation at
    medium's high fills (30/60/90) using medium's growth ratio
    anchored at large's measured 1% — same approach as the proof-size
    plot, since medium and large share identical per-probe structure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    large_style = REGIME_STYLES["large"]
    for ax, (key, _short, title) in zip(axes, LOOKUP_KINDS):
        for regime_name, regime_levels in [("small", small), ("medium", medium), ("large", large)]:
            if not regime_levels:
                continue
            style = REGIME_STYLES[regime_name]
            xs = np.array([lvl.fill_percent for lvl in regime_levels])
            p50 = np.array([percentile(lvl.server_ms[key], 50) for lvl in regime_levels])
            p10 = np.array([percentile(lvl.server_ms[key], 10) for lvl in regime_levels])
            p90 = np.array([percentile(lvl.server_ms[key], 90) for lvl in regime_levels])
            # Floor at a tiny positive value so log scale doesn't choke on
            # the small bench's sub-microsecond history latencies.
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

        ext = _large_metric_extrapolation(
            large, medium, lambda lvl, k=key: lvl.server_ms.get(k, [])
        )
        if ext is not None:
            xs_ext, ys_ext = ext
            measured_large = [l for l in large if l.fill_percent <= 4.0]
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
                label="large (extrap.)",
            )

        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("preload fill (% of true capacity)")
        ax.set_ylabel("server lookup latency (ms, log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks([1, 30, 60, 90])

    # Single legend at the bottom — same labels appear in every panel.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle("Server-side lookup latency vs. preload (median, shaded = 10th–90th percentile; dashed = extrapolation)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out_path = PLOTS_DIR / "server_lookup_vs_fill.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_audit_vs_fill(small: list[LevelStats], medium: list[LevelStats], large: list[LevelStats]) -> Path:
    """Audit figure: 2 side-by-side panels.

    Left:  auditor verify latency vs preload fill (one line per regime).
    Right: audit proof size (bytes) vs preload fill, same regimes.

    Both are expected to be ~flat with fill — the audit proof structure
    is per-epoch and per-shard, independent of dictionary state. What
    changes between regimes is the shard count: large's proof carries
    ~37 KB of cross-shard merkle data vs medium's 640 B and small's
    344 B.
    """
    fig, (ax_time, ax_size) = plt.subplots(1, 2, figsize=(12, 5))

    large_style = REGIME_STYLES["large"]

    # ---- LEFT: verify latency ----
    for regime_name, regime_levels in [("small", small), ("medium", medium), ("large", large)]:
        regime_levels = [lvl for lvl in regime_levels if lvl.audit_ms]
        if not regime_levels:
            continue
        style = REGIME_STYLES[regime_name]
        xs = np.array([lvl.fill_percent for lvl in regime_levels])
        p50 = np.array([percentile(lvl.audit_ms, 50) for lvl in regime_levels])
        p10 = np.array([percentile(lvl.audit_ms, 10) for lvl in regime_levels])
        p90 = np.array([percentile(lvl.audit_ms, 90) for lvl in regime_levels])
        ax_time.fill_between(xs, p10, p90, color=style["color"], alpha=0.18, linewidth=0)
        ax_time.plot(
            xs, p50,
            color=style["color"], marker=style["marker"], linewidth=1.8,
            markersize=6, label=style["label"],
        )

    ext = _large_metric_extrapolation(large, medium, lambda lvl: lvl.audit_ms)
    if ext is not None:
        xs_ext, ys_ext = ext
        measured_large = [l for l in large if l.fill_percent <= 4.0 and l.audit_ms]
        if measured_large:
            last = measured_large[-1]
            last_med = float(np.median(last.audit_ms))
            xs_full = np.concatenate(([last.fill_percent], xs_ext))
            ys_full = np.concatenate(([last_med], ys_ext))
        else:
            xs_full, ys_full = xs_ext, ys_ext
        ax_time.plot(
            xs_full, ys_full,
            color=large_style["color"], marker=large_style["marker"],
            markerfacecolor="none", linestyle="--", linewidth=1.4,
            markersize=6, alpha=0.85, label="large (extrap.)",
        )

    ax_time.set_yscale("log")
    ax_time.set_xlabel("preload fill (% of true capacity)")
    ax_time.set_ylabel("auditor verify latency (ms, log)")
    ax_time.set_title("Verify latency")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.set_xticks([1, 30, 60, 90])
    ax_time.set_ylim(0.1, 50)
    ax_time.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50])
    ax_time.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_time.yaxis.set_minor_formatter(plt.NullFormatter())

    # ---- RIGHT: proof size in KB ----
    for regime_name, regime_levels in [("small", small), ("medium", medium), ("large", large)]:
        regime_levels = [lvl for lvl in regime_levels if lvl.audit_bytes]
        if not regime_levels:
            continue
        style = REGIME_STYLES[regime_name]
        xs = np.array([lvl.fill_percent for lvl in regime_levels])
        # Convert bytes → KB. The proof size is identical per sample at
        # a given regime+fill, so median/p10/p90 collapse to a single
        # value — no need for a shaded band.
        p50 = np.array([percentile(lvl.audit_bytes, 50) for lvl in regime_levels]) / 1024.0
        ax_size.plot(
            xs, p50,
            color=style["color"], marker=style["marker"], linewidth=1.8,
            markersize=6, label=style["label"],
        )

    # Large proof-size extrapolation. The audit proof structure carries
    # one merkle commitment per shard and is fill-independent by
    # construction, so the projected curve is simply the measured
    # large value carried over to 30/60/90% (medium's growth ratio is
    # 1.0 — also flat — so _large_metric_extrapolation produces this
    # naturally and we stay consistent with the other panels).
    ext = _large_metric_extrapolation(large, medium, lambda lvl: lvl.audit_bytes)
    if ext is not None:
        xs_ext, ys_ext = ext
        measured_large = [l for l in large if l.fill_percent <= 4.0 and l.audit_bytes]
        if measured_large:
            last = measured_large[-1]
            last_med = float(np.median(last.audit_bytes))
            xs_full = np.concatenate(([last.fill_percent], xs_ext))
            ys_full = np.concatenate(([last_med], ys_ext))
        else:
            xs_full, ys_full = xs_ext, ys_ext
        ax_size.plot(
            xs_full, ys_full / 1024.0,
            color=large_style["color"], marker=large_style["marker"],
            markerfacecolor="none", linestyle="--", linewidth=1.4,
            markersize=6, alpha=0.85, label="large (extrap.)",
        )

    ax_size.set_yscale("log")
    ax_size.set_xlabel("preload fill (% of true capacity)")
    ax_size.set_ylabel("audit proof size (KB, log)")
    ax_size.set_title("Proof size")
    ax_size.grid(True, which="both", alpha=0.3)
    ax_size.set_xticks([1, 30, 60, 90])
    # Dense ticks covering 0.1 KB .. 100 KB so small (~0.34 KB),
    # medium (~0.63 KB), and large (~37 KB) are all readable.
    ax_size.set_ylim(0.1, 100)
    ax_size.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100])
    ax_size.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_size.yaxis.set_minor_formatter(plt.NullFormatter())

    # Single shared legend under both panels — handles/labels are
    # identical between the two axes (same 3 regimes + same large
    # extrapolation), so we pull from ax_time and place it at figure
    # bottom-center.
    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle("Auditor verify cost vs. preload (median, shaded = 10th–90th percentile; dashed = extrapolation)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out_path = PLOTS_DIR / "audit_vs_fill.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_publish_vs_batch(small: list[LevelStats], medium: list[LevelStats], large: list[LevelStats]) -> Path:
    """Publish median time vs. batch size, faceted by regime, one
    curve per fill level (color = fill %).

    Three panels (small / medium / large) share log-log axes. Inside
    each panel a curve is drawn per fill: median line + 10th-90th percentile band.
    x-axis is batch size as a percentage of the dictionary's true
    capacity (2**true_log_capacity) so the regimes are on the same
    normalized scale and curve separation between fills (if any) is
    immediately visible.
    """
    regimes = [("small", small), ("medium", medium), ("large", large)]
    populated = [(n, lvls) for n, lvls in regimes if lvls]
    if not populated:
        fig, _ = plt.subplots(figsize=(7, 5))
        out_path = PLOTS_DIR / "publish_vs_batch.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return out_path

    # Per-panel y-axis (sharey=False) so the large panel can show its
    # extrapolated values without squishing small/medium.
    fig, axes = plt.subplots(1, len(populated), figsize=(5 * len(populated), 5), sharey=False)
    if len(populated) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")

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

    for ax, (regime_name, regime_levels) in zip(axes, populated):
        true_cap = 1 << regime_levels[0].true_log_capacity
        levels_sorted = sorted(regime_levels, key=lambda l: l.fill_percent)
        measured_fills = [lvl.fill_percent for lvl in levels_sorted if lvl.publish_ms]
        if not measured_fills:
            ax.set_title(f"{regime_name} (no data)")
            continue
        # Add extrapolated fills only for large (open-addressing
        # theory scaling, projected to 30/60/90 %).
        extra_fills = list(EXTRAP_FILLS_PUBLISH) if regime_name == "large" else []
        all_fills_sorted = sorted(set(measured_fills) | set(extra_fills))
        # Sample the colormap in [0.15, 0.85] across the union, so
        # extrapolated high-fill curves get the bright end and stay
        # visually consistent with the measured low-fill curves.
        if len(all_fills_sorted) == 1:
            fill_to_color = {all_fills_sorted[0]: 0.5}
        else:
            positions = np.linspace(0.15, 0.85, len(all_fills_sorted))
            fill_to_color = dict(zip(all_fills_sorted, positions))

        for lvl in [l for l in levels_sorted if l.publish_ms]:
            color = cmap(fill_to_color[lvl.fill_percent])
            batch_sizes = np.array(sorted(lvl.publish_ms))
            xs = 100.0 * batch_sizes / true_cap
            p50 = np.array([percentile(lvl.publish_ms[b], 50) for b in batch_sizes]) / 1000.0
            p10 = np.array([percentile(lvl.publish_ms[b], 10) for b in batch_sizes]) / 1000.0
            p90 = np.array([percentile(lvl.publish_ms[b], 90) for b in batch_sizes]) / 1000.0
            ax.fill_between(xs, p10, p90, color=color, alpha=0.18, linewidth=0)
            ax.plot(
                xs,
                p50,
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=f"{lvl.fill_percent:g}%",
            )

        # Extrapolation overlay for large: empirical log-log power-law
        # fit on the measured 1-4% data, ONE fit per batch size. For
        # each target fill (30/60/90%), compute the projected median
        # latency by fitting log(time) = a + b·log(fill_pct) across
        # large's measured anchor points and plugging in the target
        # fill. Drawn as dashed curves with hollow markers so they
        # don't look like measured data.
        if regime_name == "large" and extra_fills:
            # Use ALL measured large fills as fit anchors (not just 1%).
            # With 4 points (1, 2, 3, 4%) the log-log slope captures the
            # RocksDB compaction transient honestly.
            anchors = sorted(
                [l for l in levels_sorted if l.publish_ms and l.fill_percent > 0.0],
                key=lambda l: l.fill_percent,
            )
            if len(anchors) >= 2:
                # Use union of all anchors' batch sizes, intersect-style:
                # only batches that appear in EVERY anchor are fittable.
                batch_sets = [set(a.publish_ms.keys()) for a in anchors]
                common_batches = sorted(set.intersection(*batch_sets))
                if common_batches:
                    xs = 100.0 * np.array(common_batches) / true_cap
                    for f in extra_fills:
                        ys = []
                        for b in common_batches:
                            anchor_pts = [
                                (a.fill_percent, percentile(a.publish_ms[b], 50))
                                for a in anchors
                            ]
                            fit_ms = _empirical_publish_fit(anchor_pts, f)
                            if fit_ms is None:
                                ys.append(np.nan)
                            else:
                                ys.append(fit_ms / 1000.0)  # ms → s
                        color = cmap(fill_to_color[f])
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
                            label=f"{f:g}% (extrap.)",
                        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("publish batch size (% of true capacity, log)")
        ax.set_title(REGIME_STYLES[regime_name]["label"])
        ax.grid(True, which="both", alpha=0.3)
        if regime_name == "large":
            for marker_batch, marker_label in ((50_000, "50k"), (80_000, "80k")):
                xv = 100.0 * marker_batch / true_cap
                ax.axvline(xv, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
                ax.text(
                    xv,
                    0.02,
                    marker_label,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="gray",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
                )
        ax.legend(title="fill", loc="best", frameon=False, fontsize=9)

    for ax in axes:
        ax.set_ylabel("publish latency (s, log)")
    fig.suptitle("Publish latency vs. batch fraction of capacity (median, shaded = 10th–90th percentile; dashed = extrapolation)")
    fig.tight_layout()

    out_path = PLOTS_DIR / "publish_vs_batch.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_client_lookup_vs_fill(small: list[LevelStats], medium: list[LevelStats], large: list[LevelStats]) -> Path:
    """One panel per lookup kind; small + medium overlaid.

    Each curve is the median client-side verify time; a shaded band shows
    10th-90th percentile. Client time is what the verifier pays per lookup (proof
    deserialization + pairing-based verification).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    large_style = REGIME_STYLES["large"]
    for ax, (key, _short, title) in zip(axes, LOOKUP_KINDS):
        for regime_name, regime_levels in [("small", small), ("medium", medium), ("large", large)]:
            if not regime_levels:
                continue
            style = REGIME_STYLES[regime_name]
            xs = np.array([lvl.fill_percent for lvl in regime_levels])
            p50 = np.array([percentile(lvl.client_ms[key], 50) for lvl in regime_levels])
            p10 = np.array([percentile(lvl.client_ms[key], 10) for lvl in regime_levels])
            p90 = np.array([percentile(lvl.client_ms[key], 90) for lvl in regime_levels])
            # Same eps floor as the server-latency plot: empty-history
            # verifications can be sub-microsecond and break log scale.
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

        ext = _large_metric_extrapolation(
            large, medium, lambda lvl, k=key: lvl.client_ms.get(k, [])
        )
        if ext is not None:
            xs_ext, ys_ext = ext
            measured_large = [l for l in large if l.fill_percent <= 4.0]
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
                label="large (extrap.)",
            )

        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("preload fill (% of true capacity)")
        ax.set_ylabel("client verify latency (ms, log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks([1, 30, 60, 90])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle("Client-side lookup verify latency vs. preload (median, shaded = 10th–90th percentile; dashed = extrapolation)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out_path = PLOTS_DIR / "client_lookup_vs_fill.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _project_via_medium_ratio(
    large: list[LevelStats],
    medium: list[LevelStats],
    getter,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Medium-ratio projection: apply medium's X(fill)/X(1%) to large's 1%.

    Medium and large share kzh_k=9, shard_log_capacity=27, and
    over-prov=4 — same per-probe opening structure and same open-
    addressing load factor (alpha = fill/400). So medium's relative
    growth above 4% is the best single-regime predictor for large at
    the same fills.
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
    m1 = float(np.median(m_anchor_vals))
    l1 = float(np.median(l_anchor_vals))
    if m1 == 0:
        return None
    xs, ys = [], []
    for lvl in medium:
        if lvl.fill_percent <= 4.0:
            continue
        vals = getter(lvl)
        if not vals:
            continue
        mf = float(np.median(vals))
        xs.append(lvl.fill_percent)
        ys.append(l1 * mf / m1)
    if not xs:
        return None
    return np.array(xs), np.array(ys)


def _project_via_large_fit(
    large: list[LevelStats], getter, target_fills: np.ndarray
) -> np.ndarray | None:
    """Log-log fit through large's measured 1-4% data, projected to target fills.

    Uses large's *own* measurements at its actual scale. The 1-4%
    fits a power law `log(y) = a + b * log(fill)`; the projection at
    `target_fills` assumes that local growth rate continues. Returns
    None when fewer than 2 valid anchor points are available or when
    any value is non-positive (log undefined).
    """
    measured = [l for l in large if l.fill_percent <= 4.5]
    valid: list[tuple[float, float]] = []
    for lvl in measured:
        vals = getter(lvl)
        if not vals:
            continue
        med = float(np.median(vals))
        if med <= 0:
            continue
        valid.append((lvl.fill_percent, med))
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
    medium_pred = _project_via_medium_ratio(large, medium, getter)
    if medium_pred is None:
        return None
    xs_target, ys_medium = medium_pred
    ys_large = _project_via_large_fit(large, getter, xs_target)
    if ys_large is None:
        return xs_target, ys_medium
    return xs_target, np.sqrt(ys_medium * ys_large)


def _large_proof_extrapolation(
    large: list[LevelStats], medium: list[LevelStats], key: str
) -> tuple[np.ndarray, np.ndarray] | None:
    return _large_metric_extrapolation(
        large, medium, lambda lvl: lvl.proof_bytes.get(key, [])
    )


def plot_proof_size_vs_fill(small: list[LevelStats], medium: list[LevelStats], large: list[LevelStats]) -> Path:
    """One panel per lookup kind; proof bytes for small + medium + large.

    Solid lines are measured. For the large regime we also draw a
    dashed extrapolation at fills 30/60/90% computed by transferring
    medium's growth ratio (medium has identical per-probe structure;
    only the Merkle-path depth differs, captured in large's 1% anchor).
    Each measured curve is the median; the shaded band is 10th-90th percentile.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    large_style = REGIME_STYLES["large"]
    # Render label / value panels in KB, history panels in bytes
    # (history proofs are already in the 10–20 KB range so KB doesn't
    # change much, but bytes keep the small-regime empty-history floor
    # visible).
    KB_KINDS = {"label", "value"}
    for ax, (key, _short, title) in zip(axes, LOOKUP_KINDS):
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
            measured_large = [l for l in large if l.fill_percent <= 4.0]
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
                label="large (extrap.)",
            )

        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("preload fill (% of true capacity)")
        ax.set_ylabel(f"proof size ({unit}, log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks([1, 30, 60, 90])
        if key in KB_KINDS:
            # KB values land in the 4–9 range, where the default log
            # formatter renders "4 × 10⁰" etc. Plain numbers read better.
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(fmt)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle("Lookup proof size vs. preload (median, shaded = 10th–90th percentile; dashed = extrapolation)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out_path = PLOTS_DIR / "proof_size_vs_fill.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_latency_knee_per_regime(
    small: list[LevelStats], medium: list[LevelStats], large: list[LevelStats]
) -> list[Path]:
    """Latency-knee figure, one PDF per regime.

    Each PDF shows p99 latency vs achieved QPS, with one curve per fill
    level. The "knee" is where latency starts to blow up as offered load
    approaches the system's QPS ceiling — that's the planner-relevant
    operating point.

    Skips a regime entirely when none of its levels carry concurrency-
    sweep data (older bench outputs don't have it).
    """
    out_paths: list[Path] = []
    for regime_name, lvls in [("small", small), ("medium", medium), ("large", large)]:
        with_sweep = [l for l in lvls if l.concurrency_sweep]
        if not with_sweep:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = plt.get_cmap("viridis")
        fills = sorted({l.fill_percent for l in with_sweep})
        positions = (
            [0.5] if len(fills) == 1 else np.linspace(0.15, 0.85, len(fills))
        )
        fill_to_color = dict(zip(fills, positions))

        sweep_kind = ""
        # Two "ceiling" notions worth tracking:
        #   * peak_qps: the highest QPS observed at ANY concurrency.
        #     For small/medium this is usually conc=1 (single-thread
        #     1/latency burst) which isn't sustainable under real load.
        #   * sustained_qps: the QPS at the HIGHEST measured concurrency.
        #     This is what the system actually delivers when stressed
        #     — the "break point" you'd plan capacity around.
        # For small/medium they differ (peak~200 vs sustained~184). For
        # large the climbing phase is visible, peak and sustained both
        # land around the same masking-pool ceiling.
        peak_qps_overall = 0.0
        sustained_qps_overall = 0.0
        for lvl in sorted(with_sweep, key=lambda l: l.fill_percent):
            if sweep_kind == "" and lvl.concurrency_sweep_kind:
                sweep_kind = lvl.concurrency_sweep_kind
            samples = sorted(lvl.concurrency_sweep, key=lambda s: s["concurrency"])
            conc = np.array([s["concurrency"] for s in samples])
            qps = np.array([s["qps"] for s in samples])
            p99 = np.array([s["latency_ms_p99"] for s in samples])
            color = cmap(fill_to_color[lvl.fill_percent])
            ax.plot(
                qps, p99,
                color=color, marker="o", linewidth=1.8,
                markersize=6, label=f"{lvl.fill_percent:g}% fill",
            )
            # Label each point with its concurrency so the (possibly
            # tangled) curve shape is self-explanatory. Small jitter
            # to keep labels off the marker itself.
            for q, l99, c in zip(qps, p99, conc):
                ax.annotate(
                    f"N={c}",
                    xy=(q, l99),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7, color=color, alpha=0.85,
                )
            peak_qps_overall = max(peak_qps_overall, float(qps.max()))
            sustained_qps_overall = max(sustained_qps_overall, float(qps[-1]))

        # Vertical dashed line at the SUSTAINED ceiling — this is the
        # QPS the system actually holds under high concurrency, not the
        # transient single-thread peak.
        if sustained_qps_overall > 0:
            ax.axvline(sustained_qps_overall, color="red", linestyle="--",
                       linewidth=1.2, alpha=0.7, zorder=1)
            _, ymax = ax.get_ylim()
            ax.annotate(
                f"sustained QPS ≈ {sustained_qps_overall:,.0f}",
                xy=(sustained_qps_overall, ymax),
                xytext=(6, -10), textcoords="offset points",
                color="red", fontsize=10, fontweight="bold",
                ha="left", va="top",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("achieved QPS (log)")
        ax.set_ylabel("p99 latency (ms, log)")
        ax.set_title(
            f"Latency knee — {regime_name} regime\n"
            f"(p99 vs. QPS, {sweep_kind or 'lookup'} RPC; one curve per fill level)"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", frameon=False, fontsize=9)
        fig.tight_layout()

        out_path = PLOTS_DIR / f"latency_knee_{regime_name}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    small = load_small()
    medium = load_medium()
    large = load_large()
    if not small and not medium and not large:
        raise SystemExit(f"no lookup JSONs found under {LOOKUP_DIR}")

    written: list[Path] = []
    written.append(plot_server_lookup_vs_fill(small, medium, large))
    written.append(plot_client_lookup_vs_fill(small, medium, large))
    written.append(plot_proof_size_vs_fill(small, medium, large))
    written.append(plot_publish_vs_batch(small, medium, large))
    written.append(plot_audit_vs_fill(small, medium, large))
    written.extend(plot_latency_knee_per_regime(small, medium, large))

    for p in written:
        size = p.stat().st_size
        rel = p.relative_to(REPO_ROOT)
        print(f"wrote {rel} ({size} bytes)")


if __name__ == "__main__":
    main()
