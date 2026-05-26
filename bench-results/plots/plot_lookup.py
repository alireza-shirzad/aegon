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

REPO_ROOT = Path(__file__).resolve().parents[2]
LOOKUP_DIR = REPO_ROOT / "bench-results" / "lookup"
SMALL_LOOKUP_DIR = REPO_ROOT / "bench-results" / "_remote-small" / "lookup"
PLOTS_DIR = REPO_ROOT / "bench-results" / "plots"

# Lookup operations and the JSON field stems for each.
LOOKUP_KINDS = [
    ("label", "label", "label lookup"),
    ("value", "value", "value lookup"),
    ("history", "value-history", "value-history lookup"),
    ("label_history", "label-history", "label-history lookup"),
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


def load_small() -> list[LevelStats]:
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
            out.append(LevelStats(
                fill_percent=float(pct),
                true_log_capacity=tlc,
                server_ms=server_ms,
                client_ms=client_ms,
                proof_bytes=proof_bytes,
                publish_ms=publish_ms,
                audit_ms=audit_ms,
                audit_bytes=audit_bytes,
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
        out.append(LevelStats(
            fill_percent=float(fp),
            true_log_capacity=tlc,
            server_ms=server_ms,
            client_ms=client_ms,
            proof_bytes=proof_bytes,
            publish_ms=publish_ms,
            audit_ms=audit_ms,
            audit_bytes=audit_bytes,
        ))
    out.sort(key=lambda x: x.fill_percent)
    return out


def percentile(vs: list[float], q: float) -> float:
    if not vs:
        return float("nan")
    return float(np.percentile(vs, q))


def plot_server_lookup_vs_fill(small: list[LevelStats], medium: list[LevelStats]) -> Path:
    """One panel per lookup kind; small + medium overlaid.

    Each curve is the median; a shaded band shows p10-p90.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    for ax, (key, _short, title) in zip(axes, LOOKUP_KINDS):
        for regime_name, regime_levels in [("small", small), ("medium", medium)]:
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
    fig.suptitle("Server-side lookup latency vs. preload (p50, shaded = p10–p90)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out_path = PLOTS_DIR / "server_lookup_vs_fill.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_audit_vs_fill(small: list[LevelStats], medium: list[LevelStats]) -> Path:
    """Auditor verify time vs. preload fill, one line per regime.

    Audit checks the epoch-invariance commitment, which is independent
    of namespace fill — so the curves are expected to be flat. Plotted
    on log y for consistency with the lookup-latency figures.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for regime_name, regime_levels in [("small", small), ("medium", medium)]:
        regime_levels = [lvl for lvl in regime_levels if lvl.audit_ms]
        if not regime_levels:
            continue
        style = REGIME_STYLES[regime_name]
        xs = np.array([lvl.fill_percent for lvl in regime_levels])
        p50 = np.array([percentile(lvl.audit_ms, 50) for lvl in regime_levels])
        p10 = np.array([percentile(lvl.audit_ms, 10) for lvl in regime_levels])
        p90 = np.array([percentile(lvl.audit_ms, 90) for lvl in regime_levels])
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

    ax.set_yscale("log")
    ax.set_xlabel("preload fill (% of true capacity)")
    ax.set_ylabel("auditor verify latency (ms, log)")
    ax.set_title("Auditor verify latency vs. preload (p50, shaded = p10–p90)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks([1, 30, 60, 90])
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    out_path = PLOTS_DIR / "audit_vs_fill.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_publish_vs_batch(small: list[LevelStats], medium: list[LevelStats]) -> Path:
    """Publish median time vs. batch size, one curve per regime.

    x-axis is batch size as a percentage of the dictionary's true
    capacity (2**true_log_capacity), so small and medium are on the
    same normalized scale. For each batch size, p10/p50/p90 are
    computed across per-sample measurements pooled from every fill
    level (publish time is largely fill-independent at fixed batch).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for regime_name, regime_levels in [("small", small), ("medium", medium)]:
        if not regime_levels:
            continue
        style = REGIME_STYLES[regime_name]
        true_cap = 1 << regime_levels[0].true_log_capacity
        # batch_size -> pooled samples across all fills
        pooled: dict[int, list[float]] = {}
        for lvl in regime_levels:
            for bs, samples in lvl.publish_ms.items():
                pooled.setdefault(bs, []).extend(samples)
        if not pooled:
            continue
        batch_sizes = np.array(sorted(pooled))
        xs = 100.0 * batch_sizes / true_cap  # percent of true capacity
        p50 = np.array([percentile(pooled[b], 50) for b in batch_sizes])
        p10 = np.array([percentile(pooled[b], 10) for b in batch_sizes])
        p90 = np.array([percentile(pooled[b], 90) for b in batch_sizes])
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

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("publish batch size (% of true capacity, log)")
    ax.set_ylabel("publish latency (ms, log)")
    ax.set_title("Publish latency vs. batch fraction of capacity (p50, shaded = p10–p90 across fills)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out_path = PLOTS_DIR / "publish_vs_batch.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_client_lookup_vs_fill(small: list[LevelStats], medium: list[LevelStats]) -> Path:
    """One panel per lookup kind; small + medium overlaid.

    Each curve is the median client-side verify time; a shaded band shows
    p10-p90. Client time is what the verifier pays per lookup (proof
    deserialization + pairing-based verification).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    for ax, (key, _short, title) in zip(axes, LOOKUP_KINDS):
        for regime_name, regime_levels in [("small", small), ("medium", medium)]:
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
    fig.suptitle("Client-side lookup verify latency vs. preload (p50, shaded = p10–p90)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out_path = PLOTS_DIR / "client_lookup_vs_fill.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_proof_size_vs_fill(small: list[LevelStats], medium: list[LevelStats]) -> Path:
    """One panel per lookup kind; proof bytes for small + medium.

    Each curve is the median; a shaded band shows p10-p90. Log y-axis
    because value-history proofs span ~30 bytes (empty history on small)
    to ~19 KiB (populated history on medium).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    for ax, (key, _short, title) in zip(axes, LOOKUP_KINDS):
        for regime_name, regime_levels in [("small", small), ("medium", medium)]:
            if not regime_levels:
                continue
            style = REGIME_STYLES[regime_name]
            xs = np.array([lvl.fill_percent for lvl in regime_levels])
            p50 = np.array([percentile(lvl.proof_bytes[key], 50) for lvl in regime_levels])
            p10 = np.array([percentile(lvl.proof_bytes[key], 10) for lvl in regime_levels])
            p90 = np.array([percentile(lvl.proof_bytes[key], 90) for lvl in regime_levels])
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
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("preload fill (% of true capacity)")
        ax.set_ylabel("proof size (bytes, log)")
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
    fig.suptitle("Lookup proof size vs. preload (p50, shaded = p10–p90)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out_path = PLOTS_DIR / "proof_size_vs_fill.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    small = load_small()
    medium = load_medium()
    if not small and not medium:
        raise SystemExit(f"no lookup JSONs found under {LOOKUP_DIR}")

    written: list[Path] = []
    written.append(plot_server_lookup_vs_fill(small, medium))
    written.append(plot_client_lookup_vs_fill(small, medium))
    written.append(plot_proof_size_vs_fill(small, medium))
    written.append(plot_publish_vs_batch(small, medium))
    written.append(plot_audit_vs_fill(small, medium))

    for p in written:
        size = p.stat().st_size
        rel = p.relative_to(REPO_ROOT)
        print(f"wrote {rel} ({size} bytes)")


if __name__ == "__main__":
    main()
