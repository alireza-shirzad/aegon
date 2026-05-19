#!/usr/bin/env python3
"""plot_setup.py — render setup-bench results into two publication-ready
PDFs (setup time, key sizes) under bench-results/plots/.

Reads:
    bench-results/setup/{small,medium,large-per-shard}.json

Each JSON has the schema emitted by aegon_setup_bench:
    label, shard_log_capacity, kzh_k, setup_seed,
    gen_duration_secs, trim_duration_secs,
    compute_secs, communication_secs,
    universal_params_bytes, prover_param_bytes, verifier_param_bytes,
    h_t_entries

Writes:
    bench-results/plots/setup_time.pdf
    bench-results/plots/setup_key_sizes.pdf

Run:
    python3 bench-results/plots/plot_setup.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # PDF backend, no DISPLAY needed.
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_DIR = REPO_ROOT / "bench-results" / "setup"
OUT_DIR = REPO_ROOT / "bench-results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical regime order + presentation labels. The large-regime
# `setup-bench` runs a *per-shard* sequential proxy (single-machine
# sequential gen at log_cap=27), since the distributed setup time is
# measured separately on the cluster. We label it as such.
REGIME_ORDER = ["small", "medium", "large-per-shard"]
REGIME_LABEL = {
    "small":           "Small\n(N = $2^{22}$)",
    "medium":          "Medium\n(N = $2^{28}$)",
    "large-per-shard": "Large\n(per shard, seq.)\nN = $2^{27}$",
}
# Colour-blind friendly palette. Same triple used by every regime-
# stratified plot — keeps the paper's figures internally consistent.
REGIME_COLOR = {
    "small":           "#4477AA",
    "medium":          "#EE6677",
    "large-per-shard": "#228833",
}


def load_records() -> list[dict]:
    records = []
    for name in REGIME_ORDER:
        path = SETUP_DIR / f"{name}.json"
        if not path.exists():
            print(f"[warn] missing {path}; skipping")
            continue
        with path.open() as f:
            records.append(json.load(f))
    if not records:
        raise SystemExit("no setup JSONs found under " + str(SETUP_DIR))
    return records


def plot_setup_time(records: list[dict]) -> Path:
    """Bar chart of total setup time per regime. Log y-axis because the
    range spans ~5 s (small) to ~several minutes (large-per-shard)."""
    labels = [REGIME_LABEL[r["label"]] for r in records]
    times = [r["gen_duration_secs"] for r in records]
    colors = [REGIME_COLOR[r["label"]] for r in records]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(records))
    bars = ax.bar(x, times, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Setup time (seconds, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("KZH-k SRS generation time")

    for bar, t in zip(bars, times):
        h = bar.get_height()
        # Format: seconds for <60, m:ss for 60..3600, else h:mm.
        if t < 60:
            txt = f"{t:.1f} s"
        elif t < 3600:
            mins = t / 60.0
            txt = f"{mins:.1f} min"
        else:
            hrs = t / 3600.0
            txt = f"{hrs:.2f} h"
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.10, txt,
                ha="center", va="bottom", fontsize=9)

    # Footnote: large-regime number is sequential proxy, not the cluster
    # measurement, so readers don't compare apples to oranges.
    fig.text(0.02, 0.02,
             "Note: the large bar is a single-machine sequential proxy. "
             "The actual distributed setup time on the 128-shard cluster "
             "is reported separately.",
             fontsize=7, style="italic", color="#555555")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = OUT_DIR / "setup_time.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def fmt_bytes(b: int) -> str:
    """Human-readable byte size for bar annotations."""
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(b)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    if v >= 100:
        return f"{v:.0f} {units[i]}"
    if v >= 10:
        return f"{v:.1f} {units[i]}"
    return f"{v:.2f} {units[i]}"


def plot_key_sizes(records: list[dict]) -> Path:
    """Grouped bar chart of prover-key and verifier-key sizes per
    regime. Log y because pk is GB-scale while vk is ~1 MB."""
    labels = [REGIME_LABEL[r["label"]] for r in records]
    pk = np.array([r["prover_param_bytes"] for r in records], dtype=float)
    vk = np.array([r["verifier_param_bytes"] for r in records], dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.arange(len(records))
    w = 0.36
    bars_pk = ax.bar(x - w / 2, pk, w, label="Prover key (pk)",
                     color="#4477AA", edgecolor="black", linewidth=0.5)
    bars_vk = ax.bar(x + w / 2, vk, w, label="Verifier key (vk)",
                     color="#CCBB44", edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Key size (bytes, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("KZH-k key sizes")
    ax.legend(frameon=False, loc="upper left")

    for bars, vals in [(bars_pk, pk), (bars_vk, vk)]:
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.15,
                    fmt_bytes(int(v)),
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "setup_key_sizes.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    records = load_records()
    print(f"loaded {len(records)} setup records: "
          + ", ".join(r["label"] for r in records))
    p1 = plot_setup_time(records)
    print(f"wrote {p1.relative_to(REPO_ROOT)}")
    p2 = plot_key_sizes(records)
    print(f"wrote {p2.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
