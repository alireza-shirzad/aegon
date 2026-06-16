#!/usr/bin/env python3
"""plot_setup.py — render setup-bench results into two publication-ready
PDFs (setup time, key sizes) under bench-results/plots/.

Reads (per regime, first match wins):
    bench-results/setup/<regime>.json          — aegon_setup_bench schema:
        label, shard_log_capacity, kzh_k, setup_seed,
        gen_duration_secs, trim_duration_secs,
        compute_secs, communication_secs,
        universal_params_bytes, prover_param_bytes, verifier_param_bytes,
        h_t_entries
    bench-results/setup/<regime>_cluster.json  — bench-cluster.sh schema:
        regime, n_shards, shard_log_capacity, kzh_k, log_n_shards,
        setup_seed, gen_seconds, gen_seconds_max/min/median/mean,
        gen_seconds_per_shard, broadcast_seconds, srs_bytes
        Cluster mode skips the trim step (each shard uses the full SRS
        as its prover param) so verifier_param_bytes is absent — those
        regimes appear in the setup-time chart but not the key-sizes
        chart.

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


def _load_one(regime: str) -> dict | None:
    """Return a normalized record for `regime`, preferring the
    aegon_setup_bench schema and falling back to the cluster schema
    (mapped to the same field names so callers stay schema-agnostic).
    Returns None when neither file exists."""
    legacy = SETUP_DIR / f"{regime}.json"
    if legacy.exists():
        with legacy.open() as f:
            return json.load(f)
    cluster = SETUP_DIR / f"{regime}_cluster.json"
    if cluster.exists():
        with cluster.open() as f:
            raw = json.load(f)
        # Cluster mode: each shard runs SRS gen sequentially on its
        # own VM, in parallel with the others (no inter-shard comms
        # during gen). Every per-shard time is an independent sample
        # of "sequential SRS gen at this shard config", so we treat
        # `gen_seconds_per_shard` as the sample distribution and use
        # the median as the central tendency, min/max as the error
        # bar. With 1 shard the distribution degenerates to a single
        # point (no error bar); with N shards we have N samples for
        # free.
        per_shard = raw.get("gen_seconds_per_shard")
        if isinstance(per_shard, list) and per_shard:
            samples = [float(x) for x in per_shard]
        else:
            samples = [float(raw.get("gen_seconds_max", raw.get("gen_seconds", 0.0)))]
        srs_bytes = int(raw["srs_bytes"])
        return {
            "label": regime,
            "shard_log_capacity": int(raw["shard_log_capacity"]),
            "kzh_k": int(raw["kzh_k"]),
            "gen_duration_secs": float(np.median(samples)),
            "gen_seconds_samples": samples,
            "universal_params_bytes": srs_bytes,
            # Cluster mode skips trim — each shard uses the full SRS
            # as its prover param. Record that so key_sizes can plot
            # pk and flag the absent vk via _has_vk below.
            "prover_param_bytes": srs_bytes,
            "_source": "cluster",
        }
    return None


def load_records() -> list[dict]:
    records = []
    for name in REGIME_ORDER:
        rec = _load_one(name)
        if rec is None:
            print(f"[warn] missing setup/{name}.json or setup/{name}_cluster.json; skipping")
            continue
        records.append(rec)
    if not records:
        raise SystemExit("no setup JSONs found under " + str(SETUP_DIR))
    return records


def _has_vk(rec: dict) -> bool:
    """Cluster-mode records don't have a trimmed verifier_param_bytes;
    the bench skips trim because each shard uses the full SRS as its
    prover param. Filter such records out of the key-sizes chart."""
    return "verifier_param_bytes" in rec


def plot_setup_time(records: list[dict]) -> Path:
    """Bar chart of total setup time per regime. Log y-axis because the
    range spans ~5 s (small) to ~several minutes (large-per-shard).
    Cluster regimes carry one sample per shard (each shard runs SRS
    gen independently on its VM); we draw min–max error bars and
    annotate the sample count. Legacy in-process records (one sample)
    appear without an error bar."""
    labels = [REGIME_LABEL[r["label"]] for r in records]
    times = [r["gen_duration_secs"] for r in records]
    colors = [REGIME_COLOR[r["label"]] for r in records]
    # Build asymmetric (lower, upper) deltas-from-bar-height per
    # regime; falls back to zero when only one sample is present so
    # matplotlib draws nothing for those bars.
    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    sample_counts: list[int] = []
    for r in records:
        samples = r.get("gen_seconds_samples", [r["gen_duration_secs"]])
        sample_counts.append(len(samples))
        med = r["gen_duration_secs"]
        yerr_lo.append(max(0.0, med - min(samples)))
        yerr_hi.append(max(0.0, max(samples) - med))

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(records))
    bars = ax.bar(
        x,
        times,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        yerr=[yerr_lo, yerr_hi],
        capsize=4,
        error_kw={"ecolor": "#222222", "elinewidth": 0.8},
    )
    ax.set_yscale("log")
    ax.set_ylabel("Setup time (seconds, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("KZH-k SRS generation time (median; whiskers = min..max across shards)")

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
    regime. Log y because pk is GB-scale while vk is ~1 MB. Cluster-
    mode regimes (no trim, no vk) are filtered out — they'd otherwise
    show up as an empty vk bar."""
    records = [r for r in records if _has_vk(r)]
    if not records:
        raise SystemExit("no setup records have verifier_param_bytes; "
                         "did all your regimes come from cluster setup?")
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
