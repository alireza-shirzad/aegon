#!/usr/bin/env python3
"""plot_publish.py — render publish-bench results into a publish-time
vs batch-size PDF, one line per fill_percent (preload level).

Reads:
    bench-results/publish/<regime>.json

The JSON schema (emitted by aegon_publish_bench) is:
    {
      "params": {...},
      "stages": [
        {
          "fill_percent": 0|30|60|90,
          "prefilled_count": <u64>,
          "batches": {
            "2":    {"samples_ms": [...], "fastest_ms": ..., "slowest_ms": ...,
                     "median_ms": ..., "mean_ms": ..., "*_commitment_bytes": ...},
            "4":    {...}, ...
          }
        }, ...
      ]
    }

Writes:
    bench-results/plots/publish_time_<regime>.pdf

Run:
    python3 bench-results/plots/plot_publish.py --regime small
    python3 bench-results/plots/plot_publish.py --regime medium
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_DIR = REPO_ROOT / "bench-results" / "publish"
OUT_DIR = REPO_ROOT / "bench-results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fill levels in canonical order. Each gets a distinct colour from the
# colour-blind-friendly Paul-Tol "bright" palette, matching the
# regime-colour convention used by plot_setup.py.
FILL_ORDER = [0, 30, 60, 90]
FILL_COLOR = {
    0:  "#4477AA",  # blue   — empty dict
    30: "#66CCEE",  # cyan   — light fill
    60: "#EE6677",  # red    — medium fill
    90: "#AA3377",  # purple — near-full
}
FILL_MARKER = {0: "o", 30: "s", 60: "^", 90: "D"}

REGIME_TITLE = {
    "small":  "Small regime: dictionary $2^{20}$, shard capacity $2^{22}$",
    "medium": "Medium regime: dictionary $2^{26}$, shard capacity $2^{28}$",
    "large":  "Large regime: dictionary $2^{32}$, 128 shards × $2^{27}$",
}


def load_publish(regime: str) -> dict:
    path = PUBLISH_DIR / f"{regime}.json"
    if not path.exists():
        raise SystemExit(f"no publish JSON at {path}")
    with path.open() as f:
        return json.load(f)


def plot_publish_time(regime: str, data: dict) -> Path:
    """Publish-time vs batch-size, one line per fill_percent. Median
    central line, fastest/slowest as a translucent band."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Stages may arrive in any order in the JSON; sort + index by
    # fill_percent so the colour mapping is stable.
    stages = {int(s["fill_percent"]): s for s in data["stages"]}

    plotted_any = False
    for pct in FILL_ORDER:
        if pct not in stages:
            print(f"[warn] {regime}: fill_percent={pct}% absent — skipping")
            continue
        stage = stages[pct]
        batches = stage["batches"]
        # Batch-size keys are strings in JSON; convert + sort.
        bs = sorted(int(k) for k in batches.keys())
        median = np.array([batches[str(b)]["median_ms"] for b in bs])
        fastest = np.array([batches[str(b)]["fastest_ms"] for b in bs])
        slowest = np.array([batches[str(b)]["slowest_ms"] for b in bs])
        prefilled = stage["prefilled_count"]

        color = FILL_COLOR[pct]
        marker = FILL_MARKER[pct]
        # fill_between for the fastest-slowest band; line for median.
        ax.fill_between(bs, fastest, slowest, color=color, alpha=0.18, linewidth=0)
        ax.plot(bs, median,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=1.6,
                label=f"fill {pct}%  ({prefilled:,} entries)")
        plotted_any = True

    if not plotted_any:
        raise SystemExit(f"no stages plotted for {regime}")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size (publish updates per epoch, log$_2$ scale)")
    ax.set_ylabel("Publish wall-clock time (ms)")
    ax.set_xticks([int(k) for k in sorted({int(b) for s in stages.values() for b in s["batches"].keys()})])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(REGIME_TITLE.get(regime, f"Regime: {regime}"))
    ax.legend(frameon=False, loc="best", fontsize=9, title="Preload level")

    # Footnote: shaded band = min/max across samples_per_batch.
    n_samples = data["params"].get("samples_per_batch", "?")
    fig.text(0.02, 0.02,
             f"Median line; shaded band = fastest/slowest across "
             f"{n_samples} samples per batch size.",
             fontsize=7, style="italic", color="#555555")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = OUT_DIR / f"publish_time_{regime}.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regime", default="small",
                   choices=("small", "medium", "large"),
                   help="which publish JSON to plot (default: small)")
    args = p.parse_args()
    data = load_publish(args.regime)
    out = plot_publish_time(args.regime, data)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
