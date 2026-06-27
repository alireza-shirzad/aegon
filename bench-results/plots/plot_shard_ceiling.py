"""Minimum-shard plot: how many shards do you need to avoid OOM at scale X?

The model is grounded in v16 medium N=8 watchdog data: shard RSS grew at
~1.95 GB per million entries during write-peak phases (the worst-case
moment is during preload before RocksDB compaction settles).  With a
~4 GB OS/coord buffer headroom, this gives a per-shard entry budget of
roughly (available_ram - 4 GB) / 1.95 GB ≈ ~30M entries on n2-standard-16
(64 GB) and ~62M on n2-highmem-16 (128 GB).

Plots minimum N_SHARDS (rounded up to next power of 2 — the bench
requires it) as a function of dictionary capacity, separated by fill
target.  Three measured-regime markers (small/medium/large from v16)
are overlaid for sanity.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "bench-results" / "plots"

# Empirical memory model from v16 medium watchdog (write-peak phase).
GB_PER_M_ENTRIES = 1.95

# Per-shard usable RAM = total RAM - OS/buffer headroom.
OS_HEADROOM_GB = 4.0

MACHINE_LABEL = "n2-standard-16 (64 GB)"
MACHINE_RAM_GB = 64.0

# Sizing assumption: dictionary is fully filled (worst case).
FILL_PCT = 100

# Measured-regime markers: (name, dict_cap, n_shards_actual)
MEASURED_REGIMES = [
    ("small",  1 << 20, 1),
    ("medium", 1 << 26, 8),
    ("large",  1 << 32, 128),
]


def max_entries_per_shard(machine_ram_gb: float) -> float:
    """Conservative ceiling on per-shard entries before OOM risk."""
    return (machine_ram_gb - OS_HEADROOM_GB) / GB_PER_M_ENTRIES * 1e6


def min_shards(cap: float, fill_pct: float, machine_ram_gb: float) -> int:
    """Minimum N_SHARDS (power of 2) so per-shard entries at the target fill
    stay under the OOM ceiling."""
    entries_total = cap * fill_pct / 100.0
    per_shard_ceiling = max_entries_per_shard(machine_ram_gb)
    raw = max(1.0, entries_total / per_shard_ceiling)
    return 1 << max(0, math.ceil(math.log2(raw)))


def plot_shard_ceiling(out_path: Path) -> Path:
    log_caps = np.arange(20, 41)
    capacities = 2.0 ** log_caps

    fig, ax = plt.subplots(figsize=(9, 5.5))

    shards = [min_shards(c, FILL_PCT, MACHINE_RAM_GB) for c in capacities]
    ax.plot(
        capacities, shards,
        color="#1f77b4",
        linewidth=2.0,
        label=f"approximate cluster size @ fill={FILL_PCT}%, {MACHINE_LABEL}",
    )

    # Overlay measured regimes (deployed N, not predicted)
    for name, cap, n_actual in MEASURED_REGIMES:
        ax.scatter([cap], [n_actual], s=80, color="black", zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(
            f"{name}\nN={n_actual} deployed",
            xy=(cap, n_actual),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("dictionary capacity (entries, log₂)")
    ax.set_ylabel("approximate cluster size (N_SHARDS, log₂)")
    ax.set_title(
        f"Approximate cluster size at fill={FILL_PCT}% on {MACHINE_LABEL}\n"
        f"(model: ~{GB_PER_M_ENTRIES:.2f} GB peak RSS per 1M per-shard entries, "
        f"derived from v16 medium watchdog data)"
    )

    # x-axis ticks at every 4th power of 2
    tick_caps = [2 ** k for k in range(20, 41, 4)]
    ax.set_xticks(tick_caps)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: _human_count(v)))

    # y-axis ticks at every power of 2 up to 16384
    yticks = [2 ** k for k in range(0, 15)]
    ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _human_count(n: float) -> str:
    if n >= 1e12:
        return f"{n/1e12:.0f}T"
    if n >= 1e9:
        return f"{n/1e9:.0f}B"
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return f"{n:.0f}"


def main():
    out = PLOTS_DIR / "shard_ceiling.pdf"
    plot_shard_ceiling(out)
    print(f"wrote {out.relative_to(REPO_ROOT)}")

    # Print the table for reference
    print()
    print(f"  per-shard ceiling: {MACHINE_LABEL} = {max_entries_per_shard(MACHINE_RAM_GB)/1e6:.1f}M entries")
    print()
    print(f"{'capacity':>10} {'approx cluster size @ fill=100%':>32}")
    for log_cap in [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]:
        cap = 2 ** log_cap
        n = min_shards(cap, FILL_PCT, MACHINE_RAM_GB)
        print(f"{_human_count(cap):>10} {n:>26}")


if __name__ == "__main__":
    main()
