#!/usr/bin/env python3
"""Plot the per-hour deployment cost of an Aegon cluster vs. dictionary
capacity.

Pure model — no bench data needed. The plot draws a stacked breakdown
(shards, coord, masking server, coord disk, shard disks, NAT) over a
continuous range of dictionary capacities, and marks the three actually-
measured regimes (small / medium / large) as scatter points.

All assumptions live in the constants block. Override at the CLI:

    python3 plot_cost.py --shard-machine n2-highmem-16 --max-fill-percent 30
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "bench-results" / "plots"

# ---------------------------------------------------------------------------
# Pricing constants — GCP us-central1, on-demand list prices (USD).
# Sourced from cloud.google.com/compute/all-pricing 2026-Q2. Update here.
# ---------------------------------------------------------------------------

MACHINE_USD_PER_HOUR: dict[str, float] = {
    "n2-standard-4":  0.1942,
    "n2-standard-16": 0.7766,
    "n2-standard-32": 1.5533,
    "n2-highmem-16":  1.0468,
    "n2-highmem-32":  2.0937,
}

# pd-* prices are $/GB/month → divide by (30*24) for $/GB/hour.
PD_USD_PER_GB_MONTH: dict[str, float] = {
    "pd-ssd":      0.170,
    "pd-balanced": 0.100,
    "pd-standard": 0.040,
}

def disk_usd_per_gb_hour(kind: str) -> float:
    return PD_USD_PER_GB_MONTH[kind] / (30.0 * 24.0)

# Cloud NAT gateway fixed hourly charge (egress data is intra-VPC and
# free between cluster nodes). One gateway per region.
NAT_USD_PER_HOUR = 0.044

# Empirical on-disk footprint per dictionary entry, including the
# RocksDB LSM overhead (compression on, ~10× write amplification at
# steady state). Derived from the run-#6 measurement: 4 TB used at
# 172M entries → ~24 KB/entry. Used to size the coord disk vs.
# dictionary capacity at a given max fill level.
DISK_BYTES_PER_ENTRY = 24 * 1024  # 24 KB

# ---------------------------------------------------------------------------
# Cluster-config knobs (matches scripts/bench-cluster.sh defaults).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterConfig:
    shard_machine: str = "n2-standard-16"
    coord_machine: str = "n2-standard-16"
    masking_machine: str = "n2-standard-16"
    shard_boot_disk_gb: int = 30
    shard_boot_disk_kind: str = "pd-balanced"
    coord_disk_kind: str = "pd-ssd"
    # Per-shard polynomial capacity (log2). Default matches medium and
    # large in the bench: shard_log_capacity = 27 → 2^27 slots per
    # shard. Over-provisioning factor 4 (LOG2_OVER_PROVISIONING_FACTOR
    # = 2 in akd/src/aegon/config.rs), so each shard holds 2^25 = ~33M
    # dictionary entries.
    shard_log_capacity: int = 27
    overprov_factor: int = 4
    # Pre-load fill fraction we size the coord disk for. The disk
    # has to fit the deepest fill the bench will reach; sizing for
    # 100% gives a steady-state operational estimate.
    max_fill_percent: float = 100.0

    def entries_per_shard(self) -> int:
        return (1 << self.shard_log_capacity) // self.overprov_factor

    def n_shards_for(self, dict_capacity: float) -> int:
        """Round up to a power of two — the bench requires it."""
        raw = max(1.0, dict_capacity / self.entries_per_shard())
        return 1 << max(0, math.ceil(math.log2(raw)))

    def coord_disk_gb_for(self, dict_capacity: float) -> float:
        bytes_needed = dict_capacity * self.max_fill_percent / 100.0 * DISK_BYTES_PER_ENTRY
        return bytes_needed / (1024 ** 3)


@dataclass(frozen=True)
class CostBreakdown:
    shards: float
    coord: float
    masking: float
    coord_disk: float
    shard_disks: float
    nat: float

    @property
    def total(self) -> float:
        return self.shards + self.coord + self.masking + self.coord_disk + self.shard_disks + self.nat


def compute_cost(dict_capacity: float, cfg: ClusterConfig) -> tuple[CostBreakdown, int, float]:
    n = cfg.n_shards_for(dict_capacity)
    coord_disk_gb = cfg.coord_disk_gb_for(dict_capacity)
    shards = n * MACHINE_USD_PER_HOUR[cfg.shard_machine]
    coord = MACHINE_USD_PER_HOUR[cfg.coord_machine]
    masking = MACHINE_USD_PER_HOUR[cfg.masking_machine]
    coord_disk = coord_disk_gb * disk_usd_per_gb_hour(cfg.coord_disk_kind)
    shard_disks = n * cfg.shard_boot_disk_gb * disk_usd_per_gb_hour(cfg.shard_boot_disk_kind)
    nat = NAT_USD_PER_HOUR
    return (
        CostBreakdown(shards, coord, masking, coord_disk, shard_disks, nat),
        n,
        coord_disk_gb,
    )


# Three measured-regime markers — pinned to the bench's actual configs.
MEASURED_REGIMES = [
    ("small",  1 << 20, ClusterConfig(shard_log_capacity=22)),
    ("medium", 1 << 26, ClusterConfig(shard_log_capacity=27)),
    ("large",  1 << 32, ClusterConfig(shard_log_capacity=27)),
]


def plot_cost_vs_capacity(cfg: ClusterConfig, out_path: Path) -> Path:
    # Continuous capacity sweep — 2^20 (1M) to 2^40 (~1T) entries.
    log_caps = np.arange(20, 41)
    capacities = 2.0 ** log_caps
    breakdowns: list[CostBreakdown] = []
    n_shards_list: list[int] = []
    for cap in capacities:
        bd, n, _ = compute_cost(cap, cfg)
        breakdowns.append(bd)
        n_shards_list.append(n)

    shards = np.array([b.shards for b in breakdowns])
    coord = np.array([b.coord for b in breakdowns])
    masking = np.array([b.masking for b in breakdowns])
    coord_disk = np.array([b.coord_disk for b in breakdowns])
    shard_disks = np.array([b.shard_disks for b in breakdowns])
    nat = np.array([b.nat for b in breakdowns])
    total = shards + coord + masking + coord_disk + shard_disks + nat

    fig, ax = plt.subplots(figsize=(9, 5.5))
    components = [
        ("shard compute",  shards,      "#2ca02c"),
        ("coord disk",     coord_disk,  "#d62728"),
        ("shard disks",    shard_disks, "#8c564b"),
        ("coord compute",  coord,       "#1f77b4"),
        ("masking compute", masking,    "#9467bd"),
        ("NAT gateway",    nat,         "#7f7f7f"),
    ]
    ax.stackplot(
        capacities,
        [c[1] for c in components],
        labels=[c[0] for c in components],
        colors=[c[2] for c in components],
        alpha=0.85,
    )
    ax.plot(capacities, total, color="black", linewidth=1.6, label="total")

    # Mark the three measured regimes on the total curve.
    for name, cap, regime_cfg in MEASURED_REGIMES:
        bd, n, disk_gb = compute_cost(cap, regime_cfg)
        ax.scatter([cap], [bd.total], s=60, color="black", zorder=5)
        ax.annotate(
            f"{name}\n{n} shard{'s' if n != 1 else ''}, ${bd.total:.2f}/h",
            xy=(cap, bd.total),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("dictionary capacity (entries, log₂)")
    ax.set_ylabel("deployment cost (USD/hour, log)")
    ax.set_title(
        "Aegon cluster cost vs. dictionary capacity\n"
        f"(shard={cfg.shard_machine}, coord={cfg.coord_machine}, "
        f"shard_log_cap={cfg.shard_log_capacity}, disk sized for {cfg.max_fill_percent:g}% fill)"
    )

    # x-axis ticks at every 4th power of 2 for readability.
    tick_caps = [2 ** k for k in range(20, 41, 4)]
    ax.set_xticks(tick_caps)
    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda v, _: _human_count(v))
    )
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-machine",   default=ClusterConfig.shard_machine,   choices=list(MACHINE_USD_PER_HOUR))
    ap.add_argument("--coord-machine",   default=ClusterConfig.coord_machine,   choices=list(MACHINE_USD_PER_HOUR))
    ap.add_argument("--masking-machine", default=ClusterConfig.masking_machine, choices=list(MACHINE_USD_PER_HOUR))
    ap.add_argument("--coord-disk-kind", default=ClusterConfig.coord_disk_kind, choices=list(PD_USD_PER_GB_MONTH))
    ap.add_argument("--shard-log-capacity", type=int, default=ClusterConfig.shard_log_capacity)
    ap.add_argument("--max-fill-percent", type=float, default=ClusterConfig.max_fill_percent)
    ap.add_argument("--output", default=str(PLOTS_DIR / "cost_vs_capacity.pdf"))
    args = ap.parse_args()

    cfg = ClusterConfig(
        shard_machine=args.shard_machine,
        coord_machine=args.coord_machine,
        masking_machine=args.masking_machine,
        coord_disk_kind=args.coord_disk_kind,
        shard_log_capacity=args.shard_log_capacity,
        max_fill_percent=args.max_fill_percent,
    )
    out_path = Path(args.output)
    plot_cost_vs_capacity(cfg, out_path)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")

    # Print the regime table for reference.
    print()
    print(f"{'regime':<8} {'cap':>12} {'shards':>7} {'disk (TB)':>10} {'total $/h':>10} {'total $/mo':>12}")
    for name, cap, regime_cfg in MEASURED_REGIMES:
        bd, n, disk_gb = compute_cost(cap, regime_cfg)
        print(f"{name:<8} {_human_count(cap):>12} {n:>7} {disk_gb/1024:>10.2f} {bd.total:>9.2f}  {bd.total*24*30:>11.2f}")


if __name__ == "__main__":
    main()
