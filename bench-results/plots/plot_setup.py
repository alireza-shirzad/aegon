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
IRONDICT_DIR = REPO_ROOT / "bench-results" / "irondict"
AKD_DIR = REPO_ROOT / "bench-results" / "akd-mysql"
OUT_DIR = REPO_ROOT / "bench-results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical regime order + presentation labels. The large-regime
# `setup-bench` runs a *per-shard* sequential proxy (single-machine
# sequential gen at log_cap=27), since the distributed setup time is
# measured separately on the cluster. We label it as such.
REGIME_ORDER = ["small", "medium", "large-per-shard"]
REGIME_LABEL = {
    "small":           "aegon small",
    "medium":          "aegon medium",
    "large-per-shard": "aegon large",
}
# Colour-blind friendly palette. Same triple used by every regime-
# stratified plot — keeps the paper's figures internally consistent.
REGIME_COLOR = {
    "small":           "#4477AA",
    "medium":          "#EE6677",
    "large-per-shard": "#228833",
}

# irondict overlay. Same approach as the lookup/audit/publish plots:
# load per-regime IronStats-style records and render alongside aegon
# bars. Different colour palette and an "irondict-" label prefix so
# the eye distinguishes the two systems at a glance. log_capacity
# annotations include the full 1-shard size irondict actually
# operates on (no per-shard split).
IRONDICT_ORDER = ["irondict-small", "irondict-medium", "irondict-large"]
# N is the polynomial size (same convention as aegon's REGIME_LABEL
# above). With over-prov 4 these correspond to 2^20 / 2^26 / 2^32
# users — exactly matching aegon's small / medium total / large total
# user counts (aegon's large is 128 shards × 2^27 poly = 2^34 total
# poly, so irondict-large at 2^34 mirrors the whole aegon-large
# system on a single shard).
IRONDICT_LABEL = {
    "irondict-small":  "irondict small",
    "irondict-medium": "irondict medium",
    "irondict-large":  "irondict large",
}
IRONDICT_COLOR = {
    "irondict-small":  "#88CCEE",
    "irondict-medium": "#FFAABB",
    "irondict-large":  "#99DDAA",
}

# AKD overlay. AKD has no trusted setup — its only key material is a
# VRF keypair (32 B each in WhatsApp's Ed25519-based deployment), so
# every regime reads the same constants from the AKD JSON files.
AKD_ORDER = ["akd-small", "akd-medium", "akd-large"]
AKD_LABEL = {
    "akd-small":  "AKD small",
    "akd-medium": "AKD medium",
    "akd-large":  "AKD large",
}
AKD_COLOR = {
    "akd-small":  "#CCBB44",
    "akd-medium": "#EE8866",
    "akd-large":  "#AA6633",
}


def _load_akd_record(regime_key: str) -> dict | None:
    """Load AKD setup info from akd-mysql/{regime}.json. Returns the
    same record shape as aegon/irondict so the plot helpers stay
    schema-agnostic."""
    path = AKD_DIR / f"{regime_key.replace('akd-', '')}.json"
    if not path.exists():
        return None
    with path.open() as f:
        d = json.load(f)
    setup_secs = d.get("setup_secs")
    pk = d.get("prover_key_bytes")
    vk = d.get("verifier_key_bytes")
    if setup_secs is None and pk is None and vk is None:
        return None
    rec: dict = {
        "label": regime_key,
        "_source": "akd",
    }
    if setup_secs is not None:
        rec["gen_duration_secs"] = float(setup_secs)
        rec["gen_seconds_samples"] = [float(setup_secs)]
    if pk is not None:
        rec["prover_param_bytes"] = int(pk)
    if vk is not None:
        rec["verifier_param_bytes"] = int(vk)
    return rec


def _load_irondict_record(regime_key: str, log_cap: int) -> dict | None:
    """Load one irondict regime's JSON and re-shape it into the same
    record format as aegon's setup records, so plot_setup_time and
    plot_key_sizes treat both systems uniformly.

    irondict has no trim step — the prover key IS the SRS file the
    setup writes to disk. The bench file doesn't emit the SRS size,
    so we derive it: at K=7 N=2^22 the SRS structure is identical
    to aegon's (same KZH-K parameters → byte-identical SRS), and at
    K=9 N=2^28 the SRS scales linearly with N relative to aegon's
    medium-per-shard reading at K=9 N=2^27. These derivations are
    only valid because the user wired irondict's `compute_k` to
    aegon's `optimal_kzh_k` table earlier in the session.

    Returns None if no measurement exists for this regime yet (e.g.
    large.json is empty until the c4 bench completes)."""
    path = IRONDICT_DIR / f"{regime_key.replace('irondict-', '')}.json"
    if not path.exists():
        return None
    with path.open() as f:
        d = json.load(f)
    setup_ms = d.get("setup_ms")
    lookup = d.get("lookup") or {}
    ck_bytes = lookup.get("client_key_bytes")
    if setup_ms is None and ck_bytes is None:
        return None
    rec: dict = {
        "label": regime_key,
        "shard_log_capacity": log_cap,
        # K param picked by compute_k() in irondict's pcs/kzhk/mod.rs
        # (mirrors aegon's optimal_kzh_k table — wired by the user
        # before this bench ran). 22→7, 28→9, 34→11.
        "kzh_k": {22: 7, 28: 9, 34: 11}.get(log_cap, 0),
        "_source": "irondict",
    }
    if setup_ms is not None:
        rec["gen_duration_secs"] = setup_ms / 1000.0
        rec["gen_seconds_samples"] = [setup_ms / 1000.0]
    if ck_bytes is not None:
        # client_key_bytes is irondict's verifier param (the client
        # holds it to verify lookup proofs); this maps directly to
        # aegon's verifier_param_bytes.
        rec["verifier_param_bytes"] = ck_bytes
    # Prover-key size derivation (no trim → pk = SRS):
    #   - K=7 N=2^22: byte-identical to aegon small's pk = 288.7 MB,
    #     since both systems generate the same KZH-K SRS at the same
    #     (N, K). Confirmed by file-stat on the m1-megamem run
    #     before deletion.
    #   - K=9 N=2^28: aegon's medium-per-shard SRS at K=9 N=2^27 is
    #     9.82 GB (from medium_cluster.json's srs_bytes). KZH-K SRS
    #     scales linearly in N for fixed K, so irondict's medium at
    #     2× the polynomial size = ~19.6 GB. Marked as derived in
    #     the plot caption.
    SRS_BYTES_BY_LOG_CAP = {
        22: 288_679_685,          # measured on m1; identical to aegon small
        28: 2 * 9_819_476_306,    # derived: 2× aegon medium-per-shard SRS
        # 34: supplied via irondict/large.json `prover_param_bytes`
        #     manual override (bench OOMed on c3-highmem-176).
    }
    # JSON-supplied prover_param_bytes wins over the derived table.
    if "prover_param_bytes" in d:
        rec["prover_param_bytes"] = int(d["prover_param_bytes"])
    elif log_cap in SRS_BYTES_BY_LOG_CAP:
        rec["prover_param_bytes"] = SRS_BYTES_BY_LOG_CAP[log_cap]
    return rec


def _load_one(regime: str) -> dict | None:
    """Return a normalized record for `regime`, preferring the
    aegon_setup_bench schema and falling back to the cluster schema
    (mapped to the same field names so callers stay schema-agnostic).
    Returns None when neither file exists."""
    legacy = SETUP_DIR / f"{regime}.json"
    if legacy.exists():
        with legacy.open() as f:
            return json.load(f)
    # The large-regime per-shard bench file is `per-shard.json`, not
    # `large-per-shard.json` — it was a single-machine in-process run
    # at log_cap=27 with the trim step (so it carries verifier params).
    if regime == "large-per-shard":
        per_shard = SETUP_DIR / "per-shard.json"
        if per_shard.exists():
            with per_shard.open() as f:
                rec = json.load(f)
            rec["label"] = "large-per-shard"
            return rec
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
        rec = {
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
        # Cluster bench skips the trim step so no verifier_param_bytes.
        # If the per-shard.json single-machine bench (which DOES trim)
        # used the same shard config, its vk applies here too — same
        # KZH-K params → same verifier-key byte layout. Borrow it so
        # the verifier-key panel can render the cluster regimes.
        per_shard_path = SETUP_DIR / "per-shard.json"
        if per_shard_path.exists():
            with per_shard_path.open() as f:
                ps = json.load(f)
            if (int(ps.get("shard_log_capacity", -1)) == rec["shard_log_capacity"]
                    and int(ps.get("kzh_k", -1)) == rec["kzh_k"]
                    and "verifier_param_bytes" in ps):
                rec["verifier_param_bytes"] = int(ps["verifier_param_bytes"])
                rec["_verifier_param_source"] = "per-shard.json (matching shard config)"
        return rec
    return None


def load_records() -> list[dict]:
    records = []
    for name in REGIME_ORDER:
        rec = _load_one(name)
        if rec is None:
            print(f"[warn] missing setup/{name}.json or setup/{name}_cluster.json; skipping")
            continue
        records.append(rec)
    # irondict regimes are appended after aegon. Order kept consistent
    # so the bar chart reads aegon-small, aegon-medium, aegon-large,
    # irondict-small, irondict-medium, irondict-large left-to-right.
    irondict_caps = {"irondict-small": 22, "irondict-medium": 28, "irondict-large": 34}
    for name in IRONDICT_ORDER:
        rec = _load_irondict_record(name, irondict_caps[name])
        if rec is None:
            print(f"[warn] missing irondict/{name.replace('irondict-', '')}.json or no measurement; skipping")
            continue
        records.append(rec)
    for name in AKD_ORDER:
        rec = _load_akd_record(name)
        if rec is None:
            print(f"[warn] missing akd-mysql/{name.replace('akd-', '')}.json setup fields; skipping")
            continue
        records.append(rec)
    if not records:
        raise SystemExit("no setup JSONs found under " + str(SETUP_DIR))
    return records


def _record_label(rec: dict) -> str:
    """Pick the display label dict for a record (aegon, irondict, AKD)."""
    label = rec["label"]
    if label in REGIME_LABEL:
        return REGIME_LABEL[label]
    if label in IRONDICT_LABEL:
        return IRONDICT_LABEL[label]
    if label in AKD_LABEL:
        return AKD_LABEL[label]
    return label


def _record_color(rec: dict) -> str:
    label = rec["label"]
    if label in REGIME_COLOR:
        return REGIME_COLOR[label]
    if label in IRONDICT_COLOR:
        return IRONDICT_COLOR[label]
    if label in AKD_COLOR:
        return AKD_COLOR[label]
    return "#888888"


def _has_vk(rec: dict) -> bool:
    """Cluster-mode records don't have a trimmed verifier_param_bytes;
    the bench skips trim because each shard uses the full SRS as its
    prover param. Filter such records out of the key-sizes chart."""
    return "verifier_param_bytes" in rec


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


def fmt_time(t: float) -> str:
    """Human-readable setup-time annotation."""
    if t < 1.0:
        ms = t * 1000.0
        if ms >= 10:
            return f"{ms:.0f} ms"
        if ms >= 1:
            return f"{ms:.1f} ms"
        return f"{ms:.2f} ms"
    if t < 60:
        return f"{t:.1f} s"
    if t < 3600:
        return f"{t/60.0:.1f} min"
    return f"{t/3600.0:.2f} h"


# Subsets the combined plot understands. Each maps to the list of
# `record.label` strings that should appear in that PDF.
SUBSETS = {
    "small_medium": ["small", "medium",
                     "akd-small", "akd-medium",
                     "irondict-small", "irondict-medium"],
    "large":        ["large-per-shard", "akd-large", "irondict-large"],
}


def plot_setup_combined(records: list[dict], subset_name: str) -> Path:
    """One PDF per subset, three panels side by side: prover key,
    verifier key, setup time. Each panel is a bar chart with one bar
    per (system, regime) record that has the corresponding metric.
    Bars missing a metric are silently omitted (cluster-mode records
    don't have verifier_param_bytes, irondict-large currently has no
    setup_ms, etc.)."""
    wanted = SUBSETS[subset_name]
    subset_recs = [r for r in records if r["label"] in wanted]
    # Preserve the user's preferred left-to-right order: aegon first,
    # then irondict, matching the entries in SUBSETS above.
    subset_recs.sort(key=lambda r: wanted.index(r["label"]))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    ax_pk, ax_vk, ax_time = axes

    def _bar_panel(ax, metric_key: str, title: str, ylabel: str,
                   fmt_fn, value_fn=lambda r, k: r.get(k)):
        rows = [(r, value_fn(r, metric_key)) for r in subset_recs]
        rows = [(r, v) for r, v in rows if v is not None and v > 0]
        if not rows:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#666666")
            ax.set_xticks([])
            ax.set_title(title)
            return
        labels = [_record_label(r) for r, _ in rows]
        vals = [v for _, v in rows]
        colors = [_record_color(r) for r, _ in rows]
        x = np.arange(len(rows))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        # Stretch the y-axis upper bound on log scale so annotations
        # sit comfortably above the tallest bar instead of clipping.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 2.0)
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.10,
                    fmt_fn(v), ha="center", va="bottom", fontsize=8)

    _bar_panel(ax_pk, "prover_param_bytes",   "Prover key",   "size (bytes, log scale)",  fmt_bytes)
    _bar_panel(ax_vk, "verifier_param_bytes", "Verifier key", "size (bytes, log scale)",  fmt_bytes)
    _bar_panel(ax_time, "gen_duration_secs",  "Setup time",   "seconds (log scale)",     fmt_time)

    fig.tight_layout()
    out = OUT_DIR / f"setup_{subset_name}.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    records = load_records()
    print(f"loaded {len(records)} setup records: "
          + ", ".join(r["label"] for r in records))
    # Clean up the two now-stale PDFs from the previous layout so the
    # plots directory doesn't accumulate orphans.
    for stale in ("setup_time.pdf", "setup_key_sizes.pdf"):
        p = OUT_DIR / stale
        if p.exists():
            p.unlink()
            print(f"removed stale {p.relative_to(REPO_ROOT)}")
    for subset in SUBSETS:
        out = plot_setup_combined(records, subset)
        print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
