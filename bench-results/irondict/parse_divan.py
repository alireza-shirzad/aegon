#!/usr/bin/env python3
"""Parse irondict's divan bench logs into a flat JSON shape that
plot_lookup.py can overlay alongside aegon and AKD.

Source logs live in /home/alrshir/irondict/bench-results/*.log (the
small+medium runs from the dev box) and on the m1-megamem-96 VM
under ~/irondict/bench-results/*.log (the large-regime run).

Output: one {small,medium,large}.json per regime, alongside this file.

Schema is irondict-specific (NOT aegon's LevelStats) because irondict
has no fill_percent dimension — every bench runs on a near-empty
tree. The plot helpers in plot_lookup.py treat irondict overlays as
"system = irondict" horizontal lines spanning the fill axis.

Each per-regime JSON contains the rows divan actually emitted —
e.g. small.json may have rows for log_capacity 22 and 26 if the dev
box bench captured both before OOMing at log_cap=32, even though only
the 22 row is "small". The plot integration picks the correct
log_capacity for each regime.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DIVAN_TIME_RE = re.compile(
    r"^\s*([\d.]+)\s+(ns|µs|us|ms|s|m)\s+│\s+([\d.]+)\s+(ns|µs|us|ms|s|m)\s+│\s+"
    r"([\d.]+)\s+(ns|µs|us|ms|s|m)\s+│\s+([\d.]+)\s+(ns|µs|us|ms|s|m)\s+│\s+"
    r"(\d+)\s+│\s+(\d+)"
)

# divan row that introduces a single-axis bench arg (e.g. "├─ 22" or
# "╰─ 22 Computing SRS" or "├─ 22 Cache miss: ..."). The arg is the
# log_capacity for setup/audit/lookups. Some benches print prelude
# text on the SAME line as the arg header (setup says "Computing SRS",
# server_lookup says "Cache miss: ..."), so we accept arbitrary
# trailing text after the digits and use a word-boundary anchor.
SINGLE_ARG_RE = re.compile(r"^\s+[├╰]─\s+(\d+)(?:\s+\S.*)?\s*$")

# divan row for a multi-axis bench
# ("├─ [log(capacity)=22, log(|update|)=4, log(init-update)=2]   ...")
MULTI_ARG_RE = re.compile(
    r"^\s+[├╰]─\s+\[log\(capacity\)=(\d+),\s+log\(\|update\|\)=(\d+),\s+log\(init-update\)=(\d+)\]"
)

# "Posted bulletin board message size: 12852 bytes" — sometimes
# appears on the same line as the divan arg header (after spaces) and
# sometimes on a subsequent line.
BB_SIZE_RE = re.compile(r"Posted bulletin board message size:\s+(\d+)\s+bytes")

# "proof size = 4784" and "client key size = 1070150" (client_lookup
# bench prints these on the line just before/after the row header).
PROOF_SIZE_RE = re.compile(r"proof size\s*=\s*(\d+)")
CK_SIZE_RE = re.compile(r"client key size\s*=\s*(\d+)")


def _parse_time_to_ms(value: float, unit: str) -> float:
    if unit == "ns":
        return value / 1e6
    if unit in ("µs", "us"):
        return value / 1000.0
    if unit == "ms":
        return value
    if unit == "s":
        return value * 1000.0
    if unit == "m":
        return value * 60_000.0
    raise ValueError(f"unknown time unit {unit!r}")


def _next_time_row(lines: list[str], start: int) -> tuple[int, dict | None]:
    """Search forward from `start` (exclusive) for the next divan time
    row. Returns (index_of_row, parsed_dict) or (len(lines), None) if
    no row appears before EOF or before the next arg header."""
    i = start
    while i < len(lines):
        line = lines[i]
        # Bail if we hit the NEXT arg header — the previous arg row had
        # no data (likely OOMed before producing a row).
        if SINGLE_ARG_RE.match(line) or MULTI_ARG_RE.match(line):
            return i, None
        m = DIVAN_TIME_RE.search(line)
        if m:
            g = m.groups()
            return i, {
                "fastest_ms": _parse_time_to_ms(float(g[0]), g[1]),
                "slowest_ms": _parse_time_to_ms(float(g[2]), g[3]),
                "median_ms": _parse_time_to_ms(float(g[4]), g[5]),
                "mean_ms": _parse_time_to_ms(float(g[6]), g[7]),
                "samples": int(g[8]),
                "iters": int(g[9]),
            }
        i += 1
    return i, None


def parse_single_axis_log(path: Path) -> dict[int, dict]:
    """Parse setup.log / audit.log / server_lookup.log / client_lookup.log.

    Returns {log_capacity: {time row + any extra fields scraped from
    prelude lines}}.
    """
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        m = SINGLE_ARG_RE.match(lines[i])
        if not m:
            i += 1
            continue
        log_cap = int(m.group(1))
        end, row = _next_time_row(lines, i + 1)
        if row is None:
            # No data row before the next arg / EOF — skip this entry.
            i = end
            continue
        # Scrape the lines BETWEEN the arg header and the time row for
        # proof size / client key size / bulletin board size — these
        # bench-specific aux fields are printed inline by the irondict
        # bench code.
        for between in lines[i + 1:end]:
            mp = PROOF_SIZE_RE.search(between)
            if mp:
                row["proof_bytes"] = int(mp.group(1))
            mc = CK_SIZE_RE.search(between)
            if mc:
                row["client_key_bytes"] = int(mc.group(1))
            mb = BB_SIZE_RE.search(between)
            if mb:
                row["bulletin_bytes"] = int(mb.group(1))
        out[log_cap] = row
        i = end + 1
    return out


def parse_multi_axis_log(path: Path) -> dict[int, dict[int, dict]]:
    """Parse server_update_keys.log / server_update_reg.log.

    Returns {log_capacity: {log_update: {time row + bulletin_bytes}}}.
    """
    out: dict[int, dict[int, dict]] = {}
    if not path.exists():
        return out
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        m = MULTI_ARG_RE.match(lines[i])
        if not m:
            i += 1
            continue
        log_cap = int(m.group(1))
        log_upd = int(m.group(2))
        # The "Posted bulletin board message size" line sometimes sits
        # ON the same arg-header line (after the bench args) and
        # sometimes on a subsequent line, so check both.
        bb = None
        mb = BB_SIZE_RE.search(lines[i])
        if mb:
            bb = int(mb.group(1))
        end, row = _next_time_row(lines, i + 1)
        if row is None:
            i = end
            continue
        if bb is None:
            for between in lines[i + 1:end]:
                mb = BB_SIZE_RE.search(between)
                if mb:
                    bb = int(mb.group(1))
                    break
        if bb is not None:
            row["bulletin_bytes"] = bb
        out.setdefault(log_cap, {})[log_upd] = row
        i = end + 1
    return out


# ---- Regime assembly ----

# Polynomial size (log2) for each regime. Mirrors aegon's
# `shard_log_capacity = true_log_capacity + LOG2_OVER_PROVISIONING_FACTOR`
# with over-prov 4 (LOG2_OVER_PROVISIONING_FACTOR=2): a "small" regime
# serves 2^20 users on a 2^22-slot polynomial, etc. irondict's
# `IronSpecification::new(capacity, _)` interprets `capacity` as the
# polynomial size directly (no internal over-prov), so we pass
# `1usize << log_cap` from these numbers.
REGIME_LOG_CAPACITY = {"small": 22, "medium": 28, "large": 34}


def _derive_audit_bytes(pk_rows: dict[int, dict], pr_rows: dict[int, dict]) -> int | None:
    """Per-epoch audit payload = one IronEpochRegMessage + one
    IronEpochKeyMessage. Pick the smallest measured batch for each;
    the publish benches don't go below batch=16 but the bulletin
    size is nearly invariant with batch for publish_reg and varies
    only mildly for publish_keys, so the approximation is close to
    what the audit bench's batch=4 input would yield.
    Returns None if neither publish bench has data."""
    def smallest(rows):
        if not rows:
            return None
        log_min = min(rows.keys())
        bb = rows[log_min].get("bulletin_bytes")
        return int(bb) if bb is not None else None
    a = smallest(pk_rows)
    b = smallest(pr_rows)
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def assemble_regime(
    regime: str,
    bench_dir_local: Path,
    bench_dir_vm: Path | None,
) -> dict:
    """Merge the local (dev-box) and VM (large) bench results for one
    regime. For small/medium we prefer dev-box numbers; for large we
    only have VM numbers (the dev box OOMs at log_cap=32).
    """
    lc = REGIME_LOG_CAPACITY[regime]

    # Pick the directory we read from. The dev box has every bench at
    # log_cap=22 and 26 with a SIGABRT at 32; the VM has the bench at
    # log_cap=32 (and possibly 22+26 if the bench file's args wasn't
    # trimmed).
    if regime == "large":
        primary = bench_dir_vm or bench_dir_local
    else:
        primary = bench_dir_local

    setup = parse_single_axis_log(primary / "setup.log").get(lc)
    audit = parse_single_axis_log(primary / "audit.log").get(lc)
    server_lookup = parse_single_axis_log(primary / "server_lookup.log").get(lc)
    client_lookup = parse_single_axis_log(primary / "client_lookup.log").get(lc)

    pub_keys = parse_multi_axis_log(primary / "server_update_keys.log").get(lc, {})
    pub_reg = parse_multi_axis_log(primary / "server_update_reg.log").get(lc, {})

    def by_batch(rows: dict[int, dict]) -> list[dict]:
        out_list = []
        for log_upd, row in sorted(rows.items()):
            out_list.append({
                "log_update": log_upd,
                "batch_size": 1 << log_upd,
                "median_ms": row.get("median_ms"),
                "fastest_ms": row.get("fastest_ms"),
                "slowest_ms": row.get("slowest_ms"),
                "mean_ms": row.get("mean_ms"),
                "bulletin_bytes": row.get("bulletin_bytes"),
            })
        return out_list

    return {
        "params": {
            "log_capacity": lc,
            "regime": regime,
        },
        "setup_ms": setup.get("median_ms") if setup else None,
        "lookup": {
            "server_ms": server_lookup.get("median_ms") if server_lookup else None,
            "client_ms": client_lookup.get("median_ms") if client_lookup else None,
            "proof_bytes": client_lookup.get("proof_bytes") if client_lookup else None,
            "client_key_bytes": client_lookup.get("client_key_bytes") if client_lookup else None,
        },
        "audit": {
            "audit_ms": audit.get("median_ms") if audit else None,
            # Audit payload size: IronAuditor::verify_update (see
            # iron-key/src/auditor/mod.rs) runs both
            # check_reg_update AND check_keys_update on the
            # bulletin board, so an auditor must consume one
            # IronEpochRegMessage + one IronEpochKeyMessage per
            # epoch. We approximate the per-epoch audit payload as
            # the sum of the smallest measured publish_reg and
            # publish_keys bulletin sizes (batch_size=16 — irondict
            # publish benches don't go below 16, so this is a
            # slight overestimate for the audit bench's batch=4
            # setup, but the bulletin size is nearly batch-
            # invariant for publish_reg and decreases mildly with
            # batch for publish_keys, so the error is small).
            "audit_bytes": _derive_audit_bytes(pub_keys, pub_reg),
        },
        "publish_keys": {"by_batch": by_batch(pub_keys)},
        "publish_reg": {"by_batch": by_batch(pub_reg)},
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "bench-results" / "irondict"
    out_dir.mkdir(parents=True, exist_ok=True)

    local = Path("/home/alrshir/irondict/bench-results")
    vm = out_dir / "vm-bench-results"  # we'll rsync into here

    for regime in ("small", "medium", "large"):
        data = assemble_regime(regime, local, vm if vm.exists() else None)
        out_path = out_dir / f"{regime}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"wrote {out_path}")
        # Quick summary
        s = data["setup_ms"]
        a = data["audit"]["audit_ms"]
        sl = data["lookup"]["server_ms"]
        cl = data["lookup"]["client_ms"]
        pk = len(data["publish_keys"]["by_batch"])
        pr = len(data["publish_reg"]["by_batch"])
        print(f"  setup={s} audit={a} server_lookup={sl} client_lookup={cl} "
              f"publish_keys_rows={pk} publish_reg_rows={pr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
