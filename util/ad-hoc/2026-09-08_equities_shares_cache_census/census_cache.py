"""Census of the cached SEC shares payloads: count, concepts, empties, all-zero payloads, mtimes.

Project: juniper-data
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-08
Status: ad-hoc — investigation (round-37 defect-register handoff validation, lane A3)
Retire when: the SEC shares cache gains a versioned key / TTL and the restatement + outlier
    findings are closed in the defect register (or the instruments are superseded by a test).
Related: juniper-ml/notes/JUNIPER_2026-08-14_JUNIPER-ECOSYSTEM_DEFECT-REGISTER.md;
    juniper-ml/prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-08_defect-register-round-38-*.md

READ-ONLY on the cache (``~/.cache/juniper_data/equities`` or ``JUNIPER_DATA_EQUITIES_CACHE_DIR``);
the network is blocked (``_sec_get`` is patched to raise) so a missing payload is reported, never
fetched. Output goes to the directory given as the first argument. Run with the JuniperData
interpreter from this directory (the scripts import ``a3lib`` from beside themselves):

    cd util/ad-hoc/2026-09-08_equities_shares_cache_census
    /opt/miniforge3/envs/JuniperData/bin/python census_cache.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — read-only census of the equities shares cache.

Reads every shares/*.json, records: file, cik, mtime, top-level keys, taxonomy,
tag, unit keys, number of points per unit, whether the payload is 'empty' under
the generator's own test (`not any(units.values())`).  Writes a CSV + summary.
Never writes into the cache directory.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

CACHE = Path(os.environ.get("JUNIPER_DATA_EQUITIES_CACHE_DIR", str(Path.home() / ".cache" / "juniper_data" / "equities")))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

rows = []
empties = []
parse_fail = []
for path in sorted((CACHE / "shares").glob("*.json")):
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        parse_fail.append((path.name, repr(exc)))
        continue
    units = data.get("units", {}) if isinstance(data, dict) else {}
    npts = {k: len(v) for k, v in units.items()}
    is_empty = not any(units.values())
    if is_empty:
        empties.append(path.name)
    rows.append(
        {
            "file": path.name,
            "cik": int(path.stem),
            "mtime_utc": mtime.isoformat(),
            "size": st.st_size,
            "top_keys": "|".join(sorted(data.keys())) if isinstance(data, dict) else type(data).__name__,
            "taxonomy": data.get("taxonomy") if isinstance(data, dict) else None,
            "tag": data.get("tag") if isinstance(data, dict) else None,
            "entityName": data.get("entityName") if isinstance(data, dict) else None,
            "unit_keys": "|".join(sorted(units.keys())),
            "n_points": sum(npts.values()),
            "n_points_by_unit": json.dumps(npts, sort_keys=True),
            "is_empty": is_empty,
        }
    )

import csv  # noqa: E402

with open(OUT / "cache_census.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

mtimes = sorted(r["mtime_utc"] for r in rows)
from collections import Counter  # noqa: E402

print(f"files parsed: {len(rows)}  parse failures: {len(parse_fail)} {parse_fail[:5]}")
print(f"empty payloads (generator test `not any(units.values())`): {len(empties)} {empties[:20]}")
print(f"mtime min: {mtimes[0]}  max: {mtimes[-1]}")
print("mtime by UTC date:", sorted(Counter(m[:10] for m in mtimes).items()))
print("taxonomy/tag counts:", Counter((r["taxonomy"], r["tag"]) for r in rows).most_common())
print("top-key signatures:", Counter(r["top_keys"] for r in rows).most_common())
print("unit-key signatures:", Counter(r["unit_keys"] for r in rows).most_common())
print("n_points: min", min(r["n_points"] for r in rows), "max", max(r["n_points"] for r in rows))
for r in rows:
    if r["cik"] in (21344, 1800):
        print("SPOTLIGHT", r)
