#!/usr/bin/env python3
"""Does the lower-median basis change any DELIVERED series against the real cache?

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-15
Status: ad-hoc -- one-off
Retire when: RETAINED -- ad-hoc scripts are kept as provenance of record (owner policy 2026-08-25)
Related: juniper-data#404; round-39 round-2 validation lane B1

Compares delivered MULTISETS, not maxima -- round 1 established that a maxima-only comparison
hides over-deletion, and this is the check that mistake earned.
"""

from __future__ import annotations

import bisect
import json
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

CACHE = Path.home() / ".cache/juniper_data/equities/shares"
FLOOR, CEILING, FACTOR = 100_000.0, 1.0e11, 100.0


def observations(path: Path) -> pd.DataFrame:
    rows = []
    for arr in (json.loads(path.read_text()).get("units") or {}).values():
        if isinstance(arr, list):
            for point in arr:
                if point.get("val") is not None and point.get("end"):
                    rows.append((point["end"], point.get("filed") or None, float(point["val"])))
    frame = pd.DataFrame(rows, columns=["end", "filed", "shares"])
    if frame.empty:
        return frame
    frame["end"] = pd.to_datetime(frame["end"], errors="coerce")
    frame["filed"] = pd.to_datetime(frame["filed"], errors="coerce")
    frame = frame.dropna(subset=["end"]).drop_duplicates(subset=["end", "filed"], keep="last")
    frame = frame.sort_values(["filed", "end"], kind="stable", na_position="first").reset_index(drop=True)
    return frame[(frame["shares"] >= FLOOR) & (frame["shares"] <= CEILING)].reset_index(drop=True)


def accepted_only(values):
    accepted, keep = [], []
    for value in values:
        ok = (not accepted) or (statistics.median(accepted) / FACTOR <= value <= statistics.median(accepted) * FACTOR)
        keep.append(ok)
        if ok:
            bisect.insort(accepted, value)
    return [v for v, k in zip(values, keep) if k]


def lower_median(values):
    seen, keep = [], []
    for value in values:
        if not seen:
            ok = True
        else:
            basis = seen[(len(seen) - 1) // 2]
            ok = basis / FACTOR <= value <= basis * FACTOR
        keep.append(ok)
        bisect.insort(seen, value)
    return [v for v, k in zip(values, keep) if k]


differing, emptied_by_new, extra_deleted, extra_kept = [], [], 0, 0
total = 0
for path in sorted(CACHE.glob("*.json")):
    frame = observations(path)
    if frame.empty:
        continue
    total += 1
    values = [float(v) for v in frame["shares"]]
    old, new = accepted_only(values), lower_median(values)
    if Counter(old) != Counter(new):
        differing.append((path.stem, len(values), len(old), len(new)))
        extra_deleted += max(0, len(old) - len(new))
        extra_kept += max(0, len(new) - len(old))
    if old and not new:
        emptied_by_new.append(path.stem)

print(f"series examined                                   : {total}")
print(f"series whose DELIVERED MULTISET differs            : {len(differing)}")
print(f"observations the lower-median basis keeps and #404 deleted : {extra_kept}")
print(f"observations the lower-median basis deletes and #404 kept  : {extra_deleted}")
print(f"series the lower-median basis empties that #404 did not    : {emptied_by_new or 'none'}")
for stem, raw, old_n, new_n in differing[:10]:
    print(f"    cik {stem}: {raw} in bounds, #404 kept {old_n}, lower-median kept {new_n}")
