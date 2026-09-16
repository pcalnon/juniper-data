#!/usr/bin/env python3
"""Compare three bases for the equities scale-typo filter against the shapes that break them.

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-15
Status: ad-hoc -- one-off
Retire when: RETAINED -- ad-hoc scripts are kept as provenance of record (owner policy 2026-08-25)
Related: juniper-data#404; round-39 round-2 validation lane B1, which found the absorbing state

Round-2 lane B1 showed that the basis juniper-data#404 shipped -- the median of the values already
ACCEPTED -- has an **absorbing state**. If the first value a series ever offers is a typo that the
absolute ceiling does not reach, then ``kept_values == [typo]`` and every genuine value afterwards
is more than the factor away from it, so nothing is ever accepted again and the whole genuine
series is deleted. There is no recovery path, because the only thing that could widen the basis is
an acceptance.

That is not hypothetical arithmetic. Both ingredients are in the bundled cache today: a position-0
scale typo occurs for real (EOG), and sub-ceiling ~1000x typos occur in four series (PNR, REG, PKG,
MAA). Only their coincidence is absent, and the shares cache has a 7-day TTL.

Three candidates:

* ``accepted_only``  -- what #404 shipped. Rejected values never enter the basis.
* ``seen_shift``     -- the median of prior SEEN values. Self-corrects, because a rejected typo is
                        still only one value among a growing sample. Its weakness is n=2: a median
                        over two disagreeing values interpolates to a point that is far from both.
* ``lower_median``   -- the LOWER median of prior seen values, i.e. ``seen[(len(seen)-1)//2]`` of
                        the sorted priors. Always an actually-observed number, never an
                        interpolation, so the n=2 pathology cannot arise; and it self-corrects for
                        the same reason ``seen_shift`` does.
"""

from __future__ import annotations

import bisect
import statistics

FACTOR = 100.0


def accepted_only(values: list[float]) -> list[float]:
    accepted: list[float] = []
    keep: list[bool] = []
    for value in values:
        if not accepted:
            ok = True
        else:
            basis = statistics.median(accepted)
            ok = basis / FACTOR <= value <= basis * FACTOR
        keep.append(ok)
        if ok:
            bisect.insort(accepted, value)
    return [v for v, k in zip(values, keep) if k]


def seen_shift(values: list[float]) -> list[float]:
    seen: list[float] = []
    keep: list[bool] = []
    for value in values:
        if not seen:
            ok = True
        else:
            basis = statistics.median(seen)
            ok = basis / FACTOR <= value <= basis * FACTOR
        keep.append(ok)
        bisect.insort(seen, value)
    return [v for v, k in zip(values, keep) if k]


def lower_median(values: list[float]) -> list[float]:
    seen: list[float] = []
    keep: list[bool] = []
    for value in values:
        if not seen:
            ok = True
        else:
            basis = seen[(len(seen) - 1) // 2]
            ok = basis / FACTOR <= value <= basis * FACTOR
        keep.append(ok)
        bisect.insort(seen, value)
    return [v for v, k in zip(values, keep) if k]


CANDIDATES = (
    ("accepted-only (SHIPPED in #404)", accepted_only),
    ("seen median, shifted", seen_shift),
    ("LOWER median of prior seen", lower_median),
)

# `genuine_below` separates the real counts from the typos in each shape.
CASES: tuple[tuple[str, list[float], float], ...] = (
    ("AIZ shape -- typo at position 1", [117_926_517.0, 116_799_796_000.0, 116_485_888.0, 111_826_599.0, 112_000_000.0], 1e10),
    ("EOG shape -- typo at position 0", [251_931_774_000.0, 252_355_378.0, 252_578_053.0, 253_152_632.0, 254_000_000.0], 1e10),
    ("the n=2 interpolation case", [1.0e8, 5.0e10, 1.01e8, 1.02e8, 1.03e8], 1e10),
    ("PNR shape -- SUB-CEILING typo at position 0", [9.84e10] + [9.87e7 + i * 1e5 for i in range(20)], 1e10),
    ("placeholder at the floor, then genuine", [100_000.0] + [1_071_666_977.0] * 20, 1e10),
)

for name, values, genuine_below in CASES:
    genuine = [v for v in values if v < genuine_below]
    print(f"\n{name}   (n={len(values)}, of which genuine {len(genuine)})")
    for label, fn in CANDIDATES:
        kept = fn(values)
        kept_genuine = [v for v in kept if v < genuine_below]
        verdict = "" if len(kept_genuine) == len(genuine) and len(kept) == len(genuine) else "   <-- WRONG"
        print(f"   {label:34} kept {len(kept):>3}/{len(values):<3}  genuine kept {len(kept_genuine):>3}/{len(genuine):<3}  max {max(kept):>18,.0f}{verdict}")
