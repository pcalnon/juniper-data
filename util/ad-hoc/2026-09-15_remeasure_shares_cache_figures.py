#!/usr/bin/env python3
"""Re-measure every cache-derived figure quoted in the equities generator's comments.

Project: juniper-data
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-15
Status: ad-hoc -- one-off
Retire when: RETAINED -- ad-hoc scripts are kept as provenance of record (owner policy 2026-08-25)
Related: juniper-data#395; register rows APD-DATA-043 / -047; round-39 validation lane B1

The comments in ``juniper_data/generators/equities/generator.py`` quote counts taken against a
485-payload sweep; the cache now holds 486. Rather than bump the denominator and leave every
numerator asserting a measurement nobody redid, this recomputes all of them, INCLUDING the ones
that change meaning under the shifted (head-typo-proof) filter.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CACHE = Path.home() / ".cache/juniper_data/equities/shares"
FLOOR, FACTOR = 100_000.0, 100.0
CEIL_OLD, CEIL_NEW = 1.0e13, 1.0e11
STALE_DAYS = 365


def raw(payload: dict) -> pd.DataFrame:
    rows = []
    for arr in (payload.get("units") or {}).values():
        if isinstance(arr, list):
            for pt in arr:
                if pt.get("val") is not None and pt.get("end"):
                    rows.append((pt["end"], pt.get("filed") or None, float(pt["val"])))
    df = pd.DataFrame(rows, columns=["end", "filed", "shares"])
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    return df.dropna(subset=["end"])


files = sorted(CACHE.glob("*.json"))
payload_count = len(files)
with_obs = 0
restatement_ciks = 0
rows_moved = 0
end_collision = 0
stopped_before_cutoff = 0
stale_series = 0
lose_a_point_old = 0
lose_a_point_new = 0
sizes = []

CUTOFF = pd.Timestamp("2025-06-01")
for path in files:
    try:
        df = raw(json.loads(path.read_text()))
    except Exception:
        continue
    sizes.append(path.stat().st_size)
    if df.empty:
        continue
    with_obs += 1

    # A restatement: the same period END filed more than once.
    per_end = df.groupby("end")["filed"].nunique()
    if (per_end > 1).any():
        restatement_ciks += 1
        rows_moved += int(df[df["end"].isin(per_end[per_end > 1].index)].shape[0])

    # An end-collision the unstable sort could reorder: same (end) with distinct filed dates
    # AND distinct values -- the case where picking the wrong row changes the answer.
    collide = df.groupby("end").filter(lambda g: g["filed"].nunique() > 1 and g["shares"].nunique() > 1)
    if len(collide):
        end_collision += 1

    last_filed = df["filed"].dropna().max()
    if pd.notna(last_filed):
        if last_filed < CUTOFF:
            stopped_before_cutoff += 1
        if (df["end"].max() - last_filed).days > STALE_DAYS:
            stale_series += 1

    obs = df.drop_duplicates(subset=["end", "filed"], keep="last")
    obs = obs.sort_values(["filed", "end"], kind="stable", na_position="first").reset_index(drop=True)

    old = obs[(obs["shares"] >= FLOOR) & (obs["shares"] <= CEIL_OLD)]
    if len(old):
        run = old["shares"].expanding(min_periods=3).median()
        keep = run.isna() | ((old["shares"] >= run / FACTOR) & (old["shares"] <= run * FACTOR))
        if int((~keep).sum()):
            lose_a_point_old += 1

    new = obs[(obs["shares"] >= FLOOR) & (obs["shares"] <= CEIL_NEW)]
    dropped_by_ceiling = len(obs) - len(new)
    if len(new):
        prior = new["shares"].expanding().median().shift(1)
        keep = prior.isna() | ((new["shares"] >= prior / FACTOR) & (new["shares"] <= prior * FACTOR))
        if int((~keep).sum()) or dropped_by_ceiling:
            lose_a_point_new += 1
    elif dropped_by_ceiling:
        lose_a_point_new += 1

print(f"payload files in cache                         : {payload_count}")
print(f"payloads carrying at least one observation     : {with_obs}")
print(f"median payload size                            : {pd.Series(sizes).median():,.0f} bytes")
print(f"CIKs carrying a restatement (end filed >1x)    : {restatement_ciks}")
print(f"rows in restated periods                       : {rows_moved:,}")
print(f"CIKs with a value-changing end collision       : {end_collision}")
print(f"series whose last filing precedes 2025-06-01   : {stopped_before_cutoff}")
print(f"series silent >{STALE_DAYS}d at their last period end : {stale_series}")
print(f"CIKs losing >=1 point, SHIPPED filter (1e13)   : {lose_a_point_old}")
print(f"CIKs losing >=1 point, SHIFTED filter (1e11)   : {lose_a_point_new}")
