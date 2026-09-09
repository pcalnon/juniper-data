"""Shared helpers: a standalone replica of _fetch_shares's parse/dedup/outlier steps, so every number can be re-derived without the package, plus loaders for the on-disk SEC cache.

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
    /opt/miniforge3/envs/JuniperData/bin/python a3lib.py /tmp/shares-census-out

Original notes
--------------
Lane A3 shared helpers — a faithful, standalone replica of the parse/dedup
step of EquitiesGenerator._fetch_shares (generator.py:916-936 @ 03b7548f) so
that every number can be re-derived without the package, plus the real
function for cross-checking.  READ-ONLY on the cache.
"""

import json
import os
from pathlib import Path

import pandas as pd

CACHE = Path(os.environ.get("JUNIPER_DATA_EQUITIES_CACHE_DIR", str(Path.home() / ".cache" / "juniper_data" / "equities")))
FACTOR = 100.0  # generator.py:79 _SHARES_OUTLIER_FACTOR


def load_payload(cik: int) -> dict:
    return json.loads((CACHE / "shares" / f"{int(cik):010d}.json").read_text())


def raw_points(data: dict) -> list[dict]:
    """Every fact across every unit key, untouched."""
    pts = []
    for unit_points in data["units"].values():
        pts.extend(unit_points)
    return pts


def dedup_latest_filed(data: dict) -> tuple[dict, dict]:
    """generator.py:916-923 verbatim: sort by (end, filed); last write per end wins."""
    best: dict[str, float] = {}
    filed_on: dict[str, str] = {}
    for unit_points in data["units"].values():
        for point in sorted(unit_points, key=lambda item: (item.get("end", ""), item.get("filed", ""))):
            if point.get("val") is not None and point.get("end"):
                best[point["end"]] = float(point["val"])
                if point.get("filed"):
                    filed_on[point["end"]] = point["filed"]
    return best, filed_on


def dedup_earliest_filed(data: dict) -> tuple[dict, dict]:
    """ALTERNATIVE: per end, keep the EARLIEST-filed fact (first publication)."""
    best: dict[str, float] = {}
    filed_on: dict[str, str] = {}
    for unit_points in data["units"].values():
        for point in sorted(unit_points, key=lambda item: (item.get("end", ""), item.get("filed", ""))):
            if point.get("val") is not None and point.get("end"):
                if point["end"] in best:
                    continue
                best[point["end"]] = float(point["val"])
                if point.get("filed"):
                    filed_on[point["end"]] = point["filed"]
    return best, filed_on


def whole_history_filter(series: pd.Series) -> pd.Series:
    """generator.py:932-934 verbatim."""
    median = float(series.median())
    if median > 0:
        series = series[(series >= median / FACTOR) & (series <= median * FACTOR)]
    return series


def to_series(best: dict) -> pd.Series:
    s = pd.Series(best)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def replica_fetch_shares(data: dict, dedup=dedup_latest_filed) -> pd.DataFrame | None:
    """Replica of _fetch_shares' post-load body (generator.py:910-948)."""
    if not data or not any(data.get("units", {}).values()):
        return None
    best, filed_on = dedup(data)
    if not best:
        return None
    series = whole_history_filter(to_series(best))
    if not len(series):
        return None
    frame = series.to_frame(name="shares")
    frame["filed"] = pd.to_datetime(pd.Series({pd.Timestamp(end): filed_on.get(end) for end in best}, dtype="object")).reindex(frame.index)
    return frame


def align_on_filed(shares: pd.DataFrame, trade_index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Replica of _condition_one lines 706-716: drop unfiled, sort (filed,end),
    dedup on filed keep last, reindex on union, ffill, reindex on trade dates."""
    known = shares.dropna(subset=["filed"])
    known = known.rename_axis("end").reset_index().sort_values(["filed", "end"]).set_index("filed")
    known = known[~known.index.duplicated(keep="last")]
    if not len(known):
        return pd.Series(float("nan"), index=trade_index), pd.Series(pd.NaT, index=trade_index)
    union = trade_index.union(known.index)
    total = known["shares"].reindex(union).sort_index().ffill().reindex(trade_index).astype("float64")
    as_of = pd.Series(known.index, index=known.index).reindex(union).sort_index().ffill().reindex(trade_index)
    return total, pd.to_datetime(as_of)


def point_in_time_series(data: dict, trade_index: pd.DatetimeIndex, apply_filter: bool = True) -> pd.Series:
    """ALTERNATIVE A2 — true as-known series: for each trade date t, among ALL
    facts with filed <= t, take the one with the latest end (tie -> latest filed).
    No dedup is done up front, so a restating filing cannot erase the original
    publication.  Optional whole-history outlier filter applied on distinct
    (end,val) values first, to isolate the dedup effect from the filter effect."""
    pts = [p for p in raw_points(data) if p.get("val") is not None and p.get("end") and p.get("filed")]
    df = pd.DataFrame({"end": pd.to_datetime([p["end"] for p in pts]), "filed": pd.to_datetime([p["filed"] for p in pts]), "val": [float(p["val"]) for p in pts]})
    if apply_filter:
        # same band as the generator, computed on the generator's own deduped series
        best, _ = dedup_latest_filed(data)
        med = float(to_series(best).median())
        if med > 0:
            df = df[(df["val"] >= med / FACTOR) & (df["val"] <= med * FACTOR)]
    df = df.sort_values(["filed", "end"]).reset_index(drop=True)
    # running "best known": walk filed ascending; a fact becomes effective on its
    # filed date only if its end >= the currently effective end (a restatement of
    # an OLDER period never displaces a newer one).
    eff_dates, eff_vals = [], []
    cur_end = pd.Timestamp.min
    for _, r in df.iterrows():
        if r["end"] >= cur_end:
            cur_end = r["end"]
            eff_dates.append(r["filed"])
            eff_vals.append(r["val"])
    eff = pd.Series(eff_vals, index=pd.DatetimeIndex(eff_dates))
    eff = eff[~eff.index.duplicated(keep="last")].sort_index()
    union = trade_index.union(eff.index)
    return eff.reindex(union).sort_index().ffill().reindex(trade_index).astype("float64")
