"""Which duplicate-``filed`` collisions does the shipped sort resolve WRONG?

Project:       Juniper
Sub-Project:   JuniperData
Application:   juniper_data
File Name:     2026-09-05_probe_duplicate_filed_resolution.py
Author:        Paul Calnon
License:       MIT License
Status:        ad-hoc, single-use

``generator.py`` aligns SEC share counts on the FILING date (the look-ahead fix).
Two facts can share one ``filed`` -- an 8-K restating an old period lands the same
day as the current 10-Q -- so the series is de-duplicated with::

    known = known.set_index("filed").sort_index()
    known = known[~known.index.duplicated(keep="last")]

Rows arrive ``end``-ascending (``_fetch_shares`` sorts on ``end``), so ``keep="last"``
is CORRECT only if the re-sort on ``filed`` preserves that incoming order among ties.
``sort_index()`` defaults to ``kind="quicksort"``, which is NOT stable. Whether the
current period or the restated old one survives is therefore arbitrary.

This reads the on-disk SEC cache only -- no network -- and reports, per ticker, every
duplicate-``filed`` collision plus which value quicksort keeps versus a stable sort.

Run::

    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-05_probe_duplicate_filed_resolution.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CACHE = Path.home() / ".cache" / "juniper_data" / "equities" / "shares"


def points(payload: dict) -> pd.DataFrame:
    """Rebuild the frame ``_fetch_shares`` returns: indexed by ``end``, ascending."""
    best: dict[str, float] = {}
    filed_on: dict[str, str] = {}
    for unit_points in payload.get("units", {}).values():
        for point in sorted(unit_points, key=lambda item: (item.get("end", ""), item.get("filed", ""))):
            if point.get("val") is not None and point.get("end"):
                best[point["end"]] = float(point["val"])
                if point.get("filed"):
                    filed_on[point["end"]] = point["filed"]
    if not best:
        return pd.DataFrame()
    series = pd.Series(best)
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    median = float(series.median())
    if median > 0:
        series = series[(series >= median / 1e6) & (series <= median * 1e6)]
    frame = series.to_frame(name="shares")
    frame["filed"] = pd.to_datetime(pd.Series({pd.Timestamp(end): filed_on.get(end) for end in best}, dtype="object")).reindex(frame.index)
    return frame


def resolve(frame: pd.DataFrame, kind: str) -> pd.Series:
    """The shipped de-duplication, parameterised on sort stability."""
    known = frame.dropna(subset=["filed"]).reset_index(names="end")
    known = known.set_index("filed").sort_index(kind=kind)
    known = known[~known.index.duplicated(keep="last")]
    return known


def main() -> None:
    files = sorted(CACHE.glob("*.json"))
    print(f"cache: {CACHE}  ({len(files)} payloads)\n")
    wrong: list[tuple[str, pd.Timestamp, float, float, float]] = []
    tickers_with_collisions = 0

    for path in files:
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        frame = points(payload)
        if frame.empty or "filed" not in frame:
            continue
        known = frame.dropna(subset=["filed"])
        if not known["filed"].duplicated().any():
            continue
        tickers_with_collisions += 1

        quick = resolve(frame, "quicksort")
        stable = resolve(frame, "stable")
        diff = quick.index[quick["shares"].values != stable["shares"].values]
        entity = payload.get("entityName", path.stem)
        for filed in diff:
            q = float(quick.loc[filed, "shares"])
            s = float(stable.loc[filed, "shares"])
            wrong.append((entity, filed, q, s, abs(q - s) / s * 100.0))

    print(f"tickers with >=1 duplicate-filed collision : {tickers_with_collisions}")
    print(f"collisions resolved DIFFERENTLY by quicksort vs stable : {len(wrong)}")
    print(f"distinct tickers affected : {len({row[0] for row in wrong})}\n")
    for entity, filed, q, s, pct in sorted(wrong, key=lambda row: -row[4]):
        print(f"  {entity[:34]:34s} filed={filed.date()}  quicksort={q:>18,.0f}  stable={s:>18,.0f}  err={pct:6.3f}%")


if __name__ == "__main__":
    main()
