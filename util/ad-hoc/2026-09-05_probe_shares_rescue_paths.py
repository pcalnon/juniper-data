#!/usr/bin/env python3
"""Probe rescue paths for the 37 tickers with no shares-outstanding concept.

Project:     Juniper
Sub-Project: juniper-data
Application: shares-coverage rescue
Author:      Paul Calnon
Version:     0.1.0
License:     MIT
Status:      ad-hoc (evidence for the rescue-ladder design)

The 2026-09-04 census found 37 of 503 S&P 500 constituents returning NOTHING
from either shares concept, so their ``total_shares`` and ``market_cap`` were
zero-filled. The owner's direction (2026-09-05) is that a zero-fill is only
acceptable as a WARNING when the value can be rescued -- by an alternate source,
modified parsing, or an alternate identifier -- and must otherwise take an
explicit fail-or-continue path.

So: how many can actually be rescued, and by what?

**The mechanism, established for META.** ``companyconcept`` and ``companyfacts``
return only facts with NO dimensional qualifiers. A multi-class filer tags shares
outstanding *per share class* (``us-gaap:StatementClassOfStockAxis``), so those
facts carry a dimension and are invisible to both endpoints -- the concept comes
back present-but-empty, or absent entirely. It is a **parsing/endpoint gap, not
missing data**.

This walks a ladder of increasingly weak substitutes and reports, per ticker,
the first rung that has facts:

  1. dei/EntityCommonStockSharesOutstanding      point-in-time, cover page  (best)
  2. us-gaap/CommonStockSharesOutstanding        point-in-time
  3. us-gaap/CommonStockSharesIssued             issued != outstanding (incl. treasury)
  4. us-gaap/WeightedAverageNumberOfSharesOutstandingBasic    period AVERAGE
  5. us-gaap/WeightedAverageNumberOfDilutedSharesOutstanding  period average, diluted

Rungs 3-5 are semantically weaker than 1-2 and must be recorded as such wherever
they are used -- a market cap computed from a period average is not the same
quantity as one computed from point-in-time shares.

Uses ``companyfacts`` (ONE request per ticker) rather than five ``companyconcept``
calls, which is both faster and gentler on SEC.

Run:
    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-05_probe_shares_rescue_paths.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

UA = {"User-Agent": "juniper-data (Juniper ML research; overtoad.research@gmail.com)"}
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MIN_INTERVAL = 0.12

LADDER = (
    ("dei", "EntityCommonStockSharesOutstanding", "point-in-time (cover page)"),
    ("us-gaap", "CommonStockSharesOutstanding", "point-in-time"),
    ("us-gaap", "CommonStockSharesIssued", "ISSUED (includes treasury)"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", "period AVERAGE (basic)"),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "period AVERAGE (diluted)"),
)

# The 37 with no usable series, from the 2026-09-04 census.
GAPS = [
    ("ABNB", 1559720), ("ABT", 1800), ("AMT", 1053507), ("BG", 1996862), ("GRMN", 1121788),
    ("HBAN", 49196), ("HCA", 860730), ("HRL", 48465), ("HUM", 49071), ("INCY", 879169),
    ("IQV", 1478242), ("KO", 21344), ("LEN", 920760), ("LII", 1069202), ("MAS", 62996),
    ("META", 1326801), ("MKC", 63754), ("PFG", 1126328), ("POOL", 945841), ("RL", 1037038),
    ("SPGI", 64040), ("STZ", 16918), ("TKO", 1973266), ("TTD", 1671933), ("UHS", 352915),
    ("VLTO", 1967680), ("VRSK", 1442145), ("WELL", 766704), ("XYZ", 1512673),
]

_last = [0.0]


def _get(url: str):
    wait = MIN_INTERVAL - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as response:  # noqa: S310
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return {"__error__": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"__error__": type(exc).__name__}
    finally:
        _last[0] = time.monotonic()


def main() -> int:
    print(f"{'ticker':<8} {'rung':>5}  {'facts':>6}  concept / verdict")
    print("-" * 92)
    outcomes = Counter()
    for ticker, cik in GAPS:
        payload = _get(FACTS_URL.format(cik=cik))
        if "__error__" in payload:
            print(f"{ticker:<8} {'-':>5}  {'-':>6}  companyfacts {payload['__error__']}")
            outcomes["companyfacts error"] += 1
            continue

        facts = payload.get("facts", {})
        for rung, (tax, tag, meaning) in enumerate(LADDER, start=1):
            entry = facts.get(tax, {}).get(tag)
            count = sum(len(rows) for rows in entry.get("units", {}).values()) if entry else 0
            if count:
                print(f"{ticker:<8} {rung:>5}  {count:>6}  {tax}/{tag} -- {meaning}")
                outcomes[f"rung {rung}: {meaning}"] += 1
                break
        else:
            available = sorted({tag for tax_tags in facts.values() for tag in tax_tags if "SharesOutstanding" in tag})
            print(f"{ticker:<8} {'none':>5}  {0:>6}  NO RESCUE. share-ish tags present: {available[:3] or 'none'}")
            outcomes["no rescue"] += 1

    print()
    print("=" * 92)
    for outcome, count in outcomes.most_common():
        print(f"  {count:>3}  {outcome}")
    print()
    print("Rungs 3-5 are NOT equivalent to 1-2: issued includes treasury shares, and a period")
    print("average is not point-in-time. Any rescue from those rungs has to say so in the")
    print("dataset's own metadata, or market_cap quietly means two different things.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
