#!/usr/bin/env python3
"""Survey SEC shares-outstanding coverage across the bundled S&P 500 universe.

Project:     Juniper
Sub-Project: juniper-data
Application: APD-DATA-018 follow-up -- the KO shares gap
Author:      Paul Calnon
Version:     0.1.0
License:     MIT
Status:      ad-hoc (evidence for the shares-coverage finding)

`KO` returns 636 bytes and ZERO facts from
``dei/EntityCommonStockSharesOutstanding`` and 404s on the ``us-gaap`` fallback,
so ``EquitiesGenerator`` silently fills its ``total_shares`` and ``market_cap``
with 0.0 (the ``fundamentals_fill="zero"`` default). A feature column that is
zero because the data is missing -- rather than because the value is zero -- is
the same silent-partial-data class the truncation work just closed, and nothing
downstream can tell the two apart.

This answers the question that closes it: **how many of the 503 are affected,
and is a third concept tag enough to fix it?**

For each ticker it tries, in order:
  1. dei/EntityCommonStockSharesOutstanding   (what the generator tries first)
  2. us-gaap/CommonStockSharesOutstanding     (the generator's fallback)
  3. us-gaap/CommonStockSharesIssued          (candidate -- issued, not outstanding)
  4. dei/EntityCommonStockSharesOutstanding on the CIK resolved from SEC's own
     company_tickers.json rather than the bundled CSV -- to separate "SEC has no
     such concept" from "our CIK is wrong".

Deliberately gentle: SEC's published 10 req/s ceiling is respected, and the run
stops early with ``--limit`` so a full sweep is an explicit choice.

Run:
    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-04_survey_sec_shares_coverage.py --limit 60
"""

from __future__ import annotations

import argparse
import csv
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
CONCEPT = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{tax}/{tag}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
MIN_INTERVAL = 0.12

CANDIDATES = (
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesIssued"),
)

_last = [0.0]


def _get(url: str):
    wait = MIN_INTERVAL - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as response:  # noqa: S310
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return None if exc.code == 404 else {"__error__": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"__error__": type(exc).__name__}
    finally:
        _last[0] = time.monotonic()


def _facts(payload) -> int:
    """Fact count, or -1 when the concept is absent / errored."""
    if not payload or "__error__" in payload or not payload.get("units"):
        return -1
    return sum(len(rows) for rows in payload["units"].values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60, help="tickers to survey (0 = all 503)")
    args = parser.parse_args()

    path = REPO / "juniper_data/generators/equities/sp500_constituents.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    sec_map = {}
    payload = _get(TICKERS_URL)
    if payload and "__error__" not in payload:
        for entry in payload.values():
            sec_map[entry["ticker"].upper()] = int(entry["cik_str"])

    verdicts = Counter()
    gaps = []
    print(f"{'ticker':<8} {'cik':>9} {'dei':>7} {'gaap-out':>9} {'gaap-iss':>9}  verdict")
    for row in rows:
        ticker = row["ticker"].strip().upper()
        cik_raw = (row.get("cik") or "").strip()
        if not cik_raw.isdigit():
            verdicts["no CIK in bundled CSV"] += 1
            print(f"{ticker:<8} {'-':>9} {'-':>7} {'-':>9} {'-':>9}  NO CIK IN CSV")
            continue
        cik = int(cik_raw)

        counts = [_facts(_get(CONCEPT.format(cik=cik, tax=tax, tag=tag))) for tax, tag in CANDIDATES]

        if counts[0] > 0:
            verdict = "ok (dei)"
        elif counts[1] > 0:
            verdict = "ok (us-gaap fallback)"
        elif counts[2] > 0:
            verdict = "FIXABLE by CommonStockSharesIssued"
        else:
            sec_cik = sec_map.get(ticker) or sec_map.get(ticker.replace(".", "-"))
            if sec_cik and sec_cik != cik and _facts(_get(CONCEPT.format(cik=sec_cik, tax="dei", tag=CANDIDATES[0][1]))) > 0:
                verdict = f"WRONG CIK in CSV (SEC says {sec_cik})"
            else:
                verdict = "NO SHARES ANYWHERE"
            gaps.append((ticker, cik, verdict))
        verdicts[verdict.split(" (")[0].split(" by")[0]] += 1
        print(f"{ticker:<8} {cik:>9} {counts[0]:>7} {counts[1]:>9} {counts[2]:>9}  {verdict}")

    print()
    print("=" * 72)
    print(f"surveyed {len(rows)} tickers")
    for verdict, count in verdicts.most_common():
        print(f"  {count:>4}  {verdict}")
    if gaps:
        print()
        print("Tickers with NO usable shares series (silently zero-filled today):")
        for ticker, cik, verdict in gaps:
            print(f"  {ticker:<8} cik={cik:<9} {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
