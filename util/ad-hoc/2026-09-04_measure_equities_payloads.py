#!/usr/bin/env python3
"""Measure equities ingest cost: wire bytes vs wall time, per symbol and per period.

Project:     Juniper
Sub-Project: juniper-data
Application: APD-DATA-018 -- equities input bound
Author:      Paul Calnon
Version:     0.1.0
License:     MIT
Status:      ad-hoc (evidence for the equities cap decision)

The csv_import half of APD-DATA-018 took a BYTE cap because its input is a file:
bytes are the thing the operator can bound without parsing. The equities half
has no input file at all -- its input is an API fan-out over (symbols x date
range), and its cost may or may not track bytes.

This measures both, separately, so the cap unit is chosen from evidence:

* **Wire bytes** for the two upstreams the generator actually uses --
  Yahoo Finance daily OHLCV (via yfinance) and SEC EDGAR XBRL companyconcept.
* **Wall time** for the same calls.
* **Derived artifact bytes**, which is what the NPZ ends up holding.

If bytes and time scale together, a byte cap transfers cleanly from csv_import.
If time is dominated by per-request overhead that is independent of payload
size, a byte cap bounds the wrong quantity and the honest unit is symbols (or
symbol-days).

Deliberately gentle on both upstreams: a handful of tickers, SEC's 10 req/s
limit respected via the generator's own interval constant.

Run:
    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-04_measure_equities_payloads.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SEC_UA = {"User-Agent": "juniper-data (Juniper ML research; overtoad.research@gmail.com)"}
SEC_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/CommonStockSharesOutstanding.json"
SEC_MIN_INTERVAL = 0.12

# One large-cap, one mid-cap, one with a long history -- enough to see whether
# per-symbol cost varies with the name or is dominated by fixed overhead.
TICKERS = ["AAPL", "KO", "JNJ"]
CIKS = {"AAPL": 320193, "KO": 21344, "JNJ": 200406}

TODAY = datetime.now(UTC).date()
PERIODS = {
    "1 day": TODAY - timedelta(days=4),
    "1 month": TODAY - timedelta(days=30),
    "1 quarter": TODAY - timedelta(days=91),
    "1 year": TODAY - timedelta(days=365),
    "5 years": TODAY - timedelta(days=365 * 5),
    "since 2000": date(2000, 1, 1),
}


def measure_yf(ticker: str, start: date, end: date) -> tuple[float, int, int]:
    """Return (seconds, rows, in-memory bytes) for one yfinance daily download."""
    import yfinance as yf

    begin = time.monotonic()
    frame = yf.download(ticker.replace(".", "-"), start=start.isoformat(), end=end.isoformat(), interval="1d", auto_adjust=False, progress=False, threads=False)
    elapsed = time.monotonic() - begin
    if frame is None or len(frame) == 0:
        return elapsed, 0, 0
    return elapsed, len(frame), int(frame.memory_usage(deep=True).sum())


def measure_sec(cik: int) -> tuple[float, int, int]:
    """Return (seconds, wire bytes, fact count) for one SEC companyconcept call."""
    url = SEC_CONCEPT_URL.format(cik=cik)
    request = urllib.request.Request(url, headers=SEC_UA)
    begin = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            raw = response.read()
    except Exception as exc:  # noqa: BLE001
        print(f"    SEC {cik}: {type(exc).__name__}: {exc}")
        return time.monotonic() - begin, 0, 0
    elapsed = time.monotonic() - begin
    facts = 0
    try:
        payload = json.loads(raw)
        for unit_rows in payload.get("units", {}).values():
            facts += len(unit_rows)
    except json.JSONDecodeError:
        # The byte count and timing are still valid measurements -- they are what
        # this function exists to report -- but a non-JSON body means the fact
        # count is unknowable, and returning 0 silently would be reported as
        # "an empty concept" rather than "a body we could not parse". Say so.
        print(f"    CIK {cik}: response was not JSON ({len(raw)} bytes); fact count unavailable")
        return elapsed, len(raw), -1
    return elapsed, len(raw), facts


def main() -> int:
    print("=" * 78)
    print("A. Yahoo Finance daily OHLCV -- one symbol, by period")
    print("=" * 78)
    print(f"{'period':<12} {'rows':>6} {'mem bytes':>11} {'B/row':>7} {'seconds':>8}")

    per_row_samples: list[float] = []
    fixed_cost_samples: list[float] = []
    for label, start in PERIODS.items():
        elapsed, rows, mem = measure_yf(TICKERS[0], start, TODAY)
        per_row = (mem / rows) if rows else 0
        if rows:
            per_row_samples.append(per_row)
        fixed_cost_samples.append(elapsed)
        print(f"{label:<12} {rows:>6} {mem:>11} {per_row:>7.1f} {elapsed:>8.2f}")
        time.sleep(1.0)

    print()
    print("=" * 78)
    print("B. Yahoo Finance -- fixed per-CALL cost across symbols (1 year each)")
    print("=" * 78)
    print(f"{'ticker':<8} {'rows':>6} {'mem bytes':>11} {'seconds':>8}")
    call_times: list[float] = []
    for ticker in TICKERS:
        elapsed, rows, mem = measure_yf(ticker, PERIODS["1 year"], TODAY)
        call_times.append(elapsed)
        print(f"{ticker:<8} {rows:>6} {mem:>11} {elapsed:>8.2f}")
        time.sleep(1.0)

    print()
    print("=" * 78)
    print("C. SEC EDGAR XBRL companyconcept -- shares outstanding")
    print("=" * 78)
    print(f"{'ticker':<8} {'wire bytes':>11} {'facts':>7} {'seconds':>8}")
    sec_bytes: list[int] = []
    sec_times: list[float] = []
    for ticker in TICKERS:
        elapsed, wire, facts = measure_sec(CIKS[ticker])
        if wire:
            sec_bytes.append(wire)
            sec_times.append(elapsed)
        print(f"{ticker:<8} {wire:>11} {facts:>7} {elapsed:>8.2f}")
        time.sleep(SEC_MIN_INTERVAL)

    print()
    print("=" * 78)
    print("D. Derived figures")
    print("=" * 78)
    if per_row_samples:
        print(f"OHLCV in-memory bytes/row (pandas)   : {statistics.median(per_row_samples):.1f}")
    if call_times:
        print(f"yfinance wall time per CALL (1y)     : median {statistics.median(call_times):.2f} s")
    if sec_bytes:
        print(f"SEC concept wire bytes               : median {statistics.median(sec_bytes):,}")
        print(f"SEC concept wall time                : median {statistics.median(sec_times):.2f} s")

    # The artifact side is analytic, not measured: EQUITIES_FEATURE_COLUMNS is a
    # fixed-width float32 matrix, so a row is exactly len(cols) * 4 bytes plus
    # the label column(s).
    from juniper_data.generators.equities.defaults import EQUITIES_FEATURE_COLUMNS

    x_bytes = len(EQUITIES_FEATURE_COLUMNS) * 4
    print(f"NPZ X bytes/row ({len(EQUITIES_FEATURE_COLUMNS)} float32 cols) : {x_bytes} (+ label)")
    print()
    print("Interpretation to check: if B's per-call seconds are ~flat across")
    print("symbols while A's bytes grow with the period, then TIME is fixed")
    print("per-request overhead and BYTES is the wrong unit for this cap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
