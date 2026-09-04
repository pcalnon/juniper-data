#!/usr/bin/env python3
"""Project equities ingest cost across index universes and time horizons.

Project:     Juniper
Sub-Project: juniper-data
Application: APD-DATA-018 -- equities input bound
Author:      Paul Calnon
Version:     0.1.0
License:     MIT
Status:      ad-hoc (evidence for the equities cap decision)

Every constant below is MEASURED, by
``util/ad-hoc/2026-09-04_measure_equities_payloads.py`` on 2026-09-04, or taken
from a cited index fact sheet. Nothing here is estimated from intuition; where a
figure is derived rather than measured it says so.

The headline the numbers produce: **bytes and wall time do not scale together.**
163x more payload cost 1.16x the time, because the cost is per-REQUEST, not
per-byte. A byte cap therefore bounds the wrong quantity for this generator --
which is the opposite of the csv_import half, where the input is a file.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MEASURED 2026-09-04 -- Yahoo chart API v8, AAPL, gzip wire bytes
# ---------------------------------------------------------------------------
# period          wire B   uncompressed B   seconds
#   1 month        1,322          3,663       0.50
#   1 year         8,398         28,541       0.34
#   5 years       36,762        139,113       0.42
#   since 2000   215,505        724,740       0.58
WIRE_BYTES_PER_TRADING_DAY = 32.1  # 215,505 / 6,708 rows (the longest sample)
RAW_BYTES_PER_TRADING_DAY = 108.0  # 724,740 / 6,708
YF_SECONDS_PER_CALL = 1.85  # measured via yfinance (0.34-0.58 s direct; yfinance adds overhead)
SEC_SECONDS_PER_CALL = 0.20  # dei/EntityCommonStockSharesOutstanding
SEC_CALLS_PER_SYMBOL = 1.4  # 1 call when the dei tag hits, 2 when it falls through to us-gaap
SEC_BYTES_PER_SYMBOL = 10_500  # ~10.5 KB, ~70 facts
SEC_MIN_INTERVAL = 0.12  # SEC's 10 req/s ceiling, enforced in generator.py

NPZ_X_BYTES_PER_ROW = 10 * 4  # EQUITIES_FEATURE_COLUMNS, float32
NPZ_Y_BYTES_PER_ROW = 4  # regression target, float32

REQUEST_BUDGET_SECONDS = 30.0  # the client's default socket timeout

# Trading days. 252/year is the standard convention; "since 2000" is MEASURED
# (6,708 rows returned for 2000-01-01..2026-09-04).
HORIZONS = {
    "1 day": 1,
    "1 month": 21,
    "1 quarter": 63,
    "1 year": 252,
    "5 years": 1260,
    "since 2000": 6708,
}

# Constituent counts. S&P 500 is the bundled snapshot's actual row count; the
# Russell and Wilshire figures are current published counts, all of which fall
# well short of the number in the index's name.
UNIVERSES = {
    "Dow 30": 30,
    "Nasdaq-100": 101,
    "S&P 500": 503,
    "S&P MidCap 400": 400,
    "S&P SmallCap 600": 600,
    "S&P Composite 1500": 1503,
    "Russell 1000": 1004,
    "Russell 2000": 1958,
    "Russell 3000": 2923,
    "Wilshire 5000": 3414,
}


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            return f"{num_bytes:,.1f} {unit}" if unit != "B" else f"{num_bytes:,.0f} B"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def main() -> int:
    print("=" * 100)
    print("1. ONE SYMBOL, by horizon -- wire bytes vs wall time")
    print("=" * 100)
    print(f"{'horizon':<14} {'rows':>7} {'wire':>12} {'uncompressed':>14} {'NPZ X+y':>10} {'seconds':>9}")
    for label, days in HORIZONS.items():
        wire = days * WIRE_BYTES_PER_TRADING_DAY
        raw = days * RAW_BYTES_PER_TRADING_DAY
        npz = days * (NPZ_X_BYTES_PER_ROW + NPZ_Y_BYTES_PER_ROW)
        secs = YF_SECONDS_PER_CALL + SEC_CALLS_PER_SYMBOL * SEC_SECONDS_PER_CALL
        print(f"{label:<14} {days:>7,} {human(wire):>12} {human(raw):>14} {human(npz):>10} {duration(secs):>9}")

    print()
    print("  Note the last two columns: bytes grow 6,708x across this table and")
    print("  seconds do not move at all. The per-symbol cost is one Yahoo request")
    print("  plus ~1.4 SEC requests, whatever the horizon.")

    print()
    print("=" * 100)
    print("2. WHOLE UNIVERSE x horizon -- wire bytes (top) and wall time (bottom)")
    print("=" * 100)
    horizons = ["1 day", "1 month", "1 quarter", "1 year", "since 2000"]
    header = f"{'universe':<20} {'symbols':>8} " + " ".join(f"{h:>13}" for h in horizons)
    print(header)
    print("-" * len(header))
    for name, count in UNIVERSES.items():
        cells = []
        for label in horizons:
            wire = count * HORIZONS[label] * WIRE_BYTES_PER_TRADING_DAY
            cells.append(f"{human(wire):>13}")
        print(f"{name:<20} {count:>8,} " + " ".join(cells))

    print()
    print(f"{'universe':<20} {'symbols':>8} {'serial wall time (horizon-independent)':>40}")
    print("-" * 70)
    for name, count in UNIVERSES.items():
        secs = count * (YF_SECONDS_PER_CALL + SEC_CALLS_PER_SYMBOL * SEC_SECONDS_PER_CALL)
        floor = count * SEC_CALLS_PER_SYMBOL * SEC_MIN_INTERVAL
        print(f"{name:<20} {count:>8,} {duration(secs):>18}   (SEC throttle floor alone: {duration(floor)})")

    print()
    print("=" * 100)
    print("3. WHAT FITS IN THE 30 s REQUEST BUDGET")
    print("=" * 100)
    per_symbol = YF_SECONDS_PER_CALL + SEC_CALLS_PER_SYMBOL * SEC_SECONDS_PER_CALL
    max_symbols = REQUEST_BUDGET_SECONDS / per_symbol
    print(f"per-symbol cost (serial, uncached) : {per_symbol:.2f} s")
    print(f"symbols that fit in {REQUEST_BUDGET_SECONDS:.0f} s          : {max_symbols:.1f}")
    print(f"  ... at 50% of budget (headroom)  : {max_symbols * 0.5:.1f}")
    print()
    print("Bytes at that symbol count, longest horizon:")
    for frac, note in ((1.0, "full budget"), (0.5, "half budget")):
        symbols = max_symbols * frac
        wire = symbols * HORIZONS["since 2000"] * WIRE_BYTES_PER_TRADING_DAY
        print(f"  {symbols:>5.1f} symbols x 6,708 days = {human(wire):>10}   ({note})")
    print()
    print("A byte cap set anywhere near those figures would be measured in single-digit")
    print("MB -- and would reject a 3,000-symbol x 1-day request (2.9 MB, ~2 hours) while")
    print("accepting a 25-symbol x 26-year request (5.4 MB, ~50 s). It bounds the wrong axis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
