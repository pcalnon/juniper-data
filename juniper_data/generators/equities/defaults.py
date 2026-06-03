"""Default constants for the equities time-series dataset generator."""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     defaults.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

# Date range: daily OHLCV from Yahoo Finance reaches back to 2000 for most
# US-listed names (verified). ``None`` end date resolves to "today" (UTC).
EQUITIES_DEFAULT_START_DATE = "2000-01-01"
EQUITIES_DEFAULT_END_DATE: str | None = None

# Cost basis is defined as the price on a configurable purchase date, per
# ticker, clamped forward to the first available trading day if the requested
# date precedes the listing.
EQUITIES_DEFAULT_PURCHASE_DATE = "2000-01-03"
EQUITIES_DEFAULT_BASIS_PRICE_FIELD = "close"

# Shares outstanding / market cap are sourced from SEC EDGAR XBRL filings,
# which reach back to ~2009. Rows before the first filing have no value; the
# fill strategy decides how the X matrix represents that gap.
#   "zero" -> fill missing total_shares / market_cap with 0.0 (X stays trainable)
#   "nan"  -> leave NaN (honest; caller must handle)
#   "drop" -> drop rows lacking shares (restricts the series to ~2009+)
EQUITIES_DEFAULT_FUNDAMENTALS_FILL = "zero"

# 52-week high/low rolling window, in trading sessions (~252 per year).
EQUITIES_DEFAULT_WEEK52_WINDOW = 252

EQUITIES_DEFAULT_NORMALIZE_FEATURES = False
EQUITIES_DEFAULT_MAX_SYMBOLS: int | None = None
EQUITIES_DEFAULT_USE_CACHE = True

# Temporal split: train = earlier dates, test = later dates (per ticker).
EQUITIES_DEFAULT_TRAIN_RATIO = 0.8
EQUITIES_DEFAULT_TEST_RATIO = 0.2

# Bundled snapshot of current S&P 500 constituents (ticker,name,cik,sector),
# used as the default universe and the ticker -> (name, CIK) map.
CONSTITUENTS_FILENAME = "sp500_constituents.csv"

# Ordered numeric columns that form the X feature matrix (all float32).
EQUITIES_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "week52_high",
    "week52_low",
    "total_shares",
    "market_cap",
    "cost_basis",
]
