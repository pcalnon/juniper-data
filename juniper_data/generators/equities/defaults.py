"""Default constants for the equities time-series dataset generator."""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     defaults.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from typing import Literal

# Date range: daily OHLCV from Yahoo Finance reaches back to 2000 for most
# US-listed names (verified). ``None`` end date resolves to "today" (UTC).
EQUITIES_DEFAULT_START_DATE = "2000-01-01"
EQUITIES_DEFAULT_END_DATE: str | None = None

# Cost basis is defined as the price on a configurable purchase date, per
# ticker, clamped forward to the first available trading day if the requested
# date precedes the listing.
EQUITIES_DEFAULT_PURCHASE_DATE = "2000-01-03"
EQUITIES_DEFAULT_BASIS_PRICE_FIELD: Literal["close", "adj_close"] = "close"

# Shares outstanding / market cap are sourced from SEC EDGAR XBRL filings,
# which reach back to ~2009. Rows before the first filing have no value; the
# fill strategy decides how the X matrix represents that gap.
#   "zero" -> fill missing total_shares / market_cap with 0.0 (X stays trainable)
#   "nan"  -> leave NaN (honest; caller must handle)
#   "drop" -> drop rows lacking shares (restricts the series to ~2009+)
#
# DEFAULT CHANGED "zero" -> "nan" (2026-09-05, owner decision). A market cap of
# 0.0 is a value no listed company can have, and nothing downstream can tell it
# apart from a measurement: it survives every dtype, range and NaN check, and a
# model trained on it learns from a number that was fabricated by a fill policy.
# NaN propagates visibly instead -- pandas and numpy consumers see it, and a
# consumer that wants the old behaviour asks for it explicitly.
#
# This is a BEHAVIOUR CHANGE for callers who relied on the default: rows before a
# ticker's first SEC filing (~2009) now carry NaN in total_shares / market_cap
# rather than 0.0. "drop" remains available for a caller who wants those rows
# gone instead, and it changes the universe size, which is why it is not the
# default either.
EQUITIES_DEFAULT_FUNDAMENTALS_FILL: Literal["zero", "nan", "drop"] = "nan"

# 52-week high/low rolling window, in trading sessions (~252 per year).
EQUITIES_DEFAULT_WEEK52_WINDOW = 252

EQUITIES_DEFAULT_NORMALIZE_FEATURES = False
EQUITIES_DEFAULT_USE_CACHE = True

# APD-DATA-018 -- re-exported, not defined here (see juniper_data/core/limits.py
# for the value's derivation and for why it cannot live in this module).
#
# This was ``None`` -- every one of the 503 bundled S&P 500 constituents -- which
# measurement put at 18-34 minutes against a 30 s request budget. The cap is in
# SYMBOLS rather than bytes because a cap's unit has to be measurable BEFORE the
# work: equities has no input to weigh, so its byte count does not exist until
# the fan-out the cap exists to bound has already run. (Corrected 2026-09-05:
# this said "bytes would bound the wrong quantity", citing an inverted byte
# comparison. Bytes here are in fact positively correlated with cost; what rules
# a byte cap out is that it is unmeasurable ex ante, not its direction.)
from juniper_data.core.limits import (  # noqa: E402  (re-export, kept beside the other defaults)
    EQUITIES_DEFAULT_ALLOW_TRUNCATION,
    EQUITIES_DEFAULT_MAX_SYMBOLS,
)

__all__ = [
    "CONSTITUENTS_FILENAME",
    "EQUITIES_DEFAULT_ALLOW_TRUNCATION",
    "EQUITIES_DEFAULT_BASIS_PRICE_FIELD",
    "EQUITIES_DEFAULT_END_DATE",
    "EQUITIES_DEFAULT_FUNDAMENTALS_FILL",
    "EQUITIES_DEFAULT_MAX_SYMBOLS",
    "EQUITIES_DEFAULT_NORMALIZE_FEATURES",
    "EQUITIES_DEFAULT_PURCHASE_DATE",
    "EQUITIES_DEFAULT_REGRESSION_TARGET",
    "EQUITIES_DEFAULT_START_DATE",
    "EQUITIES_DEFAULT_TEST_RATIO",
    "EQUITIES_DEFAULT_VAL_RATIO",
    "EQUITIES_DEFAULT_TRAIN_RATIO",
    "EQUITIES_DEFAULT_USE_CACHE",
    "EQUITIES_DEFAULT_WEEK52_WINDOW",
    "EQUITIES_FEATURE_COLUMNS",
]

# Temporal split: train = earlier dates, test = later dates (per ticker).
EQUITIES_DEFAULT_TRAIN_RATIO = 0.8
# The three-way default is 0.8 / 0.1 / 0.1: test halves to make room for the
# in-loop partition rather than train shrinking, because every existing
# baseline is measured against the train count.
EQUITIES_DEFAULT_VAL_RATIO = 0.1
EQUITIES_DEFAULT_TEST_RATIO = 0.1

# Regression-target (y_reg) representation. The raw next-day close is
# non-stationary (it trends with the price level), which a bounded-memory
# recurrent regressor extrapolates badly; the return variants are stationary
# and are the standard conditioning for forecasting on trending price data.
#   "next_close" -> raw next-day close price (default; back-compatible)
#   "return"     -> simple next-day return: next_close / close - 1
#   "log_return" -> log next-day return: ln(next_close / close)
EQUITIES_DEFAULT_REGRESSION_TARGET: Literal["next_close", "return", "log_return"] = "next_close"

# Bundled snapshot of current S&P 500 constituents (ticker,name,cik,sector),
# used as the default universe and the ticker -> (name, CIK) map.
CONSTITUENTS_FILENAME = "sp500_constituents.csv"

# Ordered numeric columns that form the X feature matrix (all float32).
# Ordered numeric columns that form the X feature matrix (all float32).
#
# ORDER IS PART OF THE CONTRACT. Existing columns keep their positions so a
# consumer indexing by position (X[:, 3] is close) is unaffected; the six added
# 2026-09-04 are appended, never interleaved.
#
# All six were already being downloaded and thrown away -- none costs an extra
# request:
#   adj_close               already parsed out of the response, then dropped
#   dividend, split_ratio   arrive on the same call once actions=True is set
#   days_since_week52_*     fall out of the rolling window already computed
#   days_since_report       the `filed` date already in the SEC shares payload
#
# The three underlying DATES ship separately as row-aligned YYYYMMDD arrays
# (week52_high_date_*, week52_low_date_*, report_date_*) rather than as feature
# columns, because a raw date in a float32 matrix is a number whose magnitude
# carries no meaning. "Days since" is the form a model can use.
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
    "adj_close",
    "dividend",
    "split_ratio",
    "days_since_week52_high",
    "days_since_week52_low",
    "days_since_report",
]
