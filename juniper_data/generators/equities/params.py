"""Parameters for the equities time-series dataset generator."""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     params.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from juniper_data.core.constants import DEFAULT_GENERATOR_SEED

from .defaults import (
    EQUITIES_DEFAULT_ALLOW_TRUNCATION,
    EQUITIES_DEFAULT_BASIS_PRICE_FIELD,
    EQUITIES_DEFAULT_END_DATE,
    EQUITIES_DEFAULT_FUNDAMENTALS_FILL,
    EQUITIES_DEFAULT_MAX_SYMBOLS,
    EQUITIES_DEFAULT_NORMALIZE_FEATURES,
    EQUITIES_DEFAULT_PURCHASE_DATE,
    EQUITIES_DEFAULT_REGRESSION_TARGET,
    EQUITIES_DEFAULT_START_DATE,
    EQUITIES_DEFAULT_TEST_RATIO,
    EQUITIES_DEFAULT_TRAIN_RATIO,
    EQUITIES_DEFAULT_USE_CACHE,
    EQUITIES_DEFAULT_VAL_RATIO,
    EQUITIES_DEFAULT_WEEK52_WINDOW,
)

_DATE_FORMAT = "%Y-%m-%d"


def _validate_date(label: str, value: str) -> None:
    """Raise ``ValueError`` if ``value`` is not an ISO ``YYYY-MM-DD`` date."""
    try:
        datetime.strptime(value, _DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


class EquitiesParams(BaseModel):
    """Configuration parameters for the equities time-series generator.

    Downloads and conditions daily S&P 500 equities data into the JuniperData
    NPZ contract: a 10-column numeric feature matrix, a one-hot next-day
    direction label, and a configurable next-day regression target (raw
    close, simple return, or log return -- see ``regression_target``).
    """

    model_config = ConfigDict(populate_by_name=True)

    symbols: list[str] | None = Field(
        default=None,
        description="Tickers to include. None = the bundled S&P 500 constituents.",
    )
    start_date: str = Field(
        default=EQUITIES_DEFAULT_START_DATE,
        description="Inclusive start date (YYYY-MM-DD) for the price history.",
    )
    end_date: str | None = Field(
        default=EQUITIES_DEFAULT_END_DATE,
        description="Exclusive end date (YYYY-MM-DD). None = today (UTC).",
    )
    purchase_date: str = Field(
        default=EQUITIES_DEFAULT_PURCHASE_DATE,
        description="Cost-basis purchase date (YYYY-MM-DD); per-ticker clamped to the first available trading day.",
    )
    basis_price_field: Literal["close", "adj_close"] = Field(
        default=EQUITIES_DEFAULT_BASIS_PRICE_FIELD,
        description="Price field used for cost basis on the purchase date.",
    )
    fundamentals_fill: Literal["zero", "nan", "drop"] = Field(
        default=EQUITIES_DEFAULT_FUNDAMENTALS_FILL,
        description="How to represent pre-2009 missing total_shares / market_cap: zero-fill, leave NaN, or drop rows.",
    )
    regression_target: Literal["next_close", "return", "log_return"] = Field(
        default=EQUITIES_DEFAULT_REGRESSION_TARGET,
        description="Representation of the y_reg target: raw next-day close, simple return (next_close/close - 1), or log return ln(next_close/close). The return variants are stationary; the raw close is not.",
    )
    week52_window: int = Field(
        default=EQUITIES_DEFAULT_WEEK52_WINDOW,
        ge=2,
        le=2520,
        description="Rolling window (trading sessions) for 52-week high/low.",
    )
    normalize_features: bool = Field(
        default=EQUITIES_DEFAULT_NORMALIZE_FEATURES,
        description="Min-max normalize each feature column to [0, 1] (fit on the full set).",
    )
    max_symbols: int | None = Field(
        default=EQUITIES_DEFAULT_MAX_SYMBOLS,
        ge=1,
        description="Cap on the number of symbols (after ordering), APD-DATA-018. A universe larger than this is REFUSED unless allow_truncation is set. Omit to use the deployment default (JUNIPER_DATA_EQUITIES_MAX_SYMBOLS); None means unbounded and is honoured only up to that deployment ceiling.",
    )
    incomplete_rows: Literal["accept", "drop"] | None = Field(
        default=None,
        description="What to do with rows whose fundamentals no rescue path could resolve, once allow_truncation has opened the gate: 'accept' keeps them (filled per fundamentals_fill) or 'drop' excludes those symbols entirely. Either way the dataset is PERMANENTLY annotated in DatasetMeta.data_quality. None inherits the deployment default (JUNIPER_DATA_EQUITIES_INCOMPLETE_ROWS). Without allow_truncation this has no effect -- the request is refused instead.",
    )
    allow_truncation: bool = Field(
        default=EQUITIES_DEFAULT_ALLOW_TRUNCATION,
        description="Accept a partial universe when it exceeds max_symbols. Default false: an oversized universe is refused with 422 rather than silently truncated to the first N tickers. When true, the leading max_symbols symbols are imported and the dataset is PERMANENTLY annotated as truncated in its metadata. Can also be enabled deployment-wide via JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION or the matching .env entry.",
    )
    use_cache: bool = Field(
        default=EQUITIES_DEFAULT_USE_CACHE,
        description="Cache raw downloads under ~/.cache/juniper_data/equities for fast re-runs.",
    )
    train_ratio: float = Field(
        default=EQUITIES_DEFAULT_TRAIN_RATIO,
        gt=0,
        le=1,
        description="Fraction of each ticker's earliest rows used for training.",
    )
    val_ratio: float = Field(
        default=EQUITIES_DEFAULT_VAL_RATIO,
        ge=0,
        le=1,
        description="Fraction of each ticker's rows used for in-loop validation, taken from the rows immediately after train and before test.",
    )
    test_ratio: float = Field(
        default=EQUITIES_DEFAULT_TEST_RATIO,
        ge=0,
        le=1,
        description="Fraction of each ticker's latest rows used for testing.",
    )
    seed: int | None = Field(
        default=DEFAULT_GENERATOR_SEED,
        ge=0,
        description=(
            "Unused for the temporal split; retained for API parity. Defaulted rather than left None purely for consistency with the other generators (juniper-data#319) -- setting it changes nothing here. Note this generator's real non-reproducibility source is elsewhere: ``end_date`` defaults to the wall clock, so the same params yield different data on different days."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> EquitiesParams:
        """Validate ratio bounds and date formats."""
        if self.train_ratio + self.val_ratio + self.test_ratio > 1.0:
            raise ValueError(f"train_ratio + val_ratio + test_ratio must not exceed 1.0, got {self.train_ratio} + {self.val_ratio} + {self.test_ratio}")
        _validate_date("start_date", self.start_date)
        _validate_date("purchase_date", self.purchase_date)
        if self.end_date is not None:
            _validate_date("end_date", self.end_date)
        return self
