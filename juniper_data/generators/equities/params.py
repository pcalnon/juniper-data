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

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from juniper_data.core.constants import DEFAULT_GENERATOR_SEED

from .defaults import (
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


def _refuse_drop_with_a_later_purchase(fundamentals_fill: str, start_date: str, purchase_date: str) -> None:
    """Refuse ``fundamentals_fill="drop"`` with a ``purchase_date`` after ``start_date``.

    ``cost_basis`` is causal (APD-DATA-042): a row before the purchase session carries NaN,
    because no row may know a price from its own future. ``drop`` removed only the rows that
    lack shares, so a later purchase date left non-finite ``cost_basis`` in an artifact the
    caller had asked to be complete (juniper-ml plan F-P2). Dropping those rows as well would
    move the start of the series to the purchase date without saying so, which is a
    contradiction in the request rather than a gap in the data.

    This applies the plan's recommended R3, pending the owner's ruling (alternatives: a
    documented finite sentinel; dropping rows without refusing). Plan: juniper-ml
    ``notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md``,
    W1.8.

    WHERE: here, in the params model, not in ``generate()``. The create route validates
    params BEFORE it hashes the ``dataset_id`` and before it looks the id up in the store, so
    an artifact minted before this rule existed can never answer a request the rule now
    refuses. A refusal raised from ``generate()`` runs only on a cache miss. It is also the
    400 of APD-DATA-014 (schema-valid, semantically wrong for this generator), the status
    every other cross-field refusal in ``_validate`` already returns, and it costs no download.

    "AFTER" COUNTS WEEKDAYS, NOT CALENDAR DAYS. The defaults are ``start_date="2000-01-01"``,
    a Saturday, and ``purchase_date="2000-01-03"``, the Monday that opens that window, so a
    plain ``purchase_date > start_date`` would refuse the defaults themselves -- and with them
    juniper-canopy's two equities registry seeds and the documented recurrence bundle, all of
    which send ``drop`` with the default dates. No US equity session falls on a weekend, so a
    weekend cannot separate the start from the purchase. Exchange holidays are not modelled:
    a start on a weekday holiday followed by a purchase on the next session is refused, and
    ``purchase_date=start_date`` is the remedy, which yields the same basis.

    Raises:
        ValueError: under ``drop``, when at least one weekday lies in
            ``[start_date, purchase_date)``. The message names both dates and the fill mode.
    """
    if fundamentals_fill != "drop":
        return
    # Parsed with the format ``_validate_date`` accepted, not handed to numpy as strings: strptime
    # takes an unpadded "2008-1-2", which numpy's datetime parser rejects with an unrelated error.
    start_day = datetime.strptime(start_date, _DATE_FORMAT).date()
    purchase_day = datetime.strptime(purchase_date, _DATE_FORMAT).date()
    weekdays = int(np.busday_count(start_day, purchase_day))
    if weekdays > 0:
        raise ValueError(
            f"purchase_date {purchase_date} is after start_date {start_date} under fundamentals_fill='drop' ({weekdays} weekday(s) apart). "
            "Rows dated before the purchase have no cost basis, so 'drop' would have to delete them and the series would silently start at the purchase instead of at start_date. "
            "Set purchase_date on or before start_date, move start_date to the purchase date, or use fundamentals_fill='nan' to keep those rows with a NaN cost_basis."
        )


class EquitiesParams(BaseModel):
    """Configuration parameters for the equities time-series generator.

    Downloads and conditions daily S&P 500 equities data into the JuniperData
    NPZ contract: a numeric feature matrix with one column per entry of
    ``EQUITIES_FEATURE_COLUMNS`` (``defaults.py``; 15 as of generator 4.0.0 -- ``adj_close`` left the
    DEFAULTS in 4.0.0, see the note beside the list), a one-hot next-day
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
        description="Cost-basis purchase date (YYYY-MM-DD); per-ticker clamped to the first available trading day. Rows before it carry a NaN cost_basis. Under fundamentals_fill='drop' a purchase_date after start_date (at least one weekday apart) is refused.",
    )
    basis_price_field: Literal["close", "adj_close"] = Field(
        default=EQUITIES_DEFAULT_BASIS_PRICE_FIELD,
        description="Price field used for cost basis on the purchase date.",
    )
    fundamentals_fill: Literal["zero", "nan", "drop"] = Field(
        default=EQUITIES_DEFAULT_FUNDAMENTALS_FILL,
        description="How to represent pre-2009 missing total_shares / market_cap: zero-fill, leave NaN, or drop rows. 'drop' also drops any row before the purchase session (its cost_basis is unknown) and refuses a purchase_date after start_date.",
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
        description="Min-max normalize each feature column to [0, 1], fit on the TRAIN partition only (falling back to the full set only when train is empty). Fitting on the full set would let validation and test rows move the scaler and leak into training -- the leak juniper-data#314 removed.",
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
    allow_truncation: bool | None = Field(
        default=None,
        description="Accept a partial universe when it exceeds max_symbols, and open the gate on unresolvable-fundamentals rows. TRI-STATE (APD-DATA-052): true opts in for this request; false REFUSES truncation for this request even where the deployment enabled it; null -- the default, and what an omitted field means -- defers to JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION (or the matching .env entry), which is itself false by default, so an omitted field on a default deployment still refuses with 422. When truncation is in force the leading max_symbols symbols are imported and the dataset is PERMANENTLY annotated as truncated in its metadata.",
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
        """Validate ratio bounds, date formats, and the ``drop`` / purchase-date contradiction."""
        if self.train_ratio + self.val_ratio + self.test_ratio > 1.0:
            raise ValueError(f"train_ratio + val_ratio + test_ratio must not exceed 1.0, got {self.train_ratio} + {self.val_ratio} + {self.test_ratio}")
        _validate_date("start_date", self.start_date)
        _validate_date("purchase_date", self.purchase_date)
        if self.end_date is not None:
            _validate_date("end_date", self.end_date)
        # After the date formats: the weekday count needs two well-formed dates.
        _refuse_drop_with_a_later_purchase(self.fundamentals_fill, self.start_date, self.purchase_date)
        return self
