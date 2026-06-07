"""Parameters for the windowed (3-D sequence) equities generator."""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     params.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from pydantic import Field

from juniper_data.generators.equities.params import EquitiesParams


class EquitiesSeqParams(EquitiesParams):
    """Parameters for the windowed equities time-series generator.

    Inherits the full equities configuration (universe, date range, cost basis,
    conditioning, normalization) and adds a ``lookback`` window length. The
    train/test boundary is a single chronological cut per ticker at
    ``round(n * train_ratio)`` mapped to that row's date; windows are assigned to
    a split by their target date. ``test_ratio`` is retained for API parity but
    does not gate the windowed cut — the test split is every window after the cut.
    """

    lookback: int = Field(
        default=64,
        ge=2,
        le=512,
        description="Number of past trading-day steps per sequence window (L).",
    )
