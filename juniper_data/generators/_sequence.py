"""Per-entity sequence windowing for time-series / irregular-Δt NPZ artifacts.

Building block for the additive 3-D NPZ contract (WS-1 of the juniper-recurse
effort). Turns a flat, per-row feature table for ONE entity (e.g. one ticker)
into fixed-length lookback windows, derives the per-step elapsed time (Δt) from
the row dates, and splits the windows into train/test by *target time* so a
train window can never reach across the test boundary.

Windowing one entity at a time makes two whole classes of leakage structurally
impossible:

* a window can never splice two entities together (a "Frankenstein" sequence),
  because this function only ever sees one entity's rows;
* a train window's target can never land at/after the test cut, because windows
  are assigned to a split by their target date, not by row index.

The companion property test ``tests/unit/test_sequence_windowing_leakage.py``
pins the invariants (I1-I5) so a future vectorized rewrite cannot silently
reintroduce either leak. See the design note
``juniper-ml/notes/JUNIPER_RECURSE_DELTA_T_HANDLING_2026-06-05.md`` -- §6.3 (this
implementation), §6.1 (the key contract), and §7 (leakage analysis + invariants).

This module is numpy-only and has no juniper-data import dependencies, so it can
be reused by any sequence generator (the first consumer is ``equities_seq``).
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     _sequence.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import datetime as _dt

import numpy as np

# Per-window keys produced for every entity (observed_mask is added separately
# because it is derived from the window count rather than accumulated per step).
_WINDOW_KEYS = ("X", "y", "y_reg", "date", "dt", "target_dt", "window_end_date", "ticker_code")


def _yyyymmdd_to_ordinal(dates: np.ndarray) -> np.ndarray:
    """Convert ``(N,)`` int YYYYMMDD dates to ``(N,)`` int64 proleptic-Gregorian ordinals."""
    y, m, d = dates // 10000, (dates // 100) % 100, dates % 100
    return np.fromiter(
        (_dt.date(int(yy), int(mm), int(dd)).toordinal() for yy, mm, dd in zip(y, m, d)),
        dtype=np.int64,
        count=len(dates),
    )


def window_one_ticker(
    feats: np.ndarray,
    dates_yyyymmdd: np.ndarray,
    y_dir: np.ndarray,
    y_reg: np.ndarray,
    ticker_code: int,
    *,
    lookback: int,
    cut_ordinal: int,
    embargo: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Build fixed-length lookback windows for one entity, split by target time.

    A window ending at row ``i`` (with ``lookback <= i + 1 <= N``) uses steps
    ``[i - lookback + 1 .. i]`` and predicts day ``i + 1``, so valid ``i`` runs
    over ``[lookback - 1, N - 2]``. No window crosses an entity boundary because
    this function only ever sees ONE entity (see the module docstring / notes §7).

    Args:
        feats: ``(N, F)`` float features, ascending by date.
        dates_yyyymmdd: ``(N,)`` int YYYYMMDD, ascending and strictly increasing.
        y_dir: ``(N, C)`` classification target aligned so row ``i`` predicts day
            ``i + 1`` (e.g. next-day direction one-hot).
        y_reg: ``(N, R)`` regression target, same alignment.
        ticker_code: integer code identifying this entity.
        lookback: window length ``L`` (number of steps per window).
        cut_ordinal: first test-period date as an ordinal; a window is train iff
            its target time is strictly before this cut.
        embargo: when true, drop test windows whose lookback reaches before the
            cut (a purged/embargoed split), enforcing strict row-disjointness.

    Returns:
        ``{"train": {...}, "test": {...}}`` where each split maps the window keys
        (``X``, ``y``, ``y_reg``, ``date``, ``dt``, ``target_dt``,
        ``window_end_date``, ``ticker_code``, ``observed_mask``) to stacked
        arrays. Empty splits keep their canonical rank so per-entity blocks
        concatenate cleanly across entities.

    Raises:
        ValueError: if ``lookback < 1`` or the dates are not strictly increasing.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    n, f = feats.shape
    ords = _yyyymmdd_to_ordinal(dates_yyyymmdd)
    if n > 1 and not np.all(np.diff(ords) > 0):
        raise ValueError("dates must be strictly increasing within an entity")

    cols: dict[str, dict[str, list]] = {k: {"train": [], "test": []} for k in _WINDOW_KEYS}

    for i in range(lookback - 1, n - 1):  # window end at i; target at day i + 1
        lo = i - lookback + 1
        target_time = int(ords[i + 1])
        split = "train" if target_time < cut_ordinal else "test"
        if split == "test" and embargo and int(ords[lo]) < cut_ordinal:
            continue  # purge windows whose lookback straddles the cut

        win_ords = ords[lo : i + 1].astype(np.float32)
        dt = np.empty(lookback, dtype=np.float32)
        dt[0] = 0.0
        dt[1:] = np.diff(win_ords)  # calendar-day gaps (weekends => 3, holidays more)

        cols["X"][split].append(feats[lo : i + 1])
        cols["y"][split].append(y_dir[i])
        cols["y_reg"][split].append(y_reg[i])
        cols["date"][split].append(dates_yyyymmdd[lo : i + 1])
        cols["dt"][split].append(dt)
        cols["target_dt"][split].append(np.float32(ords[i + 1] - ords[i]))
        cols["window_end_date"][split].append(dates_yyyymmdd[i])
        cols["ticker_code"][split].append(np.int32(ticker_code))

    empty_shapes = {
        "X": (0, lookback, f),
        "y": (0, y_dir.shape[1]),
        "y_reg": (0, y_reg.shape[1]),
        "date": (0, lookback),
        "dt": (0, lookback),
        "target_dt": (0,),
        "window_end_date": (0,),
        "ticker_code": (0,),
    }
    dtypes = {
        "X": np.float32,
        "y": np.float32,
        "y_reg": np.float32,
        "date": np.int32,
        "dt": np.float32,
        "target_dt": np.float32,
        "window_end_date": np.int32,
        "ticker_code": np.int32,
    }

    def _stack(key: str, split: str) -> np.ndarray:
        items = cols[key][split]
        if not items:
            # Preserve rank so np.concatenate across entities is well-defined.
            return np.empty(empty_shapes[key], dtype=dtypes[key])
        return np.asarray(items, dtype=dtypes[key])

    out: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "test"):
        block = {key: _stack(key, split) for key in _WINDOW_KEYS}
        # Trading-day-native: every emitted step is a real observation, so
        # observed_mask is all-ones; dt alone carries the irregularity (notes §6.2).
        block["observed_mask"] = np.ones((block["X"].shape[0], lookback), dtype=np.uint8)
        out[split] = block
    return out
