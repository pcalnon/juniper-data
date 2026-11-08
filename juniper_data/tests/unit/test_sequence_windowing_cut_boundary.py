"""Pin the cut equalities the sequence-windowing property tests cannot see.

``test_windowing_invariants`` checks that every train target precedes every
validation target (I2) and that an embargoed window starts at or after its cut
(I4). Both stay green if the ``<`` comparisons in ``window_one_ticker`` become
``<=``: the window whose target lands on the cut moves into the earlier split,
and a window that starts exactly on the cut is dropped. A train target on the
validation cut is the leak the per-entity construction exists to make
impossible, and it does not announce itself.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_sequence_windowing_cut_boundary.py
# Author:        Paul Calnon
# Version:       0.17.1
# License:       MIT License

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from juniper_data.generators._sequence import _yyyymmdd_to_ordinal, window_one_ticker

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _calendar_series(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``n`` consecutive days from 2020-01-06. ``y_reg`` is the row index."""
    start = _dt.date(2020, 1, 6)
    dates = np.empty(n, dtype=np.int32)
    for i in range(n):
        day = start + _dt.timedelta(days=i)
        dates[i] = day.year * 10000 + day.month * 100 + day.day
    feats = np.zeros((n, 2), dtype=np.float32)
    y_dir = np.zeros((n, 2), dtype=np.float32)
    y_reg = np.arange(n, dtype=np.float32).reshape(n, 1)
    return feats, dates, y_dir, y_reg


def _ordinal(dates: np.ndarray, index: int) -> int:
    return int(_yyyymmdd_to_ordinal(dates[index : index + 1])[0])


def _end_rows(block: dict[str, np.ndarray]) -> list[int]:
    if block["y_reg"].shape[0] == 0:
        return []
    return block["y_reg"][:, 0].astype(int).tolist()


def _window(dates: np.ndarray, *, lookback: int, cut_at: int, val_cut_at: int, embargo: bool) -> dict[str, dict[str, np.ndarray]]:
    feats, _dates, y_dir, y_reg = _calendar_series(len(dates))
    return window_one_ticker(
        feats,
        dates,
        y_dir,
        y_reg,
        0,
        lookback=lookback,
        cut_ordinal=_ordinal(dates, cut_at),
        val_cut_ordinal=_ordinal(dates, val_cut_at),
        embargo=embargo,
    )


def test_a_target_on_the_cut_is_the_later_split() -> None:
    """Train is strictly before the cut. The cut date itself is validation, and the val/test cut is test.

    Moving end-row 3 from validation into train would leave every remaining train
    target before every remaining validation target, so I2 would stay green.
    """
    _feats, dates, _y_dir, _y_reg = _calendar_series(8)
    out = _window(dates, lookback=2, cut_at=4, val_cut_at=6, embargo=False)

    assert _end_rows(out["train"]) == [1, 2]
    assert _end_rows(out["val"]) == [3, 4]
    assert _end_rows(out["test"]) == [5, 6]

    # End-row 3 predicts row 4, the train/val cut. End-row 5 predicts row 6, the val/test cut.
    assert _ordinal(dates, 4) == _ordinal(dates, _end_rows(out["val"])[0] + 1)
    assert _ordinal(dates, 6) == _ordinal(dates, _end_rows(out["test"])[0] + 1)
    train_targets = {_ordinal(dates, end + 1) for end in _end_rows(out["train"])}
    val_targets = {_ordinal(dates, end + 1) for end in _end_rows(out["val"])}
    assert _ordinal(dates, 4) not in train_targets
    assert _ordinal(dates, 6) not in val_targets


def test_embargo_keeps_a_window_that_starts_on_the_cut() -> None:
    """A first step on the cut does not reach back across it. One step earlier does.

    I4 only checks that whatever remains starts at or after the cut, so dropping
    the window that starts on the cut would stay green.
    """
    _feats, dates, _y_dir, _y_reg = _calendar_series(12)
    lookback = 3
    open_gate = _window(dates, lookback=lookback, cut_at=4, val_cut_at=8, embargo=False)
    purged = _window(dates, lookback=lookback, cut_at=4, val_cut_at=8, embargo=True)

    # End-row 6 starts on the train/val cut (row 4). End-row 5 starts one row before it.
    assert 6 in _end_rows(open_gate["val"]) and 6 in _end_rows(purged["val"])
    assert 5 in _end_rows(open_gate["val"]) and 5 not in _end_rows(purged["val"])
    # End-row 10 starts on the val/test cut (row 8). End-row 9 starts one row before it.
    assert 10 in _end_rows(open_gate["test"]) and 10 in _end_rows(purged["test"])
    assert 9 in _end_rows(open_gate["test"]) and 9 not in _end_rows(purged["test"])
    # Train has no preceding cut, so embargo does not touch it.
    assert _end_rows(purged["train"]) == _end_rows(open_gate["train"]) == [2]

    def _first_step(block: dict[str, np.ndarray], end_row: int) -> int:
        idx = _end_rows(block).index(end_row)
        return int(block["date"][idx, 0])

    assert _first_step(purged["val"], 6) == int(dates[4])
    assert _first_step(purged["test"], 10) == int(dates[8])
