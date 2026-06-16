"""Property tests for per-entity sequence windowing leakage invariants (I1-I5).

These pin the leakage guarantees of ``window_one_ticker``: no cross-entity
window (I1), no future leak across the train/test cut (I2), monotone per-step
time with a Δt channel consistent with the step dates (I3), embargo purging when
enabled (I4), and strictly-positive target horizons (I5). A naive
concat-then-slide windower fails I1/I2; the per-entity construction makes both
structural, and these properties pin them so a future vectorized rewrite cannot
silently reintroduce either leak. See
``juniper-ml/notes/JUNIPER_RECURSE_DELTA_T_HANDLING_2026-06-05.md`` §7.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_sequence_windowing_leakage.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from juniper_data.generators._sequence import _yyyymmdd_to_ordinal, window_one_ticker, window_regular_series, window_timed_series

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _to_ordinal(v: int) -> int:
    v = int(v)
    return _dt.date(v // 10000, (v // 100) % 100, v % 100).toordinal()


# Vectorized YYYYMMDD -> ordinal that works on 2-D window-date arrays too
# (the module helper is 1-D only); otypes pins the dtype for empty inputs.
_ords_of = np.vectorize(_to_ordinal, otypes=[np.int64])


def _ordinal_to_yyyymmdd(o: int) -> int:
    d = _dt.date.fromordinal(int(o))
    return d.year * 10000 + d.month * 100 + d.day


@st.composite
def _ticker_series(draw, min_len: int = 4, max_len: int = 40):
    """One entity: strictly-increasing IRREGULAR dates + random features/targets."""
    n = draw(st.integers(min_len, max_len))
    start = _dt.date(2000, 1, 3).toordinal()
    gaps = draw(st.lists(st.integers(1, 5), min_size=n - 1, max_size=n - 1))  # 1..5 day gaps
    ords = np.concatenate([[start], start + np.cumsum(gaps)]).astype(np.int64)
    dates = np.array([_ordinal_to_yyyymmdd(o) for o in ords], dtype=np.int32)
    flat = draw(
        st.lists(
            st.floats(-5, 5, allow_nan=False, allow_infinity=False),
            min_size=n * 3,
            max_size=n * 3,
        )
    )
    X = np.asarray(flat, dtype=np.float32).reshape(n, 3)
    y_dir = np.eye(2, dtype=np.float32)[np.arange(n) % 2]  # deterministic stand-in
    y_reg = X[:, :1].copy()
    return X, dates, y_dir, y_reg


@settings(max_examples=200, deadline=None)
@given(
    series=st.lists(_ticker_series(), min_size=1, max_size=4),
    lookback=st.integers(2, 6),
    train_ratio=st.floats(0.5, 0.9),
    embargo=st.booleans(),
)
def test_windowing_invariants(series, lookback, train_ratio, embargo):
    per_ticker = []
    for code, (X, dates, y_dir, y_reg) in enumerate(series):
        n = len(dates)
        if n <= lookback + 1:
            continue
        ords = _yyyymmdd_to_ordinal(dates)
        cut_idx = min(round(n * train_ratio), n - 1)  # clamp: always leave >= 1 test row
        cut = int(ords[cut_idx])  # this entity's own test-boundary date
        out = window_one_ticker(X, dates, y_dir, y_reg, code, lookback=lookback, cut_ordinal=cut, embargo=embargo)
        per_ticker.append((code, cut, out))

    for code, cut, out in per_ticker:
        tr, te = out["train"], out["test"]

        # I1 -- every window belongs to exactly this entity (no cross-ticker splice).
        assert np.all(tr["ticker_code"] == code)
        assert np.all(te["ticker_code"] == code)

        # I3 -- monotone time inside each window; dt[:,0]==0; dt == diff(step ordinals).
        for blk in (tr, te):
            if blk["X"].shape[0] == 0:
                continue
            step_ords = _ords_of(blk["date"])
            assert np.all(np.diff(step_ords, axis=1) > 0)
            assert np.all(blk["dt"][:, 0] == 0)
            assert np.all(blk["dt"][:, 1:] > 0)
            assert np.allclose(blk["dt"][:, 1:], np.diff(step_ords, axis=1))
            # observed_mask is all-ones in trading-day-native mode (notes §6.2).
            assert blk["observed_mask"].shape == (blk["X"].shape[0], lookback)
            assert np.all(blk["observed_mask"] == 1)

        # I5 -- target strictly after the window's last step.
        for blk in (tr, te):
            if blk["X"].shape[0]:
                assert np.all(blk["target_dt"] > 0)

        # I2 -- no future leak: every train target strictly precedes every test target.
        if tr["X"].shape[0] and te["X"].shape[0]:
            tr_targets = _ords_of(tr["window_end_date"]) + tr["target_dt"].astype(np.int64)
            te_targets = _ords_of(te["window_end_date"]) + te["target_dt"].astype(np.int64)
            assert tr_targets.max() < te_targets.min()

        # I4 -- embargo purges test windows whose lookback straddles the cut.
        if embargo and te["X"].shape[0]:
            first_step_ord = _ords_of(te["date"][:, 0])
            assert np.all(first_step_ord >= cut)


def test_concat_then_slide_would_leak():
    """Document the cross-entity splice the per-entity construction prevents."""
    # Two entities concatenated; a naive global slide of length 3 over rows
    # [A0 A1 | B0 B1 B2] yields a window (A1, B0, B1) splicing two entities.
    codes = np.array([0, 0, 1, 1, 1])
    lookback = 3
    naive = [tuple(codes[i - lookback + 1 : i + 1]) for i in range(lookback - 1, len(codes) - 1)]
    assert any(len(set(w)) > 1 for w in naive)  # at least one cross-entity window exists


@settings(max_examples=200, deadline=None)
@given(
    n_steps=st.integers(8, 200),
    lookback=st.integers(2, 12),
    horizon=st.integers(1, 6),
    sample_dt=st.floats(0.1, 5.0, allow_nan=False, allow_infinity=False),
    train_ratio=st.floats(0.5, 0.95),
)
def test_regular_windowing_invariants(n_steps, lookback, horizon, sample_dt, train_ratio):
    """``window_regular_series``: regular-Δt contract, index encoding, no future leak.

    The series value encodes its own index (``series[k] == k``), so each window's
    content reveals exactly which steps it spans -- letting us pin that a window is
    L consecutive steps, the target is the step ``horizon`` after the window end,
    and every train target strictly precedes every test target (the regular-Δt
    analog of I2).
    """
    if n_steps - lookback - horizon + 1 < 2:
        return  # too short for two windows; the windower raises (covered in the unit tests)
    series = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)  # value == index
    out = window_regular_series(series, lookback=lookback, horizon=horizon, sample_dt=sample_dt, train_ratio=train_ratio)
    n_windows = n_steps - lookback - horizon + 1

    # RR1 -- shapes.
    assert out["X_full"].shape == (n_windows, lookback, 1)
    assert out["y_full"].shape == (n_windows, 1)

    # RR2 -- regular-Δt contract: dt[:,0]==0, a constant gap, a fixed target horizon.
    assert np.all(out["dt_full"][:, 0] == 0)
    assert np.allclose(out["dt_full"][:, 1:], np.float32(sample_dt))
    assert np.allclose(out["target_dt_full"], np.float32(horizon * sample_dt))
    assert np.all(out["observed_mask_full"] == 1)

    # RR3 -- index encoding: each window is L consecutive steps; target == end + horizon.
    steps = out["X_full"][:, :, 0]
    assert np.all(np.diff(steps, axis=1) == 1)
    np.testing.assert_array_equal(out["y_full"][:, 0], steps[:, -1] + horizon)

    # RR4 -- full == train + test, chronological.
    assert n_windows == out["X_train"].shape[0] + out["X_test"].shape[0]
    np.testing.assert_array_equal(out["X_full"], np.concatenate([out["X_train"], out["X_test"]]))

    # RR5 -- no future leak: every train target strictly precedes every test target.
    if out["y_train"].shape[0] and out["y_test"].shape[0]:
        assert out["y_train"][:, 0].max() < out["y_test"][:, 0].min()


@settings(max_examples=200, deadline=None)
@given(
    n_steps=st.integers(8, 200),
    lookback=st.integers(2, 12),
    horizon=st.integers(1, 6),
    gaps=st.lists(st.floats(0.05, 10.0, allow_nan=False, allow_infinity=False), min_size=199, max_size=199),
    train_ratio=st.floats(0.5, 0.95),
)
def test_timed_windowing_invariants(n_steps, lookback, horizon, gaps, train_ratio):
    """``window_timed_series``: dt == within-window time-diffs, variable target_dt, no future leak.

    Index-encoded values (``series[k] == k``) plus strictly-increasing irregular
    times (cumulative positive gaps) let us pin that ``dt`` equals the per-step
    time differences, ``target_dt`` is the window-end -> target time gap, and every
    train target strictly precedes every test target (the irregular-Δt analog of
    the regular-windowing invariants).
    """
    if n_steps - lookback - horizon + 1 < 2:
        return  # too short for two windows; the windower raises (covered in the unit tests)
    values = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)  # value == index
    times = np.concatenate([[0.0], np.cumsum(np.asarray(gaps[: n_steps - 1]))])  # strictly increasing
    out = window_timed_series(values, times, lookback=lookback, horizon=horizon, train_ratio=train_ratio)
    n_windows = n_steps - lookback - horizon + 1

    # TR1 -- shapes.
    assert out["X_full"].shape == (n_windows, lookback, 1)
    assert out["y_full"].shape == (n_windows, 1)

    # The X value at each cell encodes the original step index it spans.
    steps = out["X_full"][:, :, 0].astype(np.int64)

    # TR2 -- dt[:,0]==0; dt[:,1:] == the within-window time differences; all > 0.
    assert np.all(out["dt_full"][:, 0] == 0)
    expected_dt = np.diff(times[steps], axis=1).astype(np.float32)
    np.testing.assert_allclose(out["dt_full"][:, 1:], expected_dt, rtol=1e-5, atol=1e-5)
    assert np.all(out["dt_full"][:, 1:] > 0)

    # TR3 -- target_dt == time(target step) - time(window end).
    ends = steps[:, -1]
    expected_target_dt = (times[ends + horizon] - times[ends]).astype(np.float32)
    np.testing.assert_allclose(out["target_dt_full"], expected_target_dt, rtol=1e-5, atol=1e-5)

    # TR4 -- index encoding: each window is L consecutive steps; target == end + horizon.
    assert np.all(np.diff(steps, axis=1) == 1)
    np.testing.assert_array_equal(out["y_full"][:, 0].astype(np.int64), ends + horizon)

    # TR5 -- full == train + test; no future leak.
    assert n_windows == out["X_train"].shape[0] + out["X_test"].shape[0]
    np.testing.assert_array_equal(out["X_full"], np.concatenate([out["X_train"], out["X_test"]]))
    if out["y_train"].shape[0] and out["y_test"].shape[0]:
        assert out["y_train"][:, 0].max() < out["y_test"][:, 0].min()
