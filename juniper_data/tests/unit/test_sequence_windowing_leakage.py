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

from juniper_data.generators._sequence import _yyyymmdd_to_ordinal, window_one_ticker

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
