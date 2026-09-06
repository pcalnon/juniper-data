"""Property tests for per-entity sequence windowing leakage invariants (I1-I5).

These pin the leakage guarantees of ``window_one_ticker``: no cross-entity
window (I1), no future leak across the train/test cut (I2), monotone per-step
time with a Δt channel consistent with the step dates (I3), embargo purging when
enabled (I4), and strictly-positive target horizons (I5). A naive
concat-then-slide windower fails I1/I2; the per-entity construction makes both
structural, and these properties pin them so a future vectorized rewrite cannot
silently reintroduce either leak. See
``juniper-ml/notes/JUNIPER_2026-06-05_JUNIPER-RECURRENCE_RECURSE-DELTA-T-HANDLING.md`` §7.
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
from juniper_data.tests.partitions import whole

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
        # Two boundaries now: train | val | test, in time order. Clamped so each
        # of the three has at least one row to claim.
        cut_idx = min(round(n * train_ratio), n - 2)
        cut_idx = max(cut_idx, 1)
        val_cut_idx = min(cut_idx + 1, n - 1)
        cut = int(ords[cut_idx])  # this entity's own validation-boundary date
        val_cut = int(ords[val_cut_idx])  # this entity's own test-boundary date
        out = window_one_ticker(X, dates, y_dir, y_reg, code, lookback=lookback, cut_ordinal=cut, val_cut_ordinal=val_cut, embargo=embargo)
        per_ticker.append((code, cut, val_cut, out))

    for code, cut, val_cut, out in per_ticker:
        tr, va, te = out["train"], out["val"], out["test"]

        # I1 -- every window belongs to exactly this entity (no cross-ticker splice).
        for blk in (tr, va, te):
            assert np.all(blk["ticker_code"] == code)

        # I3 -- monotone time inside each window; dt[:,0]==0; dt == diff(step ordinals).
        for blk in (tr, va, te):
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
        for blk in (tr, va, te):
            if blk["X"].shape[0]:
                assert np.all(blk["target_dt"] > 0)

        # I2 -- no future leak, now TRANSITIVE across three partitions: every train
        # target precedes every validation target, and every validation target
        # precedes every test target. Checking only train < test would leave the
        # validation split free to overlap either neighbour, which is exactly the
        # leak the in-loop partition must not have.
        def _targets(blk):
            return _ords_of(blk["window_end_date"]) + blk["target_dt"].astype(np.int64)

        for earlier, later in ((tr, va), (va, te), (tr, te)):
            if earlier["X"].shape[0] and later["X"].shape[0]:
                assert _targets(earlier).max() < _targets(later).min()

        # I4 -- embargo purges windows whose lookback straddles the cut PRECEDING
        # their own split, at both boundaries.
        if embargo:
            for blk, preceding in ((va, cut), (te, val_cut)):
                if blk["X"].shape[0]:
                    assert np.all(_ords_of(blk["date"][:, 0]) >= preceding)


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
    train_ratio=st.floats(0.5, 0.8),
    val_ratio=st.floats(0.05, 0.15),
)
def test_regular_windowing_invariants(n_steps, lookback, horizon, sample_dt, train_ratio, val_ratio):
    """``window_regular_series``: regular-Δt contract, index encoding, no future leak.

    The series value encodes its own index (``series[k] == k``), so each window's
    content reveals exactly which steps it spans -- letting us pin that a window is
    L consecutive steps, the target is the step ``horizon`` after the window end,
    and every train target strictly precedes every test target (the regular-Δt
    analog of I2).
    """
    if n_steps - lookback - horizon + 1 < 3:
        return  # too short for a three-way split; the windower raises (covered in the unit tests)
    series = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)  # value == index
    out = window_regular_series(series, lookback=lookback, horizon=horizon, sample_dt=sample_dt, train_ratio=train_ratio, val_ratio=val_ratio)
    n_windows = n_steps - lookback - horizon + 1

    # RR1 -- shapes.
    assert whole(out, "X").shape == (n_windows, lookback, 1)
    assert whole(out, "y").shape == (n_windows, 1)

    # RR2 -- regular-Δt contract: dt[:,0]==0, a constant gap, a fixed target horizon.
    assert np.all(whole(out, "dt")[:, 0] == 0)
    assert np.allclose(whole(out, "dt")[:, 1:], np.float32(sample_dt))
    assert np.allclose(whole(out, "target_dt"), np.float32(horizon * sample_dt))
    assert np.all(whole(out, "observed_mask") == 1)

    # RR3 -- index encoding: each window is L consecutive steps; target == end + horizon.
    steps = whole(out, "X")[:, :, 0]
    assert np.all(np.diff(steps, axis=1) == 1)
    np.testing.assert_array_equal(whole(out, "y")[:, 0], steps[:, -1] + horizon)

    # RR4 -- full == train + val + test, chronological. The identity spans THREE
    # partitions now; over train + test alone it would pass only while val is empty.
    assert out["X_val"].shape[0] > 0, "X_val must be non-empty, or RR4/RR5 hold vacuously"
    assert n_windows == out["X_train"].shape[0] + out["X_val"].shape[0] + out["X_test"].shape[0]
    np.testing.assert_array_equal(whole(out, "X"), np.concatenate([out["X_train"], out["X_val"], out["X_test"]]))

    # RR5 -- no future leak, TRANSITIVE: train targets precede val targets, which
    # precede test targets. Values encode their step index, so a plain max/min
    # comparison is exactly the chronological ordering.
    for earlier, later in (("train", "val"), ("val", "test"), ("train", "test")):
        if out[f"y_{earlier}"].shape[0] and out[f"y_{later}"].shape[0]:
            assert out[f"y_{earlier}"][:, 0].max() < out[f"y_{later}"][:, 0].min()


@settings(max_examples=200, deadline=None)
@given(
    n_steps=st.integers(8, 200),
    lookback=st.integers(2, 12),
    horizon=st.integers(1, 6),
    gaps=st.lists(st.floats(0.05, 10.0, allow_nan=False, allow_infinity=False), min_size=199, max_size=199),
    train_ratio=st.floats(0.5, 0.8),
    val_ratio=st.floats(0.05, 0.15),
)
def test_timed_windowing_invariants(n_steps, lookback, horizon, gaps, train_ratio, val_ratio):
    """``window_timed_series``: dt == within-window time-diffs, variable target_dt, no future leak.

    Index-encoded values (``series[k] == k``) plus strictly-increasing irregular
    times (cumulative positive gaps) let us pin that ``dt`` equals the per-step
    time differences, ``target_dt`` is the window-end -> target time gap, and every
    train target strictly precedes every test target (the irregular-Δt analog of
    the regular-windowing invariants).
    """
    if n_steps - lookback - horizon + 1 < 3:
        return  # too short for a three-way split; the windower raises (covered in the unit tests)
    values = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)  # value == index
    times = np.concatenate([[0.0], np.cumsum(np.asarray(gaps[: n_steps - 1]))])  # strictly increasing
    out = window_timed_series(values, times, lookback=lookback, horizon=horizon, train_ratio=train_ratio, val_ratio=val_ratio)
    n_windows = n_steps - lookback - horizon + 1

    # TR1 -- shapes.
    assert whole(out, "X").shape == (n_windows, lookback, 1)
    assert whole(out, "y").shape == (n_windows, 1)

    # The X value at each cell encodes the original step index it spans.
    steps = whole(out, "X")[:, :, 0].astype(np.int64)

    # TR2 -- dt[:,0]==0; dt[:,1:] == the within-window time differences; all > 0.
    assert np.all(whole(out, "dt")[:, 0] == 0)
    expected_dt = np.diff(times[steps], axis=1).astype(np.float32)
    np.testing.assert_allclose(whole(out, "dt")[:, 1:], expected_dt, rtol=1e-5, atol=1e-5)
    assert np.all(whole(out, "dt")[:, 1:] > 0)

    # TR3 -- target_dt == time(target step) - time(window end).
    ends = steps[:, -1]
    expected_target_dt = (times[ends + horizon] - times[ends]).astype(np.float32)
    np.testing.assert_allclose(whole(out, "target_dt"), expected_target_dt, rtol=1e-5, atol=1e-5)

    # TR4 -- index encoding: each window is L consecutive steps; target == end + horizon.
    assert np.all(np.diff(steps, axis=1) == 1)
    np.testing.assert_array_equal(whole(out, "y")[:, 0].astype(np.int64), ends + horizon)

    # TR5 -- full == train + val + test; no future leak, TRANSITIVE.
    #
    # The identity spans THREE partitions now. Over train + test alone it would
    # pass only while val is empty, so the non-empty guard below is what stops it
    # from silently degrading into a vacuous check.
    assert out["X_val"].shape[0] > 0, "X_val must be non-empty, or TR5 holds vacuously"
    assert n_windows == out["X_train"].shape[0] + out["X_val"].shape[0] + out["X_test"].shape[0]
    np.testing.assert_array_equal(whole(out, "X"), np.concatenate([out["X_train"], out["X_val"], out["X_test"]]))
    for earlier, later in (("train", "val"), ("val", "test"), ("train", "test")):
        if out[f"y_{earlier}"].shape[0] and out[f"y_{later}"].shape[0]:
            assert out[f"y_{earlier}"][:, 0].max() < out[f"y_{later}"][:, 0].min()


class TestSequenceWindowingValidation:
    """Cover the argument-validation branches of the three windowers (deterministic)."""

    @staticmethod
    def _one_ticker_args(dates: np.ndarray) -> tuple:
        n = len(dates)
        feats = np.zeros((n, 2), dtype=np.float32)
        y_dir = np.zeros((n, 2), dtype=np.float32)
        y_reg = np.zeros((n, 1), dtype=np.float32)
        return feats, dates, y_dir, y_reg

    def test_one_ticker_rejects_lookback_below_one(self) -> None:
        feats, dates, y_dir, y_reg = self._one_ticker_args(np.array([20200101, 20200102, 20200103], dtype=np.int64))
        with pytest.raises(ValueError, match="lookback must be >= 1"):
            window_one_ticker(feats, dates, y_dir, y_reg, 0, lookback=0, cut_ordinal=0, val_cut_ordinal=0)

    def test_one_ticker_rejects_non_increasing_dates(self) -> None:
        feats, dates, y_dir, y_reg = self._one_ticker_args(np.array([20200103, 20200101, 20200105], dtype=np.int64))
        with pytest.raises(ValueError, match="strictly increasing"):
            window_one_ticker(feats, dates, y_dir, y_reg, 0, lookback=1, cut_ordinal=0, val_cut_ordinal=0)

    def test_regular_series_rejects_lookback_below_one(self) -> None:
        with pytest.raises(ValueError, match="lookback must be >= 1"):
            window_regular_series(np.arange(6.0), lookback=0, horizon=1, sample_dt=1.0, train_ratio=0.5, val_ratio=0.25)

    def test_regular_series_rejects_horizon_below_one(self) -> None:
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            window_regular_series(np.arange(6.0), lookback=2, horizon=0, sample_dt=1.0, train_ratio=0.5, val_ratio=0.25)

    def test_regular_series_rejects_nonpositive_sample_dt(self) -> None:
        with pytest.raises(ValueError, match="sample_dt must be > 0"):
            window_regular_series(np.arange(6.0), lookback=2, horizon=1, sample_dt=0.0, train_ratio=0.5, val_ratio=0.25)

    def test_regular_series_accepts_1d_input(self) -> None:
        out = window_regular_series(np.arange(6.0), lookback=2, horizon=1, sample_dt=1.0, train_ratio=0.5, val_ratio=0.25)
        assert whole(out, "X").ndim == 3 and whole(out, "X").shape[2] == 1  # 1-D reshaped to (W, L, 1)

    def test_regular_series_rejects_3d_input(self) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            window_regular_series(np.zeros((6, 1, 1)), lookback=2, horizon=1, sample_dt=1.0, train_ratio=0.5, val_ratio=0.25)

    def test_regular_series_rejects_too_short(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            window_regular_series(np.arange(3.0), lookback=2, horizon=1, sample_dt=1.0, train_ratio=0.5, val_ratio=0.25)

    def test_timed_series_rejects_lookback_below_one(self) -> None:
        with pytest.raises(ValueError, match="lookback must be >= 1"):
            window_timed_series(np.arange(6.0), np.arange(6.0), lookback=0, horizon=1, train_ratio=0.5, val_ratio=0.25)

    def test_timed_series_rejects_horizon_below_one(self) -> None:
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            window_timed_series(np.arange(6.0), np.arange(6.0), lookback=2, horizon=0, train_ratio=0.5, val_ratio=0.25)

    def test_timed_series_accepts_1d_values(self) -> None:
        out = window_timed_series(np.arange(6.0), np.arange(6.0), lookback=2, horizon=1, train_ratio=0.5, val_ratio=0.25)
        assert whole(out, "X").ndim == 3 and whole(out, "X").shape[2] == 1

    def test_timed_series_rejects_3d_values(self) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            window_timed_series(np.zeros((6, 1, 1)), np.arange(6.0), lookback=2, horizon=1, train_ratio=0.5, val_ratio=0.25)

    def test_timed_series_rejects_times_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="times must be 1-D of length"):
            window_timed_series(np.arange(6.0), np.arange(5.0), lookback=2, horizon=1, train_ratio=0.5, val_ratio=0.25)

    def test_timed_series_rejects_non_increasing_times(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            window_timed_series(np.arange(6.0), np.array([0.0, 2.0, 1.0, 3.0, 4.0, 5.0]), lookback=2, horizon=1, train_ratio=0.5, val_ratio=0.25)

    def test_timed_series_rejects_too_short(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            window_timed_series(np.arange(3.0), np.arange(3.0), lookback=2, horizon=1, train_ratio=0.5, val_ratio=0.25)
