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

This module depends only on numpy and the core split helper, so it can be reused
by any sequence generator: ``equities_seq`` (irregular calendar-Δt) via
``window_one_ticker``, the ``multi_sine`` / ``mackey_glass`` / ``ar_p`` synthetics
(regular-Δt) via ``window_regular_series``, and the irregular-sampling
``irregular_sine`` synthetic via ``window_timed_series``.
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

from juniper_data.core.split import temporal_split_index

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


def window_regular_series(
    series: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    sample_dt: float,
    train_ratio: float,
) -> dict[str, np.ndarray]:
    """Window a single regular-Δt series into the additive 3-D sequence contract.

    The regular-sampling sibling of :func:`window_one_ticker`, for the synthetic
    time-series generators (``multi_sine`` / ``mackey_glass`` / ``ar_p``). The
    input is ONE series sampled at a *constant* step ``sample_dt`` -- there are no
    calendar dates, so the per-step ``dt`` is constant and ``target_dt`` is the
    fixed forecast horizon. A window ending at index ``i`` uses steps
    ``[i - lookback + 1 .. i]`` and predicts the value ``horizon`` steps later
    (index ``i + horizon``); valid ``i`` runs over ``[lookback - 1, T - 1 - horizon]``.

    Windows are emitted in chronological order and split at
    :func:`~juniper_data.core.split.temporal_split_index`, so every train target
    strictly precedes every test target -- the same no-future-leak guarantee as
    the per-entity windower, here structural because there is a single series and
    a single chronological cut. ``full`` is ``train`` followed by ``test``.

    Args:
        series: ``(T, F)`` (or ``(T,)``) float series, ascending in time.
        lookback: window length ``L`` (steps per window), ``>= 1``.
        horizon: forecast horizon ``h`` in steps (target is ``h`` steps after the
            window end), ``>= 1``.
        sample_dt: constant per-step elapsed time, ``> 0`` (the regular Δt).
        train_ratio: fraction of the earliest windows used for training, ``(0, 1]``.

    Returns:
        Flat NPZ dict mapping ``{X, y, dt, target_dt, observed_mask}_{train,test,full}``:
        ``X`` ``(W, L, F)`` f32; ``y`` ``(W, F)`` f32 (the series value at the
        horizon step); ``dt`` ``(W, L)`` f32 ``[0, sample_dt, ...]``; ``target_dt``
        ``(W,)`` f32 ``= horizon * sample_dt``; ``observed_mask`` ``(W, L)`` uint8
        all-ones. ``X_full == concatenate([X_train, X_test])``.

    Raises:
        ValueError: if ``lookback < 1``, ``horizon < 1``, ``sample_dt <= 0``, the
            series is not 1-D/2-D, or it is too short to form two windows
            (``T < lookback + horizon + 1``).
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if sample_dt <= 0:
        raise ValueError(f"sample_dt must be > 0, got {sample_dt}")

    arr = np.asarray(series, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"series must be 1-D or 2-D (T, F), got {arr.ndim}-D")

    n_steps = arr.shape[0]
    n_windows = n_steps - lookback - horizon + 1
    if n_windows < 2:
        raise ValueError(f"series too short: T={n_steps}, lookback={lookback}, horizon={horizon} yields {n_windows} window(s); need >= 2 (T >= lookback + horizon + 1)")

    # Window-end index ``i`` runs over [lookback - 1, T - 1 - horizon]; the target
    # is the value ``horizon`` steps after the end (index ``i + horizon``).
    ends = np.arange(lookback - 1, n_steps - horizon)
    starts = ends - lookback + 1
    win_idx = starts[:, None] + np.arange(lookback)[None, :]  # (W, L) row indices
    x = arr[win_idx]  # (W, L, F)
    y = arr[ends + horizon]  # (W, F)

    # Regular sampling: every in-window step is one ``sample_dt`` apart, the first
    # step has no predecessor (contract: ``dt[:, 0] == 0``), and the forecast
    # horizon is the fixed ``horizon * sample_dt``. Every step is a real
    # observation, so ``observed_mask`` is all-ones (nothing imputed/padded).
    dt = np.full((n_windows, lookback), np.float32(sample_dt), dtype=np.float32)
    dt[:, 0] = 0.0
    target_dt = np.full(n_windows, np.float32(horizon * sample_dt), dtype=np.float32)
    observed_mask = np.ones((n_windows, lookback), dtype=np.uint8)

    cut = temporal_split_index(n_windows, train_ratio)

    out: dict[str, np.ndarray] = {}
    for key, full in (("X", x), ("y", y), ("dt", dt), ("target_dt", target_dt), ("observed_mask", observed_mask)):
        out[f"{key}_train"] = full[:cut]
        out[f"{key}_test"] = full[cut:]
        out[f"{key}_full"] = full
    return out


def window_timed_series(
    values: np.ndarray,
    times: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    train_ratio: float,
) -> dict[str, np.ndarray]:
    """Window an irregularly-sampled series (explicit per-step times) into the 3-D contract.

    The irregular-Δt sibling of :func:`window_regular_series`, for synthetics that
    sample a continuous-time process at *non-uniform* times (e.g. ``irregular_sine``).
    Instead of a constant ``sample_dt`` the caller supplies the absolute sample
    ``times``; the per-step ``dt`` is derived from their differences within each
    window, so ``dt`` is genuinely non-uniform and ``target_dt`` is the (variable)
    time from the window end to its target step. A window ending at index ``i``
    uses steps ``[i - lookback + 1 .. i]`` and predicts the value ``horizon`` steps
    later (index ``i + horizon``).

    Windows are split at :func:`~juniper_data.core.split.temporal_split_index` --
    the same no-future-leak guarantee as :func:`window_regular_series`. ``full`` is
    ``train`` followed by ``test``.

    Args:
        values: ``(T, F)`` (or ``(T,)``) float series, ascending in time.
        times: ``(T,)`` strictly-increasing float sample times (aligned with
            ``values``); the irregular-Δt source.
        lookback: window length ``L`` (steps per window), ``>= 1``.
        horizon: forecast horizon ``h`` in steps, ``>= 1``.
        train_ratio: fraction of the earliest windows used for training, ``(0, 1]``.

    Returns:
        Flat NPZ dict mapping ``{X, y, dt, target_dt, observed_mask}_{train,test,full}``:
        ``X`` ``(W, L, F)`` f32; ``y`` ``(W, F)`` f32; ``dt`` ``(W, L)`` f32
        ``[0, diff(window times)]`` (non-uniform); ``target_dt`` ``(W,)`` f32
        ``= times[i + horizon] - times[i]``; ``observed_mask`` ``(W, L)`` uint8
        all-ones. ``X_full == concatenate([X_train, X_test])``.

    Raises:
        ValueError: if ``lookback < 1``, ``horizon < 1``, ``values`` is not
            1-D/2-D, ``times`` length != ``T`` or is not strictly increasing, or
            the series is too short to form two windows.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"values must be 1-D or 2-D (T, F), got {arr.ndim}-D")

    t = np.asarray(times, dtype=np.float64)  # keep absolute-time precision in float64
    n_steps = arr.shape[0]
    if t.ndim != 1 or t.shape[0] != n_steps:
        raise ValueError(f"times must be 1-D of length T={n_steps}, got shape {t.shape}")
    if n_steps > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing")

    n_windows = n_steps - lookback - horizon + 1
    if n_windows < 2:
        raise ValueError(f"series too short: T={n_steps}, lookback={lookback}, horizon={horizon} yields {n_windows} window(s); need >= 2 (T >= lookback + horizon + 1)")

    # Window-end index ``i`` runs over [lookback - 1, T - 1 - horizon]; the target
    # is the value ``horizon`` steps after the end (index ``i + horizon``).
    ends = np.arange(lookback - 1, n_steps - horizon)
    starts = ends - lookback + 1
    win_idx = starts[:, None] + np.arange(lookback)[None, :]  # (W, L) row indices
    x = arr[win_idx]  # (W, L, F)
    y = arr[ends + horizon]  # (W, F)

    # Irregular sampling: per-step dt is the within-window time gap (first step has
    # no predecessor -> dt[:, 0] == 0), and target_dt is the (variable) time from
    # the window end to its target step. Every step is a real observation, so
    # observed_mask is all-ones -- the irregularity lives in dt, not in masking.
    win_times = t[win_idx]  # (W, L)
    dt = np.zeros((n_windows, lookback), dtype=np.float32)
    dt[:, 1:] = np.diff(win_times, axis=1).astype(np.float32)
    target_dt = (t[ends + horizon] - t[ends]).astype(np.float32)
    observed_mask = np.ones((n_windows, lookback), dtype=np.uint8)

    cut = temporal_split_index(n_windows, train_ratio)

    out: dict[str, np.ndarray] = {}
    for key, full in (("X", x), ("y", y), ("dt", dt), ("target_dt", target_dt), ("observed_mask", observed_mask)):
        out[f"{key}_train"] = full[:cut]
        out[f"{key}_test"] = full[cut:]
        out[f"{key}_full"] = full
    return out
