"""Unit tests for the delay-product (bilinear capacity) irregular-Δt generator (DP-3 §8a).

Pins the additive 3-D regression contract with a GENUINELY non-uniform per-step
``dt``, the in-window BILINEAR product target ``y = x(t−τ₁)·x(t−τ₂)`` (and its
closed-form known-answer), the windowing-leakage guarantee (``y`` reads only
in-window steps; ``y_full == concat(train, test)``), the model-free CAPACITY
contract (an unrestricted linear map of the whole window leaves substantial
residual — so no linear functional of the window, hence none of the LMU memory,
can fit it, while the true product feature does), determinism, parameter
validation, and the task-type / sequence metadata dispatch.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_delay_product_generator.py
# Author:        Paul Calnon
# Version:       0.9.0
# License:       MIT License

from __future__ import annotations

import numpy as np
import pytest

from juniper_data.core.meta import compute_shape_meta, derive_sequence_meta
from juniper_data.generators.delay_product import DelayProductGenerator, DelayProductParams, get_schema
from juniper_data.tests.partitions import whole

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _assert_irregular_sequence_contract(arrays: dict, *, lookback: int, horizon: int) -> None:
    """Shared assertions for the irregular-Δt 3-D regression NPZ contract (non-uniform dt)."""
    for split in ("train", "val", "test"):
        for key in ("X", "y", "dt", "target_dt", "observed_mask"):
            assert f"{key}_{split}" in arrays, f"missing {key}_{split}"

    xf = whole(arrays, "X")
    n_windows = xf.shape[0]
    assert xf.ndim == 3 and xf.shape[1:] == (lookback, 1) and xf.dtype == np.float32
    assert whole(arrays, "y").shape == (n_windows, 1) and whole(arrays, "y").dtype == np.float32

    # Irregular Δt: first step is 0 by contract; the rest are strictly positive.
    dt = whole(arrays, "dt")
    assert dt.shape == (n_windows, lookback) and dt.dtype == np.float32
    assert np.all(dt[:, 0] == 0)
    assert np.all(dt[:, 1:] > 0)
    assert whole(arrays, "target_dt").shape == (n_windows,) and np.all(whole(arrays, "target_dt") > 0)

    mask = whole(arrays, "observed_mask")
    assert mask.shape == (n_windows, lookback) and mask.dtype == np.uint8 and np.all(mask == 1)

    # full == train + val + test, chronological. The non-empty check comes first
    # deliberately: the three-way identity also holds when val rounds to zero rows,
    # so without it this assertion would pass on exactly the defect it exists to catch.
    assert arrays["X_val"].shape[0] > 0, "val partition must be non-empty"
    assert n_windows == arrays["X_train"].shape[0] + arrays["X_val"].shape[0] + arrays["X_test"].shape[0]
    np.testing.assert_array_equal(whole(arrays, "X"), np.concatenate([arrays["X_train"], arrays["X_val"], arrays["X_test"]]))

    shape_meta = compute_shape_meta(arrays, "regression")
    assert shape_meta["n_features"] == 1
    assert shape_meta["n_classes"] is None and shape_meta["class_distribution"] is None
    seq_meta = derive_sequence_meta(arrays, "steps")
    assert seq_meta["sequence"] is True and seq_meta["lookback"] == lookback and seq_meta["time_unit"] == "steps"


class TestDelayProductGenerator:
    """End-to-end behavior of DelayProductGenerator.generate()."""

    def test_contract_and_metadata(self) -> None:
        arrays = DelayProductGenerator.generate(DelayProductParams(n_steps=400, lookback=24, horizon=2, jitter=0.5, lag1=2, lag2=9, seed=0))
        _assert_irregular_sequence_contract(arrays, lookback=24, horizon=2)

    def test_dt_is_genuinely_non_uniform(self) -> None:
        # Same irregular-Δt sampling as irregular_sine: dt varies step-to-step.
        arrays = DelayProductGenerator.generate(DelayProductParams(n_steps=400, lookback=24, jitter=0.6, sample_dt=1.0, seed=0))
        dt = whole(arrays, "dt")
        assert dt[:, 1:].std() > 0.05
        assert not np.allclose(dt[:, 1:], dt[0, 1])  # not a constant gap
        assert whole(arrays, "target_dt").std() > 0.0  # the (advisory) forecast horizon also varies

    def test_target_is_in_window_product(self) -> None:
        # y is EXACTLY the product of the two delayed in-window positions of X
        # (windowing-leakage safe: the target reads only the emitted window contents).
        lookback, lag1, lag2 = 16, 1, 7
        arrays = DelayProductGenerator.generate(DelayProductParams(n_steps=300, lookback=lookback, horizon=1, lag1=lag1, lag2=lag2, jitter=0.5, seed=3))
        p1, p2 = lookback - 1 - lag1, lookback - 1 - lag2
        # Per split, not just on full. The generator overwrites the forecast target
        # window_timed_series emits with the delay product, split by split; a split
        # omitted from that loop keeps the forecast target, and X/y for that split
        # then describe two different problems. Checking only y_full would miss it
        # entirely -- full is a separate block that gets its own overwrite.
        for split in ("train", "val", "test"):
            x = arrays[f"X_{split}"]
            assert x.shape[0] > 0, f"{split} partition must be non-empty"
            np.testing.assert_array_equal(arrays[f"y_{split}"][:, 0], x[:, p1, 0] * x[:, p2, 0])
        # And the per-split products concatenate to the full target (no split-boundary drift).
        np.testing.assert_array_equal(whole(arrays, "y"), np.concatenate([arrays["y_train"], arrays["y_val"], arrays["y_test"]]))
        assert np.all(np.isfinite(whole(arrays, "y")))

    def test_closed_form_known_answer_product(self) -> None:
        params = DelayProductParams(
            n_components=2,
            frequencies=[0.05, 0.1],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 1.0],
            noise_std=0.0,
            jitter=0.5,
            n_steps=200,
            lookback=16,
            horizon=1,
            lag1=2,
            lag2=8,
            sample_dt=1.0,
            seed=0,
        )
        arrays = DelayProductGenerator.generate(params)
        # Re-derive the exact irregular sample times + closed-form signal. Explicit
        # component lists do NOT consume the RNG (see _resolve), so the gaps draw is
        # the first RNG use -- mirroring the irregular_sine known-answer test.
        rng = np.random.default_rng(0)
        gaps = 1.0 * rng.uniform(0.5, 1.5, params.n_steps - 1)
        times = np.concatenate([[0.0], np.cumsum(gaps)])
        signal = (1.0 * np.sin(2 * np.pi * 0.05 * times) + 0.5 * np.sin(2 * np.pi * 0.1 * times + 1.0)).astype(np.float32)
        n_windows = params.n_steps - params.lookback - params.horizon + 1
        starts = np.arange(n_windows)  # window j starts at raw index j
        p1, p2 = params.lookback - 1 - params.lag1, params.lookback - 1 - params.lag2
        expected = signal[starts + p1] * signal[starts + p2]
        np.testing.assert_allclose(whole(arrays, "y")[:, 0], expected, atol=1e-4)

    def test_capacity_linear_cannot_fit_but_product_can(self) -> None:
        # The defining property (DP-3 §8a): the target is QUADRATIC in the window, so
        # an UNRESTRICTED linear map of the full flattened window (a strict superset of
        # any linear functional of the LMU memory) leaves substantial residual, while
        # the true product feature fits essentially perfectly. Model-free capacity guard.
        params = DelayProductParams(n_steps=2000, lookback=24, horizon=1, lag1=3, lag2=11, jitter=0.4, n_components=4, seed=1)
        arrays = DelayProductGenerator.generate(params)
        x = whole(arrays, "X")[:, :, 0].astype(np.float64)  # (W, L)
        y = whole(arrays, "y")[:, 0].astype(np.float64)  # (W,)

        def _r2(design: np.ndarray) -> float:
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            resid = y - design @ coef
            return 1.0 - float(resid.var()) / float(y.var())

        ones = np.ones((x.shape[0], 1))
        r2_linear = _r2(np.concatenate([x, ones], axis=1))  # best linear-in-window + bias
        p1, p2 = params.lookback - 1 - params.lag1, params.lookback - 1 - params.lag2
        r2_product = _r2(np.concatenate([(x[:, p1] * x[:, p2]).reshape(-1, 1), ones], axis=1))

        assert r2_linear < 0.85, f"linear-in-window r2={r2_linear:.4f} unexpectedly high (target should be non-linear)"
        assert r2_product > 0.999, f"true product feature r2={r2_product:.6f} should be ~1"
        assert r2_product - r2_linear > 0.3, f"capacity gap too small: linear={r2_linear:.4f} product={r2_product:.4f}"

    def test_squared_term_allowed(self) -> None:
        # lag1 == lag2 gives y = x(t−τ)^2: still a valid (even) non-linear capacity target.
        arrays = DelayProductGenerator.generate(DelayProductParams(n_steps=200, lookback=16, lag1=4, lag2=4, seed=0))
        p = 16 - 1 - 4
        np.testing.assert_array_equal(whole(arrays, "y")[:, 0], whole(arrays, "X")[:, p, 0] ** 2)
        assert np.all(whole(arrays, "y") >= 0)  # a square is non-negative

    def test_determinism(self) -> None:
        first = DelayProductGenerator.generate(DelayProductParams(n_steps=300, lookback=20, lag1=2, lag2=9, seed=7))
        second = DelayProductGenerator.generate(DelayProductParams(n_steps=300, lookback=20, lag1=2, lag2=9, seed=7))
        for key in ("dt_train", "target_dt_train", "dt_val", "dt_test"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_lag_out_of_lookback_rejected(self) -> None:
        with pytest.raises(ValueError):
            DelayProductParams(lookback=16, lag1=16)  # lag must be < lookback
        with pytest.raises(ValueError):
            DelayProductParams(lookback=16, lag2=20)

    def test_component_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            DelayProductParams(n_components=3, frequencies=[0.05, 0.1])

    def test_jitter_out_of_bounds_rejected(self) -> None:
        with pytest.raises(ValueError):
            DelayProductParams(jitter=1.0)  # jitter must be < 1 (keeps gaps positive)

    def test_short_series_rejected(self) -> None:
        with pytest.raises(ValueError):
            DelayProductParams(n_steps=10, lookback=8, horizon=3)

    def test_get_schema_has_fields(self) -> None:
        schema = get_schema()
        for field in ("lag1", "lag2", "jitter", "n_components", "frequencies", "amplitudes", "phases", "noise_std", "lookback", "horizon", "sample_dt"):
            assert field in schema["properties"], f"schema missing {field}"
