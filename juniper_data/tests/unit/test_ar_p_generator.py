"""Unit tests for the AR(p) synthetic time-series generator (juniper-data#179 §A).

Pins the additive 3-D regression contract, the exact AR(p) recurrence
known-answer (re-derived from the same seed), determinism, the warmup burn-in,
parameter validation, and the task-type / sequence metadata dispatch.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_ar_p_generator.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import numpy as np
import pytest

from juniper_data.core.meta import compute_shape_meta, derive_sequence_meta
from juniper_data.generators.ar_p import ArPGenerator, ArPParams, get_schema
from juniper_data.tests.partitions import whole

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _assert_regular_sequence_contract(arrays: dict, *, lookback: int, sample_dt: float, horizon: int) -> None:
    """Shared assertions for the regular-Δt 3-D regression NPZ contract."""
    for split in ("train", "val", "test"):
        for key in ("X", "y", "dt", "target_dt", "observed_mask"):
            assert f"{key}_{split}" in arrays, f"missing {key}_{split}"

    xf = whole(arrays, "X")
    n_windows = xf.shape[0]
    assert xf.ndim == 3 and xf.shape[1:] == (lookback, 1)
    assert xf.dtype == np.float32
    assert whole(arrays, "y").shape == (n_windows, 1) and whole(arrays, "y").dtype == np.float32

    dt = whole(arrays, "dt")
    assert dt.shape == (n_windows, lookback) and dt.dtype == np.float32
    assert np.all(dt[:, 0] == 0)
    np.testing.assert_allclose(dt[:, 1:], np.float32(sample_dt))
    assert whole(arrays, "target_dt").shape == (n_windows,)
    np.testing.assert_allclose(whole(arrays, "target_dt"), np.float32(horizon * sample_dt))

    mask = whole(arrays, "observed_mask")
    assert mask.shape == (n_windows, lookback) and mask.dtype == np.uint8
    assert np.all(mask == 1)

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


class TestArPGenerator:
    """End-to-end behavior of ArPGenerator.generate()."""

    def test_contract_and_metadata(self) -> None:
        arrays = ArPGenerator.generate(ArPParams(n_steps=400, lookback=20, horizon=1, burn_in=50, seed=0))
        _assert_regular_sequence_contract(arrays, lookback=20, sample_dt=1.0, horizon=1)

    def test_recurrence_known_answer(self) -> None:
        # Re-derive the exact trajectory from the same seed and compare.
        params = ArPParams(coefficients=[0.5, -0.3], const=0.0, sigma=0.1, burn_in=0, n_steps=300, lookback=20, horizon=1, sample_dt=1.0, seed=0)
        arrays = ArPGenerator.generate(params)

        rng = np.random.default_rng(0)
        phi = np.array([0.5, -0.3])
        order, total = 2, params.n_steps
        eps = rng.normal(0.0, 0.1, total + order)
        x = np.empty(total + order)
        x[:order] = 0.0 + eps[:order]
        for t in range(order, total + order):
            x[t] = 0.0 + phi @ x[t - order : t][::-1] + eps[t]
        expected = x[order : order + total].astype(np.float32)

        np.testing.assert_allclose(whole(arrays, "X")[0, :, 0], expected[:20], atol=1e-4)
        ends = np.arange(19, total - 1)  # lookback - 1 .. T - 1 - horizon
        np.testing.assert_allclose(whole(arrays, "y")[:, 0], expected[ends + 1], atol=1e-4)

    def test_order_p_equals_len_coefficients(self) -> None:
        # An AR(3) is accepted and produces the contract; order is len(coefficients).
        arrays = ArPGenerator.generate(ArPParams(coefficients=[0.3, 0.2, -0.1], n_steps=300, lookback=16, burn_in=20, seed=2))
        _assert_regular_sequence_contract(arrays, lookback=16, sample_dt=1.0, horizon=1)

    def test_determinism(self) -> None:
        first = ArPGenerator.generate(ArPParams(n_steps=300, lookback=20, seed=5))
        second = ArPGenerator.generate(ArPParams(n_steps=300, lookback=20, seed=5))
        np.testing.assert_array_equal(whole(first, "X"), whole(second, "X"))

    def test_burn_in_changes_start(self) -> None:
        no_warmup = ArPGenerator.generate(ArPParams(n_steps=300, lookback=20, burn_in=0, seed=0))
        warmup = ArPGenerator.generate(ArPParams(n_steps=300, lookback=20, burn_in=100, seed=0))
        assert whole(no_warmup, "X").shape == whole(warmup, "X").shape
        assert not np.allclose(whole(no_warmup, "X")[0, :, 0], whole(warmup, "X")[0, :, 0])

    def test_empty_coefficients_rejected(self) -> None:
        with pytest.raises(ValueError):
            ArPParams(coefficients=[])

    def test_short_series_rejected(self) -> None:
        with pytest.raises(ValueError):
            ArPParams(n_steps=10, lookback=8, horizon=3)

    def test_get_schema_has_ar_fields(self) -> None:
        schema = get_schema()
        for field in ("coefficients", "const", "sigma", "burn_in", "lookback", "horizon"):
            assert field in schema["properties"], f"schema missing {field}"
