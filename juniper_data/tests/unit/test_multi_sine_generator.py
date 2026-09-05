"""Unit tests for the multi-sine synthetic time-series generator (juniper-data#179 §A).

Pins the additive 3-D regression contract (``X`` ``(W, L, 1)``, regression target
``y`` ``(W, 1)``, constant per-step ``dt``, all-ones ``observed_mask``,
``full == train + test``), the closed-form known-answer signal (noise-free), the
``sample_dt`` time scaling, determinism, parameter validation, and the
task-type / sequence metadata dispatch.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_multi_sine_generator.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import numpy as np
import pytest

from juniper_data.core.meta import compute_shape_meta, derive_sequence_meta
from juniper_data.generators.multi_sine import MultiSineGenerator, MultiSineParams, get_schema

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _assert_regular_sequence_contract(arrays: dict, *, lookback: int, sample_dt: float, horizon: int) -> None:
    """Shared assertions for the regular-Δt 3-D regression NPZ contract."""
    for split in ("train", "val", "test", "full"):
        for key in ("X", "y", "dt", "target_dt", "observed_mask"):
            assert f"{key}_{split}" in arrays, f"missing {key}_{split}"

    xf = arrays["X_full"]
    n_windows = xf.shape[0]
    assert xf.ndim == 3 and xf.shape[1:] == (lookback, 1)
    assert xf.dtype == np.float32
    assert arrays["y_full"].shape == (n_windows, 1) and arrays["y_full"].dtype == np.float32

    # Regular Δt: first step is 0 by contract, the rest a constant sample_dt.
    dt = arrays["dt_full"]
    assert dt.shape == (n_windows, lookback) and dt.dtype == np.float32
    assert np.all(dt[:, 0] == 0)
    np.testing.assert_allclose(dt[:, 1:], np.float32(sample_dt))
    assert arrays["target_dt_full"].shape == (n_windows,)
    np.testing.assert_allclose(arrays["target_dt_full"], np.float32(horizon * sample_dt))

    # observed_mask is all-ones uint8 (nothing imputed/padded).
    mask = arrays["observed_mask_full"]
    assert mask.shape == (n_windows, lookback) and mask.dtype == np.uint8
    assert np.all(mask == 1)

    # full == train + test (chronological).
    # full == train + val + test, chronological. The non-empty check comes first
    # deliberately: the three-way identity also holds when val rounds to zero rows,
    # so without it this assertion would pass on exactly the defect it exists to catch.
    assert arrays["X_val"].shape[0] > 0, "val partition must be non-empty"
    assert n_windows == arrays["X_train"].shape[0] + arrays["X_val"].shape[0] + arrays["X_test"].shape[0]
    np.testing.assert_array_equal(arrays["X_full"], np.concatenate([arrays["X_train"], arrays["X_val"], arrays["X_test"]]))

    # Task-type / sequence metadata dispatch: regression leaves classification
    # fields None; the 3-D X is reported as a sequence with lookback L.
    shape_meta = compute_shape_meta(arrays, "regression")
    assert shape_meta["n_features"] == 1
    assert shape_meta["n_classes"] is None and shape_meta["class_distribution"] is None
    seq_meta = derive_sequence_meta(arrays, "steps")
    assert seq_meta["sequence"] is True and seq_meta["lookback"] == lookback and seq_meta["time_unit"] == "steps"


class TestMultiSineGenerator:
    """End-to-end behavior of MultiSineGenerator.generate()."""

    def test_contract_and_metadata(self) -> None:
        arrays = MultiSineGenerator.generate(MultiSineParams(n_steps=400, lookback=24, horizon=2, sample_dt=1.0, seed=0))
        _assert_regular_sequence_contract(arrays, lookback=24, sample_dt=1.0, horizon=2)

    def test_closed_form_known_answer(self) -> None:
        # Noise-free explicit components => the exact superposition (known answer).
        params = MultiSineParams(
            n_components=2,
            frequencies=[0.05, 0.1],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 1.0],
            noise_std=0.0,
            n_steps=200,
            lookback=16,
            horizon=1,
            sample_dt=1.0,
            seed=0,
        )
        arrays = MultiSineGenerator.generate(params)
        k = np.arange(params.n_steps)
        expected = 1.0 * np.sin(2 * np.pi * 0.05 * k) + 0.5 * np.sin(2 * np.pi * 0.1 * k + 1.0)
        np.testing.assert_allclose(arrays["X_full"][0, :, 0], expected[:16], atol=1e-4)
        # Every target is the signal exactly ``horizon`` steps after the window end.
        ends = np.arange(15, params.n_steps - 1)  # lookback - 1 .. T - 1 - horizon
        np.testing.assert_allclose(arrays["y_full"][:, 0], expected[ends + 1], atol=1e-4)

    def test_sample_dt_scales_time_axis(self) -> None:
        # With sample_dt != 1 the per-step dt equals sample_dt and t = k * sample_dt.
        params = MultiSineParams(n_components=1, frequencies=[0.5], amplitudes=[1.0], phases=[0.0], noise_std=0.0, n_steps=100, lookback=10, horizon=1, sample_dt=0.1, seed=0)
        arrays = MultiSineGenerator.generate(params)
        np.testing.assert_allclose(arrays["dt_full"][:, 1:], np.float32(0.1))
        k = np.arange(params.n_steps)
        expected = np.sin(2 * np.pi * 0.5 * (k * 0.1))
        np.testing.assert_allclose(arrays["X_full"][0, :, 0], expected[:10], atol=1e-4)

    def test_determinism(self) -> None:
        first = MultiSineGenerator.generate(MultiSineParams(n_steps=300, lookback=20, seed=7))
        second = MultiSineGenerator.generate(MultiSineParams(n_steps=300, lookback=20, seed=7))
        for key in ("X_full", "y_full", "dt_full"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_noise_changes_signal_not_shape(self) -> None:
        clean = MultiSineGenerator.generate(MultiSineParams(n_steps=300, lookback=20, noise_std=0.0, seed=1))
        noisy = MultiSineGenerator.generate(MultiSineParams(n_steps=300, lookback=20, noise_std=0.5, seed=1))
        assert clean["X_full"].shape == noisy["X_full"].shape
        assert not np.allclose(clean["X_full"], noisy["X_full"])

    def test_component_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            MultiSineParams(n_components=3, frequencies=[0.05, 0.1])  # len 2 != n_components 3

    def test_short_series_rejected(self) -> None:
        with pytest.raises(ValueError):
            MultiSineParams(n_steps=10, lookback=8, horizon=3)  # W = 10 - 8 - 3 + 1 = 0

    def test_get_schema_has_component_fields(self) -> None:
        schema = get_schema()
        for field in ("n_components", "frequencies", "amplitudes", "phases", "noise_std", "lookback", "horizon", "sample_dt"):
            assert field in schema["properties"], f"schema missing {field}"
