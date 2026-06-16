"""Unit tests for the irregular-Δt sine synthetic generator (juniper-data#179 §A).

Pins the additive 3-D regression contract with a GENUINELY non-uniform per-step
``dt`` (and variable ``target_dt``), the closed-form known-answer signal evaluated
at the irregular sample times, the ``jitter`` irregularity control, determinism,
parameter validation, and the task-type / sequence metadata dispatch.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_irregular_sine_generator.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import numpy as np
import pytest

from juniper_data.core.meta import compute_shape_meta, derive_sequence_meta
from juniper_data.generators.irregular_sine import IrregularSineGenerator, IrregularSineParams, get_schema

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _assert_irregular_sequence_contract(arrays: dict, *, lookback: int, horizon: int) -> None:
    """Shared assertions for the irregular-Δt 3-D regression NPZ contract (non-uniform dt)."""
    for split in ("train", "test", "full"):
        for key in ("X", "y", "dt", "target_dt", "observed_mask"):
            assert f"{key}_{split}" in arrays, f"missing {key}_{split}"

    xf = arrays["X_full"]
    n_windows = xf.shape[0]
    assert xf.ndim == 3 and xf.shape[1:] == (lookback, 1) and xf.dtype == np.float32
    assert arrays["y_full"].shape == (n_windows, 1) and arrays["y_full"].dtype == np.float32

    # Irregular Δt: first step is 0 by contract; the rest are strictly positive
    # (strictly-increasing sample times) -- but NOT constant (asserted separately).
    dt = arrays["dt_full"]
    assert dt.shape == (n_windows, lookback) and dt.dtype == np.float32
    assert np.all(dt[:, 0] == 0)
    assert np.all(dt[:, 1:] > 0)
    assert arrays["target_dt_full"].shape == (n_windows,) and np.all(arrays["target_dt_full"] > 0)

    mask = arrays["observed_mask_full"]
    assert mask.shape == (n_windows, lookback) and mask.dtype == np.uint8 and np.all(mask == 1)

    assert n_windows == arrays["X_train"].shape[0] + arrays["X_test"].shape[0]
    np.testing.assert_array_equal(arrays["X_full"], np.concatenate([arrays["X_train"], arrays["X_test"]]))

    shape_meta = compute_shape_meta(arrays, "regression")
    assert shape_meta["n_features"] == 1
    assert shape_meta["n_classes"] is None and shape_meta["class_distribution"] is None
    seq_meta = derive_sequence_meta(arrays, "steps")
    assert seq_meta["sequence"] is True and seq_meta["lookback"] == lookback and seq_meta["time_unit"] == "steps"


class TestIrregularSineGenerator:
    """End-to-end behavior of IrregularSineGenerator.generate()."""

    def test_contract_and_metadata(self) -> None:
        arrays = IrregularSineGenerator.generate(IrregularSineParams(n_steps=400, lookback=24, horizon=2, jitter=0.5, seed=0))
        _assert_irregular_sequence_contract(arrays, lookback=24, horizon=2)

    def test_dt_is_genuinely_non_uniform(self) -> None:
        # The whole point of this generator: dt varies step-to-step (unlike the
        # regular synthetics, where window_regular_series gives a constant dt).
        arrays = IrregularSineGenerator.generate(IrregularSineParams(n_steps=400, lookback=24, jitter=0.6, sample_dt=1.0, seed=0))
        dt = arrays["dt_full"]
        assert dt[:, 1:].std() > 0.05
        assert not np.allclose(dt[:, 1:], dt[0, 1])  # not a constant gap
        assert arrays["target_dt_full"].std() > 0.0  # the forecast horizon also varies

    def test_closed_form_known_answer_at_irregular_times(self) -> None:
        params = IrregularSineParams(
            n_components=2,
            frequencies=[0.05, 0.1],
            amplitudes=[1.0, 0.5],
            phases=[0.0, 1.0],
            noise_std=0.0,
            jitter=0.5,
            n_steps=200,
            lookback=16,
            horizon=1,
            sample_dt=1.0,
            seed=0,
        )
        arrays = IrregularSineGenerator.generate(params)
        # Re-derive the exact irregular sample times + the closed-form signal.
        rng = np.random.default_rng(0)
        gaps = 1.0 * rng.uniform(0.5, 1.5, params.n_steps - 1)
        times = np.concatenate([[0.0], np.cumsum(gaps)])
        signal = 1.0 * np.sin(2 * np.pi * 0.05 * times) + 0.5 * np.sin(2 * np.pi * 0.1 * times + 1.0)
        np.testing.assert_allclose(arrays["X_full"][0, :, 0], signal[:16], atol=1e-4)
        ends = np.arange(15, params.n_steps - 1)  # lookback - 1 .. T - 1 - horizon
        np.testing.assert_allclose(arrays["y_full"][:, 0], signal[ends + 1], atol=1e-4)
        # dt within the first window equals the per-step time differences.
        np.testing.assert_allclose(arrays["dt_full"][0, 1:], np.diff(times[:16]).astype(np.float32), atol=1e-5)

    def test_jitter_controls_irregularity(self) -> None:
        # Larger jitter => a more dispersed per-step dt.
        low = IrregularSineGenerator.generate(IrregularSineParams(n_steps=400, lookback=24, jitter=0.1, seed=0))
        high = IrregularSineGenerator.generate(IrregularSineParams(n_steps=400, lookback=24, jitter=0.8, seed=0))
        assert high["dt_full"][:, 1:].std() > low["dt_full"][:, 1:].std()

    def test_determinism(self) -> None:
        first = IrregularSineGenerator.generate(IrregularSineParams(n_steps=300, lookback=20, seed=7))
        second = IrregularSineGenerator.generate(IrregularSineParams(n_steps=300, lookback=20, seed=7))
        for key in ("X_full", "y_full", "dt_full", "target_dt_full"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_component_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            IrregularSineParams(n_components=3, frequencies=[0.05, 0.1])

    def test_jitter_out_of_bounds_rejected(self) -> None:
        with pytest.raises(ValueError):
            IrregularSineParams(jitter=1.0)  # jitter must be < 1 (keeps gaps positive)

    def test_short_series_rejected(self) -> None:
        with pytest.raises(ValueError):
            IrregularSineParams(n_steps=10, lookback=8, horizon=3)

    def test_get_schema_has_fields(self) -> None:
        schema = get_schema()
        for field in ("jitter", "n_components", "frequencies", "amplitudes", "phases", "noise_std", "lookback", "horizon", "sample_dt"):
            assert field in schema["properties"], f"schema missing {field}"
