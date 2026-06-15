"""Unit tests for the Mackey-Glass synthetic time-series generator (juniper-data#179 §A).

Pins the additive 3-D regression contract, the discrete-Euler known-answer
trajectory off the constant history, boundedness on the chaotic attractor,
determinism, the transient discard, parameter validation, and the task-type /
sequence metadata dispatch.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_mackey_glass_generator.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import numpy as np
import pytest

from juniper_data.core.meta import compute_shape_meta, derive_sequence_meta
from juniper_data.generators.mackey_glass import MackeyGlassGenerator, MackeyGlassParams, get_schema

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _assert_regular_sequence_contract(arrays: dict, *, lookback: int, sample_dt: float, horizon: int) -> None:
    """Shared assertions for the regular-Δt 3-D regression NPZ contract."""
    for split in ("train", "test", "full"):
        for key in ("X", "y", "dt", "target_dt", "observed_mask"):
            assert f"{key}_{split}" in arrays, f"missing {key}_{split}"

    xf = arrays["X_full"]
    n_windows = xf.shape[0]
    assert xf.ndim == 3 and xf.shape[1:] == (lookback, 1)
    assert xf.dtype == np.float32
    assert arrays["y_full"].shape == (n_windows, 1) and arrays["y_full"].dtype == np.float32

    dt = arrays["dt_full"]
    assert dt.shape == (n_windows, lookback) and dt.dtype == np.float32
    assert np.all(dt[:, 0] == 0)
    np.testing.assert_allclose(dt[:, 1:], np.float32(sample_dt))
    assert arrays["target_dt_full"].shape == (n_windows,)
    np.testing.assert_allclose(arrays["target_dt_full"], np.float32(horizon * sample_dt))

    mask = arrays["observed_mask_full"]
    assert mask.shape == (n_windows, lookback) and mask.dtype == np.uint8
    assert np.all(mask == 1)

    assert n_windows == arrays["X_train"].shape[0] + arrays["X_test"].shape[0]
    np.testing.assert_array_equal(arrays["X_full"], np.concatenate([arrays["X_train"], arrays["X_test"]]))

    shape_meta = compute_shape_meta(arrays, "regression")
    assert shape_meta["n_features"] == 1
    assert shape_meta["n_classes"] is None and shape_meta["class_distribution"] is None
    seq_meta = derive_sequence_meta(arrays, "steps")
    assert seq_meta["sequence"] is True and seq_meta["lookback"] == lookback and seq_meta["time_unit"] == "steps"


class TestMackeyGlassGenerator:
    """End-to-end behavior of MackeyGlassGenerator.generate()."""

    def test_contract_and_metadata(self) -> None:
        arrays = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=400, lookback=32, horizon=1, discard=100, seed=0))
        _assert_regular_sequence_contract(arrays, lookback=32, sample_dt=1.0, horizon=1)

    def test_euler_known_answer_from_constant_history(self) -> None:
        # discard=0 => the series starts at the first Euler step off the x0 history.
        params = MackeyGlassParams(n_steps=100, lookback=16, horizon=1, discard=0, x0=0.5, beta=0.2, gamma=0.1, n_exp=10.0, tau=17.0, sample_dt=1.0, seed=0)
        arrays = MackeyGlassGenerator.generate(params)

        tau_steps = 17
        traj = np.full(tau_steps + 1 + params.n_steps, 0.5, dtype=np.float64)
        for k in range(tau_steps, tau_steps + params.n_steps):
            delayed = traj[k - tau_steps]
            traj[k + 1] = traj[k] + 1.0 * (0.2 * delayed / (1.0 + delayed**10) - 0.1 * traj[k])
        expected = traj[tau_steps + 1 : tau_steps + 1 + params.n_steps].astype(np.float32)

        np.testing.assert_allclose(arrays["X_full"][0, :, 0], expected[:16], atol=1e-5)
        ends = np.arange(15, params.n_steps - 1)  # lookback - 1 .. T - 1 - horizon
        np.testing.assert_allclose(arrays["y_full"][:, 0], expected[ends + 1], atol=1e-5)

    def test_bounded_on_attractor(self) -> None:
        arrays = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=1000, lookback=32, discard=250, seed=0))
        x = arrays["X_full"]
        assert np.all(np.isfinite(x))
        # The canonical regime stays in roughly [0.2, 1.4]; bound generously.
        assert x.min() > 0.0 and x.max() < 2.0

    def test_determinism(self) -> None:
        first = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=300, lookback=24, seed=0))
        second = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=300, lookback=24, seed=0))
        np.testing.assert_array_equal(first["X_full"], second["X_full"])

    def test_discard_shifts_start(self) -> None:
        no_drop = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=300, lookback=24, discard=0, seed=0))
        dropped = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=300, lookback=24, discard=50, seed=0))
        assert no_drop["X_full"].shape == dropped["X_full"].shape
        assert not np.allclose(no_drop["X_full"][0, :, 0], dropped["X_full"][0, :, 0])

    def test_short_series_rejected(self) -> None:
        with pytest.raises(ValueError):
            MackeyGlassParams(n_steps=10, lookback=8, horizon=3)

    def test_get_schema_has_dynamics_fields(self) -> None:
        schema = get_schema()
        for field in ("tau", "beta", "gamma", "n_exp", "x0", "discard", "lookback", "horizon"):
            assert field in schema["properties"], f"schema missing {field}"
