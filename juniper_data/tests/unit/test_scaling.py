"""Unit tests for core/scaling.py — advisory standardization descriptors (juniper-data#179 §A.6.5).

Pins the descriptor statistics (incl. the constant-array std guard and JSON-safety),
and that ``standardize`` / ``inverse_standardize`` are exact inverses and no-op on an
``identity`` descriptor.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_scaling.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import json

import numpy as np
import pytest

from juniper_data.core.scaling import (
    SCALING_METHOD_IDENTITY,
    SCALING_METHOD_STANDARDIZE,
    inverse_standardize,
    standardize,
    standardize_descriptor,
)

pytestmark = [pytest.mark.unit]


class TestStandardizeDescriptor:
    """standardize_descriptor() statistics + safety."""

    def test_stats(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        desc = standardize_descriptor(arr)
        assert desc["method"] == SCALING_METHOD_STANDARDIZE
        assert desc["mean"] == pytest.approx(2.5)
        assert desc["std"] == pytest.approx(float(np.asarray([1, 2, 3, 4], dtype=np.float64).std()))
        assert desc["min"] == pytest.approx(1.0) and desc["max"] == pytest.approx(4.0)

    def test_constant_array_guards_std_to_one(self) -> None:
        # A constant array has std 0; the descriptor clamps it to 1.0 so it stays invertible.
        desc = standardize_descriptor(np.full(10, 5.0))
        assert desc["std"] == 1.0 and desc["mean"] == pytest.approx(5.0)

    def test_empty_array_is_safe(self) -> None:
        desc = standardize_descriptor(np.empty(0))
        assert desc["method"] == SCALING_METHOD_STANDARDIZE and desc["std"] == 1.0

    def test_descriptor_is_json_safe(self) -> None:
        desc = standardize_descriptor(np.arange(5.0))
        json.dumps(desc)  # plain Python floats — must not raise
        assert all(isinstance(desc[k], float) for k in ("mean", "std", "min", "max"))


class TestStandardizeRoundTrip:
    """standardize / inverse_standardize are exact inverses; identity is a no-op."""

    def test_standardize_then_inverse_recovers(self) -> None:
        rng = np.random.default_rng(0)
        arr = rng.normal(3.0, 2.0, (20, 4)).astype(np.float32)
        desc = standardize_descriptor(arr)
        z = standardize(arr, desc)
        assert z.mean() == pytest.approx(0.0, abs=1e-5)
        assert z.std() == pytest.approx(1.0, abs=1e-4)
        np.testing.assert_allclose(inverse_standardize(z, desc), arr, rtol=1e-4, atol=1e-4)

    def test_identity_is_noop(self) -> None:
        arr = np.arange(6.0).reshape(2, 3).astype(np.float32)
        idesc = {"method": SCALING_METHOD_IDENTITY}
        np.testing.assert_array_equal(standardize(arr, idesc), arr)
        np.testing.assert_array_equal(inverse_standardize(arr, idesc), arr)

    def test_constant_array_round_trips(self) -> None:
        arr = np.full((5, 2), 7.0, dtype=np.float32)
        desc = standardize_descriptor(arr)
        np.testing.assert_allclose(inverse_standardize(standardize(arr, desc), desc), arr, atol=1e-5)
