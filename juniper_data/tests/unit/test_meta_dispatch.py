"""Unit tests for task-type-dispatched dataset metadata (WS-1 / juniper-data#168).

Pins ``compute_shape_meta``: classification artifacts (2-D and 3-D) still derive
``n_classes`` + ``class_distribution`` from the one-hot ``y`` and report the
trailing feature axis; regression artifacts make no classification assumption
(both fields ``None``) and never touch a one-hot label.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_meta_dispatch.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import numpy as np
import pytest

from juniper_data.core.meta import compute_shape_meta, derive_sequence_meta, pop_scaling_meta

pytestmark = [pytest.mark.unit]


def _onehot(labels: list[int], n_classes: int) -> np.ndarray:
    oh = np.zeros((len(labels), n_classes), dtype=np.float32)
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


def test_classification_2d_back_compat():
    arrays = {
        "X_train": np.zeros((6, 4), np.float32),
        "X_test": np.zeros((2, 4), np.float32),
        "y_train": _onehot([0, 1, 0, 1, 1, 0], 2),
        "y_test": _onehot([0, 1], 2),
    }
    m = compute_shape_meta(arrays, "classification")
    assert m["n_samples"] == 8
    assert m["n_train"] == 6 and m["n_test"] == 2
    assert m["n_features"] == 4
    assert m["n_classes"] == 2
    assert m["class_distribution"] == {"0": 4, "1": 4}


def test_classification_3d_uses_trailing_feature_axis():
    # X is (W, L, F); n_features must be F (last axis), NOT the lookback L.
    arrays = {
        "X_train": np.zeros((5, 7, 3), np.float32),
        "X_test": np.zeros((2, 7, 3), np.float32),
        "y_train": _onehot([0, 1, 0, 1, 0], 2),
        "y_test": _onehot([1, 0], 2),
    }
    m = compute_shape_meta(arrays, "classification")
    assert m["n_features"] == 3  # F, not L == 7
    assert m["n_classes"] == 2
    assert m["class_distribution"] == {"0": 4, "1": 3}


def test_regression_2d_makes_no_classification_assumption():
    arrays = {
        "X_train": np.zeros((6, 4), np.float32),
        "X_test": np.zeros((2, 4), np.float32),
        # Deliberately NO one-hot y_train/y_test — only a regression target.
        "y_reg_train": np.zeros((6, 1), np.float32),
        "y_reg_test": np.zeros((2, 1), np.float32),
    }
    m = compute_shape_meta(arrays, "regression")
    assert m["n_features"] == 4
    assert m["n_samples"] == 8
    assert m["n_classes"] is None
    assert m["class_distribution"] is None


def test_regression_3d_sequence():
    arrays = {
        "X_train": np.zeros((5, 7, 3), np.float32),
        "X_test": np.zeros((2, 7, 3), np.float32),
        "y_reg_train": np.zeros((5, 1), np.float32),
        "y_reg_test": np.zeros((2, 1), np.float32),
    }
    m = compute_shape_meta(arrays, "regression")
    assert m["n_features"] == 3
    assert m["n_classes"] is None
    assert m["class_distribution"] is None


def test_classification_prefers_y_full_for_distribution():
    arrays = {
        "X_train": np.zeros((2, 4), np.float32),
        "X_test": np.zeros((1, 4), np.float32),
        "y_train": _onehot([0, 1], 2),
        "y_test": _onehot([1], 2),
        "y_full": _onehot([0, 0, 1, 1, 1], 2),
    }
    m = compute_shape_meta(arrays, "classification")
    assert m["class_distribution"] == {"0": 2, "1": 3}


def test_default_task_type_is_classification():
    arrays = {
        "X_train": np.zeros((3, 2), np.float32),
        "X_test": np.zeros((1, 2), np.float32),
        "y_train": _onehot([0, 1, 1], 2),
        "y_test": _onehot([0], 2),
    }
    m = compute_shape_meta(arrays)  # no task_type argument
    assert m["n_classes"] == 2
    assert m["class_distribution"] is not None


def test_derive_sequence_meta_tabular_2d():
    arrays = {"X_train": np.zeros((6, 4), np.float32), "X_test": np.zeros((2, 4), np.float32)}
    m = derive_sequence_meta(arrays, time_unit="calendar_days")
    assert m["sequence"] is False
    assert m["lookback"] is None
    assert m["time_unit"] is None  # time_unit echoed only for sequence artifacts


def test_derive_sequence_meta_3d_sequence():
    arrays = {"X_train": np.zeros((5, 7, 3), np.float32), "X_test": np.zeros((2, 7, 3), np.float32)}
    m = derive_sequence_meta(arrays, time_unit="calendar_days")
    assert m["sequence"] is True
    assert m["lookback"] == 7
    assert m["time_unit"] == "calendar_days"


def test_derive_sequence_meta_3d_with_empty_train_split():
    # Rank is well-defined for an empty split; lookback still derived from X_train.
    arrays = {"X_train": np.zeros((0, 7, 3), np.float32), "X_test": np.zeros((2, 7, 3), np.float32)}
    m = derive_sequence_meta(arrays)
    assert m["sequence"] is True
    assert m["lookback"] == 7
    assert m["time_unit"] is None


def test_pop_scaling_meta_extracts_and_cleans():
    # The reserved "scaling" channel key is popped (so the dict stays array-only) and mapped out.
    arrays = {
        "X_train": np.zeros((2, 3, 1), np.float32),
        "scaling": {"dt_scaling": {"method": "identity"}, "target_scaling": {"y": {"method": "standardize", "mean": 1.0, "std": 2.0}}},
    }
    scaling = pop_scaling_meta(arrays)
    assert scaling["dt_scaling"] == {"method": "identity"}
    assert scaling["target_scaling"]["y"]["method"] == "standardize"
    assert "scaling" not in arrays


def test_pop_scaling_meta_absent_returns_none():
    arrays = {"X_train": np.zeros((2, 3, 1), np.float32)}
    scaling = pop_scaling_meta(arrays)
    assert scaling == {"dt_scaling": None, "target_scaling": None}
    assert "scaling" not in arrays
