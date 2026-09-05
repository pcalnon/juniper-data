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
from juniper_data.core.models import DatasetMeta

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


def test_val_partition_absent_reports_zero():
    """A two-partition artifact predating the third partition still derives cleanly."""
    arrays = {
        "X_train": np.zeros((6, 4), np.float32),
        "y_train": _onehot([0, 1, 0, 1, 0, 1], 2),
        "X_test": np.zeros((2, 4), np.float32),
        "y_test": _onehot([0, 1], 2),
    }

    meta = compute_shape_meta(arrays)

    assert meta["n_val"] == 0
    assert meta["n_train"] == 6
    assert meta["n_test"] == 2
    assert meta["n_samples"] == 8


def test_val_partition_counted_in_shape_meta():
    """n_val is reported and n_samples spans all THREE partitions."""
    arrays = {
        "X_train": np.zeros((6, 4), np.float32),
        "y_train": _onehot([0, 1, 0, 1, 0, 1], 2),
        "X_val": np.zeros((3, 4), np.float32),
        "y_val": _onehot([0, 1, 0], 2),
        "X_test": np.zeros((2, 4), np.float32),
        "y_test": _onehot([0, 1], 2),
    }

    meta = compute_shape_meta(arrays)

    assert meta["n_train"] == 6
    assert meta["n_val"] == 3
    assert meta["n_test"] == 2
    assert meta["n_samples"] == 11, "n_samples must be train + val + test, not train + test"


def test_class_distribution_without_y_full_includes_val():
    """The y_full-less fallback must stack val too, or it under-counts silently.

    ``y_full`` is dropped from the contract by decision 11, so the fallback
    becomes the normal path -- and a fallback that stacks only train + test
    would omit an entire class here while still returning a well-formed dict.
    """
    arrays = {
        "X_train": np.zeros((2, 4), np.float32),
        "y_train": _onehot([0, 0], 2),
        "X_val": np.zeros((3, 4), np.float32),
        "y_val": _onehot([1, 1, 1], 2),
        "X_test": np.zeros((2, 4), np.float32),
        "y_test": _onehot([0, 0], 2),
    }

    meta = compute_shape_meta(arrays)

    # Omitting y_val would yield {"0": 4} -- class 1 missing entirely.
    assert meta["class_distribution"] == {"0": 4, "1": 3}
    assert sum(meta["class_distribution"].values()) == meta["n_samples"]


def test_class_distribution_prefers_y_full_when_present():
    """y_full still wins when the artifact carries it, unchanged from before."""
    arrays = {
        "X_train": np.zeros((2, 4), np.float32),
        "y_train": _onehot([0, 0], 2),
        "X_val": np.zeros((1, 4), np.float32),
        "y_val": _onehot([1], 2),
        "X_test": np.zeros((1, 4), np.float32),
        "y_test": _onehot([1], 2),
        "y_full": _onehot([0, 0, 1, 1], 2),
    }

    meta = compute_shape_meta(arrays)

    assert meta["class_distribution"] == {"0": 2, "1": 2}


def test_dataset_meta_n_val_is_defaulted():
    """R-3: a required n_val would make every stored .meta.json unreadable.

    Existing artifacts are loaded with ``DatasetMeta(**meta_dict)`` from JSON
    written before the third partition existed. The field must therefore carry a
    default, and the default must be 0 -- the honest count for an artifact with
    no validation rows.
    """
    field = DatasetMeta.model_fields["n_val"]

    assert field.is_required() is False, "n_val must be defaulted or legacy .meta.json cannot load"
    assert field.default == 0


# --------------------------------------------------------------------------------------
# Empty train partition -- `n_features` must still be the TRAILING axis.
#
# `train_ratio = 0.0` is explicitly permitted (`core/split.py:60` validates
# `0.0 <= train_ratio <= 1.0`; `:70` then rounds `n_train` to 0), and `compute_shape_meta`
# runs on every dataset create (`api/routes/datasets.py:292`), so before this fix a
# fabricated `n_features = 2` was PERSISTED and SERVED for every such artifact.
#
# Each case below returned 2 on the unfixed code, so these fail without the change; the two
# non-empty controls returned the right answer before AND after, and exist so the suite
# cannot pass merely by the helper returning `shape[-1]` unconditionally.
# --------------------------------------------------------------------------------------


def test_empty_train_2d_reads_the_trailing_axis_not_two():
    arrays = {
        "X_train": np.zeros((0, 5), np.float32),
        "X_test": np.zeros((3, 5), np.float32),
        "y_train": np.zeros((0, 1), np.float32),
        "y_test": np.zeros((3, 1), np.float32),
    }
    m = compute_shape_meta(arrays, "regression")
    assert m["n_train"] == 0
    assert m["n_features"] == 5, "an empty train partition still carries its true feature count"


def test_empty_train_3d_reads_the_trailing_axis_not_the_lookback():
    # The 3-D case is the sharper one: the old fallback reported 2, which is neither the
    # feature count (3) nor the lookback (7) -- a value present in neither axis.
    arrays = {
        "X_train": np.zeros((0, 7, 3), np.float32),
        "X_test": np.zeros((2, 7, 3), np.float32),
        "y_train": np.zeros((0, 1), np.float32),
        "y_test": np.zeros((2, 1), np.float32),
    }
    m = compute_shape_meta(arrays, "regression")
    assert m["n_features"] == 3, "F, not L == 7, and not the old hardcoded 2"


@pytest.mark.parametrize(("shape", "expected"), [((4, 5), 5), ((4, 7, 3), 3)])
def test_non_empty_train_is_unchanged(shape, expected):
    # Negative control: these passed before the fix too. If they ever fail, the fix has
    # widened beyond the empty-partition case it is scoped to.
    arrays = {
        "X_train": np.zeros(shape, np.float32),
        "X_test": np.zeros((2,) + shape[1:], np.float32),
        "y_train": np.zeros((shape[0], 1), np.float32),
        "y_test": np.zeros((2, 1), np.float32),
    }
    assert compute_shape_meta(arrays, "regression")["n_features"] == expected
