"""Shape and task-type metadata derivation for dataset artifacts.

The dataset-creation route builds a :class:`~juniper_data.core.models.DatasetMeta`
from the raw NPZ array dict a generator returns. This module isolates the
array-shape + class-distribution derivation so it can dispatch on ``task_type``
and be unit-tested without standing up the FastAPI route.

Dispatch rules (WS-1 / juniper-data#168, the "lean" contract):

* ``n_features`` is the TRAILING array axis, so both 2-D tabular ``(N, F)`` and
  3-D sequence ``(W, L, F)`` artifacts report the true feature count (for 2-D,
  ``shape[-1] == shape[1]``, so this is byte-identical to the previous behaviour).
* ``classification`` artifacts derive ``n_classes`` + ``class_distribution`` from
  the one-hot ``y`` via ``argmax`` over the class axis, exactly as before.
* non-classification artifacts (e.g. ``regression``) leave ``n_classes`` and
  ``class_distribution`` as ``None`` — no one-hot / argmax assumption is made, so
  a pure-regression or time-series artifact need not fake a one-hot label.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     meta.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from typing import Any

import numpy as np

#: The only task type that populates n_classes / class_distribution.
TASK_TYPE_CLASSIFICATION = "classification"


def compute_shape_meta(
    arrays: dict[str, np.ndarray],
    task_type: str = TASK_TYPE_CLASSIFICATION,
) -> dict[str, Any]:
    """Derive shape + (classification-only) class-distribution metadata.

    Args:
        arrays: NPZ array dict with at least ``X_train`` / ``X_test`` and, for
            classification, one-hot ``y_train`` / ``y_test`` (``y_full`` optional).
        task_type: dataset task type; only ``"classification"`` populates
            ``n_classes`` and ``class_distribution``.

    Returns:
        Dict with ``n_samples``, ``n_features``, ``n_train``, ``n_test``,
        ``n_classes`` (``int | None``), and ``class_distribution``
        (``dict[str, int] | None``).
    """
    x_train = arrays["X_train"]
    x_test = arrays["X_test"]
    n_train = len(x_train)
    n_test = len(x_test)
    n_samples = n_train + n_test
    # Feature count is the trailing axis for both (N, F) and (W, L, F).
    n_features = int(x_train.shape[-1]) if n_train > 0 else 2

    n_classes: int | None = None
    class_distribution: dict[str, int] | None = None
    if task_type == TASK_TYPE_CLASSIFICATION:
        n_classes, class_distribution = _classification_meta(arrays, n_train, n_test)

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_train": n_train,
        "n_test": n_test,
        "n_classes": n_classes,
        "class_distribution": class_distribution,
    }


def _classification_meta(
    arrays: dict[str, np.ndarray],
    n_train: int,
    n_test: int,
) -> tuple[int, dict[str, int]]:
    """Class count + per-class sample counts from one-hot targets (argmax)."""
    if n_train > 0:
        n_classes = int(arrays["y_train"].shape[1])
    elif n_test > 0:
        n_classes = int(arrays["y_test"].shape[1])
    else:
        n_classes = 2

    y_full = arrays.get("y_full")
    if y_full is None:
        y_full = np.vstack([arrays["y_train"], arrays["y_test"]])
    class_labels = np.argmax(y_full, axis=1)
    unique, counts = np.unique(class_labels, return_counts=True)
    class_distribution = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    return n_classes, class_distribution
