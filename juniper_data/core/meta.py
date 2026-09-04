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

from juniper_data.core.limits import TRUNCATION_META_KEY
from juniper_data.core.scaling import SCALING_META_KEY

#: The only task type that populates n_classes / class_distribution.
TASK_TYPE_CLASSIFICATION = "classification"


def compute_shape_meta(
    arrays: dict[str, np.ndarray],
    task_type: str = TASK_TYPE_CLASSIFICATION,
) -> dict[str, Any]:
    """Derive shape + (classification-only) class-distribution metadata.

    Args:
        arrays: NPZ array dict with at least ``X_train`` / ``X_test`` and, for
            classification, one-hot ``y_train`` / ``y_test``. ``X_val`` / ``y_val``
            and ``y_full`` are optional.
        task_type: dataset task type; only ``"classification"`` populates
            ``n_classes`` and ``class_distribution``.

    Returns:
        Dict with ``n_samples``, ``n_features``, ``n_train``, ``n_val``, ``n_test``,
        ``n_classes`` (``int | None``), and ``class_distribution``
        (``dict[str, int] | None``).
    """
    x_train = arrays["X_train"]
    x_test = arrays["X_test"]
    n_train = len(x_train)
    n_test = len(x_test)
    # `X_val` is presence-conditional: a two-partition artifact predating the
    # third partition simply has none, and reports 0 rather than failing.
    n_val = len(arrays["X_val"]) if "X_val" in arrays else 0
    n_samples = n_train + n_val + n_test
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
        "n_val": n_val,
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
        # Every partition present must be stacked. Omitting `y_val` here would
        # under-count the distribution silently -- and would do so ONLY on
        # artifacts without `y_full`, which is precisely what decision 11 makes
        # the normal case.
        parts = [arrays["y_train"]]
        if "y_val" in arrays:
            parts.append(arrays["y_val"])
        parts.append(arrays["y_test"])
        y_full = np.vstack(parts)
    class_labels = np.argmax(y_full, axis=1)
    unique, counts = np.unique(class_labels, return_counts=True)
    class_distribution = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    return n_classes, class_distribution


def derive_sequence_meta(arrays: dict[str, np.ndarray], time_unit: str | None = None) -> dict[str, Any]:
    """Derive sequence-ness + lookback from the X rank (WS-1 / juniper-data#168).

    A 3-D ``X`` of shape ``(W, L, F)`` is a sequence artifact with lookback ``L``;
    a 2-D ``X`` is tabular. ``time_unit`` (generator-declared, e.g.
    ``"calendar_days"``) is echoed back only for sequence artifacts. The X rank is
    well-defined even for an empty split, so ``X_train`` is preferred with
    ``X_test`` as a fallback.

    Args:
        arrays: NPZ array dict (uses ``X_train``, falling back to ``X_test``).
        time_unit: the time unit of the per-step ``dt`` / ``t`` channels, or None.

    Returns:
        Dict with ``sequence`` (bool), ``lookback`` (``int | None``), and
        ``time_unit`` (``str | None``; None unless this is a sequence artifact).
    """
    ref = arrays.get("X_train")
    if ref is None:
        ref = arrays.get("X_test")
    if ref is None or ref.ndim != 3:
        return {"sequence": False, "lookback": None, "time_unit": None}
    return {"sequence": True, "lookback": int(ref.shape[1]), "time_unit": time_unit}


def pop_scaling_meta(arrays: dict[str, Any]) -> dict[str, Any]:
    """Pop the reserved scaling channel key from a generator's return dict.

    A generator MAY include a single reserved ``"scaling"`` entry (a plain dict,
    NOT an ndarray) carrying advisory ``dt_scaling`` / ``target_scaling``
    descriptors -- metadata that is not derivable from the final arrays (the
    standardization stats). This removes it from ``arrays`` (so the dict stays
    array-only for checksumming + NPZ persistence) and returns the two
    descriptors, each ``None`` when the generator did not report scaling.

    Args:
        arrays: the dict a generator's ``generate()`` returned; mutated in place
            to drop the reserved ``"scaling"`` key when present.

    Returns:
        ``{"dt_scaling": <dict | None>, "target_scaling": <dict | None>}``.
    """
    scaling = arrays.pop(SCALING_META_KEY, None) or {}
    return {"dt_scaling": scaling.get("dt_scaling"), "target_scaling": scaling.get("target_scaling")}


def pop_truncation_meta(arrays: dict[str, Any]) -> dict[str, Any] | None:
    """Pop the reserved truncation channel key from a generator's return dict.

    Mirrors :func:`pop_scaling_meta`. A generator that bounded its input MAY
    include a single reserved ``"truncation"`` entry (a plain dict, NOT an
    ndarray) describing what was cut. This removes it from ``arrays`` -- so the
    dict stays array-only for checksumming and NPZ persistence -- and returns
    it for storage on ``DatasetMeta``.

    Returning ``None`` rather than an empty dict is deliberate: ``DatasetMeta``
    stores ``None`` for "complete", so a reader can test the field's presence
    alone. An empty dict would be truthy-adjacent and invite
    ``if meta.truncation:`` to be read as "was it truncated" in one place and
    "did the generator report" in another.

    Args:
        arrays: the dict a generator's ``generate()`` returned; mutated in place
            to drop the reserved ``"truncation"`` key when present.

    Returns:
        The truncation descriptor, or ``None`` when nothing was truncated.
    """
    return arrays.pop(TRUNCATION_META_KEY, None) or None
