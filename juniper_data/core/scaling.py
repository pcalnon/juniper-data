"""Advisory standardization descriptors for the dt / target scaling meta channel.

WS-4 follow-up (juniper-data#179 §A; spec: juniper-ml Δt note §6.5). A generator
may report *how* its per-step ``dt`` and its regression target SHOULD be
standardized. The NPZ keeps RAW values — every contract invariant (e.g.
``dt[:, 0] == 0``) stays intact — so these descriptors are **advisory**: a
consumer (e.g. the LMU loader) standardizes at ingestion and denormalizes for
metrics, using the persisted ``mean`` / ``std``.

A descriptor is a small JSON-safe dict carried in :class:`DatasetMeta`:

* ``{"method": "identity"}`` — no transform (report values as-is); or
* ``{"method": "standardize", "mean": .., "std": .., "min": .., "max": ..}``.

``standardize`` / ``inverse_standardize`` are exact inverses (to float precision),
which the denorm round-trip test pins; they no-op on an ``identity`` descriptor.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     scaling.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from typing import Any

import numpy as np

#: Reserved key a generator MAY include in its ``generate()`` return dict to pass
#: scaling descriptors to the route (popped before NPZ persistence; see
#: ``juniper_data.core.meta.pop_scaling_meta``). Its value is a plain dict, NOT an
#: ndarray: ``{"dt_scaling": <desc|None>, "target_scaling": <{key: desc}|None>}``.
SCALING_META_KEY = "scaling"

SCALING_METHOD_IDENTITY = "identity"
SCALING_METHOD_STANDARDIZE = "standardize"

#: Below this, a sample std is treated as zero (constant data, e.g. a regular-Δt
#: ``dt``) and clamped to 1.0 so the descriptor stays invertible (no divide-by-0).
_STD_EPS = 1e-8


def standardize_descriptor(arr: np.ndarray) -> dict[str, Any]:
    """Build a ``standardize`` descriptor (mean / std / min / max) for ``arr``.

    ``std`` is clamped up to ``1.0`` when (near-)zero, so a constant array yields a
    usable, invertible descriptor (standardizing then inverting is a no-op shift to
    the mean) instead of a divide-by-zero. All stats are plain Python floats so the
    descriptor is JSON-safe for the ``.meta.json`` sidecar.

    Args:
        arr: any numeric array; flattened before computing the statistics.

    Returns:
        ``{"method": "standardize", "mean": .., "std": .., "min": .., "max": ..}``.
    """
    flat = np.asarray(arr, dtype=np.float64).ravel()
    if flat.size == 0:
        return {"method": SCALING_METHOD_STANDARDIZE, "mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0}
    std = float(flat.std())
    return {
        "method": SCALING_METHOD_STANDARDIZE,
        "mean": float(flat.mean()),
        "std": std if std >= _STD_EPS else 1.0,
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def standardize(arr: np.ndarray, descriptor: dict[str, Any]) -> np.ndarray:
    """Apply a descriptor: ``(arr - mean) / std`` for ``standardize``; a no-op otherwise.

    Returns a float32 array (matching the NPZ dtype). An ``identity`` (or unknown)
    descriptor returns ``arr`` unchanged (as float32).
    """
    if descriptor.get("method") != SCALING_METHOD_STANDARDIZE:
        return np.asarray(arr, dtype=np.float32)
    return ((np.asarray(arr, dtype=np.float64) - descriptor["mean"]) / descriptor["std"]).astype(np.float32)


def inverse_standardize(arr: np.ndarray, descriptor: dict[str, Any]) -> np.ndarray:
    """Invert a descriptor: ``arr * std + mean`` for ``standardize``; a no-op otherwise.

    The exact inverse of :func:`standardize` (to float precision). Returns float32.
    """
    if descriptor.get("method") != SCALING_METHOD_STANDARDIZE:
        return np.asarray(arr, dtype=np.float32)
    return (np.asarray(arr, dtype=np.float64) * descriptor["std"] + descriptor["mean"]).astype(np.float32)
