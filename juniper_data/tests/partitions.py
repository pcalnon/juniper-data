"""Whole-dataset views for tests, now that ``*_full`` has left the contract.

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     partitions.py
# Author:        Paul Calnon
# License:       MIT License

Decision 11 (design §9.5) removed ``X_full`` / ``y_full`` and their per-key siblings
from the NPZ contract. Roughly 180 assertions read one of them, and nearly all were
asking the same question: *what does the whole dataset look like?* -- its shape, its
dtype, its class balance.

That question is still legitimate; only its answer moved from an array the producer
shipped to a concatenation of the partitions. :func:`whole` is that concatenation, in
contract order, so those assertions keep testing what they were testing.

**It is not a compatibility shim.** Nothing in ``juniper_data`` imports this; it exists
so a test can say "the whole dataset" without re-deriving it inline in twenty files,
and so the ORDER is written down once. Production code that needs the same view should
concatenate explicitly at the point of use, where the reader can see which partitions
are involved.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

#: Contract order. Concatenating in any other order changes row indices and would make
#: a class-distribution assertion pass while a row-order one silently drifts.
PARTITIONS: tuple[str, ...] = ("train", "val", "test")


def whole(arrays: Mapping[str, np.ndarray], stem: str = "X", partitions: Iterable[str] = PARTITIONS) -> np.ndarray:
    """Concatenate one key's partitions into the whole-dataset view.

    Args:
        arrays: the generator's output dict, or a loaded NPZ mapping.
        stem: key stem -- ``"X"``, ``"y"``, ``"y_reg"``, ``"dt"``, and so on.
        partitions: which partitions to include, in order. Defaults to all three.

    Returns:
        The concatenation along axis 0. Absent partitions are skipped, so this works on
        a two-partition legacy artifact as well as a three-way one.

    Raises:
        KeyError: if no partition of ``stem`` is present at all -- an empty result would
            otherwise read as "the dataset is empty" rather than "the key is missing".
    """
    blocks = [np.asarray(arrays[f"{stem}_{p}"]) for p in partitions if f"{stem}_{p}" in arrays]
    if not blocks:
        raise KeyError(f"no {stem}_* partitions present; have {sorted(arrays)}")
    return np.concatenate(blocks, axis=0)


def n_rows(arrays: Mapping[str, np.ndarray], stem: str = "X") -> int:
    """Total row count across the partitions, without building the concatenation."""
    return sum(int(np.asarray(arrays[f"{stem}_{p}"]).shape[0]) for p in PARTITIONS if f"{stem}_{p}" in arrays)
