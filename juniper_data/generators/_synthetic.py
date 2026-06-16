"""Shared parameter base + windowing glue for the synthetic time-series generators.

The ``multi_sine`` / ``mackey_glass`` / ``ar_p`` generators are the recurse CLI
"hello-world" datasets ([OQ-5]): deterministic, seeded, offline, numpy-only
real-valued **regression** sequences (``task_type="regression"``). They share the
windowing knobs (``n_steps`` raw length, ``lookback`` / ``horizon`` /
``sample_dt`` / ``train_ratio``) and differ only in the process that produces the
raw series; each generator subclasses :class:`SyntheticSequenceParams`, adds its
process-specific fields, builds a ``(T, F)`` series, and hands it to
:func:`build_sequence_arrays` (which windows it and attaches the advisory
``scaling`` meta channel).

Unlike ``equities`` / ``equities_seq`` these have no optional extra -- they are
pure numpy, so they are the zero-dependency smoke datasets for the 3-D sequence
contract. See ``juniper-ml/notes/JUNIPER_RECURSE_MODEL_DESIGN_AND_PLAN_2026-05-31.md``
([OQ-5]) and the WS-4 follow-up tracker juniper-data#179 §A.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     _synthetic.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from juniper_data.core.scaling import (
    SCALING_META_KEY,
    SCALING_METHOD_IDENTITY,
    SCALING_METHOD_STANDARDIZE,
    standardize_descriptor,
)
from juniper_data.generators._sequence import window_regular_series

#: Declared in the GENERATOR_REGISTRY as ``time_unit`` so the dataset route echoes
#: it back for sequence artifacts. The synthetics index time by sample step, so
#: ``sample_dt`` is measured in these dimensionless steps.
SYNTHETIC_TIME_UNIT = "steps"


class SyntheticSequenceParams(BaseModel):
    """Common windowing parameters for the synthetic time-series generators.

    Subclasses add the process-specific fields (frequencies, delay, AR
    coefficients, ...). The raw series has ``n_steps`` points sampled
    ``sample_dt`` apart and is windowed into fixed-length ``lookback`` sequences
    predicting ``horizon`` steps ahead; every field is deterministic given
    ``seed``. The number of windows is ``W = n_steps - lookback - horizon + 1``,
    which the model validator pins at ``>= 2`` so both splits are non-empty.
    """

    n_steps: int = Field(default=2000, ge=8, le=1_000_000, description="Length T of the raw series before windowing.")
    lookback: int = Field(default=32, ge=2, le=512, description="Window length L (number of steps per sequence window).")
    horizon: int = Field(default=1, ge=1, le=512, description="Forecast horizon h (the target is h steps after the window end).")
    sample_dt: float = Field(default=1.0, gt=0, description="Constant per-step elapsed time (the regular Δt), measured in the generator's time_unit.")
    train_ratio: float = Field(default=0.8, gt=0, le=1, description="Fraction of the earliest windows used for training; the test split is every later window.")
    seed: int = Field(default=0, ge=0, description="Random seed for reproducibility (the synthetics are deterministic given the seed).")
    scaling: Literal["identity", "standardize"] = Field(default="identity", description="Advisory scaling reported in DatasetMeta: 'identity' (raw, no transform) or 'standardize' (train-split mean/std/min/max of dt + target). The NPZ stays RAW either way.")

    @model_validator(mode="after")
    def _validate_window_budget(self) -> SyntheticSequenceParams:
        """Ensure the raw series yields at least two windows (W = T - L - h + 1 >= 2)."""
        n_windows = self.n_steps - self.lookback - self.horizon + 1
        if n_windows < 2:
            raise ValueError(f"n_steps={self.n_steps} with lookback={self.lookback} and horizon={self.horizon} yields {n_windows} window(s); need >= 2 (n_steps >= lookback + horizon + 1).")
        return self


def build_sequence_arrays(series: np.ndarray, params: SyntheticSequenceParams) -> dict[str, Any]:
    """Window a raw ``(T, F)`` regular-Δt synthetic series into the 3-D NPZ arrays.

    Applies the shared windowing knobs from ``params`` to
    :func:`~juniper_data.generators._sequence.window_regular_series`, then attaches
    the advisory scaling channel (:func:`attach_scaling`) -- so every regular-Δt
    synthetic emits the same additive 3-D contract
    (``{X, y, dt, target_dt, observed_mask}_{train,test,full}``) plus the reserved
    ``"scaling"`` meta key.
    """
    arrays = window_regular_series(
        series,
        lookback=params.lookback,
        horizon=params.horizon,
        sample_dt=params.sample_dt,
        train_ratio=params.train_ratio,
    )
    return attach_scaling(arrays, params.scaling)


def attach_scaling(arrays: dict[str, Any], scaling: str) -> dict[str, Any]:
    """Attach the reserved ``"scaling"`` channel key (advisory dt + target descriptors).

    For ``"standardize"`` the descriptors are fit on the TRAIN split only (no test
    leakage): ``dt`` over the real per-step gaps (``dt_train[:, 1:]``, excluding the
    ``dt[:, 0] == 0`` sentinels) and the target over ``y_train``. The NPZ arrays stay
    RAW; these are recommended-transform metadata, not applied transforms
    (juniper-data#179 §A; Δt note §6.5). ``"identity"`` reports a no-op descriptor;
    ``target_scaling`` is keyed by the target-array name (``"y"``).
    """
    dt_desc: dict[str, Any]
    target_desc: dict[str, Any]
    if scaling == SCALING_METHOD_STANDARDIZE:
        dt_desc = standardize_descriptor(arrays["dt_train"][:, 1:])
        target_desc = standardize_descriptor(arrays["y_train"])
    else:
        dt_desc = {"method": SCALING_METHOD_IDENTITY}
        target_desc = {"method": SCALING_METHOD_IDENTITY}
    arrays[SCALING_META_KEY] = {"dt_scaling": dt_desc, "target_scaling": {"y": target_desc}}
    return arrays
