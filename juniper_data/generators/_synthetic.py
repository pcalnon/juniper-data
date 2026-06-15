"""Shared parameter base + windowing glue for the synthetic time-series generators.

The ``multi_sine`` / ``mackey_glass`` / ``ar_p`` generators are the recurse CLI
"hello-world" datasets ([OQ-5]): deterministic, seeded, offline, numpy-only
real-valued **regression** sequences (``task_type="regression"``). They share the
windowing knobs (``n_steps`` raw length, ``lookback`` / ``horizon`` /
``sample_dt`` / ``train_ratio``) and differ only in the process that produces the
raw series; each generator subclasses :class:`SyntheticSequenceParams`, adds its
process-specific fields, builds a ``(T, F)`` series, and hands it to
:func:`build_sequence_arrays` (a thin wrapper over
``juniper_data.generators._sequence.window_regular_series``).

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

import numpy as np
from pydantic import BaseModel, Field, model_validator

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

    @model_validator(mode="after")
    def _validate_window_budget(self) -> SyntheticSequenceParams:
        """Ensure the raw series yields at least two windows (W = T - L - h + 1 >= 2)."""
        n_windows = self.n_steps - self.lookback - self.horizon + 1
        if n_windows < 2:
            raise ValueError(f"n_steps={self.n_steps} with lookback={self.lookback} and horizon={self.horizon} yields {n_windows} window(s); need >= 2 (n_steps >= lookback + horizon + 1).")
        return self


def build_sequence_arrays(series: np.ndarray, params: SyntheticSequenceParams) -> dict[str, np.ndarray]:
    """Window a raw ``(T, F)`` synthetic series into the 3-D sequence NPZ arrays.

    A one-line wrapper that applies the shared windowing knobs from ``params`` to
    :func:`~juniper_data.generators._sequence.window_regular_series`, so every
    synthetic generator produces the byte-identical additive 3-D contract
    (``{X, y, dt, target_dt, observed_mask}_{train,test,full}``).
    """
    return window_regular_series(
        series,
        lookback=params.lookback,
        horizon=params.horizon,
        sample_dt=params.sample_dt,
        train_ratio=params.train_ratio,
    )
