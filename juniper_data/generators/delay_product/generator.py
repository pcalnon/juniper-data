"""Core numpy-only delay-product (bilinear capacity) irregular-Δt synthetic generator.

Samples the continuous-time superposition ``x(t) = Σ_i A_i sin(2π f_i t + φ_i)
(+ noise)`` at NON-uniform times (cumulative jittered gaps) — identical sampling to
``irregular_sine`` — then defines the regression target as the BILINEAR PRODUCT of
two delayed in-window values, ``y = x(t − τ₁)·x(t − τ₂)`` with ``τ₁ = lag1`` and
``τ₂ = lag2`` integer step-delays kept strictly inside the lookback.

The product is a quadratic form in the (linear) LMU memory state, so a LINEAR
readout provably cannot fit it (its r² is bounded below 1) while a nonlinear
(random-Fourier-feature) readout can approximate it — making this the
capacity-demonstrating dataset for the DP-3 readout spectrum (juniper-ml
``notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md`` §8a).

It reuses ``window_timed_series`` for the leakage-safe windowing / splitting and
then OVERWRITES ``y`` with the in-window product: the windowed ``X`` IS
``values[win_idx]``, so the product is read directly from the emitted window
contents and can never reach an out-of-window sample (windowing-leakage safe by
construction). ``y_full == concatenate([y_train, y_val, y_test])`` is preserved because
the product is computed identically per split from its own ``X`` block -- which is also
why the split list here must name EVERY partition: a split left out of it keeps the
forecast target ``window_timed_series`` emitted, silently pairing that split's windows
with a target from a different problem.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     generator.py
# Author:        Paul Calnon
# Version:       0.9.0
# License:       MIT License

from __future__ import annotations

import numpy as np

from juniper_data.generators._sequence import window_timed_series
from juniper_data.generators._synthetic import attach_scaling

from .params import DelayProductParams

VERSION = "2.0.0"

# Ranges for seeded-random component parameters when not given explicitly
# (mirrors ``irregular_sine`` / ``multi_sine``).
_FREQ_LOW, _FREQ_HIGH = 0.02, 0.15
_AMP_LOW, _AMP_HIGH = 0.5, 1.5


class DelayProductGenerator:
    """numpy-only generator for an irregular-Δt signal with a bilinear delay-product target.

    All methods are static (stateless, deterministic given ``seed``).
    """

    @staticmethod
    def generate(params: DelayProductParams) -> dict[str, np.ndarray]:
        """Generate the windowed irregular-Δt delay-product dataset.

        Returns the additive 3-D NPZ contract for train/val/test/full: ``X_{split}``
        ``(W, L, 1)`` (the irregularly-sampled signal windows), the regression target
        ``y_{split}`` ``(W, 1)`` (the in-window product
        ``x[·, L−1−lag1] · x[·, L−1−lag2]``), plus the non-uniform ``dt`` / variable
        ``target_dt`` / all-ones ``observed_mask``.

        Args:
            params: ``DelayProductParams`` (signal + sampling + delay + windowing knobs).
        """
        values, times = DelayProductGenerator._raw_series(params)
        arrays = window_timed_series(values, times, lookback=params.lookback, horizon=params.horizon, train_ratio=params.train_ratio, val_ratio=params.val_ratio)
        DelayProductGenerator._overwrite_with_product(arrays, lookback=params.lookback, lag1=params.lag1, lag2=params.lag2)
        return attach_scaling(arrays, params.scaling)

    @staticmethod
    def _overwrite_with_product(arrays: dict[str, np.ndarray], *, lookback: int, lag1: int, lag2: int) -> None:
        """Replace each split's ``y`` with the in-window delay product (in place).

        ``window_timed_series`` emits a FORECAST target (the value ``horizon`` steps
        after the window end); we replace it with the bilinear product of two delayed
        in-window steps. Window position ``p = lookback − 1 − lag`` is the step
        ``lag`` places back from the window end, so the product reads only the emitted
        window contents ``X`` (== ``values[win_idx]``) and can never reach outside the
        window. Computing it per split from that split's own ``X`` keeps
        ``y_full == concatenate([y_train, y_val, y_test])``.
        """
        p1 = lookback - 1 - lag1
        p2 = lookback - 1 - lag2
        for split in ("train", "val", "test", "full"):
            x = arrays[f"X_{split}"]  # (W, L, 1) float32
            arrays[f"y_{split}"] = (x[:, p1, 0] * x[:, p2, 0]).reshape(-1, 1).astype(np.float32)

    @staticmethod
    def _raw_series(params: DelayProductParams) -> tuple[np.ndarray, np.ndarray]:
        """Build the ``(T, 1)`` float32 signal and its ``(T,)`` float64 irregular sample times."""
        rng = np.random.default_rng(params.seed)
        k = params.n_components
        freqs = DelayProductGenerator._resolve(params.frequencies, rng, _FREQ_LOW, _FREQ_HIGH, k)
        amps = DelayProductGenerator._resolve(params.amplitudes, rng, _AMP_LOW, _AMP_HIGH, k)
        phases = DelayProductGenerator._resolve(params.phases, rng, 0.0, 2.0 * np.pi, k)

        # Irregular sample times: cumulative jittered gaps starting at t = 0. Each gap
        # is in [1 - jitter, 1 + jitter] * sample_dt, so the times are strictly
        # increasing (gap > 0 for jitter < 1).
        gaps = params.sample_dt * rng.uniform(1.0 - params.jitter, 1.0 + params.jitter, params.n_steps - 1)
        times = np.concatenate([[0.0], np.cumsum(gaps)])  # (T,) float64

        signal = (amps * np.sin(2.0 * np.pi * freqs * times[:, None] + phases)).sum(axis=1)
        if params.noise_std > 0:
            signal = signal + rng.normal(0.0, params.noise_std, params.n_steps)
        return signal.reshape(-1, 1).astype(np.float32), times

    @staticmethod
    def _resolve(values: list[float] | None, rng: np.random.Generator, low: float, high: float, k: int) -> np.ndarray:
        """Explicit per-component values as float64, or a seeded uniform draw in [low, high)."""
        if values is not None:
            return np.asarray(values, dtype=np.float64)
        return rng.uniform(low, high, k)


def get_schema() -> dict:
    """Return the JSON schema describing the generator parameters."""
    return DelayProductParams.model_json_schema()
