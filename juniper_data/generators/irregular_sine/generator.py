"""Core numpy-only irregular-Δt sine synthetic time-series generator.

Samples the continuous-time superposition ``x(t) = Σ_i A_i sin(2π f_i t + φ_i)
(+ noise)`` at NON-uniform times (cumulative jittered gaps), then windows the
result via ``window_timed_series`` so the per-step ``dt`` is genuinely
non-uniform. This is the synthetic, offline, known-answer counterpart to
``equities_seq``'s calendar-gap irregularity (juniper-data#179 §A): it exercises
the irregular-Δt contract independently of real market data, with the signal
remaining closed form at each (irregular) sample time.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     generator.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import numpy as np

from juniper_data.generators._sequence import window_timed_series

from .params import IrregularSineParams

VERSION = "1.0.0"

# Ranges for seeded-random component parameters when not given explicitly
# (mirrors ``multi_sine``).
_FREQ_LOW, _FREQ_HIGH = 0.02, 0.15
_AMP_LOW, _AMP_HIGH = 0.5, 1.5


class IrregularSineGenerator:
    """numpy-only generator for an irregularly-sampled sinusoid-superposition series.

    All methods are static (stateless, deterministic given ``seed``).
    """

    @staticmethod
    def generate(params: IrregularSineParams) -> dict[str, np.ndarray]:
        """Generate the windowed irregular-Δt sine sequence dataset.

        Returns the additive 3-D NPZ contract for train/test/full with a genuinely
        non-uniform per-step ``dt`` and a variable ``target_dt``: ``X_{split}``
        ``(W, L, 1)``, the regression target ``y_{split}`` ``(W, 1)`` (signal value
        ``horizon`` steps after the window end), plus ``dt`` / ``target_dt`` /
        ``observed_mask``.

        Args:
            params: ``IrregularSineParams`` (component + sampling spec + windowing knobs).
        """
        values, times = IrregularSineGenerator._raw_series(params)
        return window_timed_series(values, times, lookback=params.lookback, horizon=params.horizon, train_ratio=params.train_ratio)

    @staticmethod
    def _raw_series(params: IrregularSineParams) -> tuple[np.ndarray, np.ndarray]:
        """Build the ``(T, 1)`` float32 signal and its ``(T,)`` float64 irregular sample times."""
        rng = np.random.default_rng(params.seed)
        k = params.n_components
        freqs = IrregularSineGenerator._resolve(params.frequencies, rng, _FREQ_LOW, _FREQ_HIGH, k)
        amps = IrregularSineGenerator._resolve(params.amplitudes, rng, _AMP_LOW, _AMP_HIGH, k)
        phases = IrregularSineGenerator._resolve(params.phases, rng, 0.0, 2.0 * np.pi, k)

        # Irregular sample times: cumulative jittered gaps starting at t = 0. Each
        # gap is in [1 - jitter, 1 + jitter] * sample_dt, so the times are strictly
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
    return IrregularSineParams.model_json_schema()
