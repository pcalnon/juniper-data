"""Core numpy-only multi-sine synthetic time-series generator.

Produces ``x(t) = Σ_i A_i sin(2π f_i t + φ_i) (+ noise)`` -- a finite
superposition of sinusoids sampled at a constant ``sample_dt`` -- then windows it
into the additive 3-D sequence contract via ``window_regular_series``. With
``noise_std == 0`` the signal is exact closed form (the known-answer smoke
dataset); the memory required is bounded by ~one beat period, so it is a
sub-ceiling recurrent-regression test ([OQ-5], dataset-audit 2026-06-13).
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

from juniper_data.generators._synthetic import build_sequence_arrays

from .params import MultiSineParams

VERSION = "3.0.0"

# Ranges for seeded-random component parameters when not given explicitly. The
# frequency band keeps the period (1 / f, in steps at sample_dt == 1) within a
# few tens of steps, so a default lookback window spans roughly one to a few
# periods of the slowest component.
_FREQ_LOW, _FREQ_HIGH = 0.02, 0.15
_AMP_LOW, _AMP_HIGH = 0.5, 1.5


class MultiSineGenerator:
    """numpy-only generator for a superposition-of-sinusoids regression series.

    All methods are static (stateless, deterministic given ``seed``).
    """

    @staticmethod
    def generate(params: MultiSineParams) -> dict[str, np.ndarray]:
        """Generate the windowed multi-sine sequence dataset.

        Returns the additive 3-D NPZ contract for train/val/test:
        ``X_{split}`` ``(W, L, 1)``, the regression target ``y_{split}`` ``(W, 1)``
        (signal value ``horizon`` steps after the window end), plus ``dt`` /
        ``target_dt`` / ``observed_mask``.

        Args:
            params: ``MultiSineParams`` (component spec + windowing knobs).
        """
        series = MultiSineGenerator._raw_series(params)
        return build_sequence_arrays(series, params)

    @staticmethod
    def _raw_series(params: MultiSineParams) -> np.ndarray:
        """Build the ``(T, 1)`` float32 superposition signal (closed form + optional noise)."""
        rng = np.random.default_rng(params.seed)
        k = params.n_components
        freqs = MultiSineGenerator._resolve(params.frequencies, rng, _FREQ_LOW, _FREQ_HIGH, k)
        amps = MultiSineGenerator._resolve(params.amplitudes, rng, _AMP_LOW, _AMP_HIGH, k)
        phases = MultiSineGenerator._resolve(params.phases, rng, 0.0, 2.0 * np.pi, k)

        t = np.arange(params.n_steps, dtype=np.float64) * params.sample_dt  # (T,)
        # (T, K) per-component contributions summed to the (T,) signal.
        signal = (amps * np.sin(2.0 * np.pi * freqs * t[:, None] + phases)).sum(axis=1)
        if params.noise_std > 0:
            signal = signal + rng.normal(0.0, params.noise_std, params.n_steps)
        return signal.reshape(-1, 1).astype(np.float32)

    @staticmethod
    def _resolve(values: list[float] | None, rng: np.random.Generator, low: float, high: float, k: int) -> np.ndarray:
        """Explicit per-component values as float64, or a seeded uniform draw in [low, high)."""
        if values is not None:
            return np.asarray(values, dtype=np.float64)
        return rng.uniform(low, high, k)


def get_schema() -> dict:
    """Return the JSON schema describing the generator parameters."""
    return MultiSineParams.model_json_schema()
