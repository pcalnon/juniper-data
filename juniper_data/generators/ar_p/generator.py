"""Core numpy-only AR(p) synthetic time-series generator.

Generates an autoregressive process ``xₜ = c + Σ_{i=1}^p φ_i xₜ₋ᵢ + εₜ`` with
Gaussian innovations, drops a warmup transient, and windows the result into the
additive 3-D sequence contract. The conditional mean depends on EXACTLY the last
``p`` observations (a width-``p`` sufficient statistic), so the memory is
BOUNDED-WINDOW and the process is star-free-trivial -- the clearest non-counting,
sub-ceiling smoke dataset ([OQ-5], dataset-audit 2026-06-13).
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

from .params import ArPParams

VERSION = "2.0.0"


class ArPGenerator:
    """numpy-only generator for an autoregressive AR(p) regression series.

    All methods are static (stateless, deterministic given ``seed``).
    """

    @staticmethod
    def generate(params: ArPParams) -> dict[str, np.ndarray]:
        """Generate the windowed AR(p) sequence dataset.

        Returns the additive 3-D NPZ contract for train/test/full:
        ``X_{split}`` ``(W, L, 1)``, the regression target ``y_{split}`` ``(W, 1)``
        (the series value ``horizon`` steps after the window end), plus ``dt`` /
        ``target_dt`` / ``observed_mask``.

        Args:
            params: ``ArPParams`` (AR spec + windowing knobs).
        """
        series = ArPGenerator._raw_series(params)
        return build_sequence_arrays(series, params)

    @staticmethod
    def _raw_series(params: ArPParams) -> np.ndarray:
        """Run the AR(p) recurrence and return ``(n_steps, 1)`` float32."""
        rng = np.random.default_rng(params.seed)
        phi = np.asarray(params.coefficients, dtype=np.float64)
        p = phi.size
        total = params.n_steps + params.burn_in

        eps = rng.normal(0.0, params.sigma, total + p)
        x = np.empty(total + p, dtype=np.float64)
        x[:p] = params.const + eps[:p]  # warm start; washed out by burn_in
        for t in range(p, total + p):
            # phi . [x[t-1], ..., x[t-p]]  ==  Σ_i φ_i x[t-i]
            x[t] = params.const + phi @ x[t - p : t][::-1] + eps[t]

        start = p + params.burn_in
        series = x[start : start + params.n_steps]
        return series.reshape(-1, 1).astype(np.float32)


def get_schema() -> dict:
    """Return the JSON schema describing the generator parameters."""
    return ArPParams.model_json_schema()
