"""Core numpy-only Mackey-Glass synthetic time-series generator.

Integrates the Mackey-Glass delay-differential equation
``dx/dt = β x(t-τ) / (1 + x(t-τ)^n) - γ x(t)`` with the discrete Euler scheme
(integer delay ``τ_steps = round(τ / sample_dt)``, constant history ``x0``), drops
a transient, and windows the result into the additive 3-D sequence contract. The
canonical regime (β=0.2, γ=0.1, n=10, τ=17) is chaotic on a bounded attractor:
the predictor needs a Takens delay-embedding window (~τ) of FADING memory, and
the useful horizon is Lyapunov-capped -- a chaos/horizon stress test orthogonal to
the star-free counting ceiling ([OQ-5], dataset-audit 2026-06-13).
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

from .params import MackeyGlassParams

VERSION = "2.0.0"


class MackeyGlassGenerator:
    """numpy-only generator for a Mackey-Glass chaotic regression series.

    All methods are static (stateless, deterministic given ``seed`` -- the seed
    only matters when ``init_noise_std > 0``).
    """

    @staticmethod
    def generate(params: MackeyGlassParams) -> dict[str, np.ndarray]:
        """Generate the windowed Mackey-Glass sequence dataset.

        Returns the additive 3-D NPZ contract for train/test/full:
        ``X_{split}`` ``(W, L, 1)``, the regression target ``y_{split}`` ``(W, 1)``
        (the state ``horizon`` steps after the window end), plus ``dt`` /
        ``target_dt`` / ``observed_mask``.

        Args:
            params: ``MackeyGlassParams`` (dynamics + windowing knobs).
        """
        series = MackeyGlassGenerator._raw_series(params)
        return build_sequence_arrays(series, params)

    @staticmethod
    def _raw_series(params: MackeyGlassParams) -> np.ndarray:
        """Integrate the discrete Mackey-Glass map and return ``(n_steps, 1)`` float32."""
        tau_steps = max(1, round(params.tau / params.sample_dt))
        total = params.n_steps + params.discard

        # traj[0 .. tau_steps] is the constant history x(t)=x0; the Euler recurrence
        # then fills traj[tau_steps + 1 .. tau_steps + total].
        traj = np.empty(tau_steps + 1 + total, dtype=np.float64)
        traj[: tau_steps + 1] = params.x0
        if params.init_noise_std > 0:
            rng = np.random.default_rng(params.seed)
            traj[: tau_steps + 1] += rng.normal(0.0, params.init_noise_std, tau_steps + 1)

        beta, gamma, n_exp, dt = params.beta, params.gamma, params.n_exp, params.sample_dt
        for k in range(tau_steps, tau_steps + total):
            delayed = traj[k - tau_steps]
            traj[k + 1] = traj[k] + dt * (beta * delayed / (1.0 + delayed**n_exp) - gamma * traj[k])

        start = tau_steps + 1 + params.discard
        series = traj[start : start + params.n_steps]
        return series.reshape(-1, 1).astype(np.float32)


def get_schema() -> dict:
    """Return the JSON schema describing the generator parameters."""
    return MackeyGlassParams.model_json_schema()
