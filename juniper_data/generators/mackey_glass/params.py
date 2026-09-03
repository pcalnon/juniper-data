"""Parameters for the Mackey-Glass synthetic time-series generator."""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     params.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from pydantic import Field

from juniper_data.core.constants import DEFAULT_MACKEY_GLASS_INIT_NOISE_STD
from juniper_data.generators._synthetic import SyntheticSequenceParams


class MackeyGlassParams(SyntheticSequenceParams):
    """Configuration for the Mackey-Glass delay-differential regression generator.

    Integrates ``dx/dt = β x(t-τ) / (1 + x(t-τ)^n) - γ x(t)`` by the discrete
    Euler scheme ``x[k+1] = x[k] + Δt (β x[k-τ] / (1 + x[k-τ]^n) - γ x[k])`` with a
    constant history ``x(t) = x0`` for ``t in [-τ, 0]``. The defaults are the
    canonical chaotic regime (β=0.2, γ=0.1, n=10, τ=17). Inherits the windowing
    knobs from :class:`SyntheticSequenceParams`.
    """

    tau: float = Field(default=17.0, gt=0, description="Delay τ in time_unit; the canonical chaotic regime uses 17.")
    beta: float = Field(default=0.2, gt=0, description="Production coefficient β.")
    gamma: float = Field(default=0.1, gt=0, description="Decay coefficient γ.")
    n_exp: float = Field(default=10.0, gt=0, description="Nonlinearity exponent n.")
    x0: float = Field(default=0.5, gt=0, description="Constant history value x(t)=x0 for t in [-τ, 0].")
    init_noise_std: float = Field(
        default=DEFAULT_MACKEY_GLASS_INIT_NOISE_STD,
        ge=0,
        description="Std of an optional seeded Gaussian perturbation added to the history block; 0 yields an exact deterministic init. THIS IS THE KNOB THAT DECIDES WHETHER ``seed`` HAS ANY EFFECT: the seed is consumed only inside ``if init_noise_std > 0``, so at 0 every seed produces byte-identical output. Defaulted from DEFAULT_MACKEY_GLASS_INIT_NOISE_STD (juniper-data#319) so a deployment can make this generator seed-sensitive without editing code.",
    )
    discard: int = Field(default=250, ge=0, le=100_000, description="Transient steps to integrate and drop before collecting n_steps (settles onto the attractor).")
