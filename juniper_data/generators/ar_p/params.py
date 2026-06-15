"""Parameters for the AR(p) synthetic time-series generator."""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     params.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from pydantic import Field, field_validator

from juniper_data.generators._synthetic import SyntheticSequenceParams


class ArPParams(SyntheticSequenceParams):
    """Configuration for the autoregressive AR(p) regression generator.

    Generates ``xₜ = c + Σ_{i=1}^p φ_i xₜ₋ᵢ + εₜ`` with ``εₜ ~ N(0, σ²)``. The
    order ``p`` is ``len(coefficients)``; the conditional mean depends on exactly
    the last ``p`` observations (a bounded-window memory). The default is a stable
    AR(2). Inherits the windowing knobs from :class:`SyntheticSequenceParams`.
    """

    coefficients: list[float] = Field(default=[0.5, -0.3], description="AR coefficients [φ1, ..., φp]; the order p is len(coefficients). Default is a stable AR(2).")
    const: float = Field(default=0.0, description="Constant term c (the unconditional-mean offset).")
    sigma: float = Field(default=0.1, ge=0, description="Std σ of the Gaussian innovations εₜ.")
    burn_in: int = Field(default=100, ge=0, le=100_000, description="Warmup steps to generate and drop before collecting n_steps (washes out the initial condition).")

    @field_validator("coefficients")
    @classmethod
    def _validate_coefficients(cls, value: list[float]) -> list[float]:
        """The AR order p = len(coefficients) must be at least 1."""
        if len(value) < 1:
            raise ValueError("coefficients must have at least one entry (AR order p >= 1)")
        return value
