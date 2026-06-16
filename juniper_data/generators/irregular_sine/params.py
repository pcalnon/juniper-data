"""Parameters for the irregular-Δt (non-uniform sampling) sine synthetic generator."""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     params.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from pydantic import Field, model_validator

from juniper_data.generators._synthetic import SyntheticSequenceParams


class IrregularSineParams(SyntheticSequenceParams):
    """Configuration for the irregularly-sampled sinusoid-superposition generator.

    Samples the continuous-time signal ``x(t) = Σ_i A_i sin(2π f_i t + φ_i) + ε(t)``
    at NON-uniform times: the k-th inter-sample gap is
    ``sample_dt * U[1 - jitter, 1 + jitter]``, so ``sample_dt`` is the *nominal*
    step and ``jitter`` controls the irregularity (0 => regular; toward 1 =>
    highly irregular). The result exercises the per-step ``dt`` channel with
    genuine non-uniformity, independently of equities' calendar gaps. Inherits the
    windowing knobs from :class:`SyntheticSequenceParams`.
    """

    jitter: float = Field(default=0.5, ge=0, lt=1, description="Sampling irregularity: per-step gap = sample_dt * U[1 - jitter, 1 + jitter]. 0 yields regular spacing; values toward 1 are highly irregular.")
    n_components: int = Field(default=3, ge=1, le=16, description="Number K of superimposed sinusoids.")
    frequencies: list[float] | None = Field(default=None, description="Per-component frequencies (cycles per time_unit). None => seeded uniform in [0.02, 0.15].")
    amplitudes: list[float] | None = Field(default=None, description="Per-component amplitudes. None => seeded uniform in [0.5, 1.5].")
    phases: list[float] | None = Field(default=None, description="Per-component phases in radians. None => seeded uniform in [0, 2π).")
    noise_std: float = Field(default=0.0, ge=0, description="Std of additive Gaussian observation noise; 0 yields the exact closed-form signal.")

    @model_validator(mode="after")
    def _validate_component_lengths(self) -> IrregularSineParams:
        """Any explicit frequencies / amplitudes / phases list must have length n_components."""
        for name in ("frequencies", "amplitudes", "phases"):
            values = getattr(self, name)
            if values is not None and len(values) != self.n_components:
                raise ValueError(f"{name} has length {len(values)} but n_components={self.n_components}")
        return self
