"""Parameters for the multi-sine synthetic time-series generator."""

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


class MultiSineParams(SyntheticSequenceParams):
    """Configuration for the superposition-of-sinusoids regression generator.

    The raw signal is ``x(t) = Σ_i A_i sin(2π f_i t + φ_i) + ε(t)`` for
    ``n_components`` sinusoids. Frequencies / amplitudes / phases may be given
    explicitly (for a known-answer signal) or left ``None`` to be drawn from a
    seeded uniform distribution. Inherits the windowing knobs (``n_steps`` /
    ``lookback`` / ``horizon`` / ``sample_dt`` / ``train_ratio`` / ``seed``).
    """

    n_components: int = Field(default=3, ge=1, le=16, description="Number K of superimposed sinusoids.")
    frequencies: list[float] | None = Field(default=None, description="Per-component frequencies (cycles per time_unit). None => seeded uniform in [0.02, 0.15].")
    amplitudes: list[float] | None = Field(default=None, description="Per-component amplitudes. None => seeded uniform in [0.5, 1.5].")
    phases: list[float] | None = Field(default=None, description="Per-component phases in radians. None => seeded uniform in [0, 2π).")
    noise_std: float = Field(default=0.0, ge=0, description="Std of additive Gaussian observation noise; 0 yields the exact closed-form signal.")

    @model_validator(mode="after")
    def _validate_component_lengths(self) -> MultiSineParams:
        """Any explicit frequencies / amplitudes / phases list must have length n_components."""
        for name in ("frequencies", "amplitudes", "phases"):
            values = getattr(self, name)
            if values is not None and len(values) != self.n_components:
                raise ValueError(f"{name} has length {len(values)} but n_components={self.n_components}")
        return self
