"""Parameters for the delay-product (bilinear capacity) irregular-Δt synthetic generator."""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     params.py
# Author:        Paul Calnon
# Version:       0.9.0
# License:       MIT License

from __future__ import annotations

from pydantic import Field, model_validator

from juniper_data.generators._synthetic import SyntheticSequenceParams


class DelayProductParams(SyntheticSequenceParams):
    """Configuration for the delay-product (bilinear capacity) generator.

    Samples the continuous-time signal ``x(t) = Σ_i A_i sin(2π f_i t + φ_i) + ε(t)``
    at NON-uniform times (the same irregular-Δt sampling as ``irregular_sine``: the
    k-th inter-sample gap is ``sample_dt * U[1 - jitter, 1 + jitter]``), then defines
    the regression target as the **bilinear product of two delayed in-window
    values**::

        y_i = x(t_i − τ₁) · x(t_i − τ₂)

    where ``t_i`` is the window-end step and ``τ₁ = lag1`` / ``τ₂ = lag2`` are integer
    step-delays measured back from the window end, kept STRICTLY inside the lookback
    (``0 <= lag < lookback``; ``lag = 0`` is the window-end step itself). Because each
    delayed value is a (near-)linear functional of the fixed-order LMU Legendre
    memory, their product is a *quadratic form* in the memory state: a LINEAR readout
    can only form linear combinations and provably cannot represent it (its r² is
    bounded below 1), while a nonlinear readout (random Fourier features / MLP) can
    approximate it. This is the capacity instrument that exposes a clear
    nonlinear ≫ linear r² gap, complementing the near-linear synthetics where the
    linear readout is already at its ceiling (DP-3 readout-spectrum design §8a).

    Inherits the windowing knobs (``n_steps`` / ``lookback`` / ``horizon`` /
    ``sample_dt`` / ``train_ratio`` / ``seed`` / ``scaling``) from
    :class:`SyntheticSequenceParams`. ``horizon`` only positions the windows (and the
    advisory ``target_dt`` side-channel); the target itself is in-window, so the
    answer does not live at the forecast step.
    """

    lag1: int = Field(default=2, ge=0, description="First delay τ₁ in steps back from the window end (0 = the window-end step). Must be < lookback (the delayed step must lie inside the window).")
    lag2: int = Field(default=8, ge=0, description="Second delay τ₂ in steps back from the window end. Must be < lookback. May equal lag1 (yielding a squared term x(t−τ)², still a non-linear capacity target).")
    jitter: float = Field(default=0.5, ge=0, lt=1, description="Sampling irregularity: per-step gap = sample_dt * U[1 - jitter, 1 + jitter]. 0 yields regular spacing; values toward 1 are highly irregular.")
    n_components: int = Field(default=3, ge=1, le=16, description="Number K of superimposed sinusoids.")
    frequencies: list[float] | None = Field(default=None, description="Per-component frequencies (cycles per time_unit). None => seeded uniform in [0.02, 0.15].")
    amplitudes: list[float] | None = Field(default=None, description="Per-component amplitudes. None => seeded uniform in [0.5, 1.5].")
    phases: list[float] | None = Field(default=None, description="Per-component phases in radians. None => seeded uniform in [0, 2π).")
    noise_std: float = Field(default=0.0, ge=0, description="Std of additive Gaussian observation noise; 0 yields the exact closed-form signal (and an exactly-known product target).")

    @model_validator(mode="after")
    def _validate_component_lengths(self) -> DelayProductParams:
        """Any explicit frequencies / amplitudes / phases list must have length n_components."""
        for name in ("frequencies", "amplitudes", "phases"):
            values = getattr(self, name)
            if values is not None and len(values) != self.n_components:
                raise ValueError(f"{name} has length {len(values)} but n_components={self.n_components}")
        return self

    @model_validator(mode="after")
    def _validate_lags_in_lookback(self) -> DelayProductParams:
        """Both delays must address a step strictly inside the lookback window."""
        for name in ("lag1", "lag2"):
            lag = getattr(self, name)
            if lag >= self.lookback:
                raise ValueError(f"{name}={lag} must be < lookback={self.lookback} (the delayed step must lie inside the window)")
        return self
