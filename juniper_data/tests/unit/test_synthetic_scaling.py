"""Generator→meta scaling channel + denorm round-trip for the synthetics (juniper-data#179 §A.6.5).

Pins that every synthetic emits the reserved ``"scaling"`` channel key (identity by
default), that ``scaling="standardize"`` reports train-split-fit descriptors while the
NPZ arrays stay RAW, and the denorm round-trip (the §B acceptance check): standardizing
then inverting the raw values with the reported descriptor recovers them to tolerance.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_synthetic_scaling.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import numpy as np
import pytest

from juniper_data.core.meta import pop_scaling_meta
from juniper_data.core.scaling import inverse_standardize, standardize
from juniper_data.generators.ar_p import ArPGenerator, ArPParams
from juniper_data.generators.irregular_sine import IrregularSineGenerator, IrregularSineParams
from juniper_data.generators.mackey_glass import MackeyGlassGenerator, MackeyGlassParams
from juniper_data.generators.multi_sine import MultiSineGenerator, MultiSineParams

pytestmark = [pytest.mark.unit, pytest.mark.generators]

# (Generator, Params, extra non-default kwargs) for the four synthetic regression gens.
_CASES = [
    (MultiSineGenerator, MultiSineParams, {}),
    (MackeyGlassGenerator, MackeyGlassParams, {"discard": 50}),
    (ArPGenerator, ArPParams, {"burn_in": 20}),
    (IrregularSineGenerator, IrregularSineParams, {"jitter": 0.6}),
]


@pytest.mark.parametrize("gen,params_cls,extra", _CASES)
def test_default_scaling_is_identity(gen, params_cls, extra) -> None:
    arrays = gen.generate(params_cls(n_steps=200, lookback=16, **extra))
    scaling = pop_scaling_meta(arrays)
    assert scaling["dt_scaling"] == {"method": "identity"}
    assert scaling["target_scaling"] == {"y": {"method": "identity"}}
    # After the reserved key is popped, the dict is array-only (safe to checksum / NPZ-persist).
    assert all(isinstance(v, np.ndarray) for v in arrays.values())


@pytest.mark.parametrize("gen,params_cls,extra", _CASES)
def test_standardize_emits_descriptors_and_round_trips(gen, params_cls, extra) -> None:
    arrays = gen.generate(params_cls(n_steps=300, lookback=16, scaling="standardize", **extra))
    scaling = pop_scaling_meta(arrays)
    dt_desc = scaling["dt_scaling"]
    target_desc = scaling["target_scaling"]["y"]

    assert dt_desc["method"] == "standardize" and target_desc["method"] == "standardize"
    assert all(isinstance(dt_desc[k], float) for k in ("mean", "std", "min", "max"))
    # Advisory: the NPZ arrays stay RAW, so the contract still holds (dt[:, 0] == 0).
    assert np.all(arrays["dt_full"][:, 0] == 0)

    # Denorm round-trip (§B): standardize(raw) then inverse recovers raw to tolerance.
    raw_y = arrays["y_full"]
    np.testing.assert_allclose(inverse_standardize(standardize(raw_y, target_desc), target_desc), raw_y, rtol=1e-4, atol=1e-4)
    raw_dt = arrays["dt_full"][:, 1:]
    np.testing.assert_allclose(inverse_standardize(standardize(raw_dt, dt_desc), dt_desc), raw_dt, rtol=1e-4, atol=1e-4)


def test_standardize_target_descriptor_fit_on_train_split() -> None:
    # The descriptor is fit on the TRAIN split only (no test leakage).
    arrays = IrregularSineGenerator.generate(IrregularSineParams(n_steps=400, lookback=16, jitter=0.6, scaling="standardize"))
    scaling = pop_scaling_meta(arrays)
    y_train = np.asarray(arrays["y_train"], dtype=np.float64)
    assert scaling["target_scaling"]["y"]["mean"] == pytest.approx(float(y_train.mean()), rel=1e-6)
    assert scaling["target_scaling"]["y"]["std"] == pytest.approx(float(y_train.std()), rel=1e-6)


def test_irregular_dt_descriptor_has_real_spread() -> None:
    # irregular_sine's dt is non-uniform => a meaningful standardize std (not the std-guard fallback).
    arrays = IrregularSineGenerator.generate(IrregularSineParams(n_steps=400, lookback=24, jitter=0.7, scaling="standardize"))
    scaling = pop_scaling_meta(arrays)
    assert scaling["dt_scaling"]["std"] > 0.1


def test_regular_dt_descriptor_uses_std_guard() -> None:
    # A regular synthetic has constant dt => std 0 => guarded to 1.0 (still round-trips).
    arrays = MultiSineGenerator.generate(MultiSineParams(n_steps=300, lookback=16, sample_dt=1.0, scaling="standardize"))
    scaling = pop_scaling_meta(arrays)
    assert scaling["dt_scaling"]["std"] == 1.0
    assert scaling["dt_scaling"]["mean"] == pytest.approx(1.0)
