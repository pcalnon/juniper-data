#!/usr/bin/env python
"""Reproducibility at documented defaults (juniper-data#319).

Nine generators declared ``seed: int | None = Field(default=None)`` and were therefore
NOT reproducible at their documented defaults: two identical calls produced different
data, and ``shuffle_and_split`` re-drew the partition boundaries from OS entropy too.

These pin the four properties the fix has to hold simultaneously, because three of them
are in tension:

1. the DEFAULT is reproducible;
2. an EXPLICIT ``seed=None`` still opts into a fresh draw -- the escape hatch survives;
3. explicit ``None`` still receives the BUG-JD-04 cache nonce, so a seedless request
   cannot collide with a stale seedless artifact;
4. the default is overridable from the environment, and the override reaches the
   generated arrays rather than merely the params model.

A fix that satisfies (1) by making ``seed`` non-optional would break (2) and (3).
"""

import importlib
import subprocess
import sys

import numpy as np
import pytest

from juniper_data.core.constants import (
    DEFAULT_GENERATOR_SEED_ENV_VAR,
    DEFAULT_GENERATOR_SEED_FALLBACK,
)
from juniper_data.core.dataset_id import generate_dataset_id

pytestmark = pytest.mark.unit

# The 2-D generators whose default was ``None`` before #319, plus ``spiral`` as the
# pre-existing control (it always had a concrete default).
SEEDED_2D = [
    ("checkerboard", "CheckerboardParams", "CheckerboardGenerator"),
    ("circles", "CirclesParams", "CirclesGenerator"),
    ("gaussian", "GaussianParams", "GaussianGenerator"),
    ("moon", "MoonParams", "MoonGenerator"),
    ("xor", "XorParams", "XorGenerator"),
    ("spiral", "SpiralParams", "SpiralGenerator"),
]


def _load(mod: str, params_name: str, gen_name: str):
    params_cls = getattr(importlib.import_module(f"juniper_data.generators.{mod}.params"), params_name)
    gen_cls = getattr(importlib.import_module(f"juniper_data.generators.{mod}.generator"), gen_name)
    return params_cls, gen_cls


class TestDefaultsAreReproducible:
    @pytest.mark.parametrize("mod,params_name,gen_name", SEEDED_2D)
    def test_two_calls_at_defaults_are_identical(self, mod, params_name, gen_name):
        """THE property. Before #319 this failed for every generator but spiral."""
        params_cls, gen_cls = _load(mod, params_name, gen_name)
        first = gen_cls.generate(params_cls())
        second = gen_cls.generate(params_cls())
        for key in ("X_train", "y_train", "X_test", "y_test"):
            assert np.array_equal(first[key], second[key]), f"{mod}: {key} differs between two default-config calls"

    @pytest.mark.parametrize("mod,params_name,gen_name", SEEDED_2D)
    def test_default_seed_is_not_none(self, mod, params_name, gen_name):
        params_cls, _ = _load(mod, params_name, gen_name)
        assert params_cls().seed is not None, f"{mod} still defaults to a None seed"


class TestExplicitNoneStillOptsOut:
    """The escape hatch. Defaulting the seed must not remove the ability to ask for a
    fresh draw -- only change what happens when the caller says nothing."""

    def test_explicit_none_is_still_accepted(self):
        params_cls, _ = _load("moon", "MoonParams", "MoonGenerator")
        assert params_cls(seed=None).seed is None

    def test_explicit_none_still_yields_different_data(self):
        params_cls, gen_cls = _load("moon", "MoonParams", "MoonGenerator")
        first = gen_cls.generate(params_cls(seed=None))["X_train"]
        second = gen_cls.generate(params_cls(seed=None))["X_train"]
        assert not np.array_equal(first, second), "explicit seed=None should still draw fresh"

    def test_explicit_none_still_gets_the_cache_nonce(self):
        """BUG-JD-04: a seedless request must not collide with a stale seedless artifact."""
        first = generate_dataset_id("moon", "v1.0.0", {"seed": None})
        second = generate_dataset_id("moon", "v1.0.0", {"seed": None})
        assert first != second

    def test_a_concrete_seed_is_still_deterministic_in_the_id(self):
        first = generate_dataset_id("moon", "v1.0.0", {"seed": 5})
        second = generate_dataset_id("moon", "v1.0.0", {"seed": 5})
        assert first == second


class TestEnvironmentOverride:
    """Runs in a SUBPROCESS because the default resolves at import time -- setting the
    variable inside this process after juniper_data is imported would prove nothing, and
    a test that cannot fail is worse than no test."""

    @staticmethod
    def _child(env_value: str | None, expr: str) -> str:
        import os

        env = dict(os.environ)
        env.pop(DEFAULT_GENERATOR_SEED_ENV_VAR, None)
        if env_value is not None:
            env[DEFAULT_GENERATOR_SEED_ENV_VAR] = env_value
        out = subprocess.run([sys.executable, "-c", expr], capture_output=True, text=True, env=env, check=True)
        return out.stdout.strip()

    def test_override_changes_the_default(self):
        got = self._child("1234", "from juniper_data.core.constants import DEFAULT_GENERATOR_SEED as S; print(S)")
        assert got == "1234"

    def test_override_reaches_the_generated_arrays(self):
        """Not merely the params model -- the data itself."""
        expr = "import numpy as np;from juniper_data.generators.moon.params import MoonParams as P;from juniper_data.generators.moon.generator import MoonGenerator as G;print(np.array_equal(G.generate(P())['X_train'], G.generate(P(seed=7))['X_train']))"
        assert self._child("7", expr) == "True"

    @pytest.mark.parametrize("bad", ["not-an-int", "-5", "", "   "])
    def test_malformed_override_falls_back_rather_than_raising(self, bad):
        """A configuration error must not make the package unimportable."""
        got = self._child(bad, "from juniper_data.core.constants import DEFAULT_GENERATOR_SEED as S; print(S)")
        assert got == str(DEFAULT_GENERATOR_SEED_FALLBACK)

    def test_unset_uses_the_fallback(self):
        got = self._child(None, "from juniper_data.core.constants import DEFAULT_GENERATOR_SEED as S; print(S)")
        assert got == str(DEFAULT_GENERATOR_SEED_FALLBACK)
        assert DEFAULT_GENERATOR_SEED_FALLBACK == 42, "the shared default should match spiral's long-standing SPIRAL_DEFAULT_SEED"


class TestResolverDirectly:
    """Unit-test the resolver in-process, alongside the subprocess tests above.

    The subprocess tests prove the property that matters end to end -- an env override
    reaches the generated arrays -- but a subprocess earns no coverage credit, so the
    branches below would read as untested. These call the pure function directly, which
    is also the only way to exercise the cast-failure path deterministically.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1234", 1234),  # well-formed
            ("  99  ", 99),  # surrounding whitespace tolerated
            ("0", 0),  # zero is valid, not falsy-rejected
            (None, 42),  # unset -> fallback
            ("", 42),  # empty -> fallback
            ("   ", 42),  # whitespace-only -> fallback
            ("not-an-int", 42),  # unparseable -> fallback
            ("-1", 42),  # negative -> fallback
        ],
    )
    def test_int_resolution(self, monkeypatch, raw, expected):
        from juniper_data.core.constants import _resolve_env_number

        var = "JUNIPER_DATA_TEST_RESOLVER_INT"
        monkeypatch.delenv(var, raising=False)
        if raw is not None:
            monkeypatch.setenv(var, raw)
        assert _resolve_env_number(var, 42, int) == expected

    @pytest.mark.parametrize("raw,expected", [("0.25", 0.25), ("0", 0.0), ("-0.5", 0.0), ("nope", 0.0), (None, 0.0)])
    def test_float_resolution(self, monkeypatch, raw, expected):
        from juniper_data.core.constants import _resolve_env_number

        var = "JUNIPER_DATA_TEST_RESOLVER_FLOAT"
        monkeypatch.delenv(var, raising=False)
        if raw is not None:
            monkeypatch.setenv(var, raw)
        assert _resolve_env_number(var, 0.0, float) == expected

    def test_never_raises_on_a_hostile_value(self, monkeypatch):
        """A configuration error must not make the package unimportable."""
        from juniper_data.core.constants import _resolve_env_number

        var = "JUNIPER_DATA_TEST_RESOLVER_HOSTILE"
        for hostile in ("1e400", "nan", "0x10", "1,000", "٣"):
            monkeypatch.setenv(var, hostile)
            result = _resolve_env_number(var, 7, int)
            assert isinstance(result, (int, float))


class TestMackeyGlassInitNoiseStd:
    """``init_noise_std`` decides whether mackey_glass's seed does anything at all, so it
    gets the same constants/env treatment (design §9.6.5)."""

    def test_default_is_deterministic_and_documented_as_such(self):
        from juniper_data.core.constants import DEFAULT_MACKEY_GLASS_INIT_NOISE_STD
        from juniper_data.generators.mackey_glass.params import MackeyGlassParams

        assert MackeyGlassParams().init_noise_std == DEFAULT_MACKEY_GLASS_INIT_NOISE_STD

    def test_at_default_the_seed_has_no_effect(self):
        """Pins the documented behaviour rather than treating it as a defect."""
        from juniper_data.generators.mackey_glass.generator import MackeyGlassGenerator
        from juniper_data.generators.mackey_glass.params import MackeyGlassParams

        first = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=200, seed=1))["X_train"]
        second = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=200, seed=999_999))["X_train"]
        assert np.array_equal(first, second), "at init_noise_std=0 the seed is inert by design"

    def test_raising_init_noise_std_makes_the_seed_matter(self):
        from juniper_data.generators.mackey_glass.generator import MackeyGlassGenerator
        from juniper_data.generators.mackey_glass.params import MackeyGlassParams

        first = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=200, seed=1, init_noise_std=0.01))["X_train"]
        second = MackeyGlassGenerator.generate(MackeyGlassParams(n_steps=200, seed=999_999, init_noise_std=0.01))["X_train"]
        assert not np.array_equal(first, second)
