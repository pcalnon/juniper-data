"""Common pytest fixtures for juniper_data tests."""

import numpy as np
import pytest

from juniper_data.generators.spiral import SpiralGenerator, SpiralParams

# ===================================================================
# SETTINGS .env FILE ISOLATION
# ===================================================================
# pydantic-settings' ``Settings`` class is declared with
# ``env_file=".env"``, which makes every ``Settings()`` constructor
# call read the developer's local ``.env`` (gitignored, present only
# on dev machines that have run ``cp .env.example .env``). Tests that
# patch ``os.environ`` and then assert "field default applies when no
# env var is set" silently fail when a developer's local ``.env``
# defines the same variable: pydantic-settings layers .env *under*
# ``os.environ``, so a per-test ``monkeypatch.delenv(...)`` removes
# the OS-level value but leaves the .env value in effect.
#
# CI never sees this because runner checkouts have no ``.env``. The
# failure mode is local-only. Sibling cascor fix landed in cascor PR
# #309 (2026-05-26); canopy port in canopy PR #325. This is the
# juniper-data port.
#
# Counterpart regression test:
# ``juniper_data/tests/unit/test_env_file_isolation.py``.


@pytest.fixture(scope="session", autouse=True)
def _disable_settings_env_file_for_tests():
    """Stop pydantic-settings from reading a developer's local .env in tests."""
    # Import lazily so this doesn't influence pytest's plugin autoload.
    from juniper_data.api.settings import Settings, get_settings

    original_env_file = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None
    # Drop any cached Settings instance that may have been built from .env
    # before this fixture fired (e.g. via an early import side-effect).
    try:
        get_settings.cache_clear()
    except AttributeError:  # nosec B110 - cache attribute is the documented lru_cache API; absence is unexpected but recoverable
        pass
    yield
    Settings.model_config["env_file"] = original_env_file
    try:
        get_settings.cache_clear()
    except AttributeError:  # nosec B110
        pass


@pytest.fixture
def default_spiral_params() -> SpiralParams:
    """Default spiral parameters for testing."""
    return SpiralParams()


@pytest.fixture
def two_spiral_params() -> SpiralParams:
    """Parameters for a 2-spiral dataset with 100 points per spiral."""
    return SpiralParams(
        n_spirals=2,
        n_points_per_spiral=100,
        seed=42,
    )


@pytest.fixture
def three_spiral_params() -> SpiralParams:
    """Parameters for a 3-spiral dataset with 50 points per spiral."""
    return SpiralParams(
        n_spirals=3,
        n_points_per_spiral=50,
        seed=42,
    )


@pytest.fixture
def minimal_spiral_params() -> SpiralParams:
    """Minimal valid spiral parameters for fast tests."""
    return SpiralParams(
        n_spirals=2,
        n_points_per_spiral=10,
        seed=42,
    )


@pytest.fixture
def generated_two_spiral_dataset(two_spiral_params: SpiralParams) -> dict[str, np.ndarray]:
    """Generate a 2-spiral dataset for testing."""
    return SpiralGenerator.generate(two_spiral_params)


@pytest.fixture
def generated_three_spiral_dataset(three_spiral_params: SpiralParams) -> dict[str, np.ndarray]:
    """Generate a 3-spiral dataset for testing."""
    return SpiralGenerator.generate(three_spiral_params)


@pytest.fixture
def generated_minimal_dataset(minimal_spiral_params: SpiralParams) -> dict[str, np.ndarray]:
    """Generate a minimal dataset for fast tests."""
    return SpiralGenerator.generate(minimal_spiral_params)


@pytest.fixture
def sample_arrays() -> dict[str, np.ndarray]:
    """Simple sample arrays for split/shuffle testing."""
    X = np.arange(20).reshape(10, 2).astype(np.float32)
    y = np.eye(2, dtype=np.float32)[np.arange(10) % 2]
    return {"X": X, "y": y}
