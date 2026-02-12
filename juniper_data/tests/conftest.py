"""Common pytest fixtures for juniper_data tests."""

import numpy as np
import pytest

from juniper_data.generators.spiral import SpiralGenerator, SpiralParams


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
