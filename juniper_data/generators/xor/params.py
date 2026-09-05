"""Parameters for the XOR dataset generator."""

from pydantic import Field

from juniper_data.core.constants import DEFAULT_GENERATOR_SEED
from juniper_data.core.partition_params import PartitionParams

from .defaults import (
    MIN_MARGIN,
    MIN_N_POINTS_PER_QUADRANT,
    MIN_NOISE,
    XOR_DEFAULT_MARGIN,
    XOR_DEFAULT_N_POINTS_PER_QUADRANT,
    XOR_DEFAULT_NOISE,
    XOR_DEFAULT_TEST_RATIO,
    XOR_DEFAULT_TRAIN_RATIO,
    XOR_DEFAULT_X_RANGE,
    XOR_DEFAULT_Y_RANGE,
)


class XorParams(PartitionParams):
    """Configuration parameters for XOR dataset generation.

    The XOR dataset consists of 4 quadrants around the origin.
    Points in quadrants 1 and 3 (x*y > 0) belong to class 0.
    Points in quadrants 2 and 4 (x*y < 0) belong to class 1.
    """

    n_points_per_quadrant: int = Field(default=XOR_DEFAULT_N_POINTS_PER_QUADRANT, ge=MIN_N_POINTS_PER_QUADRANT, description="Number of points per quadrant")
    x_range: float = Field(
        default=XOR_DEFAULT_X_RANGE,
        gt=0,
        description="Maximum absolute x value; x is sampled from the interval [-x_range, x_range]",
    )
    y_range: float = Field(
        default=XOR_DEFAULT_Y_RANGE,
        gt=0,
        description="Maximum absolute y value; y is sampled from the interval [-y_range, y_range]",
    )
    margin: float = Field(default=XOR_DEFAULT_MARGIN, ge=MIN_MARGIN, description="Margin around axes (exclusion zone)")
    noise: float = Field(default=XOR_DEFAULT_NOISE, ge=MIN_NOISE, description="Gaussian noise level")
    seed: int | None = Field(default=DEFAULT_GENERATOR_SEED, ge=0, description="Random seed for reproducibility. Defaults to DEFAULT_GENERATOR_SEED so the documented default configuration is REPRODUCIBLE (juniper-data#319); pass None explicitly to opt into a fresh draw per call.")
    train_ratio: float = Field(default=XOR_DEFAULT_TRAIN_RATIO, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=XOR_DEFAULT_TEST_RATIO, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle dataset before train/test split")
