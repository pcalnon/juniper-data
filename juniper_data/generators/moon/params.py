"""Parameters for the two-moons dataset generator."""

from pydantic import BaseModel, Field

from .defaults import (
    MIN_N_SAMPLES,
    MIN_NOISE,
    MOON_DEFAULT_N_SAMPLES,
    MOON_DEFAULT_NOISE,
    MOON_DEFAULT_TEST_RATIO,
    MOON_DEFAULT_TRAIN_RATIO,
)


class MoonParams(BaseModel):
    """Configuration parameters for two-moons dataset generation.

    Generates a binary classification dataset where each class lies on
    one of two interleaving half-circles ("moons") in 2D space.
    """

    n_samples: int = Field(
        default=MOON_DEFAULT_N_SAMPLES,
        ge=MIN_N_SAMPLES,
        description="Total number of samples (split evenly between the two moons)",
    )
    noise: float = Field(
        default=MOON_DEFAULT_NOISE,
        ge=MIN_NOISE,
        description="Gaussian noise standard deviation added to each coordinate",
    )
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")
    train_ratio: float = Field(default=MOON_DEFAULT_TRAIN_RATIO, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=MOON_DEFAULT_TEST_RATIO, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
