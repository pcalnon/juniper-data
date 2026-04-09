"""Parameters for the concentric circles dataset generator."""

from pydantic import BaseModel, Field

from .defaults import (
    CIRCLES_DEFAULT_FACTOR,
    CIRCLES_DEFAULT_INNER_RATIO,
    CIRCLES_DEFAULT_N_SAMPLES,
    CIRCLES_DEFAULT_NOISE,
    CIRCLES_DEFAULT_OUTER_RADIUS,
    CIRCLES_DEFAULT_TEST_RATIO,
    CIRCLES_DEFAULT_TRAIN_RATIO,
    MIN_N_SAMPLES,
    MIN_NOISE,
)


class CirclesParams(BaseModel):
    """Configuration parameters for concentric circles dataset generation.

    Generates a binary classification dataset with points on two concentric
    circles - an inner circle and an outer circle.
    """

    n_samples: int = Field(default=CIRCLES_DEFAULT_N_SAMPLES, ge=MIN_N_SAMPLES, description="Total number of samples")
    outer_radius: float = Field(default=CIRCLES_DEFAULT_OUTER_RADIUS, gt=0, description="Radius of the outer circle")
    factor: float = Field(
        default=CIRCLES_DEFAULT_FACTOR,
        gt=0,
        lt=1,
        description="Scale factor between inner and outer circles (inner_radius = outer_radius * factor)",
    )
    noise: float = Field(default=CIRCLES_DEFAULT_NOISE, ge=MIN_NOISE, description="Gaussian noise level added to coordinates")
    inner_ratio: float = Field(
        default=CIRCLES_DEFAULT_INNER_RATIO,
        gt=0,
        le=1,
        description="Fraction of samples on the inner circle",
    )
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")
    train_ratio: float = Field(default=CIRCLES_DEFAULT_TRAIN_RATIO, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=CIRCLES_DEFAULT_TEST_RATIO, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
