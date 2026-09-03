"""Parameters for the checkerboard dataset generator."""

from pydantic import BaseModel, Field

from juniper_data.core.constants import DEFAULT_GENERATOR_SEED

from .defaults import (
    CHECKERBOARD_DEFAULT_N_SAMPLES,
    CHECKERBOARD_DEFAULT_N_SQUARES,
    CHECKERBOARD_DEFAULT_NOISE,
    CHECKERBOARD_DEFAULT_TEST_RATIO,
    CHECKERBOARD_DEFAULT_TRAIN_RATIO,
    CHECKERBOARD_DEFAULT_X_RANGE,
    CHECKERBOARD_DEFAULT_Y_RANGE,
    MAX_N_SQUARES,
    MIN_N_SAMPLES,
    MIN_N_SQUARES,
    MIN_NOISE,
)


class CheckerboardParams(BaseModel):
    """Configuration parameters for checkerboard dataset generation.

    Generates a checkerboard pattern classification dataset where
    alternating squares belong to different classes.
    """

    n_samples: int = Field(default=CHECKERBOARD_DEFAULT_N_SAMPLES, ge=MIN_N_SAMPLES, description="Total number of samples")
    n_squares: int = Field(
        default=CHECKERBOARD_DEFAULT_N_SQUARES,
        ge=MIN_N_SQUARES,
        le=MAX_N_SQUARES,
        description="Number of squares per side (total squares = n_squares^2)",
    )
    x_range: tuple[float, float] = Field(
        default=CHECKERBOARD_DEFAULT_X_RANGE,
        description="Range of x values (min, max)",
    )
    y_range: tuple[float, float] = Field(
        default=CHECKERBOARD_DEFAULT_Y_RANGE,
        description="Range of y values (min, max)",
    )
    noise: float = Field(default=CHECKERBOARD_DEFAULT_NOISE, ge=MIN_NOISE, description="Gaussian noise level")
    seed: int | None = Field(default=DEFAULT_GENERATOR_SEED, ge=0, description="Random seed for reproducibility. Defaults to DEFAULT_GENERATOR_SEED so the documented default configuration is REPRODUCIBLE (juniper-data#319); pass None explicitly to opt into a fresh draw per call.")
    train_ratio: float = Field(default=CHECKERBOARD_DEFAULT_TRAIN_RATIO, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=CHECKERBOARD_DEFAULT_TEST_RATIO, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
