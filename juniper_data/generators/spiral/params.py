"""Spiral dataset generator parameters.

This module defines the Pydantic model for spiral dataset generation parameters
with validation and computation methods.
"""

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from .defaults import MAX_NOISE, MAX_POINTS, MAX_ROTATIONS, MAX_SPIRALS, MIN_NOISE, MIN_POINTS, MIN_ROTATIONS, MIN_SPIRALS, SPIRAL_DEFAULT_CLOCKWISE, SPIRAL_DEFAULT_N_POINTS, SPIRAL_DEFAULT_N_ROTATIONS, SPIRAL_DEFAULT_N_SPIRALS, SPIRAL_DEFAULT_NOISE, SPIRAL_DEFAULT_RADIUS, SPIRAL_DEFAULT_SEED, SPIRAL_DEFAULT_TEST_RATIO, SPIRAL_DEFAULT_TRAIN_RATIO


class SpiralParams(BaseModel):
    """Parameters for spiral dataset generation.

    Defines the configuration for generating multi-spiral classification datasets
    with support for noise, train/test splitting, and deterministic seeding.

    Attributes:
        n_spirals: Number of spiral arms to generate.
        n_points_per_spiral: Number of points per spiral arm.
        n_rotations: Number of full rotations for each spiral.
        noise: Noise level applied to point positions.
        clockwise: Whether spirals rotate clockwise.
        seed: Random seed for reproducibility.
        train_ratio: Fraction of data for training set.
        test_ratio: Fraction of data for test set.
        shuffle: Whether to shuffle the dataset before splitting.
    """

    n_spirals: int = Field(
        default=SPIRAL_DEFAULT_N_SPIRALS,
        ge=MIN_SPIRALS,
        le=MAX_SPIRALS,
        description="Number of spiral arms to generate",
    )
    n_points_per_spiral: int = Field(
        default=SPIRAL_DEFAULT_N_POINTS,
        ge=MIN_POINTS,
        le=MAX_POINTS,
        description="Number of points per spiral arm",
    )
    n_rotations: float = Field(
        default=SPIRAL_DEFAULT_N_ROTATIONS,
        ge=MIN_ROTATIONS,
        le=MAX_ROTATIONS,
        description="Number of full rotations for each spiral",
    )
    noise: float = Field(
        default=SPIRAL_DEFAULT_NOISE,
        ge=MIN_NOISE,
        le=MAX_NOISE,
        description="Noise level applied to point positions",
    )
    clockwise: bool = Field(
        default=SPIRAL_DEFAULT_CLOCKWISE,
        description="Whether spirals rotate clockwise",
    )
    seed: Optional[int] = Field(
        default=SPIRAL_DEFAULT_SEED,
        description="Random seed for reproducibility",
    )
    train_ratio: float = Field(
        default=SPIRAL_DEFAULT_TRAIN_RATIO,
        gt=0.0,
        lt=1.0,
        description="Fraction of data for training set",
    )
    test_ratio: float = Field(
        default=SPIRAL_DEFAULT_TEST_RATIO,
        gt=0.0,
        lt=1.0,
        description="Fraction of data for test set",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the dataset before splitting",
    )
    algorithm: Literal["modern", "legacy_cascor"] = Field(
        default="modern",
        description="Generation algorithm: 'modern' (linspace+normal noise) or 'legacy_cascor' (sqrt-uniform radii + uniform noise)",
    )
    radius: float = Field(
        default=SPIRAL_DEFAULT_RADIUS,
        gt=0.0,
        le=100.0,
        description="Maximum radius for modern mode, or max distance parameter for legacy mode",
    )
    origin: Tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Origin point (x, y) for spiral center",
    )

    @model_validator(mode="after")
    def validate_ratios_sum(self) -> "SpiralParams":
        """Validate that train_ratio + test_ratio <= 1.0."""
        if self.train_ratio + self.test_ratio > 1.0:
            raise ValueError(f"train_ratio ({self.train_ratio}) + test_ratio ({self.test_ratio}) " f"must be <= 1.0, got {self.train_ratio + self.test_ratio}")
        return self

    def total_points(self) -> int:
        """Compute the total number of points in the dataset.

        Returns:
            Total number of points (n_spirals * n_points_per_spiral).
        """
        return self.n_spirals * self.n_points_per_spiral
