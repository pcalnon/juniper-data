"""Parameters for the XOR dataset generator."""

# from typing import Optional

from pydantic import BaseModel, Field


class XorParams(BaseModel):
    """Configuration parameters for XOR dataset generation.

    The XOR dataset consists of 4 quadrants around the origin.
    Points in quadrants 1 and 3 (x*y > 0) belong to class 0.
    Points in quadrants 2 and 4 (x*y < 0) belong to class 1.
    """

    n_points_per_quadrant: int = Field(default=50, ge=1, description="Number of points per quadrant")
    x_range: float = Field(default=1.0, gt=0, description="Range of x values [-x_range, x_range]")
    y_range: float = Field(default=1.0, gt=0, description="Range of y values [-y_range, y_range]")
    margin: float = Field(default=0.1, ge=0, description="Margin around axes (exclusion zone)")
    noise: float = Field(default=0.0, ge=0, description="Gaussian noise level")
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")
    train_ratio: float = Field(default=0.8, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=0.2, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
