"""Parameters for the checkerboard dataset generator."""

from pydantic import BaseModel, Field


class CheckerboardParams(BaseModel):
    """Configuration parameters for checkerboard dataset generation.

    Generates a checkerboard pattern classification dataset where
    alternating squares belong to different classes.
    """

    n_samples: int = Field(default=200, ge=2, description="Total number of samples")
    n_squares: int = Field(
        default=4,
        ge=2,
        le=16,
        description="Number of squares per side (total squares = n_squares^2)",
    )
    x_range: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Range of x values (min, max)",
    )
    y_range: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Range of y values (min, max)",
    )
    noise: float = Field(default=0.0, ge=0, description="Gaussian noise level")
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")
    train_ratio: float = Field(default=0.8, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=0.2, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
