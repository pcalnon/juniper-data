"""Parameters for the concentric circles dataset generator."""

from pydantic import BaseModel, Field


class CirclesParams(BaseModel):
    """Configuration parameters for concentric circles dataset generation.

    Generates a binary classification dataset with points on two concentric
    circles - an inner circle and an outer circle.
    """

    n_samples: int = Field(default=100, ge=2, description="Total number of samples")
    outer_radius: float = Field(default=1.0, gt=0, description="Radius of the outer circle")
    factor: float = Field(
        default=0.5,
        gt=0,
        lt=1,
        description="Scale factor between inner and outer circles (inner_radius = outer_radius * factor)",
    )
    noise: float = Field(default=0.0, ge=0, description="Gaussian noise level added to coordinates")
    inner_ratio: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Fraction of samples on the inner circle",
    )
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")
    train_ratio: float = Field(default=0.8, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=0.2, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
