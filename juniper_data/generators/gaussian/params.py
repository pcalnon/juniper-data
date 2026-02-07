"""Parameters for the Gaussian blobs dataset generator."""

from pydantic import BaseModel, Field, field_validator


class GaussianParams(BaseModel):
    """Configuration parameters for Gaussian blobs dataset generation.

    Generates a mixture-of-Gaussians classification dataset with configurable
    class centers, covariance, and noise levels.
    """

    n_classes: int = Field(default=2, ge=2, le=10, description="Number of classes/blobs")
    n_samples_per_class: int = Field(default=50, ge=1, description="Number of samples per class")
    n_features: int = Field(default=2, ge=1, description="Number of features/dimensions")
    class_std: float | list[float] = Field(
        default=1.0,
        description="Standard deviation for each class. Single value applies to all classes.",
    )
    centers: list[list[float]] | None = Field(
        default=None,
        description="List of class center coordinates. If None, centers are placed on a circle.",
    )
    center_radius: float = Field(
        default=3.0,
        gt=0,
        description="Radius for auto-placed centers when centers is None",
    )
    noise: float = Field(default=0.0, ge=0, description="Additional Gaussian noise level")
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")
    train_ratio: float = Field(default=0.8, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=0.2, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")

    @field_validator("class_std")
    @classmethod
    def validate_class_std(cls, v: float | list[float]) -> float | list[float]:
        """Validate that class_std values are positive."""
        if isinstance(v, list):
            if not all(s > 0 for s in v):
                raise ValueError("All class_std values must be positive")
        elif v <= 0:
            raise ValueError("class_std must be positive")
        return v

    @field_validator("centers")
    @classmethod
    def validate_centers(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        """Validate centers structure if provided."""
        if v is not None:
            if len(v) == 0:
                raise ValueError("centers list cannot be empty")
        return v
