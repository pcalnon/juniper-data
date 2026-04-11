"""Parameters for the Gaussian blobs dataset generator."""

from pydantic import BaseModel, Field, field_validator

from .defaults import (
    GAUSSIAN_DEFAULT_CENTER_RADIUS,
    GAUSSIAN_DEFAULT_CLASS_STD,
    GAUSSIAN_DEFAULT_N_CLASSES,
    GAUSSIAN_DEFAULT_N_FEATURES,
    GAUSSIAN_DEFAULT_N_SAMPLES_PER_CLASS,
    GAUSSIAN_DEFAULT_NOISE,
    GAUSSIAN_DEFAULT_TEST_RATIO,
    GAUSSIAN_DEFAULT_TRAIN_RATIO,
    MAX_N_CLASSES,
    MIN_N_CLASSES,
    MIN_N_FEATURES,
    MIN_N_SAMPLES_PER_CLASS,
    MIN_NOISE,
)


class GaussianParams(BaseModel):
    """Configuration parameters for Gaussian blobs dataset generation.

    Generates a mixture-of-Gaussians classification dataset with configurable
    class centers, covariance, and noise levels.
    """

    n_classes: int = Field(default=GAUSSIAN_DEFAULT_N_CLASSES, ge=MIN_N_CLASSES, le=MAX_N_CLASSES, description="Number of classes/blobs")
    n_samples_per_class: int = Field(default=GAUSSIAN_DEFAULT_N_SAMPLES_PER_CLASS, ge=MIN_N_SAMPLES_PER_CLASS, description="Number of samples per class")
    n_features: int = Field(default=GAUSSIAN_DEFAULT_N_FEATURES, ge=MIN_N_FEATURES, description="Number of features/dimensions")
    class_std: float | list[float] = Field(
        default=GAUSSIAN_DEFAULT_CLASS_STD,
        description="Standard deviation for each class. Single value applies to all classes.",
    )
    centers: list[list[float]] | None = Field(
        default=None,
        description="List of class center coordinates. If None, centers are placed on a circle.",
    )
    center_radius: float = Field(
        default=GAUSSIAN_DEFAULT_CENTER_RADIUS,
        gt=0,
        description="Radius for auto-placed centers when centers is None",
    )
    noise: float = Field(default=GAUSSIAN_DEFAULT_NOISE, ge=MIN_NOISE, description="Additional Gaussian noise level")
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")
    train_ratio: float = Field(default=GAUSSIAN_DEFAULT_TRAIN_RATIO, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=GAUSSIAN_DEFAULT_TEST_RATIO, ge=0, le=1, description="Fraction of data for testing")
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
