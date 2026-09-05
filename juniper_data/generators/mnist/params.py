"""Parameters for the MNIST dataset generator."""

from typing import Literal

from pydantic import Field

from juniper_data.core.constants import DEFAULT_GENERATOR_SEED
from juniper_data.core.partition_params import CarveOnlyPartitionParams

from .defaults import (
    MNIST_DEFAULT_DATASET,
    MNIST_DEFAULT_FLATTEN,
    MNIST_DEFAULT_NORMALIZE,
    MNIST_DEFAULT_ONE_HOT_LABELS,
    MNIST_DEFAULT_TEST_RATIO,
    MNIST_DEFAULT_TRAIN_RATIO,
)


class MnistParams(CarveOnlyPartitionParams):
    """Configuration parameters for MNIST dataset generation.

    Loads and preprocesses MNIST or Fashion-MNIST datasets from
    Hugging Face Hub.
    """

    dataset: Literal["mnist", "fashion_mnist"] = Field(
        default=MNIST_DEFAULT_DATASET,
        description="Dataset to load: 'mnist' or 'fashion_mnist'",
    )
    n_samples: int | None = Field(
        default=None,
        ge=1,
        description="Limit number of samples (None for full dataset)",
    )
    flatten: bool = Field(
        default=MNIST_DEFAULT_FLATTEN,
        description="Flatten images to 1D (784 features) or keep 2D (28x28)",
    )
    normalize: bool = Field(
        default=MNIST_DEFAULT_NORMALIZE,
        description="Normalize pixel values to [0, 1]",
    )
    one_hot_labels: bool = Field(
        default=MNIST_DEFAULT_ONE_HOT_LABELS,
        description="One-hot encode labels (10 classes)",
    )
    seed: int | None = Field(default=DEFAULT_GENERATOR_SEED, ge=0, description="Random seed for reproducibility. Defaults to DEFAULT_GENERATOR_SEED so the documented default configuration is REPRODUCIBLE (juniper-data#319); pass None explicitly to opt into a fresh draw per call.")
    train_ratio: float = Field(default=MNIST_DEFAULT_TRAIN_RATIO, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=MNIST_DEFAULT_TEST_RATIO, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
