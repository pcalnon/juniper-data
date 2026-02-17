"""MNIST dataset generator using Hugging Face datasets.

This module provides the MnistGenerator class for loading and preprocessing
MNIST and Fashion-MNIST datasets from the Hugging Face Hub.
"""

import numpy as np

from juniper_data.core.split import shuffle_and_split

from .params import MnistParams

VERSION = "1.0.0"

try:
    from datasets import load_dataset as hf_load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    hf_load_dataset = None  # type: ignore[assignment]


class MnistGenerator:
    """Generator for MNIST and Fashion-MNIST datasets.

    Loads datasets from Hugging Face Hub and converts them to the
    JuniperData format with train/test splits.

    Requires the `datasets` package: pip install datasets

    All methods are static to ensure the generator is stateless and side-effect free.
    """

    @staticmethod
    def generate(params: MnistParams) -> dict[str, np.ndarray]:
        """Generate a complete MNIST dataset with train/test splits.

        Args:
            params: MnistParams instance defining generation configuration.

        Returns:
            Dictionary containing:
                - X_train: Training features
                - y_train: Training labels
                - X_test: Test features
                - y_test: Test labels
                - X_full: Full dataset features
                - y_full: Full dataset labels

        Raises:
            ImportError: If datasets package is not installed.
        """
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face datasets package not installed. " "Install with: pip install datasets")

        X, y = MnistGenerator._load_and_preprocess(params)

        split_result = shuffle_and_split(
            X=X,
            y=y,
            train_ratio=params.train_ratio,
            test_ratio=params.test_ratio,
            seed=params.seed,
            shuffle=params.shuffle,
        )

        return {
            "X_train": split_result["X_train"],
            "y_train": split_result["y_train"],
            "X_test": split_result["X_test"],
            "y_test": split_result["y_test"],
            "X_full": X,
            "y_full": y,
        }

    @staticmethod
    def _load_and_preprocess(params: MnistParams) -> tuple[np.ndarray, np.ndarray]:
        """Load dataset from HuggingFace and preprocess.

        Args:
            params: MnistParams instance.

        Returns:
            Tuple of (X, y) arrays.
        """
        # assert hf_load_dataset is not None

        # params.dataset is validated by MnistParams (Pydantic) as Literal["mnist", "fashion_mnist"],
        # so this argument to hf_load_dataset is restricted to these known-safe values.  # nosec B615
        ds = hf_load_dataset(params.dataset, split="train")

        if params.seed is not None:
            ds = ds.shuffle(seed=params.seed)

        if params.n_samples is not None:
            ds = ds.select(range(params.n_samples))

        # Use bulk column access with numpy formatting for efficient conversion
        ds = ds.with_format("numpy")
        X = np.array(ds["image"])
        X = X.astype(np.float32) / 255.0 if params.normalize else X.astype(np.float32)
        if params.flatten:
            X = X.reshape(len(X), -1)

        labels = np.array(ds["label"])
        if params.one_hot_labels:
            n_classes = 10

            y = np.zeros((len(labels), n_classes), dtype=np.float32)
            y[np.arange(len(labels)), labels] = 1.0
        else:
            y = labels.astype(np.float32).reshape(-1, 1)

        return X, y


def get_schema() -> dict:
    """Return JSON schema describing the generator parameters.

    Returns:
        JSON schema dictionary for MnistParams.
    """
    return MnistParams.model_json_schema()
