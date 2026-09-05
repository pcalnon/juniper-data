"""MNIST dataset generator using Hugging Face datasets.

This module provides the MnistGenerator class for loading and preprocessing
MNIST and Fashion-MNIST datasets from the Hugging Face Hub.
"""

import numpy as np

from juniper_data.core.split import partition_and_assemble, resolve_counts_for_params

from .params import MnistParams

VERSION = "2.0.0"

try:
    from datasets import load_dataset as hf_load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    hf_load_dataset = None  # type: ignore[assignment]

# Canonical Hub repositories for the public parameter values. Bare canonical
# names ("mnist") are rejected by the huggingface-hub 1.x URI layer used by
# datasets >= 5 ("Repository id must be 'namespace/name'"), so the stable
# MnistParams.dataset values map to the namespaced repos the Hub migrated the
# canonical datasets to. Namespaced ids load identically on older datasets
# versions, so this mapping is safe across the entire supported range.
_HF_DATASET_REPOS = {
    "mnist": "ylecun/mnist",
    "fashion_mnist": "zalando-datasets/fashion_mnist",
}


class MnistGenerator:
    """Generator for MNIST and Fashion-MNIST datasets.

    Loads datasets from Hugging Face Hub and converts them to the
    JuniperData format with train/test splits.

    Requires the `datasets` package: pip install datasets

    All methods are static to ensure the generator is stateless and side-effect free.
    """

    @staticmethod
    def is_available() -> bool:
        """Report whether this generator can run in this deployment (D1 / I-5).

        Returns:
            True when the optional Hugging Face ``datasets`` dependency is
            importable; False otherwise (``generate`` would raise ImportError).
        """
        return HF_AVAILABLE

    @staticmethod
    def install_hint() -> str:
        """Report how to make this generator available (W-4, companion to ``is_available``).

        Single source of truth: ``generate`` raises this exact text, so the hint on
        ``GET /v1/generators`` and the 501 detail on ``POST /v1/datasets`` cannot drift.

        Returns:
            The curated, actionable install instruction for the missing dependency.
        """
        return "Hugging Face datasets package not installed. Install with: pip install datasets"

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
            raise ImportError(MnistGenerator.install_hint())

        X, y = MnistGenerator._load_and_preprocess(params)

        # Carve only: MNIST reads a fixed corpus, so there is no way to generate
        # additional rows to honour a requested train count. The params model
        # rejects additive sizing outright rather than accepting it and quietly
        # carving anyway.
        counts = resolve_counts_for_params(params, X.shape[0])

        return partition_and_assemble(X, y, counts, params.seed, params.shuffle)

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
        # so the repo id passed to hf_load_dataset comes from the closed _HF_DATASET_REPOS mapping
        # of known-safe canonical Hub repositories.
        ds = hf_load_dataset(_HF_DATASET_REPOS[params.dataset], split="train")  # nosec B615

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
