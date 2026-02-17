"""Hugging Face datasets integration for loading external datasets."""

from datetime import datetime, timezone
from typing import Any

import numpy as np

from juniper_data.core.models import DatasetMeta

from .base import DatasetStore
from .memory import InMemoryDatasetStore

try:
    from datasets import load_dataset as hf_load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    hf_load_dataset = None  # type: ignore[assignment]


class HuggingFaceDatasetStore(DatasetStore):
    """Read-only store for loading datasets from Hugging Face Hub.

    Loads datasets from Hugging Face and converts them to JuniperData format.
    Primarily used as a data source, not for persistent storage.

    Requires the `datasets` package: pip install datasets
    """

    def __init__(
        self,
        cache_store: DatasetStore | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the HF store.

        Args:
            cache_store: Optional store for caching loaded datasets.
            cache_dir: Optional local directory for HF dataset cache.

        Raises:
            ImportError: If datasets package is not installed.
        """
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face datasets package not installed. " "Install with: pip install datasets")

        self._cache_store = cache_store or InMemoryDatasetStore()
        self._cache_dir = cache_dir

    def load_hf_dataset(
        self,
        dataset_name: str,
        config_name: str | None = None,
        split: str = "train",
        feature_columns: list[str] | None = None,
        label_column: str = "label",
        n_samples: int | None = None,
        seed: int | None = None,
        flatten: bool = True,
        normalize: bool = True,
        one_hot_labels: bool = True,
        train_ratio: float = 0.8,
    ) -> tuple[str, DatasetMeta, dict[str, np.ndarray]]:
        """Load a dataset from Hugging Face and convert to JuniperData format.

        Args:
            dataset_name: HF dataset name (e.g., "mnist", "fashion_mnist").
            config_name: Optional dataset configuration.
            split: Dataset split to load.
            feature_columns: Column names for features (auto-detected if None).
            label_column: Column name for labels.
            n_samples: Optional limit on number of samples.
            seed: Random seed for shuffling/sampling.
            flatten: Flatten image data to 1D.
            normalize: Normalize features to [0, 1].
            one_hot_labels: One-hot encode labels.
            train_ratio: Ratio for train/test split.

        Returns:
            Tuple of (dataset_id, metadata, arrays).
        """
        # assert hf_load_dataset is not None

        ds = hf_load_dataset(  # nosec B615
            dataset_name,
            config_name,
            split=split,
            cache_dir=self._cache_dir,
        )

        if seed is not None:
            ds = ds.shuffle(seed=seed)

        if n_samples is not None:
            ds = ds.select(range(min(n_samples, len(ds))))

        X, y, n_classes = self._extract_features_labels(
            ds,
            feature_columns=feature_columns,
            label_column=label_column,
            flatten=flatten,
            normalize=normalize,
            one_hot_labels=one_hot_labels,
        )

        n_train = int(len(X) * train_ratio)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        config_suffix = f"-{config_name}" if config_name else ""
        dataset_id = f"hf-{dataset_name}{config_suffix}-{len(X)}"

        class_indices = y.argmax(axis=1) if one_hot_labels else y.flatten().astype(int)
        class_distribution = {str(i): int((class_indices == i).sum()) for i in range(n_classes)}
        meta = DatasetMeta(
            dataset_id=dataset_id,
            generator="huggingface",
            generator_version="1.0.0",
            params={
                "dataset_name": dataset_name,
                "config_name": config_name,
                "split": split,
                "n_samples": len(X),
                "seed": seed,
                "flatten": flatten,
                "normalize": normalize,
                "one_hot_labels": one_hot_labels,
            },
            n_samples=len(X),
            n_features=X.shape[1] if len(X.shape) > 1 else 1,
            n_classes=n_classes,
            n_train=n_train,
            n_test=len(X) - n_train,
            class_distribution=class_distribution,
            created_at=datetime.now(timezone.utc),
            tags=["huggingface", dataset_name],
        )

        arrays = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_full": X,
            "y_full": y,
        }

        self._cache_store.save(dataset_id, meta, arrays)

        return dataset_id, meta, arrays

    def _extract_features_labels(
        self,
        ds: Any,
        feature_columns: list[str] | None,
        label_column: str,
        flatten: bool,
        normalize: bool,
        one_hot_labels: bool,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Extract features and labels from HF dataset.

        Returns:
            Tuple of (X, y, n_classes).
        """
        if feature_columns is None:
            feature_columns = [col for col in ds.column_names if col not in (label_column, "idx", "id")]

        if len(feature_columns) == 1 and "image" in feature_columns[0].lower():
            X = self._extract_images(ds, feature_columns[0], flatten, normalize)
        else:
            features = []
            for col in feature_columns:
                col_data = ds[col]
                if hasattr(col_data[0], "numpy"):
                    col_data = [x.numpy() for x in col_data]
                features.append(np.array(col_data))
            X = np.column_stack(features) if len(features) > 1 else features[0]
            X = X.astype(np.float32)
            if normalize and X.max() > 1.0:
                X = X / X.max()

        labels = np.array(ds[label_column])
        n_classes = int(labels.max()) + 1

        if one_hot_labels:
            y = np.zeros((len(labels), n_classes), dtype=np.float32)
            y[np.arange(len(labels)), labels] = 1.0
        else:
            y = labels.astype(np.float32).reshape(-1, 1)

        return X, y, n_classes

    def _extract_images(
        self,
        ds: Any,
        image_column: str,
        flatten: bool,
        normalize: bool,
    ) -> np.ndarray:
        """Extract and preprocess image data."""
        images = []
        for item in ds:
            img = item[image_column]
            if hasattr(img, "convert"):
                img = np.array(img.convert("L"))
            elif hasattr(img, "numpy"):
                img = img.numpy()
            else:
                img = np.array(img)
            images.append(img)

        X = np.stack(images)

        X = X.astype(np.float32) / 255.0 if normalize else X.astype(np.float32)
        if flatten:
            X = X.reshape(len(X), -1)

        return X

    def save(
        self,
        dataset_id: str,
        meta: DatasetMeta,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Save to cache store."""
        self._cache_store.save(dataset_id, meta, arrays)

    def get_meta(self, dataset_id: str) -> DatasetMeta | None:
        """Get from cache store."""
        return self._cache_store.get_meta(dataset_id)

    def get_artifact_bytes(self, dataset_id: str) -> bytes | None:
        """Get from cache store."""
        return self._cache_store.get_artifact_bytes(dataset_id)

    def exists(self, dataset_id: str) -> bool:
        """Check cache store."""
        return self._cache_store.exists(dataset_id)

    def delete(self, dataset_id: str) -> bool:
        """Delete from cache store."""
        return self._cache_store.delete(dataset_id)

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List from cache store."""
        return self._cache_store.list_datasets(limit, offset)

    def update_meta(self, dataset_id: str, meta: DatasetMeta) -> bool:
        """Update in cache store."""
        return self._cache_store.update_meta(dataset_id, meta)

    def list_all_metadata(self) -> list[DatasetMeta]:
        """List from cache store."""
        return self._cache_store.list_all_metadata()
