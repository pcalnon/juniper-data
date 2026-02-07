"""Kaggle datasets integration for downloading and caching datasets."""

import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from juniper_data.core.models import DatasetMeta

from .base import DatasetStore
from .memory import InMemoryDatasetStore

try:
    from kaggle.api.kaggle_api_extended import KaggleApi

    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    KaggleApi = None  # type: ignore[assignment, misc]


class KaggleDatasetStore(DatasetStore):
    """Kaggle API integration for downloading datasets.

    Downloads datasets from Kaggle and caches them locally.
    Primarily used as a data source, not for persistent storage.

    Requires the `kaggle` package: pip install kaggle
    Also requires Kaggle API credentials in ~/.kaggle/kaggle.json
    or via KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    """

    def __init__(
        self,
        download_path: Optional[Path] = None,
        cache_store: Optional[DatasetStore] = None,
        auto_authenticate: bool = True,
    ) -> None:
        """Initialize the Kaggle store.

        Args:
            download_path: Path for downloading and extracting datasets.
            cache_store: Optional store for caching loaded datasets.
            auto_authenticate: Automatically authenticate with Kaggle API.

        Raises:
            ImportError: If kaggle package is not installed.
        """
        if not KAGGLE_AVAILABLE:
            raise ImportError(
                "Kaggle package not installed. "
                "Install with: pip install kaggle"
            )

        self._download_path = download_path or Path("./data/kaggle")
        self._download_path.mkdir(parents=True, exist_ok=True)
        self._cache_store = cache_store or InMemoryDatasetStore()

        self._api: Optional[Any] = None
        if auto_authenticate:
            self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Kaggle API."""
        self._api = KaggleApi()
        self._api.authenticate()

    def download_dataset(
        self,
        dataset_ref: str,
        unzip: bool = True,
        force: bool = False,
    ) -> Path:
        """Download a dataset from Kaggle.

        Args:
            dataset_ref: Dataset reference in format "owner/dataset-name".
            unzip: Whether to unzip downloaded files.
            force: Force re-download even if already exists.

        Returns:
            Path to the downloaded/extracted dataset directory.

        Raises:
            RuntimeError: If authentication failed or API not available.
        """
        if self._api is None:
            raise RuntimeError("Kaggle API not authenticated. Call _authenticate() first.")

        dataset_path = self._download_path / dataset_ref.replace("/", "_")

        if dataset_path.exists() and not force:
            return dataset_path

        dataset_path.mkdir(parents=True, exist_ok=True)

        self._api.dataset_download_files(
            dataset_ref,
            path=str(dataset_path),
            unzip=unzip,
            force=force,
        )

        return dataset_path

    def load_kaggle_dataset(
        self,
        dataset_ref: str,
        file_name: str,
        feature_columns: Optional[list[str]] = None,
        label_column: str = "label",
        delimiter: str = ",",
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        one_hot_labels: bool = True,
        normalize_features: bool = False,
        train_ratio: float = 0.8,
    ) -> tuple[str, DatasetMeta, dict[str, np.ndarray]]:
        """Download and load a CSV dataset from Kaggle.

        Args:
            dataset_ref: Dataset reference in format "owner/dataset-name".
            file_name: Name of the CSV file within the dataset.
            feature_columns: Column names for features (None = auto-detect).
            label_column: Column name for labels.
            delimiter: CSV delimiter.
            n_samples: Optional limit on number of samples.
            seed: Random seed for shuffling.
            one_hot_labels: One-hot encode labels.
            normalize_features: Normalize features to [0, 1].
            train_ratio: Ratio for train/test split.

        Returns:
            Tuple of (dataset_id, metadata, arrays).
        """
        dataset_path = self.download_dataset(dataset_ref)
        file_path = dataset_path / file_name

        if not file_path.exists():
            all_files = list(dataset_path.glob("**/*"))
            csv_files = [f for f in all_files if f.suffix.lower() == ".csv"]
            if csv_files:
                file_path = csv_files[0]
            else:
                raise FileNotFoundError(
                    f"File '{file_name}' not found in dataset. "
                    f"Available files: {[f.name for f in all_files]}"
                )

        import csv

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                data.append(row)

        if not data:
            raise ValueError("No data found in CSV file")

        if seed is not None:
            import random

            random.seed(seed)
            random.shuffle(data)

        if n_samples is not None:
            data = data[:n_samples]

        all_columns = list(data[0].keys())
        if feature_columns is None:
            feature_columns = [c for c in all_columns if c != label_column]

        features = []
        labels = []

        for row in data:
            feature_row = []
            for col in feature_columns:
                val = row.get(col, 0)
                try:
                    feature_row.append(float(val))
                except (ValueError, TypeError):
                    feature_row.append(0.0)
            features.append(feature_row)
            labels.append(row.get(label_column))

        X = np.array(features, dtype=np.float32)

        if normalize_features:
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            X_range = X_max - X_min
            X_range[X_range == 0] = 1
            X = (X - X_min) / X_range

        unique_labels = sorted([str(l) for l in set(labels)])
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        n_classes = len(unique_labels)

        label_indices = np.array([label_to_idx[str(l)] for l in labels])

        if one_hot_labels:
            y = np.zeros((len(labels), n_classes), dtype=np.float32)
            y[np.arange(len(labels)), label_indices] = 1.0
        else:
            y = label_indices.astype(np.float32).reshape(-1, 1)

        class_distribution = {}
        for i in range(n_classes):
            class_distribution[str(i)] = int((label_indices == i).sum())

        n_train = int(len(X) * train_ratio)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        dataset_id = f"kaggle-{dataset_ref.replace('/', '-')}-{len(X)}"

        meta = DatasetMeta(
            dataset_id=dataset_id,
            generator="kaggle",
            generator_version="1.0.0",
            params={
                "dataset_ref": dataset_ref,
                "file_name": file_name,
                "n_samples": len(X),
                "seed": seed,
                "normalize_features": normalize_features,
                "one_hot_labels": one_hot_labels,
            },
            n_samples=len(X),
            n_features=X.shape[1],
            n_classes=n_classes,
            n_train=n_train,
            n_test=len(X) - n_train,
            class_distribution=class_distribution,
            created_at=datetime.now(timezone.utc),
            tags=["kaggle", dataset_ref.split("/")[0]],
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

    def list_competitions(self, search: Optional[str] = None) -> list[dict]:
        """List available Kaggle competitions.

        Args:
            search: Optional search term.

        Returns:
            List of competition info dictionaries.
        """
        if self._api is None:
            raise RuntimeError("Kaggle API not authenticated.")

        competitions = self._api.competitions_list(search=search)
        return [
            {
                "ref": c.ref,
                "title": c.title,
                "deadline": c.deadline,
                "category": c.category,
            }
            for c in competitions
        ]

    def list_kaggle_datasets(
        self, search: Optional[str] = None, page: int = 1
    ) -> list[dict]:
        """List available Kaggle datasets.

        Args:
            search: Optional search term.
            page: Page number for pagination.

        Returns:
            List of dataset info dictionaries.
        """
        if self._api is None:
            raise RuntimeError("Kaggle API not authenticated.")

        datasets = self._api.dataset_list(search=search, page=page)
        return [
            {
                "ref": d.ref,
                "title": d.title,
                "size": d.totalBytes,
                "lastUpdated": d.lastUpdated,
            }
            for d in datasets
        ]

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
