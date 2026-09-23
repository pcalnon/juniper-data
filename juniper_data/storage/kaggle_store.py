"""Kaggle datasets integration for downloading and caching datasets."""

import operator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from juniper_data.core.constants import CHARSET_UTF8
from juniper_data.core.models import DatasetMeta

from .base import DatasetStore
from .external_partition import (
    EXTERNAL_STORE_DEFAULT_TEST_RATIO,
    EXTERNAL_STORE_DEFAULT_TRAIN_RATIO,
    EXTERNAL_STORE_DEFAULT_VAL_RATIO,
    EXTERNAL_STORE_VERSION,
    carve_three_way,
    external_dataset_id,
    validate_carve_ratios,
)
from .memory import InMemoryDatasetStore

#: Contract version of the emitted arrays (juniper-data#411). Module-level, like a
#: generator's ``VERSION``, so the decision-11 floor guard can enumerate it.
VERSION: str = EXTERNAL_STORE_VERSION

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
        download_path: Path | None = None,
        cache_store: DatasetStore | None = None,
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
            raise ImportError("Kaggle package not installed. Install with: pip install kaggle")

        self._download_path = download_path or Path("./data/kaggle")
        self._download_path.mkdir(parents=True, exist_ok=True)
        # JD-PERF-02: without this the metadata cache is inert for this store.
        super().__init__()
        self._cache_store = cache_store or InMemoryDatasetStore()

        self._api: Any | None = None
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
        feature_columns: list[str] | None = None,
        label_column: str = "label",
        delimiter: str = ",",
        n_samples: int | None = None,
        seed: int | None = None,
        one_hot_labels: bool = True,
        normalize_features: bool = False,
        train_ratio: float = EXTERNAL_STORE_DEFAULT_TRAIN_RATIO,
        val_ratio: float = EXTERNAL_STORE_DEFAULT_VAL_RATIO,
        test_ratio: float = EXTERNAL_STORE_DEFAULT_TEST_RATIO,
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
            train_ratio: Train's share of the loaded rows.
            val_ratio: Val's share of the loaded rows.
            test_ratio: Test's share of the loaded rows. The three may not sum to more
                than 1; rows beyond their sum are left out.

        Returns:
            Tuple of (dataset_id, metadata, arrays). ``arrays`` holds exactly the
            decision-11 contract: ``X_train``, ``y_train``, ``X_val``, ``y_val``,
            ``X_test``, ``y_test`` -- no ``*_full`` (juniper-data#411).

        Raises:
            ValueError: If the ratios are invalid (see :func:`carve_three_way`).
        """
        # Plain JSON types FIRST, because the dataset ID is a JSON hash of these values.
        # random.seed() also rejects np.int64 on Python >= 3.11. operator.index rejects a float
        # seed instead of truncating it.
        seed = None if seed is None else operator.index(seed)
        train_ratio, val_ratio, test_ratio = float(train_ratio), float(val_ratio), float(test_ratio)
        feature_columns = None if feature_columns is None else [str(c) for c in feature_columns]
        label_column = str(label_column)
        file_name = str(file_name)
        # Then validate, and before the download: a bad request must not cost a Kaggle fetch.
        # After conversion, not before, so this check and the carve's see the SAME numbers.
        validate_carve_ratios(train_ratio, val_ratio, test_ratio)

        dataset_path = self.download_dataset(dataset_ref)
        file_path = dataset_path / file_name

        if not file_path.exists():
            all_files = list(dataset_path.glob("**/*"))
            csv_files = [f for f in all_files if f.suffix.lower() == ".csv"]
            if csv_files:
                file_path = csv_files[0]
            else:
                raise FileNotFoundError(f"File '{file_name}' not found in dataset. Available files: {[f.name for f in all_files]}")

        import csv

        data = []
        with open(file_path, encoding=CHARSET_UTF8) as f:
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

        unique_labels = sorted([str(lbl) for lbl in set(labels)])
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        n_classes = len(unique_labels)

        label_indices = np.array([label_to_idx[str(lbl)] for lbl in labels])

        if one_hot_labels:
            y = np.zeros((len(labels), n_classes), dtype=np.float32)
            y[np.arange(len(labels)), label_indices] = 1.0
        else:
            y = label_indices.astype(np.float32).reshape(-1, 1)

        # Three partitions, carved, and no *_full (decision 11; juniper-data#411). The rows
        # are cut in their current order -- the seeded shuffle above is the only shuffle.
        arrays, counts = carve_three_way(X, y, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        n_emitted = counts["n_total"]

        # Decision 7: min-max statistics are fit on train ONLY and applied unchanged to val and
        # test, so held-out values may fall outside [0, 1]. They used to be fit on every row
        # before the cut, which leaked val / test statistics into train's scaling. An empty
        # train partition has nothing to fit and is left unscaled.
        if normalize_features and arrays["X_train"].shape[0] > 0:
            x_min = arrays["X_train"].min(axis=0, keepdims=True)
            x_range = arrays["X_train"].max(axis=0, keepdims=True) - x_min
            x_range[x_range == 0] = 1
            for part in ("train", "val", "test"):
                arrays[f"X_{part}"] = (arrays[f"X_{part}"] - x_min) / x_range

        # Class counts over the rows actually emitted, not rows a sub-1.0 ratio sum left out.
        emitted_indices = label_indices[:n_emitted]
        class_distribution = {str(i): int((emitted_indices == i).sum()) for i in range(n_classes)}

        params = {
            "dataset_ref": dataset_ref,
            "file_name": file_name,
            "n_samples": len(X),
            "seed": seed,
            "normalize_features": normalize_features,
            "one_hot_labels": one_hot_labels,
            "feature_columns": feature_columns,
            "label_column": label_column,
            "delimiter": delimiter,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        }
        dataset_id = external_dataset_id(f"kaggle-{dataset_ref.replace('/', '-')}", "kaggle", params)

        meta = DatasetMeta(
            dataset_id=dataset_id,
            generator="kaggle",
            generator_version=VERSION,
            params=params,
            n_samples=n_emitted,
            n_features=X.shape[1],
            n_classes=n_classes,
            n_train=counts["n_train"],
            n_val=counts["n_val"],
            n_test=counts["n_test"],
            class_distribution=class_distribution,
            created_at=datetime.now(UTC),
            tags=["kaggle", dataset_ref.split("/")[0]],
        )

        self._cache_store.save(dataset_id, meta, arrays)
        # Bypasses this store's own ``save``, so invalidate here too.
        self._invalidate_metadata_cache()

        return dataset_id, meta, arrays

    def list_competitions(self, search: str | None = None) -> list[dict]:
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

    def list_kaggle_datasets(self, search: str | None = None, page: int = 1) -> list[dict]:
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
        self._invalidate_metadata_cache()

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
        deleted = self._cache_store.delete(dataset_id)
        if deleted:
            self._invalidate_metadata_cache()
        return deleted

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List from cache store."""
        return self._cache_store.list_datasets(limit, offset)

    def update_meta(self, dataset_id: str, meta: DatasetMeta) -> bool:
        """Update in cache store."""
        updated = self._cache_store.update_meta(dataset_id, meta)
        if updated:
            self._invalidate_metadata_cache()
        return updated

    def list_all_metadata(self) -> list[DatasetMeta]:
        """List from cache store."""
        return self._cache_store.list_all_metadata()
