"""Hugging Face datasets integration for loading external datasets."""

import operator
from datetime import UTC, datetime
from typing import Any

import numpy as np

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
            raise ImportError("Hugging Face datasets package not installed. Install with: pip install datasets")

        # JD-PERF-02: without this the metadata cache is inert for this store.
        super().__init__()
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
        train_ratio: float = EXTERNAL_STORE_DEFAULT_TRAIN_RATIO,
        val_ratio: float = EXTERNAL_STORE_DEFAULT_VAL_RATIO,
        test_ratio: float = EXTERNAL_STORE_DEFAULT_TEST_RATIO,
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
        # assert hf_load_dataset is not None

        # Plain JSON types FIRST, because the dataset ID is a JSON hash of these values. np.int64 seeds,
        # datasets.Split.TRAIN (a NamedSplit) and ndarray column lists are not JSON-serialisable.
        # operator.index rejects a float seed instead of truncating it. `split` is also passed to
        # the Hub as a plain string, which it accepts.
        seed = None if seed is None else operator.index(seed)
        train_ratio, val_ratio, test_ratio = float(train_ratio), float(val_ratio), float(test_ratio)
        split = str(split)
        feature_columns = None if feature_columns is None else [str(c) for c in feature_columns]
        label_column = str(label_column)
        # Then validate, and before the download: a bad request must not cost a Hub fetch. After
        # conversion, not before, so this check and the carve's see the SAME numbers.
        # np.float32(0.8) + 0.1 + 0.1 is exactly 1.0 in float32 and 1.0000000119 once widened.
        validate_carve_ratios(train_ratio, val_ratio, test_ratio)

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

        # Three partitions, carved, and no *_full (decision 11; juniper-data#411). The rows
        # are cut in their current order -- the seeded shuffle above is the only shuffle.
        arrays, counts = carve_three_way(X, y, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        n_emitted = counts["n_total"]

        # Decision 7: a data-derived scale is fit on train ONLY and applied unchanged to val and
        # test. It used to be fit on every row before the cut, so val / test statistics leaked
        # into train's scaling. Images are exempt: their /255 is a constant, not a fit.
        if normalize and not self._is_image_source(self._resolve_feature_columns(ds, feature_columns, label_column)):
            x_train = arrays["X_train"]
            scale = float(x_train.max()) if x_train.size else 0.0
            if scale > 1.0:
                for part in ("train", "val", "test"):
                    arrays[f"X_{part}"] = arrays[f"X_{part}"] / np.float32(scale)

        params = {
            "dataset_name": dataset_name,
            "config_name": config_name,
            "split": split,
            "n_samples": len(X),
            "seed": seed,
            "flatten": flatten,
            "normalize": normalize,
            "one_hot_labels": one_hot_labels,
            "feature_columns": feature_columns,
            "label_column": label_column,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        }
        config_suffix = f"-{config_name}" if config_name else ""
        dataset_id = external_dataset_id(f"hf-{dataset_name}{config_suffix}", "huggingface", params)

        # Class counts over the rows actually emitted, not rows a sub-1.0 ratio sum left out.
        class_indices = (y.argmax(axis=1) if one_hot_labels else y.flatten().astype(int))[:n_emitted]
        class_distribution = {str(i): int((class_indices == i).sum()) for i in range(n_classes)}
        meta = DatasetMeta(
            dataset_id=dataset_id,
            generator="huggingface",
            generator_version=VERSION,
            params=params,
            n_samples=n_emitted,
            n_features=X.shape[1] if len(X.shape) > 1 else 1,
            n_classes=n_classes,
            n_train=counts["n_train"],
            n_val=counts["n_val"],
            n_test=counts["n_test"],
            class_distribution=class_distribution,
            created_at=datetime.now(UTC),
            tags=["huggingface", dataset_name],
        )

        self._cache_store.save(dataset_id, meta, arrays)
        # Bypasses this store's own ``save``, so invalidate here too -- a lazy
        # download that populated the cache store must not stay invisible.
        self._invalidate_metadata_cache()

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

        ``normalize`` applies to IMAGE sources only, whose ``/ 255`` is a constant. Tabular
        features come back unscaled: their scale is data-derived, so :meth:`load_hf_dataset`
        fits it on the train partition after the carve (decision 7).

        Returns:
            Tuple of (X, y, n_classes).
        """
        feature_columns = self._resolve_feature_columns(ds, feature_columns, label_column)

        if self._is_image_source(feature_columns):
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

        labels = np.array(ds[label_column])
        n_classes = int(labels.max()) + 1

        if one_hot_labels:
            y = np.zeros((len(labels), n_classes), dtype=np.float32)
            y[np.arange(len(labels)), labels] = 1.0
        else:
            y = labels.astype(np.float32).reshape(-1, 1)

        return X, y, n_classes

    @staticmethod
    def _resolve_feature_columns(ds: Any, feature_columns: list[str] | None, label_column: str) -> list[str]:
        """The feature columns: as given, or every column but the label and id columns."""
        if feature_columns is None:
            return [col for col in ds.column_names if col not in (label_column, "idx", "id")]
        return feature_columns

    @staticmethod
    def _is_image_source(feature_columns: list[str]) -> bool:
        """A single column named like an image is read as pixels (scaled by the constant 255)."""
        return len(feature_columns) == 1 and "image" in feature_columns[0].lower()

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
