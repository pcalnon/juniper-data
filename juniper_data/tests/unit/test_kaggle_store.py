"""Unit tests for KaggleDatasetStore."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture
def sample_meta() -> DatasetMeta:
    """Create sample metadata."""
    return DatasetMeta(
        dataset_id="test-dataset",
        generator="test",
        generator_version="1.0.0",
        params={"seed": 42},
        n_samples=100,
        n_features=2,
        n_classes=2,
        n_train=80,
        n_test=20,
        class_distribution={"0": 50, "1": 50},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_arrays() -> dict[str, np.ndarray]:
    """Create sample arrays."""
    rng = np.random.default_rng(42)
    return {
        "X_train": rng.standard_normal((80, 2)).astype(np.float32),
        "y_train": rng.standard_normal((80, 2)).astype(np.float32),
        "X_test": rng.standard_normal((20, 2)).astype(np.float32),
        "y_test": rng.standard_normal((20, 2)).astype(np.float32),
    }


@pytest.fixture
def mock_kaggle_module():
    """Create a mock kaggle module and patch it into kaggle_store."""
    mock_api_class = MagicMock()
    mock_api_instance = MagicMock()
    mock_api_class.return_value = mock_api_instance

    with patch("juniper_data.storage.kaggle_store.KAGGLE_AVAILABLE", True):
        with patch("juniper_data.storage.kaggle_store.KaggleApi", mock_api_class):
            yield mock_api_class, mock_api_instance


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Helper to write a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.unit
@pytest.mark.storage
class TestKaggleDatasetStoreInit:
    """Tests for KaggleDatasetStore initialization."""

    def test_init_default(self, mock_kaggle_module, tmp_path) -> None:
        """Initialize with default parameters."""
        mock_api_class, mock_api_instance = mock_kaggle_module
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        mock_api_instance.authenticate.assert_called_once()
        assert isinstance(store._cache_store, InMemoryDatasetStore)

    def test_init_custom_cache_store(self, mock_kaggle_module, tmp_path) -> None:
        """Initialize with custom cache store."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        custom_cache = InMemoryDatasetStore()
        store = KaggleDatasetStore(download_path=tmp_path / "kaggle", cache_store=custom_cache)
        assert store._cache_store is custom_cache

    def test_init_no_auto_authenticate(self, mock_kaggle_module, tmp_path) -> None:
        """Initialize without auto authentication."""
        _, mock_api_instance = mock_kaggle_module
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle", auto_authenticate=False)
        assert store._api is None

    def test_init_raises_without_kaggle(self) -> None:
        """Raises ImportError when kaggle package is not available."""
        with patch("juniper_data.storage.kaggle_store.KAGGLE_AVAILABLE", False):
            from juniper_data.storage.kaggle_store import KaggleDatasetStore

            with pytest.raises(ImportError, match="Kaggle package not installed"):
                KaggleDatasetStore()


@pytest.mark.unit
@pytest.mark.storage
class TestKaggleDatasetStoreDownload:
    """Tests for download_dataset operation."""

    def test_download_dataset(self, mock_kaggle_module, tmp_path) -> None:
        """Download a new dataset."""
        _, mock_api_instance = mock_kaggle_module
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")

        result = store.download_dataset("owner/dataset-name")
        assert isinstance(result, Path)
        mock_api_instance.dataset_download_files.assert_called_once()

    def test_download_dataset_cached(self, mock_kaggle_module, tmp_path) -> None:
        """Skip download when dataset directory exists."""
        _, mock_api_instance = mock_kaggle_module
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        cached_path = tmp_path / "kaggle" / "owner_dataset-name"
        cached_path.mkdir(parents=True, exist_ok=True)

        result = store.download_dataset("owner/dataset-name")
        assert result == cached_path
        mock_api_instance.dataset_download_files.assert_not_called()

    def test_download_dataset_force(self, mock_kaggle_module, tmp_path) -> None:
        """Force re-download even when cached."""
        _, mock_api_instance = mock_kaggle_module
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        cached_path = tmp_path / "kaggle" / "owner_dataset-name"
        cached_path.mkdir(parents=True, exist_ok=True)

        store.download_dataset("owner/dataset-name", force=True)
        mock_api_instance.dataset_download_files.assert_called_once()

    def test_download_dataset_not_authenticated(self, mock_kaggle_module, tmp_path) -> None:
        """Raises RuntimeError when API not authenticated."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle", auto_authenticate=False)

        with pytest.raises(RuntimeError, match="not authenticated"):
            store.download_dataset("owner/dataset")


@pytest.mark.unit
@pytest.mark.storage
class TestKaggleDatasetStoreLoadDataset:
    """Tests for load_kaggle_dataset operation."""

    def test_load_csv_dataset(self, mock_kaggle_module, tmp_path) -> None:
        """Load a CSV dataset from Kaggle."""
        _, mock_api_instance = mock_kaggle_module
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")

        dataset_dir = tmp_path / "kaggle" / "owner_iris"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"sepal_length": "5.1", "sepal_width": "3.5", "label": "0"},
            {"sepal_length": "7.0", "sepal_width": "3.2", "label": "1"},
            {"sepal_length": "6.3", "sepal_width": "3.3", "label": "2"},
            {"sepal_length": "5.0", "sepal_width": "3.6", "label": "0"},
            {"sepal_length": "6.7", "sepal_width": "3.1", "label": "1"},
        ]
        _write_csv(dataset_dir / "data.csv", rows)

        dataset_id, meta, arrays = store.load_kaggle_dataset("owner/iris", file_name="data.csv")

        assert "kaggle-owner-iris" in dataset_id
        assert meta.generator == "kaggle"
        assert meta.n_samples == 5
        assert meta.n_features == 2
        assert meta.n_classes == 3
        assert arrays["X_full"].shape == (5, 2)

    def test_load_with_auto_detect_csv(self, mock_kaggle_module, tmp_path) -> None:
        """Auto-detect CSV when specified file not found."""
        _, mock_api_instance = mock_kaggle_module
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")

        dataset_dir = tmp_path / "kaggle" / "owner_test"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"feature": "1.0", "label": "a"},
            {"feature": "2.0", "label": "b"},
        ]
        _write_csv(dataset_dir / "actual.csv", rows)

        dataset_id, meta, arrays = store.load_kaggle_dataset("owner/test", file_name="missing.csv")
        assert meta.n_samples == 2

    def test_load_file_not_found(self, mock_kaggle_module, tmp_path) -> None:
        """Raises FileNotFoundError when no CSV found."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_test"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="not found"):
            store.load_kaggle_dataset("owner/test", file_name="missing.csv")

    def test_load_empty_csv(self, mock_kaggle_module, tmp_path) -> None:
        """Raises ValueError when CSV is empty."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_empty"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        empty_csv = dataset_dir / "data.csv"
        empty_csv.write_text("col1,col2,label\n")

        with pytest.raises(ValueError, match="No data found"):
            store.load_kaggle_dataset("owner/empty", file_name="data.csv")

    def test_load_with_seed(self, mock_kaggle_module, tmp_path) -> None:
        """Load with seed shuffles data."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_seed"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i), "label": str(i % 2)} for i in range(10)]
        _write_csv(dataset_dir / "data.csv", rows)

        _, meta1, arrays1 = store.load_kaggle_dataset("owner/seed", file_name="data.csv", seed=42)
        _, meta2, arrays2 = store.load_kaggle_dataset("owner/seed", file_name="data.csv", seed=42)

        np.testing.assert_array_equal(arrays1["X_full"], arrays2["X_full"])

    def test_load_with_n_samples(self, mock_kaggle_module, tmp_path) -> None:
        """Load with n_samples limits data."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_limit"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i), "label": str(i % 2)} for i in range(20)]
        _write_csv(dataset_dir / "data.csv", rows)

        _, meta, _ = store.load_kaggle_dataset("owner/limit", file_name="data.csv", n_samples=5)
        assert meta.n_samples == 5

    def test_load_without_one_hot(self, mock_kaggle_module, tmp_path) -> None:
        """Load without one-hot encoding."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_nohot"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i), "label": str(i % 2)} for i in range(10)]
        _write_csv(dataset_dir / "data.csv", rows)

        _, _, arrays = store.load_kaggle_dataset("owner/nohot", file_name="data.csv", one_hot_labels=False)
        assert arrays["y_full"].shape[1] == 1

    def test_load_with_normalization(self, mock_kaggle_module, tmp_path) -> None:
        """Load with feature normalization."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_norm"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i * 10), "label": str(i % 2)} for i in range(10)]
        _write_csv(dataset_dir / "data.csv", rows)

        _, _, arrays = store.load_kaggle_dataset("owner/norm", file_name="data.csv", normalize_features=True)
        assert arrays["X_full"].max() <= 1.0 + 1e-6
        assert arrays["X_full"].min() >= 0.0 - 1e-6

    def test_load_with_invalid_values(self, mock_kaggle_module, tmp_path) -> None:
        """Non-numeric feature values are treated as 0.0."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_bad"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"feature": "abc", "label": "0"},
            {"feature": "1.5", "label": "1"},
        ]
        _write_csv(dataset_dir / "data.csv", rows)

        _, _, arrays = store.load_kaggle_dataset("owner/bad", file_name="data.csv")
        assert arrays["X_full"][0, 0] == 0.0
        assert arrays["X_full"][1, 0] == 1.5

    def test_load_with_feature_columns(self, mock_kaggle_module, tmp_path) -> None:
        """Load with explicit feature columns."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_cols"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"a": "1", "b": "2", "c": "3", "label": "0"},
            {"a": "4", "b": "5", "c": "6", "label": "1"},
        ]
        _write_csv(dataset_dir / "data.csv", rows)

        _, meta, arrays = store.load_kaggle_dataset("owner/cols", file_name="data.csv", feature_columns=["a", "b"])
        assert meta.n_features == 2

    def test_load_saves_to_cache(self, mock_kaggle_module, tmp_path) -> None:
        """Load saves the result to cache store."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_cache"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": "1", "label": "0"}, {"feature": "2", "label": "1"}]
        _write_csv(dataset_dir / "data.csv", rows)

        dataset_id, _, _ = store.load_kaggle_dataset("owner/cache", file_name="data.csv")
        assert store._cache_store.exists(dataset_id)


@pytest.mark.unit
@pytest.mark.storage
class TestKaggleDatasetStoreListCompetitions:
    """Tests for list_competitions operation."""

    def test_list_competitions(self, mock_kaggle_module, tmp_path) -> None:
        """List competitions returns formatted results."""
        _, mock_api_instance = mock_kaggle_module
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")

        mock_comp = MagicMock()
        mock_comp.ref = "competition-1"
        mock_comp.title = "Test Competition"
        mock_comp.deadline = "2026-12-31"
        mock_comp.category = "Getting Started"
        mock_api_instance.competitions_list.return_value = [mock_comp]

        result = store.list_competitions(search="test")
        assert len(result) == 1
        assert result[0]["ref"] == "competition-1"
        assert result[0]["title"] == "Test Competition"

    def test_list_competitions_not_authenticated(self, mock_kaggle_module, tmp_path) -> None:
        """Raises RuntimeError when API not authenticated."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle", auto_authenticate=False)

        with pytest.raises(RuntimeError, match="not authenticated"):
            store.list_competitions()


@pytest.mark.unit
@pytest.mark.storage
class TestKaggleDatasetStoreListKaggleDatasets:
    """Tests for list_kaggle_datasets operation."""

    def test_list_kaggle_datasets(self, mock_kaggle_module, tmp_path) -> None:
        """List Kaggle datasets returns formatted results."""
        _, mock_api_instance = mock_kaggle_module
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")

        mock_dataset = MagicMock()
        mock_dataset.ref = "owner/dataset"
        mock_dataset.title = "Test Dataset"
        mock_dataset.totalBytes = 1024
        mock_dataset.lastUpdated = "2026-01-01"
        mock_api_instance.dataset_list.return_value = [mock_dataset]

        result = store.list_kaggle_datasets(search="test", page=2)
        assert len(result) == 1
        assert result[0]["ref"] == "owner/dataset"
        mock_api_instance.dataset_list.assert_called_once_with(search="test", page=2)

    def test_list_kaggle_datasets_not_authenticated(self, mock_kaggle_module, tmp_path) -> None:
        """Raises RuntimeError when API not authenticated."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle", auto_authenticate=False)

        with pytest.raises(RuntimeError, match="not authenticated"):
            store.list_kaggle_datasets()


@pytest.mark.unit
@pytest.mark.storage
class TestKaggleDatasetStoreDelegation:
    """Tests for delegated cache store operations."""

    def test_save_delegates(self, mock_kaggle_module, tmp_path, sample_meta, sample_arrays) -> None:
        """save delegates to cache store."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        store.save("test-1", sample_meta, sample_arrays)
        assert store._cache_store.exists("test-1")

    def test_get_meta_delegates(self, mock_kaggle_module, tmp_path, sample_meta, sample_arrays) -> None:
        """get_meta delegates to cache store."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        store._cache_store.save("test-1", sample_meta, sample_arrays)
        result = store.get_meta("test-1")
        assert result is not None

    def test_get_artifact_bytes_delegates(self, mock_kaggle_module, tmp_path, sample_meta, sample_arrays) -> None:
        """get_artifact_bytes delegates to cache store."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        store._cache_store.save("test-1", sample_meta, sample_arrays)
        result = store.get_artifact_bytes("test-1")
        assert result is not None

    def test_exists_delegates(self, mock_kaggle_module, tmp_path) -> None:
        """exists delegates to cache store."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        assert store.exists("nonexistent") is False

    def test_delete_delegates(self, mock_kaggle_module, tmp_path, sample_meta, sample_arrays) -> None:
        """delete delegates to cache store."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        store._cache_store.save("test-1", sample_meta, sample_arrays)
        assert store.delete("test-1") is True

    def test_list_datasets_delegates(self, mock_kaggle_module, tmp_path, sample_meta, sample_arrays) -> None:
        """list_datasets delegates to cache store."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        store._cache_store.save("test-1", sample_meta, sample_arrays)
        assert "test-1" in store.list_datasets()

    def test_update_meta_delegates(self, mock_kaggle_module, tmp_path, sample_meta, sample_arrays) -> None:
        """update_meta delegates to cache store."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        store._cache_store.save("test-1", sample_meta, sample_arrays)
        assert store.update_meta("test-1", sample_meta) is True

    def test_list_all_metadata_delegates(self, mock_kaggle_module, tmp_path, sample_meta, sample_arrays) -> None:
        """list_all_metadata delegates to cache store."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        store._cache_store.save("test-1", sample_meta, sample_arrays)
        result = store.list_all_metadata()
        assert len(result) == 1
