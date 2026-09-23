"""Unit tests for KaggleDatasetStore."""

import csv
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.memory import InMemoryDatasetStore
from juniper_data.tests.partitions import whole


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
        created_at=datetime.now(UTC),
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
        assert whole(arrays, "X").shape == (5, 2)
        # juniper-data#411: exactly the decision-11 contract -- three partitions, no *_full.
        assert set(arrays) == {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
        assert meta.generator_version == "3.0.0"
        assert "-kaggle-3.0.0-" in dataset_id

    def test_load_carves_three_partitions(self, mock_kaggle_module, tmp_path) -> None:
        """0.8 / 0.1 / 0.1 over 20 rows; meta counts match the arrays; order is kept."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_three"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i), "label": str(i % 2)} for i in range(20)]
        _write_csv(dataset_dir / "data.csv", rows)

        _, meta, arrays = store.load_kaggle_dataset("owner/three", file_name="data.csv")

        assert (meta.n_train, meta.n_val, meta.n_test) == (16, 2, 2)
        assert [arrays[f"X_{p}"].shape[0] for p in ("train", "val", "test")] == [16, 2, 2]
        assert meta.n_samples == 20
        assert sum(meta.class_distribution.values()) == 20
        np.testing.assert_array_equal(whole(arrays, "X")[:, 0], np.arange(20, dtype=np.float32))

    def test_ratios_are_honoured_and_unused_rows_are_left_out(self, mock_kaggle_module, tmp_path) -> None:
        """A ratio sum below 1 leaves the tail out of every partition and out of the meta."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_part"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i), "label": str(i % 2)} for i in range(20)]
        _write_csv(dataset_dir / "data.csv", rows)

        _, meta, arrays = store.load_kaggle_dataset("owner/part", file_name="data.csv", train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)

        assert (meta.n_train, meta.n_val, meta.n_test) == (10, 4, 2)
        assert meta.n_samples == 16
        assert sum(meta.class_distribution.values()) == 16
        assert whole(arrays, "X").shape[0] == 16

    def test_oversubscribed_ratios_raise(self, mock_kaggle_module, tmp_path) -> None:
        """train_ratio=0.9 on its own now over-asks (0.9 + 0.1 + 0.1) and fails loudly."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_over"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i), "label": str(i % 2)} for i in range(10)]
        _write_csv(dataset_dir / "data.csv", rows)

        with pytest.raises(ValueError, match="must be <= 1.0"):
            store.load_kaggle_dataset("owner/over", file_name="data.csv", train_ratio=0.9)

    def test_dataset_id_carries_version_and_params(self, mock_kaggle_module, tmp_path) -> None:
        """Same request -> same ID; a different partitioning -> a different ID."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_ids"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i), "label": str(i % 2)} for i in range(10)]
        _write_csv(dataset_dir / "data.csv", rows)

        first, _, _ = store.load_kaggle_dataset("owner/ids", file_name="data.csv", seed=3)
        again, _, _ = store.load_kaggle_dataset("owner/ids", file_name="data.csv", seed=3)
        other, _, _ = store.load_kaggle_dataset("owner/ids", file_name="data.csv", seed=3, train_ratio=0.7, val_ratio=0.2)

        assert first == again
        assert first != other

    def test_unseeded_loads_reuse_one_id_and_one_cache_entry(self, mock_kaggle_module, tmp_path) -> None:
        """No seed means no shuffle here, so the load is repeatable and must not mint a new ID."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_unseeded"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i), "label": str(i % 2)} for i in range(10)]
        _write_csv(dataset_dir / "data.csv", rows)

        ids = {store.load_kaggle_dataset("owner/unseeded", file_name="data.csv")[0] for _ in range(3)}

        assert len(ids) == 1
        assert store._cache_store.list_datasets() == list(ids)

    def test_numpy_seed_is_accepted_and_hashes_like_an_int(self, mock_kaggle_module, tmp_path) -> None:
        """np.int64 broke both the JSON-hashed ID and random.seed() (Python >= 3.11)."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_npseed"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i), "label": str(i % 2)} for i in range(10)]
        _write_csv(dataset_dir / "data.csv", rows)

        as_numpy, _, arrays_numpy = store.load_kaggle_dataset("owner/npseed", file_name="data.csv", seed=np.int64(3))
        as_int, _, arrays_int = store.load_kaggle_dataset("owner/npseed", file_name="data.csv", seed=3)

        assert as_numpy == as_int
        np.testing.assert_array_equal(whole(arrays_numpy, "X"), whole(arrays_int, "X"))

    def test_invalid_ratios_fail_before_the_download(self, mock_kaggle_module, tmp_path) -> None:
        """A bad request must not cost a Kaggle fetch."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        with patch.object(store, "download_dataset") as download:
            with pytest.raises(ValueError, match="must be <= 1.0"):
                store.load_kaggle_dataset("owner/any", file_name="data.csv", train_ratio=0.9)
            # float32 0.8 / 0.1 / 0.1 sums to 1.0 only in float32; widened it over-asks. Judged
            # as the carve sees it, and so still before the fetch.
            with pytest.raises(ValueError, match="must be <= 1.0"):
                store.load_kaggle_dataset("owner/any", file_name="data.csv", train_ratio=np.float32(0.8), val_ratio=np.float32(0.1), test_ratio=np.float32(0.1))

        download.assert_not_called()

    def test_array_columns_and_path_file_name_hash_as_plain_json(self, mock_kaggle_module, tmp_path) -> None:
        """An ndarray of column names, or a Path file name, is not JSON-serialisable."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_types"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(dataset_dir / "data.csv", [{"a": str(i), "b": str(2 * i), "label": str(i % 2)} for i in range(10)])

        odd_id, meta, _ = store.load_kaggle_dataset("owner/types", file_name=Path("data.csv"), seed=3, feature_columns=np.array(["a", "b"]))
        plain_id, _, _ = store.load_kaggle_dataset("owner/types", file_name="data.csv", seed=3, feature_columns=["a", "b"])

        assert odd_id == plain_id
        assert meta.params["feature_columns"] == ["a", "b"]
        assert meta.params["file_name"] == "data.csv"

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

        np.testing.assert_array_equal(whole(arrays1, "X"), whole(arrays2, "X"))

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
        assert whole(arrays, "y").shape[1] == 1

    def test_load_with_normalization(self, mock_kaggle_module, tmp_path) -> None:
        """Load with feature normalization."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_norm"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"feature": str(i * 10), "label": str(i % 2)} for i in range(10)]
        _write_csv(dataset_dir / "data.csv", rows)

        _, _, arrays = store.load_kaggle_dataset("owner/norm", file_name="data.csv", normalize_features=True)

        # Min-max is fit on train ONLY (decision 7; juniper-data#411). This asserted the WHOLE
        # dataset sat in [0, 1], true only for a fit over every row. Train is rows 0..7 (0..70),
        # so val (80) and test (90) escape the bound under train's statistics.
        assert arrays["X_train"].min() == pytest.approx(0.0)
        assert arrays["X_train"].max() == pytest.approx(1.0)
        np.testing.assert_allclose(arrays["X_val"][:, 0], [80.0 / 70.0], rtol=1e-6)
        np.testing.assert_allclose(arrays["X_test"][:, 0], [90.0 / 70.0], rtol=1e-6)

    def test_empty_train_partition_is_left_unscaled(self, mock_kaggle_module, tmp_path) -> None:
        """Nothing to fit on: the rows pass through raw instead of raising on an empty min()."""
        from juniper_data.storage.kaggle_store import KaggleDatasetStore

        store = KaggleDatasetStore(download_path=tmp_path / "kaggle")
        dataset_dir = tmp_path / "kaggle" / "owner_notrain"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(dataset_dir / "data.csv", [{"feature": "5", "label": "0"}, {"feature": "7", "label": "1"}])

        _, meta, arrays = store.load_kaggle_dataset("owner/notrain", file_name="data.csv", normalize_features=True, train_ratio=0.2, val_ratio=0.4, test_ratio=0.4)

        assert meta.n_train == 0
        np.testing.assert_array_equal(whole(arrays, "X")[:, 0], [5.0, 7.0])

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
        assert whole(arrays, "X")[0, 0] == 0.0
        assert whole(arrays, "X")[1, 0] == 1.5

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
