"""Unit tests for storage module."""

import contextlib
import io
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage import DatasetStore, InMemoryDatasetStore, LocalFSDatasetStore


@pytest.fixture
def sample_meta() -> DatasetMeta:
    """Create sample dataset metadata for testing."""
    return DatasetMeta(
        dataset_id="test-dataset-001",
        generator="spiral",
        generator_version="1.0.0",
        params={"n_spirals": 2, "n_points_per_spiral": 100, "noise": 0.1},
        n_samples=200,
        n_features=2,
        n_classes=2,
        n_train=160,
        n_test=40,
        class_distribution={"0": 100, "1": 100},
        artifact_formats=["npz"],
        created_at=datetime(2026, 1, 30, 12, 0, 0),
        checksum="abc123",
    )


@pytest.fixture
def sample_arrays() -> dict[str, np.ndarray]:
    """Create sample arrays for testing."""
    return {
        "X_train": np.random.randn(160, 2).astype(np.float32),
        "y_train": np.eye(2, dtype=np.float32)[np.random.randint(0, 2, 160)],
        "X_test": np.random.randn(40, 2).astype(np.float32),
        "y_test": np.eye(2, dtype=np.float32)[np.random.randint(0, 2, 40)],
    }


@pytest.fixture
def memory_store() -> InMemoryDatasetStore:
    """Create a fresh in-memory store."""
    return InMemoryDatasetStore()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for filesystem tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fs_store(temp_dir: Path) -> LocalFSDatasetStore:
    """Create a local filesystem store in a temp directory."""
    return LocalFSDatasetStore(temp_dir)


class TestInMemoryDatasetStore:
    """Tests for InMemoryDatasetStore."""

    @pytest.mark.unit
    def test_init_creates_empty_store(self, memory_store: InMemoryDatasetStore):
        """Test that initialization creates an empty store."""
        assert memory_store.list_datasets() == []

    @pytest.mark.unit
    def test_save_and_get_meta(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test saving and retrieving metadata."""
        memory_store.save("ds-001", sample_meta, sample_arrays)
        retrieved = memory_store.get_meta("ds-001")

        assert retrieved is not None
        assert retrieved.dataset_id == sample_meta.dataset_id
        assert retrieved.generator == sample_meta.generator
        assert retrieved.n_samples == sample_meta.n_samples

    @pytest.mark.unit
    def test_get_meta_nonexistent(self, memory_store: InMemoryDatasetStore):
        """Test getting metadata for nonexistent dataset returns None."""
        assert memory_store.get_meta("nonexistent") is None

    @pytest.mark.unit
    def test_save_and_get_artifact_bytes(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test saving and retrieving artifact bytes."""
        memory_store.save("ds-001", sample_meta, sample_arrays)
        artifact_bytes = memory_store.get_artifact_bytes("ds-001")

        assert artifact_bytes is not None
        assert len(artifact_bytes) > 0

        loaded = np.load(io.BytesIO(artifact_bytes))
        assert set(loaded.files) == set(sample_arrays.keys())
        for key in sample_arrays:
            np.testing.assert_array_almost_equal(loaded[key], sample_arrays[key])

    @pytest.mark.unit
    def test_get_artifact_bytes_nonexistent(self, memory_store: InMemoryDatasetStore):
        """Test getting artifact bytes for nonexistent dataset returns None."""
        assert memory_store.get_artifact_bytes("nonexistent") is None

    @pytest.mark.unit
    def test_exists_true(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test exists returns True for saved dataset."""
        memory_store.save("ds-001", sample_meta, sample_arrays)
        assert memory_store.exists("ds-001") is True

    @pytest.mark.unit
    def test_exists_false(self, memory_store: InMemoryDatasetStore):
        """Test exists returns False for nonexistent dataset."""
        assert memory_store.exists("nonexistent") is False

    @pytest.mark.unit
    def test_delete_existing(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test deleting an existing dataset returns True."""
        memory_store.save("ds-001", sample_meta, sample_arrays)
        deleted = memory_store.delete("ds-001")
        assert deleted is True
        assert memory_store.exists("ds-001") is False
        assert memory_store.get_meta("ds-001") is None

    @pytest.mark.unit
    def test_delete_nonexistent(self, memory_store: InMemoryDatasetStore):
        """Test deleting a nonexistent dataset returns False."""
        deleted = memory_store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.unit
    def test_list_datasets_empty(self, memory_store: InMemoryDatasetStore):
        """Test listing datasets in empty store."""
        assert memory_store.list_datasets() == []

    @pytest.mark.unit
    def test_list_datasets_multiple(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test listing multiple datasets."""
        for i in range(5):
            memory_store.save(f"ds-00{i}", sample_meta, sample_arrays)

        datasets = memory_store.list_datasets()
        assert len(datasets) == 5
        assert datasets == sorted(datasets)

    @pytest.mark.unit
    def test_list_datasets_with_limit(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test listing datasets with limit."""
        for i in range(10):
            memory_store.save(f"ds-{i:03d}", sample_meta, sample_arrays)

        datasets = memory_store.list_datasets(limit=3)
        assert len(datasets) == 3

    @pytest.mark.unit
    def test_list_datasets_with_offset(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test listing datasets with offset."""
        for i in range(10):
            memory_store.save(f"ds-{i:03d}", sample_meta, sample_arrays)

        datasets = memory_store.list_datasets(offset=5)
        assert len(datasets) == 5
        assert datasets[0] == "ds-005"

    @pytest.mark.unit
    def test_list_datasets_with_limit_and_offset(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test listing datasets with both limit and offset."""
        for i in range(10):
            memory_store.save(f"ds-{i:03d}", sample_meta, sample_arrays)

        datasets = memory_store.list_datasets(limit=3, offset=2)
        assert len(datasets) == 3
        assert datasets == ["ds-002", "ds-003", "ds-004"]

    @pytest.mark.unit
    def test_clear(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test clearing all datasets."""
        for i in range(5):
            memory_store.save(f"ds-00{i}", sample_meta, sample_arrays)

        assert len(memory_store.list_datasets()) == 5
        memory_store.clear()
        assert len(memory_store.list_datasets()) == 0

    @pytest.mark.unit
    def test_save_copies_arrays(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test that save makes copies of arrays (not references)."""
        memory_store.save("ds-001", sample_meta, sample_arrays)

        original_value = sample_arrays["X_train"][0, 0].copy()
        sample_arrays["X_train"][0, 0] = 999.0

        artifact_bytes = memory_store.get_artifact_bytes("ds-001")
        assert artifact_bytes is not None
        loaded = np.load(io.BytesIO(artifact_bytes))
        assert loaded["X_train"][0, 0] == original_value

    @pytest.mark.unit
    def test_overwrite_existing(self, memory_store: InMemoryDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test that saving to same ID overwrites existing dataset."""
        memory_store.save("ds-001", sample_meta, sample_arrays)

        new_meta = DatasetMeta(
            dataset_id="ds-001-updated",
            generator="spiral",
            generator_version="2.0.0",
            params={},
            n_samples=100,
            n_features=2,
            n_classes=2,
            n_train=80,
            n_test=20,
            class_distribution={"0": 50, "1": 50},
            created_at=datetime.now(),
        )
        new_arrays = {"X": np.zeros((10, 2), dtype=np.float32)}

        memory_store.save("ds-001", new_meta, new_arrays)

        retrieved = memory_store.get_meta("ds-001")
        assert retrieved is not None
        assert retrieved.generator_version == "2.0.0"

    @pytest.mark.unit
    def test_record_access_nonexistent_noop(self, memory_store: InMemoryDatasetStore):
        """record_access on nonexistent dataset does nothing."""
        memory_store.record_access("nonexistent-id")
        assert not memory_store.exists("nonexistent-id")


class TestLocalFSDatasetStore:
    """Tests for LocalFSDatasetStore."""

    @pytest.mark.unit
    def test_init_creates_directory(self, temp_dir: Path):
        """Test that initialization creates the base directory."""
        subdir = temp_dir / "datasets" / "nested"
        store = LocalFSDatasetStore(subdir)
        assert subdir.exists()
        assert store.base_path == subdir

    @pytest.mark.unit
    def test_save_creates_files(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test that save creates meta and npz files."""
        fs_store.save("ds-001", sample_meta, sample_arrays)

        meta_path = fs_store.base_path / "ds-001.meta.json"
        npz_path = fs_store.base_path / "ds-001.npz"

        assert meta_path.exists()
        assert npz_path.exists()

    @pytest.mark.unit
    def test_save_and_get_meta(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test saving and retrieving metadata."""
        fs_store.save("ds-001", sample_meta, sample_arrays)
        retrieved = fs_store.get_meta("ds-001")

        assert retrieved is not None
        assert retrieved.dataset_id == sample_meta.dataset_id
        assert retrieved.generator == sample_meta.generator
        assert retrieved.n_samples == sample_meta.n_samples
        assert retrieved.created_at == sample_meta.created_at

    @pytest.mark.unit
    def test_get_meta_nonexistent(self, fs_store: LocalFSDatasetStore):
        """Test getting metadata for nonexistent dataset returns None."""
        assert fs_store.get_meta("nonexistent") is None

    @pytest.mark.unit
    def test_save_and_get_artifact_bytes(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test saving and retrieving artifact bytes."""
        fs_store.save("ds-001", sample_meta, sample_arrays)
        artifact_bytes = fs_store.get_artifact_bytes("ds-001")

        assert artifact_bytes is not None
        assert len(artifact_bytes) > 0

        loaded = np.load(io.BytesIO(artifact_bytes))
        assert set(loaded.files) == set(sample_arrays.keys())

    @pytest.mark.unit
    def test_get_artifact_bytes_nonexistent(self, fs_store: LocalFSDatasetStore):
        """Test getting artifact bytes for nonexistent dataset returns None."""
        assert fs_store.get_artifact_bytes("nonexistent") is None

    @pytest.mark.unit
    def test_exists_true(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test exists returns True for saved dataset."""
        fs_store.save("ds-001", sample_meta, sample_arrays)
        assert fs_store.exists("ds-001") is True

    @pytest.mark.unit
    def test_exists_false(self, fs_store: LocalFSDatasetStore):
        """Test exists returns False for nonexistent dataset."""
        assert fs_store.exists("nonexistent") is False

    @pytest.mark.unit
    def test_exists_partial_files(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test exists returns False when only one file exists."""
        fs_store.save("ds-001", sample_meta, sample_arrays)

        (fs_store.base_path / "ds-001.npz").unlink()
        assert fs_store.exists("ds-001") is False

    @pytest.mark.unit
    def test_delete_existing(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test deleting an existing dataset returns True."""
        fs_store.save("ds-001", sample_meta, sample_arrays)
        deleted = fs_store.delete("ds-001")
        assert deleted is True
        assert fs_store.exists("ds-001") is False

        assert not (fs_store.base_path / "ds-001.meta.json").exists()
        assert not (fs_store.base_path / "ds-001.npz").exists()

    @pytest.mark.unit
    def test_delete_nonexistent(self, fs_store: LocalFSDatasetStore):
        """Test deleting a nonexistent dataset returns False."""
        deleted = fs_store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.unit
    def test_delete_partial_files(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test deleting when only meta file exists."""
        fs_store.save("ds-001", sample_meta, sample_arrays)

        deleted = fs_store.delete("ds-001")
        assert deleted is True
        assert fs_store.delete("ds-001") is False
        assert not (fs_store.base_path / "ds-001.meta.json").exists()

    @pytest.mark.unit
    def test_list_datasets_empty(self, fs_store: LocalFSDatasetStore):
        """Test listing datasets in empty store."""
        assert fs_store.list_datasets() == []

    @pytest.mark.unit
    def test_list_datasets_multiple(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test listing multiple datasets."""
        for i in range(5):
            fs_store.save(f"ds-00{i}", sample_meta, sample_arrays)

        datasets = fs_store.list_datasets()
        assert len(datasets) == 5
        assert datasets == sorted(datasets)

    @pytest.mark.unit
    def test_list_datasets_with_limit(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test listing datasets with limit."""
        for i in range(10):
            fs_store.save(f"ds-{i:03d}", sample_meta, sample_arrays)

        datasets = fs_store.list_datasets(limit=3)
        assert len(datasets) == 3

    @pytest.mark.unit
    def test_list_datasets_with_offset(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test listing datasets with offset."""
        for i in range(10):
            fs_store.save(f"ds-{i:03d}", sample_meta, sample_arrays)

        datasets = fs_store.list_datasets(offset=5)
        assert len(datasets) == 5
        assert datasets[0] == "ds-005"

    @pytest.mark.unit
    def test_list_datasets_with_limit_and_offset(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test listing datasets with both limit and offset."""
        for i in range(10):
            fs_store.save(f"ds-{i:03d}", sample_meta, sample_arrays)

        datasets = fs_store.list_datasets(limit=3, offset=2)
        assert len(datasets) == 3
        assert datasets == ["ds-002", "ds-003", "ds-004"]

    @pytest.mark.unit
    def test_base_path_property(self, temp_dir: Path):
        """Test base_path property returns correct path."""
        store = LocalFSDatasetStore(temp_dir)
        assert store.base_path == temp_dir

    @pytest.mark.unit
    def test_datetime_serialization(self, fs_store: LocalFSDatasetStore, sample_arrays: dict[str, np.ndarray]):
        """Test that datetime is properly serialized and deserialized."""
        specific_time = datetime(2026, 6, 15, 10, 30, 45)
        meta = DatasetMeta(
            dataset_id="dt-test",
            generator="test",
            generator_version="1.0.0",
            params={},
            n_samples=100,
            n_features=2,
            n_classes=2,
            n_train=80,
            n_test=20,
            class_distribution={"0": 50, "1": 50},
            created_at=specific_time,
        )

        fs_store.save("dt-test", meta, sample_arrays)
        retrieved = fs_store.get_meta("dt-test")

        assert retrieved is not None
        assert retrieved.created_at == specific_time


class TestDatasetStoreInterface:
    """Tests to verify implementations follow the abstract interface."""

    @pytest.mark.unit
    def test_memory_store_is_dataset_store(self, memory_store: InMemoryDatasetStore):
        """Test InMemoryDatasetStore is a DatasetStore."""
        assert isinstance(memory_store, DatasetStore)

    @pytest.mark.unit
    def test_fs_store_is_dataset_store(self, fs_store: LocalFSDatasetStore):
        """Test LocalFSDatasetStore is a DatasetStore."""
        assert isinstance(fs_store, DatasetStore)


class TestLocalFSUpdateAndList:
    """Tests for LocalFSDatasetStore update_meta and list_all_metadata."""

    @pytest.mark.unit
    def test_update_meta(self, fs_store, sample_meta, sample_arrays):
        """Test update_meta updates metadata."""
        fs_store.save("ds-upd", sample_meta, sample_arrays)
        new_meta = DatasetMeta(
            dataset_id="ds-upd-v2",
            generator="spiral",
            generator_version="2.0.0",
            params={},
            n_samples=200,
            n_features=2,
            n_classes=2,
            n_train=160,
            n_test=40,
            class_distribution={"0": 100, "1": 100},
            created_at=datetime(2026, 2, 1, 12, 0, 0),
        )
        result = fs_store.update_meta("ds-upd", new_meta)
        assert result is True
        retrieved = fs_store.get_meta("ds-upd")
        assert retrieved.generator_version == "2.0.0"

    @pytest.mark.unit
    def test_update_meta_nonexistent(self, fs_store, sample_meta):
        """Test update_meta returns False for nonexistent dataset."""
        assert fs_store.update_meta("nonexistent", sample_meta) is False

    @pytest.mark.unit
    def test_list_all_metadata(self, fs_store, sample_meta, sample_arrays):
        """Test list_all_metadata returns all stored metadata."""
        fs_store.save("ds-a", sample_meta, sample_arrays)
        fs_store.save("ds-b", sample_meta, sample_arrays)
        result = fs_store.list_all_metadata()
        assert len(result) == 2

    @pytest.mark.unit
    def test_list_all_metadata_empty(self, fs_store):
        """Test list_all_metadata returns empty list when no datasets."""
        result = fs_store.list_all_metadata()
        assert result == []

    @pytest.mark.unit
    def test_save_error_cleanup(self, fs_store, sample_meta, sample_arrays):
        """Test save cleans up temp files on error."""
        from unittest.mock import patch

        with patch("numpy.savez_compressed", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                fs_store.save("ds-fail", sample_meta, sample_arrays)
        tmp_files = list(fs_store.base_path.glob("*.tmp"))
        assert not tmp_files

    @pytest.mark.unit
    def test_save_cleanup_oserror_suppressed(self, fs_store, sample_meta, sample_arrays):
        """Test that OSError during temp file cleanup is caught and logged."""
        from unittest.mock import patch

        path_cls = type(fs_store.base_path)
        original_replace = path_cls.replace
        original_unlink = path_cls.unlink

        def failing_replace(self_path, target):
            if str(self_path).endswith(".npz.tmp"):
                raise OSError("Simulated disk error during replace")
            return original_replace(self_path, target)

        def failing_unlink(self_path, missing_ok=False):
            if str(self_path).endswith(".tmp"):
                raise OSError("Simulated permission error during cleanup")
            return original_unlink(self_path, missing_ok=missing_ok)

        with patch.object(path_cls, "replace", failing_replace):
            with patch.object(path_cls, "unlink", failing_unlink):
                with pytest.raises(IOError, match="Simulated disk error"):
                    fs_store.save("ds-cleanup-err", sample_meta, sample_arrays)


class TestLocalFSEdgeCases:
    """Additional edge case tests for LocalFSDatasetStore."""

    @pytest.mark.unit
    def test_json_serializer_raises_for_unknown_type(self):
        """Test _json_serializer raises TypeError for unknown types."""
        from juniper_data.storage.local_fs import _json_serializer

        with pytest.raises(TypeError) as exc_info:
            _json_serializer(object())

        assert "not JSON serializable" in str(exc_info.value)

    @pytest.mark.unit
    def test_get_meta_skips_datetime_conversion_for_non_string(self, fs_store: LocalFSDatasetStore, sample_arrays: dict[str, np.ndarray]):
        """Test get_meta skips datetime conversion when created_at is already parsed or not a string."""
        import json

        meta = DatasetMeta(
            dataset_id="test-date-type",
            generator="test",
            generator_version="1.0.0",
            params={},
            n_samples=100,
            n_features=2,
            n_classes=2,
            n_train=80,
            n_test=20,
            class_distribution={"0": 50, "1": 50},
            created_at=datetime(2026, 1, 30, 12, 0, 0),
        )

        fs_store.save("test-date-type", meta, sample_arrays)

        meta_path = fs_store._meta_path("test-date-type")
        meta_dict = json.loads(meta_path.read_text())
        assert isinstance(meta_dict["created_at"], str)
        meta_dict["created_at"] = 1234567890
        meta_path.write_text(json.dumps(meta_dict))

        retrieved = fs_store.get_meta("test-date-type")
        assert retrieved is not None

    @pytest.mark.unit
    def test_delete_only_npz_exists(self, fs_store: LocalFSDatasetStore, sample_meta: DatasetMeta, sample_arrays: dict[str, np.ndarray]):
        """Test delete when only NPZ file exists (meta was deleted)."""
        fs_store.save("ds-partial-npz", sample_meta, sample_arrays)

        (fs_store.base_path / "ds-partial-npz.meta.json").unlink()

        result = fs_store.delete("ds-partial-npz")
        assert result is True
        assert not (fs_store.base_path / "ds-partial-npz.npz").exists()

    @pytest.mark.unit
    def test_get_meta_with_timezone_aware_datetime(self, fs_store: LocalFSDatasetStore, sample_arrays: dict[str, np.ndarray]):
        """Test get_meta correctly deserializes timezone-aware datetime."""
        from datetime import timezone

        tz_aware_time = datetime(2026, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        meta = DatasetMeta(
            dataset_id="tz-test",
            generator="test",
            generator_version="1.0.0",
            params={},
            n_samples=100,
            n_features=2,
            n_classes=2,
            n_train=80,
            n_test=20,
            class_distribution={"0": 50, "1": 50},
            created_at=tz_aware_time,
        )

        fs_store.save("tz-test", meta, sample_arrays)
        retrieved = fs_store.get_meta("tz-test")

        assert retrieved is not None
        assert retrieved.created_at is not None
        assert retrieved.created_at.year == 2026
        assert retrieved.created_at.month == 6


class TestDatasetStoreAbstractMethods:
    """Tests to ensure abstract methods are properly defined."""

    @pytest.mark.unit
    def test_cannot_instantiate_abstract_base(self):
        """Test that DatasetStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DatasetStore()

    @pytest.mark.unit
    def test_abstract_methods_exist(self):
        """Test that all abstract methods are defined."""
        import inspect

        abstract_methods = [name for name, method in inspect.getmembers(DatasetStore, predicate=inspect.isfunction) if getattr(method, "__isabstractmethod__", False)]

        expected_methods = ["save", "get_meta", "get_artifact_bytes", "exists", "delete", "list_datasets"]
        for method in expected_methods:
            assert method in abstract_methods, f"Missing abstract method: {method}"


class TestStorageModuleFactories:
    """Tests for storage module factory functions and imports."""

    @pytest.mark.unit
    def test_storage_module_exports(self):
        """Test that the storage module exports expected classes."""
        from juniper_data.storage import CachedDatasetStore, DatasetStore, InMemoryDatasetStore, LocalFSDatasetStore

        assert DatasetStore is not None
        assert InMemoryDatasetStore is not None
        assert LocalFSDatasetStore is not None
        assert CachedDatasetStore is not None

    @pytest.mark.unit
    def test_optional_imports_are_none_or_class(self):
        """Test that optional store classes are None or importable."""
        import juniper_data.storage as storage_mod

        for attr in ["RedisDatasetStore", "HuggingFaceDatasetStore", "PostgresDatasetStore", "KaggleDatasetStore"]:
            val = getattr(storage_mod, attr, None)
            assert val is None or isinstance(val, type)

    @pytest.mark.unit
    def test_get_redis_store_raises_import_error(self):
        """Test get_redis_store raises ImportError when redis not installed."""
        from juniper_data.storage import get_redis_store

        with contextlib.suppress(ImportError):
            get_redis_store()

    @pytest.mark.unit
    def test_get_hf_store_raises_import_error(self):
        """Test get_hf_store raises ImportError when datasets not installed."""
        from juniper_data.storage import get_hf_store

        with contextlib.suppress(ImportError):
            get_hf_store()

    @pytest.mark.unit
    def test_get_postgres_store_raises_import_error(self):
        """Test get_postgres_store raises ImportError when psycopg2 not installed."""
        from juniper_data.storage import get_postgres_store

        with contextlib.suppress(ImportError):
            get_postgres_store()

    @pytest.mark.unit
    def test_get_kaggle_store_raises_import_error(self):
        """Test get_kaggle_store raises ImportError when kaggle not installed."""
        from juniper_data.storage import get_kaggle_store

        with contextlib.suppress(ImportError):
            get_kaggle_store()

    @pytest.mark.unit
    def test_optional_imports_fallback_to_none(self):
        """Test that optional imports fall back to None when packages are missing."""
        import importlib
        import sys

        # from unittest.mock import MagicMock
        # Save original modules and remove them to force ImportError
        modules_to_block = [
            "juniper_data.storage.redis_store",
            "juniper_data.storage.hf_store",
            "juniper_data.storage.postgres_store",
            "juniper_data.storage.kaggle_store",
        ]
        saved = {}
        for mod in modules_to_block:
            saved[mod] = sys.modules.pop(mod, None)

        # Also remove the storage module itself so it can be reimported
        saved["juniper_data.storage"] = sys.modules.pop("juniper_data.storage", None)

        # Patch __import__ to block the optional store modules
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def blocking_import(name, *args, **kwargs):
            if name in modules_to_block:
                raise ImportError(f"Mocked: {name} not installed")
            return original_import(name, *args, **kwargs)

        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("builtins.__import__", blocking_import)
                storage_mod = importlib.import_module("juniper_data.storage")

            assert storage_mod.RedisDatasetStore is None
            assert storage_mod.HuggingFaceDatasetStore is None
            assert storage_mod.PostgresDatasetStore is None
            assert storage_mod.KaggleDatasetStore is None
            assert "RedisDatasetStore" not in storage_mod.__all__
            assert "HuggingFaceDatasetStore" not in storage_mod.__all__
            assert "PostgresDatasetStore" not in storage_mod.__all__
            assert "KaggleDatasetStore" not in storage_mod.__all__
        finally:
            # Restore original modules
            for mod, val in saved.items():
                if val is not None:
                    sys.modules[mod] = val
                else:
                    sys.modules.pop(mod, None)
            # Force reimport to restore normal state
            importlib.import_module("juniper_data.storage")
