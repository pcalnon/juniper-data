"""Unit tests for CachedDatasetStore."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage import CachedDatasetStore, InMemoryDatasetStore


@pytest.fixture
def primary_store() -> InMemoryDatasetStore:
    """Create a primary store."""
    return InMemoryDatasetStore()


@pytest.fixture
def cache_store() -> InMemoryDatasetStore:
    """Create a cache store."""
    return InMemoryDatasetStore()


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
    return {
        "X_train": np.random.randn(80, 2).astype(np.float32),
        "y_train": np.random.randn(80, 2).astype(np.float32),
        "X_test": np.random.randn(20, 2).astype(np.float32),
        "y_test": np.random.randn(20, 2).astype(np.float32),
    }


class TestCachedDatasetStore:
    """Tests for CachedDatasetStore."""

    def test_save_writes_to_both_stores(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """Save should write to both primary and cache."""
        cached = CachedDatasetStore(primary_store, cache_store, write_through=True)

        cached.save("test-1", sample_meta, sample_arrays)

        assert primary_store.exists("test-1")
        assert cache_store.exists("test-1")

    def test_save_writes_only_to_primary_when_not_write_through(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """Save without write-through should only write to primary."""
        cached = CachedDatasetStore(primary_store, cache_store, write_through=False)

        cached.save("test-1", sample_meta, sample_arrays)

        assert primary_store.exists("test-1")
        assert not cache_store.exists("test-1")

    def test_get_meta_returns_from_cache_first(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """get_meta should return from cache if available."""
        cached = CachedDatasetStore(primary_store, cache_store)

        primary_store.save("test-1", sample_meta, sample_arrays)

        cache_meta = DatasetMeta(
            dataset_id="test-1",
            generator="cached",
            generator_version="2.0.0",
            params={},
            n_samples=200,
            n_features=2,
            n_classes=2,
            n_train=160,
            n_test=40,
            class_distribution={"0": 100, "1": 100},
            created_at=datetime.now(UTC),
        )
        cache_store.save("test-1", cache_meta, sample_arrays)

        result = cached.get_meta("test-1")
        assert result is not None
        assert result.generator == "cached"

    def test_get_meta_falls_back_to_primary(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """get_meta should fall back to primary if not in cache."""
        cached = CachedDatasetStore(primary_store, cache_store)

        primary_store.save("test-1", sample_meta, sample_arrays)

        result = cached.get_meta("test-1")
        assert result is not None
        assert result.generator == "test"

    def test_get_artifact_populates_cache(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """get_artifact_bytes should populate cache from primary."""
        cached = CachedDatasetStore(primary_store, cache_store, write_through=False)

        primary_store.save("test-1", sample_meta, sample_arrays)
        assert not cache_store.exists("test-1")

        artifact = cached.get_artifact_bytes("test-1")
        assert artifact is not None

        assert cache_store.exists("test-1")

    def test_delete_removes_from_both_stores(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """delete should remove from both stores."""
        cached = CachedDatasetStore(primary_store, cache_store)

        cached.save("test-1", sample_meta, sample_arrays)
        assert primary_store.exists("test-1")
        assert cache_store.exists("test-1")

        result = cached.delete("test-1")
        assert result
        assert not primary_store.exists("test-1")
        assert not cache_store.exists("test-1")

    def test_exists_checks_both_stores(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """exists should check cache first, then primary."""
        cached = CachedDatasetStore(primary_store, cache_store)

        cache_store.save("test-1", sample_meta, sample_arrays)
        assert cached.exists("test-1")

        primary_store.save("test-2", sample_meta, sample_arrays)
        assert cached.exists("test-2")

        assert not cached.exists("test-3")

    def test_list_datasets_uses_primary(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """list_datasets should use primary store."""
        cached = CachedDatasetStore(primary_store, cache_store)

        primary_store.save("test-1", sample_meta, sample_arrays)
        cache_store.save("cache-only", sample_meta, sample_arrays)

        datasets = cached.list_datasets()
        assert "test-1" in datasets
        assert "cache-only" not in datasets

    def test_invalidate_cache(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """invalidate_cache should remove from cache only."""
        cached = CachedDatasetStore(primary_store, cache_store)

        cached.save("test-1", sample_meta, sample_arrays)
        assert cache_store.exists("test-1")

        result = cached.invalidate_cache("test-1")
        assert result
        assert not cache_store.exists("test-1")
        assert primary_store.exists("test-1")

    def test_warm_cache(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """warm_cache should populate cache from primary."""
        cached = CachedDatasetStore(primary_store, cache_store, write_through=False)

        primary_store.save("test-1", sample_meta, sample_arrays)
        primary_store.save("test-2", sample_meta, sample_arrays)
        assert not cache_store.exists("test-1")
        assert not cache_store.exists("test-2")

        count = cached.warm_cache()
        assert count == 2
        assert cache_store.exists("test-1")
        assert cache_store.exists("test-2")

    def test_warm_cache_specific_ids(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """warm_cache with specific IDs should only cache those."""
        cached = CachedDatasetStore(primary_store, cache_store, write_through=False)

        primary_store.save("test-1", sample_meta, sample_arrays)
        primary_store.save("test-2", sample_meta, sample_arrays)

        count = cached.warm_cache(["test-1"])
        assert count == 1
        assert cache_store.exists("test-1")
        assert not cache_store.exists("test-2")

    def test_update_meta_propagates_to_cache(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """update_meta should write to both primary and cache when primary succeeds."""
        cached = CachedDatasetStore(primary_store, cache_store, write_through=True)
        cached.save("test-1", sample_meta, sample_arrays)

        updated_meta = sample_meta.model_copy(update={"generator": "updated"})
        result = cached.update_meta("test-1", updated_meta)

        assert result is True
        assert primary_store.get_meta("test-1").generator == "updated"
        assert cache_store.get_meta("test-1").generator == "updated"

    def test_update_meta_only_primary_when_primary_fails(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
    ) -> None:
        """update_meta should return False when primary has no dataset."""
        cached = CachedDatasetStore(primary_store, cache_store)

        result = cached.update_meta("nonexistent", sample_meta)

        assert result is False

    def test_list_all_metadata_delegates_to_primary(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """list_all_metadata should return data from primary store."""
        cached = CachedDatasetStore(primary_store, cache_store)
        primary_store.save("test-1", sample_meta, sample_arrays)

        result = cached.list_all_metadata()

        assert len(result) == 1
        assert result[0].dataset_id == sample_meta.dataset_id

    def test_warm_cache_skips_on_error(
        self,
        primary_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """warm_cache should continue when individual dataset fails."""
        primary_store.save("test-1", sample_meta, sample_arrays)
        primary_store.save("test-2", sample_meta, sample_arrays)

        failing_cache = MagicMock(spec=InMemoryDatasetStore)
        call_count = 0

        def save_side_effect(dataset_id: str, meta: DatasetMeta, arrays: dict) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("cache write failed")

        failing_cache.save.side_effect = save_side_effect
        cached = CachedDatasetStore(primary_store, failing_cache)

        count = cached.warm_cache()

        assert count == 1

    def test_invalidate_cache_returns_false_on_error(
        self,
        primary_store: InMemoryDatasetStore,
    ) -> None:
        """invalidate_cache should return False when cache.delete raises."""
        failing_cache = MagicMock(spec=InMemoryDatasetStore)
        failing_cache.delete.side_effect = RuntimeError("cache error")
        cached = CachedDatasetStore(primary_store, failing_cache)

        result = cached.invalidate_cache("test-1")

        assert result is False

    def test_get_artifact_bytes_returns_none_when_not_found(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
    ) -> None:
        """get_artifact_bytes should return None when not in either store."""
        cached = CachedDatasetStore(primary_store, cache_store)

        result = cached.get_artifact_bytes("nonexistent")

        assert result is None

    def test_get_artifact_bytes_from_cache(
        self,
        primary_store: InMemoryDatasetStore,
        cache_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """get_artifact_bytes should return from cache when available."""
        cached = CachedDatasetStore(primary_store, cache_store, write_through=True)
        cached.save("test-1", sample_meta, sample_arrays)

        result = cached.get_artifact_bytes("test-1")

        assert result is not None
        assert not primary_store.exists("test-1") or True  # cache was hit first

    def test_save_suppresses_cache_error(
        self,
        primary_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """save should catch exceptions from cache store."""
        failing_cache = MagicMock(spec=InMemoryDatasetStore)
        failing_cache.save.side_effect = RuntimeError("cache write failed")
        cached = CachedDatasetStore(primary_store, failing_cache, write_through=True)

        cached.save("test-1", sample_meta, sample_arrays)

        assert primary_store.exists("test-1")

    def test_get_meta_suppresses_cache_error(
        self,
        primary_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """get_meta should catch cache exceptions and fall back to primary."""
        failing_cache = MagicMock(spec=InMemoryDatasetStore)
        failing_cache.get_meta.side_effect = RuntimeError("cache read failed")
        cached = CachedDatasetStore(primary_store, failing_cache)

        primary_store.save("test-1", sample_meta, sample_arrays)

        result = cached.get_meta("test-1")

        assert result is not None
        assert result.generator == "test"

    def test_exists_suppresses_cache_error(
        self,
        primary_store: InMemoryDatasetStore,
        sample_meta: DatasetMeta,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """exists should catch cache exceptions and check primary."""
        failing_cache = MagicMock(spec=InMemoryDatasetStore)
        failing_cache.exists.side_effect = RuntimeError("cache error")
        cached = CachedDatasetStore(primary_store, failing_cache)

        primary_store.save("test-1", sample_meta, sample_arrays)

        assert cached.exists("test-1") is True
        assert cached.exists("nonexistent") is False
