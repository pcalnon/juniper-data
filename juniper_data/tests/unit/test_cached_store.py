"""Unit tests for CachedDatasetStore."""

from datetime import datetime, timezone

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
        created_at=datetime.now(timezone.utc),
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
            created_at=datetime.now(timezone.utc),
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
