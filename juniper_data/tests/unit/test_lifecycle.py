"""Unit tests for dataset lifecycle management features (DATA-016).

Tests for:
- Dataset expiration / TTL
- Bulk operations (filtering, batch delete)
- Dataset tagging
- Usage tracking / access counts
- Statistics
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.memory import InMemoryDatasetStore

# from typing import Dict


def _create_test_meta(
    dataset_id: str,
    generator: str = "spiral",
    n_samples: int = 100,
    tags: list[str] | None = None,
    ttl_seconds: int | None = None,
    created_at: datetime | None = None,
) -> DatasetMeta:
    """Create a test DatasetMeta instance."""
    now = created_at or datetime.now(UTC)
    expires_at = None
    if ttl_seconds is not None:
        expires_at = now + timedelta(seconds=ttl_seconds)

    return DatasetMeta(
        dataset_id=dataset_id,
        generator=generator,
        generator_version="1.0.0",
        params={"n_spirals": 2},
        n_samples=n_samples,
        n_features=2,
        n_classes=2,
        n_train=int(n_samples * 0.8),
        n_test=int(n_samples * 0.2),
        class_distribution={"0": n_samples // 2, "1": n_samples // 2},
        artifact_formats=["npz"],
        created_at=now,
        tags=tags or [],
        ttl_seconds=ttl_seconds,
        expires_at=expires_at,
    )


def _create_test_arrays(n_samples: int = 100) -> dict[str, np.ndarray]:
    """Create minimal test arrays."""
    n_train = int(n_samples * 0.8)
    n_test = n_samples - n_train
    return {
        "X_train": np.zeros((n_train, 2), dtype=np.float32),
        "y_train": np.zeros((n_train, 2), dtype=np.float32),
        "X_test": np.zeros((n_test, 2), dtype=np.float32),
        "y_test": np.zeros((n_test, 2), dtype=np.float32),
        "X_full": np.zeros((n_samples, 2), dtype=np.float32),
        "y_full": np.zeros((n_samples, 2), dtype=np.float32),
    }


@pytest.fixture
def store() -> InMemoryDatasetStore:
    """Create a fresh in-memory store."""
    return InMemoryDatasetStore()


@pytest.mark.unit
class TestDatasetTags:
    """Tests for dataset tagging functionality."""

    def test_create_dataset_with_tags(self, store: InMemoryDatasetStore) -> None:
        """Dataset can be created with tags."""
        meta = _create_test_meta("ds-1", tags=["train", "spiral", "v1"])
        store.save("ds-1", meta, _create_test_arrays())

        retrieved = store.get_meta("ds-1")
        assert retrieved is not None
        assert retrieved.tags == ["train", "spiral", "v1"]

    def test_update_meta_adds_tags(self, store: InMemoryDatasetStore) -> None:
        """Tags can be added via update_meta."""
        meta = _create_test_meta("ds-1", tags=["original"])
        store.save("ds-1", meta, _create_test_arrays())

        meta.tags = ["original", "added"]
        result = store.update_meta("ds-1", meta)
        assert result is True

        retrieved = store.get_meta("ds-1")
        assert retrieved is not None
        assert "added" in retrieved.tags

    def test_update_meta_nonexistent_returns_false(self, store: InMemoryDatasetStore) -> None:
        """update_meta returns False for nonexistent dataset."""
        meta = _create_test_meta("nonexistent")
        result = store.update_meta("nonexistent", meta)
        assert result is False


@pytest.mark.unit
class TestDatasetTTL:
    """Tests for dataset expiration / TTL."""

    def test_dataset_with_ttl_has_expires_at(self, store: InMemoryDatasetStore) -> None:
        """Dataset with TTL has expires_at set."""
        meta = _create_test_meta("ds-1", ttl_seconds=3600)
        store.save("ds-1", meta, _create_test_arrays())

        retrieved = store.get_meta("ds-1")
        assert retrieved is not None
        assert retrieved.ttl_seconds == 3600
        assert retrieved.expires_at is not None

    def test_is_expired_false_for_future_expiry(self, store: InMemoryDatasetStore) -> None:
        """Dataset with future expiry is not expired."""
        meta = _create_test_meta("ds-1", ttl_seconds=3600)
        store.save("ds-1", meta, _create_test_arrays())

        assert store.is_expired(meta) is False

    def test_is_expired_true_for_past_expiry(self, store: InMemoryDatasetStore) -> None:
        """Dataset with past expiry is expired."""
        past_time = datetime.now(UTC) - timedelta(hours=2)
        meta = _create_test_meta("ds-1", ttl_seconds=3600, created_at=past_time)
        store.save("ds-1", meta, _create_test_arrays())

        assert store.is_expired(meta) is True

    def test_is_expired_false_for_no_ttl(self, store: InMemoryDatasetStore) -> None:
        """Dataset without TTL never expires."""
        meta = _create_test_meta("ds-1", ttl_seconds=None)
        store.save("ds-1", meta, _create_test_arrays())

        assert store.is_expired(meta) is False

    def test_delete_expired_removes_expired_datasets(self, store: InMemoryDatasetStore) -> None:
        """delete_expired removes only expired datasets."""
        past_time = datetime.now(UTC) - timedelta(hours=2)
        meta1 = _create_test_meta("expired-1", ttl_seconds=3600, created_at=past_time)
        meta2 = _create_test_meta("expired-2", ttl_seconds=3600, created_at=past_time)
        meta3 = _create_test_meta("valid-1", ttl_seconds=3600)
        meta4 = _create_test_meta("no-ttl")

        store.save("expired-1", meta1, _create_test_arrays())
        store.save("expired-2", meta2, _create_test_arrays())
        store.save("valid-1", meta3, _create_test_arrays())
        store.save("no-ttl", meta4, _create_test_arrays())

        deleted = store.delete_expired()

        assert set(deleted) == {"expired-1", "expired-2"}
        assert store.exists("valid-1")
        assert store.exists("no-ttl")
        assert not store.exists("expired-1")
        assert not store.exists("expired-2")


@pytest.mark.unit
class TestDatasetFiltering:
    """Tests for dataset filtering functionality."""

    @pytest.fixture
    def populated_store(self, store: InMemoryDatasetStore) -> InMemoryDatasetStore:
        """Create a store with multiple datasets for filtering tests."""
        now = datetime.now(UTC)

        datasets = [
            ("ds-1", "spiral", 100, ["train", "v1"], now - timedelta(days=5)),
            ("ds-2", "spiral", 200, ["train", "v2"], now - timedelta(days=3)),
            ("ds-3", "spiral", 50, ["test", "v1"], now - timedelta(days=1)),
            ("ds-4", "xor", 100, ["train"], now - timedelta(hours=12)),
            ("ds-5", "xor", 300, ["train", "v2"], now - timedelta(hours=1)),
        ]

        for dataset_id, gen, n_samples, tags, created in datasets:
            meta = _create_test_meta(dataset_id, generator=gen, n_samples=n_samples, tags=tags, created_at=created)
            store.save(dataset_id, meta, _create_test_arrays(n_samples))

        return store

    def test_filter_by_generator(self, populated_store: InMemoryDatasetStore) -> None:
        """Filter datasets by generator name."""
        datasets, total = populated_store.filter_datasets(generator="spiral")
        assert total == 3
        assert all(d.generator == "spiral" for d in datasets)

    def test_filter_by_tags_any(self, populated_store: InMemoryDatasetStore) -> None:
        """Filter datasets by tags (any match)."""
        datasets, total = populated_store.filter_datasets(tags=["v1", "v2"], tags_match="any")
        assert total == 4

    def test_filter_by_tags_all(self, populated_store: InMemoryDatasetStore) -> None:
        """Filter datasets by tags (all must match)."""
        datasets, total = populated_store.filter_datasets(tags=["train", "v2"], tags_match="all")
        assert total == 2
        for d in datasets:
            assert "train" in d.tags
            assert "v2" in d.tags

    def test_filter_by_created_after(self, populated_store: InMemoryDatasetStore) -> None:
        """Filter datasets created after a date."""
        cutoff = datetime.now(UTC) - timedelta(days=2)
        datasets, total = populated_store.filter_datasets(created_after=cutoff)
        assert total == 3

    def test_filter_by_created_before(self, populated_store: InMemoryDatasetStore) -> None:
        """Filter datasets created before a date."""
        cutoff = datetime.now(UTC) - timedelta(days=2)
        datasets, total = populated_store.filter_datasets(created_before=cutoff)
        assert total == 2

    def test_filter_by_sample_count(self, populated_store: InMemoryDatasetStore) -> None:
        """Filter datasets by sample count range."""
        datasets, total = populated_store.filter_datasets(min_samples=100, max_samples=200)
        assert total == 3
        for d in datasets:
            assert 100 <= d.n_samples <= 200

    def test_filter_pagination(self, populated_store: InMemoryDatasetStore) -> None:
        """Filter with pagination."""
        datasets_page1, total = populated_store.filter_datasets(limit=2, offset=0)
        datasets_page2, _ = populated_store.filter_datasets(limit=2, offset=2)

        assert total == 5
        assert len(datasets_page1) == 2
        assert len(datasets_page2) == 2

        ids_page1 = {d.dataset_id for d in datasets_page1}
        ids_page2 = {d.dataset_id for d in datasets_page2}
        assert ids_page1.isdisjoint(ids_page2)

    def test_filter_excludes_expired_by_default(self, store: InMemoryDatasetStore) -> None:
        """Expired datasets are excluded by default."""
        past_time = datetime.now(UTC) - timedelta(hours=2)
        meta_expired = _create_test_meta("expired", ttl_seconds=3600, created_at=past_time)
        meta_valid = _create_test_meta("valid")

        store.save("expired", meta_expired, _create_test_arrays())
        store.save("valid", meta_valid, _create_test_arrays())

        datasets, total = store.filter_datasets(include_expired=False)
        assert total == 1
        assert datasets[0].dataset_id == "valid"

    def test_filter_includes_expired_when_requested(self, store: InMemoryDatasetStore) -> None:
        """Expired datasets are included when requested."""
        past_time = datetime.now(UTC) - timedelta(hours=2)
        meta_expired = _create_test_meta("expired", ttl_seconds=3600, created_at=past_time)
        meta_valid = _create_test_meta("valid")

        store.save("expired", meta_expired, _create_test_arrays())
        store.save("valid", meta_valid, _create_test_arrays())

        datasets, total = store.filter_datasets(include_expired=True)
        assert total == 2


@pytest.mark.unit
class TestBatchDelete:
    """Tests for batch delete functionality."""

    def test_batch_delete_existing(self, store: InMemoryDatasetStore) -> None:
        """Batch delete existing datasets."""
        for i in range(5):
            meta = _create_test_meta(f"ds-{i}")
            store.save(f"ds-{i}", meta, _create_test_arrays())

        deleted, not_found = store.batch_delete(["ds-0", "ds-2", "ds-4"])

        assert set(deleted) == {"ds-0", "ds-2", "ds-4"}
        assert not_found == []
        assert store.exists("ds-1")
        assert store.exists("ds-3")
        assert not store.exists("ds-0")

    def test_batch_delete_mixed(self, store: InMemoryDatasetStore) -> None:
        """Batch delete with some nonexistent IDs."""
        meta = _create_test_meta("ds-1")
        store.save("ds-1", meta, _create_test_arrays())

        deleted, not_found = store.batch_delete(["ds-1", "nonexistent-1", "nonexistent-2"])

        assert deleted == ["ds-1"]
        assert set(not_found) == {"nonexistent-1", "nonexistent-2"}

    def test_batch_delete_all_nonexistent(self, store: InMemoryDatasetStore) -> None:
        """Batch delete with all nonexistent IDs."""
        deleted, not_found = store.batch_delete(["fake-1", "fake-2"])

        assert deleted == []
        assert set(not_found) == {"fake-1", "fake-2"}


@pytest.mark.unit
class TestAccessTracking:
    """Tests for access tracking functionality."""

    def test_record_access_updates_timestamp(self, store: InMemoryDatasetStore) -> None:
        """record_access updates last_accessed_at."""
        meta = _create_test_meta("ds-1")
        store.save("ds-1", meta, _create_test_arrays())

        store.record_access("ds-1")

        retrieved = store.get_meta("ds-1")
        assert retrieved is not None
        assert retrieved.last_accessed_at is not None
        assert retrieved.access_count == 1

    def test_record_access_increments_count(self, store: InMemoryDatasetStore) -> None:
        """record_access increments access_count."""
        meta = _create_test_meta("ds-1")
        store.save("ds-1", meta, _create_test_arrays())

        store.record_access("ds-1")
        store.record_access("ds-1")
        store.record_access("ds-1")

        retrieved = store.get_meta("ds-1")
        assert retrieved is not None
        assert retrieved.access_count == 3


@pytest.mark.unit
class TestDatasetStats:
    """Tests for aggregate statistics functionality."""

    def test_stats_empty_store(self, store: InMemoryDatasetStore) -> None:
        """Stats for empty store."""
        stats = store.get_stats()

        assert stats["total_datasets"] == 0
        assert stats["total_samples"] == 0
        assert stats["by_generator"] == {}
        assert stats["by_tag"] == {}

    def test_stats_populated_store(self, store: InMemoryDatasetStore) -> None:
        """Stats for populated store."""
        meta1 = _create_test_meta("ds-1", generator="spiral", n_samples=100, tags=["train", "v1"])
        meta2 = _create_test_meta("ds-2", generator="spiral", n_samples=200, tags=["train", "v2"])
        meta3 = _create_test_meta("ds-3", generator="xor", n_samples=50, tags=["test"])

        store.save("ds-1", meta1, _create_test_arrays(100))
        store.save("ds-2", meta2, _create_test_arrays(200))
        store.save("ds-3", meta3, _create_test_arrays(50))

        stats = store.get_stats()

        assert stats["total_datasets"] == 3
        assert stats["total_samples"] == 350
        assert stats["by_generator"] == {"spiral": 2, "xor": 1}
        assert stats["by_tag"] == {"train": 2, "v1": 1, "v2": 1, "test": 1}

    def test_stats_counts_expired(self, store: InMemoryDatasetStore) -> None:
        """Stats includes expired count."""
        past_time = datetime.now(UTC) - timedelta(hours=2)
        meta_expired = _create_test_meta("expired", ttl_seconds=3600, created_at=past_time)
        meta_valid = _create_test_meta("valid")

        store.save("expired", meta_expired, _create_test_arrays())
        store.save("valid", meta_valid, _create_test_arrays())

        stats = store.get_stats()

        assert stats["expired_count"] == 1


@pytest.mark.unit
class TestListAllMetadata:
    """Tests for list_all_metadata functionality."""

    def test_list_all_metadata_empty(self, store: InMemoryDatasetStore) -> None:
        """list_all_metadata returns empty list for empty store."""
        result = store.list_all_metadata()
        assert result == []

    def test_list_all_metadata_returns_all(self, store: InMemoryDatasetStore) -> None:
        """list_all_metadata returns all stored metadata."""
        for i in range(5):
            meta = _create_test_meta(f"ds-{i}")
            store.save(f"ds-{i}", meta, _create_test_arrays())

        result = store.list_all_metadata()
        assert len(result) == 5
        ids = {m.dataset_id for m in result}
        assert ids == {"ds-0", "ds-1", "ds-2", "ds-3", "ds-4"}
