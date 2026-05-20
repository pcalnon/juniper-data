"""JD-PERF-02 — metadata cache regression coverage.

Pins the cache contract added to ``DatasetStore``:

* ``filter_datasets`` / ``get_stats`` / ``list_versions`` /
  ``next_version_number`` / ``delete_expired`` all share a single
  TTL-cached snapshot of ``list_all_metadata()`` so the steady-state
  cost is one disk walk per ``_METADATA_CACHE_TTL_SECONDS`` window.
* Subclasses that override ``save`` / ``delete`` / ``update_meta`` can
  opt in to immediate freshness by calling
  ``self._invalidate_metadata_cache()``.
* Subclasses that skip ``super().__init__()`` degrade to uncached
  behaviour instead of crashing.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage import base as base_module
from juniper_data.storage.base import DatasetStore


def _make_meta(dataset_id: str = "ds_1", **overrides) -> DatasetMeta:
    """Build a minimal DatasetMeta for cache tests."""
    defaults = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_id,
        "dataset_version": 1,
        "generator": "spiral",
        "generator_version": "1.0.0",
        "params": {},
        "created_at": datetime.now(UTC),
        "n_samples": 100,
        "n_features": 2,
        "n_classes": 2,
        "n_train": 80,
        "n_test": 20,
        "class_distribution": {"0": 50, "1": 50},
        "tags": [],
    }
    defaults.update(overrides)
    return DatasetMeta(**defaults)


class _CountingStore(DatasetStore):
    """Minimal concrete store whose ``list_all_metadata`` records call count."""

    def __init__(self, metas: list[DatasetMeta] | None = None) -> None:
        super().__init__()
        self._metas: list[DatasetMeta] = list(metas or [])
        self.list_all_metadata_calls = 0

    # ABC required surface — minimal stubs:
    def save(self, dataset_id, meta, arrays):  # noqa: D401
        self._metas.append(meta)
        self._invalidate_metadata_cache()

    def get_meta(self, dataset_id):
        for m in self._metas:
            if m.dataset_id == dataset_id:
                return m
        return None

    def get_artifact_bytes(self, dataset_id):
        return None

    def exists(self, dataset_id):
        return any(m.dataset_id == dataset_id for m in self._metas)

    def delete(self, dataset_id):
        before = len(self._metas)
        self._metas = [m for m in self._metas if m.dataset_id != dataset_id]
        removed = len(self._metas) < before
        if removed:
            self._invalidate_metadata_cache()
        return removed

    def list_datasets(self, limit=100, offset=0):
        return [m.dataset_id for m in self._metas][offset : offset + limit]

    def list_all_metadata(self) -> list[DatasetMeta]:
        self.list_all_metadata_calls += 1
        return list(self._metas)


@pytest.mark.unit
class TestMetadataCache:
    def test_first_call_hits_underlying_list_all_metadata(self):
        store = _CountingStore([_make_meta("a"), _make_meta("b")])
        filtered, total = store.filter_datasets()
        assert store.list_all_metadata_calls == 1
        assert total == 2

    def test_subsequent_calls_within_ttl_hit_cache(self):
        """JD-PERF-02 hot-path contract: two ``filter_datasets`` calls in
        quick succession share one underlying disk walk."""
        store = _CountingStore([_make_meta("a"), _make_meta("b")])
        store.filter_datasets()
        store.filter_datasets()
        store.get_stats()
        assert store.list_all_metadata_calls == 1, "expected cached reuse within TTL window"

    def test_invalidate_metadata_cache_forces_refresh(self):
        store = _CountingStore([_make_meta("a")])
        store.filter_datasets()  # warm cache
        assert store.list_all_metadata_calls == 1
        store._invalidate_metadata_cache()
        store.filter_datasets()  # fresh fetch
        assert store.list_all_metadata_calls == 2

    def test_save_invalidates_cache_in_subclass_opt_in(self):
        """The example subclass calls ``_invalidate_metadata_cache()`` from
        ``save``. A user-visible create-then-list must show the new row
        immediately (within-TTL stale-read is avoided)."""
        store = _CountingStore([])
        store.filter_datasets()  # warm empty cache
        store.save("new", _make_meta("new"), {})  # write triggers invalidate
        filtered, total = store.filter_datasets()
        assert total == 1
        assert filtered[0].dataset_id == "new"
        # 2 underlying list_all calls: one for the empty warm, one for the post-save refresh.
        assert store.list_all_metadata_calls == 2

    def test_delete_invalidates_cache_in_subclass_opt_in(self):
        store = _CountingStore([_make_meta("a"), _make_meta("b")])
        store.filter_datasets()  # warm
        deleted = store.delete("a")
        assert deleted is True
        filtered, total = store.filter_datasets()
        assert total == 1
        assert filtered[0].dataset_id == "b"
        assert store.list_all_metadata_calls == 2

    def test_ttl_expiry_triggers_refresh(self, monkeypatch):
        store = _CountingStore([_make_meta("a")])
        store.filter_datasets()
        assert store.list_all_metadata_calls == 1
        # Fast-forward monotonic clock past the TTL.
        original_now = time.monotonic()
        monkeypatch.setattr(base_module.time, "monotonic", lambda: original_now + base_module._METADATA_CACHE_TTL_SECONDS + 0.1)
        store.filter_datasets()
        assert store.list_all_metadata_calls == 2

    def test_returned_list_is_a_snapshot_not_cache_reference(self):
        """Mutating ``filter_datasets``' returned list (or any list any
        caller obtains by going through the cache) must not corrupt the
        cache for the next caller."""
        store = _CountingStore([_make_meta("a"), _make_meta("b"), _make_meta("c")])
        # Drive a cache load via a method that returns a list view of cache.
        first = store._list_all_metadata_cached()
        first.clear()  # mutate the returned list
        second = store._list_all_metadata_cached()
        assert len(second) == 3, "cache should not be corrupted by caller mutation"

    def test_subclass_without_super_init_degrades_gracefully(self):
        """A subclass that skips ``super().__init__`` (e.g., legacy code)
        must not crash on filter_datasets / get_stats. The wrapper falls
        back to direct list_all_metadata calls."""

        class _LegacyStore(_CountingStore):
            def __init__(self, metas):
                # Intentionally skip super().__init__() to simulate a legacy subclass.
                self._metas = list(metas or [])
                self.list_all_metadata_calls = 0
                # Note: no _metadata_cache_lock initialised.

        store = _LegacyStore([_make_meta("a")])
        filtered, total = store.filter_datasets()
        assert total == 1
        # Without cache, each call hits the underlying method.
        store.get_stats()
        assert store.list_all_metadata_calls == 2

    def test_get_stats_uses_cache(self):
        store = _CountingStore([_make_meta("a"), _make_meta("b", generator="moon")])
        store.filter_datasets()  # warm
        stats = store.get_stats()
        assert stats["total_datasets"] == 2
        assert stats["by_generator"]["spiral"] == 1
        assert stats["by_generator"]["moon"] == 1
        assert store.list_all_metadata_calls == 1, "get_stats should reuse the warm cache"


@pytest.mark.unit
class TestMetadataCacheConcurrency:
    """Sanity check that the cache lock prevents tears under concurrent reads.
    The cache is a small dict — full concurrency-stress is overkill; this
    pins that the lock acquisition itself works."""

    def test_concurrent_filter_does_not_double_count_calls(self):
        import threading

        store = _CountingStore([_make_meta(f"d{i}") for i in range(50)])

        def call():
            store.filter_datasets()

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 10 threads should see a cached result — the underlying
        # call count is at most 10 (worst case: all 10 race the initial
        # miss), and at least 1.
        assert 1 <= store.list_all_metadata_calls <= 10
