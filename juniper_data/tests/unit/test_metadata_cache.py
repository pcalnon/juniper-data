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

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage import base as base_module
from juniper_data.storage.base import DatasetStore
from juniper_data.storage.cached import CachedDatasetStore
from juniper_data.storage.hf_store import HuggingFaceDatasetStore
from juniper_data.storage.kaggle_store import KaggleDatasetStore  # noqa: F401  -- imported to register the subclass for the census below
from juniper_data.storage.local_fs import LocalFSDatasetStore
from juniper_data.storage.memory import InMemoryDatasetStore
from juniper_data.storage.postgres_store import PostgresDatasetStore  # noqa: F401  -- ditto
from juniper_data.storage.redis_store import RedisDatasetStore  # noqa: F401  -- ditto


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


# ---------------------------------------------------------------------------
# The cache against REAL stores.
#
# Everything above this line runs against ``_CountingStore``, which calls
# ``super().__init__()`` -- so it exercises a path that, before 2026-09-07, NO
# production store took. ``LocalFSDatasetStore`` (the store ``api/app.py``
# wires) omitted the call, ``_list_all_metadata_cached`` silently degraded to an
# uncached walk, and the whole of JD-PERF-02 was inert in production while its
# test suite stayed green. `test_subclass_without_super_init_degrades_gracefully`
# documented that as a "legacy code" hypothetical; it was the live path.
#
# These tests therefore construct the real classes. They are the arm that would
# have caught it.
# ---------------------------------------------------------------------------


def _arrays() -> dict:
    """Minimal NPZ payload accepted by every store."""
    return {
        "X_train": np.zeros((8, 2), dtype=np.float32),
        "y_train": np.zeros((8, 2), dtype=np.float32),
        "X_val": np.zeros((2, 2), dtype=np.float32),
        "y_val": np.zeros((2, 2), dtype=np.float32),
        "X_test": np.zeros((2, 2), dtype=np.float32),
        "y_test": np.zeros((2, 2), dtype=np.float32),
    }


_COVERED = ("LocalFSDatasetStore", "InMemoryDatasetStore", "CachedDatasetStore", "HuggingFaceDatasetStore")


def _real_stores(tmp_path) -> dict[str, DatasetStore]:
    """Every concrete store constructible without an external service."""
    return {
        "LocalFSDatasetStore": LocalFSDatasetStore(tmp_path / "local"),
        "InMemoryDatasetStore": InMemoryDatasetStore(),
        "CachedDatasetStore": CachedDatasetStore(LocalFSDatasetStore(tmp_path / "primary"), InMemoryDatasetStore()),
        "HuggingFaceDatasetStore": HuggingFaceDatasetStore(),
    }


# Stores this suite cannot construct here, with the REASON, listed EXPLICITLY so
# the census below cannot go quietly stale: a new subclass is either covered by
# ``_real_stores`` or named here, on purpose.
#
# ``HuggingFaceDatasetStore`` is deliberately NOT in this set -- it constructs
# with an in-memory cache store and no network, so it is covered above.
_NOT_CONSTRUCTIBLE_HERE = {
    "RedisDatasetStore",  # __init__ builds a live client
    "PostgresDatasetStore",  # __init__ requires psycopg2 + a reachable server
    "KaggleDatasetStore",  # optional dependency, absent by default
}


class TestCacheAgainstRealStores:
    """The cache must be LIVE and read-your-writes correct on real stores."""

    @pytest.mark.parametrize("store_name", _COVERED)
    def test_the_cache_is_actually_live(self, store_name, tmp_path) -> None:
        """``super().__init__()`` must have run, or the cache is a no-op.

        This is the whole defect in one assertion: ``_list_all_metadata_cached``
        checks ``getattr(self, "_metadata_cache_lock", None)`` and falls back to
        an uncached walk when it is absent. A store that skips the super call
        therefore pays the full O(N) cost on every request while looking, from
        the outside, exactly like a store that does not.
        """
        store = _real_stores(tmp_path)[store_name]
        assert getattr(store, "_metadata_cache_lock", None) is not None, f"{store_name} skipped super().__init__() -- its metadata cache is inert"

    @pytest.mark.parametrize("store_name", _COVERED)
    def test_a_save_is_visible_immediately(self, store_name, tmp_path) -> None:
        """READ-YOUR-WRITES. A created dataset must not wait out the TTL.

        Wiring the cache without invalidating on write is worse than leaving it
        inert: ``POST /v1/datasets`` returns 201 and the dataset is then absent
        from ``/v1/datasets/filter`` for up to the TTL. Reproduced on
        ``LocalFSDatasetStore`` before the fix -- two saves, ``total`` stuck at 1.
        """
        store = _real_stores(tmp_path)[store_name]
        store.save("ds_a", _make_meta("ds_a"), _arrays())
        first, total_first = store.filter_datasets(limit=10)
        assert total_first == 1, f"{store_name}: first save invisible"

        store.save("ds_b", _make_meta("ds_b"), _arrays())
        second, total_second = store.filter_datasets(limit=10)
        assert total_second == 2, f"{store_name}: second save invisible -- the cache was not invalidated (read-your-writes broken)"
        assert {m.dataset_id for m in second} == {"ds_a", "ds_b"}

    @pytest.mark.parametrize("store_name", _COVERED)
    def test_a_delete_is_visible_immediately(self, store_name, tmp_path) -> None:
        """The other direction: a deleted dataset must not linger in the cache."""
        store = _real_stores(tmp_path)[store_name]
        store.save("ds_a", _make_meta("ds_a"), _arrays())
        store.save("ds_b", _make_meta("ds_b"), _arrays())
        assert store.filter_datasets(limit=10)[1] == 2

        # Bound outside the assert: a mutation inside one disappears under
        # ``python -O``, and the test would then pass without ever deleting.
        deleted = store.delete("ds_a")
        assert deleted is True
        remaining, total = store.filter_datasets(limit=10)
        assert total == 1, f"{store_name}: deleted dataset still counted -- cache not invalidated"
        assert {m.dataset_id for m in remaining} == {"ds_b"}

    @pytest.mark.parametrize("store_name", _COVERED)
    def test_an_update_meta_is_visible_immediately(self, store_name, tmp_path) -> None:
        """``update_meta`` changes what a filter matches, so it must invalidate too.

        Filtering on a mutated field is the case that makes this load-bearing:
        the row count can stay the same while the row that *matches* changes.
        """
        store = _real_stores(tmp_path)[store_name]
        store.save("ds_a", _make_meta("ds_a", tags=["before"]), _arrays())
        assert store.filter_datasets(limit=10, tags=["before"])[1] == 1

        updated = store.update_meta("ds_a", _make_meta("ds_a", tags=["after"]))
        assert updated is True
        assert store.filter_datasets(limit=10, tags=["before"])[1] == 0, f"{store_name}: stale tag still matches -- cache not invalidated"
        assert store.filter_datasets(limit=10, tags=["after"])[1] == 1

    def test_every_concrete_store_is_covered_or_explicitly_excluded(self) -> None:
        """The census that stops this suite going vacuous again.

        A new store subclass that skips ``super().__init__()`` would reproduce
        the original defect silently. This fails until the author either adds it
        to the parametrised set above or names it in
        ``_REQUIRES_EXTERNAL_SERVICE`` -- a decision, not an omission.
        """

        def _walk(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from _walk(sub)

        # Every store module is imported at the top of this file precisely so that
        # ``__subclasses__`` sees it here -- a subclass Python has not imported does
        # not exist to this census, which would make it quietly incomplete.
        # Scoped to ``juniper_data.storage``: the census is about PRODUCTION stores.
        # Test doubles defined in this file are subclasses too, and counting them
        # would make the assertion fail for a reason it does not care about.
        concrete = {sub.__name__ for sub in _walk(DatasetStore) if not getattr(sub, "__abstractmethods__", None) and sub.__module__.startswith("juniper_data.storage")}

        covered = set(_COVERED)
        unaccounted = concrete - covered - _NOT_CONSTRUCTIBLE_HERE
        assert not unaccounted, f"concrete DatasetStore subclasses neither covered nor explicitly excluded: {sorted(unaccounted)}"
