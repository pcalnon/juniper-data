"""Phase 3D regression tests for `DatasetStore.record_access` atomicity.

CONC-12 / BUG-JD-11 — `record_access` performs a non-atomic
read-modify-write (`get_meta` → in-memory increment → `update_meta`).
Two concurrent requests racing on the same dataset both used to read
the same count, both increment locally, and both write the *same* new
value, so one access was silently dropped on every collision.

The fix wraps the entire sequence in `with self._version_lock:`. The
class-level lock means access counting is now serialized cross-instance
within a single process; per-process best-effort still applies in
multi-process deployments (see BUG-JD-05), but for the in-process race
the increment is now exact.

Tests below use `InMemoryDatasetStore` so they exercise the abstract
`record_access` directly, no filesystem or Postgres needed. To make the
read-modify-write window observable under CPython's GIL (which makes
the bare dict assignment essentially atomic), `update_meta` is wrapped
with a small sleep so concurrent threads can race on the round-trip.
Without the lock the test fails reliably with N concurrent calls
producing fewer than N increments; with the lock the count is exactly N.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage import InMemoryDatasetStore


def _meta() -> DatasetMeta:
    return DatasetMeta(
        dataset_id="conc12",
        generator="spiral",
        generator_version="1.0.0",
        params={},
        n_samples=10,
        n_features=2,
        n_classes=2,
        n_train=8,
        n_test=2,
        class_distribution={"0": 5, "1": 5},
        artifact_formats=["npz"],
        created_at=datetime.now(UTC),
        checksum="x",
    )


def _arrays() -> dict[str, np.ndarray]:
    return {
        "X_train": np.zeros((8, 2), dtype=np.float32),
        "y_train": np.zeros((8, 2), dtype=np.float32),
        "X_test": np.zeros((2, 2), dtype=np.float32),
        "y_test": np.zeros((2, 2), dtype=np.float32),
    }


@pytest.mark.unit
class TestRecordAccessAtomicity:
    """CONC-12 / BUG-JD-11 regression cover."""

    def test_record_access_holds_version_lock_across_read_and_write(self, monkeypatch):
        """`record_access` must hold `_version_lock` across get_meta + update_meta.

        Verifies the lock is held at every observed point in the
        read-modify-write sequence by swapping `_version_lock` for a
        tracing wrapper and recording the held-state at each call. The
        pre-fix code never acquired the lock, so the tracing wrapper
        records all-False; the fix records all-True.
        """

        class _TracingLock:
            def __init__(self) -> None:
                self._real = threading.RLock()
                self._depth = 0

            @property
            def held(self) -> bool:
                return self._depth > 0

            def __enter__(self):
                self._real.acquire()
                self._depth += 1
                return self

            def __exit__(self, *_a):
                self._depth -= 1
                self._real.release()

        store = InMemoryDatasetStore()
        store.save("conc12", _meta(), _arrays())

        tracing_lock = _TracingLock()
        # The lock lives on the class — replace it for the duration of this
        # test. monkeypatch.setattr restores it automatically.
        monkeypatch.setattr(InMemoryDatasetStore, "_version_lock", tracing_lock)

        observations: list = []
        original_get_meta = store.get_meta
        original_update_meta = store.update_meta

        def traced_get_meta(dataset_id: str):
            observations.append(("get_meta", tracing_lock.held))
            return original_get_meta(dataset_id)

        def traced_update_meta(dataset_id: str, meta: DatasetMeta) -> bool:
            observations.append(("update_meta", tracing_lock.held))
            return original_update_meta(dataset_id, meta)

        monkeypatch.setattr(store, "get_meta", traced_get_meta)
        monkeypatch.setattr(store, "update_meta", traced_update_meta)

        store.record_access("conc12")

        assert observations == [("get_meta", True), ("update_meta", True)], f"CONC-12 regressed: lock was not held across record_access — observed {observations!r}"

    def test_concurrent_record_access_no_lost_updates(self, monkeypatch):
        """N concurrent record_access calls must produce exactly N increments.

        Widens the read-modify-write race window deterministically by
        copying the meta inside `get_meta` (so each thread gets its own
        snapshot — the InMemoryDatasetStore normally returns the shared
        reference, masking the race). With the lock the increment is
        serialized; without it the threads collide.
        """
        from copy import deepcopy

        store = InMemoryDatasetStore()
        store.save("conc12", _meta(), _arrays())

        original_get_meta = store.get_meta
        original_update_meta = store.update_meta

        def copy_get_meta(dataset_id: str):
            # Force per-thread snapshot so increments don't accidentally
            # share an object reference and trivially compose.
            meta = original_get_meta(dataset_id)
            return deepcopy(meta) if meta is not None else None

        def slow_update_meta(dataset_id: str, meta: DatasetMeta) -> bool:
            # Widen the gap between `get_meta` and the actual write so
            # concurrent threads observably race on the snapshotted count.
            time.sleep(0.001)
            return original_update_meta(dataset_id, meta)

        monkeypatch.setattr(store, "get_meta", copy_get_meta)
        monkeypatch.setattr(store, "update_meta", slow_update_meta)

        n_threads = 16
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            store.record_access("conc12")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = original_get_meta("conc12")
        assert final is not None
        assert final.access_count == n_threads, f"CONC-12 race: {n_threads} concurrent record_access calls produced only {final.access_count} increments — lost updates"

    def test_record_access_nonexistent_still_noop(self):
        """Locking the read-modify-write must not change the existing no-op behaviour."""
        store = InMemoryDatasetStore()
        store.record_access("missing")
        assert not store.exists("missing")

    def test_record_access_updates_last_accessed_at(self):
        """The timestamp + counter increment still both apply after the fix."""
        store = InMemoryDatasetStore()
        store.save("conc12", _meta(), _arrays())
        before = datetime.now(UTC)

        store.record_access("conc12")

        after_meta = store.get_meta("conc12")
        assert after_meta is not None
        assert after_meta.access_count == 1
        assert after_meta.last_accessed_at is not None
        assert after_meta.last_accessed_at >= before
