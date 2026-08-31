"""Chunked artifact delivery — ``DatasetStore.open_artifact_stream``.

Defect-register ``APD-DATA-016``. ``download_artifact`` returned a
``StreamingResponse`` wrapping ``io.BytesIO(get_artifact_bytes(...))``. That name
is misleading in a way that matters: the ``BytesIO`` wrapper bounds the **socket
buffer**, not process memory. The whole artifact existed in RAM before the
response object did, once per concurrent download, so peak RSS scaled with
artifact size × concurrency while the endpoint advertised "streaming".

What these tests pin
--------------------

* **LocalFS genuinely chunks.** The decisive arm is not "the bytes round-trip" —
  a whole-file read round-trips too. It is that a small ``chunk_size`` yields
  **more than one** chunk, which only a real incremental read can do.
* **The base default still works, and is honest about being whole-read.** It
  yields exactly one chunk, so a backend that cannot do better is unchanged.
  This is why the method is not ``@abstractmethod``: adding it widened the
  interface without a flag day across seven stores.
* **Absence is decided eagerly.** ``None`` must come back from the *call*, not
  from the first iteration. A generator body does not execute until iterated, so
  an existence check placed inside one would defer the 404 until after the route
  had already committed to a 200 and sent headers. This is the arm that fails if
  someone "simplifies" the override by moving the check inside ``_chunks``.
* **The route streams and still 404s**, and the delivered bytes are identical to
  the materialised ones — the change is a memory profile, not a wire change.
"""

from datetime import datetime

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.local_fs import LocalFSDatasetStore
from juniper_data.storage.memory import InMemoryDatasetStore

pytestmark = pytest.mark.unit


@pytest.fixture
def meta() -> DatasetMeta:
    return DatasetMeta(
        dataset_id="stream-001",
        generator="spiral",
        generator_version="1.0.0",
        params={},
        n_samples=64,
        n_features=2,
        n_classes=2,
        n_train=48,
        n_test=16,
        class_distribution={"0": 32, "1": 32},
        created_at=datetime(2026, 1, 30, 12, 0, 0),
    )


@pytest.fixture
def arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "X_full": rng.standard_normal((64, 2)).astype("float32"),
        "y_full": rng.integers(0, 2, size=64).astype("float32"),
    }


@pytest.fixture
def local_store(tmp_path, meta, arrays) -> LocalFSDatasetStore:
    store = LocalFSDatasetStore(base_path=str(tmp_path))
    store.save(meta.dataset_id, meta, arrays)
    return store


@pytest.fixture
def memory_store(meta, arrays) -> InMemoryDatasetStore:
    store = InMemoryDatasetStore()
    store.save(meta.dataset_id, meta, arrays)
    return store


class TestLocalFsStreamsIncrementally:
    def test_small_chunk_size_yields_many_chunks(self, local_store, meta):
        """The decisive arm: a whole-file read cannot produce more than one chunk."""
        chunks = list(local_store.open_artifact_stream(meta.dataset_id, chunk_size=64))
        assert len(chunks) > 1, "a single chunk means the artifact was materialised whole"

    def test_chunk_size_is_honoured(self, local_store, meta):
        chunks = list(local_store.open_artifact_stream(meta.dataset_id, chunk_size=64))
        assert all(len(c) <= 64 for c in chunks)
        # Every chunk but the last should be full, or the reader is not filling them.
        assert all(len(c) == 64 for c in chunks[:-1])

    def test_bytes_are_identical_to_the_materialised_read(self, local_store, meta):
        """A memory-profile change must not be a wire change."""
        streamed = b"".join(local_store.open_artifact_stream(meta.dataset_id, chunk_size=64))
        assert streamed == local_store.get_artifact_bytes(meta.dataset_id)


class TestBaseDefaultFallback:
    def test_inheriting_backend_yields_exactly_one_chunk(self, memory_store, meta):
        """The default is a whole read and does not pretend otherwise."""
        chunks = list(memory_store.open_artifact_stream(meta.dataset_id))
        assert len(chunks) == 1

    def test_default_bytes_match_get_artifact_bytes(self, memory_store, meta):
        streamed = b"".join(memory_store.open_artifact_stream(meta.dataset_id))
        assert streamed == memory_store.get_artifact_bytes(meta.dataset_id)


class TestAbsenceIsDecidedEagerly:
    """``None`` from the call, not from the first iteration.

    If the existence check moves inside the generator body, these return a
    generator object instead of ``None`` — and the route, which branches on
    ``is None`` to raise its 404, would instead return 200 with an empty body.
    """

    def test_local_fs_returns_none_not_a_generator(self, local_store):
        assert local_store.open_artifact_stream("no-such-dataset") is None

    def test_default_returns_none_not_a_generator(self, memory_store):
        assert memory_store.open_artifact_stream("no-such-dataset") is None
