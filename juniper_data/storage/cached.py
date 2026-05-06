"""Cached dataset storage wrapper for composable caching layers."""

import contextlib
import logging

import numpy as np

from juniper_data.api.observability import set_datasets_cached
from juniper_data.core.models import DatasetMeta
from juniper_data.storage.constants import DEFAULT_LIST_LIMIT, DEFAULT_LIST_OFFSET

logger = logging.getLogger(__name__)
from .base import DatasetStore

# Probe limit used when sampling the cache backend for the
# ``juniper_data_datasets_cached`` gauge. Mirrors the limit used by
# :meth:`CachedDatasetStore.warm_cache` so the gauge reflects the same
# population that warm_cache would touch. Cache backends are expected
# to be in-memory (Redis / InMemoryDatasetStore) so a SCAN over 10k
# keys is cheap relative to a dataset save/load.
_CACHE_COUNT_PROBE_LIMIT: int = 10_000


class CachedDatasetStore(DatasetStore):
    """Composable caching wrapper for dataset storage.

    Wraps a primary store with a cache store for read-through caching.
    Writes go to both stores; reads check cache first, then primary.

    Example:
        primary = LocalFSDatasetStore(Path("./data"))
        cache = RedisDatasetStore(host="localhost")
        store = CachedDatasetStore(primary, cache)
    """

    def __init__(
        self,
        primary: DatasetStore,
        cache: DatasetStore,
        write_through: bool = True,
    ) -> None:
        """Initialize the cached store.

        Args:
            primary: Primary (persistent) storage backend.
            cache: Cache storage backend (e.g., Redis, InMemory).
            write_through: If True, writes go to both stores. If False,
            writes only go to primary and cache is populated on read.
        """
        self._primary = primary
        self._cache = cache
        self._write_through = write_through

    def _emit_cached_count(self) -> None:
        """Update the ``juniper_data_datasets_cached`` Prometheus gauge.

        Probes the cache backend for its current dataset population and
        publishes the count via :func:`juniper_data.api.observability.set_datasets_cached`.
        Failures (cache backend unavailable, metric registry not yet
        initialised, etc.) are swallowed so observability never breaks
        the storage path -- mirrors the ``contextlib.suppress(Exception)``
        discipline used everywhere else in this class.
        """
        try:
            count = len(self._cache.list_datasets(limit=_CACHE_COUNT_PROBE_LIMIT))
            set_datasets_cached(count)
        except Exception:
            logger.debug("Failed to update juniper_data_datasets_cached gauge", exc_info=True)

    def save(
        self,
        dataset_id: str,
        meta: DatasetMeta,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Save dataset to primary store (and optionally cache).

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Dataset metadata.
            arrays: Dictionary of numpy arrays.
        """
        self._primary.save(dataset_id, meta, arrays)

        if self._write_through:
            with contextlib.suppress(Exception):
                self._cache.save(dataset_id, meta, arrays)
            self._emit_cached_count()

    def get_meta(self, dataset_id: str) -> DatasetMeta | None:
        """Get metadata, checking cache first.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dataset metadata if found, None otherwise.
        """
        with contextlib.suppress(Exception):
            cached = self._cache.get_meta(dataset_id)
            if cached is not None:
                return cached
        return self._primary.get_meta(dataset_id)

    def get_artifact_bytes(self, dataset_id: str) -> bytes | None:
        """Get artifact bytes, checking cache first.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            NPZ bytes if found, None otherwise.
        """
        with contextlib.suppress(Exception):
            cached = self._cache.get_artifact_bytes(dataset_id)
            if cached is not None:
                return cached
        artifact = self._primary.get_artifact_bytes(dataset_id)

        if artifact is not None:
            populated = False
            with contextlib.suppress(Exception):
                meta = self._primary.get_meta(dataset_id)
                if meta is not None:
                    import io

                    with np.load(io.BytesIO(artifact)) as npz:
                        arrays = {k: npz[k] for k in npz.files}
                    self._cache.save(dataset_id, meta, arrays)
                    populated = True
            if populated:
                self._emit_cached_count()
        return artifact

    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists in either store.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset exists, False otherwise.
        """
        with contextlib.suppress(Exception):
            if self._cache.exists(dataset_id):
                return True
        return self._primary.exists(dataset_id)

    def delete(self, dataset_id: str) -> bool:
        """Delete dataset from both stores.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset was deleted from primary, False otherwise.
        """
        cache_touched = False
        with contextlib.suppress(Exception):
            self._cache.delete(dataset_id)
            cache_touched = True
        if cache_touched:
            self._emit_cached_count()
        return self._primary.delete(dataset_id)

    def list_datasets(self, limit: int = DEFAULT_LIST_LIMIT, offset: int = DEFAULT_LIST_OFFSET) -> list[str]:
        """List datasets from primary store.

        Args:
            limit: Maximum number of dataset IDs to return.
            offset: Number of dataset IDs to skip.

        Returns:
            List of dataset IDs.
        """
        return self._primary.list_datasets(limit, offset)

    def update_meta(self, dataset_id: str, meta: DatasetMeta) -> bool:
        """Update metadata in both stores.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Updated dataset metadata.

        Returns:
            True if the dataset was updated in primary, False otherwise.
        """
        result = self._primary.update_meta(dataset_id, meta)

        if result:
            with contextlib.suppress(Exception):
                self._cache.update_meta(dataset_id, meta)
        return result

    def list_all_metadata(self) -> list[DatasetMeta]:
        """List all metadata from primary store.

        Returns:
            List of all DatasetMeta objects.
        """
        return self._primary.list_all_metadata()

    def invalidate_cache(self, dataset_id: str) -> bool:
        """Invalidate a specific entry in the cache.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if entry was removed from cache, False otherwise.
        """
        try:
            result = self._cache.delete(dataset_id)
        except Exception:
            return False
        self._emit_cached_count()
        return result

    def warm_cache(self, dataset_ids: list[str] | None = None) -> int:
        """Populate cache from primary store.

        Args:
            dataset_ids: Specific IDs to cache, or None for all.

        Returns:
            Number of datasets cached.
        """
        if dataset_ids is None:
            dataset_ids = self._primary.list_datasets(limit=10000)

        cached_count = 0
        for dataset_id in dataset_ids:
            try:
                meta = self._primary.get_meta(dataset_id)
                artifact = self._primary.get_artifact_bytes(dataset_id)

                if meta is not None and artifact is not None:
                    import io

                    with np.load(io.BytesIO(artifact)) as npz:
                        arrays = {k: npz[k] for k in npz.files}
                        self._cache.save(dataset_id, meta, arrays)
                    cached_count += 1
            except Exception:
                logger.warning("Failed to cache dataset %s", dataset_id, exc_info=True)
                continue

        if cached_count > 0:
            self._emit_cached_count()
        return cached_count
