"""Cached dataset storage wrapper for composable caching layers."""

import contextlib
import logging

import numpy as np

from juniper_data.core.models import DatasetMeta

logger = logging.getLogger(__name__)
from .base import DatasetStore


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
            with contextlib.suppress(Exception):
                meta = self._primary.get_meta(dataset_id)
                if meta is not None:
                    import io

                    with np.load(io.BytesIO(artifact)) as npz:
                        arrays = {k: npz[k] for k in npz.files}
                    self._cache.save(dataset_id, meta, arrays)
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
        with contextlib.suppress(Exception):
            self._cache.delete(dataset_id)
        return self._primary.delete(dataset_id)

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
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
            return self._cache.delete(dataset_id)
        except Exception:
            return False

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

        return cached_count
