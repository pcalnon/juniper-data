"""Abstract base class for dataset storage."""

import threading
from abc import ABC, abstractmethod

# from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from juniper_data.core.models import DatasetMeta

# from typing import Dict, List, Optional


class DatasetStore(ABC):
    """Abstract dataset storage interface.

    Provides a common interface for storing and retrieving datasets,
    supporting different backends (in-memory, local filesystem, cloud, etc.).
    """

    _version_lock = threading.Lock()

    @abstractmethod
    def save(
        self,
        dataset_id: str,
        meta: DatasetMeta,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Save dataset metadata and arrays.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Dataset metadata.
            arrays: Dictionary of numpy arrays (e.g., X_train, y_train, etc.).

        Raises:
            IOError: If the save operation fails.
        """
        pass

    @abstractmethod
    def get_meta(self, dataset_id: str) -> DatasetMeta | None:
        """Get dataset metadata.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dataset metadata if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_artifact_bytes(self, dataset_id: str) -> bytes | None:
        """Get dataset artifact as bytes (NPZ format).

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            NPZ file contents as bytes if found, None otherwise.
        """
        pass

    @abstractmethod
    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset exists, False otherwise.
        """
        pass

    @abstractmethod
    def delete(self, dataset_id: str) -> bool:
        """Delete dataset.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset was deleted, False if it didn't exist.
        """
        pass

    @abstractmethod
    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List dataset IDs.

        Args:
            limit: Maximum number of dataset IDs to return.
            offset: Number of dataset IDs to skip.

        Returns:
            List of dataset IDs.
        """
        pass

    def update_meta(self, dataset_id: str, meta: DatasetMeta) -> bool:
        """Update dataset metadata.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Updated dataset metadata.

        Returns:
            True if the dataset was updated, False if it didn't exist.
        """
        raise NotImplementedError("update_meta not implemented for this storage backend")

    def list_all_metadata(self) -> list[DatasetMeta]:
        """List all dataset metadata (for filtering/stats).

        Returns:
            List of all DatasetMeta objects.
        """
        raise NotImplementedError("list_all_metadata not implemented for this storage backend")

    def record_access(self, dataset_id: str) -> None:
        """Record an access to a dataset (updates last_accessed_at and access_count).

        Args:
            dataset_id: Unique identifier for the dataset.

        CONC-12 / BUG-JD-11 (Phase 3D): the read-modify-write on
        ``access_count`` happens across three calls — ``get_meta``, the
        in-memory increment, and ``update_meta``. Two concurrent requests
        racing on the same dataset both used to read the same count, both
        increment locally, and both write back the *same* new value, so
        one access was silently lost from the counter on every collision.
        Hold the existing ``_version_lock`` across the whole sequence so
        the increment is atomic from the perspective of any other thread
        in the same process. The plan's caveat applies: this is a
        per-process lock, so multi-process deployments accept best-effort
        counting (see BUG-JD-05). Access counting is informational so
        this is an acceptable trade-off.
        """
        with self._version_lock:
            meta = self.get_meta(dataset_id)
            if meta is not None:
                meta.last_accessed_at = datetime.now(UTC)
                meta.access_count += 1
                self.update_meta(dataset_id, meta)

    def is_expired(self, meta: DatasetMeta) -> bool:
        """Check if a dataset has expired based on its TTL.

        Args:
            meta: Dataset metadata.

        Returns:
            True if the dataset has expired, False otherwise.
        """
        if meta.expires_at is None:
            return False
        return datetime.now(UTC) > meta.expires_at

    def delete_expired(self) -> list[str]:
        """Delete all expired datasets.

        Returns:
            List of dataset IDs that were deleted.
        """
        deleted: list[str] = []
        deleted.extend(meta.dataset_id for meta in self.list_all_metadata() if self.is_expired(meta) and self.delete(meta.dataset_id))
        return deleted

    def list_versions(self, dataset_name: str) -> list[DatasetMeta]:
        """List all versions of a named dataset, sorted by version ascending.

        Args:
            dataset_name: The logical dataset name.

        Returns:
            List of DatasetMeta objects sorted by version number ascending.
        """
        all_meta = self.list_all_metadata()
        versions = [m for m in all_meta if m.dataset_name == dataset_name]
        versions.sort(key=lambda m: m.dataset_version or 0)
        return versions

    def get_latest_version(self, dataset_name: str) -> DatasetMeta | None:
        """Get the latest version of a named dataset.

        Args:
            dataset_name: The logical dataset name.

        Returns:
            DatasetMeta for the latest version, or None if no versions exist.
        """
        versions = self.list_versions(dataset_name)
        return versions[-1] if versions else None

    def next_version_number(self, dataset_name: str) -> int:
        """Get the next version number for a named dataset.

        Note: This method is NOT concurrency-safe on its own. For atomic
        version allocation during save, use ``save_versioned()`` instead.

        Args:
            dataset_name: The logical dataset name.

        Returns:
            The next version number (1 if no versions exist).
        """
        versions = self.list_versions(dataset_name)
        if not versions:
            return 1
        return max(m.dataset_version or 0 for m in versions) + 1

    def save_versioned(
        self,
        dataset_id: str,
        meta: DatasetMeta,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Atomically allocate a version number and save.

        If ``meta.dataset_name`` is set and ``meta.dataset_version`` is None,
        the next version number is computed and assigned under a lock so that
        concurrent callers cannot receive the same version.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Dataset metadata. ``dataset_version`` is set in-place.
            arrays: Dictionary of numpy arrays.
        """
        if meta.dataset_name is not None and meta.dataset_version is None:
            with self._version_lock:
                meta.dataset_version = self.next_version_number(meta.dataset_name)
                self.save(dataset_id, meta, arrays)
        else:
            self.save(dataset_id, meta, arrays)

    def filter_datasets(
        self,
        generator: str | None = None,
        tags: list[str] | None = None,
        tags_match: str = "any",
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_samples: int | None = None,
        max_samples: int | None = None,
        include_expired: bool = False,
        dataset_name: str | None = None,
        dataset_version: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[DatasetMeta], int]:
        """Filter datasets by various criteria.

        Args:
            generator: Filter by generator name.
            tags: Filter by tags.
            tags_match: "any" (OR) or "all" (AND) for tag matching.
            created_after: Filter by creation date (after).
            created_before: Filter by creation date (before).
            min_samples: Minimum number of samples.
            max_samples: Maximum number of samples.
            include_expired: Include expired datasets.
            dataset_name: Filter by logical dataset name.
            dataset_version: Filter by dataset version number.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (filtered metadata list, total count before pagination).
        """
        all_meta = self.list_all_metadata()
        filtered = []

        for meta in all_meta:
            if not include_expired and self.is_expired(meta):
                continue
            if generator is not None and meta.generator != generator:
                continue
            if dataset_name is not None and meta.dataset_name != dataset_name:
                continue
            if dataset_version is not None and meta.dataset_version != dataset_version:
                continue
            if tags is not None:
                if tags_match == "all":
                    if any(t not in meta.tags for t in tags):
                        continue
                elif all(t not in meta.tags for t in tags):
                    continue
            if created_after is not None and meta.created_at < created_after:
                continue
            if created_before is not None and meta.created_at > created_before:
                continue
            if min_samples is not None and meta.n_samples < min_samples:
                continue
            if max_samples is not None and meta.n_samples > max_samples:
                continue
            filtered.append(meta)

        filtered.sort(key=lambda m: m.created_at, reverse=True)
        total = len(filtered)
        return filtered[offset : offset + limit], total

    def batch_delete(self, dataset_ids: list[str]) -> tuple[list[str], list[str]]:
        """Delete multiple datasets.

        Args:
            dataset_ids: List of dataset IDs to delete.

        Returns:
            Tuple of (deleted IDs, not found IDs).
        """
        deleted = []
        not_found = []
        for dataset_id in dataset_ids:
            try:
                ok = self.delete(dataset_id)
            except ValueError:
                # JD-SEC-01: reject traversal attempts without failing the
                # entire batch — classify as not_found so the response still
                # returns cleanly and legitimate IDs in the same request are
                # not penalised.
                not_found.append(dataset_id)
                continue
            if ok:
                deleted.append(dataset_id)
            else:
                not_found.append(dataset_id)
        return deleted, not_found

    def get_stats(self) -> dict[str, object]:
        """Get aggregate statistics about stored datasets.

        Returns:
            Dictionary with statistics.
        """
        all_meta = self.list_all_metadata()

        if not all_meta:
            return {
                "total_datasets": 0,
                "total_samples": 0,
                "by_generator": {},
                "by_tag": {},
                "oldest_created_at": None,
                "newest_created_at": None,
                "expired_count": 0,
            }

        by_generator: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        total_samples = 0
        expired_count = 0
        created_times = []

        for meta in all_meta:
            by_generator[meta.generator] = by_generator.get(meta.generator, 0) + 1
            for tag in meta.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1
            total_samples += meta.n_samples
            created_times.append(meta.created_at)
            if self.is_expired(meta):
                expired_count += 1

        return {
            "total_datasets": len(all_meta),
            "total_samples": total_samples,
            "by_generator": by_generator,
            "by_tag": by_tag,
            "oldest_created_at": min(created_times),
            "newest_created_at": max(created_times),
            "expired_count": expired_count,
        }
