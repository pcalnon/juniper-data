"""Abstract base class for dataset storage."""

from abc import ABC, abstractmethod

# from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np

from juniper_data.core.models import DatasetMeta

# from typing import Dict, List, Optional


class DatasetStore(ABC):
    """Abstract dataset storage interface.

    Provides a common interface for storing and retrieving datasets,
    supporting different backends (in-memory, local filesystem, cloud, etc.).
    """

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
        """
        meta = self.get_meta(dataset_id)
        if meta is not None:
            meta.last_accessed_at = datetime.now(timezone.utc)
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
        return datetime.now(timezone.utc) > meta.expires_at

    def delete_expired(self) -> list[str]:
        """Delete all expired datasets.

        Returns:
            List of dataset IDs that were deleted.
        """
        deleted: list[str] = []
        deleted.extend(meta.dataset_id for meta in self.list_all_metadata() if self.is_expired(meta) and self.delete(meta.dataset_id))
        return deleted

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
            if self.delete(dataset_id):
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
