"""In-memory dataset store for testing and development."""

import io

import numpy as np

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.base import DatasetStore



class InMemoryDatasetStore(DatasetStore):
    """In-memory implementation of DatasetStore.

    Stores datasets in dictionaries. Useful for testing and development.
    Data is lost when the process exits.
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        self._metadata: dict[str, DatasetMeta] = {}
        self._arrays: dict[str, dict[str, np.ndarray]] = {}

    def save(
        self,
        dataset_id: str,
        meta: DatasetMeta,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Save dataset metadata and arrays to memory.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Dataset metadata.
            arrays: Dictionary of numpy arrays.
        """
        self._metadata[dataset_id] = meta
        self._arrays[dataset_id] = {k: v.copy() for k, v in arrays.items()}

    def get_meta(self, dataset_id: str) -> DatasetMeta | None:
        """Get dataset metadata from memory.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dataset metadata if found, None otherwise.
        """
        return self._metadata.get(dataset_id)

    def get_artifact_bytes(self, dataset_id: str) -> bytes | None:
        """Get dataset artifact as NPZ bytes.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            NPZ file contents as bytes if found, None otherwise.
        """
        arrays = self._arrays.get(dataset_id)
        if arrays is None:
            return None

        buffer = io.BytesIO()
        # Sort keys to ensure stable NPZ artifact bytes regardless of dict construction order.
        sorted_arrays = {key: arrays[key] for key in sorted(arrays.keys())}
        np.savez_compressed(buffer, **sorted_arrays)
        buffer.seek(0)
        return buffer.read()

    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists in memory.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset exists, False otherwise.
        """
        return dataset_id in self._metadata

    def delete(self, dataset_id: str) -> bool:
        """Delete dataset from memory.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset was deleted, False if it didn't exist.
        """
        if dataset_id not in self._metadata:
            return False

        del self._metadata[dataset_id]
        del self._arrays[dataset_id]
        return True

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List dataset IDs from memory.

        Args:
            limit: Maximum number of dataset IDs to return.
            offset: Number of dataset IDs to skip.

        Returns:
            List of dataset IDs.
        """
        all_ids = sorted(self._metadata.keys())
        return all_ids[offset : offset + limit]

    def clear(self) -> None:
        """Clear all stored datasets. Useful for test cleanup."""
        self._metadata.clear()
        self._arrays.clear()

    def update_meta(self, dataset_id: str, meta: DatasetMeta) -> bool:
        """Update dataset metadata in memory.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Updated dataset metadata.

        Returns:
            True if the dataset was updated, False if it didn't exist.
        """
        if dataset_id not in self._metadata:
            return False
        self._metadata[dataset_id] = meta
        return True

    def list_all_metadata(self) -> list[DatasetMeta]:
        """List all dataset metadata from memory.

        Returns:
            List of all DatasetMeta objects.
        """
        return list(self._metadata.values())
