"""Abstract base class for dataset storage."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from juniper_data.core.models import DatasetMeta


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
        arrays: Dict[str, np.ndarray],
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
    def get_meta(self, dataset_id: str) -> Optional[DatasetMeta]:
        """Get dataset metadata.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dataset metadata if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_artifact_bytes(self, dataset_id: str) -> Optional[bytes]:
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
    def list_datasets(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List dataset IDs.

        Args:
            limit: Maximum number of dataset IDs to return.
            offset: Number of dataset IDs to skip.

        Returns:
            List of dataset IDs.
        """
        pass
