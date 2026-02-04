"""Local filesystem dataset store."""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.base import DatasetStore


def _json_serializer(obj: Any) -> str:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class LocalFSDatasetStore(DatasetStore):
    """Local filesystem implementation of DatasetStore.

    Stores datasets as JSON metadata files and NPZ array files.

    Storage layout:
        {base_path}/{dataset_id}.meta.json
        {base_path}/{dataset_id}.npz
    """

    def __init__(self, base_path: Path) -> None:
        """Initialize the local filesystem store.

        Args:
            base_path: Base directory for storing datasets.
                       Created if it doesn't exist.
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, dataset_id: str) -> Path:
        """Get path to metadata file."""
        return self._base_path / f"{dataset_id}.meta.json"

    def _npz_path(self, dataset_id: str) -> Path:
        """Get path to NPZ file."""
        return self._base_path / f"{dataset_id}.npz"

    def save(
        self,
        dataset_id: str,
        meta: DatasetMeta,
        arrays: Dict[str, np.ndarray],
    ) -> None:
        """Save dataset metadata and arrays to filesystem.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Dataset metadata.
            arrays: Dictionary of numpy arrays.

        Raises:
            IOError: If the save operation fails.
        """
        meta_path = self._meta_path(dataset_id)
        npz_path = self._npz_path(dataset_id)

        meta_json = json.dumps(
            meta.model_dump(),
            default=_json_serializer,
            indent=2,
        )
        meta_path.write_text(meta_json, encoding="utf-8")

        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]  # numpy stubs incomplete for **kwargs
        buffer.seek(0)
        npz_path.write_bytes(buffer.read())

    def get_meta(self, dataset_id: str) -> Optional[DatasetMeta]:
        """Get dataset metadata from filesystem.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dataset metadata if found, None otherwise.
        """
        meta_path = self._meta_path(dataset_id)
        if not meta_path.exists():
            return None

        meta_json = meta_path.read_text(encoding="utf-8")
        meta_dict = json.loads(meta_json)

        if "created_at" in meta_dict and isinstance(meta_dict["created_at"], str):
            meta_dict["created_at"] = datetime.fromisoformat(meta_dict["created_at"])

        return DatasetMeta(**meta_dict)

    def get_artifact_bytes(self, dataset_id: str) -> Optional[bytes]:
        """Get dataset artifact as NPZ bytes.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            NPZ file contents as bytes if found, None otherwise.
        """
        npz_path = self._npz_path(dataset_id)
        if not npz_path.exists():
            return None

        return npz_path.read_bytes()

    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists on filesystem.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if both metadata and NPZ files exist, False otherwise.
        """
        return self._meta_path(dataset_id).exists() and self._npz_path(dataset_id).exists()

    def delete(self, dataset_id: str) -> bool:
        """Delete dataset from filesystem.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset was deleted, False if it didn't exist.
        """
        meta_path = self._meta_path(dataset_id)
        npz_path = self._npz_path(dataset_id)

        if not meta_path.exists() and not npz_path.exists():
            return False

        if meta_path.exists():
            meta_path.unlink()
        if npz_path.exists():
            npz_path.unlink()

        return True

    def list_datasets(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List dataset IDs from filesystem.

        Finds datasets by globbing for .meta.json files.

        Args:
            limit: Maximum number of dataset IDs to return.
            offset: Number of dataset IDs to skip.

        Returns:
            List of dataset IDs.
        """
        meta_files = sorted(self._base_path.glob("*.meta.json"))
        dataset_ids = [f.stem.replace(".meta", "") for f in meta_files]
        return dataset_ids[offset : offset + limit]

    @property
    def base_path(self) -> Path:
        """Get the base storage path."""
        return self._base_path
