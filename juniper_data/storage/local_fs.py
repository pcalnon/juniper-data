"""Local filesystem dataset store."""

import io
import json
import logging
from datetime import datetime
from pathlib import Path

# from typing import Any, Dict, List, Optional
from typing import Any

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
            base_path: Base directory for storing datasets. Created if it doesn't exist.
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
        arrays: dict[str, np.ndarray],
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

        # Write to temporary files first, then atomically replace the final files
        tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
        tmp_npz_path = npz_path.with_suffix(npz_path.suffix + ".tmp")

        meta_json = json.dumps(
            meta.model_dump(),
            default=_json_serializer,
            indent=2,
        )

        try:
            # Write metadata JSON to temporary file
            tmp_meta_path.write_text(meta_json, encoding="utf-8")

            # Write NPZ data to temporary file
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]  # numpy stubs incomplete for **kwargs
            buffer.seek(0)
            tmp_npz_path.write_bytes(buffer.read())

            # Atomically replace final files with the temporary ones.
            # Write NPZ first so we never have metadata without its NPZ.
            tmp_npz_path.replace(npz_path)
            tmp_meta_path.replace(meta_path)
        except Exception:
            # Best-effort cleanup of temporary files on failure
            try:
                tmp_meta_path.unlink(missing_ok=True)
            except OSError:
                logging.debug(
                    "Failed to remove temporary metadata file %s during cleanup",
                    tmp_meta_path,
                    exc_info=True,
                )
            try:
                tmp_npz_path.unlink(missing_ok=True)
            except OSError:
                logging.debug(
                    "Failed to remove temporary NPZ file %s during cleanup",
                    tmp_npz_path,
                    exc_info=True,
                )
            raise

    def get_meta(self, dataset_id: str) -> DatasetMeta | None:
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

        return DatasetMeta(**meta_dict)

    def get_artifact_bytes(self, dataset_id: str) -> bytes | None:
        """Get dataset artifact as NPZ bytes.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            NPZ file contents as bytes if found, None otherwise.
        """
        npz_path = self._npz_path(dataset_id)
        return npz_path.read_bytes() if npz_path.exists() else None

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

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
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

    def update_meta(self, dataset_id: str, meta: DatasetMeta) -> bool:
        """Update dataset metadata on filesystem.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Updated dataset metadata.

        Returns:
            True if the dataset was updated, False if it didn't exist.
        """
        meta_path = self._meta_path(dataset_id)
        if not meta_path.exists():
            return False

        meta_json = json.dumps(
            meta.model_dump(),
            default=_json_serializer,
            indent=2,
        )
        meta_path.write_text(meta_json, encoding="utf-8")
        return True

    def list_all_metadata(self) -> list[DatasetMeta]:
        """List all dataset metadata from filesystem.

        Returns:
            List of all DatasetMeta objects.
        """
        result = []
        for meta_file in self._base_path.glob("*.meta.json"):
            dataset_id = meta_file.stem.replace(".meta", "")
            meta = self.get_meta(dataset_id)
            if meta is not None:
                result.append(meta)
        return result
