"""Artifact utilities for NPZ file handling and checksum computation."""

import hashlib
import io
from pathlib import Path

import numpy as np


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Save arrays to NPZ file.

    Args:
        path: Path to save the NPZ file.
        arrays: Dictionary mapping array names to numpy arrays.
    """
    np.savez(path, **arrays)  # type: ignore[arg-type]  # numpy stubs incomplete for **kwargs


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load arrays from NPZ file.

    Args:
        path: Path to the NPZ file.

    Returns:
        Dictionary mapping array names to numpy arrays.
    """
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def arrays_to_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Convert arrays to NPZ bytes for streaming response.

    Args:
        arrays: Dictionary mapping array names to numpy arrays.

    Returns:
        Bytes representation of the NPZ file.
    """
    buffer = io.BytesIO()
    # Ensure a stable serialization order by sorting keys before saving.
    ordered_arrays = {key: arrays[key] for key in sorted(arrays.keys())}
    np.savez(buffer, **ordered_arrays)  # type: ignore[arg-type]  # numpy stubs incomplete for **kwargs
    buffer.seek(0)
    return buffer.read()


def compute_checksum(arrays: dict[str, np.ndarray]) -> str:
    """Compute SHA-256 checksum of arrays for integrity verification.

    The checksum is computed over the NPZ byte representation of the arrays,
    ensuring consistent results across different systems.

    Args:
        arrays: Dictionary mapping array names to numpy arrays.

    Returns:
        SHA-256 hex digest of the arrays.
    """
    data = arrays_to_bytes(arrays)
    return hashlib.sha256(data).hexdigest()
