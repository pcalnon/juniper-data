"""Core module for Juniper Data."""

from juniper_data.core.artifacts import arrays_to_bytes, compute_checksum, load_npz, save_npz
from juniper_data.core.dataset_id import generate_dataset_id
from juniper_data.core.models import (
    CreateDatasetRequest,
    CreateDatasetResponse,
    DatasetMeta,
    GeneratorInfo,
    PreviewData,
)
from juniper_data.core.split import shuffle_and_split, shuffle_data, split_data

__all__ = [
    # Dataset ID
    "generate_dataset_id",
    # Split utilities
    "shuffle_and_split",
    "shuffle_data",
    "split_data",
    # Models
    "CreateDatasetRequest",
    "CreateDatasetResponse",
    "DatasetMeta",
    "GeneratorInfo",
    "PreviewData",
    # Artifacts
    "arrays_to_bytes",
    "compute_checksum",
    "load_npz",
    "save_npz",
]
