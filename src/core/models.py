"""Core Pydantic models for dataset metadata and API responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetMeta(BaseModel):
    """Dataset metadata (always small, JSON-safe)."""

    # Identity
    dataset_id: str
    generator: str
    generator_version: str

    # Generation Parameters
    params: Dict[str, Any]

    # Shape Information
    n_samples: int
    n_features: int
    n_classes: int
    n_train: int
    n_test: int

    # Class Distribution (str keys for JSON compatibility)
    class_distribution: Dict[str, int]

    # Artifacts
    artifact_formats: List[str] = Field(default_factory=lambda: ["npz"])

    # Timestamps
    created_at: datetime

    # Optional
    checksum: Optional[str] = None


class CreateDatasetRequest(BaseModel):
    """Request model for creating a new dataset."""

    generator: str
    params: Dict[str, Any] = Field(default_factory=dict)
    persist: bool = True


class CreateDatasetResponse(BaseModel):
    """Response model for dataset creation."""

    dataset_id: str
    generator: str
    meta: DatasetMeta
    artifact_url: str


class GeneratorInfo(BaseModel):
    """Information about an available generator."""

    name: str
    version: str
    description: str
    params_schema: Dict[str, Any] = Field(alias="schema")  # JSON schema for params


class PreviewData(BaseModel):
    """Preview subset of a dataset for visualization."""

    n_samples: int
    X_sample: List[List[float]]
    y_sample: List[List[float]]
