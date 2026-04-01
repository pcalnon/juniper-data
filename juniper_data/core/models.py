"""Core Pydantic models for dataset metadata and API responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetMeta(BaseModel):
    """Dataset metadata (always small, JSON-safe)."""

    # Identity
    dataset_id: str
    generator: str
    generator_version: str

    # Generation Parameters
    params: dict[str, Any]

    # Shape Information
    n_samples: int
    n_features: int
    n_classes: int
    n_train: int
    n_test: int

    # Class Distribution (str keys for JSON compatibility)
    class_distribution: dict[str, int]

    # Artifacts
    artifact_formats: list[str] = Field(default_factory=lambda: ["npz"])

    # Timestamps
    created_at: datetime

    # Optional fields
    checksum: str | None = None

    # Lifecycle management (DATA-016)
    tags: list[str] = Field(default_factory=list)
    ttl_seconds: int | None = None
    expires_at: datetime | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0


class CreateDatasetRequest(BaseModel):
    """Request model for creating a new dataset."""

    generator: str
    params: dict[str, Any] = Field(default_factory=dict)
    persist: bool = True
    tags: list[str] = Field(default_factory=list)
    ttl_seconds: int | None = Field(default=None, ge=1, description="Time-to-live in seconds")


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
    params_schema: dict[str, Any] = Field(alias="schema")  # JSON schema for params


class PreviewData(BaseModel):
    """Preview subset of a dataset for visualization."""

    n_samples: int
    X_sample: list[list[float]]
    y_sample: list[list[float]]


class DatasetListFilter(BaseModel):
    """Filter criteria for listing datasets."""

    generator: str | None = None
    tags: list[str] | None = None
    tags_match: str = Field(default="any", pattern="^(any|all)$")
    created_after: datetime | None = None
    created_before: datetime | None = None
    min_samples: int | None = Field(default=None, ge=1)
    max_samples: int | None = Field(default=None, ge=1)
    include_expired: bool = False


class DatasetListResponse(BaseModel):
    """Response model for filtered dataset listing."""

    datasets: list[DatasetMeta]
    total: int
    limit: int
    offset: int


class BatchDeleteRequest(BaseModel):
    """Request model for batch delete operation."""

    dataset_ids: list[str] = Field(min_length=1, max_length=100)


class BatchDeleteResponse(BaseModel):
    """Response model for batch delete operation."""

    deleted: list[str]
    not_found: list[str]
    total_deleted: int


class UpdateTagsRequest(BaseModel):
    """Request model for updating dataset tags."""

    add_tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)


class BatchCreateItem(BaseModel):
    """Single item in a batch create request."""

    generator: str
    params: dict[str, Any] = Field(default_factory=dict)
    persist: bool = True
    tags: list[str] = Field(default_factory=list)
    ttl_seconds: int | None = Field(default=None, ge=1)


class BatchCreateRequest(BaseModel):
    """Request model for batch create operation."""

    datasets: list[BatchCreateItem] = Field(min_length=1, max_length=50)


class BatchCreateResultItem(BaseModel):
    """Result for a single item in a batch create response."""

    index: int
    dataset_id: str | None = None
    generator: str
    success: bool
    error: str | None = None
    artifact_url: str | None = None


class BatchCreateResponse(BaseModel):
    """Response model for batch create operation."""

    results: list[BatchCreateResultItem]
    total_created: int
    total_failed: int


class BatchUpdateTagsRequest(BaseModel):
    """Request model for batch tag update operation."""

    dataset_ids: list[str] = Field(min_length=1, max_length=100)
    add_tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)


class BatchUpdateTagsResponse(BaseModel):
    """Response model for batch tag update operation."""

    updated: list[str]
    not_found: list[str]
    total_updated: int


class BatchExportRequest(BaseModel):
    """Request model for batch export operation."""

    dataset_ids: list[str] = Field(min_length=1, max_length=50)


class DatasetStats(BaseModel):
    """Aggregate statistics about stored datasets."""

    total_datasets: int
    total_samples: int
    by_generator: dict[str, int]
    by_tag: dict[str, int]
    oldest_created_at: datetime | None = None
    newest_created_at: datetime | None = None
    expired_count: int = 0
