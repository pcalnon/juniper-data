"""Core Pydantic models for dataset metadata and API responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from juniper_data.core.constants import (
    BATCH_CREATE_MAX_ITEMS,
    BATCH_DELETE_MAX_ITEMS,
    BATCH_EXPORT_MAX_ITEMS,
    BATCH_MIN_ITEMS,
    BATCH_UPDATE_TAGS_MAX_ITEMS,
    CREATED_BY_MAX_LENGTH,
    DESCRIPTION_MAX_LENGTH,
)


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
    # Task type: "classification" | "regression". The classification-only fields
    # (n_classes, class_distribution) are optional so regression / 3-D time-series
    # artifacts need not fake a one-hot label (WS-1 / juniper-data#168).
    task_type: str = "classification"
    n_classes: int | None = None
    n_train: int
    # Rows in the in-loop validation partition of the three-way train/val/test
    # contract (design decision O-1). DEFAULTED, and it must stay defaulted:
    # every stored `.meta.json` predating the third partition is loaded with
    # `DatasetMeta(**meta_dict)` (storage/local_fs.py), so a required field with
    # no default would make every existing artifact unreadable. 0 is the honest
    # value for a two-partition artifact -- it has no validation rows.
    n_val: int = 0
    n_test: int

    # Class Distribution (str keys for JSON compatibility); None when not classification.
    class_distribution: dict[str, int] | None = None

    # Sequence / time-series metadata (WS-1 / juniper-data#168); False/None for
    # tabular artifacts. The route derives `sequence` + `lookback` from the X
    # rank; `time_unit` is generator-declared. Per-step Δt lives in the NPZ
    # (dt_ / observed_mask_).
    sequence: bool = False
    lookback: int | None = None
    time_unit: str | None = None

    # Advisory scaling descriptors (WS-4 / juniper-data#179 §A; Δt note §6.5).
    # A generator MAY report how its per-step `dt` / regression target should be
    # standardized; the NPZ keeps RAW values, so these are recommended-transform
    # metadata (the consumer normalizes at ingestion + denorms for metrics), NOT
    # applied transforms. JSON-safe dicts: `{"method": "identity"}` or
    # `{"method": "standardize", "mean": .., "std": .., "min": .., "max": ..}`;
    # `target_scaling` is keyed by target-array name (e.g. `{"y": <desc>}`).
    dt_scaling: dict[str, Any] | None = None
    target_scaling: dict[str, Any] | None = None

    # APD-DATA-018: PERMANENT record that this dataset is partial.
    #
    # None means complete. A dict means the source exceeded its byte cap and the
    # caller authorised a partial import; it carries `truncated`, `reason`,
    # `bytes_read`, `bytes_total`, `cap_bytes` and `records_imported`.
    #
    # This is metadata, not a warning: it is persisted with the artifact and
    # survives every later read, so a consumer that never saw the HTTP response
    # -- a trainer loading the NPZ months later -- still learns the data is a
    # prefix of its source. The HTTP response is transient; this is not.
    truncation: dict[str, Any] | None = None

    # PERMANENT record that something in this dataset is degraded or missing.
    #
    # None means clean. A dict means at least one of:
    #   `degraded`  -- a value was recovered from a WEAKER source than the primary
    #                  one (e.g. a period-average share count rather than a
    #                  point-in-time one, which makes market_cap a different
    #                  quantity for those symbols);
    #   `unrescued` -- no source produced a value, so those rows carry the fill.
    # plus `rows_affected` and the `policy` (accept / drop) that was applied.
    #
    # Separate from `truncation` on purpose: truncation says how much is MISSING,
    # this says what is WRONG with what is present. A consumer has to be able to
    # ask those independently.
    data_quality: dict[str, Any] | None = None

    # Artifacts
    artifact_formats: list[str] = Field(default_factory=lambda: ["npz"])

    # Timestamps
    created_at: datetime

    # Optional fields
    checksum: str | None = None

    # Versioning (CAN-DEF-005)
    dataset_name: str | None = None
    dataset_version: int | None = None
    parent_dataset_id: str | None = None
    description: str | None = None
    created_by: str | None = None

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
    name: str | None = Field(default=None, description="Logical dataset name for version tracking")
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_LENGTH)
    created_by: str | None = Field(default=None, max_length=CREATED_BY_MAX_LENGTH)
    parent_dataset_id: str | None = Field(default=None, description="ID of parent dataset for lineage")


class CreateDatasetResponse(BaseModel):
    """Response model for dataset creation."""

    dataset_id: str
    generator: str
    meta: DatasetMeta
    artifact_url: str


class GeneratorInfo(BaseModel):
    """Information about a registered generator."""

    name: str
    version: str
    description: str
    # D1 (I-5): False when an optional dependency this generator needs is missing
    # in the running deployment (e.g. mnist without HF ``datasets``). Defaults to
    # True so older payloads / constructors without the flag mean "available".
    available: bool = True
    # W-4: the generator's own curated install hint — the same string its guarded
    # ``ImportError`` carries, and therefore the same text the 501 on ``POST /v1/datasets``
    # returns for a DECLARED capability gap (ERR-08 / APD-DATA-004 keeps undeclared ones
    # behind a correlation id). Without it ``available: false`` says a generator cannot run
    # and nothing at all about what would fix that, so a preflight has nowhere to send an
    # operator: juniper-ml's experiment driver refuses an unavailable generator with "see
    # GET /v1/generators for the install hint" against a payload that carried none.
    # None for the generators that declare no optional dependency.
    install_hint: str | None = None
    params_schema: dict[str, Any] = Field(alias="schema")  # JSON schema for params


class PreviewData(BaseModel):
    """Preview subset of a dataset for visualization."""

    n_samples: int
    X_sample: list[list[float]]
    y_sample: list[list[float]]


class DatasetListResponse(BaseModel):
    """Response model for filtered dataset listing."""

    datasets: list[DatasetMeta]
    total: int
    limit: int
    offset: int
    # APD-DATA-011: position of the last returned row in the total order
    # ``(created_at DESC, dataset_id ASC)``. Pass it back as ``cursor`` to get the next
    # page without the skip/duplicate drift that re-slicing by ``offset`` suffers under
    # concurrent writes. Optional with a default so adding it does not break any existing
    # client; ``None`` on an empty page, since there is no position to name.
    next_cursor: str | None = None


class DatasetVersionListResponse(BaseModel):
    """Response for version listing endpoint."""

    dataset_name: str
    versions: list[DatasetMeta]
    total: int
    latest_version: int | None = None


class BatchDeleteRequest(BaseModel):
    """Request model for batch delete operation."""

    dataset_ids: list[str] = Field(min_length=BATCH_MIN_ITEMS, max_length=BATCH_DELETE_MAX_ITEMS)


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
    name: str | None = Field(default=None, description="Logical dataset name for version tracking")
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_LENGTH)
    created_by: str | None = Field(default=None, max_length=CREATED_BY_MAX_LENGTH)
    parent_dataset_id: str | None = Field(default=None, description="ID of parent dataset for lineage")


class BatchCreateRequest(BaseModel):
    """Request model for batch create operation."""

    datasets: list[BatchCreateItem] = Field(min_length=BATCH_MIN_ITEMS, max_length=BATCH_CREATE_MAX_ITEMS)


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

    dataset_ids: list[str] = Field(min_length=BATCH_MIN_ITEMS, max_length=BATCH_UPDATE_TAGS_MAX_ITEMS)
    add_tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)


class BatchUpdateTagsResponse(BaseModel):
    """Response model for batch tag update operation."""

    updated: list[str]
    not_found: list[str]
    total_updated: int


class BatchExportRequest(BaseModel):
    """Request model for batch export operation."""

    dataset_ids: list[str] = Field(min_length=BATCH_MIN_ITEMS, max_length=BATCH_EXPORT_MAX_ITEMS)


class DatasetStats(BaseModel):
    """Aggregate statistics about stored datasets."""

    total_datasets: int
    total_samples: int
    by_generator: dict[str, int]
    by_tag: dict[str, int]
    oldest_created_at: datetime | None = None
    newest_created_at: datetime | None = None
    expired_count: int = 0
