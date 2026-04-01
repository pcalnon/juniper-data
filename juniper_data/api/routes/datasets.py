"""Dataset endpoints for creating, listing, and retrieving datasets."""

import io
from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from juniper_data.core.artifacts import compute_checksum
from juniper_data.core.dataset_id import generate_dataset_id
from juniper_data.core.models import (
    BatchCreateRequest,
    BatchCreateResponse,
    BatchCreateResultItem,
    BatchDeleteRequest,
    BatchDeleteResponse,
    BatchExportRequest,
    BatchUpdateTagsRequest,
    BatchUpdateTagsResponse,
    CreateDatasetRequest,
    CreateDatasetResponse,
    DatasetListResponse,
    DatasetMeta,
    DatasetStats,
    DatasetVersionListResponse,
    PreviewData,
    UpdateTagsRequest,
)
from juniper_data.storage import DatasetStore

from .generators import GENERATOR_REGISTRY

# from typing import List, Optional


router = APIRouter(prefix="/datasets", tags=["datasets"])

_store: DatasetStore | None = None


def get_store() -> DatasetStore:
    """Dependency to get the dataset store."""
    if _store is None:
        raise HTTPException(status_code=500, detail="Storage not initialized")
    return _store


def set_store(store: DatasetStore) -> None:
    """Set the dataset store (called during app startup)."""
    global _store
    _store = store


@router.post("", response_model=CreateDatasetResponse, status_code=201)
async def create_dataset(
    request: CreateDatasetRequest,
    store: DatasetStore = Depends(get_store),
) -> CreateDatasetResponse:
    """Create or generate a new dataset.

    If a dataset with the same parameters already exists, returns the existing
    metadata without regeneration (caching behavior).

    Args:
        request: Dataset creation request with generator name and parameters.
        store: Dataset storage backend.

    Returns:
        Dataset metadata and artifact URL.

    Raises:
        HTTPException: 400 if generator not found or parameters invalid.
    """
    if request.generator not in GENERATOR_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown generator '{request.generator}'. Available: {list(GENERATOR_REGISTRY.keys())}",
        )

    generator_info = GENERATOR_REGISTRY[request.generator]
    generator_class = generator_info["generator"]
    params_class = generator_info["params_class"]
    version = generator_info["version"]

    try:
        params = params_class(**request.params)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {e}")

    dataset_id = generate_dataset_id(
        generator=request.generator,
        version=version,
        params=params.model_dump(),
    )

    existing_meta = store.get_meta(dataset_id)
    if existing_meta is not None:
        return CreateDatasetResponse(
            dataset_id=dataset_id,
            generator=request.generator,
            meta=existing_meta,
            artifact_url=f"/v1/datasets/{dataset_id}/artifact",
        )

    arrays = generator_class.generate(params)

    checksum = compute_checksum(arrays)

    n_train = len(arrays["X_train"])
    n_test = len(arrays["X_test"])
    n_samples = n_train + n_test
    n_features = arrays["X_train"].shape[1] if n_train > 0 else 2
    n_classes = arrays["y_train"].shape[1] if n_train > 0 else params.n_spirals

    y_full = arrays.get("y_full", np.vstack([arrays["y_train"], arrays["y_test"]]))
    class_labels = np.argmax(y_full, axis=1)
    unique, counts = np.unique(class_labels, return_counts=True)
    class_distribution = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    now = datetime.now(UTC)
    expires_at = None
    if request.ttl_seconds is not None:
        expires_at = now + timedelta(seconds=request.ttl_seconds)

    # Auto-increment version for named datasets
    version_num = store.next_version_number(request.name) if request.name is not None else None

    meta = DatasetMeta(
        dataset_id=dataset_id,
        generator=request.generator,
        generator_version=version,
        params=params.model_dump(),
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_train=n_train,
        n_test=n_test,
        class_distribution=class_distribution,
        artifact_formats=["npz"],
        created_at=now,
        checksum=checksum,
        dataset_name=request.name,
        dataset_version=version_num,
        description=request.description,
        created_by=request.created_by,
        parent_dataset_id=request.parent_dataset_id,
        tags=request.tags,
        ttl_seconds=request.ttl_seconds,
        expires_at=expires_at,
    )

    if request.persist:
        store.save(dataset_id, meta, arrays)

    return CreateDatasetResponse(
        dataset_id=dataset_id,
        generator=request.generator,
        meta=meta,
        artifact_url=f"/v1/datasets/{dataset_id}/artifact",
    )


@router.get("", response_model=list[str])
async def list_datasets(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    store: DatasetStore = Depends(get_store),
) -> list[str]:
    """List all dataset IDs.

    Args:
        limit: Maximum number of dataset IDs to return.
        offset: Number of dataset IDs to skip.
        store: Dataset storage backend.

    Returns:
        List of dataset IDs.
    """
    return store.list_datasets(limit=limit, offset=offset)


@router.get("/filter", response_model=DatasetListResponse)
async def filter_datasets(
    generator: str | None = Query(default=None, description="Filter by generator name"),
    tags: str | None = Query(default=None, description="Comma-separated list of tags to filter by"),
    tags_match: str = Query(default="any", pattern="^(any|all)$", description="Tag matching mode: 'any' (OR) or 'all' (AND)"),
    created_after: datetime | None = Query(default=None, description="Filter by creation date (after)"),
    created_before: datetime | None = Query(default=None, description="Filter by creation date (before)"),
    min_samples: int | None = Query(default=None, ge=1, description="Minimum number of samples"),
    max_samples: int | None = Query(default=None, ge=1, description="Maximum number of samples"),
    include_expired: bool = Query(default=False, description="Include expired datasets"),
    dataset_name: str | None = Query(default=None, description="Filter by logical dataset name"),
    dataset_version: int | None = Query(default=None, description="Filter by dataset version number"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    store: DatasetStore = Depends(get_store),
) -> DatasetListResponse:
    """Filter datasets by various criteria.

    Args:
        generator: Filter by generator name.
        tags: Comma-separated list of tags.
        tags_match: Tag matching mode: 'any' (OR) or 'all' (AND).
        created_after: Filter by creation date (after).
        created_before: Filter by creation date (before).
        min_samples: Minimum number of samples.
        max_samples: Maximum number of samples.
        include_expired: Include expired datasets.
        dataset_name: Filter by logical dataset name.
        dataset_version: Filter by dataset version number.
        limit: Maximum number of results.
        offset: Number of results to skip.
        store: Dataset storage backend.

    Returns:
        Filtered list of dataset metadata with pagination info.
    """
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    datasets, total = store.filter_datasets(
        generator=generator,
        tags=tag_list,
        tags_match=tags_match,
        created_after=created_after,
        created_before=created_before,
        min_samples=min_samples,
        max_samples=max_samples,
        include_expired=include_expired,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        limit=limit,
        offset=offset,
    )

    return DatasetListResponse(
        datasets=datasets,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/stats", response_model=DatasetStats)
async def get_dataset_stats(
    store: DatasetStore = Depends(get_store),
) -> DatasetStats:
    """Get aggregate statistics about stored datasets.

    Args:
        store: Dataset storage backend.

    Returns:
        Dataset statistics.
    """
    stats = store.get_stats()
    return DatasetStats(**stats)  # type: ignore[arg-type]


@router.post("/batch-delete", response_model=BatchDeleteResponse)
async def batch_delete_datasets(
    request: BatchDeleteRequest,
    store: DatasetStore = Depends(get_store),
) -> BatchDeleteResponse:
    """Delete multiple datasets in a single request.

    Args:
        request: Batch delete request with list of dataset IDs.
        store: Dataset storage backend.

    Returns:
        Batch delete response with deleted and not found IDs.
    """
    deleted, not_found = store.batch_delete(request.dataset_ids)

    return BatchDeleteResponse(
        deleted=deleted,
        not_found=not_found,
        total_deleted=len(deleted),
    )


@router.post("/batch-create", response_model=BatchCreateResponse, status_code=201)
async def batch_create_datasets(
    request: BatchCreateRequest,
    store: DatasetStore = Depends(get_store),
) -> BatchCreateResponse:
    """Create multiple datasets in a single request.

    Each item is processed independently; failures in one item do not
    affect others. Results include per-item success/failure status.

    Args:
        request: Batch create request with list of dataset specifications.
        store: Dataset storage backend.

    Returns:
        Batch create response with per-item results.
    """
    results: list[BatchCreateResultItem] = []
    total_created = 0
    total_failed = 0

    for idx, item in enumerate(request.datasets):
        try:
            # Reuse the single-create logic
            create_req = CreateDatasetRequest(
                generator=item.generator,
                params=item.params,
                persist=item.persist,
                tags=item.tags,
                ttl_seconds=item.ttl_seconds,
                name=item.name,
                description=item.description,
                created_by=item.created_by,
                parent_dataset_id=item.parent_dataset_id,
            )
            resp = await create_dataset(create_req, store)
            results.append(
                BatchCreateResultItem(
                    index=idx,
                    dataset_id=resp.dataset_id,
                    generator=item.generator,
                    success=True,
                    artifact_url=resp.artifact_url,
                )
            )
            total_created += 1
        except HTTPException as e:
            results.append(
                BatchCreateResultItem(
                    index=idx,
                    generator=item.generator,
                    success=False,
                    error=e.detail,
                )
            )
            total_failed += 1
        except Exception as e:
            results.append(
                BatchCreateResultItem(
                    index=idx,
                    generator=item.generator,
                    success=False,
                    error=str(e),
                )
            )
            total_failed += 1

    return BatchCreateResponse(
        results=results,
        total_created=total_created,
        total_failed=total_failed,
    )


@router.patch("/batch-tags", response_model=BatchUpdateTagsResponse)
async def batch_update_tags(
    request: BatchUpdateTagsRequest,
    store: DatasetStore = Depends(get_store),
) -> BatchUpdateTagsResponse:
    """Add or remove tags from multiple datasets.

    Args:
        request: Batch tag update request with dataset IDs and tag changes.
        store: Dataset storage backend.

    Returns:
        Batch tag update response with updated and not found IDs.
    """
    updated: list[str] = []
    not_found: list[str] = []

    for dataset_id in request.dataset_ids:
        meta = store.get_meta(dataset_id)
        if meta is None:
            not_found.append(dataset_id)
            continue

        current_tags = set(meta.tags)
        current_tags.update(request.add_tags)
        current_tags -= set(request.remove_tags)
        meta.tags = sorted(current_tags)
        store.update_meta(dataset_id, meta)
        updated.append(dataset_id)

    return BatchUpdateTagsResponse(
        updated=updated,
        not_found=not_found,
        total_updated=len(updated),
    )


@router.post("/batch-export")
async def batch_export_datasets(
    request: BatchExportRequest,
    store: DatasetStore = Depends(get_store),
) -> StreamingResponse:
    """Export multiple datasets as a ZIP archive of NPZ files.

    Args:
        request: Batch export request with list of dataset IDs.
        store: Dataset storage backend.

    Returns:
        Streaming response with ZIP file containing NPZ artifacts.

    Raises:
        HTTPException: 404 if none of the requested datasets exist.
    """
    import zipfile

    buffer = io.BytesIO()
    found_count = 0

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for dataset_id in request.dataset_ids:
            artifact_bytes = store.get_artifact_bytes(dataset_id)
            if artifact_bytes is not None:
                zf.writestr(f"{dataset_id}.npz", artifact_bytes)
                found_count += 1

    if found_count == 0:
        raise HTTPException(status_code=404, detail="None of the requested datasets were found")

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=datasets.zip"},
    )


@router.post("/cleanup-expired", response_model=list[str])
async def cleanup_expired_datasets(
    store: DatasetStore = Depends(get_store),
) -> list[str]:
    """Delete all expired datasets.

    Args:
        store: Dataset storage backend.

    Returns:
        List of deleted dataset IDs.
    """
    return store.delete_expired()


@router.get("/versions", response_model=DatasetVersionListResponse)
async def list_dataset_versions(
    name: str = Query(description="Dataset name to list versions for"),
    store: DatasetStore = Depends(get_store),
) -> DatasetVersionListResponse:
    """List all versions of a named dataset.

    Args:
        name: Logical dataset name to list versions for.
        store: Dataset storage backend.

    Returns:
        Version list response with all versions sorted by version number.
    """
    versions = store.list_versions(name)
    latest = versions[-1].dataset_version if versions else None
    return DatasetVersionListResponse(
        dataset_name=name,
        versions=versions,
        total=len(versions),
        latest_version=latest,
    )


@router.get("/latest", response_model=DatasetMeta)
async def get_latest_version(
    name: str = Query(description="Dataset name to get latest version of"),
    store: DatasetStore = Depends(get_store),
) -> DatasetMeta:
    """Get the latest version of a named dataset.

    Args:
        name: Logical dataset name to get the latest version of.
        store: Dataset storage backend.

    Returns:
        Dataset metadata for the latest version.

    Raises:
        HTTPException: 404 if no versions found for the given name.
    """
    meta = store.get_latest_version(name)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"No versions found for dataset '{name}'")
    return meta


@router.get("/{dataset_id}", response_model=DatasetMeta)
async def get_dataset_metadata(
    dataset_id: str,
    store: DatasetStore = Depends(get_store),
) -> DatasetMeta:
    """Get metadata for a specific dataset.

    Args:
        dataset_id: Unique dataset identifier.
        store: Dataset storage backend.

    Returns:
        Dataset metadata.

    Raises:
        HTTPException: 404 if dataset not found.
    """
    meta = store.get_meta(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return meta


@router.get("/{dataset_id}/artifact")
async def download_artifact(
    dataset_id: str,
    store: DatasetStore = Depends(get_store),
) -> StreamingResponse:
    """Download dataset artifact as NPZ file.

    Args:
        dataset_id: Unique dataset identifier.
        store: Dataset storage backend.

    Returns:
        Streaming response with NPZ file contents.

    Raises:
        HTTPException: 404 if dataset not found.
    """
    artifact_bytes = store.get_artifact_bytes(dataset_id)
    if artifact_bytes is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    return StreamingResponse(
        io.BytesIO(artifact_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={dataset_id}.npz"},
    )


@router.get("/{dataset_id}/preview", response_model=PreviewData)
async def preview_dataset(
    dataset_id: str,
    n: int = Query(default=100, ge=1, le=1000),
    store: DatasetStore = Depends(get_store),
) -> PreviewData:
    """Preview first N samples of a dataset as JSON.

    Args:
        dataset_id: Unique dataset identifier.
        n: Number of samples to preview (default 100, max 1000).
        store: Dataset storage backend.

    Returns:
        Preview data with sample features and labels.

    Raises:
        HTTPException: 404 if dataset not found.
    """
    artifact_bytes = store.get_artifact_bytes(dataset_id)
    if artifact_bytes is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    with np.load(io.BytesIO(artifact_bytes)) as data:
        if "X_full" in data and "y_full" in data:
            X = data["X_full"]
            y = data["y_full"]
        else:
            X = np.vstack([data["X_train"], data["X_test"]])
            y = np.vstack([data["y_train"], data["y_test"]])

    n_samples = min(n, len(X))

    return PreviewData(
        n_samples=n_samples,
        X_sample=X[:n_samples].tolist(),
        y_sample=y[:n_samples].tolist(),
    )


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: str,
    store: DatasetStore = Depends(get_store),
) -> None:
    """Delete a dataset.

    Args:
        dataset_id: Unique dataset identifier.
        store: Dataset storage backend.

    Raises:
        HTTPException: 404 if dataset not found.
    """
    deleted = store.delete(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")


@router.patch("/{dataset_id}/tags", response_model=DatasetMeta)
async def update_dataset_tags(
    dataset_id: str,
    request: UpdateTagsRequest,
    store: DatasetStore = Depends(get_store),
) -> DatasetMeta:
    """Add or remove tags from a dataset.

    Args:
        dataset_id: Unique dataset identifier.
        request: Tags to add and/or remove.
        store: Dataset storage backend.

    Returns:
        Updated dataset metadata.

    Raises:
        HTTPException: 404 if dataset not found.
    """
    meta = store.get_meta(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    current_tags = set(meta.tags)
    current_tags.update(request.add_tags)
    current_tags -= set(request.remove_tags)
    meta.tags = sorted(current_tags)

    store.update_meta(dataset_id, meta)
    return meta
