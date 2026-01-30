"""Dataset endpoints for creating, listing, and retrieving datasets."""

import io
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from juniper_data.core.dataset_id import generate_dataset_id
from juniper_data.core.models import (
    CreateDatasetRequest,
    CreateDatasetResponse,
    DatasetMeta,
    PreviewData,
)
from juniper_data.storage import DatasetStore

from .generators import GENERATOR_REGISTRY

router = APIRouter(prefix="/datasets", tags=["datasets"])

_store: DatasetStore = None


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
            detail=f"Unknown generator '{request.generator}'. "
            f"Available: {list(GENERATOR_REGISTRY.keys())}",
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

    n_train = len(arrays["X_train"])
    n_test = len(arrays["X_test"])
    n_samples = n_train + n_test
    n_features = arrays["X_train"].shape[1] if n_train > 0 else 2
    n_classes = arrays["y_train"].shape[1] if n_train > 0 else params.n_spirals

    y_full = arrays.get("y_full", np.vstack([arrays["y_train"], arrays["y_test"]]))
    class_labels = np.argmax(y_full, axis=1)
    unique, counts = np.unique(class_labels, return_counts=True)
    class_distribution = {str(int(k)): int(v) for k, v in zip(unique, counts)}

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
        created_at=datetime.now(timezone.utc),
    )

    if request.persist:
        store.save(dataset_id, meta, arrays)

    return CreateDatasetResponse(
        dataset_id=dataset_id,
        generator=request.generator,
        meta=meta,
        artifact_url=f"/v1/datasets/{dataset_id}/artifact",
    )


@router.get("", response_model=List[str])
async def list_datasets(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    store: DatasetStore = Depends(get_store),
) -> List[str]:
    """List all dataset IDs.

    Args:
        limit: Maximum number of dataset IDs to return.
        offset: Number of dataset IDs to skip.
        store: Dataset storage backend.

    Returns:
        List of dataset IDs.
    """
    return store.list_datasets(limit=limit, offset=offset)


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
