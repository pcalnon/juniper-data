"""Dataset endpoints for creating, listing, and retrieving datasets."""

import asyncio
import io
import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from starlette import status

logger = logging.getLogger(__name__)

from juniper_data.api.constants import (
    API_PREFIX,
    BATCH_EXPORT_MANIFEST_NAME,
    BINARY_MEDIA_TYPE,
    GENERATION_STATUS_ERROR,
    GENERATION_STATUS_SUCCESS,
    POST_CACHE_HIT,
    POST_CACHE_MISS,
)
from juniper_data.api.observability import record_dataset_generation, record_dataset_post
from juniper_data.core.artifacts import compute_checksum
from juniper_data.core.constants import TAGS_MATCH_DEFAULT, TAGS_MATCH_PATTERN
from juniper_data.core.dataset_id import generate_dataset_id
from juniper_data.core.limits import IncompleteDataError, InputTooLargeError
from juniper_data.core.meta import compute_shape_meta, derive_sequence_meta, pop_data_quality_meta, pop_scaling_meta, pop_truncation_meta
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
from juniper_data.storage.base import encode_cursor
from juniper_data.storage.constants import JSON_INDENT_DEFAULT

from .generators import GENERATOR_REGISTRY

# from typing import List, Optional


router = APIRouter(prefix="/datasets", tags=["datasets"])

_store: DatasetStore | None = None


def get_store() -> DatasetStore:
    """Dependency to get the dataset store."""
    if _store is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Storage not initialized")
    return _store


def set_store(store: DatasetStore) -> None:
    """Set the dataset store (called during app startup)."""
    global _store
    _store = store


@router.post("", operation_id="create_dataset", response_model=CreateDatasetResponse, status_code=status.HTTP_201_CREATED)
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
        HTTPException: 400 if generator not found or parameters invalid;
            501 if the generator's optional dependencies are missing in this
            deployment (D1 / I-5 — the detail carries the install hint).
    """
    if request.generator not in GENERATOR_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown generator '{request.generator}'. Available: {list(GENERATOR_REGISTRY.keys())}",
        )

    generator_info = GENERATOR_REGISTRY[request.generator]
    generator_class = generator_info["generator"]
    params_class = generator_info["params_class"]
    version = generator_info["version"]

    # APD-DATA-014 -- the 400-vs-422 rule, stated rather than inferred.
    #
    # Both of this endpoint's rejection paths are "the caller sent something
    # wrong", and which status you get used to fall out of *where* validation
    # happened rather than out of a decision. The rule is:
    #
    #   422  the request violates the DECLARED schema, and FastAPI rejects it at
    #        the boundary before this function runs (``params`` not a mapping,
    #        ``ttl_seconds=0``, a missing required field).
    #   400  the request is schema-valid but SEMANTICALLY wrong for the generator
    #        it names -- an unknown generator above, or params this generator
    #        rejects here. ``params`` is typed ``dict[str, Any]``, so the boundary
    #        cannot check it; only the resolved ``params_class`` can.
    #
    # ``except ValueError`` alone is the whole catch: ``pydantic.ValidationError``
    # subclasses ``ValueError`` (verified against pydantic v2 -- its MRO is
    # ValidationError -> ValueError -> Exception). Naming both in the tuple, as
    # this did, implied a distinction between them that does not exist and was
    # precisely the "the split falls out of the MRO, not design" finding.
    try:
        params = params_class(**request.params)
    except ValueError as e:
        record_dataset_post(
            generator=request.generator,
            status=GENERATION_STATUS_ERROR,
            cache=POST_CACHE_MISS,
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid parameters: {e}") from e

    dataset_id = generate_dataset_id(
        generator=request.generator,
        version=version,
        params=params.model_dump(),
    )

    existing_meta = await asyncio.to_thread(store.get_meta, dataset_id)
    if existing_meta is not None:
        # METRICS-MON R4.5: cache hits short-circuit the generator path,
        # so ``record_dataset_generation`` is not called — but the POST
        # still happened and operators need to see the request volume
        # (deterministic re-POSTs, cascor retraining, etc. would
        # otherwise be invisible). ``status="success"`` because returning
        # the cached meta is a successful POST outcome from the caller's
        # perspective.
        record_dataset_post(
            generator=request.generator,
            status=GENERATION_STATUS_SUCCESS,
            cache=POST_CACHE_HIT,
        )
        return CreateDatasetResponse(
            dataset_id=dataset_id,
            generator=request.generator,
            meta=existing_meta,
            artifact_url=f"{API_PREFIX}/datasets/{dataset_id}/artifact",
        )

    # SEC-04 / JD-PERF-01 / CONC-04: move the potentially CPU-bound
    # generator call off the async event-loop thread so concurrent HTTP
    # requests are not blocked while a dataset is being synthesized.
    # Generator classes are stateless today (verified in Phase 1D survey),
    # so running them in asyncio's default executor is safe.
    # BUG-JD-07: record dataset generation duration + count to Prometheus
    gen_start = time.monotonic()
    try:
        # arrays = generator_class.generate(params)
        arrays = await asyncio.to_thread(generator_class.generate, params)
    except (InputTooLargeError, IncompleteDataError) as e:
        record_dataset_generation(generator=request.generator, status=GENERATION_STATUS_ERROR, duration=time.monotonic() - gen_start)
        record_dataset_post(
            generator=request.generator,
            status=GENERATION_STATUS_ERROR,
            cache=POST_CACHE_MISS,
        )
        # APD-DATA-018: the caller's input exceeds its cap, or part of the dataset
        # could not be produced correctly, and neither the
        # request nor the deployment authorised a partial import. This is a
        # caller-fixable condition with a deterministic remedy, so it must not
        # reach the bare re-raise below and surface as a 500.
        #
        # 422 is chosen because it is ALREADY on this API's surface (the
        # app-level RequestValidationError handler answers 422), so this adds no
        # status code. That matters: `APD-DATA-022` -- the row that would
        # document a new code in `responses={}` -- is parked as an owner
        # decision, and a fix for one row must not force work inside a parked
        # one.
        #
        # ERR-08: the message is safe to echo. It is curated in
        # `InputTooLargeError`, interpolates only the caller's own file_path
        # plus two integers this service computed, and names the exact remedy.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        ) from e
    except ImportError as e:
        record_dataset_generation(generator=request.generator, status=GENERATION_STATUS_ERROR, duration=time.monotonic() - gen_start)
        record_dataset_post(
            generator=request.generator,
            status=GENERATION_STATUS_ERROR,
            cache=POST_CACHE_MISS,
        )
        # D1 (I-5): a generator raising ImportError means an optional dependency
        # is missing in this deployment — a deterministic capability gap, not an
        # internal error. Surface it as 501 Not Implemented with the generator's
        # actionable install hint (e.g. "pip install datasets") instead of letting
        # the bare re-raise below mask it as a generic 500. 503 is deliberately
        # avoided: it invites client retries and health-tooling misreads for a
        # condition that will not clear on its own.
        # ERR-08 (APD-DATA-004): only a DECLARED capability gap may echo the
        # exception text. Every generator that guards an optional dependency
        # raises a curated message carrying the install hint, and that is what
        # D1 exists to surface. But this ``except`` also catches ImportErrors
        # raised *beneath* the guard -- a broken native extension, a partial
        # install, a failed lazy import inside the third-party package -- and
        # those messages routinely carry filesystem paths. Echoing them would
        # defeat the ERR-08 control that batch-create applies to the sibling
        # ``except Exception`` branch, because batch-create copies ``e.detail``
        # verbatim; the single-create path returns it to the caller directly.
        #
        # ``is_available()`` is the discriminator: the three optional-dependency
        # generators check it first, so when the guard fires it reports False.
        # A generator that reports itself available -- or does not declare the
        # method at all, as the thirteen numpy-only generators do not -- has
        # failed for a reason the caller must not be shown.
        availability_check = getattr(generator_class, "is_available", None)
        dependency_declared_missing = callable(availability_check) and not availability_check()

        if dependency_declared_missing:
            detail = f"Generator '{request.generator}' is not available in this deployment: {e}"
        else:
            error_id = uuid.uuid4().hex[:12]
            # The generator name is deliberately NOT interpolated here. It is
            # request-supplied, and this module's ERR-08 logging idiom (see the
            # batch-create handler below) keeps caller-controlled strings out of
            # log records entirely rather than sanitizing them per call site --
            # juniper-data has no ``_sanitize_for_log`` helper to lean on.
            # ``logger.exception`` emits the traceback, which names the failing
            # generator module and class, and ``error_id`` is the join key back
            # to the 501 the caller received.
            logger.exception("A generator reported itself available but raised ImportError [error_id=%s]", error_id)
            detail = f"Generator '{request.generator}' is not available in this deployment (ref: {error_id})"

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=detail,
        ) from e
    except Exception:
        record_dataset_generation(generator=request.generator, status=GENERATION_STATUS_ERROR, duration=time.monotonic() - gen_start)
        # METRICS-MON R4.5: also bump the POST counter on the error
        # branch so the post_total / generations_total ratio surfaces
        # generator-error rate (vs cache-hit-rate).
        record_dataset_post(
            generator=request.generator,
            status=GENERATION_STATUS_ERROR,
            cache=POST_CACHE_MISS,
        )
        raise
    record_dataset_generation(generator=request.generator, status=GENERATION_STATUS_SUCCESS, duration=time.monotonic() - gen_start)
    record_dataset_post(
        generator=request.generator,
        status=GENERATION_STATUS_SUCCESS,
        cache=POST_CACHE_MISS,
    )

    # WS-4 (#179 §A): pull the advisory dt/target scaling descriptors out of the
    # reserved "scaling" channel key BEFORE checksum + NPZ persist, so the stored
    # arrays stay array-only. None for generators that do not report scaling.
    scaling_meta = pop_scaling_meta(arrays)

    # APD-DATA-018: same reserved-channel discipline as scaling above -- pull the
    # truncation descriptor out BEFORE checksum + NPZ persist so the stored
    # arrays stay array-only. None for every generator that did not truncate,
    # which is all of them except a csv_import run that was explicitly
    # authorised to produce a partial dataset.
    truncation_meta = pop_truncation_meta(arrays)

    # Same reserved-channel discipline: pull the data-quality descriptor out
    # before checksum + NPZ persist. None for a clean dataset.
    data_quality_meta = pop_data_quality_meta(arrays)

    checksum = compute_checksum(arrays)

    # WS-1 (#168): dispatch shape/class metadata on the generator's declared
    # task_type so regression / 3-D sequence artifacts traverse the route
    # without a forced one-hot/argmax. n_features is the trailing axis, so 2-D
    # tabular and 3-D sequence X are both handled.
    task_type = generator_info.get("task_type", "classification")
    shape_meta = compute_shape_meta(arrays, task_type)

    # WS-1 PR3: sequence-ness + lookback are derived from the X rank; time_unit
    # is generator-declared in the registry (like task_type). Per-step Δt lives
    # in the NPZ (dt_ / observed_mask_); dynamic scaling stats defer to WS-4.
    seq_meta = derive_sequence_meta(arrays, generator_info.get("time_unit"))

    now = datetime.now(UTC)
    expires_at = None
    if request.ttl_seconds is not None:
        expires_at = now + timedelta(seconds=request.ttl_seconds)

    meta = DatasetMeta(
        dataset_id=dataset_id,
        generator=request.generator,
        generator_version=version,
        params=params.model_dump(),
        n_samples=shape_meta["n_samples"],
        n_features=shape_meta["n_features"],
        task_type=task_type,
        n_classes=shape_meta["n_classes"],
        n_train=shape_meta["n_train"],
        n_val=shape_meta["n_val"],
        n_test=shape_meta["n_test"],
        class_distribution=shape_meta["class_distribution"],
        sequence=seq_meta["sequence"],
        lookback=seq_meta["lookback"],
        time_unit=seq_meta["time_unit"],
        dt_scaling=scaling_meta["dt_scaling"],
        target_scaling=scaling_meta["target_scaling"],
        truncation=truncation_meta,
        data_quality=data_quality_meta,
        artifact_formats=["npz"],
        created_at=now,
        checksum=checksum,
        dataset_name=request.name,
        dataset_version=None,  # Assigned atomically by save_versioned()
        description=request.description,
        created_by=request.created_by,
        parent_dataset_id=request.parent_dataset_id,
        tags=request.tags,
        ttl_seconds=request.ttl_seconds,
        expires_at=expires_at,
    )

    if request.persist:
        # save_versioned() atomically allocates the version number under a lock
        # to prevent concurrent requests from receiving the same version.
        await asyncio.to_thread(store.save_versioned, dataset_id, meta, arrays)
    elif request.name is not None:
        # Non-persisted: preview the next version (no race since no write)
        meta.dataset_version = await asyncio.to_thread(store.next_version_number, request.name)

    return CreateDatasetResponse(
        dataset_id=dataset_id,
        generator=request.generator,
        meta=meta,
        artifact_url=f"{API_PREFIX}/datasets/{dataset_id}/artifact",
    )


@router.get("", operation_id="list_datasets", response_model=list[str])
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
    return await asyncio.to_thread(store.list_datasets, limit=limit, offset=offset)


@router.get("/filter", operation_id="filter_datasets", response_model=DatasetListResponse)
async def filter_datasets(
    generator: str | None = Query(default=None, description="Filter by generator name"),
    tags: str | None = Query(default=None, description="Comma-separated list of tags to filter by"),
    tags_match: str = Query(default=TAGS_MATCH_DEFAULT, pattern=TAGS_MATCH_PATTERN, description="Tag matching mode: 'any' (OR) or 'all' (AND)"),
    created_after: datetime | None = Query(default=None, description="Filter by creation date (after)"),
    created_before: datetime | None = Query(default=None, description="Filter by creation date (before)"),
    min_samples: int | None = Query(default=None, ge=1, description="Minimum number of samples"),
    max_samples: int | None = Query(default=None, ge=1, description="Maximum number of samples"),
    include_expired: bool = Query(default=False, description="Include expired datasets"),
    dataset_name: str | None = Query(default=None, description="Filter by logical dataset name"),
    dataset_version: int | None = Query(default=None, description="Filter by dataset version number"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    cursor: str | None = Query(default=None, description="Opaque cursor from a previous response's next_cursor. Stable under concurrent writes; mutually exclusive with offset."),
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
        cursor: Opaque cursor naming the last row of a previous page.
        store: Dataset storage backend.

    Returns:
        Filtered list of dataset metadata with pagination info.

    Raises:
        HTTPException: 400 if ``cursor`` is malformed, or if ``cursor`` and a non-zero
            ``offset`` are combined.

    APD-DATA-011: two pagination modes, and ``next_cursor`` is always returned so a
    caller can move from one to the other without a round trip.

    * ``offset`` re-slices the current result set, so a row inserted or deleted before
      the offset shifts every later page -- reproduced, an insert between two fetches
      returned the same dataset on both. Kept because it is the existing contract and
      is fine for a one-shot page.
    * ``cursor`` names a position in the total order and asks for what strictly follows,
      which nothing inserted or deleted ahead of it can shift.

    Combining them is rejected rather than silently resolved: passing both means the
    caller believes one of them is doing something it is not.
    """
    if cursor is not None and offset:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pass either 'cursor' or 'offset', not both: a cursor already names where the page starts.",
        )

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        datasets, total = await asyncio.to_thread(
            store.filter_datasets,
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
            cursor=cursor,
        )
    except ValueError as exc:
        # decode_cursor rejects a token it did not issue. Schema-valid string,
        # semantically wrong -> 400, per the rule stated in create_dataset.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DatasetListResponse(
        datasets=datasets,
        total=total,
        limit=limit,
        offset=offset,
        # Always emitted, in both modes: it is simply the last returned row's position,
        # so an offset-mode caller can switch to stable pagination at any point. ``None``
        # on an empty page because there is no position to name.
        next_cursor=encode_cursor(datasets[-1]) if datasets else None,
    )


@router.get("/stats", operation_id="get_dataset_stats", response_model=DatasetStats)
async def get_dataset_stats(
    store: DatasetStore = Depends(get_store),
) -> DatasetStats:
    """Get aggregate statistics about stored datasets.

    Args:
        store: Dataset storage backend.

    Returns:
        Dataset statistics.
    """
    stats = await asyncio.to_thread(store.get_stats)
    return DatasetStats(**stats)  # type: ignore[arg-type]


@router.post("/batch-delete", operation_id="batch_delete_datasets", response_model=BatchDeleteResponse)
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
    deleted, not_found = await asyncio.to_thread(store.batch_delete, request.dataset_ids)

    return BatchDeleteResponse(
        deleted=deleted,
        not_found=not_found,
        total_deleted=len(deleted),
    )


@router.post("/batch-create", operation_id="batch_create_datasets", response_model=BatchCreateResponse, status_code=status.HTTP_201_CREATED)
async def batch_create_datasets(
    request: BatchCreateRequest,
    response: Response,
    store: DatasetStore = Depends(get_store),
) -> BatchCreateResponse:
    """Create multiple datasets in a single request.

    Each item is processed independently; failures in one item do not
    affect others. Results include per-item success/failure status.

    Status code: ``201 Created`` when at least one dataset was created,
    ``200 OK`` when none was. The decorator's ``201`` is only a default —
    a batch in which every item failed created no resource, so reporting
    "Created" tells a caller that checks only the status line the exact
    opposite of what happened. ``200`` is not "success" here either; it
    means the batch was processed and the per-item ``results`` are the
    authority, which is the contract the three sibling batch routes
    already use (they declare no ``status_code`` at all).

    Deliberately not ``207 Multi-Status``: that would be a new response
    semantic for every existing client to learn, and per-item statuses
    are a redesign of this endpoint's contract rather than a correction
    of this one false claim.

    Args:
        request: Batch create request with list of dataset specifications.
        response: Injected so the status can depend on the outcome.
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
        except Exception:
            # ERR-08: do not surface raw exception strings — they can leak
            # filesystem paths or internal type details. Log the full
            # traceback server-side with a short correlation ID and return
            # the ID so support can look up the incident.
            error_id = uuid.uuid4().hex[:12]
            logger.exception("Batch create item %d failed [error_id=%s]", idx, error_id)
            results.append(
                BatchCreateResultItem(
                    index=idx,
                    generator=item.generator,
                    success=False,
                    error=f"Dataset creation failed (ref: {error_id})",
                )
            )
            total_failed += 1

    if total_created == 0:
        response.status_code = status.HTTP_200_OK

    return BatchCreateResponse(
        results=results,
        total_created=total_created,
        total_failed=total_failed,
    )


@router.patch("/batch-tags", operation_id="batch_update_tags", response_model=BatchUpdateTagsResponse)
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

    # BUG-JD-10 (2026-05-05 audit): ``store.get_meta`` and
    # ``store.update_meta`` are synchronous filesystem I/O. Offload
    # each call to a thread so the FastAPI event loop stays
    # responsive during large batches — same pattern as
    # ``generator_class.generate`` above.
    for dataset_id in request.dataset_ids:
        meta = await asyncio.to_thread(store.get_meta, dataset_id)
        if meta is None:
            not_found.append(dataset_id)
            continue

        current_tags = set(meta.tags)
        current_tags.update(request.add_tags)
        current_tags -= set(request.remove_tags)
        meta.tags = sorted(current_tags)
        await asyncio.to_thread(store.update_meta, dataset_id, meta)
        updated.append(dataset_id)

    return BatchUpdateTagsResponse(
        updated=updated,
        not_found=not_found,
        total_updated=len(updated),
    )


@router.post("/batch-export", operation_id="batch_export_datasets")
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

    # BUG-JD-01: Stream the ZIP instead of accumulating the entire archive in
    # memory. Previously, `io.BytesIO()` held every selected NPZ + zip metadata
    # until the response was ready, which is an OOM risk once callers batch many
    # large datasets. We now drive `zipfile.ZipFile` through a chunk-buffer that
    # the generator drains after each entry, so peak memory stays proportional
    # to a single NPZ rather than the sum of the export.
    #
    # Pre-check existence up front so we can still return 404 when *none* of
    # the requested datasets exist without first committing to a response body.
    # `exists()` is a cheap metadata check on every backend, but it is still
    # filesystem (or network) I/O — gather() each call off the event loop so a
    # large batch doesn't stall concurrent requests.
    exists_flags = await asyncio.gather(*(asyncio.to_thread(store.exists, dsid) for dsid in request.dataset_ids))
    present_ids = [dsid for dsid, exists in zip(request.dataset_ids, exists_flags) if exists]
    if not present_ids:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="None of the requested datasets were found")

    # APD-DATA-010: a dataset the caller asked for can be absent for two reasons -- it
    # did not exist at this pre-check, or it disappeared between here and its artifact
    # read. Both used to drop it from the archive with no signal at all: HTTP 200, a ZIP
    # with fewer members than ids requested, and nothing saying which.
    missing: dict[str, str] = {dsid: "not_found" for dsid in request.dataset_ids if dsid not in set(present_ids)}

    class _ChunkBuffer:
        """File-like sink that accumulates bytes for the streaming generator to drain.

        ``zipfile.ZipFile`` needs an object with ``write``/``flush``/``close``
        methods. We record writes in a list and hand them off to the outer
        generator between entries so the response body can be emitted without
        buffering the whole archive.
        """

        def __init__(self) -> None:
            self._chunks: list[bytes] = []

        def write(self, data: bytes) -> int:
            if data:
                self._chunks.append(bytes(data))
            return len(data)

        def drain(self) -> bytes:
            if not self._chunks:
                return b""
            joined = b"".join(self._chunks)
            self._chunks.clear()
            return joined

        def flush(self) -> None:  # pragma: no cover - required by ZipFile protocol
            return None

    exported: list[str] = []

    def _stream_zip():
        buf = _ChunkBuffer()
        # ZIP_STORED is required for streaming-friendly archives: with ZIP_DEFLATED
        # the zipfile module would need to seek back to patch the local-file-header
        # with the final size, which is not possible once chunks have been yielded.
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for dataset_id in present_ids:
                artifact_bytes = store.get_artifact_bytes(dataset_id)
                if artifact_bytes is None:
                    # Raced with a concurrent deletion. Recorded rather than skipped
                    # quietly (APD-DATA-010) -- the caller cannot otherwise tell this
                    # export apart from one where it never asked for the dataset.
                    missing[dataset_id] = "vanished_during_export"
                    continue
                exported.append(dataset_id)
                zf.writestr(f"{dataset_id}.npz", artifact_bytes)
                chunk = buf.drain()
                if chunk:
                    yield chunk

            # APD-DATA-010: the archive carries its own account of what is in it, but
            # ONLY when something is absent -- a complete export stays byte-identical to
            # what this endpoint has always produced, so no existing consumer changes.
            #
            # It has to live inside the ZIP. The response is a streamed 200: the status
            # line and headers are already on the wire before the first artifact is read,
            # so neither can report a dataset that vanishes mid-stream. The manifest is
            # written after the loop, when the full picture is known.
            if missing:
                # Counts only. ``dataset_ids`` is caller-controlled, and this module's
                # ERR-08 idiom keeps caller strings out of log records (see the
                # batch-create handler); the ids themselves go to the caller, who sent
                # them, not into the log.
                logger.warning(
                    "Batch export omitted %d of %d requested datasets (%d not found, %d vanished mid-export)",
                    len(missing),
                    len(request.dataset_ids),
                    sum(1 for reason in missing.values() if reason == "not_found"),
                    sum(1 for reason in missing.values() if reason == "vanished_during_export"),
                )
                zf.writestr(
                    BATCH_EXPORT_MANIFEST_NAME,
                    json.dumps(
                        {
                            "requested": list(request.dataset_ids),
                            "exported": exported,
                            "missing": missing,
                        },
                        indent=JSON_INDENT_DEFAULT,
                    ),
                )
                chunk = buf.drain()
                if chunk:
                    yield chunk
        # ``with`` exit writes the central directory + EOCD to the buffer.
        trailing = buf.drain()
        if trailing:
            yield trailing

    return StreamingResponse(
        _stream_zip(),
        media_type=BINARY_MEDIA_TYPE,
        headers={"Content-Disposition": "attachment; filename=datasets.zip"},
    )


@router.post("/cleanup-expired", operation_id="cleanup_expired_datasets", response_model=list[str])
async def cleanup_expired_datasets(
    store: DatasetStore = Depends(get_store),
) -> list[str]:
    """Delete all expired datasets.

    Args:
        store: Dataset storage backend.

    Returns:
        List of deleted dataset IDs.
    """
    return await asyncio.to_thread(store.delete_expired)


@router.get("/versions", operation_id="list_dataset_versions", response_model=DatasetVersionListResponse)
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
    versions = await asyncio.to_thread(store.list_versions, name)
    latest = versions[-1].dataset_version if versions else None
    return DatasetVersionListResponse(
        dataset_name=name,
        versions=versions,
        total=len(versions),
        latest_version=latest,
    )


@router.get("/latest", operation_id="get_latest_version", response_model=DatasetMeta)
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
    meta = await asyncio.to_thread(store.get_latest_version, name)
    if meta is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No versions found for dataset '{name}'")
    return meta


@router.get("/{dataset_id}", operation_id="get_dataset_metadata", response_model=DatasetMeta)
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
    meta = await asyncio.to_thread(store.get_meta, dataset_id)
    if meta is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_id}' not found")
    # BUG-JD-08: record access asynchronously to avoid blocking I/O on read paths
    asyncio.get_event_loop().call_soon(lambda: store.record_access(dataset_id))
    return meta


@router.get("/{dataset_id}/artifact", operation_id="download_artifact")
async def download_artifact(
    dataset_id: str,
    store: DatasetStore = Depends(get_store),
) -> StreamingResponse:
    """Download dataset artifact as NPZ file.

    The body is produced incrementally via
    :meth:`~juniper_data.storage.base.DatasetStore.open_artifact_stream`, so peak
    memory is bounded by the chunk size rather than by artifact size — on a backend
    that overrides the default. Backends that inherit the base implementation still
    read the artifact whole, so the memory bound is a property of the *store*, not
    of this route (defect-register ``APD-DATA-016``).

    Args:
        dataset_id: Unique dataset identifier.
        store: Dataset storage backend.

    Returns:
        Streaming response with NPZ file contents.

    Raises:
        HTTPException: 404 if dataset not found.
    """
    # APD-DATA-016: stream the artifact rather than materialise it. The previous
    # form read the whole NPZ into memory and wrapped it in ``io.BytesIO``, which
    # bounds the SOCKET BUFFER but not process memory -- peak RSS was the full
    # artifact, once per concurrent download, while the name "streaming" invited
    # the opposite assumption. ``open_artifact_stream`` yields it in chunks; the
    # base-class default still falls back to a whole read, so backends that cannot
    # do better are unchanged and only this route's memory profile improves.
    #
    # ``to_thread`` wraps the OPEN (an existence check plus a file handle on
    # LocalFS) because that is the blocking part which must complete before the
    # response exists -- the 404 decision cannot be deferred into the generator, or
    # the route would already have committed to a 200. Subsequent chunk reads
    # happen as the ASGI server pulls them.
    artifact_stream = await asyncio.to_thread(store.open_artifact_stream, dataset_id)
    if artifact_stream is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_id}' not found")

    # BUG-JD-08: record access asynchronously to avoid blocking I/O on read paths
    asyncio.get_event_loop().call_soon(lambda: store.record_access(dataset_id))

    return StreamingResponse(
        artifact_stream,
        media_type=BINARY_MEDIA_TYPE,
        headers={"Content-Disposition": f"attachment; filename={dataset_id}.npz"},
    )


@router.get("/{dataset_id}/preview", operation_id="preview_dataset", response_model=PreviewData)
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
    artifact_bytes = await asyncio.to_thread(store.get_artifact_bytes, dataset_id)
    if artifact_bytes is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_id}' not found")

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


@router.delete("/{dataset_id}", operation_id="delete_dataset", status_code=status.HTTP_204_NO_CONTENT)
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
    deleted = await asyncio.to_thread(store.delete, dataset_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_id}' not found")


@router.patch("/{dataset_id}/tags", operation_id="update_dataset_tags", response_model=DatasetMeta)
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
    # APD-DATA-006: the read-modify-write must happen inside ONE hop, under the
    # store's ``_version_lock``. Doing it here across two ``asyncio.to_thread``
    # calls left a window in which ``record_access`` -- which fires on every
    # metadata read and every artifact download, and which rewrites the whole
    # metadata document under that lock -- could write back a pre-edit snapshot
    # and silently discard the tag change.
    meta = await asyncio.to_thread(store.update_tags, dataset_id, request.add_tags, request.remove_tags)
    if meta is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_id}' not found")
    return meta
