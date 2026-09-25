"""Dataset endpoints for creating, listing, and retrieving datasets."""

import asyncio
import io
import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter
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
from juniper_data.api.http_cache import CACHE_CONTROL_NO_STORE, CACHE_CONTROL_REVALIDATE, PrerenderedJSONResponse, body_etag, combine_field_lines, read_precondition_status, weak_etag, write_preconditions_hold
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
    DatasetAccessStats,
    DatasetListResponse,
    DatasetMeta,
    DatasetStats,
    DatasetVersionListResponse,
    PreviewData,
    PublicDatasetMeta,
    UpdateTagsRequest,
)
from juniper_data.storage import DatasetStore
from juniper_data.storage.base import InvalidDatasetIdError, PreconditionFailedError, StorageContainmentError, decode_cursor, encode_cursor
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


# The access recorder: ONE thread, of its own, that runs ``record_access`` for the reads.
# ``record_access`` takes the store's ``_version_lock`` and cross-process lock, and a create,
# a tag edit or a delete holds those -- in this process or, for the file lock, in another. It
# used to run on the event loop (``call_soon``), so the loop, and with it every request and
# ``/v1/health``, waited for whichever writer held them: seconds, for a large create
# (round-1 validation of the #438 fix-forward, lane B M-1). It must not run on the loop's
# default executor either: that pool serves the routes' own store I/O, and accesses queued
# behind a held lock would take its threads. So a read hands its access over and never waits.
# One worker keeps the recordings in the order the reads were answered; a failure is logged by
# type only, because its message can carry the caller's id (ERR-08).
_ACCESS_RECORDER_THREAD_NAME = "juniper-data-access-recorder"
_access_recorder: ThreadPoolExecutor | None = None
_access_recorder_guard = threading.Lock()


def _record_access(store: DatasetStore, dataset_id: str) -> None:
    try:
        store.record_access(dataset_id)
    except Exception as exc:  # noqa: BLE001 -- an access that cannot be recorded must not fail anything else
        logger.warning("Could not record an access (%s)", type(exc).__name__)


def record_access_later(store: DatasetStore, dataset_id: str) -> None:
    """Hand one access to the recorder thread. Never waits for it, or for any lock."""
    global _access_recorder
    with _access_recorder_guard:
        if _access_recorder is None:
            _access_recorder = ThreadPoolExecutor(max_workers=1, thread_name_prefix=_ACCESS_RECORDER_THREAD_NAME)
        _access_recorder.submit(_record_access, store, dataset_id)


def drain_access_recorder(timeout: float | None = None) -> None:
    """Wait until every access handed over so far is recorded.

    Blocking: call it from a worker thread, never on the event loop. ``GET /{dataset_id}/access``
    does, so the counters it serves include every access answered before it.
    """
    with _access_recorder_guard:
        recorder = _access_recorder
    if recorder is not None:
        recorder.submit(lambda: None).result(timeout)


def shutdown_access_recorder() -> None:
    """Record what is queued, then stop the recorder thread (the app's lifespan calls this on shutdown)."""
    global _access_recorder
    with _access_recorder_guard:
        recorder, _access_recorder = _access_recorder, None
    if recorder is not None:
        recorder.shutdown(wait=True)


# The one serializer for a single dataset's metadata representation. Typed with the
# PUBLIC model, so a stored ``DatasetMeta`` handed to it loses its access counters
# (APD-DATA-032; see ``PublicDatasetMeta``).
_PUBLIC_META = TypeAdapter(PublicDatasetMeta)
_ACCESS_STATS = TypeAdapter(DatasetAccessStats)

# The validator and caching headers each route sends, declared so the published OpenAPI
# says what the wire carries (juniper_data/api/http_cache.py has the semantics).
_H_STRONG_ETAG = {"description": "Strong entity-tag: the SHA-256 of the exact response body.", "schema": {"type": "string"}}
_H_WEAK_ETAG = {"description": 'Weak entity-tag, W/"<checksum>": same arrays, not necessarily the same bytes. Absent when the dataset has no stored checksum.', "schema": {"type": "string"}}
_H_CACHE_CONTROL = {"description": "private, no-cache", "schema": {"type": "string"}}
_H_CONTENT_LOCATION = {"description": "The canonical /v1/datasets/<dataset_id> of the representation returned.", "schema": {"type": "string"}}
_R_304 = {"description": "Not Modified: `If-None-Match` names the current representation. No body."}
_R_412_READ = {"description": "Precondition Failed: `If-Match` does not name the current representation."}
_R_412_WRITE = {"description": "Precondition Failed: `If-Match` does not name, or `If-None-Match` names, the current representation, or either field is malformed. Nothing was changed."}

_CONDITIONAL_READ = {status.HTTP_200_OK: {"headers": {"ETag": _H_STRONG_ETAG, "Cache-Control": _H_CACHE_CONTROL}}, status.HTTP_304_NOT_MODIFIED: _R_304, status.HTTP_412_PRECONDITION_FAILED: _R_412_READ}
_CONDITIONAL_READ_LATEST = {status.HTTP_200_OK: {"headers": {"ETag": _H_STRONG_ETAG, "Cache-Control": _H_CACHE_CONTROL, "Content-Location": _H_CONTENT_LOCATION}}, status.HTTP_304_NOT_MODIFIED: _R_304, status.HTTP_412_PRECONDITION_FAILED: _R_412_READ}
_CONDITIONAL_READ_ARTIFACT = {status.HTTP_200_OK: {"headers": {"ETag": _H_WEAK_ETAG, "Cache-Control": _H_CACHE_CONTROL}}, status.HTTP_304_NOT_MODIFIED: _R_304, status.HTTP_412_PRECONDITION_FAILED: _R_412_READ}
_CONDITIONAL_WRITE = {status.HTTP_200_OK: {"headers": {"ETag": _H_STRONG_ETAG, "Cache-Control": _H_CACHE_CONTROL, "Content-Location": _H_CONTENT_LOCATION}}, status.HTTP_412_PRECONDITION_FAILED: _R_412_WRITE}

# ``list[str]``: a list-valued header may arrive on several lines, and RFC 9110 §5.3 makes
# those one list. Typed ``str`` FastAPI would keep only the first line.
_IF_NONE_MATCH = Header(
    default=None,
    description="Entity-tag(s) of a representation the client already holds. On a read, a match answers 304 with no body (APD-DATA-017).",
)
_IF_MATCH = Header(
    default=None,
    description="Entity-tag(s) the current representation must carry (strong comparison); otherwise 412 (APD-DATA-017).",
)


def _precondition_failed() -> HTTPException:
    return HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Precondition Failed: the representation does not satisfy the request's If-Match / If-None-Match")


def _canonical_path(dataset_id: str) -> str:
    """The canonical URI of one dataset's metadata, in the form ``artifact_url`` uses."""
    return f"{API_PREFIX}/datasets/{dataset_id}"


def _close_artifact_stream(stream: Iterator[bytes]) -> None:
    """Release an opened artifact stream that will not be sent.

    LocalFS returns a generator that opens its file only once iterated, and closing it
    before then is a no-op that also stops it from ever opening one; another store's
    iterator may hold a resource from the start. The base default, a tuple iterator, has
    nothing to close.
    """
    close = getattr(stream, "close", None)
    if callable(close):
        close()


def _metadata_response(meta: DatasetMeta, if_match: str | None, if_none_match: str | None, *, content_location: str | None = None) -> Response:
    """Render one dataset's metadata with its strong ``ETag``: a 200, a 304, or a 412.

    Preconditions follow RFC 9110 §13.2.2 (``read_precondition_status``); callers have
    already established that the dataset exists, so a precondition is never evaluated
    against a target that would have been a 404 (§13.2.1).

    The body is rendered HERE rather than left to ``response_model``, so the ETag is
    the hash of the exact bytes sent (APD-DATA-017) -- correct by construction, whatever
    those bytes are. They are rendered the way FastAPI renders a ``response_model``
    with the default response class: its ``dump_json`` fast path, the annotated type's
    JSON bytes straight from pydantic-core, ``by_alias``. That is what keeps the wire
    format unchanged. The obvious alternative, ``JSONResponse`` over a JSON-mode dump,
    is NOT byte-identical -- ``json.dumps`` writes ``1e-07`` where pydantic-core writes
    ``1e-7`` -- and FastAPI's choice between the two has moved across versions, so
    ``test_bytes_match_fastapi_rendering_of_the_public_model`` pins the equality
    against a real ``response_model`` route instead of trusting this paragraph.
    """
    body = _PUBLIC_META.dump_json(meta, by_alias=True)
    headers = {"ETag": body_etag(body), "Cache-Control": CACHE_CONTROL_REVALIDATE}
    if content_location is not None:
        headers["Content-Location"] = content_location
    outcome = read_precondition_status(if_match, if_none_match, headers["ETag"])
    if outcome == status.HTTP_412_PRECONDITION_FAILED:
        raise _precondition_failed()
    if outcome == status.HTTP_304_NOT_MODIFIED:
        # RFC 9110 §15.4.5: a 304 carries the validator and caching fields the 200 would
        # have, and no body.
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    return PrerenderedJSONResponse(content=body, headers=headers)


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

    # Bind the DEPLOYMENT policy onto the params before hashing. `model_dump` fills
    # Field defaults, so an omitted `max_bytes` hashes as the schema default (128 MiB)
    # no matter what the deployment ceiling is -- and a cache entry written under a
    # loose ceiling is then served verbatim to a request made under a tighter one.
    # Lowering the bound would have no effect on anything already cached.
    # Generators without the hook are unaffected.
    binder = getattr(generator_class, "bind_deployment_defaults", None)
    if callable(binder):
        params = binder(params)

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
        # save_versioned() re-checks existence and saves under the store's locks, allocating the
        # version number there too. The check above ran before generating, which can take
        # seconds: a create of the same id, or a conditional PATCH of it, may have landed since.
        # Then nothing is written, and the dataset that is there is the one returned -- as on
        # the cache hit above.
        meta = await asyncio.to_thread(store.save_versioned, dataset_id, meta, arrays)
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

    if cursor is not None:
        # Decoded HERE, before the store is touched, so that the cursor's own error keeps its
        # 400 and its detail. This used to wrap the whole store call, and every ValueError the
        # store raised -- a stored metadata file that would not parse, or one the store refused
        # to follow out of its root -- came back as a 400 whose detail was the exception's
        # text, naming a dataset id. Now the app's handler answers those: a file that leads
        # out of the root is a 500; one that does not parse is still a 400, with a generic
        # detail -- a known issue.
        try:
            decode_cursor(cursor)
        except ValueError as exc:
            # decode_cursor rejects a token it did not issue. Schema-valid string,
            # semantically wrong -> 400, per the rule stated in create_dataset.
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

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

    Each dataset's edit is applied through the store's ``update_tags``, under the same locks as
    ``PATCH /{dataset_id}/tags``, so a batch edit is ordered before or after a conditional tag
    edit of the same dataset, never inside it. ``If-Match`` / ``If-None-Match`` are not
    evaluated here.

    Args:
        request: Batch tag update request with dataset IDs and tag changes.
        store: Dataset storage backend.

    Returns:
        Batch tag update response with updated and not found IDs.
    """
    updated: list[str] = []
    not_found: list[str] = []

    # One locked read-modify-write per dataset. Reading with ``get_meta`` and writing with
    # ``update_meta`` in two unlocked hops, as this did, let the edit land between a
    # conditional PATCH's passed check and its write, which then erased it. BUG-JD-10
    # (2026-05-05 audit): the store call is synchronous I/O, so it runs off the event loop --
    # same pattern as ``generator_class.generate`` above.
    for dataset_id in request.dataset_ids:
        meta = await asyncio.to_thread(store.update_tags, dataset_id, request.add_tags, request.remove_tags)
        if meta is None:
            not_found.append(dataset_id)
            continue
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


@router.get("/latest", operation_id="get_latest_version", response_model=PublicDatasetMeta, responses=_CONDITIONAL_READ_LATEST)
async def get_latest_version(
    name: str = Query(description="Dataset name to get latest version of"),
    store: DatasetStore = Depends(get_store),
    if_match: list[str] | None = _IF_MATCH,
    if_none_match: list[str] | None = _IF_NONE_MATCH,
) -> Response:
    """Get the latest version of a named dataset.

    The body is the same representation ``GET /{dataset_id}`` serves for that version,
    with the same ``ETag``, and ``Content-Location`` names that canonical URI
    (APD-DATA-029): two URIs return this representation, and without the header a
    cache or client cannot tell they are one resource. A 307 to the canonical URI was
    ruled out -- it costs every caller a round trip.

    Args:
        name: Logical dataset name to get the latest version of.
        store: Dataset storage backend.
        if_match: Entity-tag(s) the representation must carry; otherwise 412.
        if_none_match: Entity-tag(s) the client already holds; a match answers 304.

    Returns:
        Dataset metadata for the latest version, or an empty 304. Unlike
        ``GET /{dataset_id}``, this route has never recorded an access, and a 304 here
        does not either.

    Raises:
        HTTPException: 404 if no versions found for the given name.
    """
    meta = await asyncio.to_thread(store.get_latest_version, name)
    if meta is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No versions found for dataset '{name}'")
    return _metadata_response(meta, combine_field_lines(if_match), combine_field_lines(if_none_match), content_location=_canonical_path(meta.dataset_id))


@router.get("/{dataset_id}", operation_id="get_dataset_metadata", response_model=PublicDatasetMeta, responses=_CONDITIONAL_READ)
async def get_dataset_metadata(
    dataset_id: str,
    store: DatasetStore = Depends(get_store),
    if_match: list[str] | None = _IF_MATCH,
    if_none_match: list[str] | None = _IF_NONE_MATCH,
) -> Response:
    """Get metadata for a specific dataset.

    Carries a strong ``ETag`` -- the SHA-256 of the exact body -- and answers a
    matching ``If-None-Match`` with an empty 304 (APD-DATA-017). The access counters
    are not in the body (APD-DATA-032): see ``GET /{dataset_id}/access``.

    Args:
        dataset_id: Unique dataset identifier.
        store: Dataset storage backend.
        if_match: Entity-tag(s) the representation must carry; otherwise 412.
        if_none_match: Entity-tag(s) the client already holds; a match answers 304.

    Returns:
        Dataset metadata, or an empty 304.

    Raises:
        HTTPException: 404 if dataset not found.
    """
    meta = await asyncio.to_thread(store.get_meta, dataset_id)
    if meta is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_id}' not found")
    response = _metadata_response(meta, combine_field_lines(if_match), combine_field_lines(if_none_match))
    # BUG-JD-08: the access is recorded off the request path, on the access recorder's own
    # thread -- never on the event loop, because ``record_access`` waits for the store's locks.
    # A 304 is recorded too: revalidating a held copy is a read of the dataset. A 412 raised
    # above is not -- nothing was read.
    record_access_later(store, dataset_id)
    return response


@router.get("/{dataset_id}/access", operation_id="get_dataset_access_stats", response_model=DatasetAccessStats)
async def get_dataset_access_stats(
    dataset_id: str,
    store: DatasetStore = Depends(get_store),
) -> Response:
    """Get the access counters of a dataset -- where they live since APD-DATA-032.

    ``access_count`` and ``last_accessed_at`` change on every ``GET /{dataset_id}`` and
    every artifact download that answers 200 or 304 (a 412 reads nothing and records
    nothing; nor does a download whose stored metadata leads out of the storage root,
    which has no readable document to count in), which is exactly why they could not stay
    in the metadata body: no strong ``ETag`` can describe a representation that differs on
    every read. They are still maintained as before; this is where they are read. Reading
    them is NOT itself recorded as an access -- the count would then describe its own
    observation.

    A read hands its access to a thread of its own and does not wait for it to be recorded.
    This waits for every access handed over before it, so the counters it serves include
    every read answered before this request.

    Args:
        dataset_id: Unique dataset identifier.
        store: Dataset storage backend.

    Returns:
        The dataset's access counters, marked ``no-store``.

    Raises:
        HTTPException: 404 if dataset not found.
    """
    await asyncio.to_thread(drain_access_recorder)
    meta = await asyncio.to_thread(store.get_meta, dataset_id)
    if meta is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_id}' not found")
    stats = DatasetAccessStats(dataset_id=meta.dataset_id, access_count=meta.access_count, last_accessed_at=meta.last_accessed_at)
    return PrerenderedJSONResponse(content=_ACCESS_STATS.dump_json(stats, by_alias=True), headers={"Cache-Control": CACHE_CONTROL_NO_STORE})


@router.get("/{dataset_id}/artifact", operation_id="download_artifact", responses=_CONDITIONAL_READ_ARTIFACT)
async def download_artifact(
    dataset_id: str,
    store: DatasetStore = Depends(get_store),
    if_match: list[str] | None = _IF_MATCH,
    if_none_match: list[str] | None = _IF_NONE_MATCH,
) -> Response:
    """Download dataset artifact as NPZ file.

    The body is produced incrementally via
    :meth:`~juniper_data.storage.base.DatasetStore.open_artifact_stream`, so peak
    memory is bounded by the chunk size rather than by artifact size — on a backend
    that overrides the default. Backends that inherit the base implementation still
    read the artifact whole, so the memory bound is a property of the *store*, not
    of this route (defect-register ``APD-DATA-016``).

    The response carries a WEAK ``ETag``, ``W/"<checksum>"``: the stored ``checksum``, which
    is NOT a hash of the transferred bytes -- ``juniper_data/api/http_cache.py`` sets out what
    it does and does not cover. A matching ``If-None-Match`` is answered with an empty 304
    (APD-DATA-017). ``If-Match`` compares strongly, so on this route only ``*`` satisfies it.

    Args:
        dataset_id: Unique dataset identifier.
        store: Dataset storage backend.
        if_match: Entity-tag(s) the representation must carry; otherwise 412.
        if_none_match: Entity-tag(s) the client already holds; a match answers 304.

    Returns:
        Streaming response with NPZ file contents, or an empty 304.

    Raises:
        HTTPException: 404 if dataset not found; 412 if ``If-Match`` does not hold.
    """
    # APD-DATA-017: the validator comes from the metadata, read BEFORE the artifact is
    # opened, so a matching If-None-Match costs a metadata read and an existence check --
    # no artifact I/O. A dataset whose metadata carries no checksum gets no ETag.
    #
    # Unreadable metadata must not cost the download: before validators existed this
    # route never read the metadata, so a corrupt ``.meta.json`` still served the
    # artifact. It still does, without a validator -- and so does a metadata file the store
    # refuses to follow out of its root (``StorageContainmentError``), a storage fault. A
    # malformed ``dataset_id`` is not unreadable metadata but the caller's error, so it is
    # re-raised to the 400 every other route gives it (the app's ``ValueError`` handler). The
    # warning names the exception TYPE only: a message or traceback can carry the
    # caller-controlled id, and ERR-08 keeps caller strings out of log records.
    metadata_readable = True
    metadata_refused = False
    try:
        meta = await asyncio.to_thread(store.get_meta, dataset_id)
    except InvalidDatasetIdError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any other metadata read failure degrades to "no validator", never to a failed download
        logger.warning("Artifact download: dataset metadata unreadable (%s); serving without an ETag", type(exc).__name__)
        meta = None
        metadata_readable = False
        metadata_refused = isinstance(exc, StorageContainmentError)
    etag = weak_etag(meta.checksum) if meta is not None and meta.checksum else None
    headers = {"Cache-Control": CACHE_CONTROL_REVALIDATE}
    if etag is not None:
        headers["ETag"] = etag
    conditional = bool(if_match or if_none_match)
    evaluated = False
    # Unreadable metadata skips the existence gate: ``exists`` consults the same metadata (on
    # LocalFS it raises for a metadata file the store refuses to follow), so the artifact alone
    # decides whether a representation exists, and the preconditions are judged after the open
    # below, as for an orphan -- against no validator.
    if conditional and metadata_readable:
        # RFC 9110 §13.2.1: preconditions are evaluated only for a target the request would
        # otherwise have served. ``exists`` is the store's own contract (LocalFS: metadata
        # AND artifact on disk) and reads no artifact bytes. A target it denies falls
        # through to the open below, which 404s when there is no artifact -- never a 304 for
        # data that is gone. A store whose ``exists`` checks the metadata alone (the
        # in-memory store does) can still answer 304 for a metadata-without-artifact
        # corruption.
        if await asyncio.to_thread(store.exists, dataset_id):
            evaluated = True
            outcome = read_precondition_status(combine_field_lines(if_match), combine_field_lines(if_none_match), etag)
            if outcome == status.HTTP_412_PRECONDITION_FAILED:
                raise _precondition_failed()
            if outcome == status.HTTP_304_NOT_MODIFIED:
                # Revalidating a held copy is a read of the dataset, as on the metadata route.
                record_access_later(store, dataset_id)
                return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)

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

    if conditional and not evaluated:
        # An ORPHANED artifact: ``exists`` denied the target (on LocalFS, the metadata is
        # gone), or the metadata could not be read and the gate was skipped, yet the artifact
        # opened, so a representation exists and is about to be sent.
        # RFC 9110 §13.1.1 still binds -- when If-Match is false the method MUST NOT be
        # performed -- so the preconditions are evaluated here, against no validator: ``*``
        # matches (a representation exists) and no listed tag can. A stream that will not be
        # sent is closed, releasing what it holds.
        outcome = read_precondition_status(combine_field_lines(if_match), combine_field_lines(if_none_match), None)
        if outcome is not None:
            _close_artifact_stream(artifact_stream)
            if outcome == status.HTTP_412_PRECONDITION_FAILED:
                raise _precondition_failed()
            if not metadata_refused:
                record_access_later(store, dataset_id)
            return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)

    # BUG-JD-08: the access is recorded on the access recorder's own thread, never on the
    # event loop. Not when the store REFUSED the metadata (``StorageContainmentError``, a file
    # that leads out of the root): ``record_access`` reads the same metadata, so it would only
    # fail again. Any other failure may be transient -- a descriptor limit, say -- so the access
    # is still handed over, and a failure there is logged by type only (ERR-08).
    if not metadata_refused:
        record_access_later(store, dataset_id)

    return StreamingResponse(
        artifact_stream,
        media_type=BINARY_MEDIA_TYPE,
        headers={**headers, "Content-Disposition": f"attachment; filename={dataset_id}.npz"},
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
        # The TRAINING partition, per design §9.5.4 item 3. This read ``X_full`` with a
        # ``train + test`` fallback -- which was risk R-5 surviving in the preview path,
        # silently skipping the validation rows whenever the artifact had them.
        #
        # Serving train rather than a reassembled whole is the design's stated choice and
        # matches juniper-data-client's fake, so the two cannot disagree. It does shift the
        # semantics for a SEQUENCE artifact, where train is the chronologically earliest
        # block rather than a random sample.
        X = data["X_train"]
        y = data["y_train"]

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

    The delete takes the same locks as ``PATCH /{dataset_id}/tags``, so it is ordered before or
    after a conditional tag edit of the same dataset, never between its check and its write.
    ``If-Match`` is not evaluated here.

    Args:
        dataset_id: Unique dataset identifier.
        store: Dataset storage backend.

    Raises:
        HTTPException: 404 if dataset not found.
    """
    deleted = await asyncio.to_thread(store.delete_under_lock, dataset_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_id}' not found")


@router.patch("/{dataset_id}/tags", operation_id="update_dataset_tags", response_model=PublicDatasetMeta, responses=_CONDITIONAL_WRITE)
async def update_dataset_tags(
    dataset_id: str,
    request: UpdateTagsRequest,
    store: DatasetStore = Depends(get_store),
    if_match: list[str] | None = _IF_MATCH,
    if_none_match: list[str] | None = _IF_NONE_MATCH,
) -> Response:
    """Add or remove tags from a dataset.

    The body is the updated metadata representation and carries its new ``ETag``
    (APD-DATA-017), so a client can hold the edited copy without a second read. With
    ``If-Match`` it is an optimistic-concurrency write: the precondition is evaluated
    against the CURRENT representation inside the store's locks, and every route that
    creates, edits or deletes a dataset takes the same locks, so the check cannot pass and
    then lose a race to another writer. That holds across processes on one host with the
    default (LocalFS) store, and within one process with the Redis and Postgres stores. It
    does not hold for ``CachedDatasetStore``, whose cache fills write without either lock;
    the service wires LocalFS. A dataset removed from outside the locks before the write is a
    404, not a 200 with an ``ETag`` -- unless, on LocalFS, the removal lands between the
    write's own existence check and its rename. ``If-None-Match`` naming the current
    representation (``*`` included) is a 412 here, not a 304 -- RFC 9110 §13.1.2 for a
    method other than GET -- and so is either field when it is malformed: a write fails
    closed on a precondition it cannot read. No other write route evaluates preconditions:
    an ``If-Match`` sent to ``DELETE /{dataset_id}`` or ``PATCH /batch-tags`` is ignored.

    Args:
        dataset_id: Unique dataset identifier.
        request: Tags to add and/or remove.
        store: Dataset storage backend.
        if_match: Entity-tag(s) the current representation must carry; otherwise 412.
        if_none_match: Entity-tag(s) the current representation must NOT carry; otherwise 412.

    Returns:
        Updated dataset metadata.

    Raises:
        HTTPException: 404 if dataset not found; 412 if a precondition fails (nothing written).
    """
    # APD-DATA-006: the read-modify-write must happen inside ONE hop, under the
    # store's ``_version_lock``. Doing it here across two ``asyncio.to_thread``
    # calls left a window in which ``record_access`` -- which fires on every
    # ``GET /{dataset_id}`` and every artifact download (not on a list, ``/filter``,
    # ``/latest``, ``/versions``, a preview or ``/access``), and which rewrites the whole
    # metadata document under that lock -- could write back a pre-edit snapshot
    # and silently discard the tag change.
    if_match_field, if_none_match_field = combine_field_lines(if_match), combine_field_lines(if_none_match)

    def precondition(current: DatasetMeta) -> bool:
        # The ETag the current representation would carry, computed exactly as the read
        # routes compute it -- and evaluated inside the store's lock, by update_tags.
        return write_preconditions_hold(if_match_field, if_none_match_field, body_etag(_PUBLIC_META.dump_json(current, by_alias=True)))

    conditional = if_match_field is not None or if_none_match_field is not None
    try:
        meta = await asyncio.to_thread(store.update_tags, dataset_id, request.add_tags, request.remove_tags, precondition if conditional else None)
    except PreconditionFailedError:
        raise _precondition_failed() from None
    if meta is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_id}' not found")
    # The request target (``.../tags``) has no GET of its own, so Content-Location names the
    # resource this body and its ETag describe (RFC 9110 §8.7; RFC 5789 §2.1 pairs them).
    return _metadata_response(meta, None, None, content_location=_canonical_path(dataset_id))
