"""Health check endpoints for container orchestration.

Provides three health check endpoints:
- /v1/health: Combined health check (backward compatible)
- /v1/health/live: Liveness probe — runs an in-process tick within a strict
  budget; returns 503 if the tick fails or exceeds the budget so the
  orchestrator can restart wedged pods.
- /v1/health/ready: Readiness probe — returns 200 when all required
  dependencies are healthy, 200 with status "degraded" when only optional
  dependencies are unhealthy, and 503 when a required dependency is
  unhealthy so load balancers can shed traffic without parsing the body.

See ``notes/code-review/METRICS_MONITORING_R1.2_PROBE_DESIGN_2026-04-27.md``
in juniper-ml for the cross-repo contract this implements (R1.2 / seed-02
and seed-03).
"""

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Request, Response

from juniper_data import __version__, provenance
from juniper_data.api.models.health import DependencyStatus, ReadinessResponse
from juniper_data.api.settings import Settings, get_settings

router = APIRouter(tags=["health"])


def _settings_from_request(request: Request) -> Settings:
    """Resolve settings from the request app's state, falling back to the
    cached process-wide ``get_settings()``.

    Reading from ``app.state.settings`` ensures tests that override
    settings via ``create_app(settings=...)`` use the test value rather
    than the lru-cached process settings.
    """
    return getattr(request.app.state, "settings", None) or get_settings()


# R1.2: liveness tick must complete in-process within this wall-clock budget.
# Helm timeoutSeconds (5–10) wraps this with headroom; the budget catches
# event-loop stalls and CPU starvation that the no-op ``return {"status":
# "alive"}`` could not.
LIVENESS_TICK_BUDGET_MS = 250

# R1.2: header surfaces readiness state to ``kubectl describe pod`` /
# ``curl -I`` without requiring body parsing.
READINESS_HEADER = "X-Juniper-Readiness"


def _liveness_tick(settings: Settings) -> None:
    """Run the juniper-data liveness tick.

    Pure in-process work: confirms the configured storage path resolves to
    a directory. Raises if it does not.
    """
    if not Path(settings.storage_path).is_dir():
        raise RuntimeError(f"storage path not a directory: {settings.storage_path}")


# PERF-JD-01: short-lived cache for the readiness probe's dataset count.
# Plain ``len(list(storage_path.glob("*.npz")))`` is O(n) per probe and
# orchestrators poll readiness every few seconds — a stale-tolerant 5s
# cache cuts the steady-state cost to one glob per cache window without
# instrumenting every dataset save / delete path (which would couple the
# write hot path to readiness bookkeeping and miss out-of-band changes
# like manual filesystem edits or test fixtures).
#
# Stored as ``(cached_at_perf_counter, is_dir, dataset_count, storage_path_str)``
# so a change to the configured storage path (test fixtures install
# ``create_app(settings=...)`` with a tmpdir) invalidates the cache.
# Concurrent probes that race the cache miss are benign: at most a few
# extra globs in a narrow window; the last writer wins.
_PROBE_CACHE_TTL_SECONDS = 5.0
_probe_cache: tuple[float, bool, int, str] | None = None


def _probe_storage(storage_path: Path) -> tuple[bool, int]:
    """Filesystem probe for the readiness route.

    Bundles the ``is_dir()`` stat and the ``*.npz`` glob into a single
    helper so the readiness route takes one ``asyncio.to_thread``
    hop instead of two. Returns ``(is_dir, dataset_count)``;
    ``dataset_count`` is 0 when the path isn't a directory.

    PERF-JD-01: results are cached for ``_PROBE_CACHE_TTL_SECONDS`` to
    keep steady-state readiness probes O(1) on the hot path. Per-test
    invalidation uses :func:`_reset_probe_cache`.
    """
    global _probe_cache
    now = time.perf_counter()
    path_str = str(storage_path)
    cached = _probe_cache
    if cached is not None and now - cached[0] < _PROBE_CACHE_TTL_SECONDS and cached[3] == path_str:
        return cached[1], cached[2]

    if not storage_path.is_dir():
        _probe_cache = (now, False, 0, path_str)
        return False, 0
    count = len(list(storage_path.glob("*.npz")))
    _probe_cache = (now, True, count, path_str)
    return True, count


def _reset_probe_cache() -> None:
    """Drop the readiness probe's cached dataset count.

    Exposed for tests that need to observe fresh filesystem state
    without waiting out the TTL.
    """
    global _probe_cache
    _probe_cache = None


@router.get("/health")
async def health_check() -> dict:
    """Combined health check endpoint (backward compatible).

    Always returns 200 while the process can respond. Reserved for legacy
    integrations; new probes should use ``/health/live`` or
    ``/health/ready``.

    Response schema (API-02 shared base):

    - ``status``  — always ``"ok"`` on success.
    - ``version`` — the package version of this service.
    - ``service`` — canonical service identifier (``"juniper-data"``);
      matches the cross-service ``{status, version, service}`` base
      shared by juniper-cascor and juniper-canopy so monitoring tools
      can tell health responses apart without inspecting the URL.
    """
    return {
        "status": "ok",
        "version": __version__,
        "service": "juniper-data",
        # Build provenance (juniper-ml notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md):
        # source git SHA + ISO-8601 build date baked into the image. ``None``
        # outside a provenance-stamped image; lets ``make doctor`` detect drift.
        "git_sha": provenance.git_sha(),
        "build_date": provenance.build_date(),
    }


@router.get("/health/live")
async def liveness_probe(request: Request, response: Response) -> dict:
    """Liveness probe — runs an in-process tick within a strict budget.

    Returns 200 with ``{"status": "alive", "tick": "juniper-data",
    "duration_ms": N}`` when the tick succeeds within
    ``LIVENESS_TICK_BUDGET_MS``. Returns 503 with ``{"status":
    "unresponsive", ...}`` otherwise.
    """
    settings = _settings_from_request(request)
    started = time.perf_counter()
    try:
        _liveness_tick(settings)
    except Exception as exc:  # noqa: BLE001 — health probe must surface every failure
        duration_ms = int((time.perf_counter() - started) * 1000)
        response.status_code = 503
        return {
            "status": "unresponsive",
            "tick": "juniper-data",
            "error": str(exc),
            "duration_ms": duration_ms,
        }

    duration_ms = int((time.perf_counter() - started) * 1000)
    if duration_ms > LIVENESS_TICK_BUDGET_MS:
        response.status_code = 503
        return {
            "status": "unresponsive",
            "tick": "juniper-data",
            "error": f"tick exceeded budget: {duration_ms}ms > {LIVENESS_TICK_BUDGET_MS}ms",
            "duration_ms": duration_ms,
        }

    return {
        "status": "alive",
        "tick": "juniper-data",
        "duration_ms": duration_ms,
    }


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_probe(request: Request, response: Response) -> ReadinessResponse:
    """Readiness probe — drives orchestrator traffic decisions via status code.

    Status code semantics:

    - 200, body status="ready"     — all required deps healthy.
    - 200, body status="degraded"  — required deps healthy, an optional dep
      unhealthy. juniper-data has no optional deps; this branch is
      unreachable for this service.
    - 503, body status="not_ready" — at least one required dep unhealthy.

    Sets ``X-Juniper-Readiness`` header to mirror body status so probe
    diagnostics surface in orchestrator logs.
    """
    settings = _settings_from_request(request)
    storage_path = Path(settings.storage_path)

    # Probe filesystem off the event loop. ``is_dir()`` is a stat
    # syscall and ``glob()`` walks the directory; both block on slow
    # disks. Bundled into a single ``to_thread`` call so the readiness
    # probe takes one thread-hop, not two.
    is_dir, dataset_count = await asyncio.to_thread(_probe_storage, storage_path)

    storage_dep = (
        DependencyStatus(
            name="Dataset Storage",
            status="healthy",
            message=f"{storage_path} ({dataset_count} datasets)",
        )
        if is_dir
        else DependencyStatus(
            name="Dataset Storage",
            status="unhealthy",
            message=f"{storage_path} not found or not a directory",
        )
    )

    # Storage is the sole REQUIRED dep for juniper-data; degraded is unreachable.
    if storage_dep.status == "healthy":
        overall = "ready"
    else:
        overall = "not_ready"
        response.status_code = 503

    response.headers[READINESS_HEADER] = overall

    return ReadinessResponse(
        status=overall,
        version=__version__,
        service="juniper-data",
        git_sha=provenance.git_sha(),
        build_date=provenance.build_date(),
        dependencies={"storage": storage_dep},
    )
