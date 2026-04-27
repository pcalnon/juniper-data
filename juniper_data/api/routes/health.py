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

import time
from pathlib import Path

from fastapi import APIRouter, Request, Response

from juniper_data import __version__
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


@router.get("/health")
async def health_check() -> dict:
    """Combined health check endpoint (backward compatible).

    Always returns 200 while the process can respond. Reserved for legacy
    integrations; new probes should use ``/health/live`` or
    ``/health/ready``.
    """
    return {"status": "ok", "version": __version__}


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

    if storage_path.is_dir():
        dataset_count = len(list(storage_path.glob("*.npz")))
        storage_dep = DependencyStatus(
            name="Dataset Storage",
            status="healthy",
            message=f"{storage_path} ({dataset_count} datasets)",
        )
    else:
        storage_dep = DependencyStatus(
            name="Dataset Storage",
            status="unhealthy",
            message=f"{storage_path} not found or not a directory",
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
        dependencies={"storage": storage_dep},
    )
