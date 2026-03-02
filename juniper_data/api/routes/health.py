"""Health check endpoints for container orchestration.

Provides three health check endpoints:
- /v1/health: Combined health check (backward compatible)
- /v1/health/live: Liveness probe - is the process running?
- /v1/health/ready: Readiness probe - is the service ready to accept traffic?
"""

from pathlib import Path

from fastapi import APIRouter

from juniper_data import __version__
from juniper_data.api.models.health import DependencyStatus, ReadinessResponse
from juniper_data.api.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Combined health check endpoint (backward compatible).

    Returns:
        Dictionary with service status and version.
    """
    return {"status": "ok", "version": __version__}


@router.get("/health/live")
async def liveness_probe() -> dict:
    """Liveness probe for container orchestration.

    Used by Kubernetes/Docker to determine if the container should be restarted.
    Returns success if the Python process is running and can respond to requests.

    Returns:
        Dictionary with liveness status.
    """
    return {"status": "alive"}


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_probe() -> ReadinessResponse:
    """Readiness probe for container orchestration.

    Reports ready when storage is accessible. Includes dependency status
    for the dataset storage directory.

    Returns:
        ReadinessResponse with storage dependency health.
    """
    settings = get_settings()
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

    overall = "ready" if storage_dep.status == "healthy" else "degraded"
    return ReadinessResponse(
        status=overall,
        version=__version__,
        service="juniper-data",
        dependencies={"storage": storage_dep},
    )
