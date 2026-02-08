"""Health check endpoints for container orchestration.

Provides three health check endpoints:
- /v1/health: Combined health check (backward compatible)
- /v1/health/live: Liveness probe - is the process running?
- /v1/health/ready: Readiness probe - is the service ready to accept traffic?
"""

from fastapi import APIRouter

from juniper_data import __version__

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


@router.get("/health/ready")
async def readiness_probe() -> dict:
    """Readiness probe for container orchestration.

    Used by Kubernetes/Docker to determine if the container can accept traffic.
    Returns success if the service is fully initialized and ready to handle requests.

    Returns:
        Dictionary with readiness status and version.
    """
    return {"status": "ready", "version": __version__}
