"""Health check endpoint."""

from fastapi import APIRouter

from juniper_data import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Dictionary with service status and version.
    """
    return {"status": "ok", "version": __version__}
