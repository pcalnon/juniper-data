"""Health check response models for standardized readiness reporting."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class DependencyStatus(BaseModel):
    """Health status of a single dependency."""

    name: str
    status: Literal["healthy", "unhealthy", "degraded", "not_configured"]
    latency_ms: float | None = None
    message: str | None = None


class ReadinessResponse(BaseModel):
    """Standard /v1/health/ready response for all Juniper services."""

    status: Literal["ready", "degraded", "not_ready"]
    version: str
    service: str
    # BUG-JD-06: timezone-aware UTC timestamp for consistency across services
    timestamp: float = Field(default_factory=lambda: datetime.now(UTC).timestamp())
    dependencies: dict[str, DependencyStatus] = {}
    details: dict[str, object] = {}
