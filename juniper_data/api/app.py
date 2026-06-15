"""FastAPI application factory and configuration."""

import functools
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette import status

from juniper_data import __version__, provenance
from juniper_data.storage import LocalFSDatasetStore

from .middleware import RequestBodyLimitMiddleware, SecurityHeadersMiddleware, SecurityMiddleware
from .observability import (
    MetricsAuthMiddleware,
    PrometheusMiddleware,
    RequestIdMiddleware,
    configure_logging,
    configure_sentry,
    get_prometheus_app,
    set_build_info,
)
from .routes import datasets, generators, health
from .security import APIKeyAuth, RateLimiter
from .settings import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    settings: Settings = app.state.settings
    storage_path = Path(settings.storage_path)
    store = LocalFSDatasetStore(storage_path)
    datasets.set_store(store)

    configure_logging(settings.log_level, settings.log_format, "juniper-data")
    configure_sentry(settings.sentry_dsn, "juniper-data", __version__, send_pii=settings.sentry_send_pii, traces_sample_rate=settings.sentry_traces_sample_rate)
    if settings.metrics_enabled:
        set_build_info("juniper_data", __version__, git_sha=provenance.git_sha(), build_date=provenance.build_date())

    logger = logging.getLogger("juniper_data")
    logger.info(f"JuniperData API v{__version__} starting")
    # ``Path.absolute()`` is pure path manipulation (no I/O); the
    # ASYNC240 rule is over-conservative here and flags every
    # ``pathlib.Path`` method without distinguishing stat-bound ones
    # from text-only ones. Lifespan startup is also a one-shot
    # event, not a request handler — even if there were I/O it
    # wouldn't block per-request latency.
    logger.info(f"Storage path: {storage_path.absolute()}")  # noqa: ASYNC240

    yield

    logger.info("JuniperData API shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override. If not provided,
                  settings are loaded from environment variables.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    # Disable interactive API docs when authentication is enabled (production).
    docs_enabled = not settings.api_keys
    app = FastAPI(
        title="Juniper Data API",
        description="Dataset generation and management service for the Juniper ecosystem",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    app.state.settings = settings

    # CORS: only enable when origins are explicitly configured.
    allow_credentials = bool(settings.cors_origins) and "*" not in settings.cors_origins

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Request body size limit
    app.add_middleware(RequestBodyLimitMiddleware)

    # Security headers (outermost — runs on every response)
    app.add_middleware(SecurityHeadersMiddleware)

    api_key_auth = APIKeyAuth(settings.api_keys)
    rate_limiter = RateLimiter(
        requests_per_minute=settings.rate_limit_requests_per_minute,
        enabled=settings.rate_limit_enabled,
    )
    app.add_middleware(
        SecurityMiddleware,
        api_key_auth=api_key_auth,
        rate_limiter=rate_limiter,
    )

    # Observability middleware (added after SecurityMiddleware, before CORS)
    # Middleware execution is LIFO: last added runs first.
    # Order: RequestIdMiddleware → PrometheusMiddleware → SecurityMiddleware → SecurityHeaders → CORS
    if settings.metrics_enabled:
        app.add_middleware(PrometheusMiddleware, service_name="juniper-data", namespace="juniper_data")
    app.add_middleware(RequestIdMiddleware)

    app.include_router(health.router, prefix="/v1")
    app.include_router(generators.router, prefix="/v1")
    app.include_router(datasets.router, prefix="/v1")

    # Mount Prometheus metrics endpoint (SEC-16: wrap with trusted-IP
    # auth because ASGI sub-app mounts bypass SecurityMiddleware).
    if settings.metrics_enabled:
        app.mount(
            "/metrics",
            MetricsAuthMiddleware(get_prometheus_app(), settings.metrics_trusted_ips),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logging.getLogger("juniper_data").debug("Validation error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid request parameters"},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("juniper_data").exception("Unhandled exception")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    return app


@functools.lru_cache(maxsize=1)
def get_app() -> FastAPI:
    """Return the singleton FastAPI app instance (lazy factory).

    Use with uvicorn's factory mode::

        uvicorn --factory juniper_data.api.app:get_app

    or programmatically::

        uvicorn.run("juniper_data.api.app:get_app", factory=True)

    The first call builds the app via :func:`create_app` with default
    settings; subsequent calls return the same instance from
    ``functools.lru_cache``. Replaces the previous module-level
    ``app = create_app()`` (CLN-JD-03), which read environment variables
    and registered middleware at import time. Tests that need a fresh
    instance with overridden settings should continue to call
    :func:`create_app` directly with explicit ``Settings``.
    """
    return create_app()
