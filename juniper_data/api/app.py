"""FastAPI application factory and configuration."""

import functools
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from juniper_service_core import enforce_auth_posture
from pydantic_core import PydanticSerializationError
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
    # SEC-F01 (HO-2): boot-time auth-posture self-check. An empty/blank
    # JUNIPER_DATA_API_KEYS secret silently disables APIKeyAuth and the API
    # serves OPEN (docs exposed, protected routes unauthenticated) behind a
    # healthy health check; make that posture loud here, before serving
    # begins. The intended posture comes from JUNIPER_DATA_REQUIRE_AUTH
    # (settings.require_auth; default false): false = loud WARNING only
    # (bare/dev profile), true = a missing/blank key is a boot FAILURE
    # (CRITICAL + AuthPostureError) — set true wherever secrets are
    # provisioned (the composed juniper-deploy stack). Bypass with
    # JUNIPER_SKIP_AUTH_POSTURE_CHECK=1 (logged loudly).
    enforce_auth_posture(
        settings.api_keys,
        require_auth=settings.require_auth,
        service_name="juniper-data",
        logger=logger,
    )
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

    # Request body size limit
    app.add_middleware(RequestBodyLimitMiddleware)

    # Security headers
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

    # Observability middleware
    if settings.metrics_enabled:
        app.add_middleware(PrometheusMiddleware, service_name="juniper-data", namespace="juniper_data")
    app.add_middleware(RequestIdMiddleware)

    # CORS: only enable when origins are explicitly configured.
    #
    # Registered LAST so it executes OUTERMOST. Starlette's ``add_middleware``
    # prepends, so execution order is the reverse of registration order:
    #
    #   CORS → RequestId → Prometheus → Security → SecurityHeaders
    #        → RequestBodyLimit → routes
    #
    # CORS must sit OUTSIDE SecurityMiddleware. A browser preflight carries no
    # ``X-API-Key``: the browser generates the preflight itself, and
    # author-defined headers ride only on the actual request that follows. So
    # with CORS innermost every preflight to a non-exempt path was answered 401
    # and the browser never issued the real request. Outermost also attaches the CORS
    # headers to error responses (401/429), so a browser can surface the real
    # status instead of an opaque CORS failure.
    allow_credentials = bool(settings.cors_origins) and "*" not in settings.cors_origins

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

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
        # ``PydanticSerializationError`` subclasses ValueError, but it is a SERVER
        # fault: the app failed to serialise its own response. Reporting it as 400
        # misattributes our defect to the caller, hides it from 5xx alerting, and
        # replaces the diagnostic with "Invalid request parameters". Classify it as
        # the 500 it is, and log at exception level so the traceback survives.
        #
        # juniper-cascor carries the same handler and pre-empts the common
        # numpy-scalar case with a ``coerce_native_scalars`` helper; juniper-data
        # has no such helper, so before this branch existed every serialisation
        # fault here was reported as a client error.
        if isinstance(exc, PydanticSerializationError):
            logging.getLogger("juniper_data").exception("Response serialization failed")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"},
            )
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
