"""FastAPI application factory and configuration."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from juniper_data import __version__
from juniper_data.storage import LocalFSDatasetStore

from .middleware import RequestBodyLimitMiddleware, SecurityHeadersMiddleware, SecurityMiddleware
from .observability import (
    PrometheusMiddleware,
    RequestIdMiddleware,
    configure_logging,
    configure_sentry,
    get_prometheus_app,
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
    configure_sentry(settings.sentry_dsn, "juniper-data", __version__)

    logger = logging.getLogger("juniper_data")
    logger.info(f"JuniperData API v{__version__} starting")
    logger.info(f"Storage path: {storage_path.absolute()}")

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
        app.add_middleware(PrometheusMiddleware, service_name="juniper-data")
    app.add_middleware(RequestIdMiddleware)

    app.include_router(health.router, prefix="/v1")
    app.include_router(generators.router, prefix="/v1")
    app.include_router(datasets.router, prefix="/v1")

    # Mount Prometheus metrics endpoint
    if settings.metrics_enabled:
        app.mount("/metrics", get_prometheus_app())

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logging.getLogger("juniper_data").debug("Validation error: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid request parameters"},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("juniper_data").exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()
