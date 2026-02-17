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

from .middleware import SecurityMiddleware
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

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
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

    app = FastAPI(
        title="Juniper Data API",
        description="Dataset generation and management service for the Juniper ecosystem",
        version=__version__,
        lifespan=lifespan,
    )

    app.state.settings = settings

    # Only allow credentialed CORS requests when origins are explicitly specified.
    # Browsers do not permit Access-Control-Allow-Credentials: true with a wildcard
    # origin (Access-Control-Allow-Origin: "*"), so the default ["*"] intentionally
    # disables credentials unless concrete origins are configured.
    allow_credentials = bool(settings.cors_origins) and "*" not in settings.cors_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    app.include_router(health.router, prefix="/v1")
    app.include_router(generators.router, prefix="/v1")
    app.include_router(datasets.router, prefix="/v1")

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
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
