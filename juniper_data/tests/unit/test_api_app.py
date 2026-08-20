"""Unit tests for the FastAPI application factory and configuration."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_core import PydanticSerializationError

from juniper_data import __version__
from juniper_data.api.app import create_app, lifespan
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        storage_path="/tmp/juniper_test",
        host="127.0.0.1",
        port=8100,
        log_level="DEBUG",
        cors_origins=["http://localhost:3000"],
    )


@pytest.fixture
def memory_store() -> InMemoryDatasetStore:
    """Create in-memory store for testing."""
    return InMemoryDatasetStore()


@pytest.mark.unit
class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_create_app_returns_fastapi_instance(self, test_settings: Settings) -> None:
        """Test create_app returns a FastAPI instance."""
        app = create_app(settings=test_settings)
        assert isinstance(app, FastAPI)

    def test_create_app_sets_title(self, test_settings: Settings) -> None:
        """Test app has correct title."""
        app = create_app(settings=test_settings)
        assert app.title == "Juniper Data API"

    def test_create_app_sets_version(self, test_settings: Settings) -> None:
        """Test app has correct version."""
        app = create_app(settings=test_settings)
        assert app.version == __version__

    def test_create_app_stores_settings(self, test_settings: Settings) -> None:
        """Test settings are stored in app state."""
        app = create_app(settings=test_settings)
        assert app.state.settings == test_settings

    def test_create_app_includes_health_router(self, test_settings: Settings) -> None:
        """Test health router is included."""
        app = create_app(settings=test_settings)
        # fastapi >=0.137 wraps included routers in ``_IncludedRouter`` objects
        # (``path`` is ``None``), so included sub-paths are no longer flat in
        # ``app.routes``. The OpenAPI schema reflects registered REST routes
        # robustly across fastapi versions.
        assert "/v1/health" in app.openapi()["paths"]

    def test_create_app_includes_generators_router(self, test_settings: Settings) -> None:
        """Test generators router is included."""
        app = create_app(settings=test_settings)
        assert "/v1/generators" in app.openapi()["paths"]

    def test_create_app_includes_datasets_router(self, test_settings: Settings) -> None:
        """Test datasets router is included."""
        app = create_app(settings=test_settings)
        assert "/v1/datasets" in app.openapi()["paths"]

    def test_create_app_uses_default_settings_when_none_provided(self) -> None:
        """Test create_app loads settings from environment when not provided."""
        with patch("juniper_data.api.app.get_settings") as mock_get:
            mock_settings = Settings()
            mock_get.return_value = mock_settings
            app = create_app(settings=None)
            mock_get.assert_called_once()
            assert app.state.settings == mock_settings

    def test_create_app_cors_middleware_added(self, test_settings: Settings) -> None:
        """Test CORS middleware is configured."""
        app = create_app(settings=test_settings)
        middleware_classes = [getattr(m.cls, "__name__", None) for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes


@pytest.fixture
def cors_auth_settings() -> Settings:
    """Settings with BOTH a CORS origin and an API key, so auth is actually active.

    The plain ``test_settings`` fixture configures no API key, which leaves
    ``SecurityMiddleware`` permissive -- a preflight would pass there for the
    wrong reason.
    """
    return Settings(
        storage_path="/tmp/juniper_test",
        cors_origins=["http://localhost:3000"],
        api_keys=["preflight-test-key"],
    )


@pytest.mark.unit
class TestCorsPreflight:
    """CORS must execute OUTSIDE SecurityMiddleware.

    Regression coverage for APD-DATA-035 (sibling of APD-CASCOR-001b). CORS was
    registered first, which under Starlette's prepending ``add_middleware`` made
    it the INNERMOST layer -- so ``SecurityMiddleware`` saw browser preflights
    first and answered them 401. A preflight carries no ``X-API-Key`` by
    specification, so no browser could ever reach a protected endpoint.
    """

    def test_cors_executes_outside_security_middleware(self, cors_auth_settings: Settings) -> None:
        """Order is the contract: index 0 runs outermost, so CORS must precede Security."""
        app = create_app(settings=cors_auth_settings)
        order = [getattr(m.cls, "__name__", None) for m in app.user_middleware]

        assert "CORSMiddleware" in order, order
        assert "SecurityMiddleware" in order, order
        assert order.index("CORSMiddleware") < order.index("SecurityMiddleware"), f"CORS must run outside SecurityMiddleware, got outermost-first order {order}"

    def test_preflight_to_protected_path_is_not_answered_401(self, cors_auth_settings: Settings) -> None:
        """The defect itself: a genuine preflight must get CORS headers, not 401."""
        client = TestClient(create_app(settings=cors_auth_settings))

        response = client.options(
            "/v1/generators",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.status_code != 401, "preflight was rejected by auth; it carries no API key by design"
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_preflight_from_disallowed_origin_is_still_rejected(self, cors_auth_settings: Settings) -> None:
        """Negative control: moving CORS outermost must not accept arbitrary origins."""
        client = TestClient(create_app(settings=cors_auth_settings))

        response = client.options(
            "/v1/generators",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.headers.get("access-control-allow-origin") is None
        assert response.status_code == 400

    def test_non_preflight_options_still_requires_auth(self, cors_auth_settings: Settings) -> None:
        """The auth surface must not widen.

        This is why the fix is a reorder and not an ``OPTIONS`` bypass inside
        ``_is_exempt``: a bypass would exempt every ``OPTIONS`` request, while
        CORS short-circuits only a genuine preflight (one carrying
        ``Access-Control-Request-Method``).
        """
        client = TestClient(create_app(settings=cors_auth_settings))

        with_origin = client.options("/v1/generators", headers={"Origin": "http://localhost:3000"})
        bare = client.options("/v1/generators")

        assert with_origin.status_code == 401
        assert bare.status_code == 401

    def test_auth_failure_still_carries_cors_headers(self, cors_auth_settings: Settings) -> None:
        """Outermost CORS also annotates error responses.

        Without this a browser sees an opaque CORS failure instead of the real
        401, which is why the misordering was so hard to diagnose from the
        client side.
        """
        client = TestClient(create_app(settings=cors_auth_settings))

        response = client.get("/v1/generators", headers={"Origin": "http://localhost:3000"})

        assert response.status_code == 401
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


@pytest.mark.unit
class TestExceptionHandlers:
    """Tests for custom exception handlers."""

    def test_value_error_returns_400(self, test_settings: Settings, memory_store: InMemoryDatasetStore) -> None:
        """Test ValueError is handled with 400 status."""
        app = create_app(settings=test_settings)
        datasets.set_store(memory_store)

        @app.get("/test-value-error")
        async def raise_value_error():
            raise ValueError("Test error message")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-value-error")

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid request parameters"

    def test_serialization_fault_returns_500_not_400(self, test_settings: Settings, memory_store: InMemoryDatasetStore) -> None:
        """APD-DATA-034: a serialisation fault is the server's, not the caller's.

        ``PydanticSerializationError`` subclasses ``ValueError``, so the blanket
        handler reported the app's own failure to serialise a response as a 400 --
        invisible to 5xx alerting, misattributed to the client, and stripped of its
        diagnostic by the generic "Invalid request parameters" message. Unlike
        juniper-cascor, juniper-data has no ``coerce_native_scalars`` helper
        pre-empting the common case, so every such fault landed here.
        """
        app = create_app(settings=test_settings)
        datasets.set_store(memory_store)

        @app.get("/test-serialization-error")
        async def raise_serialization_error():
            raise PydanticSerializationError("Unable to serialize unknown type: <class 'numpy.float32'>")

        client = TestClient(app, raise_server_exceptions=False)

        with patch("logging.Logger.exception"):
            response = client.get("/test-serialization-error")

        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"

    def test_general_exception_returns_500(self, test_settings: Settings, memory_store: InMemoryDatasetStore) -> None:
        """Test unhandled Exception returns 500 status."""
        app = create_app(settings=test_settings)
        datasets.set_store(memory_store)

        @app.get("/test-general-error")
        async def raise_general_error():
            raise RuntimeError("Unexpected error")

        client = TestClient(app, raise_server_exceptions=False)

        with patch("logging.Logger.exception"):
            response = client.get("/test-general-error")

        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"


@pytest.mark.unit
class TestLifespan:
    """Tests for the lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_store(self, test_settings: Settings) -> None:
        """Test lifespan sets up the dataset store."""
        app = FastAPI()
        app.state.settings = test_settings

        with patch("juniper_data.api.app.LocalFSDatasetStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store

            with patch("juniper_data.api.app.datasets") as mock_datasets:
                async with lifespan(app):
                    MockStore.assert_called_once()
                    mock_datasets.set_store.assert_called_once_with(mock_store)

    @pytest.mark.asyncio
    async def test_lifespan_logs_startup_message(self, test_settings: Settings) -> None:
        """Test lifespan logs startup message."""
        app = FastAPI()
        app.state.settings = test_settings

        with patch("juniper_data.api.app.LocalFSDatasetStore"):
            with patch("juniper_data.api.app.datasets"):
                with patch("logging.Logger.info") as mock_info:
                    async with lifespan(app):
                        startup_calls = [call for call in mock_info.call_args_list if "starting" in str(call).lower()]
                        assert len(startup_calls) >= 1

    @pytest.mark.asyncio
    async def test_lifespan_logs_shutdown_message(self, test_settings: Settings) -> None:
        """Test lifespan logs shutdown message."""
        app = FastAPI()
        app.state.settings = test_settings

        with patch("juniper_data.api.app.LocalFSDatasetStore"):
            with patch("juniper_data.api.app.datasets"):
                with patch("logging.Logger.info") as mock_info:
                    async with lifespan(app):
                        pass

                    shutdown_calls = [call for call in mock_info.call_args_list if "shutting" in str(call).lower()]
                    assert len(shutdown_calls) >= 1

    @pytest.mark.asyncio
    async def test_lifespan_configures_logging(self, test_settings: Settings) -> None:
        """Test lifespan configures logging with correct level."""
        app = FastAPI()
        app.state.settings = test_settings

        with patch("juniper_data.api.app.LocalFSDatasetStore"):
            with patch("juniper_data.api.app.datasets"):
                with patch("juniper_data.api.app.configure_logging") as mock_config:
                    async with lifespan(app):
                        mock_config.assert_called_once_with(
                            test_settings.log_level,
                            test_settings.log_format,
                            "juniper-data",
                        )


@pytest.mark.unit
class TestGetAppFactory:
    """Tests for the cached get_app() factory (CLN-JD-03)."""

    def test_get_app_returns_fastapi(self) -> None:
        """get_app() returns a FastAPI instance."""
        from juniper_data.api.app import get_app

        assert isinstance(get_app(), FastAPI)

    def test_get_app_has_correct_title(self) -> None:
        """get_app() returns an app with the expected title."""
        from juniper_data.api.app import get_app

        assert get_app().title == "Juniper Data API"

    def test_get_app_is_cached(self) -> None:
        """get_app() returns the same singleton instance across calls."""
        from juniper_data.api.app import get_app

        assert get_app() is get_app()
