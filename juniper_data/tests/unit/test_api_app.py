"""Unit tests for the FastAPI application factory and configuration."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
        port=8200,
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
        routes = [route.path for route in app.routes]
        assert "/v1/health" in routes

    def test_create_app_includes_generators_router(self, test_settings: Settings) -> None:
        """Test generators router is included."""
        app = create_app(settings=test_settings)
        routes = [route.path for route in app.routes]
        assert "/v1/generators" in routes

    def test_create_app_includes_datasets_router(self, test_settings: Settings) -> None:
        """Test datasets router is included."""
        app = create_app(settings=test_settings)
        routes = [route.path for route in app.routes]
        assert "/v1/datasets" in routes

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
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes


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
        assert response.json()["detail"] == "Test error message"

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
                with patch("logging.basicConfig") as mock_config:
                    async with lifespan(app):
                        mock_config.assert_called_once()
                        call_kwargs = mock_config.call_args[1]
                        assert call_kwargs["level"] == logging.DEBUG


@pytest.mark.unit
class TestGlobalApp:
    """Tests for the global app instance."""

    def test_global_app_exists(self) -> None:
        """Test global app is created at module level."""
        from juniper_data.api.app import app

        assert isinstance(app, FastAPI)

    def test_global_app_has_correct_title(self) -> None:
        """Test global app has correct title."""
        from juniper_data.api.app import app

        assert app.title == "Juniper Data API"
