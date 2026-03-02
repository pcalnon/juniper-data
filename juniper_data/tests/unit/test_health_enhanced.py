"""Tests for enhanced health check endpoints with dependency status."""

import pytest
from fastapi.testclient import TestClient

from juniper_data import __version__
from juniper_data.api.app import create_app
from juniper_data.api.models.health import DependencyStatus, ReadinessResponse
from juniper_data.api.settings import Settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear cached settings between tests."""
    from juniper_data.api.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings(tmp_path, monkeypatch) -> Settings:
    """Create test settings with a real storage directory."""
    storage = tmp_path / "datasets"
    storage.mkdir()
    monkeypatch.setenv("JUNIPER_DATA_STORAGE_PATH", str(storage))
    return Settings(storage_path=str(storage))


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    """Create a test client."""
    app = create_app(settings=test_settings)
    return TestClient(app)


@pytest.mark.unit
class TestDependencyStatusModel:
    """Test DependencyStatus Pydantic model."""

    def test_healthy_status(self):
        dep = DependencyStatus(name="Test", status="healthy", latency_ms=1.5, message="ok")
        assert dep.name == "Test"
        assert dep.status == "healthy"
        assert dep.latency_ms == 1.5

    def test_unhealthy_status(self):
        dep = DependencyStatus(name="Test", status="unhealthy", message="connection refused")
        assert dep.status == "unhealthy"
        assert dep.latency_ms is None

    def test_not_configured_status(self):
        dep = DependencyStatus(name="Optional", status="not_configured")
        assert dep.status == "not_configured"
        assert dep.message is None


@pytest.mark.unit
class TestReadinessResponseModel:
    """Test ReadinessResponse Pydantic model."""

    def test_ready_response(self):
        resp = ReadinessResponse(status="ready", version="0.4.2", service="juniper-data")
        assert resp.status == "ready"
        assert resp.version == "0.4.2"
        assert resp.service == "juniper-data"
        assert resp.timestamp > 0
        assert resp.dependencies == {}
        assert resp.details == {}

    def test_degraded_response_with_deps(self):
        dep = DependencyStatus(name="Storage", status="unhealthy", message="not found")
        resp = ReadinessResponse(
            status="degraded",
            version="0.4.2",
            service="juniper-data",
            dependencies={"storage": dep},
        )
        assert resp.status == "degraded"
        assert resp.dependencies["storage"].status == "unhealthy"


@pytest.mark.unit
class TestEnhancedReadinessEndpoint:
    """Test enhanced /v1/health/ready endpoint."""

    def test_readiness_with_valid_storage(self, client, test_settings):
        """Ready when storage directory exists."""
        response = client.get("/v1/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["version"] == __version__
        assert body["service"] == "juniper-data"
        assert "timestamp" in body
        assert body["dependencies"]["storage"]["status"] == "healthy"
        assert body["dependencies"]["storage"]["name"] == "Dataset Storage"

    def test_readiness_with_datasets(self, client, test_settings, tmp_path):
        """Storage message includes dataset count."""
        storage = tmp_path / "datasets"
        (storage / "test1.npz").touch()
        (storage / "test2.npz").touch()
        response = client.get("/v1/health/ready")
        body = response.json()
        assert "2 datasets" in body["dependencies"]["storage"]["message"]

    def test_readiness_with_missing_storage(self):
        """Degraded when storage directory is missing."""
        settings = Settings(storage_path="/nonexistent/path/datasets")
        app = create_app(settings=settings)
        c = TestClient(app)
        response = c.get("/v1/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["dependencies"]["storage"]["status"] == "unhealthy"
        assert "not found" in body["dependencies"]["storage"]["message"]


@pytest.mark.unit
class TestBackwardCompatibleEndpoints:
    """Test that /v1/health and /v1/health/live are unchanged."""

    def test_health_check_unchanged(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__

    def test_liveness_unchanged(self, client):
        response = client.get("/v1/health/live")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "alive"
