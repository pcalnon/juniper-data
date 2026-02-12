"""Integration tests for the FastAPI REST API.

Tests cover all endpoints:
- Health check
- Generators listing and schema
- Dataset CRUD operations
- Artifact download
- Preview functionality
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from juniper_data import __version__
from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture
def memory_store() -> InMemoryDatasetStore:
    """Create a fresh in-memory store for each test."""
    return InMemoryDatasetStore()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(storage_path="/tmp/juniper_data_test")


@pytest.fixture
def client(memory_store: InMemoryDatasetStore, test_settings: Settings) -> TestClient:
    """Create a test client with in-memory storage."""
    app = create_app(settings=test_settings)
    datasets.set_store(memory_store)
    return TestClient(app)


@pytest.fixture
def spiral_request() -> dict:
    """Default spiral dataset creation request."""
    return {
        "generator": "spiral",
        "params": {
            "n_spirals": 2,
            "n_points_per_spiral": 50,
            "seed": 42,
        },
        "persist": True,
    }


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for the /v1/health endpoints."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """GET /v1/health returns {"status": "ok"}."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client: TestClient) -> None:
        """Response includes version string."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["version"] == __version__

    def test_liveness_probe(self, client: TestClient) -> None:
        """GET /v1/health/live returns liveness status."""
        response = client.get("/v1/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_probe(self, client: TestClient) -> None:
        """GET /v1/health/ready returns readiness status with version."""
        response = client.get("/v1/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["version"] == __version__


@pytest.mark.integration
class TestGeneratorsEndpoint:
    """Tests for the /v1/generators endpoints."""

    def test_list_generators(self, client: TestClient) -> None:
        """GET /v1/generators returns list with "spiral"."""
        response = client.get("/v1/generators")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

        generator_names = [g["name"] for g in data]
        assert "spiral" in generator_names

    def test_get_generator_schema(self, client: TestClient) -> None:
        """GET /v1/generators/spiral/schema returns valid schema."""
        response = client.get("/v1/generators/spiral/schema")

        assert response.status_code == 200
        schema = response.json()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "n_spirals" in schema["properties"]
        assert "n_points_per_spiral" in schema["properties"]

    def test_unknown_generator_404(self, client: TestClient) -> None:
        """GET /v1/generators/unknown/schema returns 404."""
        response = client.get("/v1/generators/unknown/schema")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "unknown" in data["detail"].lower()


@pytest.mark.integration
class TestDatasetsEndpoint:
    """Tests for the /v1/datasets endpoints."""

    def test_create_spiral_dataset(self, client: TestClient, spiral_request: dict) -> None:
        """POST /v1/datasets creates dataset and returns meta."""
        response = client.post("/v1/datasets", json=spiral_request)

        assert response.status_code == 201
        data = response.json()
        assert "dataset_id" in data
        assert "meta" in data
        assert data["generator"] == "spiral"
        assert data["meta"]["generator"] == "spiral"
        assert data["meta"]["n_samples"] == 100

    def test_create_returns_artifact_url(self, client: TestClient, spiral_request: dict) -> None:
        """Response includes artifact_url."""
        response = client.post("/v1/datasets", json=spiral_request)

        assert response.status_code == 201
        data = response.json()
        assert "artifact_url" in data
        assert "/v1/datasets/" in data["artifact_url"]
        assert "/artifact" in data["artifact_url"]

    def test_list_datasets(self, client: TestClient, spiral_request: dict) -> None:
        """GET /v1/datasets returns list after creation."""
        client.post("/v1/datasets", json=spiral_request)

        response = client.get("/v1/datasets")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_dataset_meta(self, client: TestClient, spiral_request: dict) -> None:
        """GET /v1/datasets/{id} returns metadata."""
        create_response = client.post("/v1/datasets", json=spiral_request)
        dataset_id = create_response.json()["dataset_id"]

        response = client.get(f"/v1/datasets/{dataset_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert data["generator"] == "spiral"
        assert "n_samples" in data

    def test_get_dataset_404(self, client: TestClient) -> None:
        """GET /v1/datasets/nonexistent returns 404."""
        response = client.get("/v1/datasets/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_delete_dataset(self, client: TestClient, spiral_request: dict) -> None:
        """DELETE /v1/datasets/{id} returns 204."""
        create_response = client.post("/v1/datasets", json=spiral_request)
        dataset_id = create_response.json()["dataset_id"]

        response = client.delete(f"/v1/datasets/{dataset_id}")

        assert response.status_code == 204

        get_response = client.get(f"/v1/datasets/{dataset_id}")
        assert get_response.status_code == 404

    def test_caching_same_params(self, client: TestClient, spiral_request: dict) -> None:
        """Same params twice returns same dataset_id (no regeneration)."""
        response1 = client.post("/v1/datasets", json=spiral_request)
        response2 = client.post("/v1/datasets", json=spiral_request)

        assert response1.status_code == 201
        assert response2.status_code == 201

        data1 = response1.json()
        data2 = response2.json()
        assert data1["dataset_id"] == data2["dataset_id"]


@pytest.mark.integration
class TestArtifactEndpoint:
    """Tests for the /v1/datasets/{id}/artifact endpoint."""

    def test_download_artifact(self, client: TestClient, spiral_request: dict) -> None:
        """GET /v1/datasets/{id}/artifact returns NPZ bytes."""
        create_response = client.post("/v1/datasets", json=spiral_request)
        dataset_id = create_response.json()["dataset_id"]

        response = client.get(f"/v1/datasets/{dataset_id}/artifact")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert len(response.content) > 0

        with np.load(io.BytesIO(response.content)) as data:
            assert len(data.files) > 0

    def test_artifact_contains_expected_keys(self, client: TestClient, spiral_request: dict) -> None:
        """NPZ has X_train, y_train, X_test, y_test."""
        create_response = client.post("/v1/datasets", json=spiral_request)
        dataset_id = create_response.json()["dataset_id"]

        response = client.get(f"/v1/datasets/{dataset_id}/artifact")

        assert response.status_code == 200

        with np.load(io.BytesIO(response.content)) as data:
            assert "X_train" in data.files
            assert "y_train" in data.files
            assert "X_test" in data.files
            assert "y_test" in data.files


@pytest.mark.integration
class TestPreviewEndpoint:
    """Tests for the /v1/datasets/{id}/preview endpoint."""

    def test_preview_returns_samples(self, client: TestClient, spiral_request: dict) -> None:
        """GET /v1/datasets/{id}/preview returns JSON with samples."""
        create_response = client.post("/v1/datasets", json=spiral_request)
        dataset_id = create_response.json()["dataset_id"]

        response = client.get(f"/v1/datasets/{dataset_id}/preview")

        assert response.status_code == 200
        data = response.json()
        assert "n_samples" in data
        assert "X_sample" in data
        assert "y_sample" in data
        assert isinstance(data["X_sample"], list)
        assert isinstance(data["y_sample"], list)
        assert len(data["X_sample"]) > 0
        assert len(data["y_sample"]) > 0

    def test_preview_respects_n_param(self, client: TestClient, spiral_request: dict) -> None:
        """?n=10 returns 10 samples."""
        create_response = client.post("/v1/datasets", json=spiral_request)
        dataset_id = create_response.json()["dataset_id"]

        response = client.get(f"/v1/datasets/{dataset_id}/preview?n=10")

        assert response.status_code == 200
        data = response.json()
        assert data["n_samples"] == 10
        assert len(data["X_sample"]) == 10
        assert len(data["y_sample"]) == 10
