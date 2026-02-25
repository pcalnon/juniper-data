"""Unit tests for API route modules."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(storage_path="/tmp/juniper_test")


@pytest.fixture
def memory_store() -> InMemoryDatasetStore:
    """Create in-memory store for testing."""
    return InMemoryDatasetStore()


@pytest.fixture
def client(memory_store: InMemoryDatasetStore, test_settings: Settings) -> TestClient:
    """Create a test client with in-memory storage."""
    app = create_app(settings=test_settings)
    datasets.set_store(memory_store)
    return TestClient(app)


@pytest.mark.unit
class TestDatasetsRouteModule:
    """Tests for the datasets route module functions."""

    def test_get_store_raises_when_not_initialized(self) -> None:
        """Test get_store raises 500 when store is None."""
        datasets._store = None

        with pytest.raises(HTTPException) as exc_info:
            datasets.get_store()

        assert exc_info.value.status_code == 500
        assert "not initialized" in exc_info.value.detail

    def test_set_store_sets_global_store(self, memory_store: InMemoryDatasetStore) -> None:
        """Test set_store correctly sets the global store."""
        datasets.set_store(memory_store)
        assert datasets._store is memory_store

    def test_get_store_returns_store_when_initialized(self, memory_store: InMemoryDatasetStore) -> None:
        """Test get_store returns store when initialized."""
        datasets.set_store(memory_store)
        store = datasets.get_store()
        assert store is memory_store


@pytest.mark.unit
class TestDatasetsEndpointEdgeCases:
    """Tests for edge cases in datasets endpoints."""

    def test_create_dataset_unknown_generator(self, client: TestClient) -> None:
        """Test creating dataset with unknown generator returns 400."""
        request = {"generator": "unknown_generator", "params": {}, "persist": True}
        response = client.post("/v1/datasets", json=request)

        assert response.status_code == 400
        data = response.json()
        assert "Unknown generator" in data["detail"]
        assert "unknown_generator" in data["detail"]

    def test_create_dataset_invalid_params(self, client: TestClient) -> None:
        """Test creating dataset with invalid params returns 400."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": "not_an_integer", "n_points_per_spiral": 100},
            "persist": True,
        }
        response = client.post("/v1/datasets", json=request)

        assert response.status_code == 400
        data = response.json()
        assert "Invalid parameters" in data["detail"]

    def test_create_dataset_without_persist(self, client: TestClient) -> None:
        """Test creating dataset with persist=False does not save."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
            "persist": False,
        }
        response = client.post("/v1/datasets", json=request)

        assert response.status_code == 201
        data = response.json()
        dataset_id = data["dataset_id"]

        get_response = client.get(f"/v1/datasets/{dataset_id}")
        assert get_response.status_code == 404

    def test_download_artifact_not_found(self, client: TestClient) -> None:
        """Test downloading artifact for non-existent dataset returns 404."""
        response = client.get("/v1/datasets/nonexistent-id/artifact")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    def test_preview_not_found(self, client: TestClient) -> None:
        """Test previewing non-existent dataset returns 404."""
        response = client.get("/v1/datasets/nonexistent-id/preview")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    def test_delete_not_found(self, client: TestClient) -> None:
        """Test deleting non-existent dataset returns 404."""
        response = client.delete("/v1/datasets/nonexistent-id")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    def test_preview_uses_x_full_y_full_when_available(self, client: TestClient) -> None:
        """Test preview uses X_full/y_full arrays when available."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
            "persist": True,
        }
        response = client.post("/v1/datasets", json=request)
        assert response.status_code == 201

        dataset_id = response.json()["dataset_id"]

        preview_response = client.get(f"/v1/datasets/{dataset_id}/preview?n=10")
        assert preview_response.status_code == 200
        data = preview_response.json()
        assert data["n_samples"] == 10
        assert len(data["X_sample"]) == 10

    def test_list_datasets_with_pagination(self, client: TestClient) -> None:
        """Test listing datasets with limit and offset."""
        for i in range(5):
            request = {
                "generator": "spiral",
                "params": {"n_spirals": 2, "n_points_per_spiral": 10, "seed": i},
                "persist": True,
            }
            client.post("/v1/datasets", json=request)

        response = client.get("/v1/datasets?limit=2&offset=1")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_create_dataset_returns_existing(self, client: TestClient) -> None:
        """Creating same dataset twice returns cached version."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
            "persist": True,
        }
        response1 = client.post("/v1/datasets", json=request)
        assert response1.status_code == 201
        response2 = client.post("/v1/datasets", json=request)
        assert response2.status_code == 201
        assert response1.json()["dataset_id"] == response2.json()["dataset_id"]

    def test_create_dataset_with_ttl(self, client: TestClient) -> None:
        """Creating dataset with TTL sets expires_at."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 99},
            "persist": True,
            "ttl_seconds": 3600,
        }
        response = client.post("/v1/datasets", json=request)
        assert response.status_code == 201
        meta = response.json()["meta"]
        assert meta["expires_at"] is not None

    def test_get_dataset_stats(self, client: TestClient) -> None:
        """Stats endpoint returns aggregate statistics."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
            "persist": True,
        }
        client.post("/v1/datasets", json=request)
        response = client.get("/v1/datasets/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_datasets" in data
        assert data["total_datasets"] >= 1

    def test_batch_delete(self, client: TestClient) -> None:
        """Batch delete removes multiple datasets."""
        ids = []
        for i in range(3):
            request = {
                "generator": "spiral",
                "params": {"n_spirals": 2, "n_points_per_spiral": 10, "seed": i + 1000},
                "persist": True,
            }
            resp = client.post("/v1/datasets", json=request)
            ids.append(resp.json()["dataset_id"])
        response = client.post("/v1/datasets/batch-delete", json={"dataset_ids": ids + ["nonexistent-id"]})
        assert response.status_code == 200
        data = response.json()
        assert data["total_deleted"] == 3
        assert "nonexistent-id" in data["not_found"]

    def test_cleanup_expired(self, client: TestClient) -> None:
        """Cleanup expired endpoint returns list."""
        response = client.post("/v1/datasets/cleanup-expired")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_download_artifact(self, client: TestClient) -> None:
        """Download artifact returns NPZ bytes."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
            "persist": True,
        }
        resp = client.post("/v1/datasets", json=request)
        dataset_id = resp.json()["dataset_id"]
        response = client.get(f"/v1/datasets/{dataset_id}/artifact")
        assert response.status_code == 200
        assert len(response.content) > 0

    def test_update_tags(self, client: TestClient) -> None:
        """PATCH tags endpoint adds and removes tags."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
            "persist": True,
            "tags": ["original"],
        }
        resp = client.post("/v1/datasets", json=request)
        dataset_id = resp.json()["dataset_id"]
        patch_resp = client.patch(
            f"/v1/datasets/{dataset_id}/tags",
            json={"add_tags": ["new-tag"], "remove_tags": ["original"]},
        )
        assert patch_resp.status_code == 200
        data = patch_resp.json()
        assert "new-tag" in data["tags"]
        assert "original" not in data["tags"]

    def test_update_tags_not_found(self, client: TestClient) -> None:
        """PATCH tags for nonexistent dataset returns 404."""
        response = client.patch("/v1/datasets/nonexistent/tags", json={"add_tags": ["tag"]})
        assert response.status_code == 404

    def test_filter_datasets(self, client: TestClient) -> None:
        """Filter datasets endpoint returns filtered results."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
            "persist": True,
            "tags": ["test-filter"],
        }
        client.post("/v1/datasets", json=request)
        response = client.get("/v1/datasets/filter?generator=spiral&tags=test-filter")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "total" in data

    def test_get_dataset_metadata(self, client: TestClient) -> None:
        """GET /v1/datasets/{id} returns metadata for existing dataset."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
            "persist": True,
        }
        resp = client.post("/v1/datasets", json=request)
        dataset_id = resp.json()["dataset_id"]

        response = client.get(f"/v1/datasets/{dataset_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset_id
        assert data["generator"] == "spiral"

    def test_preview_stacks_train_test_when_no_full_arrays(
        self, memory_store: InMemoryDatasetStore, test_settings: Settings
    ) -> None:
        """Test preview stacks X_train/X_test when X_full/y_full not available."""
        from datetime import datetime

        import numpy as np

        from juniper_data.core.models import DatasetMeta

        app = create_app(settings=test_settings)
        datasets.set_store(memory_store)
        client = TestClient(app)

        meta = DatasetMeta(
            dataset_id="test-no-full",
            generator="spiral",
            generator_version="1.0.0",
            params={"n_spirals": 2},
            n_samples=20,
            n_features=2,
            n_classes=2,
            n_train=16,
            n_test=4,
            class_distribution={"0": 10, "1": 10},
            created_at=datetime.now(),
        )

        arrays = {
            "X_train": np.random.randn(16, 2).astype(np.float32),
            "y_train": np.eye(2, dtype=np.float32)[np.random.randint(0, 2, 16)],
            "X_test": np.random.randn(4, 2).astype(np.float32),
            "y_test": np.eye(2, dtype=np.float32)[np.random.randint(0, 2, 4)],
        }
        memory_store.save("test-no-full", meta, arrays)

        response = client.get("/v1/datasets/test-no-full/preview?n=10")

        assert response.status_code == 200
        data = response.json()
        assert data["n_samples"] == 10
        assert len(data["X_sample"]) == 10


@pytest.mark.unit
class TestGeneratorsEndpoint:
    """Tests for the generators route module."""

    def test_list_generators_returns_list(self, client: TestClient) -> None:
        """Test list generators returns list of generators."""
        response = client.get("/v1/generators")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_list_generators_contains_spiral(self, client: TestClient) -> None:
        """Test list generators contains spiral generator."""
        response = client.get("/v1/generators")

        data = response.json()
        names = [g["name"] for g in data]
        assert "spiral" in names

    def test_get_schema_unknown_generator(self, client: TestClient) -> None:
        """Test getting schema for unknown generator returns 404."""
        response = client.get("/v1/generators/unknown/schema")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_get_schema_spiral_generator(self, client: TestClient) -> None:
        """Test getting schema for spiral generator."""
        response = client.get("/v1/generators/spiral/schema")

        assert response.status_code == 200
        data = response.json()
        assert "properties" in data
        assert "n_spirals" in data["properties"]


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for the health route module."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Test health endpoint returns ok status."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client: TestClient) -> None:
        """Test health endpoint includes version."""
        from juniper_data import __version__

        response = client.get("/v1/health")

        data = response.json()
        assert data["version"] == __version__

    def test_liveness_probe(self, client: TestClient) -> None:
        """Test liveness probe returns alive status."""
        response = client.get("/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_probe(self, client: TestClient) -> None:
        """Test readiness probe returns ready status with version."""
        response = client.get("/v1/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "version" in data
