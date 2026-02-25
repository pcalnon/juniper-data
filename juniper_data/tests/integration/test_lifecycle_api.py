"""Integration tests for dataset lifecycle management API endpoints (DATA-016).

Tests for:
- POST /v1/datasets with tags and TTL
- GET /v1/datasets/filter
- POST /v1/datasets/batch-delete
- PATCH /v1/datasets/{id}/tags
- GET /v1/datasets/stats
- POST /v1/datasets/cleanup-expired
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore

# from typing import Dict


@pytest.fixture
def lifecycle_store() -> InMemoryDatasetStore:
    """Create a fresh in-memory store for lifecycle tests."""
    return InMemoryDatasetStore()


@pytest.fixture
def lifecycle_settings() -> Settings:
    """Create lifecycle test settings."""
    return Settings(storage_path="/tmp/juniper_data_lifecycle_test")


@pytest.fixture
def lifecycle_client(lifecycle_store: InMemoryDatasetStore, lifecycle_settings: Settings) -> TestClient:
    """Create a lifecycle test client with in-memory storage."""
    app = create_app(settings=lifecycle_settings)
    datasets.set_store(lifecycle_store)
    return TestClient(app)


def _create_spiral_request(
    n_points: int = 50,
    seed: int = 42,
    tags: list[str] | None = None,
    ttl_seconds: int | None = None,
) -> dict:
    """Create a spiral dataset request."""
    request = {
        "generator": "spiral",
        "params": {"n_spirals": 2, "n_points_per_spiral": n_points, "seed": seed},
        "persist": True,
    }
    if tags:
        request["tags"] = tags
    if ttl_seconds:
        request["ttl_seconds"] = ttl_seconds
    return request


@pytest.mark.integration
class TestCreateDatasetWithLifecycle:
    """Tests for creating datasets with lifecycle features."""

    def test_create_dataset_with_tags(self, lifecycle_client: TestClient) -> None:
        """Create dataset with tags."""
        request = _create_spiral_request(tags=["train", "experiment-1"])
        response = lifecycle_client.post("/v1/datasets", json=request)

        assert response.status_code == 201
        meta = response.json()["meta"]
        assert "train" in meta["tags"]
        assert "experiment-1" in meta["tags"]

    def test_create_dataset_with_ttl(self, lifecycle_client: TestClient) -> None:
        """Create dataset with TTL."""
        request = _create_spiral_request(ttl_seconds=3600)
        response = lifecycle_client.post("/v1/datasets", json=request)

        assert response.status_code == 201
        meta = response.json()["meta"]
        assert meta["ttl_seconds"] == 3600
        assert meta["expires_at"] is not None


@pytest.mark.integration
class TestFilterDatasets:
    """Tests for the filter datasets endpoint."""

    @pytest.fixture
    def populated_client(self, lifecycle_client: TestClient) -> TestClient:
        """Create multiple datasets for filtering tests."""
        requests = [
            _create_spiral_request(n_points=50, seed=1, tags=["train", "v1"]),
            _create_spiral_request(n_points=100, seed=2, tags=["train", "v2"]),
            _create_spiral_request(n_points=150, seed=3, tags=["test", "v1"]),
            _create_spiral_request(n_points=200, seed=4, tags=["test", "v2"]),
        ]
        for req in requests:
            lifecycle_client.post("/v1/datasets", json=req)
        return lifecycle_client

    def test_filter_by_generator(self, populated_client: TestClient) -> None:
        """Filter by generator name."""
        response = populated_client.get("/v1/datasets/filter?generator=spiral")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 4
        assert all(d["generator"] == "spiral" for d in data["datasets"])

    def test_filter_by_tags_any(self, populated_client: TestClient) -> None:
        """Filter by tags with any match."""
        response = populated_client.get("/v1/datasets/filter?tags=train&tags_match=any")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    def test_filter_by_tags_all(self, populated_client: TestClient) -> None:
        """Filter by tags with all match."""
        response = populated_client.get("/v1/datasets/filter?tags=train,v1&tags_match=all")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

    def test_filter_by_sample_count(self, populated_client: TestClient) -> None:
        """Filter by sample count range."""
        response = populated_client.get("/v1/datasets/filter?min_samples=250&max_samples=350")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

    def test_filter_with_pagination(self, populated_client: TestClient) -> None:
        """Filter with pagination."""
        response = populated_client.get("/v1/datasets/filter?limit=2&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 4
        assert len(data["datasets"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0


@pytest.mark.integration
class TestBatchDelete:
    """Tests for the batch delete endpoint."""

    def test_batch_delete_existing(self, lifecycle_client: TestClient) -> None:
        """Batch delete existing datasets."""
        ids = []
        for seed in range(3):
            response = lifecycle_client.post("/v1/datasets", json=_create_spiral_request(seed=seed))
            ids.append(response.json()["dataset_id"])

        response = lifecycle_client.post("/v1/datasets/batch-delete", json={"dataset_ids": ids[:2]})

        assert response.status_code == 200
        data = response.json()
        assert len(data["deleted"]) == 2
        assert data["not_found"] == []
        assert data["total_deleted"] == 2

        for deleted_id in ids[:2]:
            get_response = lifecycle_client.get(f"/v1/datasets/{deleted_id}")
            assert get_response.status_code == 404

        get_response = lifecycle_client.get(f"/v1/datasets/{ids[2]}")
        assert get_response.status_code == 200

    def test_batch_delete_mixed(self, lifecycle_client: TestClient) -> None:
        """Batch delete with some nonexistent IDs."""
        response = lifecycle_client.post("/v1/datasets", json=_create_spiral_request(seed=42))
        existing_id = response.json()["dataset_id"]

        response = lifecycle_client.post(
            "/v1/datasets/batch-delete", json={"dataset_ids": [existing_id, "fake-id-1", "fake-id-2"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] == [existing_id]
        assert set(data["not_found"]) == {"fake-id-1", "fake-id-2"}


@pytest.mark.integration
class TestUpdateTags:
    """Tests for the update tags endpoint."""

    def test_add_tags(self, lifecycle_client: TestClient) -> None:
        """Add tags to existing dataset."""
        response = lifecycle_client.post("/v1/datasets", json=_create_spiral_request(tags=["original"]))
        dataset_id = response.json()["dataset_id"]

        response = lifecycle_client.patch(
            f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["new-tag-1", "new-tag-2"]}
        )

        assert response.status_code == 200
        tags = response.json()["tags"]
        assert "original" in tags
        assert "new-tag-1" in tags
        assert "new-tag-2" in tags

    def test_remove_tags(self, lifecycle_client: TestClient) -> None:
        """Remove tags from existing dataset."""
        response = lifecycle_client.post("/v1/datasets", json=_create_spiral_request(tags=["keep", "remove"]))
        dataset_id = response.json()["dataset_id"]

        response = lifecycle_client.patch(f"/v1/datasets/{dataset_id}/tags", json={"remove_tags": ["remove"]})

        assert response.status_code == 200
        tags = response.json()["tags"]
        assert "keep" in tags
        assert "remove" not in tags

    def test_add_and_remove_tags(self, lifecycle_client: TestClient) -> None:
        """Add and remove tags in single request."""
        response = lifecycle_client.post("/v1/datasets", json=_create_spiral_request(tags=["a", "b"]))
        dataset_id = response.json()["dataset_id"]

        response = lifecycle_client.patch(
            f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["c"], "remove_tags": ["a"]}
        )

        assert response.status_code == 200
        tags = response.json()["tags"]
        assert set(tags) == {"b", "c"}

    def test_update_tags_not_found(self, lifecycle_client: TestClient) -> None:
        """Update tags on nonexistent dataset returns 404."""
        response = lifecycle_client.patch("/v1/datasets/nonexistent-id/tags", json={"add_tags": ["test"]})
        assert response.status_code == 404


@pytest.mark.integration
class TestDatasetStats:
    """Tests for the stats endpoint."""

    def test_stats_empty(self, lifecycle_client: TestClient) -> None:
        """Stats for empty store."""
        response = lifecycle_client.get("/v1/datasets/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_datasets"] == 0
        assert data["total_samples"] == 0

    def test_stats_populated(self, lifecycle_client: TestClient) -> None:
        """Stats for populated store."""
        lifecycle_client.post("/v1/datasets", json=_create_spiral_request(n_points=50, seed=1, tags=["train"]))
        lifecycle_client.post("/v1/datasets", json=_create_spiral_request(n_points=100, seed=2, tags=["train", "v2"]))
        lifecycle_client.post("/v1/datasets", json=_create_spiral_request(n_points=150, seed=3, tags=["test"]))

        response = lifecycle_client.get("/v1/datasets/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_datasets"] == 3
        assert data["total_samples"] == 600
        assert data["by_generator"] == {"spiral": 3}
        assert data["by_tag"]["train"] == 2


@pytest.mark.integration
class TestCleanupExpired:
    """Tests for the cleanup-expired endpoint."""

    def test_cleanup_expired_none(self, lifecycle_client: TestClient) -> None:
        """Cleanup with no expired datasets."""
        lifecycle_client.post("/v1/datasets", json=_create_spiral_request(seed=1))
        lifecycle_client.post("/v1/datasets", json=_create_spiral_request(seed=2))

        response = lifecycle_client.post("/v1/datasets/cleanup-expired")

        assert response.status_code == 200
        assert response.json() == []

    def test_cleanup_expired_with_ttl(
        self, lifecycle_client: TestClient, lifecycle_store: InMemoryDatasetStore
    ) -> None:
        """Cleanup datasets with expired TTL requires manipulating store directly."""
        response = lifecycle_client.post("/v1/datasets", json=_create_spiral_request(seed=1, ttl_seconds=3600))
        dataset_id = response.json()["dataset_id"]

        meta = lifecycle_store.get_meta(dataset_id)
        assert meta is not None
        meta.expires_at = datetime.now(UTC) - timedelta(hours=1)
        lifecycle_store.update_meta(dataset_id, meta)

        response = lifecycle_client.post("/v1/datasets/cleanup-expired")

        assert response.status_code == 200
        deleted = response.json()
        assert dataset_id in deleted

        get_response = lifecycle_client.get(f"/v1/datasets/{dataset_id}")
        assert get_response.status_code == 404
