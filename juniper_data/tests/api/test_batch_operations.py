"""Tests for batch dataset operations.

Covers the three batch endpoints:
- POST /v1/datasets/batch-create
- PATCH /v1/datasets/batch-tags
- POST /v1/datasets/batch-export
"""

import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(storage_path="/tmp/juniper_test_batch")


@pytest.fixture
def memory_store() -> InMemoryDatasetStore:
    """Create a fresh in-memory store for each test."""
    return InMemoryDatasetStore()


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
        "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
        "persist": True,
    }


def _create_spiral(client: TestClient, seed: int, tags: list[str] | None = None) -> str:
    """Helper: create a persisted spiral dataset and return its ID."""
    request: dict = {
        "generator": "spiral",
        "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": seed},
        "persist": True,
    }
    if tags is not None:
        request["tags"] = tags
    resp = client.post("/v1/datasets", json=request)
    assert resp.status_code == 201
    return resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# POST /v1/datasets/batch-create
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestBatchCreate:
    """Tests for POST /v1/datasets/batch-create."""

    def test_create_two_spirals_successfully(self, client: TestClient) -> None:
        """Creating two valid spiral datasets returns 201 with both succeeding."""
        request = {
            "datasets": [
                {
                    "generator": "spiral",
                    "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 1},
                    "persist": True,
                },
                {
                    "generator": "spiral",
                    "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 2},
                    "persist": True,
                },
            ]
        }
        response = client.post("/v1/datasets/batch-create", json=request)

        assert response.status_code == 201
        data = response.json()
        assert data["total_created"] == 2
        assert data["total_failed"] == 0
        assert len(data["results"]) == 2

        for idx, result in enumerate(data["results"]):
            assert result["index"] == idx
            assert result["success"] is True
            assert result["dataset_id"] is not None
            assert result["generator"] == "spiral"
            assert result["artifact_url"] is not None
            assert result["error"] is None

    def test_partial_success_with_invalid_generator(self, client: TestClient) -> None:
        """Mix of valid and invalid generators produces partial success."""
        request = {
            "datasets": [
                {
                    "generator": "spiral",
                    "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 10},
                    "persist": True,
                },
                {
                    "generator": "nonexistent_gen",
                    "params": {},
                    "persist": True,
                },
                {
                    "generator": "spiral",
                    "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 11},
                    "persist": True,
                },
            ]
        }
        response = client.post("/v1/datasets/batch-create", json=request)

        assert response.status_code == 201
        data = response.json()
        assert data["total_created"] == 2
        assert data["total_failed"] == 1

        # First item succeeds
        assert data["results"][0]["success"] is True
        assert data["results"][0]["index"] == 0

        # Second item fails
        assert data["results"][1]["success"] is False
        assert data["results"][1]["index"] == 1
        assert data["results"][1]["error"] is not None
        assert "Unknown generator" in data["results"][1]["error"]
        assert data["results"][1]["dataset_id"] is None

        # Third item succeeds
        assert data["results"][2]["success"] is True
        assert data["results"][2]["index"] == 2

    def test_empty_datasets_list_returns_422(self, client: TestClient) -> None:
        """Empty datasets list triggers Pydantic validation error (422)."""
        request: dict = {"datasets": []}
        response = client.post("/v1/datasets/batch-create", json=request)

        assert response.status_code == 422

    def test_result_fields_present(self, client: TestClient) -> None:
        """Each result item has index, dataset_id, generator, success fields."""
        request = {
            "datasets": [
                {
                    "generator": "spiral",
                    "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 20},
                    "persist": True,
                }
            ]
        }
        response = client.post("/v1/datasets/batch-create", json=request)

        assert response.status_code == 201
        result = response.json()["results"][0]
        assert "index" in result
        assert "dataset_id" in result
        assert "generator" in result
        assert "success" in result

    def test_counts_correct_on_all_failures(self, client: TestClient) -> None:
        """When all items fail, total_created is 0 and total_failed matches count."""
        request = {
            "datasets": [
                {"generator": "bad_gen_1", "params": {}, "persist": True},
                {"generator": "bad_gen_2", "params": {}, "persist": True},
            ]
        }
        response = client.post("/v1/datasets/batch-create", json=request)

        assert response.status_code == 201
        data = response.json()
        assert data["total_created"] == 0
        assert data["total_failed"] == 2

    def test_batch_create_with_tags(self, client: TestClient) -> None:
        """Batch-created datasets respect tags in each item."""
        request = {
            "datasets": [
                {
                    "generator": "spiral",
                    "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 30},
                    "persist": True,
                    "tags": ["batch-test", "spiral"],
                }
            ]
        }
        response = client.post("/v1/datasets/batch-create", json=request)

        assert response.status_code == 201
        dataset_id = response.json()["results"][0]["dataset_id"]

        meta_resp = client.get(f"/v1/datasets/{dataset_id}")
        assert meta_resp.status_code == 200
        assert "batch-test" in meta_resp.json()["tags"]

    def test_batch_create_invalid_params(self, client: TestClient) -> None:
        """Invalid generator params for one item produces failure for that item only."""
        request = {
            "datasets": [
                {
                    "generator": "spiral",
                    "params": {"n_spirals": "not_a_number"},
                    "persist": True,
                },
                {
                    "generator": "spiral",
                    "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 40},
                    "persist": True,
                },
            ]
        }
        response = client.post("/v1/datasets/batch-create", json=request)

        assert response.status_code == 201
        data = response.json()
        assert data["total_created"] == 1
        assert data["total_failed"] == 1
        assert data["results"][0]["success"] is False
        assert data["results"][1]["success"] is True


# ---------------------------------------------------------------------------
# PATCH /v1/datasets/batch-tags
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestBatchUpdateTags:
    """Tests for PATCH /v1/datasets/batch-tags."""

    def test_add_tags_to_existing_datasets(self, client: TestClient) -> None:
        """Adding tags to existing datasets succeeds."""
        id1 = _create_spiral(client, seed=100)
        id2 = _create_spiral(client, seed=101)

        response = client.patch(
            "/v1/datasets/batch-tags",
            json={"dataset_ids": [id1, id2], "add_tags": ["new-tag", "experiment"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_updated"] == 2
        assert set(data["updated"]) == {id1, id2}
        assert data["not_found"] == []

        # Verify tags were actually applied
        meta1 = client.get(f"/v1/datasets/{id1}").json()
        meta2 = client.get(f"/v1/datasets/{id2}").json()
        assert "new-tag" in meta1["tags"]
        assert "experiment" in meta1["tags"]
        assert "new-tag" in meta2["tags"]
        assert "experiment" in meta2["tags"]

    def test_remove_tags_from_existing_datasets(self, client: TestClient) -> None:
        """Removing tags from existing datasets succeeds."""
        id1 = _create_spiral(client, seed=110, tags=["keep-me", "remove-me"])
        id2 = _create_spiral(client, seed=111, tags=["keep-me", "remove-me", "also-keep"])

        response = client.patch(
            "/v1/datasets/batch-tags",
            json={"dataset_ids": [id1, id2], "remove_tags": ["remove-me"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_updated"] == 2

        meta1 = client.get(f"/v1/datasets/{id1}").json()
        meta2 = client.get(f"/v1/datasets/{id2}").json()
        assert "remove-me" not in meta1["tags"]
        assert "keep-me" in meta1["tags"]
        assert "remove-me" not in meta2["tags"]
        assert "also-keep" in meta2["tags"]

    def test_mix_existing_and_nonexisting_ids(self, client: TestClient) -> None:
        """Mix of existing and non-existing IDs reports both updated and not_found."""
        id1 = _create_spiral(client, seed=120)

        response = client.patch(
            "/v1/datasets/batch-tags",
            json={
                "dataset_ids": [id1, "nonexistent-dataset-id"],
                "add_tags": ["tagged"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_updated"] == 1
        assert id1 in data["updated"]
        assert "nonexistent-dataset-id" in data["not_found"]

    def test_add_and_remove_tags_simultaneously(self, client: TestClient) -> None:
        """Adding and removing tags in one request works correctly."""
        id1 = _create_spiral(client, seed=130, tags=["old-tag"])

        response = client.patch(
            "/v1/datasets/batch-tags",
            json={
                "dataset_ids": [id1],
                "add_tags": ["new-tag"],
                "remove_tags": ["old-tag"],
            },
        )

        assert response.status_code == 200
        meta = client.get(f"/v1/datasets/{id1}").json()
        assert "new-tag" in meta["tags"]
        assert "old-tag" not in meta["tags"]

    def test_all_nonexisting_ids(self, client: TestClient) -> None:
        """All non-existing dataset IDs results in total_updated=0."""
        response = client.patch(
            "/v1/datasets/batch-tags",
            json={
                "dataset_ids": ["fake-id-1", "fake-id-2"],
                "add_tags": ["tag"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_updated"] == 0
        assert len(data["not_found"]) == 2

    def test_empty_dataset_ids_returns_422(self, client: TestClient) -> None:
        """Empty dataset_ids list triggers validation error (422)."""
        response = client.patch(
            "/v1/datasets/batch-tags",
            json={"dataset_ids": [], "add_tags": ["tag"]},
        )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/datasets/batch-export
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestBatchExport:
    """Tests for POST /v1/datasets/batch-export."""

    def test_export_two_existing_datasets(self, client: TestClient) -> None:
        """Exporting two existing datasets returns a valid ZIP with NPZ files."""
        id1 = _create_spiral(client, seed=200)
        id2 = _create_spiral(client, seed=201)

        response = client.post(
            "/v1/datasets/batch-export",
            json={"dataset_ids": [id1, id2]},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            names = zf.namelist()
            assert f"{id1}.npz" in names
            assert f"{id2}.npz" in names
            assert len(names) == 2

    def test_zip_contains_valid_npz_files(self, client: TestClient) -> None:
        """Each file inside the ZIP is a valid NPZ with expected array keys."""
        dataset_id = _create_spiral(client, seed=210)

        response = client.post(
            "/v1/datasets/batch-export",
            json={"dataset_ids": [dataset_id]},
        )

        assert response.status_code == 200

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            npz_bytes = zf.read(f"{dataset_id}.npz")
            with np.load(io.BytesIO(npz_bytes)) as data:
                assert "X_train" in data.files
                assert "y_train" in data.files
                assert "X_test" in data.files
                assert "y_test" in data.files

    def test_all_nonexisting_datasets_returns_404(self, client: TestClient) -> None:
        """When none of the requested datasets exist, returns 404."""
        response = client.post(
            "/v1/datasets/batch-export",
            json={"dataset_ids": ["nonexistent-1", "nonexistent-2"]},
        )

        assert response.status_code == 404
        data = response.json()
        assert "none of the requested datasets were found" in data["detail"].lower()

    def test_mix_existing_and_nonexisting_returns_zip_with_found_only(self, client: TestClient) -> None:
        """Mix of existing and non-existing IDs returns ZIP with only found datasets."""
        dataset_id = _create_spiral(client, seed=220)

        response = client.post(
            "/v1/datasets/batch-export",
            json={"dataset_ids": [dataset_id, "nonexistent-id"]},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            names = zf.namelist()
            assert f"{dataset_id}.npz" in names
            assert len(names) == 1

    def test_empty_dataset_ids_returns_422(self, client: TestClient) -> None:
        """Empty dataset_ids list triggers validation error (422)."""
        response = client.post(
            "/v1/datasets/batch-export",
            json={"dataset_ids": []},
        )

        assert response.status_code == 422

    def test_export_content_disposition_header(self, client: TestClient) -> None:
        """Response includes Content-Disposition header for download."""
        dataset_id = _create_spiral(client, seed=230)

        response = client.post(
            "/v1/datasets/batch-export",
            json={"dataset_ids": [dataset_id]},
        )

        assert response.status_code == 200
        assert "content-disposition" in response.headers
        assert "datasets.zip" in response.headers["content-disposition"]
