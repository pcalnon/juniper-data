"""Unit tests for API route modules."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Create test settings backed by an existing tmp directory.

    Using tmp_path ensures the storage directory actually exists so that
    R1.2 readiness probe (503 when storage missing) does not fail the
    legacy health-endpoint smoke tests.
    """
    storage = tmp_path / "juniper_data_storage"
    storage.mkdir()
    return Settings(storage_path=str(storage))


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

    def test_preview_stacks_train_test_when_no_full_arrays(self, memory_store: InMemoryDatasetStore, test_settings: Settings) -> None:
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
class TestGeneratorAvailability:
    """D1 (I-5): generator availability surfaced — registry/schema ``available`` flag + the 501 capability seam."""

    def test_list_generators_available_flag_present_for_all(self, client: TestClient) -> None:
        """Every registry listing entry carries a boolean ``available`` flag."""
        response = client.get("/v1/generators")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        for entry in data:
            assert "available" in entry, f"generator '{entry['name']}' is missing the available flag"
            assert isinstance(entry["available"], bool)

    def test_list_generators_synthetics_available(self, client: TestClient) -> None:
        """Numpy-only synthetic generators are always available."""
        response = client.get("/v1/generators")

        by_name = {g["name"]: g for g in response.json()}
        for name in ("spiral", "xor", "gaussian", "circles", "moon", "checkerboard", "multi_sine", "mackey_glass", "ar_p", "irregular_sine", "delay_product"):
            assert by_name[name]["available"] is True, f"synthetic generator '{name}' should be available"

    def test_list_generators_capability_gated_flags_match_modules(self, client: TestClient) -> None:
        """mnist/equities/equities_seq report their real module-level dependency availability."""
        from juniper_data.generators.equities.generator import EQUITIES_DEPS_AVAILABLE
        from juniper_data.generators.mnist.generator import HF_AVAILABLE

        response = client.get("/v1/generators")

        by_name = {g["name"]: g for g in response.json()}
        assert by_name["mnist"]["available"] is HF_AVAILABLE
        assert by_name["equities"]["available"] is EQUITIES_DEPS_AVAILABLE
        assert by_name["equities_seq"]["available"] is EQUITIES_DEPS_AVAILABLE

    def test_list_generators_mnist_unavailable_when_hf_missing(self, client: TestClient) -> None:
        """mnist reports available=False in the listing when the HF datasets package is absent."""
        with patch("juniper_data.generators.mnist.generator.HF_AVAILABLE", False):
            response = client.get("/v1/generators")

        assert response.status_code == 200
        by_name = {g["name"]: g for g in response.json()}
        assert by_name["mnist"]["available"] is False

    def test_get_schema_carries_available_flag(self, client: TestClient) -> None:
        """The per-generator schema response carries a top-level ``available`` flag alongside the intact schema."""
        response = client.get("/v1/generators/spiral/schema")

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert "properties" in data
        assert "n_spirals" in data["properties"]

    def test_get_schema_mnist_unavailable_when_hf_missing(self, client: TestClient) -> None:
        """The mnist schema response reports available=False without HF datasets."""
        with patch("juniper_data.generators.mnist.generator.HF_AVAILABLE", False):
            response = client.get("/v1/generators/mnist/schema")

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
        assert "properties" in data

    def test_create_dataset_unavailable_generator_returns_501_with_hint(self, client: TestClient) -> None:
        """POST /v1/datasets for a generator missing its optional dependency returns 501 + the install hint, never a masked 500."""
        request = {"generator": "mnist", "params": {}, "persist": False}
        with patch("juniper_data.generators.mnist.generator.HF_AVAILABLE", False):
            response = client.post("/v1/datasets", json=request)

        assert response.status_code == 501
        detail = response.json()["detail"]
        assert "mnist" in detail
        assert "pip install datasets" in detail

    # ------------------------------------------------------------------
    # APD-DATA-004: the 501 detail must not echo an UNDECLARED ImportError.
    #
    # The register filed this against batch-create's ``except HTTPException``
    # branch, which copies ``e.detail`` verbatim beside a sibling branch that
    # redacts. It is verified here on BOTH paths, because single-create returns
    # the same detail to the caller directly -- batch-create amplifies the
    # leak, it is not the source of it. The source is the 501 construction.
    # ------------------------------------------------------------------

    #: Shaped like a real broken-native-extension ImportError: it carries an
    #: absolute filesystem path, unlike the generators' curated install hints.
    LEAKY_IMPORT_ERROR = "libcudart.so.11.0: cannot open shared object file at /opt/miniforge3/envs/JuniperData/lib"

    @staticmethod
    def _available_but_broken_entry() -> dict:
        """A registry entry whose generator reports itself AVAILABLE and still raises ImportError."""

        class AvailableButBroken:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def generate(params):
                raise ImportError(TestGeneratorAvailability.LEAKY_IMPORT_ERROR)

        return {**datasets.GENERATOR_REGISTRY["mnist"], "generator": AvailableButBroken}

    def test_undeclared_importerror_is_not_echoed_by_create(self, client: TestClient) -> None:
        """A generator reporting itself available has failed for a reason the caller must not see."""
        request = {"generator": "mnist", "params": {}, "persist": False}

        with patch.dict(datasets.GENERATOR_REGISTRY, {"mnist": self._available_but_broken_entry()}):
            response = client.post("/v1/datasets", json=request)

        assert response.status_code == 501
        assert "libcudart" not in response.text
        assert "/opt/" not in response.text
        detail = response.json()["detail"]
        assert "mnist" in detail
        assert "ref: " in detail

    def test_undeclared_importerror_is_not_echoed_through_batch_create(self, client: TestClient) -> None:
        """The same leak through the branch the register named: batch-create copies ``e.detail`` verbatim."""
        request = {"generator": "mnist", "params": {}, "persist": False}

        with patch.dict(datasets.GENERATOR_REGISTRY, {"mnist": self._available_but_broken_entry()}):
            response = client.post("/v1/datasets/batch-create", json={"datasets": [request]})

        assert response.status_code == 201
        assert "libcudart" not in response.text
        assert "/opt/" not in response.text
        assert "ref: " in response.json()["results"][0]["error"]

    def test_declared_missing_dependency_keeps_the_install_hint_through_batch_create(self, client: TestClient) -> None:
        """Negative control: redaction must not swallow the D1 hint for a genuine capability gap.

        Without this, a fix that simply redacted every ImportError would pass
        the two tests above while destroying the behaviour D1 was added for.
        """
        request = {"generator": "mnist", "params": {}, "persist": False}

        with patch("juniper_data.generators.mnist.generator.HF_AVAILABLE", False):
            response = client.post("/v1/datasets/batch-create", json={"datasets": [request]})

        assert response.status_code == 201
        error = response.json()["results"][0]["error"]
        assert "pip install datasets" in error
        assert "ref: " not in error


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

    def test_health_includes_service_identifier(self, client: TestClient) -> None:
        """API-02: health endpoint includes the ``service`` field naming this service.

        Part of the shared ``{status, version, service}`` base schema across
        juniper-data, juniper-cascor, and juniper-canopy so cross-service
        monitoring tools can tell health responses apart without parsing
        the URL.
        """
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "juniper-data"

    def test_liveness_probe(self, client: TestClient) -> None:
        """Test liveness probe returns alive status."""
        response = client.get("/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_probe(self, client: TestClient) -> None:
        """Test readiness probe returns ReadinessResponse with version."""
        response = client.get("/v1/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ready", "degraded")
        assert "version" in data
        assert data["service"] == "juniper-data"
        assert "dependencies" in data


def _create_spiral(client: TestClient, seed: int) -> str:
    """Create a persisted spiral dataset via the API and return its ID."""
    request = {"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 40, "seed": seed}, "persist": True}
    resp = client.post("/v1/datasets", json=request)
    assert resp.status_code == 201
    return resp.json()["dataset_id"]


@pytest.mark.unit
class TestBatchEndpoints:
    """Cover the batch endpoints' error/edge branches in the unit lane.

    ``juniper_data/tests/api/test_batch_operations.py`` exercises these routes,
    but it lives under ``tests/api`` (``@pytest.mark.api``) which the CI
    ``unit and not slow`` + ``juniper_data/tests/unit`` scope does not collect,
    so the batch error paths were invisible to the per-file coverage gate. These
    unit tests use the in-memory-store TestClient (no external services).
    """

    def test_batch_create_partial_success_on_http_error(self, client: TestClient) -> None:
        """A bad-generator item is reported via the ``except HTTPException`` branch."""
        request = {
            "datasets": [
                {"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 40, "seed": 1}, "persist": True},
                {"generator": "does_not_exist", "params": {}, "persist": True},
            ]
        }
        response = client.post("/v1/datasets/batch-create", json=request)
        assert response.status_code == 201
        data = response.json()
        assert data["total_created"] == 1
        assert data["total_failed"] == 1
        failed = [item for item in data["results"] if not item["success"]]
        assert len(failed) == 1
        assert failed[0]["error"]

    def test_batch_create_opaque_error_ref_on_unexpected_failure(self, client: TestClient, memory_store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-HTTP store failure hits the ``except Exception`` branch (ERR-08 opaque ref)."""

        def _boom(*_args, **_kwargs):
            raise RuntimeError("disk gone")

        monkeypatch.setattr(memory_store, "save_versioned", _boom)
        request = {"datasets": [{"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 40, "seed": 2}, "persist": True}]}
        response = client.post("/v1/datasets/batch-create", json=request)
        assert response.status_code == 201
        data = response.json()
        assert data["total_created"] == 0
        assert data["total_failed"] == 1
        error = data["results"][0]["error"]
        assert "Dataset creation failed (ref:" in error
        assert "disk gone" not in error  # raw exception detail must never leak

    def test_batch_update_tags_updates_and_reports_not_found(self, client: TestClient) -> None:
        """batch-tags updates existing datasets and lists unknown IDs as not_found."""
        id1 = _create_spiral(client, seed=3)
        id2 = _create_spiral(client, seed=4)
        request = {"dataset_ids": [id1, id2, "missing-id"], "add_tags": ["alpha", "beta"], "remove_tags": []}
        response = client.patch("/v1/datasets/batch-tags", json=request)
        assert response.status_code == 200
        data = response.json()
        assert set(data["updated"]) == {id1, id2}
        assert data["not_found"] == ["missing-id"]
        assert data["total_updated"] == 2

    def test_batch_export_skips_raced_deletion(self, client: TestClient, memory_store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
        """An ID that exists but whose artifact is None is skipped mid-stream (raced delete)."""
        id1 = _create_spiral(client, seed=5)
        monkeypatch.setattr(memory_store, "get_artifact_bytes", lambda _dataset_id: None)
        response = client.post("/v1/datasets/batch-export", json={"dataset_ids": [id1]})
        # The raced item is skipped; the endpoint still streams a (well-formed) archive.
        assert response.status_code == 200
