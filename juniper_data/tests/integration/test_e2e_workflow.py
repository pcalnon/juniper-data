"""End-to-End integration tests for the complete JuniperData workflow.

These tests verify the full flow:
1. Start JuniperData service (via TestClient)
2. Create dataset via REST API
3. Download NPZ artifact
4. Verify data integrity (shapes, dtypes, determinism)

Marked with @pytest.mark.slow for weekly CI runs.
"""

import io
from typing import Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture
def e2e_store() -> InMemoryDatasetStore:
    """Create a fresh in-memory store for E2E tests."""
    return InMemoryDatasetStore()


@pytest.fixture
def e2e_settings() -> Settings:
    """Create E2E test settings."""
    return Settings(storage_path="/tmp/juniper_data_e2e_test")


@pytest.fixture
def e2e_client(e2e_store: InMemoryDatasetStore, e2e_settings: Settings) -> TestClient:
    """Create an E2E test client with in-memory storage."""
    app = create_app(settings=e2e_settings)
    datasets.set_store(e2e_store)
    return TestClient(app)


@pytest.mark.integration
@pytest.mark.slow
class TestE2EModernAlgorithm:
    """E2E tests for the modern spiral generation algorithm."""

    @pytest.fixture
    def modern_request(self) -> Dict:
        """Request for modern algorithm spiral dataset."""
        return {
            "generator": "spiral",
            "params": {
                "n_spirals": 2,
                "n_points_per_spiral": 100,
                "seed": 42,
                "algorithm": "modern",
                "noise": 0.1,
                "train_ratio": 0.8,
                "test_ratio": 0.2,
            },
            "persist": True,
        }

    def test_e2e_create_download_verify_modern(self, e2e_client: TestClient, modern_request: Dict) -> None:
        """Complete E2E flow: create dataset, download NPZ, verify integrity."""
        create_response = e2e_client.post("/v1/datasets", json=modern_request)
        assert create_response.status_code == 201
        dataset_id = create_response.json()["dataset_id"]

        artifact_response = e2e_client.get(f"/v1/datasets/{dataset_id}/artifact")
        assert artifact_response.status_code == 200
        assert artifact_response.headers["content-type"] == "application/octet-stream"

        with np.load(io.BytesIO(artifact_response.content)) as data:
            assert "X_train" in data.files
            assert "y_train" in data.files
            assert "X_test" in data.files
            assert "y_test" in data.files
            assert "X_full" in data.files
            assert "y_full" in data.files

            X_train = data["X_train"]
            y_train = data["y_train"]
            X_test = data["X_test"]
            y_test = data["y_test"]
            X_full = data["X_full"]
            y_full = data["y_full"]

            assert X_train.dtype == np.float32
            assert y_train.dtype == np.float32
            assert X_test.dtype == np.float32
            assert y_test.dtype == np.float32
            assert X_full.dtype == np.float32
            assert y_full.dtype == np.float32

            n_total = 2 * 100
            n_train = int(n_total * 0.8)
            n_test = n_total - n_train
            n_spirals = 2

            assert X_train.shape == (n_train, 2)
            assert y_train.shape == (n_train, n_spirals)
            assert X_test.shape == (n_test, 2)
            assert y_test.shape == (n_test, n_spirals)
            assert X_full.shape == (n_total, 2)
            assert y_full.shape == (n_total, n_spirals)

    def test_e2e_deterministic_with_seed(self, e2e_client: TestClient, modern_request: Dict) -> None:
        """Same seed produces identical data (determinism verification)."""
        create_response1 = e2e_client.post("/v1/datasets", json=modern_request)
        dataset_id1 = create_response1.json()["dataset_id"]
        artifact_response1 = e2e_client.get(f"/v1/datasets/{dataset_id1}/artifact")

        modern_request["params"]["seed"] = 42
        create_response2 = e2e_client.post("/v1/datasets", json=modern_request)
        dataset_id2 = create_response2.json()["dataset_id"]
        artifact_response2 = e2e_client.get(f"/v1/datasets/{dataset_id2}/artifact")

        assert dataset_id1 == dataset_id2

        with np.load(io.BytesIO(artifact_response1.content)) as data1:
            with np.load(io.BytesIO(artifact_response2.content)) as data2:
                np.testing.assert_array_equal(data1["X_full"], data2["X_full"])
                np.testing.assert_array_equal(data1["y_full"], data2["y_full"])

    def test_e2e_different_seed_different_data(self, e2e_client: TestClient, modern_request: Dict) -> None:
        """Different seeds produce different data."""
        modern_request["params"]["seed"] = 42
        create_response1 = e2e_client.post("/v1/datasets", json=modern_request)
        dataset_id1 = create_response1.json()["dataset_id"]
        artifact_response1 = e2e_client.get(f"/v1/datasets/{dataset_id1}/artifact")

        modern_request["params"]["seed"] = 123
        create_response2 = e2e_client.post("/v1/datasets", json=modern_request)
        dataset_id2 = create_response2.json()["dataset_id"]
        artifact_response2 = e2e_client.get(f"/v1/datasets/{dataset_id2}/artifact")

        assert dataset_id1 != dataset_id2

        with np.load(io.BytesIO(artifact_response1.content)) as data1:
            with np.load(io.BytesIO(artifact_response2.content)) as data2:
                assert not np.array_equal(data1["X_full"], data2["X_full"])


@pytest.mark.integration
@pytest.mark.slow
class TestE2ELegacyCascorAlgorithm:
    """E2E tests for the legacy_cascor spiral generation algorithm."""

    @pytest.fixture
    def legacy_request(self) -> Dict:
        """Request for legacy_cascor algorithm spiral dataset."""
        return {
            "generator": "spiral",
            "params": {
                "n_spirals": 2,
                "n_points_per_spiral": 100,
                "seed": 42,
                "algorithm": "legacy_cascor",
                "radius": 10.0,
                "origin": [0.0, 0.0],
                "noise": 0.1,
                "train_ratio": 0.8,
                "test_ratio": 0.2,
            },
            "persist": True,
        }

    def test_e2e_create_download_verify_legacy(self, e2e_client: TestClient, legacy_request: Dict) -> None:
        """Complete E2E flow for legacy_cascor algorithm."""
        create_response = e2e_client.post("/v1/datasets", json=legacy_request)
        assert create_response.status_code == 201
        dataset_id = create_response.json()["dataset_id"]

        artifact_response = e2e_client.get(f"/v1/datasets/{dataset_id}/artifact")
        assert artifact_response.status_code == 200

        with np.load(io.BytesIO(artifact_response.content)) as data:
            expected_keys = ["X_train", "y_train", "X_test", "y_test", "X_full", "y_full"]
            for key in expected_keys:
                assert key in data.files, f"Missing key: {key}"

            X_full = data["X_full"]
            y_full = data["y_full"]

            assert X_full.dtype == np.float32
            assert y_full.dtype == np.float32

            n_total = 2 * 100
            assert X_full.shape == (n_total, 2)
            assert y_full.shape == (n_total, 2)

    def test_e2e_legacy_vs_modern_different(self, e2e_client: TestClient) -> None:
        """Legacy and modern algorithms produce different data with same seed."""
        base_params = {
            "n_spirals": 2,
            "n_points_per_spiral": 50,
            "seed": 42,
            "noise": 0.1,
        }

        modern_request = {
            "generator": "spiral",
            "params": {**base_params, "algorithm": "modern"},
            "persist": True,
        }
        legacy_request = {
            "generator": "spiral",
            "params": {**base_params, "algorithm": "legacy_cascor", "radius": 10.0},
            "persist": True,
        }

        modern_response = e2e_client.post("/v1/datasets", json=modern_request)
        legacy_response = e2e_client.post("/v1/datasets", json=legacy_request)

        modern_id = modern_response.json()["dataset_id"]
        legacy_id = legacy_response.json()["dataset_id"]

        assert modern_id != legacy_id

        modern_artifact = e2e_client.get(f"/v1/datasets/{modern_id}/artifact")
        legacy_artifact = e2e_client.get(f"/v1/datasets/{legacy_id}/artifact")

        with np.load(io.BytesIO(modern_artifact.content)) as modern_data:
            with np.load(io.BytesIO(legacy_artifact.content)) as legacy_data:
                assert not np.array_equal(modern_data["X_full"], legacy_data["X_full"])


@pytest.mark.integration
@pytest.mark.slow
class TestE2EDataContract:
    """E2E tests verifying the NPZ data contract for consumers."""

    @pytest.fixture
    def contract_request(self) -> Dict:
        """Standard request for data contract verification."""
        return {
            "generator": "spiral",
            "params": {
                "n_spirals": 2,
                "n_points_per_spiral": 50,
                "seed": 12345,
                "train_ratio": 0.7,
                "test_ratio": 0.3,
            },
            "persist": True,
        }

    def test_e2e_npz_keys_contract(self, e2e_client: TestClient, contract_request: Dict) -> None:
        """Verify NPZ contains exactly the expected keys per data contract."""
        create_response = e2e_client.post("/v1/datasets", json=contract_request)
        dataset_id = create_response.json()["dataset_id"]
        artifact_response = e2e_client.get(f"/v1/datasets/{dataset_id}/artifact")

        with np.load(io.BytesIO(artifact_response.content)) as data:
            expected_keys = {"X_train", "y_train", "X_test", "y_test", "X_full", "y_full"}
            actual_keys = set(data.files)
            assert actual_keys == expected_keys, f"Keys mismatch: expected {expected_keys}, got {actual_keys}"

    def test_e2e_feature_dimensions(self, e2e_client: TestClient, contract_request: Dict) -> None:
        """Verify features have 2 dimensions (x, y coordinates)."""
        create_response = e2e_client.post("/v1/datasets", json=contract_request)
        dataset_id = create_response.json()["dataset_id"]
        artifact_response = e2e_client.get(f"/v1/datasets/{dataset_id}/artifact")

        with np.load(io.BytesIO(artifact_response.content)) as data:
            assert data["X_train"].shape[1] == 2
            assert data["X_test"].shape[1] == 2
            assert data["X_full"].shape[1] == 2

    def test_e2e_one_hot_labels(self, e2e_client: TestClient, contract_request: Dict) -> None:
        """Verify labels are one-hot encoded with correct class count."""
        create_response = e2e_client.post("/v1/datasets", json=contract_request)
        dataset_id = create_response.json()["dataset_id"]
        artifact_response = e2e_client.get(f"/v1/datasets/{dataset_id}/artifact")

        with np.load(io.BytesIO(artifact_response.content)) as data:
            y_full = data["y_full"]
            n_spirals = contract_request["params"]["n_spirals"]

            assert y_full.shape[1] == n_spirals

            row_sums = y_full.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(len(y_full)))

            assert set(np.unique(y_full)) == {0.0, 1.0}

    def test_e2e_train_test_split_ratios(self, e2e_client: TestClient, contract_request: Dict) -> None:
        """Verify train/test split matches requested ratios."""
        create_response = e2e_client.post("/v1/datasets", json=contract_request)
        dataset_id = create_response.json()["dataset_id"]
        artifact_response = e2e_client.get(f"/v1/datasets/{dataset_id}/artifact")

        with np.load(io.BytesIO(artifact_response.content)) as data:
            n_train = len(data["X_train"])
            n_test = len(data["X_test"])
            n_full = len(data["X_full"])

            assert n_train + n_test == n_full

            expected_train_ratio = 0.7
            actual_train_ratio = n_train / n_full
            assert abs(actual_train_ratio - expected_train_ratio) < 0.05

    def test_e2e_metadata_consistency(self, e2e_client: TestClient, contract_request: Dict) -> None:
        """Verify metadata matches actual data dimensions."""
        create_response = e2e_client.post("/v1/datasets", json=contract_request)
        data = create_response.json()
        dataset_id = data["dataset_id"]
        meta = data["meta"]

        artifact_response = e2e_client.get(f"/v1/datasets/{dataset_id}/artifact")

        with np.load(io.BytesIO(artifact_response.content)) as npz_data:
            assert meta["n_samples"] == len(npz_data["X_full"])
            assert meta["n_train"] == len(npz_data["X_train"])
            assert meta["n_test"] == len(npz_data["X_test"])
            assert meta["n_features"] == npz_data["X_full"].shape[1]
            assert meta["n_classes"] == npz_data["y_full"].shape[1]


@pytest.mark.integration
@pytest.mark.slow
class TestE2EErrorHandling:
    """E2E tests for error handling scenarios."""

    def test_e2e_invalid_generator_name(self, e2e_client: TestClient) -> None:
        """Invalid generator name returns error (400 or 404)."""
        request = {
            "generator": "nonexistent_generator",
            "params": {},
            "persist": True,
        }
        response = e2e_client.post("/v1/datasets", json=request)
        assert response.status_code in (400, 404)
        assert "detail" in response.json()

    def test_e2e_invalid_params(self, e2e_client: TestClient) -> None:
        """Invalid parameters return 400/422."""
        request = {
            "generator": "spiral",
            "params": {
                "n_spirals": -1,
                "n_points_per_spiral": 100,
            },
            "persist": True,
        }
        response = e2e_client.post("/v1/datasets", json=request)
        assert response.status_code in (400, 422)

    def test_e2e_nonexistent_dataset_artifact(self, e2e_client: TestClient) -> None:
        """Requesting artifact for nonexistent dataset returns 404."""
        response = e2e_client.get("/v1/datasets/nonexistent-id-12345/artifact")
        assert response.status_code == 404

    def test_e2e_delete_and_verify_gone(self, e2e_client: TestClient) -> None:
        """Deleted dataset cannot be retrieved."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 10, "seed": 1},
            "persist": True,
        }
        create_response = e2e_client.post("/v1/datasets", json=request)
        dataset_id = create_response.json()["dataset_id"]

        get_response = e2e_client.get(f"/v1/datasets/{dataset_id}")
        assert get_response.status_code == 200

        delete_response = e2e_client.delete(f"/v1/datasets/{dataset_id}")
        assert delete_response.status_code == 204

        get_after_delete = e2e_client.get(f"/v1/datasets/{dataset_id}")
        assert get_after_delete.status_code == 404

        artifact_after_delete = e2e_client.get(f"/v1/datasets/{dataset_id}/artifact")
        assert artifact_after_delete.status_code == 404
