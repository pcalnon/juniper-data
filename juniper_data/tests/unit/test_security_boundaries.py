"""Security boundary tests for JuniperData.

Tests for path traversal prevention, input injection, parameter bounds
enforcement, and resource exhaustion protection.

Source: RD-006 (TEST_SUITE_AUDIT_DATA_CLAUDE.md Section 1.8, TEST_SUITE_AUDIT_DATA_AMP_.md)
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.core.models import (
    BatchDeleteRequest,
    CreateDatasetRequest,
    DatasetMeta,
)
from juniper_data.generators.csv_import.params import CsvImportParams
from juniper_data.generators.spiral.params import SpiralParams
from juniper_data.storage import LocalFSDatasetStore
from juniper_data.storage.memory import InMemoryDatasetStore

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_dir():
    """Create a temporary directory for filesystem tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fs_store(temp_dir: Path) -> LocalFSDatasetStore:
    """Create a local filesystem store in a temporary directory."""
    return LocalFSDatasetStore(temp_dir)


@pytest.fixture
def sample_meta() -> DatasetMeta:
    """Create sample dataset metadata for testing."""
    return DatasetMeta(
        dataset_id="test-dataset-001",
        generator="spiral",
        generator_version="1.0.0",
        params={"n_spirals": 2, "n_points_per_spiral": 100},
        n_samples=200,
        n_features=2,
        n_classes=2,
        n_train=160,
        n_test=40,
        class_distribution={"0": 100, "1": 100},
        created_at=datetime(2026, 1, 30, 12, 0, 0),
    )


@pytest.fixture
def sample_arrays() -> dict[str, np.ndarray]:
    """Create sample arrays for testing."""
    rng = np.random.default_rng(42)
    return {
        "X_train": rng.standard_normal((160, 2)).astype(np.float32),
        "y_train": np.eye(2, dtype=np.float32)[rng.integers(0, 2, 160)],
        "X_test": rng.standard_normal((40, 2)).astype(np.float32),
        "y_test": np.eye(2, dtype=np.float32)[rng.integers(0, 2, 40)],
    }


@pytest.fixture
def memory_store() -> InMemoryDatasetStore:
    """Create in-memory store for testing."""
    return InMemoryDatasetStore()


@pytest.fixture
def client(memory_store: InMemoryDatasetStore) -> TestClient:
    """Create a test client with in-memory storage."""
    settings = Settings(storage_path="/tmp/juniper_test")
    app = create_app(settings=settings)
    datasets.set_store(memory_store)
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════════
# TestPathTraversalPrevention
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPathTraversalPrevention:
    """Tests that path traversal attacks via dataset_id are contained."""

    def test_storage_path_with_dotdot_rejected(self, fs_store: LocalFSDatasetStore, temp_dir: Path) -> None:
        """JD-SEC-01: dataset_id with '..' must be rejected before touching the filesystem."""
        malicious_id = "../../../etc/passwd"
        with pytest.raises(ValueError):
            fs_store._meta_path(malicious_id)
        with pytest.raises(ValueError):
            fs_store._npz_path(malicious_id)

    def test_storage_absolute_path_in_dataset_id_rejected(self, fs_store: LocalFSDatasetStore, temp_dir: Path) -> None:
        """JD-SEC-01: absolute-path IDs must be rejected (they would otherwise escape base via Path()/Path())."""
        with pytest.raises(ValueError):
            fs_store._meta_path("/etc/shadow")

    def test_dataset_id_with_null_bytes_rejected(self, fs_store: LocalFSDatasetStore, temp_dir: Path) -> None:
        """JD-SEC-01: null bytes are outside the allowlist and must be rejected up front."""
        with pytest.raises(ValueError):
            fs_store._meta_path("dataset\x00.meta.json")

    def test_api_dataset_id_with_path_traversal(self, client: TestClient) -> None:
        """API endpoints must reject traversal identifiers with a client error.

        FastAPI's path parsing strips ``..`` segments before dispatch, so the
        request typically resolves to a collection URL (404/405) rather than
        reaching the handler. The storage-layer defense now also translates
        any traversal that does reach the handler into an HTTP 400 via the
        ValueError → 400 application exception handler.
        """
        traversal_ids = [
            "../../../etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "dataset/../../../etc/shadow",
        ]
        for malicious_id in traversal_ids:
            response = client.get(f"/v1/datasets/{malicious_id}")
            assert response.status_code in (400, 404, 405, 422), f"Unexpected status for ID '{malicious_id}': {response.status_code}"

    def test_api_artifact_download_with_traversal(self, client: TestClient) -> None:
        """Artifact download must not serve files outside storage via traversal."""
        response = client.get("/v1/datasets/../../etc/passwd/artifact")
        assert response.status_code in (400, 404, 422)

    def test_batch_delete_with_traversal_ids(self, client: TestClient) -> None:
        """Batch delete must classify traversal IDs as not_found without failing the batch (JD-SEC-01)."""
        response = client.post(
            "/v1/datasets/batch-delete",
            json={"dataset_ids": ["../../../etc/passwd", "valid-id"]},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "../../../etc/passwd" in payload.get("not_found", [])


# ═══════════════════════════════════════════════════════════════════════════════
# TestCsvImportPathSecurity
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestCsvImportPathSecurity:
    """Tests for path traversal risks in CSV import file_path parameter."""

    def test_absolute_path_outside_working_dir(self) -> None:
        """CSV import with absolute path to sensitive file should be rejected by path traversal protection."""
        params = CsvImportParams(file_path="/etc/shadow")
        from juniper_data.generators.csv_import.generator import CsvImportGenerator

        # Path traversal protection raises ValueError before file access
        with pytest.raises(ValueError, match="Path traversal detected"):
            CsvImportGenerator.generate(params)

    def test_relative_path_traversal(self) -> None:
        """CSV import with relative traversal path should be rejected by path traversal protection."""
        params = CsvImportParams(file_path="../../../etc/passwd")
        from juniper_data.generators.csv_import.generator import CsvImportGenerator

        with pytest.raises(ValueError, match="Path traversal detected"):
            CsvImportGenerator.generate(params)

    def test_file_path_with_null_bytes(self) -> None:
        """CSV import with null bytes in path should fail."""
        params = CsvImportParams(file_path="/tmp/test\x00malicious.csv")
        from juniper_data.generators.csv_import.generator import CsvImportGenerator

        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            CsvImportGenerator.generate(params)

    def test_csv_import_via_api_with_traversal_path(self) -> None:
        """CSV import through the API with traversal path should be rejected by path traversal protection."""
        settings = Settings(storage_path="/tmp/juniper_test")
        app = create_app(settings=settings)
        datasets.set_store(InMemoryDatasetStore())
        test_client = TestClient(app)

        response = test_client.post(
            "/v1/datasets",
            json={
                "generator": "csv_import",
                "params": {"file_path": "../../../etc/passwd"},
            },
        )
        # Path traversal protection raises ValueError, caught by the app's
        # ValueError handler which returns 400 with a generic message
        assert response.status_code == 400
        # Verify no file contents or internal paths are leaked in the error response
        assert "root:" not in response.text
        assert "etc/passwd" not in response.text


# ═══════════════════════════════════════════════════════════════════════════════
# TestInputBoundaryEnforcement
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestInputBoundaryEnforcement:
    """Tests for Pydantic parameter bound enforcement."""

    def test_spiral_n_points_at_maximum(self) -> None:
        """Spiral generator should accept n_points at maximum boundary."""
        params = SpiralParams(n_points_per_spiral=10000)
        assert params.n_points_per_spiral == 10000

    def test_spiral_n_points_above_maximum(self) -> None:
        """Spiral generator should reject n_points above maximum."""
        with pytest.raises(ValidationError):
            SpiralParams(n_points_per_spiral=10001)

    def test_spiral_n_points_below_minimum(self) -> None:
        """Spiral generator should reject n_points below minimum."""
        with pytest.raises(ValidationError):
            SpiralParams(n_points_per_spiral=9)

    def test_spiral_negative_noise(self) -> None:
        """Spiral generator should reject negative noise values."""
        with pytest.raises(ValidationError):
            SpiralParams(noise=-0.1)

    def test_spiral_train_test_ratio_sum_exceeds_one(self) -> None:
        """Spiral generator should reject train_ratio + test_ratio > 1.0."""
        with pytest.raises(ValidationError):
            SpiralParams(train_ratio=0.8, test_ratio=0.3)

    def test_spiral_n_spirals_at_boundaries(self) -> None:
        """Spiral generator should enforce n_spirals bounds (2-10)."""
        with pytest.raises(ValidationError):
            SpiralParams(n_spirals=1)
        with pytest.raises(ValidationError):
            SpiralParams(n_spirals=11)
        # Boundaries should work
        params_min = SpiralParams(n_spirals=2)
        assert params_min.n_spirals == 2
        params_max = SpiralParams(n_spirals=10)
        assert params_max.n_spirals == 10

    def test_api_extreme_n_points_rejected(self, client: TestClient) -> None:
        """API should reject extreme n_points values via Pydantic validation."""
        response = client.post(
            "/v1/datasets",
            json={
                "generator": "spiral",
                "params": {"n_points_per_spiral": 999999999},
            },
        )
        assert response.status_code == 400
        assert "Invalid parameters" in response.json()["detail"]

    def test_api_negative_parameters_rejected(self, client: TestClient) -> None:
        """API should reject negative parameter values."""
        response = client.post(
            "/v1/datasets",
            json={
                "generator": "spiral",
                "params": {"n_points_per_spiral": -100},
            },
        )
        assert response.status_code == 400

    def test_api_string_in_numeric_field(self, client: TestClient) -> None:
        """API should reject string values in numeric parameter fields."""
        response = client.post(
            "/v1/datasets",
            json={
                "generator": "spiral",
                "params": {"n_points_per_spiral": "DROP TABLE datasets"},
            },
        )
        assert response.status_code == 400

    def test_ttl_seconds_zero_rejected(self) -> None:
        """TTL of zero should be rejected (minimum is 1)."""
        with pytest.raises(ValidationError):
            CreateDatasetRequest(generator="spiral", ttl_seconds=0)

    def test_ttl_seconds_negative_rejected(self) -> None:
        """Negative TTL should be rejected."""
        with pytest.raises(ValidationError):
            CreateDatasetRequest(generator="spiral", ttl_seconds=-1)

    def test_batch_delete_empty_list_rejected(self) -> None:
        """Batch delete with empty list should be rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            BatchDeleteRequest(dataset_ids=[])

    def test_batch_delete_exceeds_max_rejected(self) -> None:
        """Batch delete with >100 IDs should be rejected (max_length=100)."""
        with pytest.raises(ValidationError):
            BatchDeleteRequest(dataset_ids=[f"id-{i}" for i in range(101)])

    def test_batch_delete_at_max_accepted(self) -> None:
        """Batch delete with exactly 100 IDs should be accepted."""
        request = BatchDeleteRequest(dataset_ids=[f"id-{i}" for i in range(100)])
        assert len(request.dataset_ids) == 100

    def test_list_limit_boundaries(self, client: TestClient) -> None:
        """List endpoint should enforce limit bounds (1-1000)."""
        # Below minimum
        response = client.get("/v1/datasets?limit=0")
        assert response.status_code == 422

        # Above maximum
        response = client.get("/v1/datasets?limit=1001")
        assert response.status_code == 422

        # At boundaries
        response = client.get("/v1/datasets?limit=1")
        assert response.status_code == 200
        response = client.get("/v1/datasets?limit=1000")
        assert response.status_code == 200

    def test_list_offset_negative_rejected(self, client: TestClient) -> None:
        """List endpoint should reject negative offset."""
        response = client.get("/v1/datasets?offset=-1")
        assert response.status_code == 422

    def test_preview_n_boundaries(self, client: TestClient) -> None:
        """Preview endpoint should enforce n bounds (1-1000)."""
        # Below minimum
        response = client.get("/v1/datasets/some-id/preview?n=0")
        assert response.status_code == 422

        # Above maximum
        response = client.get("/v1/datasets/some-id/preview?n=1001")
        assert response.status_code == 422

    def test_filter_tags_match_pattern_enforcement(self, client: TestClient) -> None:
        """Filter endpoint should only accept 'any' or 'all' for tags_match."""
        response = client.get("/v1/datasets/filter?tags_match=invalid")
        assert response.status_code == 422

        response = client.get("/v1/datasets/filter?tags_match=any")
        assert response.status_code == 200
        response = client.get("/v1/datasets/filter?tags_match=all")
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# TestResourceExhaustion
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestResourceExhaustion:
    """Tests for resource exhaustion protection."""

    def test_very_large_n_points_rejected_by_pydantic(self) -> None:
        """Generators should reject unreasonably large point counts."""
        with pytest.raises(ValidationError):
            SpiralParams(n_points_per_spiral=10001)

    def test_api_rejects_very_large_dataset_request(self, client: TestClient) -> None:
        """API should reject dataset generation requests with extreme parameters."""
        response = client.post(
            "/v1/datasets",
            json={
                "generator": "spiral",
                "params": {"n_spirals": 10, "n_points_per_spiral": 10001},
            },
        )
        assert response.status_code == 400

    def test_batch_delete_max_enforcement(self, client: TestClient) -> None:
        """Batch delete should enforce maximum of 100 IDs per request."""
        response = client.post(
            "/v1/datasets/batch-delete",
            json={"dataset_ids": [f"id-{i}" for i in range(101)]},
        )
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# TestAPIBoundaries
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestAPIBoundaries:
    """Tests for API-level input handling and malformed request resilience."""

    def test_malformed_json_body(self, client: TestClient) -> None:
        """API should return 422 for malformed JSON in request body."""
        response = client.post(
            "/v1/datasets",
            content=b"not valid json{{{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_field(self, client: TestClient) -> None:
        """API should return 422 when required 'generator' field is missing."""
        response = client.post("/v1/datasets", json={"params": {}})
        assert response.status_code == 422

    def test_wrong_type_for_generator(self, client: TestClient) -> None:
        """API should return 422 when generator is not a string."""
        response = client.post("/v1/datasets", json={"generator": 12345})
        # FastAPI will attempt coercion; integer may be cast to string
        # The key check is that it doesn't crash
        assert response.status_code in (201, 400, 422)

    def test_extra_fields_ignored(self, client: TestClient) -> None:
        """API should handle unexpected extra fields gracefully."""
        response = client.post(
            "/v1/datasets",
            json={
                "generator": "spiral",
                "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
                "persist": False,
                "evil_field": "<script>alert('xss')</script>",
                "__proto__": {"admin": True},
            },
        )
        # Extra fields should be ignored, request should succeed
        assert response.status_code == 201

    def test_dataset_id_special_characters(self, client: TestClient) -> None:
        """API should handle dataset IDs with special characters without crashing."""
        special_ids = [
            "id with spaces",
            "id<script>alert(1)</script>",
            "id'; DROP TABLE datasets;--",
            "a" * 1000,  # very long ID
        ]
        for special_id in special_ids:
            response = client.get(f"/v1/datasets/{special_id}")
            # Should return 404 (not found), not 500
            assert response.status_code in (404, 422), f"Unexpected status for ID '{special_id[:50]}': {response.status_code}"

    def test_dataset_id_non_printable_characters(self) -> None:
        """Dataset IDs with non-printable characters are rejected at HTTP level."""
        import httpx

        # Non-printable ASCII characters (tabs, newlines) are invalid in URLs
        # and rejected by the HTTP client before reaching the API
        with pytest.raises(httpx.InvalidURL):
            httpx.URL("http://localhost/v1/datasets/id\t\n\r")

    def test_tags_with_special_characters(self, client: TestClient) -> None:
        """Dataset creation with special characters in tags should not crash."""
        response = client.post(
            "/v1/datasets",
            json={
                "generator": "spiral",
                "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 42},
                "persist": False,
                "tags": [
                    "normal-tag",
                    "<script>alert('xss')</script>",
                    "'; DROP TABLE datasets;--",
                    "a" * 500,
                ],
            },
        )
        # Tags are stored as-is (no injection risk in JSON/Pydantic models)
        assert response.status_code == 201

    def test_empty_body_rejected(self, client: TestClient) -> None:
        """API should return 422 for empty body on POST."""
        response = client.post(
            "/v1/datasets",
            content=b"",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_content_type_mismatch(self, client: TestClient) -> None:
        """API should handle wrong content type gracefully."""
        response = client.post(
            "/v1/datasets",
            content=b"generator=spiral",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 422

    def test_generator_name_injection(self, client: TestClient) -> None:
        """Generator name with injection payloads should return 400."""
        injection_names = [
            "'; DROP TABLE generators;--",
            "../generators/spiral",
            "__import__('os').system('rm -rf /')",
        ]
        for name in injection_names:
            response = client.post(
                "/v1/datasets",
                json={"generator": name, "params": {}},
            )
            assert response.status_code == 400
            assert "Unknown generator" in response.json()["detail"]
