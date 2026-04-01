"""Unit tests for dataset versioning (CAN-DEF-005).

Tests cover versioning fields on models, auto-increment logic,
storage versioning methods, filter extensions, and API endpoints.
"""

import threading
from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.core.models import DatasetMeta
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(storage_path="/tmp/juniper_test_versioning")


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
def sample_arrays() -> dict[str, np.ndarray]:
    """Create sample arrays for testing."""
    return {
        "X_train": np.random.randn(16, 2).astype(np.float32),
        "y_train": np.eye(2, dtype=np.float32)[np.random.randint(0, 2, 16)],
        "X_test": np.random.randn(4, 2).astype(np.float32),
        "y_test": np.eye(2, dtype=np.float32)[np.random.randint(0, 2, 4)],
    }


def _make_meta(
    dataset_id: str,
    dataset_name: str | None = None,
    dataset_version: int | None = None,
    parent_dataset_id: str | None = None,
    description: str | None = None,
    created_by: str | None = None,
    checksum: str | None = None,
) -> DatasetMeta:
    """Helper to create a DatasetMeta with versioning fields."""
    return DatasetMeta(
        dataset_id=dataset_id,
        generator="spiral",
        generator_version="1.0.0",
        params={"n_spirals": 2},
        n_samples=20,
        n_features=2,
        n_classes=2,
        n_train=16,
        n_test=4,
        class_distribution={"0": 10, "1": 10},
        created_at=datetime.now(UTC),
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        parent_dataset_id=parent_dataset_id,
        description=description,
        created_by=created_by,
        checksum=checksum,
    )


def _create_named_spiral(client: TestClient, seed: int, name: str, **kwargs) -> dict:
    """Helper: create a persisted named spiral dataset and return full response JSON."""
    request: dict = {
        "generator": "spiral",
        "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": seed},
        "persist": True,
        "name": name,
        **kwargs,
    }
    resp = client.post("/v1/datasets", json=request)
    assert resp.status_code == 201
    return resp.json()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetMetaVersioningFields:
    """Tests for versioning fields on DatasetMeta."""

    def test_versioning_fields_default_to_none(self) -> None:
        """All versioning fields default to None for backward compat."""
        meta = _make_meta("ds-001")
        assert meta.dataset_name is None
        assert meta.dataset_version is None
        assert meta.parent_dataset_id is None
        assert meta.description is None
        assert meta.created_by is None

    def test_versioning_fields_set_correctly(self) -> None:
        """Versioning fields can be set on DatasetMeta."""
        meta = _make_meta(
            "ds-002",
            dataset_name="my-dataset",
            dataset_version=3,
            parent_dataset_id="ds-001",
            description="A test dataset",
            created_by="test-user",
        )
        assert meta.dataset_name == "my-dataset"
        assert meta.dataset_version == 3
        assert meta.parent_dataset_id == "ds-001"
        assert meta.description == "A test dataset"
        assert meta.created_by == "test-user"

    def test_backward_compat_existing_datasets_without_name(self) -> None:
        """Existing datasets without versioning fields still work."""
        meta = DatasetMeta(
            dataset_id="legacy-001",
            generator="spiral",
            generator_version="1.0.0",
            params={"n_spirals": 2},
            n_samples=100,
            n_features=2,
            n_classes=2,
            n_train=80,
            n_test=20,
            class_distribution={"0": 50, "1": 50},
            created_at=datetime.now(UTC),
        )
        assert meta.dataset_name is None
        assert meta.dataset_version is None
        assert meta.generator == "spiral"


# ---------------------------------------------------------------------------
# Storage versioning methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStorageVersioningMethods:
    """Tests for versioning methods on DatasetStore."""

    def test_list_versions_returns_sorted(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """list_versions returns all versions sorted by version ascending."""
        # Save versions out of order
        meta_v2 = _make_meta("ds-v2", dataset_name="experiment", dataset_version=2)
        meta_v1 = _make_meta("ds-v1", dataset_name="experiment", dataset_version=1)
        meta_v3 = _make_meta("ds-v3", dataset_name="experiment", dataset_version=3)
        memory_store.save("ds-v2", meta_v2, sample_arrays)
        memory_store.save("ds-v1", meta_v1, sample_arrays)
        memory_store.save("ds-v3", meta_v3, sample_arrays)

        versions = memory_store.list_versions("experiment")
        assert len(versions) == 3
        assert versions[0].dataset_version == 1
        assert versions[1].dataset_version == 2
        assert versions[2].dataset_version == 3

    def test_list_versions_empty_for_unknown(self, memory_store: InMemoryDatasetStore) -> None:
        """list_versions returns empty list for unknown name."""
        assert memory_store.list_versions("nonexistent") == []

    def test_list_versions_excludes_other_names(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """list_versions only returns datasets with the matching name."""
        meta_a = _make_meta("ds-a", dataset_name="alpha", dataset_version=1)
        meta_b = _make_meta("ds-b", dataset_name="beta", dataset_version=1)
        memory_store.save("ds-a", meta_a, sample_arrays)
        memory_store.save("ds-b", meta_b, sample_arrays)

        versions = memory_store.list_versions("alpha")
        assert len(versions) == 1
        assert versions[0].dataset_name == "alpha"

    def test_get_latest_version_returns_highest(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """get_latest_version returns the dataset with highest version."""
        for v in [1, 2, 3]:
            meta = _make_meta(f"ds-v{v}", dataset_name="experiment", dataset_version=v)
            memory_store.save(f"ds-v{v}", meta, sample_arrays)

        latest = memory_store.get_latest_version("experiment")
        assert latest is not None
        assert latest.dataset_version == 3

    def test_get_latest_version_returns_none_for_unknown(self, memory_store: InMemoryDatasetStore) -> None:
        """get_latest_version returns None for unknown name."""
        assert memory_store.get_latest_version("nonexistent") is None

    def test_next_version_number_starts_at_1(self, memory_store: InMemoryDatasetStore) -> None:
        """next_version_number returns 1 for a new name."""
        assert memory_store.next_version_number("new-dataset") == 1

    def test_next_version_number_increments(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """next_version_number returns max version + 1."""
        for v in [1, 2, 3]:
            meta = _make_meta(f"ds-v{v}", dataset_name="experiment", dataset_version=v)
            memory_store.save(f"ds-v{v}", meta, sample_arrays)

        assert memory_store.next_version_number("experiment") == 4

    def test_next_version_number_ignores_legacy_none_versions(
        self,
        memory_store: InMemoryDatasetStore,
        sample_arrays: dict[str, np.ndarray],
    ) -> None:
        """Legacy entries without a numeric version do not block incrementing."""
        legacy_meta = _make_meta("ds-legacy", dataset_name="experiment", dataset_version=None)
        v2_meta = _make_meta("ds-v2", dataset_name="experiment", dataset_version=2)
        memory_store.save("ds-legacy", legacy_meta, sample_arrays)
        memory_store.save("ds-v2", v2_meta, sample_arrays)

        assert memory_store.next_version_number("experiment") == 3

    def test_filter_by_dataset_name(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """filter_datasets filters by dataset_name."""
        meta_a = _make_meta("ds-a", dataset_name="alpha", dataset_version=1)
        meta_b = _make_meta("ds-b", dataset_name="beta", dataset_version=1)
        memory_store.save("ds-a", meta_a, sample_arrays)
        memory_store.save("ds-b", meta_b, sample_arrays)

        results, total = memory_store.filter_datasets(dataset_name="alpha")
        assert total == 1
        assert results[0].dataset_name == "alpha"

    def test_filter_by_dataset_version(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """filter_datasets filters by dataset_version."""
        meta_v1 = _make_meta("ds-v1", dataset_name="experiment", dataset_version=1)
        meta_v2 = _make_meta("ds-v2", dataset_name="experiment", dataset_version=2)
        memory_store.save("ds-v1", meta_v1, sample_arrays)
        memory_store.save("ds-v2", meta_v2, sample_arrays)

        results, total = memory_store.filter_datasets(dataset_version=2)
        assert total == 1
        assert results[0].dataset_version == 2

    def test_filter_by_name_and_version(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """filter_datasets filters by both dataset_name and dataset_version."""
        meta_a1 = _make_meta("ds-a1", dataset_name="alpha", dataset_version=1)
        meta_a2 = _make_meta("ds-a2", dataset_name="alpha", dataset_version=2)
        meta_b1 = _make_meta("ds-b1", dataset_name="beta", dataset_version=1)
        memory_store.save("ds-a1", meta_a1, sample_arrays)
        memory_store.save("ds-a2", meta_a2, sample_arrays)
        memory_store.save("ds-b1", meta_b1, sample_arrays)

        results, total = memory_store.filter_datasets(dataset_name="alpha", dataset_version=2)
        assert total == 1
        assert results[0].dataset_id == "ds-a2"

    def test_parent_dataset_id_lineage(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """parent_dataset_id correctly tracks lineage."""
        parent = _make_meta("ds-parent", dataset_name="lineage", dataset_version=1)
        child = _make_meta("ds-child", dataset_name="lineage", dataset_version=2, parent_dataset_id="ds-parent")
        memory_store.save("ds-parent", parent, sample_arrays)
        memory_store.save("ds-child", child, sample_arrays)

        child_meta = memory_store.get_meta("ds-child")
        assert child_meta is not None
        assert child_meta.parent_dataset_id == "ds-parent"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVersioningAPIEndpoints:
    """Tests for versioning-related API endpoints."""

    def test_create_dataset_with_name_assigns_version_1(self, client: TestClient) -> None:
        """Creating a dataset with a name assigns version 1."""
        data = _create_named_spiral(client, seed=500, name="my-experiment")
        meta = data["meta"]
        assert meta["dataset_name"] == "my-experiment"
        assert meta["dataset_version"] == 1

    def test_create_second_dataset_same_name_assigns_version_2(self, client: TestClient) -> None:
        """Creating a second dataset with the same name assigns version 2."""
        _create_named_spiral(client, seed=500, name="my-experiment")
        data2 = _create_named_spiral(client, seed=501, name="my-experiment")
        meta2 = data2["meta"]
        assert meta2["dataset_name"] == "my-experiment"
        assert meta2["dataset_version"] == 2

    def test_create_dataset_without_name_has_no_version(self, client: TestClient) -> None:
        """Creating a dataset without a name has no version."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 510},
            "persist": True,
        }
        resp = client.post("/v1/datasets", json=request)
        assert resp.status_code == 201
        meta = resp.json()["meta"]
        assert meta["dataset_name"] is None
        assert meta["dataset_version"] is None

    def test_checksum_is_computed_and_stored(self, client: TestClient) -> None:
        """Checksum is computed and stored on dataset creation."""
        data = _create_named_spiral(client, seed=520, name="checksum-test")
        meta = data["meta"]
        assert meta["checksum"] is not None
        assert len(meta["checksum"]) == 64  # SHA-256 hex digest

    def test_checksum_on_unnamed_dataset(self, client: TestClient) -> None:
        """Checksum is computed even for unnamed datasets."""
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 521},
            "persist": True,
        }
        resp = client.post("/v1/datasets", json=request)
        assert resp.status_code == 201
        meta = resp.json()["meta"]
        assert meta["checksum"] is not None
        assert len(meta["checksum"]) == 64

    def test_versioning_fields_in_api_response(self, client: TestClient) -> None:
        """Versioning fields are present in API response metadata."""
        data = _create_named_spiral(
            client,
            seed=530,
            name="api-test",
            description="Test description",
            created_by="test-user",
        )
        meta = data["meta"]
        assert meta["dataset_name"] == "api-test"
        assert meta["dataset_version"] == 1
        assert meta["description"] == "Test description"
        assert meta["created_by"] == "test-user"

    def test_parent_dataset_id_in_api(self, client: TestClient) -> None:
        """parent_dataset_id is passed through in API creation."""
        data1 = _create_named_spiral(client, seed=540, name="lineage-test")
        parent_id = data1["dataset_id"]

        data2 = _create_named_spiral(client, seed=541, name="lineage-test", parent_dataset_id=parent_id)
        meta2 = data2["meta"]
        assert meta2["parent_dataset_id"] == parent_id
        assert meta2["dataset_version"] == 2

    def test_versions_endpoint(self, client: TestClient) -> None:
        """GET /versions returns all versions of a named dataset."""
        _create_named_spiral(client, seed=600, name="versioned")
        _create_named_spiral(client, seed=601, name="versioned")
        _create_named_spiral(client, seed=602, name="versioned")

        resp = client.get("/v1/datasets/versions?name=versioned")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset_name"] == "versioned"
        assert data["total"] == 3
        assert data["latest_version"] == 3
        assert len(data["versions"]) == 3
        # Verify sorted by version ascending
        versions = [v["dataset_version"] for v in data["versions"]]
        assert versions == [1, 2, 3]

    def test_versions_endpoint_empty(self, client: TestClient) -> None:
        """GET /versions returns empty list for unknown name."""
        resp = client.get("/v1/datasets/versions?name=nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["versions"] == []
        assert data["latest_version"] is None

    def test_latest_endpoint(self, client: TestClient) -> None:
        """GET /latest returns the latest version of a named dataset."""
        _create_named_spiral(client, seed=700, name="latest-test")
        _create_named_spiral(client, seed=701, name="latest-test")

        resp = client.get("/v1/datasets/latest?name=latest-test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset_name"] == "latest-test"
        assert data["dataset_version"] == 2

    def test_latest_endpoint_404(self, client: TestClient) -> None:
        """GET /latest returns 404 for unknown name."""
        resp = client.get("/v1/datasets/latest?name=nonexistent")
        assert resp.status_code == 404
        assert "No versions found" in resp.json()["detail"]

    def test_filter_endpoint_with_dataset_name(self, client: TestClient) -> None:
        """Filter endpoint accepts dataset_name parameter."""
        _create_named_spiral(client, seed=800, name="filter-name")
        _create_named_spiral(client, seed=801, name="other-name")

        resp = client.get("/v1/datasets/filter?dataset_name=filter-name")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["datasets"][0]["dataset_name"] == "filter-name"

    def test_filter_endpoint_with_dataset_version(self, client: TestClient) -> None:
        """Filter endpoint accepts dataset_version parameter."""
        _create_named_spiral(client, seed=810, name="filter-version")
        _create_named_spiral(client, seed=811, name="filter-version")

        resp = client.get("/v1/datasets/filter?dataset_version=1")
        assert resp.status_code == 200
        data = resp.json()
        assert all(d["dataset_version"] == 1 for d in data["datasets"])

    def test_filter_endpoint_with_dataset_name_and_version(self, client: TestClient) -> None:
        """Filter endpoint combines dataset_name and dataset_version."""
        _create_named_spiral(client, seed=820, name="combined-filter")
        _create_named_spiral(client, seed=821, name="combined-filter")
        _create_named_spiral(client, seed=822, name="other-filter")

        resp = client.get("/v1/datasets/filter?dataset_name=combined-filter&dataset_version=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["datasets"][0]["dataset_name"] == "combined-filter"
        assert data["datasets"][0]["dataset_version"] == 2

    def test_cached_dataset_returns_without_version_assignment(self, client: TestClient) -> None:
        """Returning a cached dataset does not re-assign version."""
        data1 = _create_named_spiral(client, seed=900, name="cached-test")
        dataset_id = data1["dataset_id"]

        # Same params -> same dataset_id -> cached
        request = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 900},
            "persist": True,
            "name": "cached-test",
        }
        resp2 = client.post("/v1/datasets", json=request)
        assert resp2.status_code == 201
        data2 = resp2.json()
        # Returns same dataset_id (cached)
        assert data2["dataset_id"] == dataset_id
        # Version should be 1 (not 2) since it was cached
        assert data2["meta"]["dataset_version"] == 1

    def test_cached_dataset_with_different_name_keeps_original_name(self, client: TestClient) -> None:
        """Cache hits keep original metadata even when requested with a new name."""
        data1 = _create_named_spiral(client, seed=910, name="canonical-name")
        dataset_id = data1["dataset_id"]

        response = client.post(
            "/v1/datasets",
            json={
                "generator": "spiral",
                "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 910},
                "persist": True,
                "name": "alternate-name",
            },
        )
        assert response.status_code == 201
        data2 = response.json()
        assert data2["dataset_id"] == dataset_id
        assert data2["meta"]["dataset_name"] == "canonical-name"
        assert data2["meta"]["dataset_version"] == 1

        versions_resp = client.get("/v1/datasets/versions?name=alternate-name")
        assert versions_resp.status_code == 200
        assert versions_resp.json()["total"] == 0

    def test_batch_create_with_versioning(self, client: TestClient) -> None:
        """Batch create passes versioning fields through."""
        request = {
            "datasets": [
                {
                    "generator": "spiral",
                    "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 950},
                    "persist": True,
                    "name": "batch-versioned",
                    "description": "Batch item 1",
                    "created_by": "batch-user",
                },
                {
                    "generator": "spiral",
                    "params": {"n_spirals": 2, "n_points_per_spiral": 50, "seed": 951},
                    "persist": True,
                    "name": "batch-versioned",
                    "description": "Batch item 2",
                },
            ]
        }
        resp = client.post("/v1/datasets/batch-create", json=request)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_created"] == 2

        # Verify versions were assigned
        versions_resp = client.get("/v1/datasets/versions?name=batch-versioned")
        versions_data = versions_resp.json()
        assert versions_data["total"] == 2
        version_nums = [v["dataset_version"] for v in versions_data["versions"]]
        assert version_nums == [1, 2]


# ---------------------------------------------------------------------------
# Concurrency regression tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVersionConcurrency:
    """Regression tests for atomic version allocation (PR #15 review)."""

    def test_save_versioned_no_duplicate_versions(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """Concurrent save_versioned() calls must produce unique version numbers."""
        n_threads = 10
        results: list[int | None] = [None] * n_threads
        barrier = threading.Barrier(n_threads)

        def save_thread(idx: int) -> None:
            meta = _make_meta(f"ds-concurrent-{idx}", dataset_name="race-test")
            barrier.wait()  # all threads start together
            memory_store.save_versioned(f"ds-concurrent-{idx}", meta, sample_arrays)
            results[idx] = meta.dataset_version

        threads = [threading.Thread(target=save_thread, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assigned_versions = sorted(results)
        assert assigned_versions == list(range(1, n_threads + 1))

    def test_save_versioned_sets_version_on_meta(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """save_versioned() sets dataset_version in-place on meta."""
        meta = _make_meta("ds-sv1", dataset_name="sv-test")
        assert meta.dataset_version is None
        memory_store.save_versioned("ds-sv1", meta, sample_arrays)
        assert meta.dataset_version == 1

    def test_save_versioned_skips_lock_when_version_already_set(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """save_versioned() delegates directly to save() when version is pre-set."""
        meta = _make_meta("ds-preset", dataset_name="preset-test", dataset_version=42)
        memory_store.save_versioned("ds-preset", meta, sample_arrays)
        assert meta.dataset_version == 42
        stored = memory_store.get_meta("ds-preset")
        assert stored is not None
        assert stored.dataset_version == 42

    def test_save_versioned_no_name_no_version(self, memory_store: InMemoryDatasetStore, sample_arrays: dict[str, np.ndarray]) -> None:
        """save_versioned() with no dataset_name leaves version as None."""
        meta = _make_meta("ds-anon")
        memory_store.save_versioned("ds-anon", meta, sample_arrays)
        assert meta.dataset_version is None
