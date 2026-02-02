"""Integration tests for storage workflows.

Tests cover end-to-end scenarios:
- Generate → Store → Retrieve → Verify
- Cross-store compatibility
- Full dataset lifecycle
"""

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from juniper_data.core.artifacts import compute_checksum
from juniper_data.core.models import DatasetMeta
from juniper_data.generators.spiral import SpiralGenerator, SpiralParams
from juniper_data.storage import InMemoryDatasetStore, LocalFSDatasetStore


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def spiral_params() -> SpiralParams:
    """Standard spiral parameters for testing."""
    return SpiralParams(
        n_spirals=2,
        n_points_per_spiral=100,
        noise=0.1,
        seed=42,
    )


def create_dataset_meta(dataset_id: str, params: SpiralParams, X: np.ndarray, y: np.ndarray, n_train: int, n_test: int) -> DatasetMeta:
    """Helper to create DatasetMeta from generated data."""
    n_classes = y.shape[1] if len(y.shape) > 1 else len(np.unique(y))
    class_counts = np.sum(y, axis=0).astype(int) if len(y.shape) > 1 else np.bincount(y.astype(int))

    return DatasetMeta(
        dataset_id=dataset_id,
        generator="spiral",
        generator_version="1.0.0",
        params=params.model_dump(),
        n_samples=len(X),
        n_features=X.shape[1],
        n_classes=n_classes,
        n_train=n_train,
        n_test=n_test,
        class_distribution={str(i): int(c) for i, c in enumerate(class_counts)},
        created_at=datetime.now(),
    )


@pytest.mark.integration
class TestGenerateStoreRetrieveWorkflow:
    """Tests for the complete generate → store → retrieve workflow."""

    def test_memory_store_full_workflow(self, spiral_params: SpiralParams):
        """Generate spiral → store in memory → retrieve and verify."""
        store = InMemoryDatasetStore()

        data = SpiralGenerator.generate(spiral_params)
        X, y = data["X_full"], data["y_full"]
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        arrays = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

        meta = create_dataset_meta("spiral-mem-001", spiral_params, X, y, len(X_train), len(X_test))

        store.save("spiral-mem-001", meta, arrays)

        assert store.exists("spiral-mem-001")
        retrieved_meta = store.get_meta("spiral-mem-001")
        assert retrieved_meta.dataset_id == "spiral-mem-001"
        assert retrieved_meta.n_samples == 200

        artifact_bytes = store.get_artifact_bytes("spiral-mem-001")
        loaded = np.load(io.BytesIO(artifact_bytes))
        np.testing.assert_array_equal(loaded["X_train"], X_train)
        np.testing.assert_array_equal(loaded["y_train"], y_train)

    def test_fs_store_full_workflow(self, temp_storage_dir: Path, spiral_params: SpiralParams):
        """Generate spiral → store to filesystem → retrieve and verify."""
        store = LocalFSDatasetStore(temp_storage_dir)

        data = SpiralGenerator.generate(spiral_params)
        X, y = data["X_full"], data["y_full"]
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        arrays = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

        meta = create_dataset_meta("spiral-fs-001", spiral_params, X, y, len(X_train), len(X_test))

        store.save("spiral-fs-001", meta, arrays)

        assert store.exists("spiral-fs-001")
        assert (temp_storage_dir / "spiral-fs-001.meta.json").exists()
        assert (temp_storage_dir / "spiral-fs-001.npz").exists()

        retrieved_meta = store.get_meta("spiral-fs-001")
        assert retrieved_meta.dataset_id == "spiral-fs-001"
        assert retrieved_meta.n_samples == 200

        artifact_bytes = store.get_artifact_bytes("spiral-fs-001")
        loaded = np.load(io.BytesIO(artifact_bytes))
        np.testing.assert_array_equal(loaded["X_train"], X_train)

    def test_persistence_across_store_instances(self, temp_storage_dir: Path, spiral_params: SpiralParams):
        """Data persists when creating new store instance."""
        store1 = LocalFSDatasetStore(temp_storage_dir)

        data = SpiralGenerator.generate(spiral_params)
        X, y = data["X_full"], data["y_full"]
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        arrays = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
        meta = create_dataset_meta("persist-test", spiral_params, X, y, len(X_train), len(X_test))

        store1.save("persist-test", meta, arrays)

        store2 = LocalFSDatasetStore(temp_storage_dir)

        assert store2.exists("persist-test")
        retrieved = store2.get_meta("persist-test")
        assert retrieved.dataset_id == "persist-test"

        datasets = store2.list_datasets()
        assert "persist-test" in datasets


@pytest.mark.integration
class TestDatasetLifecycle:
    """Tests for complete dataset lifecycle operations."""

    def test_create_update_delete_lifecycle(self, temp_storage_dir: Path, spiral_params: SpiralParams):
        """Test full create → verify → delete lifecycle."""
        store = LocalFSDatasetStore(temp_storage_dir)

        data = SpiralGenerator.generate(spiral_params)
        X, y = data["X_full"], data["y_full"]
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        arrays = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
        meta = create_dataset_meta("lifecycle-test", spiral_params, X, y, len(X_train), len(X_test))

        store.save("lifecycle-test", meta, arrays)
        assert store.exists("lifecycle-test")
        assert "lifecycle-test" in store.list_datasets()

        assert store.delete("lifecycle-test")
        assert not store.exists("lifecycle-test")
        assert "lifecycle-test" not in store.list_datasets()
        assert store.get_meta("lifecycle-test") is None
        assert store.get_artifact_bytes("lifecycle-test") is None

    def test_multiple_datasets(self, temp_storage_dir: Path):
        """Store and manage multiple datasets."""
        store = LocalFSDatasetStore(temp_storage_dir)

        for i, n_points in enumerate([50, 100, 200]):
            params = SpiralParams(n_spirals=2, n_points_per_spiral=n_points, seed=i)
            data = SpiralGenerator.generate(params)
            X, y = data["X_full"], data["y_full"]
            X_train, y_train = data["X_train"], data["y_train"]
            X_test, y_test = data["X_test"], data["y_test"]

            arrays = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            meta = create_dataset_meta(f"multi-{i}", params, X, y, len(X_train), len(X_test))

            store.save(f"multi-{i}", meta, arrays)

        datasets = store.list_datasets()
        assert len(datasets) == 3
        assert all(f"multi-{i}" in datasets for i in range(3))

        store.delete("multi-1")
        datasets = store.list_datasets()
        assert len(datasets) == 2
        assert "multi-1" not in datasets


@pytest.mark.integration
class TestChecksumVerification:
    """Tests for checksum verification across storage operations."""

    def test_checksum_consistency(self, temp_storage_dir: Path, spiral_params: SpiralParams):
        """Checksum remains consistent through store/retrieve cycle."""
        store = LocalFSDatasetStore(temp_storage_dir)

        data = SpiralGenerator.generate(spiral_params)
        X, y = data["X_full"], data["y_full"]
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        arrays = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

        original_checksum = compute_checksum(arrays)

        meta = create_dataset_meta("checksum-test", spiral_params, X, y, len(X_train), len(X_test))
        meta.checksum = original_checksum

        store.save("checksum-test", meta, arrays)

        retrieved_meta = store.get_meta("checksum-test")
        assert retrieved_meta.checksum == original_checksum

        artifact_bytes = store.get_artifact_bytes("checksum-test")
        loaded = np.load(io.BytesIO(artifact_bytes))
        loaded_arrays = {k: loaded[k] for k in loaded.files}

        loaded_checksum = compute_checksum(loaded_arrays)
        assert loaded_checksum == original_checksum


@pytest.mark.integration
class TestReproducibility:
    """Tests for dataset generation reproducibility."""

    def test_seed_reproducibility(self):
        """Same seed produces identical datasets."""
        params1 = SpiralParams(n_spirals=2, n_points_per_spiral=100, seed=42)
        params2 = SpiralParams(n_spirals=2, n_points_per_spiral=100, seed=42)

        data1 = SpiralGenerator.generate(params1)
        data2 = SpiralGenerator.generate(params2)

        np.testing.assert_array_equal(data1["X_full"], data2["X_full"])
        np.testing.assert_array_equal(data1["y_full"], data2["y_full"])

    def test_different_seeds_produce_different_data(self):
        """Different seeds produce different datasets."""
        params1 = SpiralParams(n_spirals=2, n_points_per_spiral=100, seed=42)
        params2 = SpiralParams(n_spirals=2, n_points_per_spiral=100, seed=43)

        data1 = SpiralGenerator.generate(params1)
        data2 = SpiralGenerator.generate(params2)

        assert not np.allclose(data1["X_full"], data2["X_full"])
