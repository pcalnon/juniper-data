#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_storage_benchmarks.py
# Author:        Paul Calnon
# Version:       0.4.2
#
# Date Created:  2026-02-25
# Last Modified: 2026-02-25
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Performance benchmarks for storage backends.
#    Measures throughput for save, retrieve, list, and delete operations
#    on InMemoryDatasetStore and LocalFSDatasetStore.
#
# Usage:
#    # Run benchmarks with timing (disabled by default in addopts):
#    pytest juniper_data/tests/performance/test_storage_benchmarks.py --benchmark-enable -v
#
#    # Run with autosave for regression tracking:
#    pytest juniper_data/tests/performance/test_storage_benchmarks.py --benchmark-enable --benchmark-autosave
#
# References:
#    - RD-009: Performance Test Infrastructure
#    - pytest-benchmark: https://pytest-benchmark.readthedocs.io/
#####################################################################################################################################################################################################

"""Performance benchmarks for storage backends.

Benchmarks measure throughput for core storage operations (save, get_meta,
get_artifact_bytes, list, delete) on InMemoryDatasetStore (baseline) and
LocalFSDatasetStore (I/O-bound). Optional backends (Redis, PostgreSQL,
HuggingFace, Kaggle) are excluded as they require external services.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.local_fs import LocalFSDatasetStore
from juniper_data.storage.memory import InMemoryDatasetStore


def _make_meta(dataset_id: str) -> DatasetMeta:
    """Create a representative DatasetMeta for benchmarks."""
    return DatasetMeta(
        dataset_id=dataset_id,
        generator="spiral",
        generator_version="0.4.2",
        params={"n_spirals": 2, "n_points_per_spiral": 500, "seed": 42},
        n_samples=1000,
        n_features=2,
        n_classes=2,
        n_train=800,
        n_test=200,
        class_distribution={"0": 500, "1": 500},
        created_at=datetime.now(UTC),
    )


def _make_arrays(n_train: int = 800, n_test: int = 200, n_features: int = 2) -> dict[str, np.ndarray]:
    """Create representative dataset arrays for benchmarks."""
    rng = np.random.default_rng(42)
    return {
        "X_train": rng.random((n_train, n_features), dtype=np.float32),
        "y_train": rng.random((n_train, 2), dtype=np.float32),
        "X_test": rng.random((n_test, n_features), dtype=np.float32),
        "y_test": rng.random((n_test, 2), dtype=np.float32),
        "X_full": rng.random((n_train + n_test, n_features), dtype=np.float32),
        "y_full": rng.random((n_train + n_test, 2), dtype=np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# InMemoryDatasetStore Benchmarks (Baseline)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestInMemoryStoreBenchmarks:
    """Benchmark InMemoryDatasetStore operations.

    In-memory store provides the baseline measurement for storage
    operations without filesystem or network I/O overhead.
    """

    def test_save(self, benchmark):
        """Benchmark save operation."""
        store = InMemoryDatasetStore()
        meta = _make_meta("bench-save")
        arrays = _make_arrays()
        benchmark(store.save, "bench-save", meta, arrays)
        assert store.exists("bench-save")

    def test_get_meta(self, benchmark):
        """Benchmark metadata retrieval."""
        store = InMemoryDatasetStore()
        store.save("bench-meta", _make_meta("bench-meta"), _make_arrays())
        result = benchmark(store.get_meta, "bench-meta")
        assert result is not None
        assert result.dataset_id == "bench-meta"

    def test_get_artifact_bytes(self, benchmark):
        """Benchmark artifact retrieval (NPZ bytes)."""
        store = InMemoryDatasetStore()
        store.save("bench-artifact", _make_meta("bench-artifact"), _make_arrays())
        result = benchmark(store.get_artifact_bytes, "bench-artifact")
        assert result is not None
        assert len(result) > 0

    def test_exists(self, benchmark):
        """Benchmark existence check."""
        store = InMemoryDatasetStore()
        store.save("bench-exists", _make_meta("bench-exists"), _make_arrays())
        result = benchmark(store.exists, "bench-exists")
        assert result is True

    def test_list_datasets(self, benchmark):
        """Benchmark list operation with 50 datasets."""
        store = InMemoryDatasetStore()
        arrays = _make_arrays()
        for i in range(50):
            store.save(f"bench-list-{i:03d}", _make_meta(f"bench-list-{i:03d}"), arrays)
        result = benchmark(store.list_datasets, 50, 0)
        assert len(result) == 50

    def test_delete(self, benchmark):
        """Benchmark delete operation."""
        store = InMemoryDatasetStore()
        arrays = _make_arrays()

        def save_and_delete():
            store.save("bench-delete", _make_meta("bench-delete"), arrays)
            return store.delete("bench-delete")

        result = benchmark(save_and_delete)
        assert result is True


# ═══════════════════════════════════════════════════════════════════════════════
# LocalFSDatasetStore Benchmarks (I/O-bound)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestLocalFSStoreBenchmarks:
    """Benchmark LocalFSDatasetStore operations.

    Filesystem store measures I/O-bound performance for JSON metadata
    writes and NPZ artifact serialization/deserialization.
    """

    def test_save(self, benchmark, tmp_path):
        """Benchmark save operation (JSON meta + NPZ artifact)."""
        store = LocalFSDatasetStore(str(tmp_path))
        meta = _make_meta("bench-save")
        arrays = _make_arrays()
        benchmark(store.save, "bench-save", meta, arrays)
        assert store.exists("bench-save")

    def test_get_meta(self, benchmark, tmp_path):
        """Benchmark metadata retrieval from filesystem."""
        store = LocalFSDatasetStore(str(tmp_path))
        store.save("bench-meta", _make_meta("bench-meta"), _make_arrays())
        result = benchmark(store.get_meta, "bench-meta")
        assert result is not None
        assert result.dataset_id == "bench-meta"

    def test_get_artifact_bytes(self, benchmark, tmp_path):
        """Benchmark artifact retrieval from filesystem."""
        store = LocalFSDatasetStore(str(tmp_path))
        store.save("bench-artifact", _make_meta("bench-artifact"), _make_arrays())
        result = benchmark(store.get_artifact_bytes, "bench-artifact")
        assert result is not None
        assert len(result) > 0

    def test_exists(self, benchmark, tmp_path):
        """Benchmark existence check on filesystem."""
        store = LocalFSDatasetStore(str(tmp_path))
        store.save("bench-exists", _make_meta("bench-exists"), _make_arrays())
        result = benchmark(store.exists, "bench-exists")
        assert result is True

    def test_list_datasets(self, benchmark, tmp_path):
        """Benchmark list operation with 50 datasets on filesystem."""
        store = LocalFSDatasetStore(str(tmp_path))
        arrays = _make_arrays()
        for i in range(50):
            store.save(f"bench-list-{i:03d}", _make_meta(f"bench-list-{i:03d}"), arrays)
        result = benchmark(store.list_datasets, 50, 0)
        assert len(result) == 50

    def test_delete(self, benchmark, tmp_path):
        """Benchmark delete operation on filesystem."""
        store = LocalFSDatasetStore(str(tmp_path))
        arrays = _make_arrays()

        def save_and_delete():
            store.save("bench-delete", _make_meta("bench-delete"), arrays)
            return store.delete("bench-delete")

        result = benchmark(save_and_delete)
        assert result is True


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Size Scaling Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestStorageScaling:
    """Benchmark storage throughput across dataset sizes.

    Measures how save/retrieve times scale with increasing dataset
    sizes (points * features), using InMemoryDatasetStore to isolate
    serialization overhead from filesystem I/O.
    """

    @pytest.mark.parametrize(
        ("n_train", "n_test"),
        [(80, 20), (800, 200), (4000, 1000), (8000, 2000)],
        ids=["100pts", "1000pts", "5000pts", "10000pts"],
    )
    def test_save_scaling(self, benchmark, n_train, n_test):
        """Benchmark save with increasing dataset sizes."""
        store = InMemoryDatasetStore()
        meta = _make_meta("bench-scale")
        meta.n_train = n_train
        meta.n_test = n_test
        meta.n_samples = n_train + n_test
        arrays = _make_arrays(n_train=n_train, n_test=n_test)
        benchmark(store.save, "bench-scale", meta, arrays)
        assert store.exists("bench-scale")

    @pytest.mark.parametrize(
        ("n_train", "n_test"),
        [(80, 20), (800, 200), (4000, 1000), (8000, 2000)],
        ids=["100pts", "1000pts", "5000pts", "10000pts"],
    )
    def test_retrieve_scaling(self, benchmark, n_train, n_test):
        """Benchmark artifact retrieval with increasing dataset sizes."""
        store = InMemoryDatasetStore()
        meta = _make_meta("bench-scale")
        meta.n_train = n_train
        meta.n_test = n_test
        meta.n_samples = n_train + n_test
        arrays = _make_arrays(n_train=n_train, n_test=n_test)
        store.save("bench-scale", meta, arrays)
        result = benchmark(store.get_artifact_bytes, "bench-scale")
        assert result is not None
