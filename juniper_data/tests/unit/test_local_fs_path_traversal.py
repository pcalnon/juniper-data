"""Path-traversal defense tests for LocalFSDatasetStore (JD-SEC-01)."""

import tempfile
from pathlib import Path

import pytest

from juniper_data.storage.local_fs import LocalFSDatasetStore, _validate_dataset_id


@pytest.fixture
def temp_store():
    """Create a temporary LocalFSDatasetStore for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield LocalFSDatasetStore(Path(tmpdir))


class TestDatasetIdValidator:
    """Tests for the dataset_id allowlist validator."""

    @pytest.mark.parametrize(
        "dataset_id",
        [
            "spiral-v1.0.0-a3f8e12b4c567890",
            "xor-v2-0123456789abcdef",
            "abc",
            "A1.2_3-4",
            "z" * 128,
        ],
    )
    def test_accepts_valid_ids(self, dataset_id: str) -> None:
        _validate_dataset_id(dataset_id)

    @pytest.mark.parametrize(
        "dataset_id",
        [
            "",
            "..",
            "../etc/passwd",
            "foo/bar",
            "foo\\bar",
            ".hidden",
            "foo..bar",
            "  leading-space",
            "percent%encoded",
            "a" * 129,
        ],
    )
    def test_rejects_invalid_ids(self, dataset_id: str) -> None:
        with pytest.raises(ValueError):
            _validate_dataset_id(dataset_id)

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValueError):
            _validate_dataset_id(None)  # type: ignore[arg-type]


class TestLocalFSPathTraversalDefense:
    """End-to-end traversal defense at the store API surface."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "../escape",
            "../../etc/passwd",
            "foo/bar",
            "..\\escape",
        ],
    )
    def test_meta_path_rejects_traversal(self, temp_store: LocalFSDatasetStore, malicious_id: str) -> None:
        with pytest.raises(ValueError):
            temp_store._meta_path(malicious_id)

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "../escape",
            "../../etc/passwd",
            "foo/bar",
        ],
    )
    def test_npz_path_rejects_traversal(self, temp_store: LocalFSDatasetStore, malicious_id: str) -> None:
        with pytest.raises(ValueError):
            temp_store._npz_path(malicious_id)

    def test_delete_rejects_traversal(self, temp_store: LocalFSDatasetStore) -> None:
        with pytest.raises(ValueError):
            temp_store.delete("../escape")

    def test_get_meta_rejects_traversal(self, temp_store: LocalFSDatasetStore) -> None:
        with pytest.raises(ValueError):
            temp_store.get_meta("../../etc/passwd")

    def test_get_artifact_bytes_rejects_traversal(self, temp_store: LocalFSDatasetStore) -> None:
        with pytest.raises(ValueError):
            temp_store.get_artifact_bytes("foo/../bar")

    def test_exists_rejects_traversal(self, temp_store: LocalFSDatasetStore) -> None:
        with pytest.raises(ValueError):
            temp_store.exists("../outside")

    def test_resolved_path_stays_within_base(self, temp_store: LocalFSDatasetStore) -> None:
        """Valid IDs must resolve inside the configured base path."""
        meta_path = temp_store._meta_path("valid-id-123")
        assert meta_path.resolve().is_relative_to(temp_store.base_path.resolve())
