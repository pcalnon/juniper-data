"""Unit tests for core artifacts module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from juniper_data.core.artifacts import arrays_to_bytes, compute_checksum, load_npz, save_npz


@pytest.fixture
def sample_arrays() -> dict[str, np.ndarray]:
    """Create sample arrays for testing."""
    return {
        "X": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "y": np.array([[1, 0], [0, 1]], dtype=np.float32),
    }


@pytest.fixture
def temp_npz_path():
    """Create a temporary file path for NPZ files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.npz"


class TestSaveNpz:
    """Tests for save_npz function."""

    @pytest.mark.unit
    def test_save_creates_file(self, temp_npz_path: Path, sample_arrays: dict[str, np.ndarray]):
        """Test that save_npz creates the file."""
        save_npz(temp_npz_path, sample_arrays)
        assert temp_npz_path.exists()

    @pytest.mark.unit
    def test_save_correct_content(self, temp_npz_path: Path, sample_arrays: dict[str, np.ndarray]):
        """Test that saved content is correct."""
        save_npz(temp_npz_path, sample_arrays)

        with np.load(temp_npz_path) as data:
            assert set(data.files) == set(sample_arrays.keys())
            for key in sample_arrays:
                np.testing.assert_array_equal(data[key], sample_arrays[key])


class TestLoadNpz:
    """Tests for load_npz function."""

    @pytest.mark.unit
    def test_load_returns_arrays(self, temp_npz_path: Path, sample_arrays: dict[str, np.ndarray]):
        """Test that load_npz returns the correct arrays."""
        save_npz(temp_npz_path, sample_arrays)
        loaded = load_npz(temp_npz_path)

        assert isinstance(loaded, dict)
        assert set(loaded.keys()) == set(sample_arrays.keys())
        for key in sample_arrays:
            np.testing.assert_array_equal(loaded[key], sample_arrays[key])

    @pytest.mark.unit
    def test_load_returns_mutable_arrays(self, temp_npz_path: Path, sample_arrays: dict[str, np.ndarray]):
        """Test that loaded arrays are mutable (not memory-mapped)."""
        save_npz(temp_npz_path, sample_arrays)
        loaded = load_npz(temp_npz_path)

        loaded["X"][0, 0] = 999.0
        assert loaded["X"][0, 0] == 999.0


class TestArraysToBytes:
    """Tests for arrays_to_bytes function."""

    @pytest.mark.unit
    def test_returns_bytes(self, sample_arrays: dict[str, np.ndarray]):
        """Test that arrays_to_bytes returns bytes."""
        result = arrays_to_bytes(sample_arrays)
        assert isinstance(result, bytes)

    @pytest.mark.unit
    def test_bytes_contain_npz_data(self, sample_arrays: dict[str, np.ndarray]):
        """Test that bytes contain valid NPZ data."""
        import io

        result = arrays_to_bytes(sample_arrays)

        loaded = np.load(io.BytesIO(result))
        assert set(loaded.files) == set(sample_arrays.keys())
        for key in sample_arrays:
            np.testing.assert_array_equal(loaded[key], sample_arrays[key])

    @pytest.mark.unit
    def test_bytes_non_empty(self, sample_arrays: dict[str, np.ndarray]):
        """Test that result is non-empty."""
        result = arrays_to_bytes(sample_arrays)
        assert len(result) > 0


class TestComputeChecksum:
    """Tests for compute_checksum function."""

    @pytest.mark.unit
    def test_returns_hex_string(self, sample_arrays: dict[str, np.ndarray]):
        """Test that compute_checksum returns a hex string."""
        result = compute_checksum(sample_arrays)
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    @pytest.mark.unit
    def test_deterministic(self, sample_arrays: dict[str, np.ndarray]):
        """Test that checksum is deterministic for same input."""
        checksum1 = compute_checksum(sample_arrays)
        checksum2 = compute_checksum(sample_arrays)
        assert checksum1 == checksum2

    @pytest.mark.unit
    def test_different_for_different_data(self, sample_arrays: dict[str, np.ndarray]):
        """Test that checksum differs for different data."""
        checksum1 = compute_checksum(sample_arrays)

        different_arrays = {
            "X": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            "y": sample_arrays["y"],
        }
        checksum2 = compute_checksum(different_arrays)

        assert checksum1 != checksum2

    @pytest.mark.unit
    def test_sha256_format(self, sample_arrays: dict[str, np.ndarray]):
        """Test that checksum is valid SHA-256 format."""
        import hashlib

        result = compute_checksum(sample_arrays)

        try:
            int(result, 16)
            valid_hex = True
        except ValueError:
            valid_hex = False

        assert valid_hex
        assert len(result) == hashlib.sha256().digest_size * 2
