"""Unit tests for the split and shuffle utilities.

Tests cover:
- shuffle_data maintains X/y correspondence
- split_data produces correct sizes
- shuffle_and_split integration
"""

from typing import Dict

import numpy as np
import pytest

from juniper_data.core.split import shuffle_and_split, shuffle_data, split_data


@pytest.mark.unit
class TestShuffleData:
    """Tests for shuffle_data function."""

    def test_shuffle_maintains_correspondence(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify shuffling maintains correspondence between X and y."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        X_original = X.copy()

        rng = np.random.default_rng(42)
        X_shuffled, y_shuffled = shuffle_data(X, y, rng)

        assert X_shuffled.shape == X.shape
        assert y_shuffled.shape == y.shape

        for i in range(X_shuffled.shape[0]):
            x_row = X_shuffled[i]
            y_row = y_shuffled[i]
            original_idx = np.where((X_original == x_row).all(axis=1))[0][0]
            np.testing.assert_array_equal(y_row, sample_arrays["y"][original_idx])

    def test_shuffle_changes_order(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify shuffling actually changes the order."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        rng = np.random.default_rng(42)
        X_shuffled, _ = shuffle_data(X, y, rng)

        assert not np.array_equal(X, X_shuffled)

    def test_shuffle_preserves_all_values(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify shuffling preserves all original values."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        rng = np.random.default_rng(42)
        X_shuffled, y_shuffled = shuffle_data(X, y, rng)

        assert set(map(tuple, X.tolist())) == set(map(tuple, X_shuffled.tolist()))
        assert set(map(tuple, y.tolist())) == set(map(tuple, y_shuffled.tolist()))

    def test_shuffle_mismatched_samples_raises(self) -> None:
        """Verify mismatched X and y sample counts raise ValueError."""
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[:5]

        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="same number of samples"):
            shuffle_data(X, y, rng)

    def test_shuffle_deterministic_with_same_seed(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify same seed produces same shuffle order."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        rng1 = np.random.default_rng(42)
        X_shuffled1, y_shuffled1 = shuffle_data(X.copy(), y.copy(), rng1)

        rng2 = np.random.default_rng(42)
        X_shuffled2, y_shuffled2 = shuffle_data(X.copy(), y.copy(), rng2)

        np.testing.assert_array_equal(X_shuffled1, X_shuffled2)
        np.testing.assert_array_equal(y_shuffled1, y_shuffled2)


@pytest.mark.unit
class TestSplitData:
    """Tests for split_data function."""

    def test_split_produces_correct_sizes(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify split produces correct train/test sizes."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        result = split_data(X, y, train_ratio=0.8, test_ratio=0.2)

        assert result["X_train"].shape[0] == 8
        assert result["y_train"].shape[0] == 8
        assert result["X_test"].shape[0] == 2
        assert result["y_test"].shape[0] == 2

    def test_split_maintains_feature_dimensions(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify split maintains feature dimensions."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        result = split_data(X, y, train_ratio=0.6, test_ratio=0.4)

        assert result["X_train"].shape[1] == X.shape[1]
        assert result["X_test"].shape[1] == X.shape[1]
        assert result["y_train"].shape[1] == y.shape[1]
        assert result["y_test"].shape[1] == y.shape[1]

    def test_split_with_custom_ratios(self) -> None:
        """Verify custom split ratios work correctly."""
        X = np.arange(100).reshape(50, 2).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[np.arange(50) % 2]

        result = split_data(X, y, train_ratio=0.6, test_ratio=0.3)

        expected_train = int(np.round(50 * 0.6))
        expected_test = int(np.round(50 * 0.3))

        assert abs(result["X_train"].shape[0] - expected_train) <= 1
        assert abs(result["X_test"].shape[0] - expected_test) <= 1

    def test_split_no_overlap(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify train and test sets do not overlap."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        result = split_data(X, y, train_ratio=0.6, test_ratio=0.4)

        X_train = result["X_train"]
        X_test = result["X_test"]

        train_set = set(map(tuple, X_train.tolist()))
        test_set = set(map(tuple, X_test.tolist()))

        assert len(train_set & test_set) == 0

    def test_split_mismatched_samples_raises(self) -> None:
        """Verify mismatched X and y sample counts raise ValueError."""
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[:5]

        with pytest.raises(ValueError, match="same number of samples"):
            split_data(X, y, train_ratio=0.8, test_ratio=0.2)

    def test_split_invalid_train_ratio_raises(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify invalid train_ratio raises ValueError."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        with pytest.raises(ValueError, match="train_ratio"):
            split_data(X, y, train_ratio=1.5, test_ratio=0.2)

    def test_split_invalid_test_ratio_raises(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify invalid test_ratio raises ValueError."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        with pytest.raises(ValueError, match="test_ratio"):
            split_data(X, y, train_ratio=0.8, test_ratio=1.5)

    def test_split_ratios_exceed_one_raises(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify train_ratio + test_ratio > 1.0 raises ValueError."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        with pytest.raises(ValueError, match="must not exceed 1.0"):
            split_data(X, y, train_ratio=0.7, test_ratio=0.5)


@pytest.mark.unit
class TestShuffleAndSplit:
    """Tests for shuffle_and_split integration function."""

    def test_shuffle_and_split_integration(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify shuffle_and_split combines both operations correctly."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        result = shuffle_and_split(
            X=X,
            y=y,
            train_ratio=0.8,
            test_ratio=0.2,
            seed=42,
            shuffle=True,
        )

        assert "X_train" in result
        assert "y_train" in result
        assert "X_test" in result
        assert "y_test" in result

        assert result["X_train"].shape[0] == 8
        assert result["X_test"].shape[0] == 2

    def test_shuffle_and_split_deterministic(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify same seed produces identical results."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        result1 = shuffle_and_split(X, y, 0.8, 0.2, seed=42, shuffle=True)
        result2 = shuffle_and_split(X, y, 0.8, 0.2, seed=42, shuffle=True)

        np.testing.assert_array_equal(result1["X_train"], result2["X_train"])
        np.testing.assert_array_equal(result1["X_test"], result2["X_test"])

    def test_shuffle_and_split_no_shuffle(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify shuffle=False preserves original order."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        result = shuffle_and_split(X, y, 0.8, 0.2, seed=42, shuffle=False)

        np.testing.assert_array_equal(result["X_train"], X[:8])
        np.testing.assert_array_equal(result["X_test"], X[8:])

    def test_shuffle_and_split_different_seeds(self, sample_arrays: Dict[str, np.ndarray]) -> None:
        """Verify different seeds produce different shuffles."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        result1 = shuffle_and_split(X, y, 0.8, 0.2, seed=42, shuffle=True)
        result2 = shuffle_and_split(X, y, 0.8, 0.2, seed=99, shuffle=True)

        assert not np.array_equal(result1["X_train"], result2["X_train"])

    def test_split_adjusts_test_size_when_rounding_exceeds_samples(self) -> None:
        """Verify test size is adjusted when train+test rounding exceeds total samples.
        
        With 3 samples, train_ratio=0.5, test_ratio=0.5:
        - n_train = round(3 * 0.5) = round(1.5) = 2
        - n_test = round(3 * 0.5) = round(1.5) = 2
        - n_train + n_test = 4 > 3, so n_test should be adjusted to 1
        """
        X = np.arange(6).reshape(3, 2).astype(np.float32)
        y = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)

        result = split_data(X, y, train_ratio=0.5, test_ratio=0.5)

        total_split = result["X_train"].shape[0] + result["X_test"].shape[0]
        assert total_split == 3
        assert result["X_train"].shape[0] == 2
        assert result["X_test"].shape[0] == 1
