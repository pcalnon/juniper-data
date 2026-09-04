"""Unit tests for the split and shuffle utilities.

Tests cover:
- shuffle_data maintains X/y correspondence
- split_data produces correct sizes
- shuffle_and_split integration
- partition_row_counts implements the additive sizing model
- split_three_way / shuffle_and_split_three_way cut index-disjoint partitions
"""

import numpy as np
import pytest

from juniper_data.core.split import (
    DEFAULT_TEST_PERCENT,
    DEFAULT_VAL_PERCENT,
    partition_row_counts,
    shuffle_and_split,
    shuffle_and_split_three_way,
    shuffle_data,
    split_data,
    split_three_way,
    temporal_split_index,
)


@pytest.mark.unit
class TestTemporalSplitIndex:
    """Tests for the chronological (non-shuffled) split-boundary helper."""

    def test_returns_rounded_index(self) -> None:
        assert temporal_split_index(100, 0.8) == 80
        assert temporal_split_index(10, 0.7) == 7

    def test_extreme_ratios_keep_both_splits_nonempty(self) -> None:
        # train_ratio == 1.0 must still leave at least one test row.
        assert temporal_split_index(10, 1.0) == 9
        # a tiny ratio must still leave at least one train row.
        assert temporal_split_index(10, 0.01) == 1

    def test_rejects_out_of_range_ratio(self) -> None:
        with pytest.raises(ValueError):
            temporal_split_index(10, 0.0)
        with pytest.raises(ValueError):
            temporal_split_index(10, 1.5)

    def test_below_two_samples_cannot_keep_both_splits_nonempty(self) -> None:
        """With 0 or 1 rows the clamp is skipped -- there is no boundary to hold."""
        assert temporal_split_index(1, 1.0) == 1
        assert temporal_split_index(0, 0.8) == 0


@pytest.mark.unit
class TestShuffleData:
    """Tests for shuffle_data function."""

    def test_shuffle_maintains_correspondence(self, sample_arrays: dict[str, np.ndarray]) -> None:
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

    def test_shuffle_changes_order(self, sample_arrays: dict[str, np.ndarray]) -> None:
        """Verify shuffling actually changes the order."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        rng = np.random.default_rng(42)
        X_shuffled, _ = shuffle_data(X, y, rng)

        assert not np.array_equal(X, X_shuffled)

    def test_shuffle_preserves_all_values(self, sample_arrays: dict[str, np.ndarray]) -> None:
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

    def test_shuffle_deterministic_with_same_seed(self, sample_arrays: dict[str, np.ndarray]) -> None:
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

    def test_split_produces_correct_sizes(self, sample_arrays: dict[str, np.ndarray]) -> None:
        """Verify split produces correct train/test sizes."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        result = split_data(X, y, train_ratio=0.8, test_ratio=0.2)

        assert result["X_train"].shape[0] == 8
        assert result["y_train"].shape[0] == 8
        assert result["X_test"].shape[0] == 2
        assert result["y_test"].shape[0] == 2

    def test_split_maintains_feature_dimensions(self, sample_arrays: dict[str, np.ndarray]) -> None:
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

    def test_split_no_overlap(self, sample_arrays: dict[str, np.ndarray]) -> None:
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

    def test_split_invalid_train_ratio_raises(self, sample_arrays: dict[str, np.ndarray]) -> None:
        """Verify invalid train_ratio raises ValueError."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        with pytest.raises(ValueError, match="train_ratio"):
            split_data(X, y, train_ratio=1.5, test_ratio=0.2)

    def test_split_invalid_test_ratio_raises(self, sample_arrays: dict[str, np.ndarray]) -> None:
        """Verify invalid test_ratio raises ValueError."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        with pytest.raises(ValueError, match="test_ratio"):
            split_data(X, y, train_ratio=0.8, test_ratio=1.5)

    def test_split_ratios_exceed_one_raises(self, sample_arrays: dict[str, np.ndarray]) -> None:
        """Verify train_ratio + test_ratio > 1.0 raises ValueError."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        with pytest.raises(ValueError, match="must not exceed 1.0"):
            split_data(X, y, train_ratio=0.7, test_ratio=0.5)


@pytest.mark.unit
class TestShuffleAndSplit:
    """Tests for shuffle_and_split integration function."""

    def test_shuffle_and_split_integration(self, sample_arrays: dict[str, np.ndarray]) -> None:
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

    def test_shuffle_and_split_deterministic(self, sample_arrays: dict[str, np.ndarray]) -> None:
        """Verify same seed produces identical results."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        result1 = shuffle_and_split(X, y, 0.8, 0.2, seed=42, shuffle=True)
        result2 = shuffle_and_split(X, y, 0.8, 0.2, seed=42, shuffle=True)

        np.testing.assert_array_equal(result1["X_train"], result2["X_train"])
        np.testing.assert_array_equal(result1["X_test"], result2["X_test"])

    def test_shuffle_and_split_no_shuffle(self, sample_arrays: dict[str, np.ndarray]) -> None:
        """Verify shuffle=False preserves original order."""
        X = sample_arrays["X"]
        y = sample_arrays["y"]

        result = shuffle_and_split(X, y, 0.8, 0.2, seed=42, shuffle=False)

        np.testing.assert_array_equal(result["X_train"], X[:8])
        np.testing.assert_array_equal(result["X_test"], X[8:])

    def test_shuffle_and_split_different_seeds(self, sample_arrays: dict[str, np.ndarray]) -> None:
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


@pytest.mark.unit
class TestPartitionRowCounts:
    """Tests for the additive three-way sizing model (design decisions 2 and 8)."""

    def test_default_percentages_are_the_documented_100_40_30(self) -> None:
        """The design's default breakdown is train/val/test = 100/40/30."""
        assert DEFAULT_VAL_PERCENT == 40.0
        assert DEFAULT_TEST_PERCENT == 30.0

    def test_worked_example_from_the_design(self) -> None:
        """Section 6.3's worked example: n_train=1000 at 100/40/30 -> 1000/400/300."""
        counts = partition_row_counts(1000)

        assert counts["n_train"] == 1000
        assert counts["n_val"] == 400
        assert counts["n_test"] == 300
        assert counts["n_total"] == 1700

    def test_train_count_is_honoured_literally_not_carved(self) -> None:
        """The requested train count survives sizing -- val/test are ADDITIONAL rows.

        This is the property that separates the additive model from a carve-up:
        a carve of 1000 into 100/40/30 would yield a 588-row train set.
        """
        counts = partition_row_counts(1000)

        assert counts["n_train"] == 1000
        assert counts["n_total"] > 1000

    def test_decision_8_counts_are_dataset_rows_not_native_units(self) -> None:
        """n_points_per_spiral=500 x n_spirals=2 means 1000 TRAIN rows, 1700 total.

        Decision 8 (D-2) rules that the percentages denote absolute rows of the
        realised dataset, identically for every generator regardless of its
        native size knob -- never per-spiral units.
        """
        n_points_per_spiral = 500
        n_spirals = 2

        counts = partition_row_counts(n_points_per_spiral * n_spirals)

        assert counts["n_train"] == 1000
        assert counts["n_val"] == 400
        assert counts["n_test"] == 300
        assert counts["n_total"] == 1700

    def test_total_is_the_sum_of_the_three_partitions(self) -> None:
        counts = partition_row_counts(333, val_percent=17.0, test_percent=11.0)

        assert counts["n_total"] == counts["n_train"] + counts["n_val"] + counts["n_test"]

    def test_percentages_are_rounded_to_whole_rows(self) -> None:
        # 7 * 0.40 = 2.8 -> 3;  7 * 0.30 = 2.1 -> 2
        counts = partition_row_counts(7)

        assert counts["n_val"] == 3
        assert counts["n_test"] == 2
        assert counts["n_total"] == 12

    def test_zero_percentages_yield_empty_partitions(self) -> None:
        counts = partition_row_counts(50, val_percent=0.0, test_percent=0.0)

        assert counts["n_val"] == 0
        assert counts["n_test"] == 0
        assert counts["n_total"] == 50

    def test_percentages_may_exceed_one_hundred(self) -> None:
        """Percentages are relative to train, so they are not capped at 100."""
        counts = partition_row_counts(100, val_percent=150.0, test_percent=200.0)

        assert counts["n_val"] == 150
        assert counts["n_test"] == 200
        assert counts["n_total"] == 450

    def test_rejects_train_count_below_one(self) -> None:
        with pytest.raises(ValueError, match="n_train must be at least 1"):
            partition_row_counts(0)
        with pytest.raises(ValueError, match="n_train must be at least 1"):
            partition_row_counts(-5)

    def test_rejects_negative_percentages(self) -> None:
        with pytest.raises(ValueError, match="val_percent must not be negative"):
            partition_row_counts(10, val_percent=-1.0)
        with pytest.raises(ValueError, match="test_percent must not be negative"):
            partition_row_counts(10, test_percent=-1.0)

    def test_rejects_non_finite_percentages(self) -> None:
        with pytest.raises(ValueError, match="val_percent must be finite"):
            partition_row_counts(10, val_percent=float("nan"))
        with pytest.raises(ValueError, match="test_percent must be finite"):
            partition_row_counts(10, test_percent=float("inf"))


@pytest.mark.unit
class TestSplitThreeWay:
    """Tests for the contiguous three-way row-count split."""

    @staticmethod
    def _unique_rows(n: int) -> tuple[np.ndarray, np.ndarray]:
        """Arrays whose every row is distinguishable, for disjointness checks."""
        X = np.arange(n, dtype=np.float32).reshape(n, 1)
        y = np.arange(n, dtype=np.float32).reshape(n, 1)
        return X, y

    def test_blocks_have_the_requested_sizes(self) -> None:
        X, y = self._unique_rows(100)

        result = split_three_way(X, y, n_train=50, n_val=20, n_test=30)

        assert result["X_train"].shape[0] == 50
        assert result["X_val"].shape[0] == 20
        assert result["X_test"].shape[0] == 30

    def test_emits_all_six_partition_keys(self) -> None:
        X, y = self._unique_rows(10)

        result = split_three_way(X, y, n_train=4, n_val=3, n_test=3)

        assert set(result) == {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}

    def test_partitions_are_index_disjoint(self) -> None:
        """The property design section 9.6.1 relies on instead of a leak guard."""
        X, y = self._unique_rows(100)

        result = split_three_way(X, y, n_train=50, n_val=20, n_test=30)

        train = set(result["X_train"].ravel().tolist())
        val = set(result["X_val"].ravel().tolist())
        test = set(result["X_test"].ravel().tolist())

        assert train & val == set()
        assert train & test == set()
        assert val & test == set()
        assert len(train) + len(val) + len(test) == 100

    def test_blocks_are_contiguous_and_ordered(self) -> None:
        X, y = self._unique_rows(10)

        result = split_three_way(X, y, n_train=4, n_val=3, n_test=3)

        assert result["X_train"].ravel().tolist() == [0.0, 1.0, 2.0, 3.0]
        assert result["X_val"].ravel().tolist() == [4.0, 5.0, 6.0]
        assert result["X_test"].ravel().tolist() == [7.0, 8.0, 9.0]

    def test_labels_track_their_features(self) -> None:
        X, y = self._unique_rows(20)

        result = split_three_way(X, y, n_train=10, n_val=5, n_test=5)

        for split in ("train", "val", "test"):
            np.testing.assert_array_equal(result[f"X_{split}"].ravel(), result[f"y_{split}"].ravel())

    def test_rows_beyond_the_total_are_left_unused(self) -> None:
        """Surplus rows are dropped, never folded into a partition."""
        X, y = self._unique_rows(100)

        result = split_three_way(X, y, n_train=10, n_val=5, n_test=5)

        used = result["X_train"].shape[0] + result["X_val"].shape[0] + result["X_test"].shape[0]
        assert used == 20
        assert result["X_test"].ravel().tolist() == [15.0, 16.0, 17.0, 18.0, 19.0]

    def test_zero_sized_partition_is_permitted(self) -> None:
        X, y = self._unique_rows(10)

        result = split_three_way(X, y, n_train=10, n_val=0, n_test=0)

        assert result["X_train"].shape[0] == 10
        assert result["X_val"].shape[0] == 0
        assert result["X_test"].shape[0] == 0

    def test_rejects_mismatched_sample_counts(self) -> None:
        X = np.zeros((10, 2), dtype=np.float32)
        y = np.zeros((9, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="same number of samples"):
            split_three_way(X, y, n_train=5, n_val=2, n_test=2)

    def test_rejects_negative_counts(self) -> None:
        X, y = self._unique_rows(10)

        with pytest.raises(ValueError, match="n_val must not be negative"):
            split_three_way(X, y, n_train=5, n_val=-1, n_test=2)

    def test_rejects_insufficient_rows(self) -> None:
        X, y = self._unique_rows(10)

        with pytest.raises(ValueError, match="Not enough rows to partition"):
            split_three_way(X, y, n_train=5, n_val=4, n_test=4)


@pytest.mark.unit
class TestShuffleAndSplitThreeWay:
    """Tests for the shuffling three-way split."""

    @staticmethod
    def _unique_rows(n: int) -> tuple[np.ndarray, np.ndarray]:
        X = np.arange(n, dtype=np.float32).reshape(n, 1)
        y = np.arange(n, dtype=np.float32).reshape(n, 1)
        return X, y

    def test_same_seed_reproduces_the_partition(self) -> None:
        X, y = self._unique_rows(60)

        first = shuffle_and_split_three_way(X, y, n_train=30, n_val=15, n_test=15, seed=42)
        second = shuffle_and_split_three_way(X, y, n_train=30, n_val=15, n_test=15, seed=42)

        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_different_seeds_give_different_partitions(self) -> None:
        X, y = self._unique_rows(60)

        first = shuffle_and_split_three_way(X, y, n_train=30, n_val=15, n_test=15, seed=1)
        second = shuffle_and_split_three_way(X, y, n_train=30, n_val=15, n_test=15, seed=2)

        assert not np.array_equal(first["X_train"], second["X_train"])

    def test_shuffle_disabled_preserves_row_order(self) -> None:
        X, y = self._unique_rows(10)

        result = shuffle_and_split_three_way(X, y, n_train=4, n_val=3, n_test=3, shuffle=False)

        assert result["X_train"].ravel().tolist() == [0.0, 1.0, 2.0, 3.0]
        assert result["X_val"].ravel().tolist() == [4.0, 5.0, 6.0]
        assert result["X_test"].ravel().tolist() == [7.0, 8.0, 9.0]

    def test_partitions_stay_disjoint_after_shuffling(self) -> None:
        """Shuffling permutes rows but cannot make two partitions share one."""
        X, y = self._unique_rows(100)

        result = shuffle_and_split_three_way(X, y, n_train=50, n_val=20, n_test=30, seed=7)

        train = set(result["X_train"].ravel().tolist())
        val = set(result["X_val"].ravel().tolist())
        test = set(result["X_test"].ravel().tolist())

        assert train & val == set()
        assert train & test == set()
        assert val & test == set()
        assert len(train | val | test) == 100

    def test_labels_still_track_features_after_shuffling(self) -> None:
        X, y = self._unique_rows(30)

        result = shuffle_and_split_three_way(X, y, n_train=15, n_val=8, n_test=7, seed=11)

        for split in ("train", "val", "test"):
            np.testing.assert_array_equal(result[f"X_{split}"].ravel(), result[f"y_{split}"].ravel())

    def test_propagates_insufficient_row_error(self) -> None:
        X, y = self._unique_rows(10)

        with pytest.raises(ValueError, match="Not enough rows to partition"):
            shuffle_and_split_three_way(X, y, n_train=5, n_val=4, n_test=4, seed=3)

    def test_sizing_model_feeds_the_splitter(self) -> None:
        """partition_row_counts and the splitter compose into the design's model."""
        counts = partition_row_counts(100)
        X, y = self._unique_rows(counts["n_total"])

        result = shuffle_and_split_three_way(
            X,
            y,
            n_train=counts["n_train"],
            n_val=counts["n_val"],
            n_test=counts["n_test"],
            seed=5,
        )

        assert result["X_train"].shape[0] == 100
        assert result["X_val"].shape[0] == 40
        assert result["X_test"].shape[0] == 30
