"""Clamp and refusal edges of ``temporal_split_indices``.

``TestTemporalSplitIndex`` pins the two-way clamp. The leakage property test
draws ``train_ratio`` from ``[0.5, 0.8]`` and ``val_ratio`` from ``[0.05, 0.15]``,
so it stays green if this helper stops reserving a row for each partition, or
if ``train_ratio + val_ratio >= 1`` becomes ``>``. Those are the cuts every
synthetic sequence generator and ``equities_seq`` share. An empty validation
split makes a no-future-leak check vacuous: there is no validation target to
precede the test target.
"""

import pytest

from juniper_data.core.split import temporal_split_indices

pytestmark = pytest.mark.unit


def _sizes(n_samples: int, bounds: tuple[int, int]) -> tuple[int, int, int]:
    train_end, val_end = bounds
    return train_end, val_end - train_end, n_samples - val_end


class TestTemporalSplitIndicesClamp:
    """For ``n >= 3`` each partition keeps one row, even when rounding would empty one."""

    def test_a_ratio_that_rounds_onto_the_last_row_still_leaves_val_and_test(self) -> None:
        # 10 * 0.95 rounds to 10 and 10 * 0.04 rounds to 0, so the unclamped cut is (10, 10).
        bounds = temporal_split_indices(10, 0.95, 0.04)
        assert bounds == (8, 9)
        assert _sizes(10, bounds) == (8, 1, 1)

    def test_a_ratio_that_rounds_to_zero_train_still_leaves_one_train_row(self) -> None:
        # Both products round to 0, so the unclamped cut is (0, 0): every row would be test.
        bounds = temporal_split_indices(10, 0.01, 0.01)
        assert bounds == (1, 2)
        assert _sizes(10, bounds) == (1, 1, 8)

    def test_a_zero_val_ratio_still_keeps_one_validation_row(self) -> None:
        # ``val_ratio`` 0 is legal, and the unclamped cut is (5, 5). The clamp still
        # reserves one validation row, because an empty middle split makes the
        # train < val < test check pass without looking at anything.
        bounds = temporal_split_indices(10, 0.5, 0.0)
        assert bounds == (5, 6)
        assert _sizes(10, bounds) == (5, 1, 4)

    def test_rounding_that_consumes_the_tail_still_leaves_one_test_row(self) -> None:
        # 10 * 0.15 is 1.5 (banker's round -> 2) and 10 * 0.8 is 8, so the unclamped cut is (2, 10).
        bounds = temporal_split_indices(10, 0.15, 0.8)
        assert bounds == (2, 9)
        assert _sizes(10, bounds) == (2, 7, 1)

    def test_three_rows_is_the_smallest_count_that_keeps_all_three_partitions(self) -> None:
        # Unclamped, 0.8 of 3 rounds to 2 and the val share rounds to 0: validation is empty.
        bounds = temporal_split_indices(3, 0.8, 0.1)
        assert bounds == (1, 2)
        assert _sizes(3, bounds) == (1, 1, 1)

    def test_two_rows_are_not_forced_into_three_partitions(self) -> None:
        # Below 3 the clamp is skipped. Applying the ``n >= 3`` bounds here would yield (0, 1).
        assert temporal_split_indices(2, 0.9, 0.05) == (2, 2)
        assert temporal_split_indices(2, 0.5, 0.0) == (1, 1)

    def test_a_split_that_already_has_three_nonempty_parts_is_left_alone(self) -> None:
        bounds = temporal_split_indices(10, 0.8, 0.1)
        assert bounds == (8, 9)
        assert _sizes(10, bounds) == (8, 1, 1)

    def test_half_values_use_bankers_rounding(self) -> None:
        # 9 * 0.5 is 4.5. ``int`` truncates to 4 and then adds ``int(3.6) == 3``, giving (4, 7).
        # Half-up rounding of 4.5 is 5, and the clamp then yields (5, 8). Banker's round stays (4, 8).
        bounds = temporal_split_indices(9, 0.5, 0.4)
        assert bounds == (4, 8)
        assert sum(_sizes(9, bounds)) == 9


class TestTemporalSplitIndicesRefusal:
    """The sum check is ``>=``, so a ratio pair that lands on 1.0 leaves no test split."""

    def test_ratios_that_sum_to_one_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must be < 1.0 to leave a test split"):
            temporal_split_indices(10, 0.9, 0.1)

    def test_train_ratio_of_one_does_not_keep_a_test_row(self) -> None:
        # The two-way helper accepts 1.0 and clamps to ``n - 1``. This one refuses:
        # 1.0 plus a legal ``val_ratio`` of 0 sums to 1.
        with pytest.raises(ValueError, match="must be < 1.0 to leave a test split"):
            temporal_split_indices(10, 1.0, 0.0)

    def test_a_zero_train_ratio_is_refused_before_the_clamp(self) -> None:
        with pytest.raises(ValueError, match=r"train_ratio must be in \(0, 1\]"):
            temporal_split_indices(10, 0.0, 0.1)

    def test_a_val_ratio_of_one_is_refused_on_its_own_bound(self) -> None:
        with pytest.raises(ValueError, match=r"val_ratio must be in \[0, 1\)"):
            temporal_split_indices(10, 0.5, 1.0)
