"""Unit tests for the two partition sizing models (design decisions 2, 6.3 and 8).

Pins the behaviour that separates ``additive`` from ``carve``:

* additive honours the size knob as the TRAIN count and generates val/test as
  ADDITIONAL rows;
* carve divides a fixed N and, when the ratios account for every row, loses
  none of them to independent rounding;
* generators that read real data reject additive outright rather than accepting
  the parameter and quietly carving.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_partition_sizing.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import numpy as np
import pytest

from juniper_data.core.partition_params import (
    DEFAULT_CARVE_ONLY_VAL_RATIO,
    DEFAULT_CARVE_VAL_RATIO,
    CarveOnlyPartitionParams,
    PartitionParams,
)
from juniper_data.core.split import (
    SIZING_MODE_ADDITIVE,
    SIZING_MODE_CARVE,
    partition_and_assemble,
    per_unit_count,
    resolve_counts_for_params,
    resolve_partition_counts,
)

pytestmark = [pytest.mark.unit]


class _Params(PartitionParams):
    """A synthetic generator's params: additive by default, carve available."""

    train_ratio: float = 0.8
    test_ratio: float = 0.2


class _RealDataParams(CarveOnlyPartitionParams):
    """A real-data generator's params: carve only."""

    train_ratio: float = 0.8
    test_ratio: float = 0.1


class TestAdditiveSizing:
    def test_train_count_is_honoured_literally(self) -> None:
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_ADDITIVE, n_native=1000)

        assert counts["n_train"] == 1000
        assert counts["n_val"] == 400
        assert counts["n_test"] == 300
        assert counts["n_total"] == 1700

    def test_raw_requirement_is_the_realised_total(self) -> None:
        """The generator must produce every row the three partitions will use."""
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_ADDITIVE, n_native=1000)

        assert counts["n_raw_required"] == counts["n_total"]

    def test_percentages_are_relative_to_train(self) -> None:
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_ADDITIVE, n_native=200, val_percent=50.0, test_percent=25.0)

        assert (counts["n_train"], counts["n_val"], counts["n_test"]) == (200, 100, 50)

    def test_ratios_are_ignored(self) -> None:
        """train_ratio / test_ratio do not participate in additive sizing."""
        with_ratios = resolve_partition_counts(sizing_mode=SIZING_MODE_ADDITIVE, n_native=100, train_ratio=0.1, test_ratio=0.9)
        without = resolve_partition_counts(sizing_mode=SIZING_MODE_ADDITIVE, n_native=100)

        assert with_ratios == without


class TestCarveSizing:
    def test_divides_the_fixed_dataset(self) -> None:
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_CARVE, n_native=1000, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        assert (counts["n_train"], counts["n_val"], counts["n_test"]) == (800, 100, 100)
        assert counts["n_raw_required"] == 1000, "a carve invents no rows"

    @pytest.mark.parametrize("n_native", [3, 4, 5, 7, 9, 10, 11, 13, 97, 100, 101])
    def test_no_row_is_lost_when_ratios_account_for_all(self, n_native: int) -> None:
        """The regression this guards cost a quarter of a four-row dataset.

        Rounding all three ratios independently gives 3 + 0 + 0 on four rows at
        0.8 / 0.1 / 0.1. The last partition must absorb the remainder.
        """
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_CARVE, n_native=n_native, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        assert counts["n_total"] == n_native, f"{n_native - counts['n_total']} row(s) silently dropped"

    def test_partial_ratios_do_not_absorb_the_remainder(self) -> None:
        """Asking for 70 % of the rows must not silently inflate test to 50 %."""
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_CARVE, n_native=100, train_ratio=0.5, val_ratio=0.0, test_ratio=0.2)

        assert (counts["n_train"], counts["n_val"], counts["n_test"]) == (50, 0, 20)
        assert counts["n_total"] == 70

    def test_oversubscription_is_trimmed_from_the_end(self) -> None:
        """Train is never trimmed -- every baseline is measured against it."""
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_CARVE, n_native=100, train_ratio=0.8, val_ratio=0.3, test_ratio=0.3)

        assert counts["n_train"] == 80
        assert counts["n_total"] <= 100

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="sizing_mode must be one of"):
            resolve_partition_counts(sizing_mode="sideways", n_native=10)

    def test_rejects_empty_dataset(self) -> None:
        with pytest.raises(ValueError, match="n_native must be at least 1"):
            resolve_partition_counts(sizing_mode=SIZING_MODE_CARVE, n_native=0)

    @pytest.mark.parametrize("field", ["train_ratio", "val_ratio", "test_ratio"])
    def test_rejects_out_of_range_ratio(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be between 0 and 1"):
            resolve_partition_counts(sizing_mode=SIZING_MODE_CARVE, n_native=10, **{field: 1.5})


class TestPerUnitCount:
    def test_divides_evenly(self) -> None:
        assert per_unit_count(1700, 2) == 850
        assert per_unit_count(1700, 4) == 425

    def test_rounds_up_so_the_generator_is_never_short(self) -> None:
        assert per_unit_count(1699, 3) == 567
        assert 567 * 3 >= 1699

    def test_rejects_zero_units(self) -> None:
        with pytest.raises(ValueError, match="n_units must be at least 1"):
            per_unit_count(100, 0)


class TestPartitionAndAssemble:
    @staticmethod
    def _arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
        X = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
        y = np.zeros((n, 2), dtype=np.float32)
        y[:, 0] = 1.0
        return X, y

    def test_full_is_the_union_of_the_three_partitions(self) -> None:
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_ADDITIVE, n_native=100)
        X, y = self._arrays(counts["n_raw_required"])

        result = partition_and_assemble(X, y, counts, seed=42, shuffle=True)

        assert result["X_full"].shape[0] == result["X_train"].shape[0] + result["X_val"].shape[0] + result["X_test"].shape[0]
        assert result["y_full"].shape[0] == result["X_full"].shape[0]

    def test_surplus_rows_do_not_break_the_length_identity(self) -> None:
        """Rounding a per-unit knob up can over-produce; full must still match."""
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_ADDITIVE, n_native=100)
        X, y = self._arrays(counts["n_raw_required"] + 7)

        result = partition_and_assemble(X, y, counts, seed=42, shuffle=True)

        assert result["X_full"].shape[0] == counts["n_total"]

    def test_unshuffled_preserves_generation_order(self) -> None:
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_CARVE, n_native=10, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        X, y = self._arrays(10)

        result = partition_and_assemble(X, y, counts, seed=None, shuffle=False)

        np.testing.assert_array_equal(result["X_full"], X)


class TestPartitionParamsModel:
    def test_synthetic_defaults_to_additive(self) -> None:
        params = _Params()

        assert params.sizing_mode == SIZING_MODE_ADDITIVE
        assert params.val_ratio == DEFAULT_CARVE_VAL_RATIO == 0.0

    def test_real_data_defaults_to_carve_with_a_validation_share(self) -> None:
        params = _RealDataParams()

        assert params.sizing_mode == SIZING_MODE_CARVE
        assert params.val_ratio == DEFAULT_CARVE_ONLY_VAL_RATIO == 0.1

    def test_real_data_rejects_additive(self) -> None:
        """Unimplementable, not merely undesirable -- so it raises."""
        with pytest.raises(ValueError, match="not available for this generator"):
            _RealDataParams(sizing_mode=SIZING_MODE_ADDITIVE)

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="sizing_mode must be"):
            _Params(sizing_mode="sideways")

    def test_rejects_oversubscribed_carve(self) -> None:
        with pytest.raises(ValueError, match="must be <= 1.0"):
            _Params(sizing_mode=SIZING_MODE_CARVE, train_ratio=0.8, val_ratio=0.3, test_ratio=0.3)

    def test_oversubscription_is_rejected_in_additive_mode_too(self) -> None:
        """Additive ignores these ratios, but a malformed value is still malformed."""
        with pytest.raises(ValueError, match="must be <= 1.0"):
            _Params(sizing_mode=SIZING_MODE_ADDITIVE, train_ratio=0.9, test_ratio=0.9)

    def test_resolve_counts_reads_the_model(self) -> None:
        counts = resolve_counts_for_params(_Params(), 1000)

        assert (counts["n_train"], counts["n_val"], counts["n_test"]) == (1000, 400, 300)

    def test_resolve_counts_follows_the_mode(self) -> None:
        counts = resolve_counts_for_params(_Params(sizing_mode=SIZING_MODE_CARVE), 1000)

        assert counts["n_train"] == 800
        assert counts["n_raw_required"] == 1000
