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
from pydantic import ValidationError

from juniper_data.core.partition_params import (
    DEFAULT_CARVE_ONLY_VAL_RATIO,
    DEFAULT_CARVE_VAL_RATIO,
    CarveOnlyPartitionParams,
    PartitionParams,
    rescale_generator_params,
)
from juniper_data.core.split import (
    SIZING_MODE_ADDITIVE,
    SIZING_MODE_CARVE,
    partition_and_assemble,
    per_unit_count,
    resolve_counts_for_params,
    resolve_partition_counts,
)
from juniper_data.generators.spiral.defaults import MAX_POINTS
from juniper_data.generators.spiral.generator import SpiralGenerator
from juniper_data.generators.spiral.params import SpiralParams

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

    @pytest.mark.parametrize("n_native", [5, 15, 25, 35, 65, 105])
    @pytest.mark.parametrize("train_ratio,val_ratio,test_ratio", [(0.7, 0.3, 0.0), (0.9, 0.1, 0.0), (0.5, 0.5, 0.0)])
    def test_remainder_overflow_does_not_go_negative(self, n_native: int, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
        """train+val rounding up past N must not produce a negative test count.

        0.7 / 0.3 / 0.0 over 5 or 25 rows is the concrete crash: independently
        rounded train+val is 6 (or 26), remainder absorption assigns
        ``n_test = -1``, and ``split_three_way`` raises on a valid request.
        """
        counts = resolve_partition_counts(
            sizing_mode=SIZING_MODE_CARVE,
            n_native=n_native,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        assert min(counts["n_train"], counts["n_val"], counts["n_test"]) >= 0
        assert counts["n_total"] == n_native
        assert counts["n_train"] + counts["n_val"] + counts["n_test"] == n_native
        X = np.zeros((n_native, 2), dtype=np.float32)
        y = np.zeros((n_native, 1), dtype=np.float32)
        split = partition_and_assemble(X, y, counts, seed=0, shuffle=False)
        assert split["X_full"].shape[0] == n_native


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


class TestResourceBounds:
    """The two review findings on juniper-data#361, both confirmed before fixing."""

    def test_percentages_are_bounded(self) -> None:
        """Unbounded percentages are a resource-exhaustion vector (CWE-770).

        These are request parameters on a public endpoint and they MULTIPLY the
        rows a generator allocates: ``val_percent=1e6`` turned a 200-row request
        into a 4,000,200-row allocation.
        """
        with pytest.raises(ValueError):
            _Params(val_percent=1e6)
        with pytest.raises(ValueError):
            _Params(test_percent=1e6)

    def test_percentages_may_still_exceed_one_hundred(self) -> None:
        """The bound must not break "percentages are relative to train"."""
        params = _Params(val_percent=150.0, test_percent=200.0)

        assert params.val_percent == 150.0
        assert params.test_percent == 200.0

    def test_rescale_revalidates_against_the_generator_cap(self) -> None:
        """``model_copy`` does NOT re-validate in pydantic v2; rescale must.

        Additive sizing multiplies the size knob by 1.7 at the default
        breakdown, so a spiral request at the documented maximum lands at 17 000
        against a cap of 10 000 -- a value the same model rejects when built
        directly. Without re-validation the cap is simply bypassed.
        """
        params = SpiralParams(n_spirals=2, n_points_per_spiral=MAX_POINTS, seed=1)

        with pytest.raises(ValidationError):
            rescale_generator_params(params, n_points_per_spiral=MAX_POINTS + 1)

    def test_rescale_returns_a_validated_model_on_the_happy_path(self) -> None:
        params = SpiralParams(n_spirals=2, n_points_per_spiral=100, seed=1)
        rescaled = rescale_generator_params(params, n_points_per_spiral=170)

        assert isinstance(rescaled, SpiralParams)
        assert rescaled.n_points_per_spiral == 170
        assert rescaled.seed == 1, "unrelated fields must survive the rescale"

    def test_generator_refuses_a_request_that_would_exceed_its_cap(self) -> None:
        """End-to-end: the bypass becomes a validation error, not an allocation."""
        params = SpiralParams(n_spirals=2, n_points_per_spiral=MAX_POINTS, seed=1)

        with pytest.raises(ValidationError):
            SpiralGenerator.generate(params)
