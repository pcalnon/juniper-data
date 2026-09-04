#!/usr/bin/env python
"""Normaliser fit scope: train only, never the full set (juniper-data#314).

Three generators fit their feature normaliser on the FULL matrix -- including rows that
belong to the test partition -- and then applied those statistics to the training features.
For ``equities`` and ``equities_seq`` the test rows are chronologically LATER, so this was
look-ahead leakage: training features scaled by a maximum that exists only in their future.

Decision 7 of the ecosystem partition design requires the fit to come from ``train`` alone.

WHAT MAKES THESE TESTS ABLE TO FAIL. A full-matrix fit bounds *every* partition by
construction, so "train is within [0, 1]" holds under both the correct and the broken
implementation and proves nothing on its own. The discriminating assertion is the opposite
one: a test row that lands OUTSIDE [0, 1]. That can only happen when the statistics came
from a strictly smaller set of rows.

``csv_import`` is included because it had the same defect for a different reason -- it
normalised inside ``_load_and_preprocess``, which runs BEFORE the split exists, so a
train-only fit was not merely wrong there but impossible until the order was changed.
"""

import csv
import os
import pathlib
import tempfile

import numpy as np
import pytest

# Import the routes module first: juniper_data.generators.csv_import cannot be imported on
# its own (circular import via api.routes.generators -- juniper-data#316), and pytest may
# collect this file before anything else has completed that cycle.
import juniper_data.api.routes.generators  # noqa: F401  # isort: skip
from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams  # isort: skip

pytestmark = pytest.mark.unit


def _write_csv(directory: str, name: str = "d.csv") -> None:
    """Rows whose LATER half exceeds the earlier half's range on every feature.

    With ``shuffle=False`` the split is positional, so the test partition is exactly the
    high-valued tail -- which is what makes an out-of-range test row detectable.
    """
    path = pathlib.Path(directory, name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["f0", "f1", "label"])
        for i in range(20):
            writer.writerow([i, i * 2, i % 2])


@pytest.fixture(scope="module")
def csv_dir():
    """ONE directory for the whole module, and the settings cache cleared around it.

    ``csv_import`` resolves its import root through ``get_settings()``, which is cached, so a
    per-test temporary directory silently does not take effect: the first test's path is
    cached and every later test looks for its file in a directory that no longer exists. The
    symptom is a FileNotFoundError that looks like a fixture bug rather than a caching one.
    """
    from juniper_data.api.settings import get_settings

    previous = os.environ.get("JUNIPER_DATA_IMPORT_DIR")
    with tempfile.TemporaryDirectory(prefix="juniper_fitscope_") as tmp:
        _write_csv(tmp)
        _write_constant_csv(tmp)
        _write_single_row_csv(tmp)
        os.environ["JUNIPER_DATA_IMPORT_DIR"] = tmp
        if hasattr(get_settings, "cache_clear"):
            get_settings.cache_clear()
        try:
            yield tmp
        finally:
            if previous is None:
                os.environ.pop("JUNIPER_DATA_IMPORT_DIR", None)
            else:
                os.environ["JUNIPER_DATA_IMPORT_DIR"] = previous
            if hasattr(get_settings, "cache_clear"):
                get_settings.cache_clear()


def _write_constant_csv(directory: str) -> None:
    """A dataset with a zero-range feature, to exercise the divide-by-zero guard."""
    path = pathlib.Path(directory, "const.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["f0", "f1", "label"])
        for i in range(10):
            writer.writerow([5, i, i % 2])


def _write_single_row_csv(directory: str) -> None:
    """One row so ``round(n * train_ratio)`` can empty the training partition."""
    path = pathlib.Path(directory, "single.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["f0", "f1", "label"])
        writer.writerow([10, 20, 1])


class TestCsvImportFitScope:
    def test_train_is_bounded(self, csv_dir):
        out = CsvImportGenerator.generate(CsvImportParams(file_path="d.csv", label_column="label", normalize_features=True, shuffle=False, train_ratio=0.5, test_ratio=0.5))
        train = out["X_train"]
        assert train.min() >= -1e-6 and train.max() <= 1.0 + 1e-6

    def test_test_partition_escapes_the_bound(self, csv_dir):
        """THE discriminating assertion -- a full-matrix fit would clamp this to 1.0."""
        out = CsvImportGenerator.generate(CsvImportParams(file_path="d.csv", label_column="label", normalize_features=True, shuffle=False, train_ratio=0.5, test_ratio=0.5))
        assert out["X_test"].max() > 1.0 + 1e-6

    def test_disabled_normalisation_leaves_values_raw(self, csv_dir):
        """``normalize_features`` defaults False; the common path must not move."""
        out = CsvImportGenerator.generate(CsvImportParams(file_path="d.csv", label_column="label", shuffle=False))
        assert out["X_train"].max() > 1.0

    def test_shapes_and_partition_sizes_are_unchanged(self, csv_dir):
        """The fix reorders normalisation; it must not change what is split or how much."""
        common = {"file_path": "d.csv", "label_column": "label", "shuffle": False, "train_ratio": 0.5, "test_ratio": 0.5}
        raw = CsvImportGenerator.generate(CsvImportParams(**common))
        normed = CsvImportGenerator.generate(CsvImportParams(**common, normalize_features=True))
        for key in ("X_train", "X_test", "X_full", "y_train", "y_test", "y_full"):
            assert raw[key].shape == normed[key].shape, f"{key} shape moved"

    def test_a_constant_column_does_not_divide_by_zero(self, csv_dir):
        """A zero-range feature gets a range of 1 rather than producing inf/nan."""
        out = CsvImportGenerator.generate(CsvImportParams(file_path="const.csv", label_column="label", normalize_features=True, shuffle=False))
        assert np.isfinite(out["X_train"]).all()
        assert np.isfinite(out["X_test"]).all()

    def test_empty_training_partition_falls_back_to_full_without_nan(self, csv_dir):
        """1 row + train_ratio 0.4 rounds to zero train rows: fit on full, stay finite.

        ``X_train if X_train.shape[0] else X_full`` is the empty-train guard. A full-set
        fit here is not a leak — there is no training partition to contaminate.
        """
        out = CsvImportGenerator.generate(CsvImportParams(file_path="single.csv", label_column="label", normalize_features=True, shuffle=False, train_ratio=0.4, test_ratio=0.6))
        assert out["X_train"].shape[0] == 0
        assert out["X_test"].shape[0] == 1
        assert np.isfinite(out["X_test"]).all()
        assert np.isfinite(out["X_full"]).all()
        assert out["X_test"].min() >= -1e-6 and out["X_test"].max() <= 1.0 + 1e-6

    def test_empty_test_partition_is_applied_without_error(self, csv_dir):
        """``test_ratio=0`` yields an empty X_test; ``_apply_minmax`` must pass it through."""
        out = CsvImportGenerator.generate(CsvImportParams(file_path="d.csv", label_column="label", normalize_features=True, shuffle=False, train_ratio=1.0, test_ratio=0.0))
        assert out["X_test"].shape[0] == 0
        train = out["X_train"]
        assert train.min() >= -1e-6 and train.max() <= 1.0 + 1e-6


class TestCsvImportMinmaxHelpers:
    """Direct coverage of the helpers ``generate`` gained when the split moved first.

    Unlike the equities ``_raw_features`` helper (which needs a conditioned frame schema),
    these operate on plain float32 matrices, so they can be asserted without a skip-on-mismatch
    try/except.
    """

    def test_fit_uses_only_the_rows_it_is_given(self):
        """THE helper-level discriminating assertion — later rows must not enter the stats."""
        train = np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32)
        later = np.array([[10.0, 20.0]], dtype=np.float32)
        minimum, scale = CsvImportGenerator._fit_minmax(train)
        np.testing.assert_array_equal(minimum, np.array([[0.0, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(scale, np.array([[2.0, 4.0]], dtype=np.float32))
        applied_train = CsvImportGenerator._apply_minmax(train, minimum, scale)
        assert applied_train.min() >= -1e-6 and applied_train.max() <= 1.0 + 1e-6
        applied_later = CsvImportGenerator._apply_minmax(later, minimum, scale)
        assert applied_later.max() > 1.0 + 1e-6

    def test_apply_passes_empty_matrix_through(self):
        empty = np.zeros((0, 2), dtype=np.float32)
        minimum = np.array([[0.0, 0.0]], dtype=np.float32)
        scale = np.array([[1.0, 1.0]], dtype=np.float32)
        out = CsvImportGenerator._apply_minmax(empty, minimum, scale)
        assert out.shape == (0, 2)
        np.testing.assert_array_equal(out, empty)

    def test_fit_zero_range_column_uses_scale_of_one(self):
        matrix = np.array([[5.0, 0.0], [5.0, 2.0]], dtype=np.float32)
        minimum, scale = CsvImportGenerator._fit_minmax(matrix)
        assert scale[0, 0] == 1.0
        applied = CsvImportGenerator._apply_minmax(matrix, minimum, scale)
        assert np.isfinite(applied).all()
        assert applied[0, 0] == 0.0


# NOTE: the flat equities fit scope is asserted END TO END in
# ``test_equities_generator.py`` (``test_normaliser_is_fit_on_train_not_full``).
# The windowed sibling has its own independent fit (``temporal_split_index`` then
# concatenated training rows) and is asserted in
# ``test_equities_seq_generator.py`` (``test_normaliser_is_fit_on_train_not_full``).
#
# A helper-level equities fit-scope test was drafted and REMOVED rather than shipped:
# constructing a frame by hand did not satisfy ``_raw_features``' expected columns, so it
# could only be written with a try/except that skipped on schema mismatch -- a test that
# reports success without executing its assertion. That is the failure mode this whole arc
# keeps finding, and it is not worth adding one deliberately for coverage of a path already
# covered end to end.
