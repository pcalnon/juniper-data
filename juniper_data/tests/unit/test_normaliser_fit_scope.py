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

from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams

# NOTE: this used to require pre-importing ``juniper_data.api.routes.generators`` for its side
# effect, because ``csv_import`` could not be imported on its own (juniper-data#316). That
# workaround is gone -- the cycle was broken by deferring ``create_app`` in
# ``juniper_data/api/__init__.py``. If this import starts failing again, the cycle is back.

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


# NOTE: the two equities generators are asserted END TO END in their own suites, each against
# the real conditioned-frame schema via that module's mocked-download helpers:
#
#   * equities      -> test_equities_generator.py::test_normaliser_is_fit_on_train_not_full
#   * equities_seq  -> test_equities_seq_generator.py::
#                        test_normaliser_is_fit_on_train_rows_not_the_full_frame
#
# An earlier revision of this comment claimed the equities test covered BOTH. It does not --
# they are separate generators with separate fit paths, and equities_seq had no fit-scope
# coverage at all until the test named above was added. Corrected after review.
#
# A helper-level test was drafted here and REMOVED rather than shipped: constructing a frame
# by hand did not satisfy ``_raw_features``' expected columns, so it could only be written
# with a try/except that skipped on schema mismatch -- a test that reports success without
# executing its assertion. That is the failure mode this arc keeps finding, and it is not
# worth adding one deliberately for coverage of a path already covered end to end.
