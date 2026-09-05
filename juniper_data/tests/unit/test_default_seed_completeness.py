#!/usr/bin/env python
"""The #319/#322 default-seed contract for the generators SEEDED_2D missed.

``test_default_generator_seed.py`` generate-tests the 2-D synthetics. #322 also
defaulted ``csv_import``, ``arc_agi``, and ``equities``. ``csv_import`` is the
one that matters: it shuffles by default, and its existing ``test_default_values``
never mentions ``seed``. Reverting that default to ``None`` recreates the original
defect -- two identical imports of the same file produce different splits -- and
every existing csv_import test still passes, because they either pin an explicit
seed or never compare two default-config calls.

``equities`` documents that its seed is unused for the temporal split; the params
pin is still load-bearing so a revert to ``None`` fails. ``mnist`` already asserts
``seed == 42``.
"""

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.core.constants import DEFAULT_GENERATOR_SEED
from juniper_data.core.limits import CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION, CSV_IMPORT_DEFAULT_MAX_BYTES
from juniper_data.generators.arc_agi.params import ArcAgiParams
from juniper_data.generators.equities.params import EquitiesParams

# Load-bearing side effect, NOT an unused import. ``juniper_data.generators.csv_import``
# cannot be imported on its own -- it hits a circular import through
# ``api.routes.generators`` (juniper-data#316) -- until create_app is deferred.
importlib.import_module("juniper_data.api.routes.generators")

from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams  # noqa: E402

pytestmark = pytest.mark.unit

N_ROWS = 40


@pytest.fixture
def import_dir(tmp_path: Path):
    """Patch get_settings to a temporary import directory with real cap values."""
    mock_settings = MagicMock()
    mock_settings.import_dir = str(tmp_path)
    mock_settings.csv_import_max_bytes = CSV_IMPORT_DEFAULT_MAX_BYTES
    mock_settings.csv_import_allow_truncation = CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION
    with patch("juniper_data.generators.csv_import.generator.get_settings", return_value=mock_settings):
        yield tmp_path


def _write_distinct_csv(directory: Path, name: str = "rows.csv") -> str:
    """Write enough uniquely-valued rows that a reshuffle is visible."""
    lines = ["feature1,feature2,label"]
    for i in range(N_ROWS):
        lines.append(f"{float(i)},{float(i + 100)},{'A' if i % 2 == 0 else 'B'}")
    (directory / name).write_text("\n".join(lines) + "\n")
    return name


class TestCsvImportDefaultSeed:
    def test_default_seed_is_the_shared_constant(self):
        """THE missing pin. ``test_default_values`` never mentions seed."""
        params = CsvImportParams(file_path="rows.csv")
        assert params.seed == DEFAULT_GENERATOR_SEED
        assert params.seed is not None

    def test_default_shuffle_is_true(self):
        """If shuffle defaults to False the seed is inert and the generate pins go vacuous."""
        assert CsvImportParams(file_path="rows.csv").shuffle is True

    def test_two_calls_at_defaults_are_identical(self, import_dir: Path):
        """The #319 property, on the generator that consumes a caller-supplied file."""
        name = _write_distinct_csv(import_dir)
        first = CsvImportGenerator.generate(CsvImportParams(file_path=name))
        second = CsvImportGenerator.generate(CsvImportParams(file_path=name))
        for key in ("X_train", "y_train", "X_test", "y_test"):
            assert np.array_equal(first[key], second[key]), f"{key} differs between two default-config imports"

    def test_explicit_none_still_draws_fresh(self, import_dir: Path):
        """The escape hatch must survive defaulting the seed."""
        name = _write_distinct_csv(import_dir)
        first = CsvImportGenerator.generate(CsvImportParams(file_path=name, seed=None))["X_train"]
        second = CsvImportGenerator.generate(CsvImportParams(file_path=name, seed=None))["X_train"]
        assert not np.array_equal(first, second), "explicit seed=None should still reshuffle the import"


class TestRemainingParamsDefaults:
    def test_arc_agi_default_seed_is_the_shared_constant(self):
        assert ArcAgiParams().seed == DEFAULT_GENERATOR_SEED
        assert ArcAgiParams().seed is not None

    def test_equities_default_seed_is_the_shared_constant(self):
        """Unused for the temporal split, but #322 still defaulted it for API parity."""
        assert EquitiesParams().seed == DEFAULT_GENERATOR_SEED
        assert EquitiesParams().seed is not None
