"""Guards for leftovers the #361 sizing suite cannot see.

``test_partition_sizing.py`` pins the two models and the mixin stubs.
``test_normaliser_fit_scope.py`` pins train-only fit, but every case there sets
``val_ratio=0.0``. ``test_split.py`` pins the cut itself. None of those can
catch:

* a real-data params model that mixes ``PartitionParams`` (carve default, but
  additive still accepted and then silently carved);
* ``resolve_counts_for_params`` dropping ``val_ratio`` via the getattr default
  of 0.0 (the additive resolve tests never read carve ratios);
* overflow trimmed from val first, or only from test, so long as train is 80
  and the total fits;
* ``X_full`` assembled from the raw array rather than the three partitions
  (shape identity still holds);
* surplus taken from the raw last-class block rather than the shuffled tail;
* ``csv_import`` applying train-fit stats to train/test/full but leaving a
  non-empty ``X_val`` on the raw scale.
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.core.limits import (
    CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION,
    CSV_IMPORT_DEFAULT_MAX_BYTES,
)
from juniper_data.core.split import (
    SIZING_MODE_ADDITIVE,
    SIZING_MODE_CARVE,
    partition_and_assemble,
    resolve_counts_for_params,
    resolve_partition_counts,
)
from juniper_data.generators.arc_agi.params import ArcAgiParams
from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams
from juniper_data.generators.mnist.params import MnistParams

pytestmark = [pytest.mark.unit]


def _blocked_arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Class-blocked rows with a unique id in column 0.

    First half is class 0, second half class 1. Dropping the raw tail therefore
    removes only class 1 -- the property surplus-from-shuffled-tail must not
    have.
    """
    X = np.zeros((n, 2), dtype=np.float32)
    X[:, 0] = np.arange(n, dtype=np.float32)
    X[n // 2 :, 1] = 1.0
    y = np.zeros((n, 2), dtype=np.float32)
    y[: n // 2, 0] = 1.0
    y[n // 2 :, 1] = 1.0
    return X, y


class TestRealDataParamsRejectAdditive:
    """The mixin stub cannot see a generator that kept a carve default."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: MnistParams(sizing_mode=SIZING_MODE_ADDITIVE),
            lambda: ArcAgiParams(sizing_mode=SIZING_MODE_ADDITIVE),
            lambda: CsvImportParams(file_path="data.csv", sizing_mode=SIZING_MODE_ADDITIVE),
        ],
        ids=["mnist", "arc_agi", "csv_import"],
    )
    def test_asking_for_additive_raises(self, factory) -> None:
        with pytest.raises(ValueError, match="not available for this generator"):
            factory()


class TestResolveCountsReadsCarveValRatio:
    """Additive resolve tests never read ``val_ratio``; a getattr miss is silent."""

    @pytest.mark.parametrize(
        "params",
        [
            MnistParams(),
            ArcAgiParams(),
            CsvImportParams(file_path="data.csv"),
        ],
        ids=["mnist", "arc_agi", "csv_import"],
    )
    def test_default_carve_is_800_100_100(self, params) -> None:
        counts = resolve_counts_for_params(params, 1000)

        assert (counts["n_train"], counts["n_val"], counts["n_test"]) == (800, 100, 100)
        assert counts["n_raw_required"] == 1000


class TestOverflowTrimIsTwoStageAndExact:
    """``test_oversubscription_is_trimmed_from_the_end`` allows val-first trim."""

    def test_test_is_exhausted_before_val_is_cut(self) -> None:
        counts = resolve_partition_counts(
            sizing_mode=SIZING_MODE_CARVE,
            n_native=100,
            train_ratio=0.8,
            val_ratio=0.3,
            test_ratio=0.3,
        )

        assert (counts["n_train"], counts["n_val"], counts["n_test"]) == (80, 20, 0)

    def test_val_is_still_cut_when_test_is_already_empty(self) -> None:
        counts = resolve_partition_counts(
            sizing_mode=SIZING_MODE_CARVE,
            n_native=100,
            train_ratio=0.8,
            val_ratio=0.4,
            test_ratio=0.0,
        )

        assert (counts["n_train"], counts["n_val"], counts["n_test"]) == (80, 20, 0)


class TestAssembleFromPartitionsNotRaw:
    def test_full_is_the_vstack_of_the_three_partitions(self) -> None:
        counts = resolve_partition_counts(sizing_mode=SIZING_MODE_ADDITIVE, n_native=50)
        X, y = _blocked_arrays(counts["n_raw_required"] + 7)

        result = partition_and_assemble(X, y, counts, seed=42, shuffle=True)

        np.testing.assert_array_equal(
            result["X_full"],
            np.vstack([result["X_train"], result["X_val"], result["X_test"]]),
        )
        np.testing.assert_array_equal(
            result["y_full"],
            np.vstack([result["y_train"], result["y_val"], result["y_test"]]),
        )

    def test_empty_val_still_assembles(self) -> None:
        counts = resolve_partition_counts(
            sizing_mode=SIZING_MODE_CARVE,
            n_native=10,
            train_ratio=0.8,
            val_ratio=0.0,
            test_ratio=0.2,
        )
        X, y = _blocked_arrays(10)

        result = partition_and_assemble(X, y, counts, seed=None, shuffle=False)

        assert result["X_val"].shape == (0, 2)
        np.testing.assert_array_equal(
            result["X_full"],
            np.vstack([result["X_train"], result["X_val"], result["X_test"]]),
        )

    def test_surplus_is_the_shuffled_tail_not_the_last_class(self) -> None:
        """Last-class drop would keep every class-0 row and drop 20 class-1 rows."""
        n = 100
        surplus = 20
        counts = {
            "n_train": 40,
            "n_val": 20,
            "n_test": 20,
            "n_total": n - surplus,
        }
        X, y = _blocked_arrays(n)

        result = partition_and_assemble(X, y, counts, seed=42, shuffle=True)

        used_ids = set(result["X_full"][:, 0].tolist())
        raw_tail_ids = set(X[-surplus:, 0].tolist())

        assert used_ids & raw_tail_ids, "raw tail (all class 1) was dropped wholesale"
        class0 = int(result["y_full"][:, 0].sum())
        assert class0 < n // 2, "class 0 was never charged for the surplus"


class TestCsvImportValUsesTrainFitStats:
    """Fit-scope cases set ``val_ratio=0.0``, so a skipped ``X_val`` apply is green."""

    def test_nonempty_val_is_scaled_by_the_train_fit(self, tmp_path: Path) -> None:
        path = tmp_path / "d.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["f0", "f1", "label"])
            for i in range(20):
                writer.writerow([i, i * 2, i % 2])

        mock_settings = MagicMock()
        mock_settings.import_dir = str(tmp_path)
        mock_settings.csv_import_max_bytes = CSV_IMPORT_DEFAULT_MAX_BYTES
        mock_settings.csv_import_allow_truncation = CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION

        with patch(
            "juniper_data.generators.csv_import.generator.get_settings",
            return_value=mock_settings,
        ):
            out = CsvImportGenerator.generate(
                CsvImportParams(
                    file_path="d.csv",
                    label_column="label",
                    normalize_features=True,
                    shuffle=False,
                    train_ratio=0.5,
                    val_ratio=0.2,
                    test_ratio=0.3,
                )
            )

        assert out["X_val"].shape[0] == 4

        raw_val = np.array([[10, 20], [11, 22], [12, 24], [13, 26]], dtype=np.float32)
        train_min = np.array([[0.0, 0.0]], dtype=np.float32)
        train_scale = np.array([[9.0, 18.0]], dtype=np.float32)
        expected = ((raw_val - train_min) / train_scale).astype(np.float32)

        assert not np.allclose(out["X_val"], raw_val), "X_val was left on the raw scale"
        np.testing.assert_allclose(out["X_val"], expected, rtol=1e-5, atol=1e-6)
        assert out["X_val"].max() > 1.0 + 1e-6, "X_val was fit on itself or the full set"
        np.testing.assert_array_equal(
            out["X_full"],
            np.vstack([out["X_train"], out["X_val"], out["X_test"]]),
        )
