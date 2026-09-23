"""Every generator's artifact must load with ``allow_pickle=False`` (juniper-data#429).

``np.savez`` and ``np.savez_compressed`` PICKLE any object-dtype array, and every consumer
loads with numpy's default ``allow_pickle=False``: juniper-data-client's
``download_artifact_npz`` materialises every key, and so does this service's own cached store
when it populates its cache. One object array therefore makes the WHOLE artifact unloadable,
not just the one key.

arc_agi's ``task_ids`` was exactly that. The extras mechanism re-added the key after decision 11
as ``np.array(task_ids, dtype=object)``, and no consumer could download an arc_agi artifact
from then on. Nothing caught it, because every generator test asserts on the returned dict and
none of them round-trips it through a loader.

This is a FLEET check, like ``TestEveryGeneratorBumpedForDecision11``. It enumerates
``GENERATOR_REGISTRY`` rather than naming generators, so a generator added later is covered the
day it lands. A generator whose default params need a network or a file gets an offline builder
in ``_SOURCED_BUILDERS``; every other one is built from its default params, the way a bare
``POST /v1/datasets`` would build it.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.api.routes.generators import GENERATOR_REGISTRY
from juniper_data.core.limits import CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION, CSV_IMPORT_DEFAULT_MAX_BYTES
from juniper_data.core.meta import pop_data_quality_meta, pop_scaling_meta, pop_truncation_meta

pytestmark = [pytest.mark.unit, pytest.mark.generators]

_PARTITION_KEYS = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")


def _route_arrays(arrays: dict[str, Any]) -> dict[str, np.ndarray]:
    """Strip the reserved non-array channels exactly as ``POST /v1/datasets`` does before it persists."""
    pop_scaling_meta(arrays)
    pop_truncation_meta(arrays)
    pop_data_quality_meta(arrays)
    return arrays


def _load_like_every_consumer(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Write as the stores do (``np.savez_compressed``), then read back EVERY key with ``allow_pickle=False``.

    ``allow_pickle=False`` is passed explicitly rather than inherited from numpy's default, so
    this check keeps its meaning even if a future numpy changes the default.
    """
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]  # numpy stubs incomplete for **kwargs
    with np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


# --- offline builders for the generators whose default params reach a network or a file -----------


def _build_arc_agi(tmp_path: Path) -> dict[str, Any]:
    from juniper_data.generators.arc_agi import ArcAgiGenerator, ArcAgiParams

    training = tmp_path / "arc" / "training"
    training.mkdir(parents=True)
    for index, name in enumerate(("task_a", "task_bb", "task_ccc")):
        task = {
            "train": [{"input": [[index, 1], [2, 3]], "output": [[3, 2], [1, index]]}] * 3,
            "test": [{"input": [[index]], "output": [[index]]}],
        }
        (training / f"{name}.json").write_text(json.dumps(task), encoding="utf-8")
    return ArcAgiGenerator.generate(ArcAgiParams(source="local", local_path=str(tmp_path / "arc"), subset="training", pad_to=4, seed=7))


def _build_csv_import(tmp_path: Path) -> dict[str, Any]:
    from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams

    with (tmp_path / "data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["f0", "f1", "label"])
        for i in range(40):
            writer.writerow([i, i * 2, i % 2])

    settings = MagicMock()
    settings.import_dir = str(tmp_path)
    settings.csv_import_max_bytes = CSV_IMPORT_DEFAULT_MAX_BYTES
    settings.csv_import_allow_truncation = CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION
    with patch("juniper_data.generators.csv_import.generator.get_settings", return_value=settings):
        return CsvImportGenerator.generate(CsvImportParams(file_path="data.csv", label_column="label", normalize_features=True, seed=7))


def _build_mnist(_tmp_path: Path) -> dict[str, Any]:
    from juniper_data.generators.mnist import MnistGenerator, MnistParams

    n_samples = 30
    images = np.random.default_rng(0).integers(0, 255, (n_samples, 28, 28), dtype=np.uint8)
    labels = np.arange(n_samples) % 10

    formatted = MagicMock()
    formatted.__getitem__ = MagicMock(side_effect=lambda key: labels if key == "label" else images)
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=n_samples)
    dataset.shuffle.return_value = dataset
    dataset.select.return_value = dataset
    dataset.with_format.return_value = formatted

    with patch("juniper_data.generators.mnist.generator.HF_AVAILABLE", True), patch("juniper_data.generators.mnist.generator.hf_load_dataset", MagicMock(return_value=dataset)):
        return MnistGenerator.generate(MnistParams(n_samples=n_samples, seed=7))


@contextmanager
def _equities_sources(start: str, periods: int, filed: tuple[str, str]) -> Iterator[None]:
    """Serve synthetic OHLCV and SEC share history in place of yfinance and EDGAR."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("yfinance")
    from juniper_data.generators.equities import generator as eq_gen

    def ohlcv(seed: int):
        index = pd.bdate_range(start=start, periods=periods)
        rng = np.random.default_rng(seed)
        walk = rng.normal(0.1, 1.0, periods).cumsum()
        close = 100.0 + walk - walk.min() + 1.0
        return pd.DataFrame(
            {
                "Open": close + rng.normal(0.0, 0.2, periods),
                "High": close + np.abs(rng.normal(0.5, 0.2, periods)),
                "Low": close - np.abs(rng.normal(0.5, 0.2, periods)),
                "Close": close,
                "Adj Close": close,
                "Volume": rng.integers(1_000_000, 5_000_000, periods).astype(float),
            },
            index=index,
        )

    frames = {"AAPL": ohlcv(1), "MSFT": ohlcv(2)}
    shares = pd.DataFrame(
        {"shares": [1_000_000_000.0, 1_100_000_000.0], "filed": [pd.Timestamp(filed[0]), pd.Timestamp(filed[1])]},
        index=pd.to_datetime([pd.Timestamp(start) - pd.Timedelta(days=5), pd.Timestamp(filed[1]) - pd.Timedelta(days=30)]),
    )

    def fake_download(symbol, **_kwargs):
        frame = frames.get(symbol)
        return frame.copy() if frame is not None else pd.DataFrame()

    with patch.object(eq_gen.yf, "download", side_effect=fake_download), patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(lambda _cik, _use_cache: shares)):
        yield


def _build_equities(_tmp_path: Path) -> dict[str, Any]:
    from juniper_data.generators.equities import EquitiesGenerator, EquitiesParams

    with _equities_sources("2008-01-01", 600, ("2008-01-04", "2009-01-15")):
        return EquitiesGenerator.generate(EquitiesParams(symbols=["AAPL", "MSFT"], start_date="2008-01-01", end_date="2011-01-01", use_cache=False))


def _build_equities_seq(_tmp_path: Path) -> dict[str, Any]:
    from juniper_data.generators.equities_seq import EquitiesSeqGenerator, EquitiesSeqParams

    with _equities_sources("2008-01-01", 400, ("2008-01-04", "2009-01-15")):
        return EquitiesSeqGenerator.generate(EquitiesSeqParams(symbols=["AAPL", "MSFT"], start_date="2008-01-01", end_date="2011-01-01", use_cache=False, lookback=5))


_SOURCED_BUILDERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "arc_agi": _build_arc_agi,
    "csv_import": _build_csv_import,
    "equities": _build_equities,
    "equities_seq": _build_equities_seq,
    "mnist": _build_mnist,
}


def _build(name: str, tmp_path: Path) -> dict[str, np.ndarray]:
    builder = _SOURCED_BUILDERS.get(name)
    if builder is not None:
        return _route_arrays(builder(tmp_path))
    info = GENERATOR_REGISTRY[name]
    return _route_arrays(info["generator"].generate(info["params_class"]()))


class TestTheCheckCanFail:
    """Without these the fleet test below could pass by checking nothing."""

    def test_an_object_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match="allow_pickle=False"):
            _load_like_every_consumer({"X_train": np.zeros((2, 2), np.float32), "task_ids": np.array(["a", "b"], dtype=object)})

    def test_a_unicode_array_is_loaded_intact(self) -> None:
        loaded = _load_like_every_consumer({"task_ids": np.array(["a", "bb"], dtype=np.str_)})
        assert loaded["task_ids"].dtype.kind == "U"
        assert loaded["task_ids"].tolist() == ["a", "bb"]

    def test_every_sourced_builder_names_a_registered_generator(self) -> None:
        """A builder left behind by a renamed generator would silently stop covering anything."""
        assert set(_SOURCED_BUILDERS) <= set(GENERATOR_REGISTRY)


class TestEveryGeneratorArtifactLoadsWithoutPickle:
    @pytest.mark.parametrize("name", sorted(GENERATOR_REGISTRY))
    def test_the_artifact_round_trips_through_an_allow_pickle_false_load(self, name: str, tmp_path: Path) -> None:
        arrays = _build(name, tmp_path)

        missing = [key for key in _PARTITION_KEYS if key not in arrays]
        assert not missing, f"{name}: the builder produced no {missing}; the check below would have covered a non-artifact"

        pickled = {key: str(value.dtype) for key, value in arrays.items() if value.dtype.hasobject}
        assert not pickled, f"{name}: {pickled} would be PICKLED by np.savez, and every consumer loads with allow_pickle=False, so no consumer could load this artifact at all. Emit a fixed-width dtype instead (np.str_ for text, as equities does for ticker_vocab)."

        loaded = _load_like_every_consumer(arrays)
        assert sorted(loaded) == sorted(arrays)
        for key, value in arrays.items():
            assert loaded[key].dtype == value.dtype, f"{name}: {key} came back as {loaded[key].dtype}, not {value.dtype}"
            np.testing.assert_array_equal(loaded[key], value, err_msg=f"{name}: {key} did not survive the round trip")
