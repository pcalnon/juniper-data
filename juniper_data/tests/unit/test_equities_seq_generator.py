"""Unit tests for the windowed (3-D sequence) equities generator (WS-1 / #168).

Network sources are mocked (same approach as ``test_equities_generator``), and
the synthetic dates use ``pd.bdate_range`` so weekend gaps (Fri -> Mon) are real
3-calendar-day jumps -- i.e. the irregular Δt is exercised end to end. Asserts
the additive 3-D contract: ``X`` is ``(W, L, F)``; ``dt`` is the per-step
calendar-day gap matching the step dates; all contract keys present;
``observed_mask`` all-ones; targets aligned; ``full`` == ``train`` + ``test``;
and the windowing leakage guarantee (per ticker, every train target precedes
every test target).
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_equities_seq_generator.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import datetime as _dt
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from juniper_data.generators.equities import generator as eq_gen  # noqa: E402
from juniper_data.generators.equities_seq import EquitiesSeqGenerator, EquitiesSeqParams, get_schema  # noqa: E402
from juniper_data.generators.equities_seq import generator as esq_gen  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _ohlcv(start: str = "2008-01-01", periods: int = 400, seed: int = 0):
    """Synthetic daily OHLCV on business days (weekend gaps -> irregular Δt)."""
    index = pd.bdate_range(start=start, periods=periods)
    rng = np.random.default_rng(seed)
    walk = rng.normal(0.1, 1.0, periods).cumsum()
    close = 100.0 + walk - walk.min() + 1.0
    high = close + np.abs(rng.normal(0.5, 0.2, periods))
    low = close - np.abs(rng.normal(0.5, 0.2, periods))
    open_ = close + rng.normal(0.0, 0.2, periods)
    volume = rng.integers(1_000_000, 5_000_000, periods).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": volume}, index=index)


def _shares(start: str = "2009-06-30"):
    series = pd.Series({pd.Timestamp(start): 1_000_000_000.0, pd.Timestamp("2010-06-30"): 1_100_000_000.0})
    series.index = pd.to_datetime(series.index)
    return series


@contextmanager
def _mocked(ohlcv_map: dict, shares):
    """Patch yfinance.download and SEC share fetching (reused by the sibling pipeline)."""

    def fake_download(symbol, **_kwargs):
        frame = ohlcv_map.get(symbol)
        return frame.copy() if frame is not None else pd.DataFrame()

    def fake_shares(_cik, _use_cache):
        return shares

    with patch.object(eq_gen.yf, "download", side_effect=fake_download), patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(fake_shares)):
        yield


def _generate(symbols, ohlcv_map, shares=None, *, lookback: int = 5, **overrides):
    params = EquitiesSeqParams(symbols=symbols, start_date="2008-01-01", end_date="2011-01-01", use_cache=False, lookback=lookback, **overrides)
    with _mocked(ohlcv_map, shares):
        return EquitiesSeqGenerator.generate(params)


def _ord(yyyymmdd: int) -> int:
    v = int(yyyymmdd)
    return _dt.date(v // 10000, (v // 100) % 100, v % 100).toordinal()


class TestEquitiesSeqGenerator:
    """End-to-end behavior of EquitiesSeqGenerator.generate()."""

    def test_contract_keys_and_3d_shapes(self) -> None:
        lookback = 6
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=1), "MSFT": _ohlcv(seed=2)}, _shares(), lookback=lookback)
        for split in ("train", "test", "full"):
            for key in ("X", "y", "y_reg", "date", "dt", "target_dt", "window_end_date", "ticker_code", "observed_mask"):
                assert f"{key}_{split}" in arrays, f"missing {key}_{split}"
        assert "ticker_vocab" in arrays

        n_windows = arrays["X_full"].shape[0]
        assert arrays["X_full"].shape == (n_windows, lookback, 10)
        assert arrays["X_full"].dtype == np.float32
        assert arrays["y_full"].shape == (n_windows, 2)
        assert arrays["y_reg_full"].shape == (n_windows, 1)
        assert arrays["date_full"].shape == (n_windows, lookback)
        assert arrays["dt_full"].shape == (n_windows, lookback)
        assert arrays["target_dt_full"].shape == (n_windows,)
        assert arrays["window_end_date_full"].shape == (n_windows,)
        assert arrays["observed_mask_full"].shape == (n_windows, lookback)
        assert arrays["ticker_vocab"].tolist() == ["AAPL", "MSFT"]

    def test_full_equals_train_plus_test(self) -> None:
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=3)}, _shares(), lookback=5)
        assert arrays["X_full"].shape[0] == arrays["X_train"].shape[0] + arrays["X_test"].shape[0]

    def test_dt_is_calendar_gap_with_weekend_jumps(self) -> None:
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=4)}, _shares(), lookback=5)
        dt = arrays["dt_full"]
        assert dt.dtype == np.float32
        assert np.all(dt[:, 0] == 0)  # first step has no predecessor
        assert np.all(dt[:, 1:] > 0)  # strictly positive gaps
        # business-day cadence => 1-day gaps within a week, 3-day gaps over weekends
        assert np.any(np.isclose(dt[:, 1:], 3.0)), "expected Fri->Mon weekend gaps of 3 calendar days"
        # dt must equal the diff of the per-step date ordinals (notes I3).
        ords = np.vectorize(_ord)(arrays["date_full"])
        np.testing.assert_allclose(dt[:, 1:], np.diff(ords, axis=1).astype(np.float32))

    def test_observed_mask_all_ones(self) -> None:
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=5)}, _shares(), lookback=4)
        assert np.all(arrays["observed_mask_full"] == 1)
        assert arrays["observed_mask_full"].dtype == np.uint8

    def test_targets_onehot_and_positive_horizon(self) -> None:
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=6)}, _shares(), lookback=5)
        assert np.allclose(arrays["y_full"].sum(axis=1), 1.0)  # valid one-hot
        assert np.all(arrays["target_dt_full"] > 0)  # forecast horizon strictly positive

    def test_no_future_leak_per_ticker(self) -> None:
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=7), "MSFT": _ohlcv(seed=8)}, _shares(), lookback=6, train_ratio=0.7)
        for code in range(len(arrays["ticker_vocab"])):
            tr = arrays["ticker_code_train"] == code
            te = arrays["ticker_code_test"] == code
            if not tr.any() or not te.any():
                continue
            tr_targets = np.vectorize(_ord)(arrays["window_end_date_train"][tr]) + arrays["target_dt_train"][tr].astype(np.int64)
            te_targets = np.vectorize(_ord)(arrays["window_end_date_test"][te]) + arrays["target_dt_test"][te].astype(np.int64)
            assert tr_targets.max() < te_targets.min(), "every train target must precede every test target (no leak)"

    def test_get_schema_includes_lookback(self) -> None:
        schema = get_schema()
        assert "lookback" in schema["properties"]

    def test_regression_target_flows_through(self) -> None:
        # equities_seq inherits regression_target; log_return -> stationary y_reg.
        frame = _ohlcv(seed=9)
        nc = _generate(["AAPL"], {"AAPL": frame}, _shares(), lookback=5)
        lr = _generate(["AAPL"], {"AAPL": frame}, _shares(), lookback=5, regression_target="log_return")
        assert nc["y_reg_full"].shape == lr["y_reg_full"].shape
        assert "regression_target" in get_schema()["properties"]
        # next_close tracks the ~100 price level; log returns are centered near zero.
        assert abs(float(lr["y_reg_full"].mean())) < abs(float(nc["y_reg_full"].mean()))
        assert float(np.abs(lr["y_reg_full"]).mean()) < 1.0


class TestEquitiesSeqGeneratorBranches:
    """Cover generate()'s guard / skip / normalize branches (offline, deterministic)."""

    def test_generate_raises_without_equities_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(esq_gen, "EQUITIES_DEPS_AVAILABLE", False)
        with pytest.raises(ImportError, match="equities"):
            EquitiesSeqGenerator.generate(EquitiesSeqParams(symbols=["AAPL"], lookback=5))

    def test_generate_skips_ticker_whose_download_raises(self) -> None:
        good = _ohlcv(seed=20)

        def fake_download(symbol, **_kwargs):
            if symbol == "MSFT":
                raise RuntimeError("download exploded")
            return good.copy()

        with patch.object(eq_gen.yf, "download", side_effect=fake_download), patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(lambda *_a: None)):
            arrays = EquitiesSeqGenerator.generate(EquitiesSeqParams(symbols=["AAPL", "MSFT"], start_date="2008-01-01", end_date="2011-01-01", use_cache=False, lookback=5, fundamentals_fill="zero"))
        assert arrays["ticker_vocab"].tolist() == ["AAPL"]

    def test_generate_handles_ticker_with_no_data(self) -> None:
        # MSFT is absent from the ohlcv map -> empty frame -> the "no data" else branch.
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=21)}, _shares(), lookback=5)
        assert arrays["ticker_vocab"].tolist() == ["AAPL"]

    def test_generate_raises_when_no_symbol_has_data(self) -> None:
        with pytest.raises(ValueError, match="No data"):
            _generate(["MSFT"], {}, _shares(), lookback=5)

    def test_generate_with_normalized_features(self) -> None:
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=22), "MSFT": _ohlcv(seed=23)}, _shares(), lookback=5, normalize_features=True)
        assert arrays["X_full"].shape[0] > 0

    def test_generate_skips_ticker_shorter_than_lookback(self) -> None:
        # MSFT conditions to ~7 rows (<= lookback + 1 = 21) so no window is built for it,
        # while AAPL (400 rows) produces windows -> the per-ticker skip branch fires.
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=24, periods=400), "MSFT": _ohlcv(seed=25, periods=8)}, _shares(), lookback=20)
        assert "AAPL" in arrays["ticker_vocab"].tolist()
        assert arrays["X_full"].shape[0] > 0

    def test_generate_raises_when_all_tickers_too_short(self) -> None:
        with pytest.raises(ValueError, match="lookback"):
            _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=26, periods=8), "MSFT": _ohlcv(seed=27, periods=8)}, _shares(), lookback=20)
