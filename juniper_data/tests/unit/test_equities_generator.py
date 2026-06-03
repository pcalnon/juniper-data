"""Unit tests for the equities (S&P 500) time-series generator.

Network sources (Yahoo Finance via yfinance, SEC EDGAR shares) are mocked so
the suite runs fast and offline. Requires the optional ``equities`` extra
(pandas + yfinance); skipped otherwise.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_equities_generator.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from juniper_data.core.artifacts import load_npz, save_npz  # noqa: E402
from juniper_data.generators.equities import VERSION, EquitiesGenerator, EquitiesParams, get_schema  # noqa: E402
from juniper_data.generators.equities import generator as eq_gen  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]

_FEATURES = ["open", "high", "low", "close", "volume", "week52_high", "week52_low", "total_shares", "market_cap", "cost_basis"]


def _ohlcv(start: str = "2008-01-01", periods: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily OHLCV frame with yfinance-style capitalized columns."""
    index = pd.bdate_range(start=start, periods=periods)
    rng = np.random.default_rng(seed)
    walk = rng.normal(0.1, 1.0, periods).cumsum()
    close = 100.0 + walk - walk.min() + 1.0
    high = close + np.abs(rng.normal(0.5, 0.2, periods))
    low = close - np.abs(rng.normal(0.5, 0.2, periods))
    open_ = close + rng.normal(0.0, 0.2, periods)
    volume = rng.integers(1_000_000, 5_000_000, periods).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": volume}, index=index)


def _shares(start: str = "2009-06-30") -> pd.Series:
    """Synthetic shares-outstanding step series (first filing ~mid-2009)."""
    series = pd.Series({pd.Timestamp(start): 1_000_000_000.0, pd.Timestamp("2010-06-30"): 1_100_000_000.0})
    series.index = pd.to_datetime(series.index)
    return series


@contextmanager
def _mocked(ohlcv_map: dict, shares):
    """Patch yfinance.download and SEC share fetching for the duration."""

    def fake_download(symbol, **_kwargs):
        frame = ohlcv_map.get(symbol)
        return frame.copy() if frame is not None else pd.DataFrame()

    def fake_shares(_cik, _use_cache):
        return shares

    with patch.object(eq_gen.yf, "download", side_effect=fake_download), patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(fake_shares)):
        yield


def _generate(symbols, ohlcv_map, shares=None, **overrides):
    """Run the generator against mocked sources with sensible test defaults."""
    params = EquitiesParams(symbols=symbols, start_date="2008-01-01", end_date="2011-01-01", use_cache=False, **overrides)
    with _mocked(ohlcv_map, shares):
        return EquitiesGenerator.generate(params)


class TestEquitiesGenerator:
    """End-to-end behavior of EquitiesGenerator.generate()."""

    def test_keys_shapes_and_dtypes(self) -> None:
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=1), "MSFT": _ohlcv(seed=2)}, _shares())
        for key in ("X_train", "y_train", "X_test", "y_test", "X_full", "y_full", "y_reg_full", "ticker_code_full", "date_full", "ticker_vocab"):
            assert key in arrays, f"missing {key}"

        n = arrays["X_full"].shape[0]
        assert arrays["X_full"].shape == (n, 10)
        assert arrays["y_full"].shape == (n, 2)
        assert arrays["y_reg_full"].shape == (n, 1)
        assert arrays["X_full"].dtype == np.float32
        assert arrays["y_full"].dtype == np.float32
        assert arrays["ticker_code_full"].dtype == np.int32
        assert arrays["date_full"].dtype == np.int32
        assert arrays["ticker_vocab"].tolist() == ["AAPL", "MSFT"]
        # train + test partition the full set (temporal split, no overlap/loss).
        assert arrays["X_train"].shape[0] + arrays["X_test"].shape[0] == n

    def test_direction_target_is_onehot_and_correct(self) -> None:
        frame = _ohlcv(seed=3)
        arrays = _generate(["AAPL"], {"AAPL": frame}, _shares())
        # All rows sum to exactly one (valid one-hot).
        assert np.allclose(arrays["y_full"].sum(axis=1), 1.0)
        # Single ticker, temporal order preserved -> compare to the source.
        close = frame["Close"].to_numpy()
        expected_up = (close[1:] > close[:-1]).astype(np.float32)
        np.testing.assert_array_equal(arrays["y_full"][:, 1], expected_up[: arrays["y_full"].shape[0]])

    def test_next_close_regression_target(self) -> None:
        frame = _ohlcv(seed=4)
        arrays = _generate(["AAPL"], {"AAPL": frame}, _shares())
        close = frame["Close"].to_numpy().astype(np.float32)
        np.testing.assert_allclose(arrays["y_reg_full"][:, 0], close[1 : arrays["y_reg_full"].shape[0] + 1], rtol=1e-5)

    def test_temporal_split_ordering_per_ticker(self) -> None:
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=5), "MSFT": _ohlcv(seed=6)}, _shares(), train_ratio=0.7, test_ratio=0.3)
        for code in range(len(arrays["ticker_vocab"])):
            train_dates = arrays["date_train"][arrays["ticker_code_train"] == code]
            test_dates = arrays["date_test"][arrays["ticker_code_test"] == code]
            assert train_dates.max() <= test_dates.min(), "train must precede test"

    def test_week52_high_low(self) -> None:
        frame = _ohlcv(seed=7)
        arrays = _generate(["AAPL"], {"AAPL": frame}, _shares(), week52_window=252)
        rows = arrays["X_full"].shape[0]
        expected_high = frame["High"].rolling(252, min_periods=1).max().to_numpy()[:rows]
        expected_low = frame["Low"].rolling(252, min_periods=1).min().to_numpy()[:rows]
        np.testing.assert_allclose(arrays["X_full"][:, _FEATURES.index("week52_high")], expected_high, rtol=1e-5)
        np.testing.assert_allclose(arrays["X_full"][:, _FEATURES.index("week52_low")], expected_low, rtol=1e-5)

    def test_cost_basis_uses_purchase_date(self) -> None:
        frame = _ohlcv(seed=8)
        # Purchase mid-series: basis = last close on or before that date.
        purchase = "2009-01-05"
        arrays = _generate(["AAPL"], {"AAPL": frame}, _shares(), purchase_date=purchase)
        expected = float(frame.loc[frame.index <= pd.Timestamp(purchase), "Close"].iloc[-1])
        basis_column = arrays["X_full"][:, _FEATURES.index("cost_basis")]
        assert np.allclose(basis_column, np.float32(expected)), "cost basis is constant per ticker = price on purchase date"

    def test_fundamentals_fill_zero(self) -> None:
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=9)}, _shares(), fundamentals_fill="zero")
        shares_col = arrays["X_full"][:, _FEATURES.index("total_shares")]
        mcap_col = arrays["X_full"][:, _FEATURES.index("market_cap")]
        # Pre-2009 rows exist and are zero-filled; post-filing rows are positive.
        assert (shares_col == 0).any() and (shares_col > 0).any()
        assert (mcap_col == 0).any() and (mcap_col > 0).any()

    def test_fundamentals_fill_drop(self) -> None:
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=10)}, _shares(), fundamentals_fill="drop")
        shares_col = arrays["X_full"][:, _FEATURES.index("total_shares")]
        assert (shares_col > 0).all(), "drop mode keeps only rows with known shares"

    def test_fundamentals_fill_nan(self) -> None:
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=11)}, _shares(), fundamentals_fill="nan")
        shares_col = arrays["X_full"][:, _FEATURES.index("total_shares")]
        assert np.isnan(shares_col).any(), "nan mode leaves pre-filing rows missing"

    def test_no_shares_data_yields_zero_fundamentals(self) -> None:
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=12)}, shares=None, fundamentals_fill="zero")
        shares_col = arrays["X_full"][:, _FEATURES.index("total_shares")]
        assert np.all(shares_col == 0), "no SEC data -> all shares zero under zero-fill"

    def test_shares_fetch_failure_keeps_price_data(self) -> None:
        # A transient SEC failure must not discard the ticker's OHLCV/targets.
        frame = _ohlcv(seed=18)

        def boom(_cik, _use_cache):
            raise RuntimeError("SEC unreachable")

        def fake_download(symbol, **_kwargs):
            return frame.copy() if symbol == "AAPL" else pd.DataFrame()

        with patch.object(eq_gen.yf, "download", side_effect=fake_download), patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(boom)):
            arrays = EquitiesGenerator.generate(EquitiesParams(symbols=["AAPL"], start_date="2008-01-01", end_date="2011-01-01", use_cache=False, fundamentals_fill="zero"))

        assert arrays["ticker_vocab"].tolist() == ["AAPL"]
        assert arrays["X_full"].shape[0] > 0
        assert np.all(arrays["X_full"][:, _FEATURES.index("total_shares")] == 0)

    def test_fetch_shares_filters_filer_scale_error(self) -> None:
        # A SEC XBRL cover-page typo (1e6x too large, like ORCL 2012-09-17)
        # must be rejected so market cap stays plausible.
        payload = {
            "units": {
                "shares": [
                    {"end": "2009-06-30", "val": 1.0e9, "filed": "2009-07-01"},
                    {"end": "2009-09-30", "val": 1.0e15, "filed": "2009-10-01"},
                    {"end": "2010-06-30", "val": 1.1e9, "filed": "2010-07-01"},
                    {"end": "2011-06-30", "val": 1.2e9, "filed": "2011-07-01"},
                ]
            }
        }
        with patch.object(eq_gen, "_sec_get", return_value=payload):
            series = EquitiesGenerator._fetch_shares(320193, use_cache=False)
        assert series is not None
        assert len(series) == 3, "the 1e15 outlier should be dropped"
        assert series.max() < 1e13

    def test_normalize_features_bounds(self) -> None:
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=13), "MSFT": _ohlcv(seed=14)}, _shares(), normalize_features=True)
        matrix = arrays["X_full"]
        assert matrix.min() >= -1e-6 and matrix.max() <= 1.0 + 1e-6

    def test_extra_arrays_survive_roundtrip(self, tmp_path) -> None:
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=15), "MSFT": _ohlcv(seed=16)}, _shares())
        path = tmp_path / "equities.npz"
        save_npz(path, arrays)
        restored = load_npz(path)
        assert sorted(restored) == sorted(arrays)
        for key in ("y_reg_full", "ticker_code_full", "date_full", "ticker_vocab"):
            assert key in restored
        np.testing.assert_array_equal(restored["y_reg_full"], arrays["y_reg_full"])
        assert restored["ticker_vocab"].tolist() == arrays["ticker_vocab"].tolist()

    def test_unknown_symbol_skipped_not_fatal(self) -> None:
        # AAPL returns data; NVDA returns an empty frame and is skipped (not fatal).
        arrays = _generate(["AAPL", "NVDA"], {"AAPL": _ohlcv(seed=17)}, _shares())
        assert arrays["ticker_vocab"].tolist() == ["AAPL"]

    def test_all_symbols_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="No data"):
            _generate(["NVDA"], {})

    def test_missing_deps_raises_importerror(self) -> None:
        with patch.object(eq_gen, "EQUITIES_DEPS_AVAILABLE", False):
            with pytest.raises(ImportError, match="equities"):
                EquitiesGenerator.generate(EquitiesParams(symbols=["AAPL"]))


class TestEquitiesParams:
    """Validation behavior of EquitiesParams."""

    def test_version_string(self) -> None:
        assert VERSION == "1.0.0"

    def test_get_schema_returns_json_schema(self) -> None:
        schema = get_schema()
        assert schema["type"] == "object"
        assert "purchase_date" in schema["properties"]
        assert "fundamentals_fill" in schema["properties"]

    def test_invalid_ratio_sum_rejected(self) -> None:
        with pytest.raises(ValueError, match="train_ratio \\+ test_ratio"):
            EquitiesParams(train_ratio=0.8, test_ratio=0.3)

    def test_invalid_date_rejected(self) -> None:
        with pytest.raises(ValueError, match="start_date"):
            EquitiesParams(start_date="01-01-2000")

    def test_defaults_select_full_universe(self) -> None:
        params = EquitiesParams()
        assert params.symbols is None
        assert params.start_date == "2000-01-01"
        assert params.fundamentals_fill == "zero"
