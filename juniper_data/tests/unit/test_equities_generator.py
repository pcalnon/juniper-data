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

import json
import urllib.error
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.core import limits as eq_limits

pd = pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from juniper_data.core.artifacts import load_npz, save_npz  # noqa: E402
from juniper_data.generators.equities import VERSION, EquitiesGenerator, EquitiesParams, get_schema  # noqa: E402
from juniper_data.generators.equities import generator as eq_gen  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]

_FEATURES = ["open", "high", "low", "close", "volume", "week52_high", "week52_low", "total_shares", "market_cap", "cost_basis"]


def _ohlcv(start: str = "2008-01-01", periods: int = 600, seed: int = 0):
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


def _shares(start: str = "2009-06-30"):
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

    def test_regression_target_default_is_next_close(self) -> None:
        # The default mode reproduces the original raw-close target byte-for-byte.
        frame = _ohlcv(seed=20)
        default = _generate(["AAPL"], {"AAPL": frame}, _shares())
        explicit = _generate(["AAPL"], {"AAPL": frame}, _shares(), regression_target="next_close")
        np.testing.assert_array_equal(default["y_reg_full"], explicit["y_reg_full"])

    def test_regression_target_return_and_log_return(self) -> None:
        # `return` / `log_return` produce the stationary next-day return targets.
        frame = _ohlcv(seed=21)
        n = _generate(["AAPL"], {"AAPL": frame}, _shares())["y_reg_full"].shape[0]
        ret = _generate(["AAPL"], {"AAPL": frame}, _shares(), regression_target="return")
        logret = _generate(["AAPL"], {"AAPL": frame}, _shares(), regression_target="log_return")
        close = frame["Close"].to_numpy(dtype=np.float64)
        next_close, close_aligned = close[1 : n + 1], close[:n]
        np.testing.assert_allclose(ret["y_reg_full"][:, 0], (next_close / close_aligned - 1.0).astype(np.float32), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(logret["y_reg_full"][:, 0], np.log(next_close / close_aligned).astype(np.float32), rtol=1e-6, atol=1e-7)
        assert ret["y_reg_full"].shape == (n, 1)
        # Raw close tracks the ~100 price level (non-stationary); returns are centered near zero.
        assert abs(float(ret["y_reg_full"].mean())) < 1.0

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
        """TRAIN is bounded by [0, 1]; later partitions are NOT (juniper-data#314).

        This previously asserted the bound on ``X_full``, which passed only because the
        normaliser was fit on the full matrix -- including the chronologically-later test
        rows -- and then applied to train. That is look-ahead leakage, and the assertion was
        pinning it in place without naming it.

        Under a train-only fit the bound moves to ``X_train`` alone. Test rows exceeding the
        training range is CORRECT: that excursion is real out-of-sample movement, and scaling
        it away is exactly what the leak was doing.
        """
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=13), "MSFT": _ohlcv(seed=14)}, _shares(), normalize_features=True)
        train = arrays["X_train"]
        assert train.min() >= -1e-6 and train.max() <= 1.0 + 1e-6, "the fitted partition must be bounded"

    def test_normaliser_is_fit_on_train_not_full(self) -> None:
        """The regression guard. Would fail if the fit reverted to the full matrix.

        A full-matrix fit bounds EVERY partition by construction, so an out-of-range test row
        is positive evidence that the statistics came from train alone.
        """
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=13), "MSFT": _ohlcv(seed=14)}, _shares(), normalize_features=True)
        assert arrays["X_test"].max() > 1.0 + 1e-6, "test rows should exceed the training range; a full-fit would clamp them to 1.0"

    def test_unnormalised_output_is_unaffected(self) -> None:
        """``normalize_features`` defaults to False, so the common path must not move."""
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=13), "MSFT": _ohlcv(seed=14)}, _shares())
        assert arrays["X_train"].max() > 1.0, "raw features are not in [0, 1]"

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

    def test_is_available_reflects_deps_flag(self) -> None:
        """is_available mirrors EQUITIES_DEPS_AVAILABLE (D1 / I-5 availability surface)."""
        assert EquitiesGenerator.is_available() is True  # extra installed in this suite (importorskip above)
        with patch.object(eq_gen, "EQUITIES_DEPS_AVAILABLE", False):
            assert EquitiesGenerator.is_available() is False


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

    def test_regression_target_default_and_schema(self) -> None:
        assert EquitiesParams().regression_target == "next_close"
        assert "regression_target" in get_schema()["properties"]

    def test_invalid_regression_target_rejected(self) -> None:
        with pytest.raises(ValueError):
            EquitiesParams(regression_target="returns")  # not a Literal member


def _fake_urlopen_response(payload: dict):
    """A ``urlopen``-compatible context manager whose ``read()`` yields ``payload`` JSON."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class TestEquitiesGeneratorInternals:
    """Direct coverage of the fetch / cache / conditioning helpers (offline, deterministic).

    ``time.sleep`` is patched to a no-op in the SEC-retry tests so the throttle /
    backoff paths run instantly; caches are redirected to ``tmp_path`` via the
    module-level ``_CACHE_DIR``.
    """

    def test_sec_get_returns_parsed_json_on_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(eq_gen.time, "sleep", lambda *_a, **_k: None)
        payload = {"cik": 320193, "units": {}}
        monkeypatch.setattr(eq_gen.urllib.request, "urlopen", lambda *_a, **_k: _fake_urlopen_response(payload))
        assert eq_gen._sec_get("https://data.sec.gov/x") == payload

    def test_sec_get_returns_none_on_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(eq_gen.time, "sleep", lambda *_a, **_k: None)

        def _not_found(*_a, **_k):
            raise urllib.error.HTTPError("https://data.sec.gov/x", 404, "Not Found", {}, None)

        monkeypatch.setattr(eq_gen.urllib.request, "urlopen", _not_found)
        assert eq_gen._sec_get("https://data.sec.gov/x") is None

    def test_sec_get_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(eq_gen.time, "sleep", lambda *_a, **_k: None)
        payload = {"ok": 1}
        state = {"calls": 0}

        def _flaky(*_a, **_k):
            state["calls"] += 1
            if state["calls"] == 1:
                raise urllib.error.HTTPError("https://data.sec.gov/x", 503, "Busy", {}, None)
            return _fake_urlopen_response(payload)

        monkeypatch.setattr(eq_gen.urllib.request, "urlopen", _flaky)
        assert eq_gen._sec_get("https://data.sec.gov/x", retries=3) == payload
        assert state["calls"] == 2

    def test_sec_get_raises_after_exhausting_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(eq_gen.time, "sleep", lambda *_a, **_k: None)

        def _down(*_a, **_k):
            raise urllib.error.URLError("network down")

        monkeypatch.setattr(eq_gen.urllib.request, "urlopen", _down)
        with pytest.raises(urllib.error.URLError):
            eq_gen._sec_get("https://data.sec.gov/x", retries=2)

    def test_load_sec_ticker_map_reads_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        (tmp_path / "company_tickers.json").write_text(json.dumps({"0": {"ticker": "aapl", "title": "Apple Inc", "cik_str": 320193}}))
        result = eq_gen.EquitiesGenerator._load_sec_ticker_map()
        assert result["AAPL"] == {"name": "Apple Inc", "cik": 320193}

    def test_load_sec_ticker_map_fetches_and_caches(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(eq_gen, "_sec_get", lambda *_a, **_k: {"0": {"ticker": "msft", "title": "Microsoft", "cik_str": 789019}})
        result = eq_gen.EquitiesGenerator._load_sec_ticker_map()
        assert result["MSFT"]["cik"] == 789019
        assert (tmp_path / "company_tickers.json").exists()

    def test_load_sec_ticker_map_empty_when_unavailable(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(eq_gen, "_sec_get", lambda *_a, **_k: None)
        assert eq_gen.EquitiesGenerator._load_sec_ticker_map() == {}

    def test_fetch_shares_reads_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        cache = tmp_path / "shares" / f"{320193:010d}.json"
        cache.parent.mkdir(parents=True)
        cache.write_text(json.dumps({"units": {"shares": [{"end": "2009-06-30", "val": 1.0e9, "filed": "2009-07-01"}]}}))
        series = eq_gen.EquitiesGenerator._fetch_shares(320193, use_cache=True)
        assert series is not None
        assert len(series) == 1

    def test_fetch_shares_fetches_and_caches(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        payload = {"units": {"shares": [{"end": "2009-06-30", "val": 1.0e9, "filed": "2009-07-01"}, {"end": "2010-06-30", "val": 1.1e9, "filed": "2010-07-01"}]}}
        monkeypatch.setattr(eq_gen, "_sec_get", lambda *_a, **_k: payload)
        series = eq_gen.EquitiesGenerator._fetch_shares(999999, use_cache=True)
        assert series is not None
        assert len(series) == 2
        assert (tmp_path / "shares" / "0000999999.json").exists()

    def test_fetch_shares_returns_none_when_no_concept_data(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(eq_gen, "_sec_get", lambda *_a, **_k: None)
        assert eq_gen.EquitiesGenerator._fetch_shares(111, use_cache=False) is None

    def test_fetch_shares_returns_none_when_no_usable_points(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(eq_gen, "_sec_get", lambda *_a, **_k: {"units": {"shares": [{"end": None, "val": None}]}})
        assert eq_gen.EquitiesGenerator._fetch_shares(222, use_cache=False) is None

    def test_download_ohlcv_reads_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        cache = tmp_path / "ohlcv" / "AAPL_2008-01-01_2011-01-01.csv"
        cache.parent.mkdir(parents=True)
        eq_gen.EquitiesGenerator._normalize_ohlcv_columns(_ohlcv(periods=10)).to_csv(cache)
        result = eq_gen.EquitiesGenerator._download_ohlcv("AAPL", "2008-01-01", "2011-01-01", use_cache=True)
        assert result is not None
        assert len(result) == 10

    def test_download_ohlcv_returns_none_when_normalized_empty(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        junk = pd.DataFrame({"Junk": [1.0, 2.0]}, index=pd.bdate_range("2008-01-01", periods=2))
        monkeypatch.setattr(eq_gen.yf, "download", lambda *_a, **_k: junk)
        assert eq_gen.EquitiesGenerator._download_ohlcv("AAPL", "2008-01-01", "2011-01-01", use_cache=False) is None

    def test_download_ohlcv_writes_cache_after_download(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(eq_gen.yf, "download", lambda *_a, **_k: _ohlcv(periods=12).copy())
        result = eq_gen.EquitiesGenerator._download_ohlcv("AAPL", "2008-01-01", "2011-01-01", use_cache=True)
        assert result is not None
        assert len(result) == 12
        assert (tmp_path / "ohlcv" / "AAPL_2008-01-01_2011-01-01.csv").exists()

    def test_normalize_ohlcv_columns_flattens_multiindex(self) -> None:
        idx = pd.bdate_range("2008-01-01", periods=3)
        frame = pd.DataFrame({("Open", "AAPL"): [1.0, 2.0, 3.0], ("High", "AAPL"): [2.0, 3.0, 4.0], ("Low", "AAPL"): [0.5, 1.5, 2.5], ("Close", "AAPL"): [1.0, 2.0, 3.0], ("Volume", "AAPL"): [10.0, 20.0, 30.0]}, index=idx)
        frame.columns = pd.MultiIndex.from_tuples(list(frame.columns))
        out = eq_gen.EquitiesGenerator._normalize_ohlcv_columns(frame)
        assert "open" in out.columns
        assert "close" in out.columns

    def test_resolve_symbols_uses_sec_map_for_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        constituents = {"AAPL": {"name": "Apple", "cik": 320193, "sector": "Tech"}}
        monkeypatch.setattr(eq_gen.EquitiesGenerator, "_load_sec_ticker_map", staticmethod(lambda: {"ZZZZ": {"name": "Zeta Corp", "cik": 111}}))
        ordered, meta, truncation = eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(symbols=["AAPL", "ZZZZ"]), constituents)
        assert ordered == ["AAPL", "ZZZZ"]
        assert meta["ZZZZ"]["cik"] == 111
        assert truncation is None

    def test_resolve_symbols_defaults_to_the_whole_universe_when_it_fits(self) -> None:
        """The default universe is still every constituent -- when it fits the cap.

        Renamed from ``…_defaults_to_full_universe``: under APD-DATA-018 the
        default is bounded, so "full" is only true below the cap. The two-name
        universe here is well under 14 and must be untouched and unannotated.
        """
        constituents = {"MSFT": {"name": "MS", "cik": 789019, "sector": "Tech"}, "AAPL": {"name": "Apple", "cik": 320193, "sector": "Tech"}}
        ordered, meta, truncation = eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(), constituents)
        assert ordered == ["AAPL", "MSFT"]
        assert meta is constituents
        assert truncation is None

    def test_resolve_symbols_respects_max_symbols(self) -> None:
        """A tightened cap is honoured -- but only with the opt-in.

        Under APD-DATA-018 a cap that would cut is a REFUSAL by default, so this
        arm now has to say it accepts a partial universe. The bare slice it used
        to exercise silently returned two of three.
        """
        constituents = {name: {"name": name, "cik": i, "sector": ""} for i, name in enumerate(["A", "B", "C"])}
        ordered, _meta, truncation = eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(max_symbols=2, allow_truncation=True), constituents)
        assert ordered == ["A", "B"]
        assert truncation["requested"] == 3
        assert truncation["imported"] == 2
        assert truncation["cap"] == 2

    def test_generate_skips_ticker_whose_download_raises(self) -> None:
        good = _ohlcv(seed=30)

        def fake_download(symbol, **_kwargs):
            if symbol == "MSFT":
                raise RuntimeError("download exploded")
            return good.copy()

        with patch.object(eq_gen.yf, "download", side_effect=fake_download), patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(lambda *_a: None)):
            arrays = EquitiesGenerator.generate(EquitiesParams(symbols=["AAPL", "MSFT"], start_date="2008-01-01", end_date="2011-01-01", use_cache=False, fundamentals_fill="zero"))
        assert arrays["ticker_vocab"].tolist() == ["AAPL"]

    def test_generate_clips_test_split_when_rounding_overshoots(self) -> None:
        # 8 business days condition to 7 rows; train=test=0.5 -> round(3.5)=4 each
        # -> 4 + 4 > 7, exercising the ``n_test = n_rows - n_train`` clip.
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(periods=8, seed=31)}, _shares(), train_ratio=0.5, test_ratio=0.5)
        n = arrays["X_full"].shape[0]
        assert n == 7
        assert arrays["X_train"].shape[0] + arrays["X_test"].shape[0] == n

    def test_condition_one_returns_none_when_too_short(self) -> None:
        with patch.object(eq_gen.yf, "download", return_value=_ohlcv(periods=1)):
            result = eq_gen.EquitiesGenerator._condition_one("AAPL", {}, EquitiesParams(use_cache=False), "2011-01-01")
        assert result is None

    def test_static_array_helpers_handle_empty_frame(self) -> None:
        empty = pd.DataFrame()
        assert eq_gen.EquitiesGenerator._features(empty, None).shape == (0, 10)
        assert eq_gen.EquitiesGenerator._direction_onehot(empty).shape == (0, 2)
        assert eq_gen.EquitiesGenerator._regression_target(empty, "next_close").shape == (0, 1)
        assert eq_gen.EquitiesGenerator._dates_yyyymmdd(empty).shape == (0,)


@pytest.mark.unit
class TestUniverseSymbolCap:
    """APD-DATA-018, equities half: bound the fan-out, and never cut it silently.

    The cap is in **symbols**, not bytes, and that is the load-bearing choice.
    Measurement on 2026-09-04 put 163x the payload at 1.16x the time, because
    cost is per request: one symbol over 26 years is 210 KB and ~2 s, while the
    Russell 3000 over *one day* is 92 KB and 1.7-3.2 h. A byte cap would admit
    the expensive request and reject the cheap one.
    """

    @staticmethod
    def _universe(count: int) -> dict[str, dict[str, object]]:
        return {f"T{i:03d}": {"name": f"Name {i}", "cik": 1000 + i, "sector": ""} for i in range(count)}

    def test_oversized_universe_is_refused_by_default(self) -> None:
        """The default is refusal. A partial universe never reaches a caller who did not ask."""
        with pytest.raises(eq_limits.InputTooLargeError) as excinfo:
            eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(), self._universe(40))
        assert excinfo.value.unit == "symbols"
        assert excinfo.value.cap == 14
        assert excinfo.value.actual == 40
        assert "allow_truncation" in str(excinfo.value)
        assert "JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION" in str(excinfo.value)

    def test_refusal_is_a_value_error(self) -> None:
        """Subclassing ValueError is load-bearing: a missed catch lands on 400, not 500."""
        with pytest.raises(ValueError):
            eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(), self._universe(40))

    def test_opt_in_truncates_and_annotates(self) -> None:
        """An authorised cut produces a partial universe AND its permanent record."""
        ordered, _meta, truncation = eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(allow_truncation=True), self._universe(40))
        assert len(ordered) == 14
        assert truncation["truncated"] is True
        assert truncation["reason"] == "universe_exceeded_symbol_cap"
        assert truncation["unit"] == "symbols"
        assert truncation["cap"] == 14
        assert truncation["requested"] == 40
        assert truncation["imported"] == 14

    def test_the_kept_prefix_is_deterministic(self) -> None:
        """Which symbols survive must not depend on iteration or download order.

        A truncated dataset that differs run to run is not reproducible, and the
        annotation would describe a universe nobody can reconstruct.
        """
        universe = self._universe(40)
        first, _m1, _t1 = eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(allow_truncation=True), universe)
        second, _m2, _t2 = eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(allow_truncation=True), dict(reversed(list(universe.items()))))
        assert first == second == sorted(universe)[:14]

    def test_request_cannot_RAISE_the_deployment_cap(self) -> None:
        """A request may only lower the bound -- including via max_symbols=None."""
        with pytest.raises(eq_limits.InputTooLargeError) as excinfo:
            eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(max_symbols=9999), self._universe(40))
        assert excinfo.value.cap == 14

        ordered, _meta, truncation = eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(max_symbols=None, allow_truncation=True), self._universe(40))
        assert len(ordered) == 14, "max_symbols=None must mean 'no request limit', not 'unbounded'"
        assert truncation["cap"] == 14

    def test_deployment_opt_in_truncates_without_a_request_parameter(self) -> None:
        """The env-var / .env surface works on its own, for CLI callers."""
        settings = MagicMock()
        settings.equities_max_symbols = 14
        settings.equities_allow_truncation = True
        with patch("juniper_data.api.settings.get_settings", return_value=settings):
            ordered, _meta, truncation = eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(), self._universe(40))
        assert len(ordered) == 14
        assert truncation["truncated"] is True

    def test_universe_at_exactly_the_cap_is_not_truncated(self) -> None:
        """The boundary is inclusive: 14 of 14 is complete, not a cut."""
        ordered, _meta, truncation = eq_gen.EquitiesGenerator._resolve_symbols(EquitiesParams(), self._universe(14))
        assert len(ordered) == 14
        assert truncation is None

    def test_generate_puts_the_annotation_on_the_returned_arrays(self) -> None:
        """The descriptor has to reach ``generate()``'s output, not just the resolver.

        Resolving the cut and *reporting* it are separate steps, and only the
        second is what a consumer ever sees. A mutation that dropped the channel
        assignment broke nothing until this arm existed -- every other test in
        this class calls ``_resolve_symbols`` directly and would stay green with
        the bound enforced and the record silently discarded.
        """
        tickers = [f"T{i:02d}" for i in range(16)]
        ohlcv = {ticker: _ohlcv(seed=index) for index, ticker in enumerate(tickers)}
        arrays = _generate(tickers, ohlcv, _shares(), allow_truncation=True)

        annotation = arrays[eq_limits.TRUNCATION_META_KEY]
        assert annotation["reason"] == "universe_exceeded_symbol_cap"
        assert annotation["requested"] == 16
        assert annotation["imported"] == 14
        # records_imported is filled in after conditioning, so it must be a real
        # row count -- not the -1 placeholder the resolver leaves behind.
        assert annotation["records_imported"] == arrays["X_full"].shape[0] > 0
        assert len(arrays["ticker_vocab"]) == 14

    def test_generate_refuses_an_oversized_universe(self) -> None:
        """The refusal survives all the way out through ``generate()``."""
        tickers = [f"T{i:02d}" for i in range(16)]
        ohlcv = {ticker: _ohlcv(seed=index) for index, ticker in enumerate(tickers)}
        with pytest.raises(eq_limits.InputTooLargeError):
            _generate(tickers, ohlcv, _shares())

    def test_generate_omits_the_key_entirely_when_nothing_was_cut(self) -> None:
        """Absence, not a falsy descriptor -- the same contract csv_import keeps."""
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=1), "MSFT": _ohlcv(seed=2)}, _shares())
        assert eq_limits.TRUNCATION_META_KEY not in arrays

    def test_default_cap_matches_the_measured_budget(self) -> None:
        """14 is the measured figure, not a round number.

        30 s request budget / 2.1 s per symbol = 14.1. If this constant moves,
        the measurement behind it (util/ad-hoc/2026-09-04_measure_equities_payloads.py)
        has to move with it.
        """
        assert eq_limits.EQUITIES_DEFAULT_MAX_SYMBOLS == 14
        assert eq_limits.EQUITIES_DEFAULT_ALLOW_TRUNCATION is False
