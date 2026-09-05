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
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.core import limits as eq_limits

pd = pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from juniper_data.core.artifacts import load_npz, save_npz  # noqa: E402
from juniper_data.generators.equities import VERSION, EquitiesGenerator, EquitiesParams, get_schema  # noqa: E402
from juniper_data.generators.equities import generator as eq_gen  # noqa: E402
from juniper_data.generators.equities.defaults import EQUITIES_FEATURE_COLUMNS  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]

# Derived from the constant, not restated. This list was a hand-maintained copy
# and went stale the moment the matrix widened from 10 to 16 columns.
_FEATURES = list(EQUITIES_FEATURE_COLUMNS)


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
    """Synthetic shares-outstanding history, as ``_fetch_shares`` now returns it.

    A DataFrame of ``shares`` + ``filed``, not a bare Series: the filing date is
    what feeds ``report_date`` / ``days_since_report``, and it is deliberately
    LATER than the period end it describes, because that lag is the thing those
    columns exist to represent.
    """
    return pd.DataFrame(
        {"shares": [1_000_000_000.0, 1_100_000_000.0], "filed": [pd.Timestamp("2009-08-14"), pd.Timestamp("2010-08-13")]},
        index=pd.to_datetime([pd.Timestamp(start), pd.Timestamp("2010-06-30")]),
    )


def _shares_quarterly(periods: int = 40, restated: int | None = None, restated_filed_with: int | None = None):
    """A realistic multi-year shares history, optionally with a same-day restatement.

    ``_shares()`` has two rows and the second is filed past the mocked frame's last
    trade date, so only ONE filing is ever reachable -- which leaves every ordering
    and de-duplication path structurally unexercised. This builds ``periods``
    quarterly filings, each disclosed 32 days after the period it describes.

    When ``restated`` is given, that row's ``filed`` is moved onto the same day as
    ``restated_filed_with``'s, reproducing the real shape that broke the alignment:
    an 8-K restating an old period lands the same day as the current 10-Q. Both
    rows then collapse to one under ``duplicated(keep="last")``, and only the
    LATER period end is the right survivor.
    """
    ends = pd.to_datetime([f"{2001 + index // 4}-{1 + 3 * (index % 4):02d}-20" for index in range(periods)])
    filed = [end + pd.Timedelta(days=32) for end in ends]
    if restated is not None and restated_filed_with is not None:
        filed[restated] = filed[restated_filed_with]
    return pd.DataFrame({"shares": 1_000_000_000.0 + np.arange(periods) * 1_000_000.0, "filed": filed}, index=ends)


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
        for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "X_full", "y_full", "y_reg_full", "ticker_code_full", "date_full", "ticker_vocab"):
            assert key in arrays, f"missing {key}"

        n = arrays["X_full"].shape[0]
        assert arrays["X_full"].shape == (n, len(EQUITIES_FEATURE_COLUMNS))
        assert arrays["y_full"].shape == (n, 2)
        assert arrays["y_reg_full"].shape == (n, 1)
        assert arrays["X_full"].dtype == np.float32
        assert arrays["y_full"].dtype == np.float32
        assert arrays["ticker_code_full"].dtype == np.int32
        assert arrays["date_full"].dtype == np.int32
        assert arrays["ticker_vocab"].tolist() == ["AAPL", "MSFT"]
        # train + val + test partition the full set (temporal split, no overlap/loss).
        # The default ratios are 0.8 / 0.1 / 0.1, so all three are non-empty here and
        # the sum is exact rather than a lower bound.
        assert arrays["X_val"].shape[0] > 0, "val partition must be non-empty"
        assert arrays["X_train"].shape[0] + arrays["X_val"].shape[0] + arrays["X_test"].shape[0] == n

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
        arrays = _generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=5), "MSFT": _ohlcv(seed=6)}, _shares(), train_ratio=0.6, val_ratio=0.1, test_ratio=0.3)
        for code in range(len(arrays["ticker_vocab"])):
            per_split = {s: arrays[f"date_{s}"][arrays[f"ticker_code_{s}"] == code] for s in ("train", "val", "test")}
            assert all(d.size for d in per_split.values()), "each partition must claim rows for every ticker"
            # Transitive, not just train < test: checking the endpoints alone would
            # leave val free to overlap either neighbour, and val is the split early
            # stopping reads.
            for earlier, later in (("train", "val"), ("val", "test"), ("train", "test")):
                assert per_split[earlier].max() <= per_split[later].min(), f"{earlier} must precede {later}"

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
        # allow_truncation opts in to the zero-fill this test is ABOUT. Without it
        # the generator now refuses, which is the point of the new contract.
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=12)}, shares=None, fundamentals_fill="zero", allow_truncation=True)
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
            arrays = EquitiesGenerator.generate(EquitiesParams(symbols=["AAPL"], start_date="2008-01-01", end_date="2011-01-01", use_cache=False, fundamentals_fill="zero", allow_truncation=True))

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
            frame = EquitiesGenerator._fetch_shares(320193, use_cache=False)
        assert frame is not None
        assert len(frame) == 3, "the 1e15 outlier should be dropped"
        assert frame["shares"].max() < 1e13
        # The filing date rides along on the same payload -- that is what makes
        # report_date / days_since_report cost no extra request. It must be the
        # FILED date, not the period end, and here it is a day later.
        assert frame["filed"].iloc[0] == pd.Timestamp("2009-07-01")
        assert frame.index[0] == pd.Timestamp("2009-06-30")

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

    def test_post_cut_rows_do_not_move_train_statistics(self) -> None:
        """Exact-cut identity: a row at or after n_train must not enter the fit.

        ``X_test.max() > 1`` proves the fit is not the full matrix, but still passes if the
        cut is merely *near* the split. Multiplying the later half of OHLCV by 100 leaves
        train-row inputs unchanged; the only way ``X_train`` can move is if a spiked row
        leaked into the statistics. ``train_ratio=0.5`` aligns the split with the spike.
        """
        clean = _ohlcv(seed=41)
        spiked = clean.copy()
        spiked.iloc[len(spiked) // 2 :] = spiked.iloc[len(spiked) // 2 :] * 100.0
        kwargs = {"normalize_features": True, "train_ratio": 0.5, "val_ratio": 0.0, "test_ratio": 0.5}
        baseline = _generate(["AAPL"], {"AAPL": clean}, _shares(), **kwargs)
        shifted = _generate(["AAPL"], {"AAPL": spiked}, _shares(), **kwargs)
        np.testing.assert_allclose(shifted["X_train"], baseline["X_train"], rtol=1e-5, atol=1e-6)
        assert not np.allclose(shifted["X_test"], baseline["X_test"]), "the spike must land in test so the comparison is live"

    def test_empty_training_partition_falls_back_without_nan(self) -> None:
        """When every ticker rounds to zero train rows, fit on full rather than emit NaN stats.

        Two business days condition to one row (the last is dropped for the next-day target).
        ``round(1 * 0.4) == 0`` empties train; the fallback must keep test/full finite.

        The shares frame is filed BEFORE the price rows on purpose. The module ``_shares()``
        fixture files in 2009-2010, which is after this two-day 2008 window -- and under the
        point-in-time alignment that lands NO shares at all, so the run dies on
        ``IncompleteDataError`` before the normaliser is ever reached. That guard is correct;
        it is simply not what this test is about, and satisfying it with
        ``allow_truncation=True`` would have changed the scenario rather than the fixture.
        """
        early_shares = pd.DataFrame(
            {"shares": [1_000_000_000.0], "filed": [pd.Timestamp("2007-08-14")]},
            index=pd.to_datetime([pd.Timestamp("2007-06-30")]),
        )
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(periods=2, seed=40)}, early_shares, normalize_features=True, train_ratio=0.4, val_ratio=0.0, test_ratio=0.6)
        assert arrays["X_train"].shape[0] == 0
        assert arrays["X_test"].shape[0] == 1
        assert np.isfinite(arrays["X_test"]).all()
        assert np.isfinite(arrays["X_full"]).all()


class TestEquitiesParams:
    """Validation behavior of EquitiesParams."""

    def test_version_string(self) -> None:
        # 2.0.0 since the val partition: the dataset ID hashes this version, so a
        # seeded request that produced a two-way artifact must not resolve to the
        # same ID now that the same params produce a three-way one (risk R-1).
        assert VERSION == "2.0.0"

    def test_get_schema_returns_json_schema(self) -> None:
        schema = get_schema()
        assert schema["type"] == "object"
        assert "purchase_date" in schema["properties"]
        assert "fundamentals_fill" in schema["properties"]

    def test_invalid_ratio_sum_rejected(self) -> None:
        with pytest.raises(ValueError, match="train_ratio \\+ val_ratio \\+ test_ratio"):
            EquitiesParams(train_ratio=0.8, test_ratio=0.3)

    def test_default_val_ratio_participates_in_the_sum(self) -> None:
        """0.7 + 0.3 was legal two-way and is refused three-way.

        The validation share is not free: it comes out of the same 1.0. A caller
        who wants the old two-way division has to say ``val_ratio=0.0`` and mean it,
        rather than have the generator quietly shrink test to make room.
        """
        with pytest.raises(ValueError, match="train_ratio \\+ val_ratio \\+ test_ratio"):
            EquitiesParams(train_ratio=0.7, test_ratio=0.3)
        assert EquitiesParams(train_ratio=0.7, val_ratio=0.0, test_ratio=0.3).val_ratio == 0.0

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

    def test_an_empty_concept_does_not_suppress_the_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """A present-but-EMPTY dei concept must not stop the us-gaap fallback.

        SEC answers 200 with ``{"units": {"shares": {}}}`` for some filers -- KO,
        ABT and others. The guard was ``if payload and payload.get("units")``, and
        that dict is **truthy**, so the loop accepted the empty concept and broke
        before trying the fallback. BIIB has 42 usable facts under us-gaap and was
        getting none of them: ``total_shares`` and ``market_cap`` silently became
        0.0 under the default ``fundamentals_fill="zero"``, indistinguishable
        downstream from a real measurement.

        Truthiness is the wrong test for "has data" whenever the API can return an
        empty container.
        """
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        empty_dei: dict[str, Any] = {"units": {"shares": {}}}
        populated_gaap: dict[str, Any] = {"units": {"shares": [{"end": "2009-06-30", "val": 1.0e9, "filed": "2009-08-14"}]}}

        calls: list[str] = []

        def fake_get(url: str) -> dict[str, Any]:
            calls.append(url)
            return empty_dei if "dei" in url else populated_gaap

        monkeypatch.setattr(eq_gen, "_sec_get", fake_get)
        frame = eq_gen.EquitiesGenerator._fetch_shares(875045, use_cache=False)

        assert len(calls) == 2, "the empty dei concept must not short-circuit the fallback"
        assert frame is not None and len(frame) == 1
        assert frame["shares"].iloc[0] == 1.0e9

    def test_an_all_empty_concept_set_still_returns_none(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """When every tag is empty there genuinely is no data -- KO's real shape."""
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(eq_gen, "_sec_get", lambda _url: {"units": {"shares": {}}})
        assert eq_gen.EquitiesGenerator._fetch_shares(21344, use_cache=False) is None

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
        assert len(result) == 10, "10 ROWS from the cached frame -- not a feature count"

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
            arrays = EquitiesGenerator.generate(EquitiesParams(symbols=["AAPL", "MSFT"], start_date="2008-01-01", end_date="2011-01-01", use_cache=False, fundamentals_fill="zero", allow_truncation=True))
        assert arrays["ticker_vocab"].tolist() == ["AAPL"]

    def test_generate_clips_test_split_when_rounding_overshoots(self) -> None:
        # 8 business days condition to 7 rows; 0.5/0.25/0.25 rounds to 4 + 2 + 2 = 8
        # against 7 rows, so one row of overflow has to be given back.
        # allow_truncation: this window is 8 business days from 2008-01-01, entirely
        # BEFORE the first filing in _shares() (2009-08-14), so under filed-date
        # alignment there is genuinely no shares data for these rows and the
        # generator now refuses by default. Split arithmetic is what is under test.
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(periods=8, seed=31)}, _shares(), train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, allow_truncation=True)
        n = arrays["X_full"].shape[0]
        assert n == 7
        assert arrays["X_train"].shape[0] + arrays["X_val"].shape[0] + arrays["X_test"].shape[0] == n
        # Trimmed test-first, and train is never trimmed: shrinking train to fund a
        # rounding artifact would change what the model was fit on, which is the one
        # thing a split-arithmetic fix must not do.
        assert arrays["X_train"].shape[0] == 4
        assert arrays["X_val"].shape[0] == 2
        assert arrays["X_test"].shape[0] == 1

    def test_condition_one_returns_none_when_too_short(self) -> None:
        with patch.object(eq_gen.yf, "download", return_value=_ohlcv(periods=1)):
            result = eq_gen.EquitiesGenerator._condition_one("AAPL", {}, EquitiesParams(use_cache=False), "2011-01-01")
        assert result is None

    def test_static_array_helpers_handle_empty_frame(self) -> None:
        empty = pd.DataFrame()
        assert eq_gen.EquitiesGenerator._features(empty, None).shape == (0, len(EQUITIES_FEATURE_COLUMNS))
        assert eq_gen.EquitiesGenerator._direction_onehot(empty).shape == (0, 2)
        assert eq_gen.EquitiesGenerator._regression_target(empty, "next_close").shape == (0, 1)
        assert eq_gen.EquitiesGenerator._dates_yyyymmdd(empty).shape == (0,)


@pytest.mark.unit
class TestUniverseSymbolCap:
    """APD-DATA-018, equities half: bound the fan-out, and never cut it silently.

    The cap is in **symbols**, not bytes, and that is the load-bearing choice.
    A cap's unit must be something the server can measure BEFORE doing the work.
    `csv_import` has a file in hand, so bytes are a `stat`. `equities` has no
    input at all -- a request is a ticker list and a date range, and its byte
    count does not exist until the API fan-out the cap exists to bound has
    already run. The symbol count is known with zero network calls, which is
    where `_resolve_symbols` raises.

    (Corrected 2026-09-05: this docstring argued the choice from "163x the
    payload at 1.16x the time ... a byte cap would admit the expensive request
    and reject the cheap one". The byte arithmetic was inverted -- the published
    92 KB omitted the per-request envelope on 2,923 separate calls, and the real
    figure is ~2.07 MB, larger than the 210 KB single-symbol request. Bytes are
    positively correlated with cost here. The cap is unchanged.)
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

    def test_bind_deployment_defaults_puts_effective_policy_in_dump(self) -> None:
        """The cache key must follow the resolved cap and opt-in, not Field defaults.

        omit-max_symbols and an explicit 14 schema default resolve to the SAME
        effective cap (the deployment ceiling, after the clamp). Binding must
        record that ceiling -- otherwise a later restart that raises the cap
        reuses the truncated artifact. Global allow_truncation must appear in
        the dump for the same reason.
        """
        settings = MagicMock()
        settings.equities_max_symbols = 7
        settings.equities_allow_truncation = False
        with patch("juniper_data.api.settings.get_settings", return_value=settings):
            omitted = eq_gen.EquitiesGenerator.bind_deployment_defaults(EquitiesParams(allow_truncation=True))
            explicit_default = eq_gen.EquitiesGenerator.bind_deployment_defaults(EquitiesParams(allow_truncation=True, max_symbols=eq_limits.EQUITIES_DEFAULT_MAX_SYMBOLS))
        assert omitted.max_symbols == 7
        assert explicit_default.max_symbols == 7
        assert omitted.model_dump()["max_symbols"] == explicit_default.model_dump()["max_symbols"]

        settings.equities_max_symbols = eq_limits.EQUITIES_DEFAULT_MAX_SYMBOLS
        with patch("juniper_data.api.settings.get_settings", return_value=settings):
            omitted_wide = eq_gen.EquitiesGenerator.bind_deployment_defaults(EquitiesParams(allow_truncation=True))
        assert omitted_wide.max_symbols == eq_limits.EQUITIES_DEFAULT_MAX_SYMBOLS
        assert omitted.model_dump()["max_symbols"] != omitted_wide.model_dump()["max_symbols"]

        settings.equities_max_symbols = 7
        settings.equities_allow_truncation = True
        with patch("juniper_data.api.settings.get_settings", return_value=settings):
            inherited = eq_gen.EquitiesGenerator.bind_deployment_defaults(EquitiesParams())
        assert inherited.allow_truncation is True
        assert inherited.model_dump()["allow_truncation"] is True


@pytest.mark.unit
class TestFreeFields:
    """The six columns added 2026-09-04 that cost no extra request.

    Each was already being downloaded and discarded. The tests below check the
    VALUES, not just the presence of a column -- a field that is present and
    wrong is worse than one that is absent, because nothing downstream will
    question it.
    """

    def test_the_matrix_widened_by_exactly_the_free_fields(self) -> None:
        """Order is part of the contract: existing columns keep their positions."""
        assert EQUITIES_FEATURE_COLUMNS[:10] == ["open", "high", "low", "close", "volume", "week52_high", "week52_low", "total_shares", "market_cap", "cost_basis"]
        assert EQUITIES_FEATURE_COLUMNS[10:] == ["adj_close", "dividend", "split_ratio", "days_since_week52_high", "days_since_week52_low", "days_since_report"]

    def test_splits_and_dividends_ride_the_same_download(self) -> None:
        """``actions=True`` on the existing call -- no second request."""
        captured = {}

        def spy(symbol, **kwargs):
            captured.update(kwargs)
            return _ohlcv(seed=11)

        with patch.object(eq_gen.yf, "download", side_effect=spy):
            eq_gen.EquitiesGenerator._download_ohlcv("AAPL", "2008-01-01", "2011-01-01", use_cache=False)
        assert captured.get("actions") is True

    def test_absent_action_columns_mean_zero_not_missing(self) -> None:
        """A ticker with no dividends or splits gets 0.0, never NaN.

        yfinance omits the columns entirely in that case. Treating absent as
        missing would put NaN into a float32 feature column for the majority of
        rows of the majority of tickers.
        """
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=12)}, _shares())
        for column in ("dividend", "split_ratio"):
            values = arrays["X_full"][:, _FEATURES.index(column)]
            assert not np.isnan(values).any()
            assert np.all(values == 0.0)

    def test_week52_dates_point_at_the_reported_extreme(self) -> None:
        """The date must identify the row whose value ``week52_high`` reports.

        This is the arm that would catch an off-by-one in the window alignment --
        a date one row off still looks entirely plausible in isolation.
        """
        frame = _ohlcv(seed=13)
        arrays = _generate(["AAPL"], {"AAPL": frame}, _shares())
        highs = arrays["X_full"][:, _FEATURES.index("week52_high")]
        dates = arrays["date_full"]
        high_dates = arrays["week52_high_date_full"]
        for row in (0, len(highs) // 3, len(highs) - 1):
            position = int(np.where(dates == high_dates[row])[0][0])
            assert arrays["X_full"][position, _FEATURES.index("high")] == pytest.approx(highs[row], rel=1e-6)

    def test_days_since_week52_high_is_never_negative(self) -> None:
        """A trailing window cannot place its extreme in the future."""
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=14)}, _shares())
        for column in ("days_since_week52_high", "days_since_week52_low"):
            values = arrays["X_full"][:, _FEATURES.index(column)]
            assert values.min() >= 0

    def test_report_date_is_the_filing_date_not_the_period_end(self) -> None:
        """The distinction is the whole point: only ``filed`` is safe to condition on.

        ``_shares()`` reports period end 2009-06-30 filed 2009-08-14. A row after
        the filing must carry the FILED date; using the period end would leak the
        figure roughly six weeks before it was public.
        """
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=15)}, _shares())
        reported = arrays["report_date_full"]
        seen = {int(value) for value in reported if value != 0}
        assert 20090814 in seen or 20100813 in seen
        assert 20090630 not in seen, "period end must not be used as the reporting date"

    def test_days_since_report_matches_the_dates(self) -> None:
        """The derived column and the date array must agree, or one of them is wrong."""
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=16)}, _shares())
        ages = arrays["X_full"][:, _FEATURES.index("days_since_report")]
        trade = arrays["date_full"]
        report = arrays["report_date_full"]
        for row in range(0, len(ages), max(1, len(ages) // 5)):
            if report[row] == 0:
                continue
            expected = (pd.Timestamp(str(trade[row])) - pd.Timestamp(str(report[row]))).days
            assert ages[row] == pytest.approx(expected)

    def test_adj_close_is_carried_not_recomputed(self) -> None:
        """``Adj Close`` was already in the response and dropped at the feature step."""
        frame = _ohlcv(seed=17)
        arrays = _generate(["AAPL"], {"AAPL": frame}, _shares())
        adj = arrays["X_full"][:, _FEATURES.index("adj_close")]
        np.testing.assert_allclose(adj, frame["Adj Close"].to_numpy()[: len(adj)], rtol=1e-5)

    def test_rolling_extreme_positions_matches_a_naive_scan(self) -> None:
        """The strided implementation must agree with the obvious O(n*w) one.

        The fast version exists because ``rolling(...).apply()`` would add seconds
        per ticker; correctness is pinned against the slow version rather than
        assumed from the construction.
        """
        rng = np.random.default_rng(4)
        values = rng.normal(size=120)
        for window in (1, 5, 30, 250):
            fast = eq_gen.EquitiesGenerator._rolling_extreme_positions(values, window, take_max=True)
            naive = [int(np.argmax(values[max(0, i - min(window, len(values)) + 1) : i + 1])) + max(0, i - min(window, len(values)) + 1) for i in range(len(values))]
            assert fast.tolist() == naive, f"window={window}"

    def test_shares_are_not_visible_before_they_were_filed(self) -> None:
        """No row may carry a figure that was not public on its own trade date.

        The alignment used to key on the period END and forward-fill, so a figure
        reached every trade date between the period it described and the filing
        that disclosed it, and `market_cap` inherited that. A live 2013-2021 AAPL
        run surfaced ``days_since_report`` at **-19 days** -- a negative age is the
        leak stating itself -- on 325 of 2,266 rows (14.3%).

        ``end`` on the dei cover-page tag is an AS-OF date, not a fiscal period
        end. An earlier version of this docstring said AAPL's quarter ending
        2021-03-27 went unfiled until 2021-04-29, "five weeks": AAPL's series has
        no 2021-03 point at all, and that filing carries ``end=2021-04-16``, a
        13-day gap. The -19 comes from four 2015-2016 filings. The leak was real;
        the worked example was not.

        Same class as juniper-data#314's normalisation leak, and the same rule:
        no quantity that was not knowable at a row's date may reach that row.

        THE THIRD ARM IS THE LOAD-BEARING ONE. The first two read only
        ``report_date``, so an implementation that keeps a correct filing date
        while shifting ``total_shares`` itself into the future -- a look-ahead in
        the exact quantity this test is named for -- passes both of them. Only a
        check of the VALUE against what had actually been filed catches it.
        """
        shares = _shares_quarterly()
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=18)}, shares)
        ages = arrays["X_full"][:, _FEATURES.index("days_since_report")]
        assert ages.min() >= 0, "a negative filing age means the row saw its own future"

        trade = arrays["date_full"]
        report = arrays["report_date_full"]
        future = [(int(t), int(r)) for t, r in zip(trade, report, strict=False) if r != 0 and r > t]
        assert not future, f"report_date after the trade date on {len(future)} row(s): {future[:3]}"

        published = shares.dropna(subset=["filed"]).sort_values("filed")
        observed = arrays["X_full"][:, _FEATURES.index("total_shares")]
        checked = 0
        for row in range(len(trade)):
            day = pd.Timestamp(str(trade[row]))
            visible = published[published["filed"] <= day]
            if visible.empty:
                continue
            expected = float(visible["shares"].iloc[-1])
            assert observed[row] == pytest.approx(expected, rel=1e-9), f"row {row} ({day.date()}) carries {observed[row]:,.0f} shares; only {expected:,.0f} had been filed by then"
            checked += 1
        assert checked > 100, f"only {checked} rows had a published figure to compare against -- the arm is vacuous"

    def test_a_same_day_restatement_keeps_the_current_period(self) -> None:
        """Two facts filed the same day: the LATER period end must survive.

        An 8-K restating an old quarter can be filed on the same day as the
        current 10-Q. Only one survives ``duplicated(keep="last")``, and what
        "shares outstanding" means on that filing date is the current period's
        figure, not the restated old one.

        This used to be resolved by ``set_index("filed").sort_index()``, which is
        correct only if the re-sort preserves the incoming ``end``-ascending order
        among ties. ``sort_index()`` defaults to ``kind="quicksort"``, which is not
        stable, so it did not. Measured over the 485-payload SEC cache: 54 tickers
        have such a collision and **15, across 9 tickers, kept the restated OLD
        figure** -- DVA by 10.4%, O'Reilly by 6.8%, KO by 0.9%, and ADSK, which
        sits at position 12 of the DEFAULT 14-symbol universe, by 0.26%. Each is a
        silently wrong ``market_cap`` on every row until the next filing.

        The fixture reproduces that misordering against the old implementation.
        The assertion states the INVARIANT rather than the misordering, so it keeps
        its meaning if a future numpy happens to order this particular case right.
        """
        shares = _shares_quarterly(restated=0, restated_filed_with=32)
        tie = shares["filed"].iloc[32]
        current = float(shares["shares"].iloc[32])
        restated = float(shares["shares"].iloc[0])
        assert tie == shares["filed"].iloc[0], "fixture must put both filings on one day"

        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=19)}, shares)
        trade = arrays["date_full"]
        observed = arrays["X_full"][:, _FEATURES.index("total_shares")]
        next_filing = min(value for value in shares["filed"] if value > tie)
        window = [row for row in range(len(trade)) if tie <= pd.Timestamp(str(trade[row])) < next_filing]
        assert window, "fixture must leave trade rows between the tied filing and the next one"

        for row in window:
            assert observed[row] == pytest.approx(current, rel=1e-9), f"row {row} ({pd.Timestamp(str(trade[row])).date()}) carries {observed[row]:,.0f} -- the RESTATED {restated:,.0f}, not the current period's {current:,.0f}"

    def test_a_point_with_no_filing_date_is_dropped_not_guessed(self) -> None:
        """Unknown publication time cannot be approximated by the period end.

        Falling back to the period end is exactly the leak above. Dropping the
        point is the honest alternative -- the row then has no shares figure,
        which ``fundamentals_fill`` already handles.
        """
        undated = pd.DataFrame(
            {"shares": [1_000_000_000.0], "filed": [pd.NaT]},
            index=pd.to_datetime([pd.Timestamp("2009-06-30")]),
        )
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=19)}, undated, allow_truncation=True)
        shares = arrays["X_full"][:, _FEATURES.index("total_shares")]
        assert np.all(shares == 0.0), "an undated filing must not populate total_shares"
        assert np.all(arrays["report_date_full"] == 0)

    def test_rolling_extreme_positions_rejects_an_ambiguous_request(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            eq_gen.EquitiesGenerator._rolling_extreme_positions(np.zeros(3), 2, take_max=True, take_min=True)


@pytest.mark.unit
class TestUnresolvableFundamentals:
    """The fail / accept / drop contract for rows no rescue path could recover.

    The owner's direction (2026-09-05): a zero-filled fundamental is acceptable
    only as a **warning** when the value can be rescued. When it cannot, the
    caller must take an explicit path — and the default is to refuse, because a
    dataset silently carrying `market_cap = 0.0` is exactly the failure this
    whole area exists to prevent.
    """

    @staticmethod
    def _no_shares(seed: int = 40):
        return {"AAPL": _ohlcv(seed=seed), "MSFT": _ohlcv(seed=seed + 1)}

    def test_default_is_refusal(self) -> None:
        """Unset gate ⇒ the data load fails, with a message naming the remedy."""
        with pytest.raises(eq_limits.IncompleteDataError) as excinfo:
            _generate(["AAPL", "MSFT"], self._no_shares(), shares=None)
        message = str(excinfo.value)
        assert "allow_truncation" in message
        assert "incomplete_rows" in message
        assert "JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION" in message
        assert excinfo.value.unrescued == ["AAPL", "MSFT"]
        assert excinfo.value.rows_affected > 0

    def test_refusal_is_a_value_error(self) -> None:
        """So a missed catch lands on 400, never a 500."""
        with pytest.raises(ValueError):
            _generate(["AAPL"], {"AAPL": _ohlcv(seed=42)}, shares=None)

    def test_accept_keeps_the_rows_and_annotates_them(self) -> None:
        """Canopy's choice 1 / the CLI's opted-in path."""
        arrays = _generate(["AAPL", "MSFT"], self._no_shares(43), shares=None, allow_truncation=True)
        quality = arrays[eq_limits.DATA_QUALITY_META_KEY]
        assert quality["complete"] is False
        assert quality["policy"] == "accept"
        assert sorted(quality["unrescued"]) == ["AAPL", "MSFT"]
        assert quality["rows_affected"] == arrays["X_full"].shape[0]
        assert len(arrays["ticker_vocab"]) == 2, "accept must not remove the symbols"

    def test_drop_removes_the_symbols_and_says_so(self) -> None:
        """Canopy's choice 2 — the rows are gone, and the record of that is not."""
        good = _shares()
        ohlcv = {"AAPL": _ohlcv(seed=44), "MSFT": _ohlcv(seed=45)}

        def selective(cik, _use_cache):  # noqa: ANN001, ANN202
            return good if cik == 320193 else None

        params = EquitiesParams(symbols=["AAPL", "MSFT"], start_date="2008-01-01", end_date="2011-01-01", use_cache=False, allow_truncation=True, incomplete_rows="drop")
        with patch.object(eq_gen.yf, "download", side_effect=lambda symbol, **_k: ohlcv[symbol].copy()), patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(selective)):
            arrays = EquitiesGenerator.generate(params)

        assert arrays["ticker_vocab"].tolist() == ["AAPL"], "the unresolvable symbol must be gone"
        quality = arrays[eq_limits.DATA_QUALITY_META_KEY]
        assert quality["policy"] == "drop"
        # Dropping still annotates. A dataset that quietly contains fewer symbols
        # than were asked for is the same silent-partial-data problem wearing a
        # different costume -- naming what went is the point.
        assert list(quality["unrescued"]) == ["MSFT"]
        assert quality["rows_affected"] == 0, "dropped rows are not IN the dataset to be affected"

    def test_drop_that_empties_the_dataset_still_fails(self) -> None:
        """Dropping everything is not a successful load of nothing."""
        with pytest.raises(eq_limits.IncompleteDataError, match="leaves no dataset"):
            _generate(["AAPL"], {"AAPL": _ohlcv(seed=46)}, shares=None, allow_truncation=True, incomplete_rows="drop")

    def test_a_clean_dataset_carries_no_annotation(self) -> None:
        """Absence is the signal, exactly as for truncation."""
        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=47)}, _shares())
        assert eq_limits.DATA_QUALITY_META_KEY not in arrays

    def test_a_clean_dataset_does_not_depend_on_the_gate(self) -> None:
        """The knobs must not change a dataset that has nothing wrong with it."""
        without = _generate(["AAPL"], {"AAPL": _ohlcv(seed=48)}, _shares())
        with_gate = _generate(["AAPL"], {"AAPL": _ohlcv(seed=48)}, _shares(), allow_truncation=True)
        np.testing.assert_array_equal(without["X_full"], with_gate["X_full"])

    def test_deployment_gate_works_without_a_request_parameter(self) -> None:
        """The env-var / .env surface, for a command-line caller."""
        settings = MagicMock()
        settings.equities_max_symbols = 14
        settings.equities_allow_truncation = True
        settings.equities_incomplete_rows = "accept"
        with patch("juniper_data.api.settings.get_settings", return_value=settings):
            arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=49)}, shares=None)
        assert arrays[eq_limits.DATA_QUALITY_META_KEY]["policy"] == "accept"

    def test_a_degraded_rescue_is_recorded_separately_from_an_absent_one(self) -> None:
        """A period average is a RESCUE, not a gap -- and not the same as a real one.

        A market cap built on a period-average share count is a different quantity
        from one built on point-in-time shares. Merging the two categories would
        hide that from anyone comparing symbols.
        """
        degraded_shares = _shares()
        degraded_shares["shares_quality"] = eq_gen.SHARES_QUALITY_PERIOD_AVERAGE
        degraded_shares["shares_origin"] = eq_gen.SHARES_SOURCE_FACTS

        arrays = _generate(["AAPL"], {"AAPL": _ohlcv(seed=50)}, degraded_shares)
        quality = arrays[eq_limits.DATA_QUALITY_META_KEY]
        assert quality["degraded"] == {"AAPL": "period_average"}
        assert quality["unrescued"] == {}
        # Degraded alone must NOT trip the refusal -- the value was recovered.
        assert quality["policy"] == "accept"
