"""Edges of the W1.8 cost-basis drop that ``TestCostBasisUnderDrop`` cannot see.

``TestCostBasisUnderDrop`` (``test_equities_generator.py``) refuses a later
``purchase_date`` and drops one weekend row inside ``_condition_one``. Its price
fixture sets ``Adj Close`` equal to ``Close``, its row-drop case stops at
``_condition_one`` instead of building an artifact, and none of its cases builds an
``equities_seq`` artifact. The three behaviors below are the ones a refactor of that
block can break while every existing assertion stays green.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from juniper_data.generators.equities import EquitiesGenerator, EquitiesParams  # noqa: E402
from juniper_data.generators.equities import generator as eq_gen  # noqa: E402
from juniper_data.generators.equities.defaults import EQUITIES_FEATURE_COLUMNS  # noqa: E402
from juniper_data.generators.equities_seq import EquitiesSeqGenerator, EquitiesSeqParams  # noqa: E402
from juniper_data.tests.partitions import whole  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]

_COST = EQUITIES_FEATURE_COLUMNS.index("cost_basis")
_PURCHASE = "2009-08-17"
_AUG13 = (pd.Timestamp("2009-08-13"), 7.0, 3.0)
_AUG14 = (pd.Timestamp("2009-08-14"), 13.0, 5.0)


def _sessions(start: str, periods: int, purchase_close: float, purchase_adj: float | None = None) -> Any:
    """Business-day OHLCV whose first session is the purchase, at a known price."""
    index = pd.bdate_range(start=start, periods=periods)
    close = np.full(periods, 80.0)
    close[0] = purchase_close
    adj = np.full(periods, 70.0)
    adj[0] = purchase_close if purchase_adj is None else purchase_adj
    high = np.maximum(close, adj) + 1.0
    low = np.minimum(close, adj) - 1.0
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Adj Close": adj, "Volume": np.full(periods, 1_000_000.0)},
        index=index,
    )


def _prefixed(frame: Any, specs: tuple[tuple[Any, float, float], ...]) -> Any:
    """Provider rows dated before ``start_date``. Shares are already public, so the shares drop keeps them."""
    rows = []
    for when, close, adj in specs:
        row = frame.iloc[[0]].copy()
        row.index = pd.DatetimeIndex([when])
        row["Open"] = close
        row["High"] = max(close, adj) + 1.0
        row["Low"] = min(close, adj) - 1.0
        row["Close"] = close
        row["Adj Close"] = adj
        rows.append(row)
    return pd.concat([*rows, frame])


def _shares() -> Any:
    """One filing, public before every row in these fixtures, so a missing share count cannot explain a dropped row."""
    return pd.DataFrame(
        {"shares": [1_000_000_000.0], "filed": [pd.Timestamp("2009-07-15")]},
        index=pd.to_datetime([pd.Timestamp("2009-06-30")]),
    )


@contextmanager
def _mocked(ohlcv_map: dict):
    def fake_download(symbol, **_kwargs):
        frame = ohlcv_map.get(symbol)
        return frame.copy() if frame is not None else pd.DataFrame()

    def fake_shares(_cik, _use_cache):
        return _shares()

    with patch.object(eq_gen.yf, "download", side_effect=fake_download), patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(fake_shares)):
        yield


def _generate(kind: str, ohlcv_map: dict, **overrides):
    common = {"symbols": list(ohlcv_map), "start_date": _PURCHASE, "purchase_date": _PURCHASE, "end_date": "2010-01-01", "use_cache": False, "normalize_features": False}
    common.update(overrides)
    with _mocked(ohlcv_map):
        if kind == "equities":
            return EquitiesGenerator.generate(EquitiesParams(**common))
        return EquitiesSeqGenerator.generate(EquitiesSeqParams(lookback=4, **common))


def _basis_dates_codes(arrays: dict):
    features = whole(arrays, "X")
    dates = whole(arrays, "date")
    codes = whole(arrays, "ticker_code")
    basis = features[:, :, _COST] if features.ndim == 3 else features[:, _COST]
    return basis, dates, codes


def _code(arrays: dict, ticker: str) -> int:
    return arrays["ticker_vocab"].tolist().index(ticker)


def _yyyymmdd(when: Any) -> int:
    return int(when.strftime("%Y%m%d"))


class TestProviderRowsBeforeTheStart:
    """A row before ``start_date`` survives the weekday check and the shares drop.

    The check counts weekdays between the two request dates, so a provider row
    dated earlier never makes the request illegal. ``drop`` still has to take it
    out of the artifact, and the basis has to be the purchase session's own price.
    """

    @pytest.mark.parametrize("kind", ["equities", "equities_seq"])
    def test_drop_omits_the_early_rows_and_keeps_each_tickers_purchase_price(self, kind: str, caplog: pytest.LogCaptureFixture) -> None:
        aapl = _prefixed(_sessions(_PURCHASE, 40, purchase_close=100.0), (_AUG13, _AUG14))
        msft = _sessions(_PURCHASE, 40, purchase_close=200.0)
        with caplog.at_level("WARNING", logger=eq_gen.__name__):
            arrays = _generate(kind, {"AAPL": aapl, "MSFT": msft}, fundamentals_fill="drop")

        basis, dates, codes = _basis_dates_codes(arrays)
        aapl_code, msft_code = _code(arrays, "AAPL"), _code(arrays, "MSFT")
        early_dates = {_yyyymmdd(_AUG13[0]), _yyyymmdd(_AUG14[0])}
        assert early_dates.isdisjoint(set(np.asarray(dates).ravel().tolist()))
        assert _yyyymmdd(pd.Timestamp(_PURCHASE)) in set(np.asarray(dates[codes == aapl_code]).ravel().tolist())
        assert np.allclose(basis[codes == aapl_code], np.float32(100.0))
        assert np.allclose(basis[codes == msft_code], np.float32(200.0))
        assert "AAPL dropped 2 row(s) dated before its purchase session 2009-08-17" in caplog.text
        assert "MSFT dropped" not in caplog.text

    @pytest.mark.parametrize("kind", ["equities", "equities_seq"])
    def test_nan_fill_still_emits_the_early_rows_with_an_absent_basis(self, kind: str) -> None:
        aapl = _prefixed(_sessions(_PURCHASE, 40, purchase_close=100.0), (_AUG13, _AUG14))
        arrays = _generate(kind, {"AAPL": aapl}, fundamentals_fill="nan")
        basis, dates, codes = _basis_dates_codes(arrays)
        aapl_dates = np.asarray(dates[codes == _code(arrays, "AAPL")]).ravel()
        aapl_basis = np.asarray(basis[codes == _code(arrays, "AAPL")], dtype=np.float64).ravel()
        for when, _close, _adj in (_AUG13, _AUG14):
            at = np.flatnonzero(aapl_dates == _yyyymmdd(when))
            assert at.size > 0, f"{when.date()} must survive under nan"
            assert np.isnan(aapl_basis[at]).all()
        on_purchase = aapl_dates == _yyyymmdd(pd.Timestamp(_PURCHASE))
        assert on_purchase.any()
        assert np.allclose(aapl_basis[on_purchase], np.float32(100.0))


class TestMissingPurchasePrice:
    """A NaN price on the purchase session is a different defect from a pre-purchase row.

    The drop is index-based for that reason. Deleting every row whose ``cost_basis``
    is NaN would erase the ticker, because the basis value itself is NaN and every
    later session inherits it.
    """

    @pytest.mark.parametrize("kind", ["equities", "equities_seq"])
    def test_later_sessions_survive_and_do_not_inherit_the_early_price(self, kind: str) -> None:
        frame = _prefixed(_sessions(_PURCHASE, 40, purchase_close=100.0), (_AUG13, _AUG14))
        purchase = pd.Timestamp(_PURCHASE)
        frame.loc[purchase, ["Close", "Adj Close"]] = np.nan
        later = pd.bdate_range(start=purchase, periods=2)[1]
        arrays = _generate(kind, {"AAPL": frame}, fundamentals_fill="drop")

        basis, dates, _codes = _basis_dates_codes(arrays)
        seen = set(np.asarray(dates).ravel().tolist())
        assert whole(arrays, "X").shape[0] > 0
        assert _yyyymmdd(later) in seen
        # Aug 13's own next close is finite, so only the pre-purchase drop removes it.
        # Aug 14 sits directly before the missing purchase price and loses its target either way.
        assert _yyyymmdd(_AUG13[0]) not in seen
        assert _yyyymmdd(purchase) not in seen
        assert np.isnan(np.asarray(basis, dtype=np.float64)).all()


class TestBasisPriceField:
    """``basis_price_field`` chooses the purchase session's price. The other column is a different number here."""

    @pytest.mark.parametrize("kind", ["equities", "equities_seq"])
    def test_adj_close_is_used_when_asked_and_close_is_used_by_default(self, kind: str) -> None:
        frame = _prefixed(_sessions(_PURCHASE, 40, purchase_close=100.0, purchase_adj=42.5), (_AUG13,))
        adjusted = _generate(kind, {"AAPL": frame}, fundamentals_fill="drop", basis_price_field="adj_close")
        raw = _generate(kind, {"AAPL": frame}, fundamentals_fill="drop", basis_price_field="close")
        adjusted_basis, adjusted_dates, _codes = _basis_dates_codes(adjusted)
        raw_basis, raw_dates, _raw_codes = _basis_dates_codes(raw)
        assert _yyyymmdd(_AUG13[0]) not in set(np.asarray(adjusted_dates).ravel().tolist())
        assert _yyyymmdd(_AUG13[0]) not in set(np.asarray(raw_dates).ravel().tolist())
        assert np.allclose(adjusted_basis, np.float32(42.5))
        assert np.allclose(raw_basis, np.float32(100.0))

    @pytest.mark.parametrize("kind", ["equities", "equities_seq"])
    def test_a_provider_frame_without_adj_close_still_yields_the_close(self, kind: str) -> None:
        frame = _sessions(_PURCHASE, 40, purchase_close=100.0).drop(columns=["Adj Close"])
        arrays = _generate(kind, {"AAPL": frame}, fundamentals_fill="drop", basis_price_field="adj_close")
        basis, _dates, _codes = _basis_dates_codes(arrays)
        assert np.allclose(basis, np.float32(100.0))
