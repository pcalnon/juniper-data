"""Complementary APD-DATA-018 coverage the trigger suite cannot see.

#354 pins the symbol cap on the flat ``equities`` generator: refusal,
annotation, clamp, and three ``generate()``-level arms that exist because the
E3 mutation (drop the channel-key assignment) survived when every test called
``_resolve_symbols`` directly.

``equities_seq`` reuses that resolver and *also* writes the reserved key --
the PR itself calls dropping the record "the worse of the two halves to
skip" -- but ``test_equities_seq_generator.py`` only ever asks for 1-2
symbols, so deleting the assignment in ``equities_seq/generator.py`` stays
green. The mutation matrix imports ``EQUITIES_SEQ`` and never mutates it.

The rest of this file pins the surfaces ``TestUniverseSymbolCap`` cannot see
even on the flat path: an operator ceiling *below* the measured 14 (every
settings mock there is 14, so a hard-coded constant would pass), the
``gt=0`` / env-var Settings fields, the shared ``InputTooLargeError``
renderer (40 symbols must not become ``0.0 MB``), and the unified
descriptor keys that replaced ``cap_bytes`` / ``bytes_read`` / ``bytes_total``.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from juniper_data.core.limits import (
    EQUITIES_DEFAULT_MAX_SYMBOLS,
    REASON_SYMBOL_CAP,
    TRUNCATION_META_KEY,
    UNIT_BYTES,
    UNIT_SYMBOLS,
    InputTooLargeError,
    build_truncation_meta,
)
from juniper_data.generators.equities import EquitiesParams
from juniper_data.generators.equities import generator as eq_gen

pd = pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from juniper_data.generators.equities_seq import EquitiesSeqGenerator, EquitiesSeqParams  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]


def _ohlcv(start: str = "2008-01-01", periods: int = 40, seed: int = 0):
    """Short synthetic OHLCV -- enough rows for lookback=5, cheap per ticker."""
    index = pd.bdate_range(start=start, periods=periods)
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.0, 1.0, periods).cumsum()
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000.0,
        },
        index=index,
    )


def _shares() -> object:
    """Synthetic shares-outstanding history, as ``_fetch_shares`` now returns it.

    A DataFrame of ``shares`` + ``filed``, not a bare Series. The PR this suite was
    harvested from predates the point-in-time filing-date alignment, and its Series
    fixture made ``_condition_one`` raise ``Series.dropna() got an unexpected keyword
    argument 'subset'`` -- every ticker skipped, then "No data could be retrieved".
    The stale half was the FIXTURE, not the generator: a test's mock encodes the
    production contract as it stood when the test was written.
    """
    return pd.DataFrame(
        {"shares": [1_000_000_000.0, 1_100_000_000.0], "filed": [pd.Timestamp("2009-08-14"), pd.Timestamp("2010-08-13")]},
        index=pd.to_datetime([pd.Timestamp("2009-06-30"), pd.Timestamp("2010-06-30")]),
    )


@contextmanager
def _mocked(ohlcv_map: dict, shares):
    def fake_download(symbol, **_kwargs):
        frame = ohlcv_map.get(symbol)
        return frame.copy() if frame is not None else pd.DataFrame()

    with (
        patch.object(eq_gen.yf, "download", side_effect=fake_download),
        patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(lambda *_a: shares)),
    ):
        yield


def _seq_generate(symbols: list[str], ohlcv_map: dict, **overrides):
    params = EquitiesSeqParams(
        symbols=symbols,
        start_date="2008-01-01",
        end_date="2011-01-01",
        use_cache=False,
        lookback=5,
        **overrides,
    )
    with _mocked(ohlcv_map, _shares()):
        return EquitiesSeqGenerator.generate(params)


def _ticker_map(count: int) -> tuple[list[str], dict]:
    tickers = [f"T{i:02d}" for i in range(count)]
    return tickers, {ticker: _ohlcv(seed=index) for index, ticker in enumerate(tickers)}


class TestEquitiesSeqSymbolCap:
    """E3-seq: the annotation has to survive ``equities_seq.generate()``, not just the resolver."""

    def test_generate_puts_the_annotation_on_the_returned_arrays(self) -> None:
        """Deleting the channel assignment in equities_seq/generator.py must go red.

        The flat generator grew this arm after E3 survived. The sibling never
        did, and inheriting the bound while dropping the record is the half
        the PR named as worse to skip.
        """
        tickers, ohlcv = _ticker_map(16)
        arrays = _seq_generate(tickers, ohlcv, allow_truncation=True)

        annotation = arrays[TRUNCATION_META_KEY]
        assert annotation["truncated"] is True
        assert annotation["reason"] == "universe_exceeded_symbol_cap"
        assert annotation["unit"] == "symbols"
        assert annotation["requested"] == 16
        assert annotation["imported"] == 14
        assert annotation["cap"] == 14
        # Resolver leaves -1; generate() must overwrite with a real window count.
        assert annotation["records_imported"] == arrays["X_full"].shape[0]
        assert annotation["records_imported"] > 0
        assert annotation["records_imported"] != 16
        assert len(arrays["ticker_vocab"]) == 14
        assert arrays["ticker_vocab"].tolist() == tickers[:14]

    def test_generate_refuses_an_oversized_universe(self) -> None:
        """The refusal must survive all the way out through equities_seq.generate()."""
        tickers, ohlcv = _ticker_map(16)
        with pytest.raises(InputTooLargeError) as excinfo:
            _seq_generate(tickers, ohlcv)
        assert excinfo.value.unit == "symbols"
        assert excinfo.value.actual == 16
        assert excinfo.value.cap == 14

    def test_generate_omits_the_key_entirely_when_nothing_was_cut(self) -> None:
        """Absence, not a falsy descriptor -- the same contract the flat generator keeps."""
        arrays = _seq_generate(["AAPL", "MSFT"], {"AAPL": _ohlcv(seed=1), "MSFT": _ohlcv(seed=2)})
        assert TRUNCATION_META_KEY not in arrays
        assert arrays["ticker_vocab"].tolist() == ["AAPL", "MSFT"]

    def test_schema_inherits_the_bound_fields(self) -> None:
        """EquitiesSeqParams subclasses EquitiesParams -- the fields must stay visible."""
        schema = EquitiesSeqParams.model_json_schema()
        assert "max_symbols" in schema["properties"]
        assert "allow_truncation" in schema["properties"]


class TestEquitiesCapSurfacesTheTriggerCannotSee:
    """Surfaces TestUniverseSymbolCap stays green on, even when they break."""

    @staticmethod
    def _universe(count: int) -> dict[str, dict[str, object]]:
        return {f"T{i:03d}": {"name": f"Name {i}", "cik": 1000 + i, "sector": ""} for i in range(count)}

    def test_operator_ceiling_below_the_measured_default_is_honoured(self) -> None:
        """A deployment of 5 must win over the schema default of 14.

        TestUniverseSymbolCap mocks ``equities_max_symbols = 14``, so replacing
        ``settings.equities_max_symbols`` with the constant 14 would stay green
        there. The clamp is only load-bearing when the operator tightens it.
        """
        settings = MagicMock()
        settings.equities_max_symbols = 5
        settings.equities_allow_truncation = True
        with patch("juniper_data.api.settings.get_settings", return_value=settings):
            ordered, _meta, truncation = eq_gen.EquitiesGenerator._resolve_symbols(
                EquitiesParams(allow_truncation=True),
                self._universe(40),
            )
        assert len(ordered) == 5
        assert truncation["cap"] == 5
        assert truncation["imported"] == 5
        assert truncation["requested"] == 40

    def test_settings_reject_a_non_positive_symbol_cap(self) -> None:
        """A mistyped env var must fail deployment, not empty the universe."""
        from juniper_data.api.settings import Settings

        with pytest.raises(ValidationError):
            Settings(equities_max_symbols=-1)
        with pytest.raises(ValidationError):
            Settings(equities_max_symbols=0)

    def test_env_binds_equities_max_symbols(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """JUNIPER_DATA_EQUITIES_MAX_SYMBOLS is one of the three required surfaces."""
        from juniper_data.api.settings import Settings

        monkeypatch.setenv("JUNIPER_DATA_EQUITIES_MAX_SYMBOLS", "7")
        settings = Settings()
        assert settings.equities_max_symbols == 7

    def test_env_binds_equities_allow_truncation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION is the deployment opt-in surface."""
        from juniper_data.api.settings import Settings

        monkeypatch.setenv("JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION", "true")
        settings = Settings()
        assert settings.equities_allow_truncation is True

    def test_settings_default_matches_the_measured_constant(self) -> None:
        from juniper_data.api.settings import Settings

        settings = Settings()
        assert settings.equities_max_symbols == EQUITIES_DEFAULT_MAX_SYMBOLS == 14
        assert settings.equities_allow_truncation is False

    def test_refusal_message_renders_symbols_not_megabytes(self) -> None:
        """``_describe`` must not use the byte formatter for a symbol cap.

        40 / (1024*1024) is 0.0, so a unit-blind MB path would say
        ``0.0 MB over the 0.0 MB cap`` and every existing assertion
        (``allow_truncation`` in the message, ``.unit == "symbols"``) would
        still pass.
        """
        err = InputTooLargeError(
            source="The requested universe",
            unit=UNIT_SYMBOLS,
            cap=14,
            actual=40,
            opt_in_env="JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION",
        )
        text = str(err)
        assert "40 symbols" in text
        assert "14 symbols" in text
        assert "0.0 MB" not in text
        assert "allow_truncation" in text
        assert "JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION" in text

        byte_err = InputTooLargeError(
            source="Source 'wide.csv'",
            unit=UNIT_BYTES,
            cap=128 * 1024 * 1024,
            actual=200 * 1024 * 1024,
            opt_in_env="JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION",
        )
        byte_text = str(byte_err)
        assert "MB" in byte_text
        assert "symbols" not in byte_text

    def test_unified_descriptor_has_no_byte_specific_keys(self) -> None:
        """One shape for both generators. The hour-old cap_bytes keys must not return."""
        descriptor = build_truncation_meta(
            reason=REASON_SYMBOL_CAP,
            unit=UNIT_SYMBOLS,
            cap=14,
            requested=40,
            imported=14,
            records_imported=100,
        )
        assert descriptor == {
            "truncated": True,
            "reason": "universe_exceeded_symbol_cap",
            "unit": "symbols",
            "cap": 14,
            "requested": 40,
            "imported": 14,
            "records_imported": 100,
        }
        for retired in ("cap_bytes", "bytes_read", "bytes_total", "bytes_imported"):
            assert retired not in descriptor

    def test_params_reject_max_symbols_of_zero(self) -> None:
        """``ge=1``: a zero cap would slice the universe empty rather than bound it."""
        with pytest.raises(ValidationError):
            EquitiesParams(max_symbols=0)
        assert EquitiesParams(max_symbols=None).max_symbols is None
