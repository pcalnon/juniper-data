"""W5.7 producer guards: lost symbols (F-P5), the close guard (F-P6), the action columns (F-P8).

Plan: juniper-ml ``notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md``
row W5.7. Motivating case: ``notes/JUNIPER_2026-10-04_JUNIPER-RECURRENCE_EQUITIES-CV-BLOWUP-INVESTIGATION.md``
§1.1 -- two mints of ``equities_seq-6.0.0-15505731cba5b86d`` 85 s apart, the second without the
action columns, served as a complete artifact under the same ``dataset_id``.

Both generators are exercised through the mocked sources of ``test_equities_generator`` and
``test_equities_seq_generator`` (yfinance and SEC patched; no network).
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_equities_producer_guards.py
# Author:        Paul Calnon
# Version:       0.17.1
# License:       MIT License

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from juniper_data.core import limits as eq_limits
from juniper_data.core.limits import DATA_QUALITY_META_KEY, IncompleteDataError, build_data_quality_meta

pd = pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from juniper_data.generators.equities import EquitiesGenerator, EquitiesParams  # noqa: E402
from juniper_data.generators.equities import generator as eq_gen  # noqa: E402
from juniper_data.generators.equities_seq import EquitiesSeqGenerator, EquitiesSeqParams  # noqa: E402
from juniper_data.tests.partitions import whole  # noqa: E402
from juniper_data.tests.unit.test_equities_generator import _generate as _flat_generate  # noqa: E402
from juniper_data.tests.unit.test_equities_generator import _ohlcv as _flat_ohlcv  # noqa: E402
from juniper_data.tests.unit.test_equities_generator import _shares as _flat_shares  # noqa: E402
from juniper_data.tests.unit.test_equities_seq_generator import _generate as _seq_generate  # noqa: E402
from juniper_data.tests.unit.test_equities_seq_generator import _ohlcv as _seq_ohlcv  # noqa: E402
from juniper_data.tests.unit.test_equities_seq_generator import _shares as _seq_shares  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]

_LOGGER = "juniper_data.generators.equities.generator"


def _without_actions(frame):
    """The anomalous response: no ``Dividends`` / ``Stock Splits`` column at all."""
    return frame.drop(columns=["Dividends", "Stock Splits"])


class TestActionColumns:
    """F-P8: record whether the price response carried the action columns; warn on the zero-fill."""

    def test_seq_mint_without_action_columns_is_annotated_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """The plan's acceptance line: ``actions_present: false`` and a WARNING."""
        with caplog.at_level("WARNING", logger=_LOGGER):
            arrays = _seq_generate(["AAPL"], {"AAPL": _without_actions(_seq_ohlcv(seed=70))}, _seq_shares())
        quality = arrays[DATA_QUALITY_META_KEY]
        assert quality["complete"] is False
        assert quality["actions_present"] == {"AAPL": False}
        assert quality["dropped_symbols"] == {}
        assert quality["unrescued"] == {}
        warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
        assert any("AAPL" in message and "dividend / split_ratio" in message and "zero-filling" in message for message in warnings), warnings

    def test_flat_mint_without_action_columns_is_annotated(self) -> None:
        arrays = _flat_generate(["AAPL"], {"AAPL": _without_actions(_flat_ohlcv(seed=71))}, _flat_shares())
        assert arrays[DATA_QUALITY_META_KEY]["actions_present"] == {"AAPL": False}

    def test_one_missing_column_is_enough(self) -> None:
        """Either column absent is an incomplete response; the WARNING names the one that is."""
        arrays = _flat_generate(["AAPL"], {"AAPL": _flat_ohlcv(seed=72).drop(columns=["Stock Splits"])}, _flat_shares())
        assert arrays[DATA_QUALITY_META_KEY]["actions_present"] == {"AAPL": False}

    def test_a_complete_response_carries_no_annotation_and_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Absence of the block stays the signal: the fixtures carry both columns, as yfinance does."""
        with caplog.at_level("WARNING", logger=_LOGGER):
            arrays = _seq_generate(["AAPL"], {"AAPL": _seq_ohlcv(seed=73)}, _seq_shares())
        assert DATA_QUALITY_META_KEY not in arrays
        assert not [record for record in caplog.records if "zero-filling" in record.getMessage()]

    def test_the_zero_fill_itself_is_unchanged(self) -> None:
        """The arrays are what they were: the guard adds a record, not a different artifact."""
        before = _flat_generate(["AAPL"], {"AAPL": _flat_ohlcv(seed=74)}, _flat_shares())
        after = _flat_generate(["AAPL"], {"AAPL": _without_actions(_flat_ohlcv(seed=74))}, _flat_shares())
        np.testing.assert_array_equal(whole(before, "X"), whole(after, "X"))

    def test_every_symbol_is_listed_when_one_is_incomplete(self) -> None:
        """The map covers the whole universe, so a consumer sees which mints to distrust."""
        arrays = _flat_generate(["AAPL", "MSFT"], {"AAPL": _flat_ohlcv(seed=75), "MSFT": _without_actions(_flat_ohlcv(seed=76))}, _flat_shares())
        assert arrays[DATA_QUALITY_META_KEY]["actions_present"] == {"AAPL": True, "MSFT": False}

    def test_the_flag_does_not_trip_the_gate(self) -> None:
        """Refusal is a request-level choice: the generator cannot know whether actions exist in range."""
        arrays = _flat_generate(["AAPL"], {"AAPL": _without_actions(_flat_ohlcv(seed=77))}, _flat_shares())
        assert arrays[DATA_QUALITY_META_KEY]["policy"] == "accept"


class TestLostRequestedSymbols:
    """F-P5: a requested symbol that yields nothing is refused under the shut gate, recorded under the open one."""

    def test_flat_lost_requested_symbol_is_refused_by_default(self) -> None:
        with pytest.raises(IncompleteDataError) as excinfo:
            _flat_generate(["AAPL", "NVDA"], {"AAPL": _flat_ohlcv(seed=80)}, _flat_shares())
        message = str(excinfo.value)
        assert excinfo.value.unrescued == ["NVDA"]
        assert excinfo.value.rows_affected == 0
        assert "NVDA: no usable price data in range" in message
        assert "allow_truncation" in message and "JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION" in message
        assert "fabricated" not in message, "no rows exist to carry a fabricated value"
        assert "incomplete_rows" not in message, "accept / drop chooses between rows that exist"

    def test_seq_lost_requested_symbol_is_refused_by_default(self) -> None:
        with pytest.raises(IncompleteDataError, match="MSFT"):
            _seq_generate(["AAPL", "MSFT"], {"AAPL": _seq_ohlcv(seed=81)}, _seq_shares())

    def test_refusal_is_a_value_error(self) -> None:
        """So a missed catch lands on 400, never a 500; the route maps the subclass to 422."""
        with pytest.raises(ValueError):
            _flat_generate(["AAPL", "NVDA"], {"AAPL": _flat_ohlcv(seed=82)}, _flat_shares())

    def test_open_gate_mints_without_the_symbol_and_records_it(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger=_LOGGER):
            arrays = _flat_generate(["AAPL", "NVDA"], {"AAPL": _flat_ohlcv(seed=83)}, _flat_shares(), allow_truncation=True)
        assert arrays["ticker_vocab"].tolist() == ["AAPL"]
        quality = arrays[DATA_QUALITY_META_KEY]
        assert quality["complete"] is False
        assert quality["dropped_symbols"] == {"NVDA": "no usable price data in range"}
        assert quality["unrescued"] == {} and quality["rows_affected"] == 0
        assert quality["actions_present"] == {"AAPL": True}, "only symbols IN the arrays carry the flag"
        assert any("NVDA" in record.getMessage() and "absent from the dataset" in record.getMessage() for record in caplog.records)

    def test_a_conditioning_failure_records_its_exception(self) -> None:
        good = _seq_ohlcv(seed=84)

        def fake_download(symbol, **_kwargs):
            if symbol == "MSFT":
                raise RuntimeError("download exploded")
            return good.copy()

        params = EquitiesSeqParams(symbols=["AAPL", "MSFT"], start_date="2008-01-01", end_date="2011-01-01", use_cache=False, lookback=5, allow_truncation=True)
        with patch.object(eq_gen.yf, "download", side_effect=fake_download), patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(lambda *_a: _seq_shares())):
            arrays = EquitiesSeqGenerator.generate(params)
        assert arrays[DATA_QUALITY_META_KEY]["dropped_symbols"] == {"MSFT": "conditioning failed: download exploded"}

    def test_seq_too_few_rows_for_a_window_is_a_loss(self) -> None:
        """The windowing skip was the third silent shrink; it is classified before the policy."""
        with pytest.raises(IncompleteDataError, match="lookback") as excinfo:
            _seq_generate(["AAPL", "MSFT"], {"AAPL": _seq_ohlcv(seed=85, periods=400), "MSFT": _seq_ohlcv(seed=86, periods=8)}, _seq_shares(), lookback=20)
        assert excinfo.value.unrescued == ["MSFT"]

    def test_seq_short_ticker_stays_in_the_vocab_with_zero_windows(self) -> None:
        """Deliberate: the code allocation is unchanged, so an existing dataset_id keeps its arrays."""
        arrays = _seq_generate(["AAPL", "MSFT"], {"AAPL": _seq_ohlcv(seed=87, periods=400), "MSFT": _seq_ohlcv(seed=88, periods=8)}, _seq_shares(), lookback=20, allow_truncation=True)
        assert arrays["ticker_vocab"].tolist() == ["AAPL", "MSFT"]
        assert not (whole(arrays, "ticker_code") == 1).any()
        assert "no window" in arrays[DATA_QUALITY_META_KEY]["dropped_symbols"]["MSFT"]

    def test_default_universe_losses_are_recorded_but_never_refused(self) -> None:
        """Nothing was asked for by name, so there is nothing to refuse -- only to record."""
        constituents = {"AAPL": {"name": "Apple", "cik": 320193, "sector": ""}, "MSFT": {"name": "Microsoft", "cik": 789019, "sector": ""}}
        params = EquitiesParams(symbols=None, start_date="2008-01-01", end_date="2011-01-01", use_cache=False)
        frames = {"AAPL": _flat_ohlcv(seed=89)}
        with (
            patch.object(eq_gen.EquitiesGenerator, "_load_constituents", staticmethod(lambda: constituents)),
            patch.object(eq_gen.yf, "download", side_effect=lambda symbol, **_k: frames[symbol].copy() if symbol in frames else pd.DataFrame()),
            patch.object(eq_gen.EquitiesGenerator, "_fetch_shares", staticmethod(lambda *_a: _flat_shares())),
        ):
            arrays = EquitiesGenerator.generate(params)
        assert arrays["ticker_vocab"].tolist() == ["AAPL"]
        assert arrays[DATA_QUALITY_META_KEY]["dropped_symbols"] == {"MSFT": "no usable price data in range"}

    def test_deployment_gate_opens_it_for_a_cli_caller(self) -> None:
        settings = MagicMock()
        settings.equities_max_symbols = 14
        settings.equities_allow_truncation = True
        settings.equities_incomplete_rows = "accept"
        with patch("juniper_data.api.settings.get_settings", return_value=settings):
            arrays = _flat_generate(["AAPL", "NVDA"], {"AAPL": _flat_ohlcv(seed=90)}, _flat_shares())
        assert list(arrays[DATA_QUALITY_META_KEY]["dropped_symbols"]) == ["NVDA"]

    def test_an_unrescued_fundamental_still_takes_precedence_in_the_message(self) -> None:
        """Both defects under the shut gate: the fundamentals refusal is the one raised (its remedy differs)."""
        with pytest.raises(IncompleteDataError, match="fabricated") as excinfo:
            _flat_generate(["AAPL", "NVDA"], {"AAPL": _flat_ohlcv(seed=91)}, shares=None)
        assert excinfo.value.unrescued == ["AAPL"]

    def test_every_symbol_lost_is_still_the_old_no_data_error(self) -> None:
        """Nothing conditioned: the existing refusal fires before the policy, unchanged."""
        with pytest.raises(ValueError, match="No data"):
            _flat_generate(["NVDA"], {}, _flat_shares(), allow_truncation=True)


class TestCloseGuard:
    """F-P6: ``close`` / ``next_close`` finite and positive before a return-type target."""

    @staticmethod
    def _with_bad_close(frame, value: float, at: int = 100):
        frame = frame.copy()
        frame.iloc[at, frame.columns.get_loc("Close")] = value
        return frame

    @pytest.mark.parametrize("mode", ["return", "log_return"])
    @pytest.mark.parametrize("value", [0.0, -3.0, np.inf])
    def test_flat_refuses_a_return_target_over_a_bad_close(self, mode: str, value: float) -> None:
        with pytest.raises(ValueError, match=r"AAPL has 2 row\(s\) with a non-positive or non-finite close") as excinfo:
            _flat_generate(["AAPL"], {"AAPL": self._with_bad_close(_flat_ohlcv(seed=92), value)}, _flat_shares(), regression_target=mode)
        # Two rows: the session itself (its `close`) and the one before it (its `next_close`).
        assert f"regression_target={mode!r}" in str(excinfo.value)
        assert "next_close" in str(excinfo.value), "the message offers the raw-price target as the remedy"

    @pytest.mark.parametrize("mode", ["return", "log_return"])
    def test_seq_refuses_a_return_target_over_a_zero_close(self, mode: str) -> None:
        with pytest.raises(ValueError, match="MSFT has 2 row"):
            _seq_generate(["AAPL", "MSFT"], {"AAPL": _seq_ohlcv(seed=93), "MSFT": self._with_bad_close(_seq_ohlcv(seed=94), 0.0)}, _seq_shares(), regression_target=mode)

    def test_the_message_names_the_first_offending_session(self) -> None:
        frame = self._with_bad_close(_flat_ohlcv(seed=95), 0.0, at=100)
        first = frame.index[99].date()  # the row BEFORE the zero: its next_close is the zero
        with pytest.raises(ValueError, match=str(first)):
            _flat_generate(["AAPL"], {"AAPL": frame}, _flat_shares(), regression_target="log_return")

    def test_next_close_is_not_guarded(self) -> None:
        """A zero price is then a value, not a divisor -- the raw target mints as before."""
        arrays = _flat_generate(["AAPL"], {"AAPL": self._with_bad_close(_flat_ohlcv(seed=96), 0.0)}, _flat_shares(), regression_target="next_close")
        assert np.isfinite(whole(arrays, "y_reg")).all()

    def test_a_clean_frame_mints_a_finite_target(self) -> None:
        arrays = _flat_generate(["AAPL"], {"AAPL": _flat_ohlcv(seed=97)}, _flat_shares(), regression_target="log_return")
        assert np.isfinite(whole(arrays, "y_reg")).all()

    def test_the_guard_is_a_plain_value_error_not_a_422(self) -> None:
        """A bad price is not caller-fixable by opting in; it is a 400, not the incomplete-data 422."""
        with pytest.raises(ValueError) as excinfo:
            _flat_generate(["AAPL"], {"AAPL": self._with_bad_close(_flat_ohlcv(seed=98), 0.0)}, _flat_shares(), regression_target="return")
        assert not isinstance(excinfo.value, IncompleteDataError)


class TestDescriptorAndError:
    """The two shared pieces in ``core/limits.py``."""

    def test_builder_is_none_when_every_category_is_clean(self) -> None:
        assert build_data_quality_meta(unrescued={}, degraded={}, rows_affected=0, policy="accept", dropped={}, actions_present={"AAPL": True}) is None
        assert build_data_quality_meta(unrescued={}, degraded={}, rows_affected=0, policy="accept") is None

    def test_builder_flags_a_dropped_symbol_alone(self) -> None:
        meta = build_data_quality_meta(unrescued={}, degraded={}, rows_affected=0, policy="accept", dropped={"NVDA": "no usable price data in range"})
        assert meta == {"complete": False, "unrescued": {}, "degraded": {}, "dropped_symbols": {"NVDA": "no usable price data in range"}, "actions_present": {}, "rows_affected": 0, "policy": "accept"}

    def test_builder_flags_a_missing_action_column_alone(self) -> None:
        meta = build_data_quality_meta(unrescued={}, degraded={}, rows_affected=0, policy="accept", actions_present={"MSFT": False, "AAPL": True})
        assert meta is not None and meta["complete"] is False
        assert list(meta["actions_present"]) == ["AAPL", "MSFT"], "sorted, like the other maps"

    def test_builder_keeps_the_old_shape_for_the_old_callers(self) -> None:
        meta = build_data_quality_meta(unrescued={"KO": "x"}, degraded={}, rows_affected=3, policy="accept")
        assert meta is not None
        assert meta["dropped_symbols"] == {} and meta["actions_present"] == {}
        assert meta["unrescued"] == {"KO": "x"} and meta["rows_affected"] == 3

    def test_error_wording_follows_rows_affected(self) -> None:
        with_rows = str(IncompleteDataError(detail="d.", unrescued=["A"], rows_affected=5, opt_in_env="E"))
        assert "5 row(s) would carry fabricated values" in with_rows and "incomplete_rows='drop'" in with_rows
        without = str(IncompleteDataError(detail="d.", unrescued=["A"], rows_affected=0, opt_in_env="E"))
        assert "fewer symbols than were requested" in without and "data_quality.dropped_symbols" in without
        assert "fabricated" not in without and "incomplete_rows" not in without
        assert "E=true" in without and "allow_truncation=true" in without

    def test_the_default_policy_constant_is_unchanged(self) -> None:
        assert eq_limits.EQUITIES_DEFAULT_INCOMPLETE_ROWS == "accept"
