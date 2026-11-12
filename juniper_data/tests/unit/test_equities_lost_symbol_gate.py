"""The lost-symbol gate under an explicit refusal, and the bound on its 422 body.

W5.7 (F-P5) refuses a requested symbol that yielded no usable data by raising from
``_apply_incomplete_policy``, on the branch after the unrescued-fundamentals raise.
The two tests named ``test_explicit_false_refuses_against_a_deployment_opt_in`` cover
the tri-state for the symbol cap and for unrescued shares. The gate the shares one
reaches, ``_resolve_incomplete_policy``, is the gate this raise reads, so an ``or``
with the deployment opt-in there already fails it. Neither test enters this raise,
though: both stay green if the lost-symbol branch stops refusing under an explicit
``false``, or if ``incomplete_rows='drop'`` is honoured while the gate is shut. The
default-refusal tests stay green too, since none of them sends an explicit ``false``
or ``drop``.

The 422 body is ``str(IncompleteDataError)``. The detail's reason list and the
``Affected`` list each keep eight symbols; the exception's ``unrescued`` keeps every
one. One lost ticker cannot see either slice.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_equities_lost_symbol_gate.py
# Author:        Paul Calnon
# Version:       0.17.1
# License:       MIT License

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from juniper_data.core.limits import IncompleteDataError

pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from juniper_data.tests.unit.test_equities_generator import _generate as _flat_generate  # noqa: E402
from juniper_data.tests.unit.test_equities_generator import _ohlcv as _flat_ohlcv  # noqa: E402
from juniper_data.tests.unit.test_equities_generator import _shares as _flat_shares  # noqa: E402
from juniper_data.tests.unit.test_equities_seq_generator import _generate as _seq_generate  # noqa: E402
from juniper_data.tests.unit.test_equities_seq_generator import _ohlcv as _seq_ohlcv  # noqa: E402
from juniper_data.tests.unit.test_equities_seq_generator import _shares as _seq_shares  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.generators]

_REASON = "no usable price data in range"


def _deployment_opted_in() -> MagicMock:
    settings = MagicMock()
    settings.equities_max_symbols = 14
    settings.equities_allow_truncation = True
    settings.equities_incomplete_rows = "drop"
    return settings


def _refuse(generate, symbols: list[str], frames: dict, shares) -> IncompleteDataError:
    """Explicit false, with drop requested, against a deployment that allows truncation."""
    with patch("juniper_data.api.settings.get_settings", return_value=_deployment_opted_in()):
        with pytest.raises(IncompleteDataError) as excinfo:
            generate(symbols, frames, shares, allow_truncation=False, incomplete_rows="drop")
    return excinfo.value


def _lost_request(n: int, frame) -> tuple[list[str], dict]:
    """``n`` lost tickers plus AAPL, which the shares fixture can rescue.

    A keeper with no shares data would raise the fundamentals refusal first, and
    the lost-symbol message — the one these tests read — would never be built.
    The last lost ticker is requested first. An unsorted slice of the loss map
    would name it; the message is required to name the alphabetically first ones.
    """
    lost = [f"L{i}" for i in range(n)]
    symbols = [lost[-1], *lost[:-1], "AAPL"]
    return symbols, {"AAPL": frame}


class TestLostSymbolGate:
    """Explicit ``false`` shuts the lost-symbol gate, and the 422 body stays bounded."""

    def test_explicit_false_refuses_a_lost_symbol_when_the_deployment_opted_in(self) -> None:
        """APD-DATA-052 on the F-P5 raise, not on the symbol cap or the shares raise.

        ``incomplete_rows='drop'`` is set on the request and on the deployment. Drop
        chooses among rows that exist, and only after the gate is open. A lost symbol
        has no such rows, so drop must not mint a universe that simply omits it.
        """
        error = _refuse(_flat_generate, ["AAPL", "NVDA"], {"AAPL": _flat_ohlcv(seed=80)}, _flat_shares())
        message = str(error)
        assert error.unrescued == ["NVDA"]
        assert error.rows_affected == 0
        assert "NVDA: no usable price data in range" in message
        assert "fewer symbols than were requested" in message
        assert "fabricated" not in message
        assert "incomplete_rows" not in message

    def test_explicit_false_refuses_a_lost_symbol_on_the_sequence_generator(self) -> None:
        """The sequence generator reaches the same raise; a private copy must not grow an ``or``."""
        error = _refuse(_seq_generate, ["AAPL", "MSFT"], {"AAPL": _seq_ohlcv(seed=81)}, _seq_shares())
        assert error.unrescued == ["MSFT"]
        assert error.rows_affected == 0
        assert "fabricated" not in str(error)
        assert "incomplete_rows" not in str(error)

    def test_eight_lost_symbols_are_named_in_full(self) -> None:
        """Eight is the last count that still fits. An off-by-one ellipsis would hide the last name."""
        symbols, frames = _lost_request(8, _flat_ohlcv(seed=82))
        with pytest.raises(IncompleteDataError) as excinfo:
            _flat_generate(symbols, frames, _flat_shares())
        error = excinfo.value
        message = str(error)
        assert error.unrescued == [f"L{i}" for i in range(8)]
        assert error.rows_affected == 0
        assert "…" not in message
        assert f"Affected (8): {', '.join(error.unrescued)}." in message
        assert message.count(_REASON) == 8

    def test_the_ninth_lost_symbol_stays_on_the_exception_and_leaves_the_message(self) -> None:
        """The route echoes ``str(error)``. The attribute is what a catcher still has."""
        symbols, frames = _lost_request(9, _flat_ohlcv(seed=83))
        with pytest.raises(IncompleteDataError) as excinfo:
            _flat_generate(symbols, frames, _flat_shares())
        error = excinfo.value
        message = str(error)
        assert error.unrescued == [f"L{i}" for i in range(9)]
        assert error.rows_affected == 0
        assert "L8" not in message
        assert message.count(_REASON) == 8
        assert "(L0: no usable price data in range;" in message
        assert "L7: no usable price data in range)." in message
        assert "Affected (9): L0, L1, L2, L3, L4, L5, L6, L7…." in message
        assert "fewer symbols than were requested" in message
        assert "fabricated" not in message
