"""Pins the equities shares-concept fallback and the APD-DATA-018 symbol cap.

#348 measured the two upstreams this generator actually uses and recorded two
production facts that the existing suite cannot see:

* ``_fetch_shares`` tries ``dei:EntityCommonStockSharesOutstanding`` then
  ``us-gaap:CommonStockSharesOutstanding``. Every existing test mocks
  ``_sec_get`` as a constant return, so the second concept is never consulted
  and a revert that drops the fallback stays green. KO's live miss (zero
  facts on the primary tag, 404 on the fallback) is this path.
* The cost axis is **symbol count**, not wire bytes. The bound is
  ``max_symbols``. Its default is ``None`` (unbounded); ``ge=1`` refuses 0.
  Only ``max_symbols=2`` against the default universe is tested.

Network is mocked. Requires the optional ``equities`` extra (pandas) for the
shares-series tests; skipped otherwise. ``_sec_get`` / params tests do not
need yfinance.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_equities_concept_fallback.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from juniper_data.generators.equities import EquitiesGenerator, EquitiesParams
from juniper_data.generators.equities import generator as eq_gen
from juniper_data.generators.equities.defaults import EQUITIES_DEFAULT_MAX_SYMBOLS

pd = pytest.importorskip("pandas")

pytestmark = [pytest.mark.unit, pytest.mark.generators]

_DEI = "dei/EntityCommonStockSharesOutstanding"
_US_GAAP = "us-gaap/CommonStockSharesOutstanding"
_FALLBACK_VAL = 2.0e9
_FALLBACK_PAYLOAD = {
    "units": {
        "shares": [
            {"end": "2010-06-30", "val": _FALLBACK_VAL, "filed": "2010-07-01"},
        ]
    }
}


def _fake_urlopen_response(payload: dict):
    """A ``urlopen``-compatible context manager whose ``read()`` yields JSON."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _dispatch_sec_get(dei_payload, us_gaap_payload):
    """Return an ``_sec_get`` stand-in that records URLs and dispatches by tag."""
    calls: list[str] = []

    def fake(url: str, retries: int = 3):
        calls.append(url)
        if _DEI in url:
            return dei_payload
        if _US_GAAP in url:
            return us_gaap_payload
        return None

    return fake, calls


class TestSharesConceptFallback:
    """``_fetch_shares`` must try the two XBRL concepts in documented order."""

    def test_falls_back_to_us_gaap_when_dei_404s(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        fake, calls = _dispatch_sec_get(None, _FALLBACK_PAYLOAD)
        monkeypatch.setattr(eq_gen, "_sec_get", fake)

        series = EquitiesGenerator._fetch_shares(21344, use_cache=False)

        assert series is not None
        assert float(series.iloc[0]) == _FALLBACK_VAL
        assert _DEI in calls[0]
        assert _US_GAAP in calls[1]

    def test_falls_back_when_dei_units_are_empty(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # ``payload.get("units")`` is ``{}`` — falsy, so the next concept is tried.
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        fake, calls = _dispatch_sec_get({"units": {}}, _FALLBACK_PAYLOAD)
        monkeypatch.setattr(eq_gen, "_sec_get", fake)

        series = EquitiesGenerator._fetch_shares(21344, use_cache=False)

        assert series is not None
        assert float(series.iloc[0]) == _FALLBACK_VAL
        assert any(_US_GAAP in url for url in calls)

    def test_empty_facts_list_does_not_consult_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """KO-shaped miss: primary returns units with zero usable facts.

        ``if payload and payload.get("units")`` treats ``{"shares": []}`` as a
        hit, so us-gaap is never asked. Pin the short-circuit so a later
        change that *does* fall through is intentional, not accidental.
        """
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        fake, calls = _dispatch_sec_get({"units": {"shares": []}}, _FALLBACK_PAYLOAD)
        monkeypatch.setattr(eq_gen, "_sec_get", fake)

        series = EquitiesGenerator._fetch_shares(21344, use_cache=False)

        assert series is None
        assert len(calls) == 1
        assert _DEI in calls[0]
        assert not any(_US_GAAP in url for url in calls)

    def test_both_concepts_missing_returns_none(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(eq_gen, "_CACHE_DIR", tmp_path)
        fake, calls = _dispatch_sec_get(None, None)
        monkeypatch.setattr(eq_gen, "_sec_get", fake)

        assert EquitiesGenerator._fetch_shares(21344, use_cache=False) is None
        assert any(_DEI in url for url in calls)
        assert any(_US_GAAP in url for url in calls)


class TestSymbolCap:
    """APD-DATA-018: the equities bound is symbol count, default unbounded."""

    def test_default_max_symbols_is_unbounded(self) -> None:
        assert EQUITIES_DEFAULT_MAX_SYMBOLS is None
        assert EquitiesParams().max_symbols is None

    def test_unbounded_max_symbols_does_not_slice(self) -> None:
        constituents = {name: {"name": name, "cik": i, "sector": ""} for i, name in enumerate(["A", "B", "C"])}
        ordered, _meta = EquitiesGenerator._resolve_symbols(EquitiesParams(max_symbols=None), constituents)
        assert ordered == ["A", "B", "C"]

    def test_max_symbols_slices_explicit_symbol_list(self) -> None:
        # Cap applies after the caller's order, not only the default universe.
        constituents = {name: {"name": name, "cik": i, "sector": ""} for i, name in enumerate(["A", "B", "C"])}
        ordered, _meta = EquitiesGenerator._resolve_symbols(
            EquitiesParams(symbols=["C", "A", "B"], max_symbols=2),
            constituents,
        )
        assert ordered == ["C", "A"]

    def test_max_symbols_zero_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EquitiesParams(max_symbols=0)


class TestSecFairAccess:
    """SEC fair-access: descriptive User-Agent + <10 req/s via ``_SEC_MIN_INTERVAL``."""

    def test_sec_get_sends_compliant_user_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(eq_gen.time, "sleep", lambda *_a, **_k: None)
        captured: dict[str, str] = {}

        def fake_urlopen(request, timeout=30):
            captured["user_agent"] = request.get_header("User-agent") or request.get_header("User-Agent")
            return _fake_urlopen_response({"ok": 1})

        monkeypatch.setattr(eq_gen.urllib.request, "urlopen", fake_urlopen)
        eq_gen._last_sec_call[0] = 0.0
        assert eq_gen._sec_get("https://data.sec.gov/x") == {"ok": 1}
        assert captured["user_agent"] is not None
        assert "juniper-data" in captured["user_agent"]

    def test_sec_get_sleeps_when_last_call_was_inside_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(eq_gen.time, "sleep", lambda seconds: sleeps.append(seconds))
        monkeypatch.setattr(eq_gen.time, "monotonic", lambda: 100.0)
        monkeypatch.setattr(
            eq_gen.urllib.request,
            "urlopen",
            lambda *_a, **_k: _fake_urlopen_response({"ok": 1}),
        )
        eq_gen._last_sec_call[0] = 100.0 - 0.01  # 10 ms ago; interval is 0.12 s

        eq_gen._sec_get("https://data.sec.gov/x")

        throttle = [s for s in sleeps if s > 0.05]
        assert throttle, f"expected a throttle sleep, got {sleeps!r}"
        assert throttle[0] == pytest.approx(eq_gen._SEC_MIN_INTERVAL - 0.01, abs=1e-9)
