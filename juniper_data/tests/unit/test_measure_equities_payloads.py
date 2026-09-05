"""Pins the APD-DATA-018 SEC measurement helper's parse-error contract.

``util/ad-hoc/2026-09-04_measure_equities_payloads.py`` prints a fact count that
a reader treats as evidence for the equities cap. An empty ``except
json.JSONDecodeError: pass`` used to return ``facts=0``, which the table reports
as "this concept has no facts" -- a real, meaningful result -- when the body
could not be parsed at all. The helper now returns ``-1`` for that case and
keeps the byte count and elapsed time (those are still valid measurements).

These tests mock ``urlopen`` so they stay offline and deterministic.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_measure_equities_payloads.py
# Author:        Paul Calnon
# Version:       0.12.0
# License:       MIT License

from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit]

_SCRIPT = Path(__file__).resolve().parents[3] / "util" / "ad-hoc" / "2026-09-04_measure_equities_payloads.py"
_CIK = 320193


def _load_script():
    spec = importlib.util.spec_from_file_location("measure_equities_payloads", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def measure_mod():
    return _load_script()


def _raw_response(body: bytes):
    """A ``urlopen``-compatible context manager whose ``read()`` yields ``body``."""
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _json_response(payload: dict) -> MagicMock:
    return _raw_response(json.dumps(payload).encode())


class TestMeasureSecParseContract:
    """``measure_sec`` must not report an unparseable body as an empty concept."""

    def test_unparseable_body_returns_minus_one_and_keeps_bytes(self, measure_mod, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """HTML / non-JSON is unknowable, not zero facts. Bytes and elapsed stay valid."""
        body = b"<html>not a companyconcept payload</html>"
        monkeypatch.setattr(measure_mod.urllib.request, "urlopen", lambda *_a, **_k: _raw_response(body))

        elapsed, nbytes, facts = measure_mod.measure_sec(_CIK)

        assert facts == -1
        assert nbytes == len(body)
        assert elapsed >= 0.0
        captured = capsys.readouterr().out
        assert "not JSON" in captured
        assert str(len(body)) in captured

    def test_empty_json_object_is_zero_facts_not_minus_one(self, measure_mod, monkeypatch: pytest.MonkeyPatch) -> None:
        """``{}`` is a parsed empty concept -- the outcome the -1 sentinel must not collide with."""
        monkeypatch.setattr(measure_mod.urllib.request, "urlopen", lambda *_a, **_k: _json_response({}))

        _elapsed, nbytes, facts = measure_mod.measure_sec(_CIK)

        assert facts == 0
        assert nbytes > 0

    def test_empty_units_object_is_zero_facts(self, measure_mod, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(measure_mod.urllib.request, "urlopen", lambda *_a, **_k: _json_response({"units": {}}))

        _elapsed, _nbytes, facts = measure_mod.measure_sec(_CIK)

        assert facts == 0

    def test_sums_facts_across_unit_keys(self, measure_mod, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "units": {
                "USD": [{"val": 1}, {"val": 2}],
                "shares": [{"val": 3}],
            }
        }
        monkeypatch.setattr(measure_mod.urllib.request, "urlopen", lambda *_a, **_k: _json_response(payload))

        _elapsed, _nbytes, facts = measure_mod.measure_sec(_CIK)

        assert facts == 3

    def test_transport_error_returns_zero_bytes_and_zero_facts(self, measure_mod, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """A failed fetch is not a parsed empty concept either, but it has no wire bytes to keep."""

        def _down(*_a, **_k):
            raise urllib.error.URLError("network down")

        monkeypatch.setattr(measure_mod.urllib.request, "urlopen", _down)

        elapsed, nbytes, facts = measure_mod.measure_sec(_CIK)

        assert nbytes == 0
        assert facts == 0
        assert elapsed >= 0.0
        assert "URLError" in capsys.readouterr().out

    def test_request_sends_sec_user_agent(self, measure_mod, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def _capture(request, timeout=None):
            captured["url"] = request.full_url
            captured["ua"] = request.get_header("User-agent")
            captured["timeout"] = timeout
            return _json_response({"units": {}})

        monkeypatch.setattr(measure_mod.urllib.request, "urlopen", _capture)

        measure_mod.measure_sec(_CIK)

        assert f"{_CIK:010d}" in str(captured["url"])
        assert captured["ua"] == measure_mod.SEC_UA["User-Agent"]
        assert captured["timeout"] == 30
