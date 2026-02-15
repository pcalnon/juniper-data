"""Unit tests for juniper_data package __init__."""

import pytest

from juniper_data import __version__, get_arc_agi_api, get_arc_agi_api_url


@pytest.mark.unit
class TestPackageInit:
    def test_version_is_string(self) -> None:
        assert isinstance(__version__, str)

    def test_get_arc_agi_api_url_returns_none_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("ARC_AGI_API", raising=False)
        assert get_arc_agi_api_url() is None

    def test_get_arc_agi_api_url_returns_value_when_set(self, monkeypatch) -> None:
        monkeypatch.setenv("ARC_AGI_API", "http://localhost:9000")
        assert get_arc_agi_api_url() == "http://localhost:9000"

    def test_get_arc_agi_api_delegates_to_url(self, monkeypatch) -> None:
        monkeypatch.setenv("ARC_AGI_API", "http://example.com")
        assert get_arc_agi_api() == "http://example.com"

    def test_get_arc_agi_api_returns_none_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("ARC_AGI_API", raising=False)
        assert get_arc_agi_api() is None
