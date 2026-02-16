"""Unit tests for juniper_data package __init__."""

from unittest.mock import MagicMock, patch

import pytest

from juniper_data import (
    __version__,
    get_arc_agi_api,
    get_arc_agi_api_url,
    get_arc_agi_arcade,
    get_arc_agi_env,
    get_arc_api_key,
    reload_arc_agi_env,
)


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

    def test_get_arc_agi_env_returns_true_when_set(self, monkeypatch) -> None:
        """get_arc_agi_env returns True when ARC_AGI_ENV is set."""
        monkeypatch.setenv("ARC_AGI_ENV", "1")
        assert get_arc_agi_env() is True

    def test_get_arc_agi_env_calls_load_dotenv_when_unset(self, monkeypatch) -> None:
        """get_arc_agi_env calls load_dotenv when ARC_AGI_ENV is not set."""
        monkeypatch.delenv("ARC_AGI_ENV", raising=False)
        with patch("juniper_data.load_dotenv", return_value=True) as mock_load:
            result = get_arc_agi_env()
            mock_load.assert_called_once()
            assert result is True

    def test_reload_arc_agi_env(self) -> None:
        """reload_arc_agi_env calls load_dotenv and returns its result."""
        with patch("juniper_data.load_dotenv", return_value=True) as mock_load:
            result = reload_arc_agi_env()
            mock_load.assert_called_once()
            assert result is True

    def test_reload_arc_agi_env_returns_false(self) -> None:
        """reload_arc_agi_env returns False when load_dotenv returns False."""
        with patch("juniper_data.load_dotenv", return_value=False):
            assert reload_arc_agi_env() is False

    def test_get_arc_api_key_returns_none_when_unset(self, monkeypatch) -> None:
        """get_arc_api_key returns None when ARC_API_KEY is not set."""
        monkeypatch.delenv("ARC_API_KEY", raising=False)
        assert get_arc_api_key() is None

    def test_get_arc_api_key_returns_value_when_set(self, monkeypatch) -> None:
        """get_arc_api_key returns the key value when set."""
        monkeypatch.setenv("ARC_API_KEY", "test-key-123")
        assert get_arc_api_key() == "test-key-123"

    def test_get_arc_api_key_returns_none_for_empty_string(self, monkeypatch) -> None:
        """get_arc_api_key returns None when ARC_API_KEY is empty string."""
        monkeypatch.setenv("ARC_API_KEY", "")
        assert get_arc_api_key() is None

    def test_get_arc_agi_arcade_returns_arcade_instance(self, monkeypatch) -> None:
        """get_arc_agi_arcade creates an Arcade instance."""
        monkeypatch.delenv("ARC_API_KEY", raising=False)
        mock_arcade = MagicMock()
        with patch("juniper_data.arc_agi.Arcade", return_value=mock_arcade):
            result = get_arc_agi_arcade()
            assert result is mock_arcade
