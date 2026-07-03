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
        """get_arc_agi_env calls load_dotenv then returns False when ARC_AGI_ENV remains unset."""
        monkeypatch.delenv("ARC_AGI_ENV", raising=False)
        with patch("juniper_data.load_dotenv", return_value=True) as mock_load:
            result = get_arc_agi_env()
            mock_load.assert_called_once()
            assert result is False

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
        """get_arc_agi_arcade creates an Arcade instance when arc-agi is installed."""
        monkeypatch.delenv("ARC_API_KEY", raising=False)
        mock_arcade = MagicMock()
        mock_arc_agi = MagicMock()
        mock_arc_agi.Arcade.return_value = mock_arcade
        with patch("juniper_data.ARC_AGI_AVAILABLE", True), patch("juniper_data.arc_agi", mock_arc_agi):
            result = get_arc_agi_arcade()
            assert result is mock_arcade

    def test_get_arc_agi_arcade_raises_when_not_installed(self) -> None:
        """get_arc_agi_arcade raises ImportError when arc-agi is not installed."""
        with patch("juniper_data.ARC_AGI_AVAILABLE", False):
            with pytest.raises(ImportError, match="arc-agi package not installed"):
                get_arc_agi_arcade()


@pytest.mark.unit
class TestPackageInitImportGuard:
    """Cover the module-level ``except ImportError`` arc-agi degradation branch."""

    def test_missing_arc_agi_sets_unavailable(self) -> None:
        """Reloading the package with ``arc_agi`` unimportable degrades gracefully.

        The ``import arc_agi`` at import time normally succeeds in this env (the
        ``arc-agi`` extra is installed), so the ``except ImportError`` fallback
        (``ARC_AGI_AVAILABLE = False``; ``arc_agi = None``) is otherwise never
        executed. Reload the package with the import blocked to exercise it, then
        restore the real module state so no other test is affected.
        """
        import builtins
        import importlib
        import sys

        import juniper_data as jd

        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "arc_agi" or name.startswith("arc_agi."):
                raise ImportError("arc_agi blocked for coverage of the fallback branch")
            return real_import(name, *args, **kwargs)

        saved = sys.modules.pop("arc_agi", None)
        try:
            with patch.object(builtins, "__import__", side_effect=_blocked_import):
                importlib.reload(jd)
            assert jd.ARC_AGI_AVAILABLE is False
            assert jd.arc_agi is None
        finally:
            if saved is not None:
                sys.modules["arc_agi"] = saved
            importlib.reload(jd)

        # After restore the real extra is importable again in this env.
        assert jd.ARC_AGI_AVAILABLE is True
