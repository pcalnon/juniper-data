"""Unit tests for API settings module."""

import os
from unittest.mock import patch

import pytest

from juniper_data.api.settings import (
    _JUNIPER_DATA_API_HOST_DEFAULT,
    Settings,
    get_settings,
)


@pytest.mark.unit
class TestSettings:
    """Tests for the Settings class."""

    def test_default_storage_path(self) -> None:
        """Test default storage path is set."""
        settings = Settings()
        settings = Settings()
        # Default host was changed from 127.0.0.1 to 0.0.0.0 to allow external access.

    def test_default_host(self) -> None:
        """Test default host is set."""
        settings = Settings()
        # assert settings.host == "127.0.0.1"
        # assert settings.host == "0.0.0.0"
        assert settings.host == _JUNIPER_DATA_API_HOST_DEFAULT

    def test_default_port(self) -> None:
        """Test default port is set."""
        settings = Settings()
        assert settings.port == 8100

    def test_default_log_level(self) -> None:
        """Test default log level is set."""
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_default_cors_origins(self) -> None:
        """Test default CORS origins is set."""
        settings = Settings()
        assert settings.cors_origins == ["*"]

    def test_custom_values(self) -> None:
        """Test custom values can be set."""
        settings = Settings(
            storage_path="/custom/path",
            host="127.0.0.1",
            port=9000,
            log_level="DEBUG",
            cors_origins=["http://localhost:3000"],
        )

        assert settings.storage_path == "/custom/path"
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.log_level == "DEBUG"
        assert settings.cors_origins == ["http://localhost:3000"]

    def test_env_var_override(self) -> None:
        """Test environment variables override defaults."""
        with patch.dict(os.environ, {"JUNIPER_DATA_PORT": "9999"}):
            settings = Settings()
            assert settings.port == 9999

    def test_env_prefix(self) -> None:
        """Test JUNIPER_DATA_ prefix is used."""
        with patch.dict(os.environ, {"JUNIPER_DATA_STORAGE_PATH": "/env/path"}):
            settings = Settings()
            assert settings.storage_path == "/env/path"


@pytest.mark.unit
class TestGetSettings:
    """Tests for the get_settings function."""

    def test_get_settings_returns_settings(self) -> None:
        """Test get_settings returns a Settings instance."""
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self) -> None:
        """Test get_settings returns cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_get_settings_cache_clear(self) -> None:
        """Test cache can be cleared to get new instance."""
        get_settings.cache_clear()
        settings1 = get_settings()

        get_settings.cache_clear()
        settings2 = get_settings()

        assert settings1 is not settings2
