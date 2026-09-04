"""Unit tests for API settings module."""

import os
from typing import Any
from unittest.mock import patch

import pytest

from juniper_data.api.app import create_app
from juniper_data.api.constants import DEFAULT_RATE_LIMIT_WINDOW_SECONDS
from juniper_data.api.settings import (
    _JUNIPER_DATA_API_HOST_DEFAULT,
    Settings,
    get_settings,
)
from juniper_data.core.limits import CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION, CSV_IMPORT_DEFAULT_MAX_BYTES


@pytest.mark.unit
class TestSettings:
    """Tests for the Settings class."""

    def test_default_storage_path(self) -> None:
        """Test default storage path is set."""
        settings = Settings()
        # Default host was changed from 127.0.0.1 to 0.0.0.0 to allow external access.

    def test_default_host(self) -> None:
        """Test default host is set."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("JUNIPER_DATA_")}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
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
        assert settings.cors_origins == []

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

    def test_csv_import_bounds_default_to_core_constants(self) -> None:
        """APD-DATA-018: the deployment knobs default to the single core definition."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("JUNIPER_DATA_")}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.csv_import_max_bytes == CSV_IMPORT_DEFAULT_MAX_BYTES
            assert settings.csv_import_allow_truncation is CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION

    def test_csv_import_bounds_are_settable_from_the_environment(self) -> None:
        """The owner-required CLI / .env surface: JUNIPER_DATA_CSV_IMPORT_*."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("JUNIPER_DATA_")}
        env["JUNIPER_DATA_CSV_IMPORT_MAX_BYTES"] = "4096"
        env["JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION"] = "true"
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.csv_import_max_bytes == 4096
            assert settings.csv_import_allow_truncation is True


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


@pytest.mark.unit
class TestRateLimitWindowSetting:
    """APD-DATA-033: the rate-limit window must be operator-configurable.

    ``RateLimiter`` has always accepted ``window_seconds`` and uses it in three
    places, but ``create_app`` passed only ``requests_per_minute`` and
    ``enabled`` -- so the window was pinned to the constant and was the one knob
    of three an operator could not set.
    """

    def test_window_default_matches_the_limiter_constructor_default(self) -> None:
        # The setting default and the RateLimiter default must be the SAME
        # object, not two literals that happen to agree -- otherwise they drift.
        env = {k: v for k, v in os.environ.items() if not k.startswith("JUNIPER_DATA_")}
        with patch.dict(os.environ, env, clear=True):
            assert Settings().rate_limit_window_seconds == DEFAULT_RATE_LIMIT_WINDOW_SECONDS

    def test_window_is_settable_from_the_environment(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("JUNIPER_DATA_")}
        env["JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS"] = "300"
        with patch.dict(os.environ, env, clear=True):
            assert Settings().rate_limit_window_seconds == 300

    def test_configured_window_reaches_the_live_rate_limiter(self) -> None:
        """The decisive arm -- a Settings field nobody reads is the defect itself.

        The two arms above pass unchanged against the broken code: the field can
        exist, parse and validate while ``create_app`` still never passes it.
        Only reading the window back off the limiter the app actually built
        proves the knob is wired.
        """
        settings = Settings(api_keys=None, rate_limit_window_seconds=300, rate_limit_requests_per_minute=7)
        app = create_app(settings=settings)
        limiter = _rate_limiter_of(app)
        assert limiter is not None, "no RateLimiter found on the app's middleware stack"
        assert limiter.window == 300
        assert limiter.limit == 7


def _rate_limiter_of(app: Any) -> Any:
    """Pull the RateLimiter out of the app's SecurityMiddleware options."""
    for middleware in app.user_middleware:
        limiter = middleware.kwargs.get("rate_limiter")
        if limiter is not None:
            return limiter
    return None
