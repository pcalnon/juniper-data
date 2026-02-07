"""API configuration settings using pydantic-settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables with the
    JUNIPER_DATA_ prefix (e.g., JUNIPER_DATA_STORAGE_PATH).

    Security Settings:
        - api_keys: Comma-separated list of valid API keys (e.g., "key1,key2").
          If empty, authentication is disabled (open access).
        - rate_limit_enabled: Enable/disable rate limiting.
        - rate_limit_requests_per_minute: Max requests per minute per client.
    """

    model_config = SettingsConfigDict(
        env_prefix="JUNIPER_DATA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    storage_path: str = "./data/datasets"
    host: str = "0.0.0.0"
    port: int = 8100
    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]

    api_keys: Optional[list[str]] = None
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: int = 60


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
