"""API configuration settings using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables with the
    JUNIPER_DATA_ prefix (e.g., JUNIPER_DATA_STORAGE_PATH).

    Security Settings:
        - api_keys: Comma-separated list of valid API keys (e.g., "key1,key2").
        - If empty, authentication is disabled (open access).
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
    # Default to localhost for safer local development; bind only to the loopback interface.
    # In production (e.g., Docker deployments), set JUNIPER_DATA_HOST=0.0.0.0 to listen on all interfaces.
    # host: str = "0.0.0.0"
    host: str = "127.0.0.1"
    port: int = 8100
    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]

    api_keys: list[str] | None = None
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: int = 60


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
