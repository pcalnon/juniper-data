"""API configuration settings using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Define Safe and Reasonable Defaults for API Model Config
_JUNIPER_DATA_ENV_PREFIX: str = "JUNIPER_DATA_"
_JUNIPER_DATA_ENV_PREFIX_DEFAULT: str = _JUNIPER_DATA_ENV_PREFIX

_JUNIPER_DATA_ENV_FILE: str = ".env"
_JUNIPER_DATA_ENV_FILE_DEFAULT: str = _JUNIPER_DATA_ENV_FILE

_JUNIPER_DATA_ENV_FILE_ENCODING: str = "utf-8"
_JUNIPER_DATA_ENV_FILE_ENCODING_DEFAULT: str = _JUNIPER_DATA_ENV_FILE_ENCODING

_JUNIPER_DATA_ENV_CASE_SENSITIVE_ENABLED: bool = True
_JUNIPER_DATA_ENV_CASE_SENSITIVE_DISABLED: bool = False
_JUNIPER_DATA_ENV_CASE_SENSITIVE_DEFAULT: bool = _JUNIPER_DATA_ENV_CASE_SENSITIVE_DISABLED

_JUNIPER_DATA_ENV_EXTRA_DISABLED: str = "ignore"
_JUNIPER_DATA_ENV_EXTRA_DEFAULT: str = _JUNIPER_DATA_ENV_EXTRA_DISABLED

# Define Safe and Reasonable Defaults for API Settings
_JUNIPER_DATA_API_DATASET_PATH: str = "./data/datasets"
_JUNIPER_DATA_API_STORAGE_PATH_DEFAULT: str = _JUNIPER_DATA_API_DATASET_PATH

_JUNIPER_DATA_API_HOST_GLOBAL: str = "0.0.0.0"   #nosec B104
_JUNIPER_DATA_API_HOST_LOCAL: str = "127.0.0.1"
_JUNIPER_DATA_API_HOST_DEFAULT: str = _JUNIPER_DATA_API_HOST_LOCAL

_JUNIPER_DATA_API_PORT: int = 8100
_JUNIPER_DATA_API_PORT_DEFAULT: int = _JUNIPER_DATA_API_PORT

_JUNIPER_DATA_API_LOGLEVEL_TRACE: str = "TRACE"
_JUNIPER_DATA_API_LOGLEVEL_VERBOSE: str = "VERBOSE"
_JUNIPER_DATA_API_LOGLEVEL_DEBUG: str = "DEBUG"
_JUNIPER_DATA_API_LOGLEVEL_INFO: str = "INFO"
_JUNIPER_DATA_API_LOGLEVEL_WARNING: str = "WARNING"
_JUNIPER_DATA_API_LOGLEVEL_ERROR: str = "ERROR"
_JUNIPER_DATA_API_LOGLEVEL_CRITICAL: str = "CRITICAL"
_JUNIPER_DATA_API_LOGLEVEL_FATAL: str = "FATAL"
_JUNIPER_DATA_API_LOGLEVEL_DEFAULT: str = _JUNIPER_DATA_API_LOGLEVEL_INFO

_JUNIPER_DATA_API_RATELIMIT_DISABLED: bool = False
_JUNIPER_DATA_API_RATELIMIT_ENABLED: bool = True
_JUNIPER_DATA_API_RATELIMIT_ACTIVE_DEFAULT: bool = _JUNIPER_DATA_API_RATELIMIT_DISABLED

_JUNIPER_DATA_API_RATELIMIT_VALUE_SLOW: int = 30  # Requests per Minute
_JUNIPER_DATA_API_RATELIMIT_VALUE_MID: int = 60  # Requests per Minute
_JUNIPER_DATA_API_RATELIMIT_VALUE_FAST: int = 120  # Requests per Minute
_JUNIPER_DATA_API_RATELIMIT_DEFAULT: int = _JUNIPER_DATA_API_RATELIMIT_VALUE_MID

_JUNIPER_DATA_API_CORS_ORIGINS_ALL: list[str] = ["*"]
_JUNIPER_DATA_API_CORS_ORIGINS_NONE: list[str] = []
_JUNIPER_DATA_API_CORS_ORIGINS_DEFAULT: list[str] = _JUNIPER_DATA_API_CORS_ORIGINS_ALL


_JUNIPER_DATA_API_KEYS_LIST_EMPTY: list[str] | None = None
_JUNIPER_DATA_API_KEYS_LIST_VALUES: list[str] | None = []
_JUNIPER_DATA_API_KEYS_LIST_DEFAULT: list[str] | None = _JUNIPER_DATA_API_KEYS_LIST_EMPTY


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
        # env_prefix="JUNIPER_DATA_",
        env_prefix=_JUNIPER_DATA_ENV_PREFIX_DEFAULT,
        # env_file=".env",
        env_file=_JUNIPER_DATA_ENV_FILE_DEFAULT,
        # env_file_encoding="utf-8",
        env_file_encoding=_JUNIPER_DATA_ENV_FILE_ENCODING_DEFAULT,
        # case_sensitive=False,
        case_sensitive=_JUNIPER_DATA_ENV_CASE_SENSITIVE_DEFAULT,
        # extra="ignore",
        extra=_JUNIPER_DATA_ENV_EXTRA_DEFAULT,
    )

    # storage_path: str = "./data/datasets"
    storage_path: str = _JUNIPER_DATA_API_STORAGE_PATH_DEFAULT

    # Default to listening on all interfaces for compatibility with containerized deployments
    # (e.g., Docker, Kubernetes). Use firewall/security groups or reverse proxies to control
    # external access, and override JUNIPER_DATA_HOST (e.g., to 127.0.0.1) if you need a
    # more restrictive binding in specific environments.
    # host: str = "0.0.0.0"
    host: str = _JUNIPER_DATA_API_HOST_DEFAULT
    # port: int = 8100
    port: int = _JUNIPER_DATA_API_PORT_DEFAULT
    # log_level: str = "INFO"
    log_level: str = _JUNIPER_DATA_API_LOGLEVEL_DEFAULT
    # cors_origins: list[str] = ["*"]
    cors_origins: list[str] = _JUNIPER_DATA_API_CORS_ORIGINS_DEFAULT

    # api_keys: list[str] | None = None
    api_keys: list[str] | None = _JUNIPER_DATA_API_KEYS_LIST_DEFAULT
    # rate_limit_enabled: bool = False
    rate_limit_enabled: bool = _JUNIPER_DATA_API_RATELIMIT_ACTIVE_DEFAULT
    # rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_minute: int = _JUNIPER_DATA_API_RATELIMIT_DEFAULT


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
