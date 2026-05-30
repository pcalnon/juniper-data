"""API configuration settings using pydantic-settings."""

# import json
from functools import lru_cache
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from juniper_data.core.secrets import get_secret

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

_JUNIPER_DATA_API_HOST_GLOBAL: str = "0.0.0.0"  # nosec B104
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
_JUNIPER_DATA_API_RATELIMIT_ACTIVE_DEFAULT: bool = _JUNIPER_DATA_API_RATELIMIT_ENABLED

_JUNIPER_DATA_API_RATELIMIT_VALUE_SLOW: int = 30  # Requests per Minute
_JUNIPER_DATA_API_RATELIMIT_VALUE_MID: int = 60  # Requests per Minute
_JUNIPER_DATA_API_RATELIMIT_VALUE_FAST: int = 120  # Requests per Minute
_JUNIPER_DATA_API_RATELIMIT_DEFAULT: int = _JUNIPER_DATA_API_RATELIMIT_VALUE_MID

_JUNIPER_DATA_API_CORS_ORIGINS_ALL: list[str] = ["*"]
_JUNIPER_DATA_API_CORS_ORIGINS_NONE: list[str] = []
_JUNIPER_DATA_API_CORS_ORIGINS_DEFAULT: list[str] = _JUNIPER_DATA_API_CORS_ORIGINS_NONE


_JUNIPER_DATA_API_KEYS_LIST_EMPTY: list[str] | None = None
_JUNIPER_DATA_API_KEYS_LIST_VALUES: list[str] | None = []
_JUNIPER_DATA_API_KEYS_LIST_DEFAULT: list[str] | None = _JUNIPER_DATA_API_KEYS_LIST_EMPTY

_JUNIPER_DATA_API_LOG_FORMAT_TEXT: str = "text"
_JUNIPER_DATA_API_LOG_FORMAT_DEFAULT: str = _JUNIPER_DATA_API_LOG_FORMAT_TEXT

_JUNIPER_DATA_API_SENTRY_DSN_NONE: str | None = None
_JUNIPER_DATA_API_SENTRY_DSN_DEFAULT: str | None = _JUNIPER_DATA_API_SENTRY_DSN_NONE

_JUNIPER_DATA_API_SENTRY_SEND_PII_DEFAULT: bool = False
_JUNIPER_DATA_API_SENTRY_TRACES_SAMPLE_RATE_DEFAULT: float = 0.1

_JUNIPER_DATA_API_METRICS_ENABLED_DISABLED: bool = False
_JUNIPER_DATA_API_METRICS_ENABLED_DEFAULT: bool = _JUNIPER_DATA_API_METRICS_ENABLED_DISABLED

# SEC-16: default allowlist for the Prometheus /metrics endpoint. The
# endpoint is mounted as an ASGI sub-app and therefore bypasses the
# router-level SecurityMiddleware, so we gate it separately on client IP.
# Defaults to loopback-only (IPv4 + IPv6); operators who scrape from a
# dedicated Prometheus host must override via
# JUNIPER_DATA_METRICS_TRUSTED_IPS. Accepts bare IPs and CIDR ranges
# (e.g. "172.18.0.0/16", "fd00::/8") — see MetricsAuthMiddleware.
_JUNIPER_DATA_API_METRICS_TRUSTED_IPS_DEFAULT: list[str] = ["127.0.0.1", "::1"]

_JUNIPER_DATA_API_IMPORT_DIR: str = "/data/imports"
_JUNIPER_DATA_API_IMPORT_DIR_DEFAULT: str = _JUNIPER_DATA_API_IMPORT_DIR


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables with the
    JUNIPER_DATA_ prefix (e.g., JUNIPER_DATA_STORAGE_PATH).

    Security Settings:
        - api_keys: JSON list of comma-separated, valid API keys (e.g., ["key1,key2"] ).
        - If empty, authentication is disabled (open access).
        - rate_limit_enabled: Enable/disable rate limiting.
        - rate_limit_requests_per_minute: Max requests per minute per client.
    """

    model_config = SettingsConfigDict(
        env_prefix=_JUNIPER_DATA_ENV_PREFIX_DEFAULT,
        env_file=_JUNIPER_DATA_ENV_FILE_DEFAULT,
        env_file_encoding=_JUNIPER_DATA_ENV_FILE_ENCODING_DEFAULT,
        case_sensitive=_JUNIPER_DATA_ENV_CASE_SENSITIVE_DEFAULT,
        extra=_JUNIPER_DATA_ENV_EXTRA_DEFAULT,
    )

    # storage_path: str = "./data/datasets"
    storage_path: str = _JUNIPER_DATA_API_STORAGE_PATH_DEFAULT

    # Default to a more restrictive binding (e.g., 127.0.0.1) for general, non-containerized environments.
    # To provide external access and allow listening on all interfaces,
    #   for compatibility with containerized deployments (e.g., Docker, Kubernetes),
    #   override JUNIPER_DATA_HOST (e.g., to 0.0.0.0).
    # Note: When setting JUNIPER_DATA_HOST to 0.0.0.0, use firewall/security groups or reverse proxies to control access.
    host: str = _JUNIPER_DATA_API_HOST_DEFAULT
    port: int = _JUNIPER_DATA_API_PORT_DEFAULT
    log_level: str = _JUNIPER_DATA_API_LOGLEVEL_DEFAULT
    cors_origins: list[str] = _JUNIPER_DATA_API_CORS_ORIGINS_DEFAULT

    # api_keys: list[str] | None = _JUNIPER_DATA_API_KEYS_LIST_DEFAULT
    # api_keys: JSON[list[str]] | None = _JUNIPER_DATA_API_KEYS_LIST_DEFAULT
    api_keys: list[str] | None = _JUNIPER_DATA_API_KEYS_LIST_DEFAULT

    @model_validator(mode="before")
    @classmethod
    def _inject_secrets(cls, data: Any) -> Any:
        """Inject file-based Docker secrets into settings data before field validation."""
        if isinstance(data, dict) and not data.get("api_keys"):
            secret_value = get_secret("JUNIPER_DATA_API_KEYS")
            if secret_value:
                data["api_keys"] = secret_value
        return data

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, v: object) -> list[str] | None:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v  # type: ignore[return-value]

    import_dir: str = _JUNIPER_DATA_API_IMPORT_DIR_DEFAULT

    rate_limit_enabled: bool = _JUNIPER_DATA_API_RATELIMIT_ACTIVE_DEFAULT
    rate_limit_requests_per_minute: int = _JUNIPER_DATA_API_RATELIMIT_DEFAULT

    log_format: str = _JUNIPER_DATA_API_LOG_FORMAT_DEFAULT
    sentry_dsn: str | None = _JUNIPER_DATA_API_SENTRY_DSN_DEFAULT
    sentry_send_pii: bool = _JUNIPER_DATA_API_SENTRY_SEND_PII_DEFAULT
    sentry_traces_sample_rate: float = _JUNIPER_DATA_API_SENTRY_TRACES_SAMPLE_RATE_DEFAULT
    metrics_enabled: bool = _JUNIPER_DATA_API_METRICS_ENABLED_DEFAULT
    # SEC-16: loopback-only by default. Set
    # ``JUNIPER_DATA_METRICS_TRUSTED_IPS='["10.0.0.5","172.18.0.0/16"]'``
    # (JSON list) or a comma-separated string. Accepts bare IP literals and
    # CIDR ranges; ``MetricsAuthMiddleware`` normalises IPv6 zone-ids and
    # IPv4-mapped IPv6 client addresses before membership check, so a
    # Docker container appearing as ``::ffff:172.18.0.5`` matches an IPv4
    # ``172.18.0.0/16`` allowlist entry.
    metrics_trusted_ips: list[str] = _JUNIPER_DATA_API_METRICS_TRUSTED_IPS_DEFAULT

    @field_validator("metrics_trusted_ips")
    @classmethod
    def _validate_metrics_trusted_ips(cls, v: list[str]) -> list[str]:
        """Fail loud at startup if any allowlist entry is unparseable.

        Without this guard a typo like ``172.18.0.0/164`` would silently
        compile to a working-but-empty allowlist that 403s every scrape.
        ``MetricsAuthMiddleware`` raises the same ``ValueError`` at
        construction time, but Settings validation surfaces it before
        the FastAPI app gets created. Uses the shared
        ``parse_trusted_networks`` from ``juniper-observability`` so
        juniper-data's fail-loud message stays in lockstep with cascor
        and any future consumer.
        """
        from juniper_observability import parse_trusted_networks

        parse_trusted_networks(v)
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
