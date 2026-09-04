"""API configuration settings using pydantic-settings."""

# import json
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from juniper_data.api.constants import DEFAULT_RATE_LIMIT_WINDOW_SECONDS
from juniper_data.core.limits import CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION, CSV_IMPORT_DEFAULT_MAX_BYTES, EQUITIES_DEFAULT_ALLOW_TRUNCATION, EQUITIES_DEFAULT_MAX_SYMBOLS
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

# APD-DATA-033: the window was the one knob of three with no operator-facing
# setting. ``RateLimiter`` has always accepted ``window_seconds`` and uses it in
# three places (``self._window``, the cache TTL, and the ``window`` property),
# but ``app.py`` never passed it, so the value was pinned to the constant.
# Sourced from ``api.constants.DEFAULT_RATE_LIMIT_WINDOW_SECONDS`` so the
# setting default and the ``RateLimiter`` constructor default cannot drift --
# the two are the same object, not two literals that happen to agree.
_JUNIPER_DATA_API_RATELIMIT_WINDOW_DEFAULT: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS

_JUNIPER_DATA_API_CORS_ORIGINS_ALL: list[str] = ["*"]
_JUNIPER_DATA_API_CORS_ORIGINS_NONE: list[str] = []
_JUNIPER_DATA_API_CORS_ORIGINS_DEFAULT: list[str] = _JUNIPER_DATA_API_CORS_ORIGINS_NONE


_JUNIPER_DATA_API_KEYS_LIST_EMPTY: list[str] | None = None
_JUNIPER_DATA_API_KEYS_LIST_VALUES: list[str] | None = []
_JUNIPER_DATA_API_KEYS_LIST_DEFAULT: list[str] | None = _JUNIPER_DATA_API_KEYS_LIST_EMPTY

_JUNIPER_DATA_API_REQUIRE_AUTH_ENABLED: bool = True
_JUNIPER_DATA_API_REQUIRE_AUTH_DISABLED: bool = False
_JUNIPER_DATA_API_REQUIRE_AUTH_DEFAULT: bool = _JUNIPER_DATA_API_REQUIRE_AUTH_DISABLED

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

# APD-DATA-018. Sourced from juniper_data.core.limits, which is the single
# definition -- core imports nothing from api or generators, so it is reachable
# from both sides without the cycle that importing the csv_import package here
# would create.
_JUNIPER_DATA_API_CSV_IMPORT_MAX_BYTES_DEFAULT: int = CSV_IMPORT_DEFAULT_MAX_BYTES
_JUNIPER_DATA_API_CSV_IMPORT_ALLOW_TRUNCATION_DEFAULT: bool = CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION
_JUNIPER_DATA_API_EQUITIES_MAX_SYMBOLS_DEFAULT: int = EQUITIES_DEFAULT_MAX_SYMBOLS
_JUNIPER_DATA_API_EQUITIES_ALLOW_TRUNCATION_DEFAULT: bool = EQUITIES_DEFAULT_ALLOW_TRUNCATION


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables with the
    JUNIPER_DATA_ prefix (e.g., JUNIPER_DATA_STORAGE_PATH).

    Security Settings:
        - api_keys: JSON list of comma-separated, valid API keys (e.g., ["key1,key2"] ).
        - If empty, authentication is disabled (open access).
        - rate_limit_enabled: Enable/disable rate limiting.
        - rate_limit_requests_per_minute: Max requests allowed per window per client.
        - rate_limit_window_seconds: Length of that window, in seconds.
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

    # SEC-F01: the INTENDED auth posture, fed to enforce_auth_posture in the
    # lifespan (env ``JUNIPER_DATA_REQUIRE_AUTH``). False (default) = an
    # unset/blank JUNIPER_DATA_API_KEYS only WARNs at boot (service runs
    # open — bare/dev profile); True = boot REFUSES (CRITICAL +
    # AuthPostureError) when no real key is configured. Set true wherever
    # secrets are provisioned (the composed juniper-deploy stack).
    require_auth: bool = _JUNIPER_DATA_API_REQUIRE_AUTH_DEFAULT

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
        """Normalise ``api_keys`` to ``list[str] | None``, dropping blank entries.

        APD-DATA-003: both branches must filter. The comma-separated-string branch
        always did; the list branch returned ``v`` untouched, so the JSON form
        ``JUNIPER_DATA_API_KEYS='[""]'`` survived as ``['']`` and enabled
        authentication that then accepted an empty ``X-API-Key``.
        ``APIKeyAuth.__init__`` also filters -- that is the load-bearing guard, and
        this is defence in depth at the boundary where the inconsistency lived.
        A list that filters down to empty becomes ``None`` (auth disabled), which
        matches the empty-string case above rather than leaving an empty list.
        """
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        if isinstance(v, (list, tuple)):
            cleaned = [k.strip() if isinstance(k, str) else k for k in v if not isinstance(k, str) or k.strip()]
            return cleaned or None
        return v  # type: ignore[return-value]

    import_dir: str = _JUNIPER_DATA_API_IMPORT_DIR_DEFAULT

    # APD-DATA-018: deployment-wide input bound for csv_import, and the
    # deployment-wide opt-in to partial imports. Both are overridable per
    # request via CsvImportParams; these are the defaults a request inherits
    # when it does not set the field explicitly.
    #
    # Because Settings is a pydantic-settings BaseSettings with env_prefix
    # "JUNIPER_DATA_" and an env_file, declaring the fields here gives the
    # owner's three required opt-in surfaces at once: request parameter,
    # JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION environment variable, and the
    # matching .env / config-file entry.
    # ``gt=0`` is load-bearing, not decoration. Python's ``read(n)`` treats a
    # NEGATIVE n as "read everything", so a cap of -1 arriving from a mistyped
    # environment variable would turn the bound into its exact opposite --
    # silently, and only on the ingestion path. Rejecting it at settings
    # construction fails the deployment loudly instead.
    csv_import_max_bytes: int = Field(default=_JUNIPER_DATA_API_CSV_IMPORT_MAX_BYTES_DEFAULT, gt=0)
    csv_import_allow_truncation: bool = _JUNIPER_DATA_API_CSV_IMPORT_ALLOW_TRUNCATION_DEFAULT

    # APD-DATA-018, equities half. Same three opt-in surfaces as csv_import
    # above -- request parameter, JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION, and the
    # matching .env entry -- and the same gt=0 reasoning: a non-positive cap
    # would make the slice below it empty rather than bounded.
    equities_max_symbols: int = Field(default=_JUNIPER_DATA_API_EQUITIES_MAX_SYMBOLS_DEFAULT, gt=0)
    equities_allow_truncation: bool = _JUNIPER_DATA_API_EQUITIES_ALLOW_TRUNCATION_DEFAULT

    rate_limit_enabled: bool = _JUNIPER_DATA_API_RATELIMIT_ACTIVE_DEFAULT
    rate_limit_requests_per_minute: int = _JUNIPER_DATA_API_RATELIMIT_DEFAULT
    rate_limit_window_seconds: int = _JUNIPER_DATA_API_RATELIMIT_WINDOW_DEFAULT

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
